// SPDX-License-Identifier: Apache-2.0
// hybrid_pipeline_scheduler.cu — M6 2D time × space pipeline implementation.
//
// Architecture:
// - MPI_Cart_create with dims [P_time, P_space]
// - time_comm: ranks with same spatial index → TD ring
// - space_comm: ranks with same time index → halo exchange
// - Each rank stores owned atoms (in Y-subdomain) sorted by zone,
//   plus ghost atoms from spatial neighbors appended after.
// - Halo exchange: pack ghost positions → MPI_Sendrecv → upload to GPU
// - TD exchange: pack zone data → MPI_Sendrecv on time_comm → unpack

#include "hybrid_pipeline_scheduler.cuh"

#include <algorithm>
#include <cstring>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"
#include "../integrator/device_nose_hoover.cuh"
#include "../integrator/device_velocity_verlet_zone.cuh"
#include "../potentials/device_morse_zone.cuh"

namespace tdmd::scheduler {

// ---- Pack / unpack helpers for TD zone exchange ----

static constexpr std::size_t kHeaderBytes = 3 * sizeof(i32);

static std::size_t packed_zone_size(i32 natoms) {
  return kHeaderBytes + static_cast<std::size_t>(natoms) * 2 * sizeof(Vec3);
}

static void pack_zone(char* buf, i32 zone_id, i32 time_step, i32 natoms,
                       const Vec3* pos, const Vec3* vel) {
  std::memcpy(buf, &zone_id, sizeof(i32));
  buf += sizeof(i32);
  std::memcpy(buf, &time_step, sizeof(i32));
  buf += sizeof(i32);
  std::memcpy(buf, &natoms, sizeof(i32));
  buf += sizeof(i32);
  auto n = static_cast<std::size_t>(natoms);
  std::memcpy(buf, pos, n * sizeof(Vec3));
  buf += static_cast<std::ptrdiff_t>(n * sizeof(Vec3));
  std::memcpy(buf, vel, n * sizeof(Vec3));
}

static void unpack_zone(const char* buf, i32& zone_id, i32& time_step,
                         i32& natoms, Vec3* pos, Vec3* vel) {
  std::memcpy(&zone_id, buf, sizeof(i32));
  buf += sizeof(i32);
  std::memcpy(&time_step, buf, sizeof(i32));
  buf += sizeof(i32);
  std::memcpy(&natoms, buf, sizeof(i32));
  buf += sizeof(i32);
  auto n = static_cast<std::size_t>(natoms);
  std::memcpy(pos, buf, n * sizeof(Vec3));
  buf += static_cast<std::ptrdiff_t>(n * sizeof(Vec3));
  std::memcpy(vel, buf, n * sizeof(Vec3));
}

// ---- Constructor ----

HybridPipelineScheduler::HybridPipelineScheduler(
    const Box& box, i32 natoms, const potentials::MorseParams& params,
    const HybridConfig& cfg, MPI_Comm world_comm)
    : cfg_(cfg),
      params_(params),
      box_(box),
      natoms_global_(natoms),
      world_comm_(world_comm),
      streams_(cfg.deterministic ? 1 : cfg.n_streams) {
  MPI_Comm_rank(world_comm_, &world_rank_);
  MPI_Comm_size(world_comm_, &world_size_);

  TDMD_ASSERT(cfg.p_time * cfg.p_space == world_size_,
              "p_time * p_space must equal MPI world size");

  // Create 2D Cartesian communicator [P_time, P_space].
  // reorder=0 so that Cartesian coordinates are predictable from world rank.
  int dims[2] = {cfg.p_time, cfg.p_space};
  int periods[2] = {1, 1};  // periodic in both dimensions
  MPI_Cart_create(world_comm_, 2, dims, periods, 0, &cart_comm_);

  // Extract subcommunicators.
  // time_comm: vary time index, fix space index.
  int time_remain[2] = {1, 0};
  MPI_Cart_sub(cart_comm_, time_remain, &time_comm_);

  // space_comm: fix time index, vary space index.
  int space_remain[2] = {0, 1};
  MPI_Cart_sub(cart_comm_, space_remain, &space_comm_);

  MPI_Comm_rank(time_comm_, &time_rank_);
  MPI_Comm_size(time_comm_, &time_size_);
  MPI_Comm_rank(space_comm_, &space_rank_);
  MPI_Comm_size(space_comm_, &space_size_);

  // TD ring neighbors on time_comm.
  time_prev_ = (time_rank_ - 1 + time_size_) % time_size_;
  time_next_ = (time_rank_ + 1) % time_size_;

  // Spatial neighbors on space_comm.
  space_prev_ = (space_rank_ - 1 + space_size_) % space_size_;
  space_next_ = (space_rank_ + 1) % space_size_;

  // Zone partition (TD zones along X).
  partition_.build(box_, params_.rc, cfg_.n_zones);
  partition_.assign_to_ranks(time_size_);
  partition_.build_zone_neighbors(params_.rc + cfg_.r_skin);
  zone_stream_.assign(static_cast<std::size_t>(partition_.n_zones()), -1);

  // TD ghost zones (across time_comm).
  td_ghost_zones_ = partition_.ghost_zones(time_rank_);
  td_ghost_time_steps_.assign(td_ghost_zones_.size(), 0);
  compute_td_boundary_zones();

  // Spatial decomposition (Y-axis).
  spatial_.build(box_, space_size_, space_rank_,
                 params_.rc + cfg_.r_skin);

  // Initialize NVT thermostat if requested.
  if (cfg_.t_target > 0) {
    integrator::NHCConfig nhc_cfg;
    nhc_cfg.t_target = cfg_.t_target;
    nhc_cfg.t_period = cfg_.t_period;
    nhc_cfg.chain_length = cfg_.nhc_length;
    i32 n_dof = 3 * natoms_global_ - 3;  // global DOF
    nhc_ = std::make_unique<integrator::NoseHooverChain>(nhc_cfg, cfg_.dt,
                                                         n_dof);
  }
}

HybridPipelineScheduler::~HybridPipelineScheduler() {
  if (space_comm_ != MPI_COMM_NULL) MPI_Comm_free(&space_comm_);
  if (time_comm_ != MPI_COMM_NULL) MPI_Comm_free(&time_comm_);
  if (cart_comm_ != MPI_COMM_NULL) MPI_Comm_free(&cart_comm_);
}

void HybridPipelineScheduler::compute_td_boundary_zones() {
  send_to_time_next_.clear();
  send_to_time_prev_.clear();

  for (i32 z = 0; z < partition_.n_zones(); ++z) {
    if (!partition_.is_local(z, time_rank_)) continue;
    bool to_next = false, to_prev = false;
    for (i32 nz : partition_.zone_neighbors(z)) {
      if (nz == z) continue;
      i32 owner = partition_.owner_rank(nz);
      if (owner == time_next_) to_next = true;
      if (owner == time_prev_) to_prev = true;
    }
    if (to_next) send_to_time_next_.push_back(z);
    if (to_prev) send_to_time_prev_.push_back(z);
  }
}

i32 HybridPipelineScheduler::td_ghost_index(i32 zone_id) const {
  for (std::size_t i = 0; i < td_ghost_zones_.size(); ++i) {
    if (td_ghost_zones_[i] == zone_id) return static_cast<i32>(i);
  }
  return -1;
}

void HybridPipelineScheduler::update_td_ghost_time_step(i32 zone_id,
                                                         i32 time_step) {
  i32 idx = td_ghost_index(zone_id);
  if (idx >= 0) {
    td_ghost_time_steps_[static_cast<std::size_t>(idx)] = time_step;
  }
}

i32 HybridPipelineScheduler::zone_time_step(i32 z_id) const {
  if (partition_.is_local(z_id, time_rank_)) {
    return partition_.zones()[static_cast<std::size_t>(z_id)].time_step;
  }
  i32 idx = td_ghost_index(z_id);
  if (idx >= 0) return td_ghost_time_steps_[static_cast<std::size_t>(idx)];
  return 0;
}

// ---- Upload ----

void HybridPipelineScheduler::upload(
    const Vec3* positions, const Vec3* velocities, const Vec3* forces,
    const i32* types, const i32* ids, const real* masses, i32 n_masses) {
  auto n = static_cast<std::size_t>(natoms_global_);

  // Make working copies of all atom data.
  h_pos_.assign(positions, positions + n);
  h_vel_.assign(velocities, velocities + n);
  h_forces_.assign(forces, forces + n);
  h_types_.assign(types, types + n);
  h_ids_.assign(ids, ids + n);

  // Partition into owned atoms (in Y-subdomain) first.
  n_owned_ = spatial_.partition_atoms(h_pos_.data(), h_vel_.data(),
                                      h_forces_.data(), h_types_.data(),
                                      h_ids_.data(), natoms_global_);

  // Truncate to owned atoms only.
  h_pos_.resize(static_cast<std::size_t>(n_owned_));
  h_vel_.resize(static_cast<std::size_t>(n_owned_));
  h_forces_.resize(static_cast<std::size_t>(n_owned_));
  h_types_.resize(static_cast<std::size_t>(n_owned_));
  h_ids_.resize(static_cast<std::size_t>(n_owned_));

  // Sort owned atoms by zone (X-axis zone partition).
  partition_.assign_atoms(h_pos_.data(), h_vel_.data(), h_forces_.data(),
                          h_types_.data(), h_ids_.data(), n_owned_);

  // Identify ghost atoms to send to spatial neighbors.
  spatial_.identify_send_ghosts(h_pos_.data(), n_owned_,
                                send_prev_indices_, send_next_indices_);

  // Perform initial halo exchange to get ghost atoms from spatial neighbors.
  // Pack send buffers for prev and next.
  std::vector<Vec3> send_prev_pos, send_prev_vel;
  std::vector<i32> send_prev_types, send_prev_ids;
  for (i32 idx : send_prev_indices_) {
    auto si = static_cast<std::size_t>(idx);
    send_prev_pos.push_back(h_pos_[si]);
    send_prev_vel.push_back(h_vel_[si]);
    send_prev_types.push_back(h_types_[si]);
    send_prev_ids.push_back(h_ids_[si]);
  }

  std::vector<Vec3> send_next_pos, send_next_vel;
  std::vector<i32> send_next_types, send_next_ids;
  for (i32 idx : send_next_indices_) {
    auto si = static_cast<std::size_t>(idx);
    send_next_pos.push_back(h_pos_[si]);
    send_next_vel.push_back(h_vel_[si]);
    send_next_types.push_back(h_types_[si]);
    send_next_ids.push_back(h_ids_[si]);
  }

  // Exchange counts with spatial neighbors.
  i32 send_next_count = static_cast<i32>(send_next_pos.size());
  i32 recv_prev_count = 0;
  MPI_Sendrecv(&send_next_count, 1, MPI_INT, space_next_, 100,
               &recv_prev_count, 1, MPI_INT, space_prev_, 100,
               space_comm_, MPI_STATUS_IGNORE);

  i32 send_prev_count = static_cast<i32>(send_prev_pos.size());
  i32 recv_next_count = 0;
  MPI_Sendrecv(&send_prev_count, 1, MPI_INT, space_prev_, 101,
               &recv_next_count, 1, MPI_INT, space_next_, 101,
               space_comm_, MPI_STATUS_IGNORE);

  // Receive ghost atoms.
  std::vector<Vec3> recv_prev_pos(static_cast<std::size_t>(recv_prev_count));
  std::vector<Vec3> recv_prev_vel(static_cast<std::size_t>(recv_prev_count));
  std::vector<i32> recv_prev_types(static_cast<std::size_t>(recv_prev_count));
  std::vector<i32> recv_prev_ids(static_cast<std::size_t>(recv_prev_count));

  std::vector<Vec3> recv_next_pos(static_cast<std::size_t>(recv_next_count));
  std::vector<Vec3> recv_next_vel(static_cast<std::size_t>(recv_next_count));
  std::vector<i32> recv_next_types(static_cast<std::size_t>(recv_next_count));
  std::vector<i32> recv_next_ids(static_cast<std::size_t>(recv_next_count));

  auto vec3_bytes = [](i32 count) {
    return static_cast<int>(static_cast<std::size_t>(count) * sizeof(Vec3));
  };
  auto i32_bytes = [](i32 count) {
    return static_cast<int>(static_cast<std::size_t>(count) * sizeof(i32));
  };

  // Exchange positions.
  MPI_Sendrecv(send_next_pos.data(), vec3_bytes(send_next_count), MPI_BYTE,
               space_next_, 110, recv_prev_pos.data(),
               vec3_bytes(recv_prev_count), MPI_BYTE, space_prev_, 110,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_pos.data(), vec3_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 111, recv_next_pos.data(),
               vec3_bytes(recv_next_count), MPI_BYTE, space_next_, 111,
               space_comm_, MPI_STATUS_IGNORE);

  // Exchange velocities.
  MPI_Sendrecv(send_next_vel.data(), vec3_bytes(send_next_count), MPI_BYTE,
               space_next_, 120, recv_prev_vel.data(),
               vec3_bytes(recv_prev_count), MPI_BYTE, space_prev_, 120,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_vel.data(), vec3_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 121, recv_next_vel.data(),
               vec3_bytes(recv_next_count), MPI_BYTE, space_next_, 121,
               space_comm_, MPI_STATUS_IGNORE);

  // Exchange types.
  MPI_Sendrecv(send_next_types.data(), i32_bytes(send_next_count), MPI_BYTE,
               space_next_, 130, recv_prev_types.data(),
               i32_bytes(recv_prev_count), MPI_BYTE, space_prev_, 130,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_types.data(), i32_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 131, recv_next_types.data(),
               i32_bytes(recv_next_count), MPI_BYTE, space_next_, 131,
               space_comm_, MPI_STATUS_IGNORE);

  // Exchange ids.
  MPI_Sendrecv(send_next_ids.data(), i32_bytes(send_next_count), MPI_BYTE,
               space_next_, 140, recv_prev_ids.data(),
               i32_bytes(recv_prev_count), MPI_BYTE, space_prev_, 140,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_ids.data(), i32_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 141, recv_next_ids.data(),
               i32_bytes(recv_next_count), MPI_BYTE, space_next_, 141,
               space_comm_, MPI_STATUS_IGNORE);

  // Collect all received ghost atoms, then deduplicate by ID.
  // Dedup is needed when space_prev == space_next (2-rank case) where
  // the same atom can arrive via both send_to_prev and send_to_next.
  struct GhostAtom { Vec3 pos, vel; i32 type, id; };
  std::vector<GhostAtom> all_ghosts;
  all_ghosts.reserve(
      static_cast<std::size_t>(recv_prev_count + recv_next_count));

  for (i32 i = 0; i < recv_prev_count; ++i) {
    auto si = static_cast<std::size_t>(i);
    all_ghosts.push_back(
        {recv_prev_pos[si], recv_prev_vel[si], recv_prev_types[si],
         recv_prev_ids[si]});
  }
  for (i32 i = 0; i < recv_next_count; ++i) {
    auto si = static_cast<std::size_t>(i);
    all_ghosts.push_back(
        {recv_next_pos[si], recv_next_vel[si], recv_next_types[si],
         recv_next_ids[si]});
  }

  // Deduplicate by ID (keep first occurrence).
  {
    std::vector<GhostAtom> unique;
    unique.reserve(all_ghosts.size());
    std::vector<bool> seen(static_cast<std::size_t>(natoms_global_) + 1, false);
    for (auto& g : all_ghosts) {
      if (!seen[static_cast<std::size_t>(g.id)]) {
        seen[static_cast<std::size_t>(g.id)] = true;
        unique.push_back(g);
      }
    }
    all_ghosts = std::move(unique);
  }

  n_ghost_ = static_cast<i32>(all_ghosts.size());
  n_total_ = n_owned_ + n_ghost_;

  auto total = static_cast<std::size_t>(n_total_);
  h_pos_.resize(total);
  h_vel_.resize(total);
  h_forces_.resize(total);
  h_types_.resize(total);
  h_ids_.resize(total);

  auto off = static_cast<std::size_t>(n_owned_);
  for (std::size_t i = 0; i < all_ghosts.size(); ++i) {
    h_pos_[off + i] = all_ghosts[i].pos;
    h_vel_[off + i] = all_ghosts[i].vel;
    h_forces_[off + i] = Vec3{0, 0, 0};
    h_types_[off + i] = all_ghosts[i].type;
    h_ids_[off + i] = all_ghosts[i].id;
  }

  // Upload to GPU. Allocate for natoms_global_ to avoid destructive resize
  // in upload_ghosts() when ghost count changes during the simulation.
  auto cap = static_cast<std::size_t>(natoms_global_);
  d_pos_.resize(cap);
  d_vel_.resize(cap);
  d_forces_.resize(cap);
  d_types_.resize(cap);
  d_ids_.resize(cap);
  d_masses_.resize(static_cast<std::size_t>(n_masses));

  d_pos_.copy_from_host(h_pos_.data(), total);
  d_vel_.copy_from_host(h_vel_.data(), total);
  d_forces_.copy_from_host(h_forces_.data(), total);
  d_types_.copy_from_host(h_types_.data(), total);
  d_ids_.copy_from_host(h_ids_.data(), total);
  d_masses_.copy_from_host(masses, static_cast<std::size_t>(n_masses));

  // Initialize zone states.
  for (auto& z : partition_.zones()) {
    z.state = ZoneState::Ready;
    z.time_step = 0;
  }
  std::fill(td_ghost_time_steps_.begin(), td_ghost_time_steps_.end(), 0);
  needs_rebuild_ = true;
}

// ---- Halo exchange (spatial ghost positions update) ----

void HybridPipelineScheduler::halo_exchange() {
  if (space_size_ == 1) return;

  // Download owned positions from GPU.
  auto no = static_cast<std::size_t>(n_owned_);
  d_pos_.copy_to_host(h_pos_.data(), no);
  d_vel_.copy_to_host(h_vel_.data(), no);

  // Re-identify ghost atoms (positions may have shifted after integration).
  spatial_.identify_send_ghosts(h_pos_.data(), n_owned_,
                                send_prev_indices_, send_next_indices_);

  // Pack send buffers.
  std::vector<Vec3> send_next_pos, send_prev_pos;
  std::vector<Vec3> send_next_vel, send_prev_vel;
  for (i32 idx : send_next_indices_) {
    auto si = static_cast<std::size_t>(idx);
    send_next_pos.push_back(h_pos_[si]);
    send_next_vel.push_back(h_vel_[si]);
  }
  for (i32 idx : send_prev_indices_) {
    auto si = static_cast<std::size_t>(idx);
    send_prev_pos.push_back(h_pos_[si]);
    send_prev_vel.push_back(h_vel_[si]);
  }

  // Exchange counts.
  i32 send_next_count = static_cast<i32>(send_next_pos.size());
  i32 recv_prev_count = 0;
  MPI_Sendrecv(&send_next_count, 1, MPI_INT, space_next_, 200,
               &recv_prev_count, 1, MPI_INT, space_prev_, 200,
               space_comm_, MPI_STATUS_IGNORE);

  i32 send_prev_count = static_cast<i32>(send_prev_pos.size());
  i32 recv_next_count = 0;
  MPI_Sendrecv(&send_prev_count, 1, MPI_INT, space_prev_, 201,
               &recv_next_count, 1, MPI_INT, space_next_, 201,
               space_comm_, MPI_STATUS_IGNORE);

  // Receive ghost positions/velocities into temporary buffers.
  std::vector<Vec3> recv_prev_pos(static_cast<std::size_t>(recv_prev_count));
  std::vector<Vec3> recv_prev_vel(static_cast<std::size_t>(recv_prev_count));
  std::vector<Vec3> recv_next_pos(static_cast<std::size_t>(recv_next_count));
  std::vector<Vec3> recv_next_vel(static_cast<std::size_t>(recv_next_count));

  auto vec3_bytes = [](i32 n) {
    return static_cast<int>(static_cast<std::size_t>(n) * sizeof(Vec3));
  };

  MPI_Sendrecv(send_next_pos.data(), vec3_bytes(send_next_count), MPI_BYTE,
               space_next_, 210, recv_prev_pos.data(),
               vec3_bytes(recv_prev_count), MPI_BYTE, space_prev_, 210,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_pos.data(), vec3_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 211, recv_next_pos.data(),
               vec3_bytes(recv_next_count), MPI_BYTE, space_next_, 211,
               space_comm_, MPI_STATUS_IGNORE);

  MPI_Sendrecv(send_next_vel.data(), vec3_bytes(send_next_count), MPI_BYTE,
               space_next_, 220, recv_prev_vel.data(),
               vec3_bytes(recv_prev_count), MPI_BYTE, space_prev_, 220,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_vel.data(), vec3_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 221, recv_next_vel.data(),
               vec3_bytes(recv_next_count), MPI_BYTE, space_next_, 221,
               space_comm_, MPI_STATUS_IGNORE);

  // Also exchange atom IDs for deduplication (with P_space=2, prev==next
  // and atoms near both Y-boundaries arrive via both channels).
  std::vector<i32> send_next_ids_h, send_prev_ids_h;
  for (i32 idx : send_next_indices_) {
    send_next_ids_h.push_back(h_ids_[static_cast<std::size_t>(idx)]);
  }
  for (i32 idx : send_prev_indices_) {
    send_prev_ids_h.push_back(h_ids_[static_cast<std::size_t>(idx)]);
  }

  // Exchange IDs.
  std::vector<i32> recv_prev_ids_h(static_cast<std::size_t>(recv_prev_count));
  std::vector<i32> recv_next_ids_h(static_cast<std::size_t>(recv_next_count));

  auto i32_bytes = [](i32 count) {
    return static_cast<int>(static_cast<std::size_t>(count) * sizeof(i32));
  };

  MPI_Sendrecv(send_next_ids_h.data(), i32_bytes(send_next_count), MPI_BYTE,
               space_next_, 230, recv_prev_ids_h.data(),
               i32_bytes(recv_prev_count), MPI_BYTE, space_prev_, 230,
               space_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_ids_h.data(), i32_bytes(send_prev_count), MPI_BYTE,
               space_prev_, 231, recv_next_ids_h.data(),
               i32_bytes(recv_next_count), MPI_BYTE, space_next_, 231,
               space_comm_, MPI_STATUS_IGNORE);

  // Collect and deduplicate.
  struct HaloFull { Vec3 pos, vel; i32 id; };
  std::vector<HaloFull> deduped;
  deduped.reserve(
      static_cast<std::size_t>(recv_prev_count + recv_next_count));
  std::vector<bool> seen(static_cast<std::size_t>(natoms_global_) + 1, false);

  for (i32 i = 0; i < recv_prev_count; ++i) {
    auto si = static_cast<std::size_t>(i);
    i32 id = recv_prev_ids_h[si];
    if (!seen[static_cast<std::size_t>(id)]) {
      seen[static_cast<std::size_t>(id)] = true;
      deduped.push_back({recv_prev_pos[si], recv_prev_vel[si], id});
    }
  }
  for (i32 i = 0; i < recv_next_count; ++i) {
    auto si = static_cast<std::size_t>(i);
    i32 id = recv_next_ids_h[si];
    if (!seen[static_cast<std::size_t>(id)]) {
      seen[static_cast<std::size_t>(id)] = true;
      deduped.push_back({recv_next_pos[si], recv_next_vel[si], id});
    }
  }

  n_ghost_ = static_cast<i32>(deduped.size());
  n_total_ = n_owned_ + n_ghost_;
  auto total = static_cast<std::size_t>(n_total_);
  h_pos_.resize(total);
  h_vel_.resize(total);
  h_ids_.resize(total);

  auto off = static_cast<std::size_t>(n_owned_);
  for (std::size_t i = 0; i < deduped.size(); ++i) {
    h_pos_[off + i] = deduped[i].pos;
    h_vel_[off + i] = deduped[i].vel;
    h_ids_[off + i] = deduped[i].id;
  }

  upload_ghosts();
  ++stats_.halo_exchanges;
}

void HybridPipelineScheduler::upload_ghosts() {
  // Device buffers are pre-allocated for natoms_global_ in upload(), so no
  // resize is ever needed here (which would destroy owned atom data).
  auto ghost_off = static_cast<std::size_t>(n_owned_);
  auto ng = static_cast<std::size_t>(n_ghost_);
  if (ng > 0) {
    d_pos_.copy_from_host(h_pos_.data() + ghost_off, ng, ghost_off);
    d_vel_.copy_from_host(h_vel_.data() + ghost_off, ng, ghost_off);
  }
}

// ---- TD boundary exchange (on time_comm) ----

void HybridPipelineScheduler::td_exchange_boundary_data() {
  if (time_size_ == 1) return;

  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  const auto& zones = partition_.zones();

  auto pack_zones = [&](const std::vector<i32>& zone_ids,
                        std::vector<char>& buf) {
    std::size_t total = sizeof(i32);
    for (i32 z_id : zone_ids) {
      auto sz = static_cast<std::size_t>(z_id);
      total += packed_zone_size(zones[sz].natoms_in_zone);
    }
    buf.resize(total);

    char* p = buf.data();
    i32 nz = static_cast<i32>(zone_ids.size());
    std::memcpy(p, &nz, sizeof(i32));
    p += sizeof(i32);

    for (i32 z_id : zone_ids) {
      auto sz = static_cast<std::size_t>(z_id);
      const auto& z = zones[sz];
      i32 count = z.natoms_in_zone;

      std::vector<Vec3> pos(static_cast<std::size_t>(count));
      std::vector<Vec3> vel(static_cast<std::size_t>(count));
      d_pos_.copy_to_host(pos.data(), static_cast<std::size_t>(count),
                          static_cast<std::size_t>(z.atom_offset));
      d_vel_.copy_to_host(vel.data(), static_cast<std::size_t>(count),
                          static_cast<std::size_t>(z.atom_offset));

      pack_zone(p, z_id, z.time_step, count, pos.data(), vel.data());
      p += static_cast<std::ptrdiff_t>(packed_zone_size(count));
    }
  };

  auto unpack_zones = [&](const std::vector<char>& buf) {
    const char* p = buf.data();
    i32 nz = 0;
    std::memcpy(&nz, p, sizeof(i32));
    p += sizeof(i32);

    for (i32 i = 0; i < nz; ++i) {
      i32 z_id = 0, ts = 0, nat = 0;
      std::memcpy(&z_id, p, sizeof(i32));
      std::memcpy(&ts, p + sizeof(i32), sizeof(i32));
      std::memcpy(&nat, p + 2 * sizeof(i32), sizeof(i32));

      std::vector<Vec3> pos(static_cast<std::size_t>(nat));
      std::vector<Vec3> vel(static_cast<std::size_t>(nat));
      unpack_zone(p, z_id, ts, nat, pos.data(), vel.data());
      p += static_cast<std::ptrdiff_t>(packed_zone_size(nat));

      update_td_ghost_time_step(z_id, ts);

      auto sz = static_cast<std::size_t>(z_id);
      const auto& z = zones[sz];
      d_pos_.copy_from_host(pos.data(), static_cast<std::size_t>(nat),
                            static_cast<std::size_t>(z.atom_offset));
      d_vel_.copy_from_host(vel.data(), static_cast<std::size_t>(nat),
                            static_cast<std::size_t>(z.atom_offset));
    }
  };

  std::vector<char> send_next_buf, send_prev_buf;
  std::vector<char> recv_prev_buf, recv_next_buf;

  pack_zones(send_to_time_next_, send_next_buf);
  pack_zones(send_to_time_prev_, send_prev_buf);

  // Exchange sizes.
  i32 send_next_size = static_cast<i32>(send_next_buf.size());
  i32 recv_prev_size = 0;
  MPI_Sendrecv(&send_next_size, 1, MPI_INT, time_next_, 310,
               &recv_prev_size, 1, MPI_INT, time_prev_, 310,
               time_comm_, MPI_STATUS_IGNORE);

  i32 send_prev_size = static_cast<i32>(send_prev_buf.size());
  i32 recv_next_size = 0;
  MPI_Sendrecv(&send_prev_size, 1, MPI_INT, time_prev_, 320,
               &recv_next_size, 1, MPI_INT, time_next_, 320,
               time_comm_, MPI_STATUS_IGNORE);

  recv_prev_buf.resize(static_cast<std::size_t>(recv_prev_size));
  recv_next_buf.resize(static_cast<std::size_t>(recv_next_size));

  // Exchange data.
  MPI_Sendrecv(send_next_buf.data(), send_next_size, MPI_BYTE,
               time_next_, 330, recv_prev_buf.data(), recv_prev_size,
               MPI_BYTE, time_prev_, 330, time_comm_, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_prev_buf.data(), send_prev_size, MPI_BYTE,
               time_prev_, 340, recv_next_buf.data(), recv_next_size,
               MPI_BYTE, time_next_, 340, time_comm_, MPI_STATUS_IGNORE);

  if (recv_prev_size > 0) unpack_zones(recv_prev_buf);
  if (recv_next_size > 0) unpack_zones(recv_next_buf);

  ++stats_.td_exchanges;
}

// ---- Core scheduler logic ----

bool HybridPipelineScheduler::check_deps(i32 z_id) const {
  ++const_cast<HybridStats&>(stats_).dep_check_calls;

  const auto& z = partition_.zones()[static_cast<std::size_t>(z_id)];
  i32 target = z.time_step;

  for (i32 nz_id : partition_.zone_neighbors(z_id)) {
    if (nz_id == z_id) continue;
    if (zone_time_step(nz_id) < target) {
      ++const_cast<HybridStats&>(stats_).dep_check_failures;
      return false;
    }
    if (partition_.is_local(nz_id, time_rank_)) {
      const auto& nz = partition_.zones()[static_cast<std::size_t>(nz_id)];
      if (nz.state == ZoneState::Computing) {
        ++const_cast<HybridStats&>(stats_).dep_check_failures;
        return false;
      }
    }
  }
  return true;
}

void HybridPipelineScheduler::launch_zone_step(i32 z_id, i32 stream_id) {
  auto& z = partition_.zones()[static_cast<std::size_t>(z_id)];
  cudaStream_t s = streams_.stream(stream_id);
  i32 first = z.atom_offset;
  i32 count = z.natoms_in_zone;

  integrator::device_half_kick_zone(d_vel_.data(), d_forces_.data(),
                                    d_types_.data(), d_masses_.data(), first,
                                    count, cfg_.dt, s);
  integrator::device_drift_zone(d_pos_.data(), d_vel_.data(), first, count,
                                cfg_.dt, box_, s);
  integrator::device_zero_forces_zone(d_forces_.data(), first, count, s);
  potentials::compute_morse_gpu_zone(
      d_pos_.data(), d_forces_.data(), nlist_.d_neighbors(), nlist_.d_offsets(),
      nlist_.d_counts(), first, count, box_, params_, nullptr, s);
  integrator::device_half_kick_zone(d_vel_.data(), d_forces_.data(),
                                    d_types_.data(), d_masses_.data(), first,
                                    count, cfg_.dt, s);

  streams_.record_event(stream_id);
  zone_stream_[static_cast<std::size_t>(z_id)] = stream_id;
  z.transition_to(ZoneState::Computing);
  ++stats_.kernel_launches;
}

void HybridPipelineScheduler::poll_completions() {
  auto& zones = partition_.zones();
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    if (!partition_.is_local(z_id, time_rank_)) continue;
    auto sz = static_cast<std::size_t>(z_id);
    if (zones[sz].state != ZoneState::Computing) continue;
    i32 sid = zone_stream_[sz];
    if (sid >= 0 && streams_.is_complete(sid)) {
      zones[sz].transition_to(ZoneState::Done);
      streams_.release(sid);
      zone_stream_[sz] = -1;
      zones[sz].state = ZoneState::Ready;
    }
  }
}

bool HybridPipelineScheduler::tick() {
  ++stats_.ticks;
  poll_completions();

  bool any_launched = false;
  auto& zones = partition_.zones();
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    if (!partition_.is_local(z_id, time_rank_)) continue;
    auto sz = static_cast<std::size_t>(z_id);
    if (zones[sz].state != ZoneState::Ready) continue;
    if (!check_deps(z_id)) continue;

    i32 sid = streams_.try_acquire();
    if (sid < 0) break;

    launch_zone_step(z_id, sid);
    any_launched = true;

    if (cfg_.deterministic) {
      TDMD_CUDA_CHECK(cudaStreamSynchronize(streams_.stream(sid)));
      poll_completions();
    }
  }
  return any_launched;
}

void HybridPipelineScheduler::rebuild_nlist() {
  if (!needs_rebuild_) return;
  // Build neighbor list over ALL atoms (owned + ghost) so that ghost atoms
  // appear as neighbors of owned atoms near subdomain boundaries.
  nlist_.build(d_pos_.data(), n_total_, box_, params_.rc, cfg_.r_skin);
  needs_rebuild_ = false;
}

void HybridPipelineScheduler::run_until(i32 target_step) {
  // Initial halo exchange + neighbor list build.
  halo_exchange();
  rebuild_nlist();

  // Initial force compute for owned atoms.
  d_forces_.zero();
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), n_owned_, box_, params_,
                                nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  i32 last_rebuild_step = 0;
  while (min_global_time_step() < target_step) {
    if (nhc_) {
      // NVT mode: drain, thermostat, advance, thermostat.
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      td_exchange_boundary_data();
      halo_exchange();

      // Pre-step NHC: compute local KE → Allreduce → scale.
      real local_ke = integrator::device_compute_ke(
          d_vel_.data(), d_types_.data(), d_masses_.data(), n_owned_);
      real global_ke = 0;
      MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM,
                     world_comm_);

      real scale1 = nhc_->half_step(global_ke);
      integrator::device_scale_velocities(d_vel_.data(), n_owned_, scale1);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());

      // Advance one step.
      tick();
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      td_exchange_boundary_data();
      halo_exchange();

      // Post-step NHC.
      local_ke = integrator::device_compute_ke(d_vel_.data(), d_types_.data(),
                                               d_masses_.data(), n_owned_);
      global_ke = 0;
      MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM,
                     world_comm_);

      real scale2 = nhc_->half_step(global_ke);
      integrator::device_scale_velocities(d_vel_.data(), n_owned_, scale2);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      // NVE mode: normal pipelined execution.
      tick();
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      td_exchange_boundary_data();
      halo_exchange();
    }

    // Rebuild neighbor list periodically.
    i32 global_min = min_global_time_step();
    if (global_min - last_rebuild_step >= cfg_.rebuild_every) {
      MPI_Barrier(world_comm_);
      needs_rebuild_ = true;
      rebuild_nlist();
      last_rebuild_step = global_min;
    }
  }

  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(world_comm_);
}

i32 HybridPipelineScheduler::min_local_time_step() const noexcept {
  i32 mn = std::numeric_limits<i32>::max();
  for (i32 z = 0; z < partition_.n_zones(); ++z) {
    if (!partition_.is_local(z, time_rank_)) continue;
    mn = std::min(mn, partition_.zones()[static_cast<std::size_t>(z)].time_step);
  }
  return mn;
}

i32 HybridPipelineScheduler::min_global_time_step() {
  i32 local_min = min_local_time_step();
  i32 global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, world_comm_);
  return global_min;
}

void HybridPipelineScheduler::download(Vec3* positions, Vec3* velocities,
                                       Vec3* forces, i32* types, i32* ids,
                                       i32 natoms) const {
  // Download only owned atoms. The caller passes the full array, but we fill
  // only the first n_owned_ elements (caller gathers via MPI if needed).
  auto n = static_cast<std::size_t>(std::min(natoms, n_owned_));
  d_pos_.copy_to_host(positions, n);
  d_vel_.copy_to_host(velocities, n);
  d_forces_.copy_to_host(forces, n);
  d_types_.copy_to_host(types, n);
  d_ids_.copy_to_host(ids, n);
}

}  // namespace tdmd::scheduler
