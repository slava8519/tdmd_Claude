// SPDX-License-Identifier: Apache-2.0
// distributed_pipeline_scheduler.cu — M5 multi-rank TD pipeline implementation.
//
// Synchronous MPI_Sendrecv after each tick round. Simple and deadlock-free.

#include "distributed_pipeline_scheduler.cuh"

#include <algorithm>
#include <cstring>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"
#include "../core/mpi_types.hpp"
#include "../integrator/device_nose_hoover.cuh"
#include "../integrator/device_velocity_verlet_zone.cuh"
#include "../potentials/device_morse_zone.cuh"

namespace tdmd::scheduler {

// ---- Pack / unpack helpers for MPI exchange ----

// Buffer layout per zone: [i32 zone_id | i32 time_step | i32 natoms | Vec3[] pos | Vec3[] vel]
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

DistributedPipelineScheduler::DistributedPipelineScheduler(
    const Box& box, i32 natoms, const potentials::MorseParams& params,
    const DistributedConfig& cfg, MPI_Comm comm)
    : cfg_(cfg),
      params_(params),
      box_(box),
      natoms_(natoms),
      comm_(comm),
      streams_(cfg.deterministic ? 1 : cfg.n_streams) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  prev_rank_ = (rank_ - 1 + size_) % size_;
  next_rank_ = (rank_ + 1) % size_;

  partition_.build(box_, params_.rc, cfg_.n_zones);
  partition_.assign_to_ranks(size_);
  partition_.build_zone_neighbors(params_.rc + cfg_.r_skin);

  zone_stream_.assign(static_cast<std::size_t>(partition_.n_zones()), -1);

  ghost_zones_ = partition_.ghost_zones(rank_);
  ghost_time_steps_.assign(ghost_zones_.size(), 0);

  compute_boundary_zones();

  // Note: first_local_atom_ and local_atom_count_ are computed in upload()
  // after assign_atoms() sets valid zone metadata. Do NOT compute them here —
  // zone atom_offset/natoms_in_zone are uninitialized before assign_atoms().

  // Initialize NVT thermostat if requested.
  if (cfg_.t_target > 0) {
    integrator::NHCConfig nhc_cfg;
    nhc_cfg.t_target = cfg_.t_target;
    nhc_cfg.t_period = cfg_.t_period;
    nhc_cfg.chain_length = cfg_.nhc_length;
    i32 n_dof = 3 * natoms_ - 3;  // global DOF
    nhc_ = std::make_unique<integrator::NoseHooverChain>(nhc_cfg, cfg_.dt,
                                                         n_dof);
  }
}

void DistributedPipelineScheduler::compute_boundary_zones() {
  send_to_next_zones_.clear();
  send_to_prev_zones_.clear();

  for (i32 z = 0; z < partition_.n_zones(); ++z) {
    if (!partition_.is_local(z, rank_)) continue;
    bool to_next = false, to_prev = false;
    for (i32 nz : partition_.zone_neighbors(z)) {
      if (nz == z) continue;
      i32 owner = partition_.owner_rank(nz);
      if (owner == next_rank_) to_next = true;
      if (owner == prev_rank_) to_prev = true;
    }
    if (to_next) send_to_next_zones_.push_back(z);
    if (to_prev) send_to_prev_zones_.push_back(z);
  }
}

i32 DistributedPipelineScheduler::ghost_index(i32 zone_id) const {
  for (std::size_t i = 0; i < ghost_zones_.size(); ++i) {
    if (ghost_zones_[i] == zone_id) return static_cast<i32>(i);
  }
  return -1;
}

void DistributedPipelineScheduler::update_ghost_time_step(i32 zone_id,
                                                          i32 time_step) {
  i32 idx = ghost_index(zone_id);
  if (idx >= 0) {
    ghost_time_steps_[static_cast<std::size_t>(idx)] = time_step;
  }
}

i32 DistributedPipelineScheduler::zone_time_step(i32 z_id) const {
  if (partition_.is_local(z_id, rank_)) {
    return partition_.zones()[static_cast<std::size_t>(z_id)].time_step;
  }
  i32 idx = ghost_index(z_id);
  if (idx >= 0) return ghost_time_steps_[static_cast<std::size_t>(idx)];
  return 0;
}

void DistributedPipelineScheduler::upload(
    const Vec3* positions, const Vec3* velocities, const Vec3* forces,
    const i32* types, const i32* ids, const real* masses, i32 n_masses) {
  auto n = static_cast<std::size_t>(natoms_);

  std::vector<Vec3> h_pos(positions, positions + n);
  std::vector<Vec3> h_vel(velocities, velocities + n);
  std::vector<Vec3> h_forces(forces, forces + n);
  std::vector<i32> h_types(types, types + n);
  std::vector<i32> h_ids(ids, ids + n);

  partition_.assign_atoms(h_pos.data(), h_vel.data(), h_forces.data(),
                          h_types.data(), h_ids.data(), natoms_);

  d_pos_.resize(n);
  d_vel_.resize(n);
  d_forces_.resize(n);
  d_types_.resize(n);
  d_ids_.resize(n);
  d_masses_.resize(static_cast<std::size_t>(n_masses));

  d_pos_.copy_from_host(h_pos.data(), n);
  d_vel_.copy_from_host(h_vel.data(), n);
  d_forces_.copy_from_host(h_forces.data(), n);
  d_types_.copy_from_host(h_types.data(), n);
  d_ids_.copy_from_host(h_ids.data(), n);
  d_masses_.copy_from_host(masses, static_cast<std::size_t>(n_masses));

  for (auto& z : partition_.zones()) {
    z.state = ZoneState::Ready;
    z.time_step = 0;
  }
  std::fill(ghost_time_steps_.begin(), ghost_time_steps_.end(), 0);
  needs_rebuild_ = true;

  // Recompute contiguous atom range for locally-owned zones.
  // Must be done AFTER assign_atoms() which sets zone atom_offset/natoms_in_zone.
  first_local_atom_ = 0;
  local_atom_count_ = 0;
  bool found_first = false;
  for (i32 z = 0; z < partition_.n_zones(); ++z) {
    if (partition_.is_local(z, rank_)) {
      const auto& zone = partition_.zones()[static_cast<std::size_t>(z)];
      if (!found_first) {
        first_local_atom_ = zone.atom_offset;
        found_first = true;
      }
      local_atom_count_ += zone.natoms_in_zone;
    }
  }
}

void DistributedPipelineScheduler::rebuild_nlist() {
  if (!needs_rebuild_) return;
  nlist_.build(d_pos_.data(), natoms_, box_, params_.rc, cfg_.r_skin);
  needs_rebuild_ = false;
}

bool DistributedPipelineScheduler::check_deps(i32 z_id) const {
  ++const_cast<DistributedStats&>(stats_).dep_check_calls;

  const auto& z = partition_.zones()[static_cast<std::size_t>(z_id)];
  i32 target = z.time_step;

  for (i32 nz_id : partition_.zone_neighbors(z_id)) {
    if (nz_id == z_id) continue;
    if (zone_time_step(nz_id) < target) {
      ++const_cast<DistributedStats&>(stats_).dep_check_failures;
      return false;
    }
    if (partition_.is_local(nz_id, rank_)) {
      const auto& nz = partition_.zones()[static_cast<std::size_t>(nz_id)];
      if (nz.state == ZoneState::Computing) {
        ++const_cast<DistributedStats&>(stats_).dep_check_failures;
        return false;
      }
    }
  }
  return true;
}

void DistributedPipelineScheduler::launch_zone_step(i32 z_id, i32 stream_id) {
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

void DistributedPipelineScheduler::poll_completions() {
  auto& zones = partition_.zones();
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    if (!partition_.is_local(z_id, rank_)) continue;
    auto sz = static_cast<std::size_t>(z_id);
    if (zones[sz].state != ZoneState::Computing) continue;
    i32 sid = zone_stream_[sz];
    if (sid >= 0 && streams_.is_complete(sid)) {
      zones[sz].transition_to(ZoneState::Done);  // time_step++
      streams_.release(sid);
      zone_stream_[sz] = -1;
      zones[sz].state = ZoneState::Ready;
    }
  }
}

bool DistributedPipelineScheduler::tick() {
  ++stats_.ticks;
  poll_completions();

  bool any_launched = false;
  auto& zones = partition_.zones();
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    if (!partition_.is_local(z_id, rank_)) continue;
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

void DistributedPipelineScheduler::exchange_boundary_data() {
  if (size_ == 1) return;

  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  const auto& zones = partition_.zones();

  // Compute send buffer sizes.
  auto pack_zones = [&](const std::vector<i32>& zone_ids,
                        std::vector<char>& buf) {
    // First pass: compute total size.
    std::size_t total = sizeof(i32);  // number of zones
    for (i32 z_id : zone_ids) {
      auto sz = static_cast<std::size_t>(z_id);
      total += packed_zone_size(zones[sz].natoms_in_zone);
    }
    buf.resize(total);

    // Second pass: pack.
    char* p = buf.data();
    i32 nz = static_cast<i32>(zone_ids.size());
    std::memcpy(p, &nz, sizeof(i32));
    p += sizeof(i32);

    for (i32 z_id : zone_ids) {
      auto sz = static_cast<std::size_t>(z_id);
      const auto& z = zones[sz];
      i32 count = z.natoms_in_zone;

      // Download zone data from GPU.
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
      // Peek header to get natoms.
      std::memcpy(&z_id, p, sizeof(i32));
      std::memcpy(&ts, p + sizeof(i32), sizeof(i32));
      std::memcpy(&nat, p + 2 * sizeof(i32), sizeof(i32));

      std::vector<Vec3> pos(static_cast<std::size_t>(nat));
      std::vector<Vec3> vel(static_cast<std::size_t>(nat));
      unpack_zone(p, z_id, ts, nat, pos.data(), vel.data());
      p += static_cast<std::ptrdiff_t>(packed_zone_size(nat));

      // Update ghost.
      update_ghost_time_step(z_id, ts);

      auto sz = static_cast<std::size_t>(z_id);
      const auto& z = zones[sz];
      d_pos_.copy_from_host(pos.data(), static_cast<std::size_t>(nat),
                            static_cast<std::size_t>(z.atom_offset));
      d_vel_.copy_from_host(vel.data(), static_cast<std::size_t>(nat),
                            static_cast<std::size_t>(z.atom_offset));
    }
  };

  // Pack send buffers.
  std::vector<char> send_to_next_buf, send_to_prev_buf;
  std::vector<char> recv_from_prev_buf, recv_from_next_buf;

  pack_zones(send_to_next_zones_, send_to_next_buf);
  pack_zones(send_to_prev_zones_, send_to_prev_buf);

  // Exchange sizes first, then data.
  i32 send_next_size = static_cast<i32>(send_to_next_buf.size());
  i32 recv_prev_size = 0;
  MPI_Sendrecv(&send_next_size, 1, MPI_INT, next_rank_, 10,
               &recv_prev_size, 1, MPI_INT, prev_rank_, 10, comm_,
               MPI_STATUS_IGNORE);

  i32 send_prev_size = static_cast<i32>(send_to_prev_buf.size());
  i32 recv_next_size = 0;
  MPI_Sendrecv(&send_prev_size, 1, MPI_INT, prev_rank_, 20,
               &recv_next_size, 1, MPI_INT, next_rank_, 20, comm_,
               MPI_STATUS_IGNORE);

  recv_from_prev_buf.resize(static_cast<std::size_t>(recv_prev_size));
  recv_from_next_buf.resize(static_cast<std::size_t>(recv_next_size));

  // Exchange data.
  MPI_Sendrecv(send_to_next_buf.data(), send_next_size, MPI_BYTE,
               next_rank_, 30, recv_from_prev_buf.data(), recv_prev_size,
               MPI_BYTE, prev_rank_, 30, comm_, MPI_STATUS_IGNORE);

  MPI_Sendrecv(send_to_prev_buf.data(), send_prev_size, MPI_BYTE,
               prev_rank_, 40, recv_from_next_buf.data(), recv_next_size,
               MPI_BYTE, next_rank_, 40, comm_, MPI_STATUS_IGNORE);

  // Unpack.
  if (recv_prev_size > 0) unpack_zones(recv_from_prev_buf);
  if (recv_next_size > 0) unpack_zones(recv_from_next_buf);

  ++stats_.exchanges;
}

void DistributedPipelineScheduler::run_until(i32 target_step) {
  rebuild_nlist();

  // Initial force compute.
  d_forces_.zero();
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), natoms_, box_, params_,
                                nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(comm_);

  i32 last_rebuild_step = 0;

  // Use global minimum so all ranks enter/exit the loop together.
  while (min_global_time_step() < target_step) {
    if (nhc_) {
      // NVT mode: drain, thermostat, advance, thermostat.
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      exchange_boundary_data();

      // Pre-step NHC: compute local KE → Allreduce → scale.
      accum_t local_ke = integrator::device_compute_ke(
          d_vel_.data() + first_local_atom_,
          d_types_.data() + first_local_atom_, d_masses_.data(),
          local_atom_count_);
      accum_t global_ke = 0;
      MPI_Allreduce(&local_ke, &global_ke, 1,
                    mpi_type<accum_t>(), MPI_SUM,
                    comm_);

      real scale1 = nhc_->half_step(global_ke);
      integrator::device_scale_velocities_zone(
          d_vel_.data(), first_local_atom_, local_atom_count_, scale1);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());

      // Advance one step.
      tick();
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      exchange_boundary_data();

      // Post-step NHC.
      local_ke = integrator::device_compute_ke(
          d_vel_.data() + first_local_atom_,
          d_types_.data() + first_local_atom_, d_masses_.data(),
          local_atom_count_);
      global_ke = 0;
      MPI_Allreduce(&local_ke, &global_ke, 1,
                    mpi_type<accum_t>(), MPI_SUM,
                    comm_);

      real scale2 = nhc_->half_step(global_ke);
      integrator::device_scale_velocities_zone(
          d_vel_.data(), first_local_atom_, local_atom_count_, scale2);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      // NVE mode: normal pipelined execution.
      tick();
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      exchange_boundary_data();
    }

    // Rebuild neighbor list (synchronized via min_global).
    i32 global_min = min_global_time_step();
    if (global_min - last_rebuild_step >= cfg_.rebuild_every) {
      MPI_Barrier(comm_);
      needs_rebuild_ = true;
      rebuild_nlist();
      last_rebuild_step = global_min;
    }
  }

  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(comm_);
}

i32 DistributedPipelineScheduler::min_local_time_step() const noexcept {
  i32 mn = std::numeric_limits<i32>::max();
  for (i32 z = 0; z < partition_.n_zones(); ++z) {
    if (!partition_.is_local(z, rank_)) continue;
    mn = std::min(mn, partition_.zones()[static_cast<std::size_t>(z)].time_step);
  }
  return mn;
}

i32 DistributedPipelineScheduler::min_global_time_step() {
  i32 local_min = min_local_time_step();
  i32 global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, comm_);
  return global_min;
}

void DistributedPipelineScheduler::download(Vec3* positions, Vec3* velocities,
                                            Vec3* forces, i32* types, i32* ids,
                                            i32 natoms) const {
  auto n = static_cast<std::size_t>(natoms);
  d_pos_.copy_to_host(positions, n);
  d_vel_.copy_to_host(velocities, n);
  d_forces_.copy_to_host(forces, n);
  d_types_.copy_to_host(types, n);
  d_ids_.copy_to_host(ids, n);
}

}  // namespace tdmd::scheduler
