// SPDX-License-Identifier: Apache-2.0
// pipeline_scheduler.cu — M4 TD pipeline scheduler implementation.

#include "pipeline_scheduler.cuh"

#include <algorithm>
#include <cstdio>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"
#include "../integrator/device_nose_hoover.cuh"
#include "../integrator/device_velocity_verlet_zone.cuh"
#include "../potentials/device_morse_zone.cuh"

namespace tdmd::scheduler {

PipelineScheduler::PipelineScheduler(const Box& box, i32 natoms,
                                     const potentials::MorseParams& params,
                                     const PipelineConfig& cfg)
    : cfg_(cfg),
      params_(params),
      box_(box),
      natoms_(natoms),
      streams_(cfg.deterministic ? 1 : cfg.n_streams) {
  // Build zone partition.
  partition_.build(box_, params_.rc, cfg_.n_zones);
  partition_.build_zone_neighbors(params_.rc + cfg_.r_skin);

  zone_stream_.assign(static_cast<std::size_t>(partition_.n_zones()), -1);
  current_dt_ = cfg_.dt;

  // Initialize NVT thermostat if requested.
  if (cfg_.t_target > 0) {
    integrator::NHCConfig nhc_cfg;
    nhc_cfg.t_target = cfg_.t_target;
    nhc_cfg.t_period = cfg_.t_period;
    nhc_cfg.chain_length = cfg_.nhc_length;
    i32 n_dof = 3 * natoms_ - 3;  // translational DOF
    nhc_ = std::make_unique<integrator::NoseHooverChain>(nhc_cfg, cfg_.dt,
                                                         n_dof);
  }
}

void PipelineScheduler::upload(const Vec3* positions, const Vec3* velocities,
                               const Vec3* forces, const i32* types,
                               const i32* ids, const real* masses,
                               i32 n_masses) {
  auto n = static_cast<std::size_t>(natoms_);

  // Sort atoms by zone on host first.
  // Make temporary copies for reordering.
  std::vector<Vec3> h_pos(positions, positions + n);
  std::vector<Vec3> h_vel(velocities, velocities + n);
  std::vector<Vec3> h_forces(forces, forces + n);
  std::vector<i32> h_types(types, types + n);
  std::vector<i32> h_ids(ids, ids + n);

  partition_.assign_atoms(h_pos.data(), h_vel.data(), h_forces.data(),
                          h_types.data(), h_ids.data(), natoms_);

  // Upload to device.
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

  // Set all zones to Ready.
  for (auto& z : partition_.zones()) {
    z.state = ZoneState::Ready;
    z.time_step = 0;
  }

  needs_rebuild_ = true;
}

void PipelineScheduler::rebuild_nlist() {
  if (!needs_rebuild_) return;
  nlist_.build(d_pos_.data(), natoms_, box_, params_.rc, cfg_.r_skin);
  needs_rebuild_ = false;
  steps_since_rebuild_ = 0;
}

bool PipelineScheduler::check_deps(i32 z_id) const {
  ++const_cast<PipelineStats&>(stats_).dep_check_calls;

  const auto& zones = partition_.zones();
  const auto& z = zones[static_cast<std::size_t>(z_id)];
  i32 target_step = z.time_step;  // zone wants to advance TO target_step+1

  const auto& nbrs = partition_.zone_neighbors(z_id);
  for (i32 nz_id : nbrs) {
    if (nz_id == z_id) continue;
    const auto& nz = zones[static_cast<std::size_t>(nz_id)];

    // I-2: neighbor must be at time_step >= target_step (i.e., T-1 where T is
    // the step zone Z is about to compute).
    if (nz.time_step < target_step) {
      ++const_cast<PipelineStats&>(stats_).dep_check_failures;
      return false;
    }

    // Neighbor must not be Computing (data would be in flux).
    // Exception: if they share no atoms in influence region (I-3).
    // For safety, disallow concurrent Computing of neighbors.
    if (nz.state == ZoneState::Computing) {
      ++const_cast<PipelineStats&>(stats_).dep_check_failures;
      return false;
    }
  }
  return true;
}

void PipelineScheduler::launch_zone_step(i32 z_id, i32 stream_id) {
  auto& z = partition_.zones()[static_cast<std::size_t>(z_id)];
  cudaStream_t s = streams_.stream(stream_id);

  i32 first = z.atom_offset;
  i32 count = z.natoms_in_zone;

  // 1. Half-kick (using old forces).
  integrator::device_half_kick_zone(d_vel_.data(), d_forces_.data(),
                                    d_types_.data(), d_masses_.data(), first,
                                    count, current_dt_, s);

  // 2. Drift (update positions).
  integrator::device_drift_zone(d_pos_.data(), d_vel_.data(), first, count,
                                current_dt_, box_, s);

  // 3. Zero forces for this zone.
  integrator::device_zero_forces_zone(d_forces_.data(), first, count, s);

  // 4. Force compute for this zone (reads all positions, writes only zone).
  potentials::compute_morse_gpu_zone(
      d_pos_.data(), d_forces_.data(), nlist_.d_neighbors(), nlist_.d_offsets(),
      nlist_.d_counts(), first, count, box_, params_, nullptr, s);

  // 5. Second half-kick (using new forces).
  integrator::device_half_kick_zone(d_vel_.data(), d_forces_.data(),
                                    d_types_.data(), d_masses_.data(), first,
                                    count, current_dt_, s);

  // Record event.
  streams_.record_event(stream_id);
  zone_stream_[static_cast<std::size_t>(z_id)] = stream_id;

  // Transition: Ready → Computing.
  z.transition_to(ZoneState::Computing);
  ++stats_.kernel_launches;
}

void PipelineScheduler::poll_completions() {
  auto& zones = partition_.zones();
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    auto sz = static_cast<std::size_t>(z_id);
    if (zones[sz].state != ZoneState::Computing) continue;

    i32 sid = zone_stream_[sz];
    if (sid >= 0 && streams_.is_complete(sid)) {
      // Computing → Done (increments time_step).
      zones[sz].transition_to(ZoneState::Done);
      streams_.release(sid);
      zone_stream_[sz] = -1;

      // In M4 single-rank: Done → Ready immediately (skip Send/Free/Recv).
      zones[sz].state = ZoneState::Ready;
    }
  }
}

bool PipelineScheduler::tick() {
  ++stats_.ticks;

  // Poll completions first.
  poll_completions();

  bool any_launched = false;
  auto& zones = partition_.zones();

  // Walk zones in linear order, find Ready ones with deps met.
  for (i32 z_id = 0; z_id < partition_.n_zones(); ++z_id) {
    auto sz = static_cast<std::size_t>(z_id);
    if (zones[sz].state != ZoneState::Ready) continue;

    if (!check_deps(z_id)) continue;

    // Try to get a stream.
    i32 sid = streams_.try_acquire();
    if (sid < 0) break;  // no free stream

    launch_zone_step(z_id, sid);
    any_launched = true;

    // In deterministic mode, synchronize after each zone.
    if (cfg_.deterministic) {
      TDMD_CUDA_CHECK(cudaStreamSynchronize(streams_.stream(sid)));
      poll_completions();
    }
  }

  return any_launched;
}

void PipelineScheduler::run_until(i32 target_step) {
  rebuild_nlist();

  // Initial force compute for all atoms (step 0 forces).
  d_forces_.zero();
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), natoms_, box_, params_,
                                nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  while (min_time_step() < target_step) {
    // Rebuild neighbor list periodically.
    ++steps_since_rebuild_;
    if (steps_since_rebuild_ >= cfg_.rebuild_every) {
      // Drain pipeline first.
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      rebuild_nlist();
    }

    // Adaptive Δt: compute new dt from v_max at the start of each step.
    if (cfg_.adaptive_dt) {
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();
      real vmax = integrator::device_compute_vmax(d_vel_.data(), natoms_);
      if (vmax > 0) {
        current_dt_ = std::min(cfg_.dt_max, cfg_.c2 * params_.rc / vmax);
        current_dt_ = std::max(current_dt_, cfg_.dt_min);
      }
    }

    if (nhc_) {
      // NVT mode: drain pipeline, apply thermostat globally, then advance
      // all zones in one synchronized step. Pipelined NVT is future work.
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();

      // Pre-step NHC half-step: compute KE, get scale factor, scale
      // velocities.
      real ke = integrator::device_compute_ke(d_vel_.data(), d_types_.data(),
                                              d_masses_.data(), natoms_);
      real scale1 = nhc_->half_step(ke);
      integrator::device_scale_velocities(d_vel_.data(), natoms_, scale1);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());

      // Advance all zones (one synchronized step).
      bool progress = tick();
      if (!progress) {
        TDMD_CUDA_CHECK(cudaDeviceSynchronize());
        poll_completions();
      }
      // Drain to ensure all zones finished this step.
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
      poll_completions();

      // Post-step NHC half-step.
      ke = integrator::device_compute_ke(d_vel_.data(), d_types_.data(),
                                         d_masses_.data(), natoms_);
      real scale2 = nhc_->half_step(ke);
      integrator::device_scale_velocities(d_vel_.data(), natoms_, scale2);
      TDMD_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      // NVE mode: normal pipelined execution.
      bool progress = tick();
      if (!progress) {
        // No zone could launch — drain pipeline and try again.
        TDMD_CUDA_CHECK(cudaDeviceSynchronize());
        poll_completions();
      }
    }
  }

  // Final sync.
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  poll_completions();
}

i32 PipelineScheduler::min_time_step() const noexcept {
  i32 mn = std::numeric_limits<i32>::max();
  for (const auto& z : partition_.zones()) {
    mn = std::min(mn, z.time_step);
  }
  return mn;
}

void PipelineScheduler::download(Vec3* positions, Vec3* velocities,
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
