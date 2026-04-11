// SPDX-License-Identifier: Apache-2.0
// fast_pipeline_scheduler.cu — batched single-GPU scheduler (ADR 0005 Phase 2).

#include "fast_pipeline_scheduler.cuh"

#include <cassert>

#include "../core/error.hpp"
#include "../integrator/device_velocity_verlet.cuh"
#include "../potentials/device_eam.cuh"
#include "../potentials/eam_alloy.hpp"

namespace tdmd::scheduler {

FastPipelineScheduler::FastPipelineScheduler(
    const Box& box, i32 natoms, const potentials::MorseParams& params,
    const FastPipelineConfig& cfg)
    : box_(box),
      natoms_(natoms),
      cfg_(cfg),
      potential_kind_(PotentialKind::Morse),
      morse_params_(params),
      cutoff_(params.rc) {
  TDMD_CUDA_CHECK(cudaStreamCreate(&compute_stream_));
}

FastPipelineScheduler::FastPipelineScheduler(
    const Box& box, i32 natoms, const potentials::EamAlloy& eam,
    const FastPipelineConfig& cfg)
    : box_(box),
      natoms_(natoms),
      cfg_(cfg),
      potential_kind_(PotentialKind::Eam),
      eam_(std::make_unique<potentials::DeviceEam>()),
      cutoff_(eam.cutoff()) {
  TDMD_CUDA_CHECK(cudaStreamCreate(&compute_stream_));
  // Upload spline tables to device. From this point on the scheduler
  // holds no reference back to the host-side EamAlloy.
  eam_->upload_tables(eam);
}

FastPipelineScheduler::~FastPipelineScheduler() {
  if (compute_stream_) {
    cudaStreamSynchronize(compute_stream_);
    cudaStreamDestroy(compute_stream_);
  }
}

void FastPipelineScheduler::upload(const PositionVec* positions,
                                   const VelocityVec* velocities,
                                   const ForceVec* forces, const i32* types,
                                   const i32* ids, const real* masses,
                                   i32 n_types) {
  auto n = static_cast<std::size_t>(natoms_);

  d_pos_.resize(n);
  d_vel_.resize(n);
  d_forces_.resize(n);
  d_types_.resize(n);
  d_ids_.resize(n);
  d_masses_.resize(static_cast<std::size_t>(n_types));

  d_pos_.copy_from_host(positions, n);
  d_vel_.copy_from_host(velocities, n);
  d_forces_.copy_from_host(forces, n);
  d_types_.copy_from_host(types, n);
  d_ids_.copy_from_host(ids, n);
  d_masses_.copy_from_host(masses, static_cast<std::size_t>(n_types));

  initial_forces_computed_ = false;
  current_step_ = 0;
  steps_since_rebuild_ = 0;
}

void FastPipelineScheduler::download(PositionVec* positions,
                                     VelocityVec* velocities,
                                     ForceVec* forces, i32* types, i32* ids,
                                     i32 natoms) const {
  auto n = static_cast<std::size_t>(natoms);
  d_pos_.copy_to_host(positions, n);
  d_vel_.copy_to_host(velocities, n);
  d_forces_.copy_to_host(forces, n);
  d_types_.copy_to_host(types, n);
  d_ids_.copy_to_host(ids, n);
}

void FastPipelineScheduler::step(bool first_step_after_bootstrap) {
  ++stats_.ticks;
  switch (potential_kind_) {
    case PotentialKind::Morse:
      step_morse(first_step_after_bootstrap);
      break;
    case PotentialKind::Eam:
      step_eam(first_step_after_bootstrap);
      break;
  }
  ++current_step_;
  ++steps_since_rebuild_;
}

void FastPipelineScheduler::step_morse(bool first_step_after_bootstrap) {
  // OPT-FUSE-1c: on the first step of a run_until window, velocities are at
  // integer time t and we bootstrap with a half-dt kick fused with drift.
  // On every subsequent step, the closing kick2 of the previous step is
  // fused with the opening kick1 of the current step into a single full-dt
  // kick, saving one launch per step. OPT-FUSE-1a register-resident
  // velocity hand-off still applies inside the fused kernel.
  if (first_step_after_bootstrap) {
    integrator::device_fused_half_kick_drift(
        d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
        d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  } else {
    integrator::device_fused_full_kick_drift(
        d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
        d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  }
  ++stats_.kernel_launches;

  // Neighbor list rebuild (amortized, every rebuild_every steps).
  // OPT-1: prefix sum now runs on the device via CUB; build() still performs
  // one 8-byte D2H for total_pairs + max_neighbors, so there is a single
  // tiny stream sync inside it — down from two N-sized round trips.
  maybe_rebuild_nlist();

  // Morse force compute. OPT-FUSE-1b: kernel writes forces with `=`, no
  // pre-zero needed.
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), natoms_, box_,
                                morse_params_, nullptr, compute_stream_);
  ++stats_.kernel_launches;
  // NOTE: second half-kick intentionally absent — lifted to run_until()
  // as a once-per-window finalize (OPT-FUSE-1c).
}

void FastPipelineScheduler::step_eam(bool first_step_after_bootstrap) {
  assert(eam_ != nullptr && "EAM step invoked without an uploaded DeviceEam");

  // OPT-FUSE-1c: half-dt bootstrap on first step, full-dt cross-step kick
  // on every other step. See step_morse() for the rationale.
  if (first_step_after_bootstrap) {
    integrator::device_fused_half_kick_drift(
        d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
        d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  } else {
    integrator::device_fused_full_kick_drift(
        d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
        d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  }
  ++stats_.kernel_launches;

  maybe_rebuild_nlist();

  // EAM 3-pass (density → embedding → force). Final force pass writes
  // with `=` (OPT-FUSE-1b), so no pre-zero. Density and embedding passes
  // write to scratch (rho_, fp_) which compute() owns.
  eam_->compute(d_pos_.data(), d_forces_.data(), d_types_.data(),
                nlist_.d_neighbors(), nlist_.d_offsets(), nlist_.d_counts(),
                natoms_, box_, nullptr, compute_stream_);
  // DeviceEam::compute issues 3 kernels in non-deterministic mode (plus
  // up to 2 extra reduction kernels in deterministic mode, which we do
  // not count here because the scheduler does not request energy).
  stats_.kernel_launches += 3;
  // NOTE: second half-kick intentionally absent — lifted to run_until()
  // as a once-per-window finalize (OPT-FUSE-1c).
}

void FastPipelineScheduler::compute_initial_forces() {
  // Build neighbor list for the first time.
  nlist_.build(d_pos_.data(), natoms_, box_, cutoff_, cfg_.r_skin,
               compute_stream_);
  ++stats_.rebuilds;
  steps_since_rebuild_ = 0;

  // OPT-FUSE-1b: force kernels write with `=`, no pre-zero needed.
  switch (potential_kind_) {
    case PotentialKind::Morse:
      potentials::compute_morse_gpu(
          d_pos_.data(), d_forces_.data(), nlist_.d_neighbors(),
          nlist_.d_offsets(), nlist_.d_counts(), natoms_, box_, morse_params_,
          nullptr, compute_stream_);
      ++stats_.kernel_launches;
      break;
    case PotentialKind::Eam:
      assert(eam_ != nullptr);
      eam_->compute(d_pos_.data(), d_forces_.data(), d_types_.data(),
                    nlist_.d_neighbors(), nlist_.d_offsets(),
                    nlist_.d_counts(), natoms_, box_, nullptr,
                    compute_stream_);
      stats_.kernel_launches += 3;
      break;
  }
}

void FastPipelineScheduler::run_until(i32 target_step) {
  if (!initial_forces_computed_) {
    compute_initial_forces();
    initial_forces_computed_ = true;
  }

  // OPT-FUSE-1c: bootstrap on the first step of this window (half-dt kick),
  // then cross-step full-dt kicks for steps 2..N, then a single finalize
  // half-kick to bring velocities from t+(N-1/2)dt to t+N dt. The finalize
  // keeps velocities on integer time at every run_until boundary, so
  // callers that alternate run_until()/download() — or that call run_until
  // one step at a time — see exactly the same state they would in the
  // unfused path.
  if (current_step_ < target_step) {
    bool first_step_after_bootstrap = true;
    while (current_step_ < target_step) {
      step(first_step_after_bootstrap);
      first_step_after_bootstrap = false;
    }
    integrator::device_half_kick(d_vel_.data(), d_forces_.data(),
                                 d_types_.data(), d_masses_.data(), natoms_,
                                 cfg_.dt, compute_stream_);
    ++stats_.kernel_launches;
  }

  // Final sync — the ONLY sync in the entire hot path. Ensures all queued
  // kernels complete before we return control to the caller.
  TDMD_CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

void FastPipelineScheduler::maybe_rebuild_nlist() {
  if (steps_since_rebuild_ >= cfg_.rebuild_every || stats_.rebuilds == 0) {
    nlist_.build(d_pos_.data(), natoms_, box_, cutoff_, cfg_.r_skin,
                 compute_stream_);
    ++stats_.rebuilds;
    steps_since_rebuild_ = 0;
  }
}

}  // namespace tdmd::scheduler
