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

void FastPipelineScheduler::step() {
  ++stats_.ticks;
  switch (potential_kind_) {
    case PotentialKind::Morse:
      step_morse();
      break;
    case PotentialKind::Eam:
      step_eam();
      break;
  }
  ++current_step_;
  ++steps_since_rebuild_;
}

void FastPipelineScheduler::step_morse() {
  // Phase 1+2: fused half-kick + drift. Register-resident velocity hand-off
  // saves one launch and one HBM round-trip on velocity vs the unfused path;
  // bit-exact equivalent (OPT-FUSE-1a).
  integrator::device_fused_half_kick_drift(
      d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
      d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 3: zero forces.
  integrator::device_zero_forces(d_forces_.data(), natoms_, compute_stream_);
  ++stats_.kernel_launches;

  // Neighbor list rebuild (amortized, every rebuild_every steps).
  // OPT-1: prefix sum now runs on the device via CUB; build() still performs
  // one 8-byte D2H for total_pairs + max_neighbors, so there is a single
  // tiny stream sync inside it — down from two N-sized round trips.
  maybe_rebuild_nlist();

  // Phase 4: Morse force compute.
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), natoms_, box_,
                                morse_params_, nullptr, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 5: second half-kick.
  integrator::device_half_kick(d_vel_.data(), d_forces_.data(),
                               d_types_.data(), d_masses_.data(), natoms_,
                               cfg_.dt, compute_stream_);
  ++stats_.kernel_launches;
}

void FastPipelineScheduler::step_eam() {
  assert(eam_ != nullptr && "EAM step invoked without an uploaded DeviceEam");

  // Phase 1+2: fused half-kick + drift (OPT-FUSE-1a).
  integrator::device_fused_half_kick_drift(
      d_vel_.data(), d_pos_.data(), d_forces_.data(), d_types_.data(),
      d_masses_.data(), natoms_, cfg_.dt, box_, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 3: zero forces.
  integrator::device_zero_forces(d_forces_.data(), natoms_, compute_stream_);
  ++stats_.kernel_launches;

  maybe_rebuild_nlist();

  // Phase 4: EAM 3-pass (density → embedding → force). All three passes
  // queue on compute_stream_; scratch zeroes inside compute() also run
  // on that stream (cudaMemsetAsync).
  eam_->compute(d_pos_.data(), d_forces_.data(), d_types_.data(),
                nlist_.d_neighbors(), nlist_.d_offsets(), nlist_.d_counts(),
                natoms_, box_, nullptr, compute_stream_);
  // DeviceEam::compute issues 3 kernels in non-deterministic mode (plus
  // up to 2 extra reduction kernels in deterministic mode, which we do
  // not count here because the scheduler does not request energy).
  stats_.kernel_launches += 3;

  // Phase 5: second half-kick.
  integrator::device_half_kick(d_vel_.data(), d_forces_.data(),
                               d_types_.data(), d_masses_.data(), natoms_,
                               cfg_.dt, compute_stream_);
  ++stats_.kernel_launches;
}

void FastPipelineScheduler::compute_initial_forces() {
  // Build neighbor list for the first time.
  nlist_.build(d_pos_.data(), natoms_, box_, cutoff_, cfg_.r_skin,
               compute_stream_);
  ++stats_.rebuilds;
  steps_since_rebuild_ = 0;

  // Zero forces and compute initial forces via the right potential.
  integrator::device_zero_forces(d_forces_.data(), natoms_, compute_stream_);
  ++stats_.kernel_launches;

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

  while (current_step_ < target_step) {
    step();
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
