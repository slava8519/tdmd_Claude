// SPDX-License-Identifier: Apache-2.0
// fast_pipeline_scheduler.cu — batched single-GPU scheduler (ADR 0005 Phase 2).

#include "fast_pipeline_scheduler.cuh"

#include "../core/error.hpp"
#include "../integrator/device_velocity_verlet.cuh"

namespace tdmd::scheduler {

FastPipelineScheduler::FastPipelineScheduler(
    const Box& box, i32 natoms, const potentials::MorseParams& params,
    const FastPipelineConfig& cfg)
    : box_(box), natoms_(natoms), params_(params), cfg_(cfg) {
  TDMD_CUDA_CHECK(cudaStreamCreate(&compute_stream_));
}

FastPipelineScheduler::~FastPipelineScheduler() {
  if (compute_stream_) {
    cudaStreamSynchronize(compute_stream_);
    cudaStreamDestroy(compute_stream_);
  }
}

void FastPipelineScheduler::upload(const Vec3* positions,
                                   const Vec3* velocities, const Vec3* forces,
                                   const i32* types, const i32* ids,
                                   const real* masses, i32 n_types) {
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

void FastPipelineScheduler::download(Vec3* positions, Vec3* velocities,
                                     Vec3* forces, i32* types, i32* ids,
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

  // Phase 1: half-kick (v += 0.5 * dt * F / m).
  integrator::device_half_kick(d_vel_.data(), d_forces_.data(),
                               d_types_.data(), d_masses_.data(), natoms_,
                               cfg_.dt, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 2: drift (x += dt * v, wrap PBC).
  integrator::device_drift(d_pos_.data(), d_vel_.data(), natoms_, cfg_.dt,
                           box_, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 3: zero forces before force compute.
  integrator::device_zero_forces(d_forces_.data(), natoms_, compute_stream_);
  ++stats_.kernel_launches;

  // Neighbor list rebuild (amortized, every rebuild_every steps).
  // Contains internal sync — this is unavoidable for the two-pass host
  // prefix sum, but happens only every rebuild_every steps.
  maybe_rebuild_nlist();

  // Phase 4: force compute (whole-system batched kernel).
  potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                nlist_.d_neighbors(), nlist_.d_offsets(),
                                nlist_.d_counts(), natoms_, box_, params_,
                                nullptr, compute_stream_);
  ++stats_.kernel_launches;

  // Phase 5: second half-kick (v += 0.5 * dt * F_new / m).
  integrator::device_half_kick(d_vel_.data(), d_forces_.data(),
                               d_types_.data(), d_masses_.data(), natoms_,
                               cfg_.dt, compute_stream_);
  ++stats_.kernel_launches;

  ++current_step_;
  ++steps_since_rebuild_;

  // No sync. CPU returns immediately. GPU queues the 5 kernels on
  // compute_stream_ and runs them in order (in-stream serialization).
}

void FastPipelineScheduler::run_until(i32 target_step) {
  // Initial force compute (once, before the first step).
  if (!initial_forces_computed_) {
    // Build neighbor list for the first time.
    nlist_.build(d_pos_.data(), natoms_, box_, params_.rc, cfg_.r_skin,
                 compute_stream_);
    ++stats_.rebuilds;
    steps_since_rebuild_ = 0;

    // Zero forces and compute initial forces.
    integrator::device_zero_forces(d_forces_.data(), natoms_, compute_stream_);
    ++stats_.kernel_launches;
    potentials::compute_morse_gpu(d_pos_.data(), d_forces_.data(),
                                  nlist_.d_neighbors(), nlist_.d_offsets(),
                                  nlist_.d_counts(), natoms_, box_, params_,
                                  nullptr, compute_stream_);
    ++stats_.kernel_launches;

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
    nlist_.build(d_pos_.data(), natoms_, box_, params_.rc, cfg_.r_skin,
                 compute_stream_);
    ++stats_.rebuilds;
    steps_since_rebuild_ = 0;
  }
}

}  // namespace tdmd::scheduler
