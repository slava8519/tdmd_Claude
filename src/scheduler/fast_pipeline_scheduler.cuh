// SPDX-License-Identifier: Apache-2.0
// fast_pipeline_scheduler.cuh — batched single-GPU TD scheduler (ADR 0005).
//
// Key differences from PipelineScheduler:
//   1. No zone state machine, no check_deps, no per-zone launches.
//   2. One force kernel covers all atoms at once (compute_morse_gpu).
//   3. Integrator phases (half_kick, drift, force, half_kick) run as
//      whole-system kernels. 5 launches per step (including zero_forces).
//   4. No cudaDeviceSynchronize in the hot loop. All kernels queue on a
//      dedicated non-default stream and serialize via in-stream ordering.
//   5. Single cudaStreamSynchronize at the end of run_until().
//
// For multi-rank TD (P_time > 1), use DistributedPipelineScheduler instead.
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../neighbors/device_neighbor_list.cuh"
#include "../potentials/device_morse.cuh"

namespace tdmd::scheduler {

/// @brief Configuration for FastPipelineScheduler.
struct FastPipelineConfig {
  real dt{real{0.001}};
  real r_skin{real{1.0}};
  i32 rebuild_every{10};
};

/// @brief Telemetry counters for FastPipelineScheduler.
struct FastPipelineStats {
  i64 ticks{0};
  i64 kernel_launches{0};
  i64 rebuilds{0};
};

/// @brief Batched single-GPU scheduler implementing ADR 0005 Phase 2.
///
/// On each step, runs 5 whole-system kernel launches on a dedicated stream:
///   half_kick -> drift -> zero_forces -> morse_force -> half_kick
///
/// No zone state machine, no per-zone launches, no cudaDeviceSynchronize
/// between steps. In-stream ordering guarantees kernel execution order.
/// CPU queues launches asynchronously — GPU catches up in the background.
class FastPipelineScheduler {
 public:
  FastPipelineScheduler(const Box& box, i32 natoms,
                        const potentials::MorseParams& params,
                        const FastPipelineConfig& cfg);
  ~FastPipelineScheduler();

  FastPipelineScheduler(const FastPipelineScheduler&) = delete;
  FastPipelineScheduler& operator=(const FastPipelineScheduler&) = delete;

  /// @brief Upload host arrays to device. Must be called before run_until().
  void upload(const Vec3* positions, const Vec3* velocities,
              const Vec3* forces, const i32* types, const i32* ids,
              const real* masses, i32 n_types);

  /// @brief Run until the given simulation step.
  /// The only sync point: cudaStreamSynchronize at the end.
  void run_until(i32 target_step);

  /// @brief Download device arrays to host.
  void download(Vec3* positions, Vec3* velocities, Vec3* forces, i32* types,
                i32* ids, i32 natoms) const;

  /// @brief Telemetry counters.
  [[nodiscard]] const FastPipelineStats& stats() const noexcept {
    return stats_;
  }

  /// @brief Current simulation step.
  [[nodiscard]] i32 current_step() const noexcept { return current_step_; }

 private:
  /// @brief One velocity-Verlet step: 5 kernel launches, no sync.
  void step();

  /// @brief Rebuild neighbor list if enough steps since last rebuild.
  /// Contains internal sync (two-pass build with host prefix sum).
  void maybe_rebuild_nlist();

  Box box_;
  i32 natoms_;
  potentials::MorseParams params_;
  FastPipelineConfig cfg_;

  DeviceBuffer<Vec3> d_pos_;
  DeviceBuffer<Vec3> d_vel_;
  DeviceBuffer<Vec3> d_forces_;
  DeviceBuffer<i32> d_types_;
  DeviceBuffer<i32> d_ids_;
  DeviceBuffer<real> d_masses_;

  neighbors::DeviceNeighborList nlist_;

  cudaStream_t compute_stream_{nullptr};

  i32 current_step_{0};
  i32 steps_since_rebuild_{0};
  bool initial_forces_computed_{false};
  FastPipelineStats stats_;
};

}  // namespace tdmd::scheduler
