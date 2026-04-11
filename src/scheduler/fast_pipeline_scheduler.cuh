// SPDX-License-Identifier: Apache-2.0
// fast_pipeline_scheduler.cuh — batched single-GPU TD scheduler (ADR 0005).
//
// Key differences from PipelineScheduler:
//   1. No zone state machine, no check_deps, no per-zone launches.
//   2. One force kernel covers all atoms at once (Morse or EAM 3-pass).
//   3. Integrator phases (kick_drift, force) run as whole-system kernels.
//      After OPT-FUSE-1a/1b/1c: 2 launches per Morse step + 1 finalize
//      half-kick per run_until window; 4 launches per EAM step + 1
//      finalize half-kick per run_until window.
//   4. No cudaDeviceSynchronize in the hot loop. All kernels queue on a
//      dedicated non-default stream and serialize via in-stream ordering.
//   5. Single cudaStreamSynchronize at the end of run_until().
//
// For multi-rank TD (P_time > 1), use DistributedPipelineScheduler instead.
#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../neighbors/device_neighbor_list.cuh"
#include "../potentials/device_morse.cuh"

namespace tdmd::potentials {
class EamAlloy;  // forward decl — full def lives in eam_alloy.hpp
class DeviceEam;  // forward decl — full def lives in device_eam.cuh
}  // namespace tdmd::potentials

namespace tdmd::scheduler {

/// @brief Which pair potential the scheduler drives.
enum class PotentialKind {
  Morse,
  Eam,
};

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
/// Per-step kernel graph after OPT-FUSE-1a/1b/1c:
///   Morse step 1 of a run_until window : fused_half_kick_drift → force
///   Morse steps 2..N of that window    : fused_full_kick_drift → force
///   After all N steps                  : half_kick (finalize)
///   EAM adds density + embedding kernels between kick_drift and force,
///   so each EAM step costs 4 launches instead of 2.
///
/// Cross-step fusion collapses the closing kick2 of step k with the
/// opening kick1 of step k+1 into a single full-dt kick. The finalize
/// half-kick is issued once per run_until window, not per step, which
/// leaves velocities at integer time t+N dt before download().
///
/// No zone state machine, no per-zone launches, no cudaDeviceSynchronize
/// between steps. In-stream ordering guarantees kernel execution order.
/// CPU queues launches asynchronously — GPU catches up in the background.
///
/// Potential is selected at construction time via overload resolution:
/// passing a MorseParams selects the Morse path, passing an EamAlloy
/// selects the EAM path. Once constructed, the potential cannot change.
class FastPipelineScheduler {
 public:
  /// @brief Construct a Morse-driven scheduler. Morse params are copied
  /// by value; the scheduler owns them.
  FastPipelineScheduler(const Box& box, i32 natoms,
                        const potentials::MorseParams& params,
                        const FastPipelineConfig& cfg);

  /// @brief Construct an EAM-driven scheduler. The EamAlloy is read
  /// only during construction — its tables are uploaded to the GPU
  /// immediately, after which the scheduler holds an owning
  /// DeviceEam with no reference back to the host-side EamAlloy.
  FastPipelineScheduler(const Box& box, i32 natoms,
                        const potentials::EamAlloy& eam,
                        const FastPipelineConfig& cfg);

  ~FastPipelineScheduler();

  FastPipelineScheduler(const FastPipelineScheduler&) = delete;
  FastPipelineScheduler& operator=(const FastPipelineScheduler&) = delete;

  /// @brief Upload host arrays to device. Must be called before run_until().
  void upload(const PositionVec* positions, const VelocityVec* velocities,
              const ForceVec* forces, const i32* types, const i32* ids,
              const real* masses, i32 n_types);

  /// @brief Run until the given simulation step.
  /// The only sync point: cudaStreamSynchronize at the end.
  void run_until(i32 target_step);

  /// @brief Download device arrays to host.
  void download(PositionVec* positions, VelocityVec* velocities,
                ForceVec* forces, i32* types, i32* ids, i32 natoms) const;

  /// @brief Telemetry counters.
  [[nodiscard]] const FastPipelineStats& stats() const noexcept {
    return stats_;
  }

  /// @brief Current simulation step.
  [[nodiscard]] i32 current_step() const noexcept { return current_step_; }

  /// @brief Which potential this scheduler drives.
  [[nodiscard]] PotentialKind potential_kind() const noexcept {
    return potential_kind_;
  }

  /// @brief Cutoff used to build the neighbor list (potential cutoff +
  /// nothing; r_skin is added inside DeviceNeighborList::build). Cached
  /// at construction time so force dispatch never has to re-query the
  /// potential for its cutoff.
  [[nodiscard]] real interaction_cutoff() const noexcept { return cutoff_; }

 private:
  /// @brief Dispatch one velocity-Verlet step to the potential-specific
  /// implementation. `first_step_after_bootstrap` is true iff this is the
  /// first step of the current run_until window — in that case velocities
  /// are at integer time t and we bootstrap with a half-dt kick; otherwise
  /// velocities are half-advanced and we apply the cross-step full-dt
  /// kick (OPT-FUSE-1c).
  void step(bool first_step_after_bootstrap);

  /// @brief Morse-specific hot path: 2 launches per step
  /// (fused_kick_drift + force). The closing half-kick is lifted out to
  /// run_until().
  void step_morse(bool first_step_after_bootstrap);

  /// @brief EAM-specific hot path: 4 launches per step
  /// (fused_kick_drift + density + embedding + force). The closing
  /// half-kick is lifted out to run_until().
  void step_eam(bool first_step_after_bootstrap);

  /// @brief Compute the initial forces once, before the first step.
  /// Runs the full potential pass so velocity Verlet has a valid F at
  /// t = 0. Shared between Morse and EAM.
  void compute_initial_forces();

  /// @brief Rebuild neighbor list if enough steps since last rebuild.
  /// Uses cutoff_ (cached at construction) as the interaction range.
  void maybe_rebuild_nlist();

  Box box_;
  i32 natoms_;
  FastPipelineConfig cfg_;

  // Potential kind + exactly one populated potential member. The
  // invariant is enforced by the constructors and asserted in the
  // hot path:
  //   kind == Morse ⇒ eam_ == nullptr
  //   kind == Eam   ⇒ eam_ != nullptr
  // The Morse branch stores params by value; the EAM branch owns a
  // DeviceEam with pre-uploaded tables.
  PotentialKind potential_kind_;
  potentials::MorseParams morse_params_{};
  std::unique_ptr<potentials::DeviceEam> eam_;

  // Cached interaction cutoff (potential-specific). Single source of
  // truth for the neighbor list builder — neither step_morse nor
  // step_eam ever re-queries the potential.
  real cutoff_{0};

  DeviceBuffer<PositionVec> d_pos_;
  DeviceBuffer<VelocityVec> d_vel_;
  DeviceBuffer<ForceVec> d_forces_;
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
