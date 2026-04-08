// SPDX-License-Identifier: Apache-2.0
// pipeline_scheduler.cuh — M4 TD pipeline scheduler.
//
// Implements the full time-decomposition pipeline on a single rank:
// - Zones at different time_step values (pipeline wave front)
// - CUDA stream pool for overlapping zone computations
// - Dependency DAG (causal check per zone-state-machine.md I-2)
// - Deterministic mode (single stream, sequential — bit-identical to M3)
#pragma once

#include <vector>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../domain/zone_partition.hpp"
#include "../neighbors/device_neighbor_list.cuh"
#include "../potentials/device_morse.cuh"
#include "stream_pool.cuh"
#include "zone.hpp"

namespace tdmd::scheduler {

struct PipelineConfig {
  real dt{0.001};         // time step (ps)
  real r_skin{1.0};       // Verlet skin
  i32 n_zones{0};         // auto
  i32 n_streams{4};       // CUDA stream pool size
  i32 rebuild_every{10};  // neighbor list rebuild interval
  bool deterministic{false};  // single stream, sequential walk
};

/// @brief Telemetry counters for the pipeline.
struct PipelineStats {
  i64 kernel_launches{0};
  i64 dep_check_calls{0};
  i64 dep_check_failures{0};
  i64 ticks{0};
};

/// @brief M4 TD pipeline scheduler (single rank, GPU).
class PipelineScheduler {
 public:
  PipelineScheduler(const Box& box, i32 natoms,
                    const potentials::MorseParams& params,
                    const PipelineConfig& cfg);

  /// @brief Upload host arrays to device. Must be called before run().
  void upload(const Vec3* positions, const Vec3* velocities,
              const Vec3* forces, const i32* types, const i32* ids,
              const real* masses, i32 n_masses);

  /// @brief Run until all zones reach target_step.
  void run_until(i32 target_step);

  /// @brief Download device arrays to host.
  void download(Vec3* positions, Vec3* velocities, Vec3* forces, i32* types,
                i32* ids, i32 natoms) const;

  [[nodiscard]] const domain::ZonePartition& partition() const noexcept {
    return partition_;
  }
  [[nodiscard]] const PipelineStats& stats() const noexcept { return stats_; }

  /// @brief Minimum time_step across all zones.
  [[nodiscard]] i32 min_time_step() const noexcept;

 private:
  PipelineConfig cfg_;
  potentials::MorseParams params_;
  Box box_;
  i32 natoms_;

  domain::ZonePartition partition_;
  StreamPool streams_;

  // Device arrays.
  DeviceBuffer<Vec3> d_pos_, d_vel_, d_forces_;
  DeviceBuffer<i32> d_types_, d_ids_;
  DeviceBuffer<real> d_masses_;

  // Neighbor list.
  neighbors::DeviceNeighborList nlist_;
  bool needs_rebuild_{true};
  i32 steps_since_rebuild_{0};

  // Per-zone: which CUDA stream is this zone using (-1 = none).
  std::vector<i32> zone_stream_;

  PipelineStats stats_;

  /// @brief Check if zone z can advance (I-2 causal dependency).
  [[nodiscard]] bool check_deps(i32 z_id) const;

  /// @brief Launch a full velocity-Verlet step for one zone on a stream.
  void launch_zone_step(i32 z_id, i32 stream_id);

  /// @brief Poll all Computing zones for completion.
  void poll_completions();

  /// @brief Rebuild neighbor list if needed.
  void rebuild_nlist();

  /// @brief One tick of the pipeline: poll, check deps, launch ready zones.
  /// @return true if any zone advanced.
  bool tick();
};

}  // namespace tdmd::scheduler
