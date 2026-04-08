// SPDX-License-Identifier: Apache-2.0
// distributed_pipeline_scheduler.cuh — M5 multi-rank TD pipeline scheduler.
//
// Each rank owns a contiguous subset of zones and holds all atoms on GPU.
// After each tick round where zones advance, boundary data is exchanged
// synchronously via MPI_Sendrecv. Ghost zone time_steps are updated to
// enable cross-rank dependency checking.
#pragma once

#include <memory>
#include <vector>

#include <mpi.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../domain/zone_partition.hpp"
#include "../integrator/device_nose_hoover.cuh"
#include "../neighbors/device_neighbor_list.cuh"
#include "../potentials/device_morse.cuh"
#include "stream_pool.cuh"
#include "zone.hpp"

namespace tdmd::scheduler {

struct DistributedConfig {
  real dt{0.001};
  real r_skin{1.0};
  i32 n_zones{0};
  i32 n_streams{4};
  i32 rebuild_every{10};
  bool deterministic{false};

  // NVT thermostat (Nosé-Hoover chain). Enabled when t_target > 0.
  real t_target{0};       // target temperature (K). 0 = NVE mode.
  real t_period{0.1};     // NHC coupling period (ps).
  i32 nhc_length{3};      // NHC chain length.
};

struct DistributedStats {
  i64 kernel_launches{0};
  i64 dep_check_calls{0};
  i64 dep_check_failures{0};
  i64 ticks{0};
  i64 exchanges{0};
};

/// @brief M5 multi-rank TD pipeline scheduler.
///
/// Uses synchronous MPI_Sendrecv for boundary exchange after each
/// pipeline tick round. Simple, correct, deadlock-free.
class DistributedPipelineScheduler {
 public:
  DistributedPipelineScheduler(const Box& box, i32 natoms,
                               const potentials::MorseParams& params,
                               const DistributedConfig& cfg,
                               MPI_Comm comm);

  void upload(const Vec3* positions, const Vec3* velocities,
              const Vec3* forces, const i32* types, const i32* ids,
              const real* masses, i32 n_masses);

  void run_until(i32 target_step);

  void download(Vec3* positions, Vec3* velocities, Vec3* forces, i32* types,
                i32* ids, i32 natoms) const;

  [[nodiscard]] const domain::ZonePartition& partition() const noexcept {
    return partition_;
  }
  [[nodiscard]] const DistributedStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] i32 rank() const noexcept { return rank_; }
  [[nodiscard]] i32 size() const noexcept { return size_; }
  [[nodiscard]] i32 min_local_time_step() const noexcept;
  [[nodiscard]] i32 min_global_time_step();

 private:
  DistributedConfig cfg_;
  potentials::MorseParams params_;
  Box box_;
  i32 natoms_;

  MPI_Comm comm_;
  i32 rank_{0}, size_{1};
  i32 prev_rank_{0}, next_rank_{0};

  domain::ZonePartition partition_;
  StreamPool streams_;

  DeviceBuffer<Vec3> d_pos_, d_vel_, d_forces_;
  DeviceBuffer<i32> d_types_, d_ids_;
  DeviceBuffer<real> d_masses_;
  neighbors::DeviceNeighborList nlist_;
  bool needs_rebuild_{true};

  std::vector<i32> zone_stream_;

  // Ghost zones and their tracked time_steps.
  std::vector<i32> ghost_zones_;
  std::vector<i32> ghost_time_steps_;

  // Boundary zones to exchange with neighbors.
  std::vector<i32> send_to_next_zones_;
  std::vector<i32> send_to_prev_zones_;

  DistributedStats stats_;

  // NVT thermostat (null in NVE mode).
  std::unique_ptr<integrator::NoseHooverChain> nhc_;

  // Atom range for locally-owned zones (contiguous after zone sort).
  i32 first_local_atom_{0};
  i32 local_atom_count_{0};

  [[nodiscard]] bool check_deps(i32 z_id) const;
  [[nodiscard]] i32 zone_time_step(i32 z_id) const;
  [[nodiscard]] i32 ghost_index(i32 z_id) const;

  void launch_zone_step(i32 z_id, i32 stream_id);
  void poll_completions();
  void rebuild_nlist();

  /// @brief Advance all local zones that can advance (one tick round).
  /// @return true if any zone advanced.
  bool tick();

  /// @brief Synchronous MPI exchange of boundary zone data.
  void exchange_boundary_data();

  void compute_boundary_zones();
  void update_ghost_time_step(i32 zone_id, i32 time_step);
};

}  // namespace tdmd::scheduler
