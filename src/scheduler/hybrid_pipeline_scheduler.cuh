// SPDX-License-Identifier: Apache-2.0
// hybrid_pipeline_scheduler.cuh — M6 2D time × space pipeline scheduler.
//
// Combines TD pipeline (time_comm ring) with spatial decomposition (space_comm
// halo exchange). Each rank is identified by a 2D Cartesian coordinate
// (time_idx, space_idx). Ranks in the same time group advance the same zones
// in lockstep with halo exchange before each force computation. Ranks in the
// same spatial subdomain form a TD ring for zone boundary data exchange.
//
// Data model: each rank stores owned_atoms (in its Y-subdomain) + ghost_atoms
// (from spatial neighbors within r_ghost of Y boundaries). Owned atoms are
// sorted by zone. Ghost atoms are appended after owned atoms.
#pragma once

#include <memory>
#include <vector>

#include <mpi.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "../domain/spatial_decomp.hpp"
#include "../domain/zone_partition.hpp"
#include "../integrator/device_nose_hoover.cuh"
#include "../neighbors/device_neighbor_list.cuh"
#include "../potentials/device_morse.cuh"
#include "stream_pool.cuh"
#include "zone.hpp"

namespace tdmd::scheduler {

struct HybridConfig {
  real dt{0.001};
  real r_skin{1.0};
  i32 n_zones{0};
  i32 n_streams{4};
  i32 rebuild_every{10};
  bool deterministic{false};
  i32 p_time{1};   ///< Number of time groups.
  i32 p_space{1};  ///< Number of spatial ranks per time group.

  // NVT thermostat (Nosé-Hoover chain). Enabled when t_target > 0.
  real t_target{0};       // target temperature (K). 0 = NVE mode.
  real t_period{0.1};     // NHC coupling period (ps).
  i32 nhc_length{3};      // NHC chain length.
};

struct HybridStats {
  i64 kernel_launches{0};
  i64 dep_check_calls{0};
  i64 dep_check_failures{0};
  i64 ticks{0};
  i64 td_exchanges{0};
  i64 halo_exchanges{0};
};

/// @brief M6 hybrid 2D time × space pipeline scheduler.
///
/// Uses MPI_Cart_create for 2D topology [P_time, P_space].
/// TD pipeline runs on time_comm (ring of time groups).
/// Halo exchange runs on space_comm (spatial neighbors within a time group).
class HybridPipelineScheduler {
 public:
  /// @param box Simulation box.
  /// @param natoms Total number of atoms in the system.
  /// @param params Morse potential parameters.
  /// @param cfg Configuration including p_time and p_space.
  /// @param world_comm MPI communicator (must have p_time * p_space ranks).
  HybridPipelineScheduler(const Box& box, i32 natoms,
                          const potentials::MorseParams& params,
                          const HybridConfig& cfg, MPI_Comm world_comm);

  ~HybridPipelineScheduler();

  void upload(const Vec3* positions, const Vec3* velocities,
              const Vec3* forces, const i32* types, const i32* ids,
              const real* masses, i32 n_masses);

  void run_until(i32 target_step);

  void download(Vec3* positions, Vec3* velocities, Vec3* forces, i32* types,
                i32* ids, i32 natoms) const;

  [[nodiscard]] const domain::ZonePartition& partition() const noexcept {
    return partition_;
  }
  [[nodiscard]] const HybridStats& stats() const noexcept { return stats_; }
  [[nodiscard]] i32 world_rank() const noexcept { return world_rank_; }
  [[nodiscard]] i32 world_size() const noexcept { return world_size_; }
  [[nodiscard]] i32 time_rank() const noexcept { return time_rank_; }
  [[nodiscard]] i32 space_rank() const noexcept { return space_rank_; }
  [[nodiscard]] i32 n_owned() const noexcept { return n_owned_; }
  [[nodiscard]] i32 n_total() const noexcept { return n_total_; }
  [[nodiscard]] i32 min_local_time_step() const noexcept;
  [[nodiscard]] i32 min_global_time_step();

 private:
  HybridConfig cfg_;
  potentials::MorseParams params_;
  Box box_;
  i32 natoms_global_;  ///< Total atoms in the system.
  i32 n_owned_{0};     ///< Atoms in this rank's spatial subdomain.
  i32 n_ghost_{0};     ///< Ghost atoms from spatial neighbors.
  i32 n_total_{0};     ///< n_owned_ + n_ghost_ on GPU.

  // MPI topology.
  MPI_Comm world_comm_;
  MPI_Comm cart_comm_{MPI_COMM_NULL};
  MPI_Comm time_comm_{MPI_COMM_NULL};   ///< TD ring (same spatial subdomain).
  MPI_Comm space_comm_{MPI_COMM_NULL};  ///< Halo exchange (same time group).
  i32 world_rank_{0}, world_size_{1};
  i32 time_rank_{0}, time_size_{1};  ///< rank/size in time_comm.
  i32 space_rank_{0}, space_size_{1};
  i32 time_prev_{0}, time_next_{0};  ///< TD ring neighbors in time_comm.
  i32 space_prev_{0}, space_next_{0};  ///< Spatial neighbors in space_comm.

  domain::ZonePartition partition_;
  domain::SpatialDecomp spatial_;
  StreamPool streams_;

  // Device buffers: sized for n_owned + ghost_capacity.
  DeviceBuffer<Vec3> d_pos_, d_vel_, d_forces_;
  DeviceBuffer<i32> d_types_, d_ids_;
  DeviceBuffer<real> d_masses_;
  neighbors::DeviceNeighborList nlist_;
  bool needs_rebuild_{true};

  // Host copies for owned atoms (needed for zone reorder and exchanges).
  std::vector<Vec3> h_pos_, h_vel_, h_forces_;
  std::vector<i32> h_types_, h_ids_;

  std::vector<i32> zone_stream_;

  // TD ghost zones (on time_comm) and their time_steps.
  std::vector<i32> td_ghost_zones_;
  std::vector<i32> td_ghost_time_steps_;
  std::vector<i32> send_to_time_next_;
  std::vector<i32> send_to_time_prev_;

  // Spatial ghost exchange bookkeeping.
  std::vector<i32> send_prev_indices_;  ///< Owned atom indices near lower Y.
  std::vector<i32> send_next_indices_;  ///< Owned atom indices near upper Y.

  HybridStats stats_;

  // NVT thermostat (null in NVE mode).
  std::unique_ptr<integrator::NoseHooverChain> nhc_;

  [[nodiscard]] bool check_deps(i32 z_id) const;
  [[nodiscard]] i32 zone_time_step(i32 z_id) const;
  [[nodiscard]] i32 td_ghost_index(i32 z_id) const;

  void launch_zone_step(i32 z_id, i32 stream_id);
  void poll_completions();
  void rebuild_nlist();
  bool tick();

  /// @brief Exchange ghost atom positions/velocities with spatial neighbors.
  void halo_exchange();

  /// @brief Exchange boundary zone data with TD ring neighbors.
  void td_exchange_boundary_data();

  void compute_td_boundary_zones();
  void update_td_ghost_time_step(i32 zone_id, i32 time_step);

  /// @brief Upload ghost atoms to GPU after halo exchange.
  void upload_ghosts();
};

}  // namespace tdmd::scheduler
