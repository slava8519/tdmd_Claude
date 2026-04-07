// SPDX-License-Identifier: Apache-2.0
// sequential_scheduler.hpp — M3 sequential zone-walked scheduler.
//
// Walks zones in linear order, all at the same time step.
// No pipeline, no MPI, single GPU. Produces bit-identical results to
// the global M2 compute.
#pragma once

#include "../core/system_state.hpp"
#include "../core/types.hpp"
#include "../domain/zone_partition.hpp"
#include "../neighbors/neighbor_list.hpp"
#include "../potentials/force_compute.hpp"
#include "../potentials/ipair.hpp"

namespace tdmd::scheduler {

/// @brief Configuration for the sequential scheduler.
struct SequentialSchedulerConfig {
  real r_skin{1.0};       // Verlet skin
  i32 rebuild_every{10};  // neighbor list rebuild interval
  i32 n_zones{0};         // number of zones (0 = auto)
};

/// @brief M3 sequential zone walker.
///
/// At each step: walks all zones in linear order, computes forces globally
/// (using the full neighbor list), integrates globally.
/// Zone metadata (state, time_step) is updated per the state machine.
///
/// This is a stepping stone to M4's pipeline scheduler.
class SequentialScheduler {
 public:
  SequentialScheduler(SystemState& state, potentials::IPairStyle& pair,
                      const SequentialSchedulerConfig& cfg);

  /// @brief Run one full time step (all zones advance by 1).
  /// @param dt Time step in ps.
  /// @return Total potential energy.
  real step(real dt);

  /// @brief Run n_steps full time steps.
  void run(i32 n_steps, real dt);

  [[nodiscard]] const domain::ZonePartition& partition() const noexcept {
    return partition_;
  }
  [[nodiscard]] i64 total_steps() const noexcept { return total_steps_; }

 private:
  SystemState& state_;
  potentials::IPairStyle& pair_;
  SequentialSchedulerConfig cfg_;

  domain::ZonePartition partition_;
  neighbors::NeighborList nlist_;

  i64 total_steps_{0};
  bool needs_rebuild_{true};

  void rebuild_if_needed();
};

}  // namespace tdmd::scheduler
