// SPDX-License-Identifier: Apache-2.0
// sequential_scheduler.cpp — M3 sequential zone walker.

#include "sequential_scheduler.hpp"

#include "../core/math.hpp"
#include "../integrator/velocity_verlet.hpp"

namespace tdmd::scheduler {

SequentialScheduler::SequentialScheduler(SystemState& state,
                                         potentials::IPairStyle& pair,
                                         const SequentialSchedulerConfig& cfg)
    : state_(state), pair_(pair), cfg_(cfg) {
  // Build zone partition.
  partition_.build(state_.box, pair_.cutoff(), cfg_.n_zones);

  // Assign atoms to zones (sorts arrays by zone).
  partition_.assign_atoms(state_.positions.data(), state_.velocities.data(),
                          state_.forces.data(), state_.types.data(),
                          state_.ids.data(), state_.natoms);

  // Precompute zone neighbors.
  real r_list = pair_.cutoff() + cfg_.r_skin;
  partition_.build_zone_neighbors(r_list);

  // Initialize all zones to Ready state (skip Free→Receiving→Received
  // since we're single-rank, no MPI).
  for (auto& z : partition_.zones()) {
    z.state = ZoneState::Ready;
  }
}

void SequentialScheduler::rebuild_if_needed() {
  if (needs_rebuild_ || nlist_.needs_rebuild(state_.positions.data(),
                                              state_.natoms)) {
    nlist_.build(state_.positions.data(), state_.natoms, state_.box,
                 pair_.cutoff(), cfg_.r_skin);
    needs_rebuild_ = false;
  }
}

real SequentialScheduler::step(real dt) {
  integrator::VelocityVerlet vv(dt);

  // Half-kick (all zones).
  vv.half_kick(state_);

  // Drift (all zones).
  vv.drift(state_);

  // Rebuild neighbor list if needed.
  rebuild_if_needed();

  // Force compute (global, using full neighbor list).
  real pe = potentials::compute_pair_forces(state_, nlist_, pair_);

  // Second half-kick (all zones).
  vv.half_kick(state_);

  // Update zone state: Ready → Computing → Done for all zones.
  for (auto& z : partition_.zones()) {
    z.state = ZoneState::Ready;
    z.transition_to(ZoneState::Computing);
    z.transition_to(ZoneState::Done);  // increments time_step
    // Reset to Ready for next step (skip MPI states in M3).
    z.state = ZoneState::Ready;
  }

  ++total_steps_;
  return pe;
}

void SequentialScheduler::run(i32 n_steps, real dt) {
  // Initial force compute.
  rebuild_if_needed();
  potentials::compute_pair_forces(state_, nlist_, pair_);

  integrator::VelocityVerlet vv(dt);

  for (i32 s = 0; s < n_steps; ++s) {
    // Half-kick.
    vv.half_kick(state_);

    // Drift.
    vv.drift(state_);

    // Rebuild check.
    if ((s + 1) % cfg_.rebuild_every == 0) {
      needs_rebuild_ = true;
    }
    rebuild_if_needed();

    // Force compute.
    potentials::compute_pair_forces(state_, nlist_, pair_);

    // Second half-kick.
    vv.half_kick(state_);

    // Update zone states.
    for (auto& z : partition_.zones()) {
      z.state = ZoneState::Ready;
      z.transition_to(ZoneState::Computing);
      z.transition_to(ZoneState::Done);
      z.state = ZoneState::Ready;
    }

    ++total_steps_;
  }
}

}  // namespace tdmd::scheduler
