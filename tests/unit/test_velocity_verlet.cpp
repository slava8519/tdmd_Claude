// SPDX-License-Identifier: Apache-2.0
// test_velocity_verlet.cpp — tests for the velocity-Verlet integrator.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/constants.hpp"
#include "core/math.hpp"
#include "core/types.hpp"
#include "integrator/velocity_verlet.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::integrator;
using namespace tdmd::potentials;
using namespace tdmd::neighbors;
using namespace tdmd::testing;

/// Compute kinetic energy in eV (metal units).
/// KE = 0.5 * m * v^2 * mvv2e
static real kinetic_energy(const SystemState& state) {
  real ke = real{0};
  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const real mass = state.masses[static_cast<std::size_t>(state.types[si])];
    ke += real{0.5} * mass * kMvv2e * length_sq(state.velocities[si]);
  }
  return ke;
}

TEST(VelocityVerlet, NVEConservation) {
  // Two Morse atoms in a large box. Run 10000 steps NVE.
  // Check that total energy drifts less than 1e-8 relative.
  SystemState state;
  state.resize(2);
  state.box.lo = {0, 0, 0};
  state.box.hi = {30, 30, 30};
  state.box.periodic = {true, true, true};

  state.positions[0] = {13.0, 15.0, 15.0};
  state.positions[1] = {17.0, 15.0, 15.0};
  state.velocities[0] = {0.5, 0.0, 0.0};
  state.velocities[1] = {-0.5, 0.0, 0.0};
  state.types[0] = 1;
  state.types[1] = 1;
  state.ids[0] = 1;
  state.ids[1] = 2;
  state.masses.resize(2);
  state.masses[1] = 63.546;  // Cu

  MorsePair pot(0.3429, 1.3588, 2.866, 9.5);
  VelocityVerlet vv(real{0.001});  // dt = 0.001 ps = 1 fs

  // Build neighbor list and compute initial forces.
  NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              pot.cutoff(), real{2.0});

  real pe = compute_pair_forces(state, nlist, pot);
  real ke = kinetic_energy(state);
  const real E0 = pe + ke;

  const int nsteps = 10000;
  for (int step = 0; step < nsteps; ++step) {
    vv.half_kick(state);
    vv.drift(state);

    // Rebuild neighbor list if needed.
    if (nlist.needs_rebuild(state.positions.data(), state.natoms)) {
      nlist.build(state.positions.data(), state.natoms, state.box,
                  pot.cutoff(), real{2.0});
    }

    pe = compute_pair_forces(state, nlist, pot);
    vv.half_kick(state);

    ke = kinetic_energy(state);
  }

  real E_final = pe + ke;
  real drift = std::abs((E_final - E0) / E0);

  // Symplectic integrator: energy should be bounded, not drifting.
  // Two-atom bounce is aggressive; 1e-5 over 10k steps is good.
  // M1 exit criterion: |dE/E| < 1e-4 over 50k steps on a bulk system.
  EXPECT_LT(drift, 1e-4) << "E0=" << E0 << " E_final=" << E_final
                          << " drift=" << drift;
}

TEST(VelocityVerlet, StepAndTimeAdvance) {
  SystemState state;
  state.resize(1);
  state.box.lo = {0, 0, 0};
  state.box.hi = {10, 10, 10};
  state.types[0] = 1;
  state.masses = {0, 1.0};
  state.positions[0] = {5, 5, 5};
  state.velocities[0] = {0, 0, 0};
  state.forces[0] = {0, 0, 0};

  VelocityVerlet vv(real{0.002});
  vv.half_kick(state);
  vv.drift(state);

  EXPECT_EQ(state.step, 1);
  EXPECT_NEAR(state.time, 0.002, kTimeTolerance);
}
