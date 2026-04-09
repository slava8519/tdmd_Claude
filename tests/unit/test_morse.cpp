// SPDX-License-Identifier: Apache-2.0
// test_morse.cpp — tests for Morse potential: analytic energy/force verification.
#include <gtest/gtest.h>

#include <cmath>

#include "core/math.hpp"
#include "core/types.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::potentials;
using namespace tdmd::testing;

// Morse parameters for Cu-Cu (typical test values).
static constexpr real kD = 0.3429;     // eV
static constexpr real kAlpha = 1.3588; // 1/A
static constexpr real kR0 = 2.866;     // A
static constexpr real kRc = 9.5;       // A

// Analytic Morse energy.
static real morse_energy(real r) {
  real dr = r - kR0;
  real exp_val = std::exp(-kAlpha * dr);
  return kD * ((real{1} - exp_val) * (real{1} - exp_val) - real{1});
}

// Analytic Morse force magnitude: F = -dU/dr.
static real morse_force(real r) {
  real dr = r - kR0;
  real exp_val = std::exp(-kAlpha * dr);
  real dudr = real{2} * kD * kAlpha * (real{1} - exp_val) * exp_val;
  return -dudr;
}

TEST(Morse, EnergyAtEquilibrium) {
  MorsePair pot(kD, kAlpha, kR0, kRc);
  real energy, fpair;
  pot.compute(kR0 * kR0, energy, fpair);
  // At r0: U = D * (0 - 1) = -D.
  EXPECT_NEAR(energy, -kD, 1e-14);
  // Force should be zero at equilibrium.
  EXPECT_NEAR(fpair, 0.0, 1e-12);
}

TEST(Morse, EnergyBeyondCutoff) {
  MorsePair pot(kD, kAlpha, kR0, kRc);
  real energy, fpair;
  pot.compute(kRc * kRc + real{1}, energy, fpair);
  EXPECT_DOUBLE_EQ(energy, 0.0);
  EXPECT_DOUBLE_EQ(fpair, 0.0);
}

TEST(Morse, AnalyticForceMatch) {
  // Test at several distances that the compute() output matches analytic formulas.
  MorsePair pot(kD, kAlpha, kR0, kRc);

  for (real r = 2.0; r < 9.0; r += 0.5) {
    real r2 = r * r;
    real energy, fpair;
    pot.compute(r2, energy, fpair);

    real expected_energy = morse_energy(r);
    EXPECT_NEAR(energy, expected_energy, kAnalyticTolerance)
        << "energy mismatch at r=" << r;

    // fpair = -dU/dr / r, so force magnitude = -fpair * r (on j directed to i)
    // Force on i from j (in direction i->j) = fpair * r_ij
    // Analytic F = -dU/dr (radial), so |force| = |fpair * r| should equal |-dU/dr|
    real computed_force = -fpair * r;  // = dU/dr
    real expected_force = -morse_force(r);  // = dU/dr
    EXPECT_NEAR(computed_force, expected_force, kAnalyticTolerance)
        << "force mismatch at r=" << r;
  }
}

TEST(Morse, TwoAtomForceCompute) {
  // Place two atoms along x-axis at distance 3.0 A.
  // Verify forces using the full force_compute pipeline.
  SystemState state;
  state.resize(2);
  state.box.lo = {0, 0, 0};
  state.box.hi = {20, 20, 20};
  state.box.periodic = {true, true, true};
  state.positions[0] = {5.0, 10.0, 10.0};
  state.positions[1] = {8.0, 10.0, 10.0};
  state.ids[0] = 1;
  state.ids[1] = 2;

  MorsePair pot(kD, kAlpha, kR0, kRc);

  neighbors::NeighborList nlist;
  nlist.build(state.positions.data(), state.natoms, state.box,
              pot.cutoff(), real{0.5});

  real pe = compute_pair_forces(state, nlist, pot);

  // Distance = 3.0 A (> r0=2.866, so atoms attract).
  real r = 3.0;
  real expected_energy = morse_energy(r);
  EXPECT_NEAR(pe, expected_energy, 1e-12);

  // LAMMPS convention: delta = r_i - r_j, fpair = -dU/dr / r.
  // Force on atom 0: F = fpair * delta = (-dU/dr / r) * (r0 - r1).
  // Atom 0 at x=5, atom 1 at x=8: delta_x = -3.
  // At r > r0, dU/dr > 0, so fpair < 0.
  // F_0x = fpair * (-3) = positive -> toward atom 1. Correct: attractive.
  real dudr = -morse_force(r);  // dU/dr (positive at r > r0)
  real expected_f0x = dudr;     // = (-dU/dr / r) * (5 - 8) = (-dudr/3)*(-3)
  EXPECT_NEAR(state.forces[0].x, expected_f0x, 1e-12);
  EXPECT_NEAR(state.forces[0].y, 0.0, 1e-14);
  EXPECT_NEAR(state.forces[0].z, 0.0, 1e-14);

  // Newton 3rd: force on atom 1 = -force on atom 0.
  EXPECT_NEAR(state.forces[1].x, -expected_f0x, 1e-12);
  EXPECT_NEAR(state.forces[1].y, 0.0, 1e-14);
  EXPECT_NEAR(state.forces[1].z, 0.0, 1e-14);
}
