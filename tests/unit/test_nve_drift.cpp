// SPDX-License-Identifier: Apache-2.0
// test_nve_drift.cpp — NVE energy conservation test on 256-atom FCC Cu.
//
// M1 exit criterion: |dE/E| < 1e-4 over 50 000 steps at dt=1 fs.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <random>

#include "core/constants.hpp"
#include "core/math.hpp"
#include "core/types.hpp"
#include "integrator/velocity_verlet.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"

#ifndef TDMD_TEST_DATA_DIR
#error "TDMD_TEST_DATA_DIR must be defined"
#endif

static std::string data_dir() { return TDMD_TEST_DATA_DIR; }

using namespace tdmd;
using namespace tdmd::potentials;
using namespace tdmd::neighbors;
using namespace tdmd::integrator;

static real kinetic_energy(const SystemState& s) {
  real ke = 0;
  for (i64 i = 0; i < s.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    real mass = s.masses[static_cast<std::size_t>(s.types[si])];
    ke += real{0.5} * mass * kMvv2e * length_sq(s.velocities[si]);
  }
  return ke;
}

/// Initialize Maxwell-Boltzmann velocities at temperature T (K).
static void init_velocities(SystemState& s, real temp, unsigned seed) {
  std::mt19937 rng(seed);
  // sigma = sqrt(kB * T / (m * mvv2e)) for velocity in A/ps.
  for (i64 i = 0; i < s.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    real mass = s.masses[static_cast<std::size_t>(s.types[si])];
    real sigma = std::sqrt(kBoltzmann * temp / (mass * kMvv2e));
    std::normal_distribution<real> dist(real{0}, sigma);
    s.velocities[si] = {dist(rng), dist(rng), dist(rng)};
  }

  // Remove net momentum.
  VelocityVec total_mom = {0, 0, 0};
  double total_mass = 0;
  for (i64 i = 0; i < s.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    double mass = s.masses[static_cast<std::size_t>(s.types[si])];
    total_mom += mass * s.velocities[si];
    total_mass += mass;
  }
  VelocityVec vcm = (1.0 / total_mass) * total_mom;
  for (i64 i = 0; i < s.natoms; ++i) {
    s.velocities[static_cast<std::size_t>(i)] -= vcm;
  }
}

TEST(NVEDrift, Morse256Atoms50kSteps) {
  auto state = io::read_lammps_data(data_dir() + "/cu_fcc_256.data");

  // Initialize at 300 K.
  init_velocities(state, real{300}, 42);

  MorsePair pot(0.3429, 1.3588, 2.866, 6.0);
  VelocityVerlet vv(real{0.001});  // 1 fs

  NeighborList nlist;
  real skin = real{1.0};
  nlist.build(state.positions.data(), state.natoms, state.box,
              pot.cutoff(), skin);

  real pe = compute_pair_forces(state, nlist, pot);
  real ke = kinetic_energy(state);
  real E0 = pe + ke;

  const int nsteps = 50000;
  for (int step = 0; step < nsteps; ++step) {
    vv.half_kick(state);
    vv.drift(state);

    if (nlist.needs_rebuild(state.positions.data(), state.natoms)) {
      nlist.build(state.positions.data(), state.natoms, state.box,
                  pot.cutoff(), skin);
    }

    pe = compute_pair_forces(state, nlist, pot);
    vv.half_kick(state);
  }

  ke = kinetic_energy(state);
  real E_final = pe + ke;
  real drift = std::abs((E_final - E0) / E0);

  std::printf("NVE 50k steps: E0=%.8f  E_final=%.8f  drift=%.2e\n",
              static_cast<double>(E0), static_cast<double>(E_final),
              static_cast<double>(drift));

  // M1 exit criterion: |dE/E| < 1e-4.
  EXPECT_LT(drift, 1e-4) << "Energy drift exceeds M1 criterion";
}
