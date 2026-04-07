// SPDX-License-Identifier: Apache-2.0
// test_sequential_scheduler.cpp — verify zone-walked MD matches global MD.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/types.hpp"
#include "integrator/velocity_verlet.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "scheduler/sequential_scheduler.hpp"

using namespace tdmd;

// Helper: compute total energy (KE + PE).
static real total_energy(const SystemState& state, real pe) {
  real ke = 0;
  for (i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    real mass = state.masses[static_cast<std::size_t>(state.types[si])];
    const Vec3& v = state.velocities[si];
    ke += real{0.5} * mass * kMvv2e * (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  return ke + pe;
}

TEST(SequentialScheduler, MatchesGlobalMD) {
  // Run 100 steps with both global MD (M2 style) and zone-walked scheduler.
  // Forces and positions must match to machine precision.
  std::string data_dir = TDMD_TEST_DATA_DIR;

  // Clone state for two independent runs.
  SystemState state_global = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
  SystemState state_zoned = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  i32 nsteps = 100;
  i32 rebuild_every = 10;

  // --- Global MD (M2 style) ---
  potentials::MorsePair morse_global(D, alpha, r0, rc);
  neighbors::NeighborList nlist_global;
  integrator::VelocityVerlet vv(dt);

  nlist_global.build(state_global.positions.data(), state_global.natoms,
                     state_global.box, rc, skin);
  potentials::compute_pair_forces(state_global, nlist_global, morse_global);

  for (i32 s = 0; s < nsteps; ++s) {
    vv.half_kick(state_global);
    vv.drift(state_global);
    if ((s + 1) % rebuild_every == 0) {
      nlist_global.build(state_global.positions.data(), state_global.natoms,
                         state_global.box, rc, skin);
    }
    potentials::compute_pair_forces(state_global, nlist_global, morse_global);
    vv.half_kick(state_global);
  }

  // --- Zone-walked MD ---
  potentials::MorsePair morse_zoned(D, alpha, r0, rc);
  scheduler::SequentialSchedulerConfig cfg;
  cfg.r_skin = skin;
  cfg.rebuild_every = rebuild_every;
  cfg.n_zones = 0;  // auto

  scheduler::SequentialScheduler sched(state_zoned, morse_zoned, cfg);
  sched.run(nsteps, dt);

  // --- Compare ---
  // Atoms may be in different order (zone-walked reorders by zone).
  // Build id→index maps for comparison.
  auto n = static_cast<std::size_t>(state_global.natoms);
  std::vector<std::size_t> global_map(n), zoned_map(n);
  for (std::size_t i = 0; i < n; ++i) {
    global_map[static_cast<std::size_t>(state_global.ids[i] - 1)] = i;
    zoned_map[static_cast<std::size_t>(state_zoned.ids[i] - 1)] = i;
  }

  real max_pos_diff = 0, max_vel_diff = 0, max_force_diff = 0;
  for (std::size_t id = 0; id < n; ++id) {
    auto ig = global_map[id];
    auto iz = zoned_map[id];

    auto dp = [](Vec3 a, Vec3 b) {
      return std::max({std::abs(a.x - b.x), std::abs(a.y - b.y),
                       std::abs(a.z - b.z)});
    };
    max_pos_diff =
        std::max(max_pos_diff,
                 dp(state_global.positions[ig], state_zoned.positions[iz]));
    max_vel_diff =
        std::max(max_vel_diff,
                 dp(state_global.velocities[ig], state_zoned.velocities[iz]));
    max_force_diff =
        std::max(max_force_diff,
                 dp(state_global.forces[ig], state_zoned.forces[iz]));
  }

  // Should be bit-identical since the only difference is atom ordering,
  // and forces use the same global neighbor list.
  // Allow tiny tolerance for floating-point reordering effects.
  EXPECT_LT(max_pos_diff, real{1e-12})
      << "position mismatch after " << nsteps << " steps";
  EXPECT_LT(max_vel_diff, real{1e-12})
      << "velocity mismatch after " << nsteps << " steps";
  EXPECT_LT(max_force_diff, real{1e-10})
      << "force mismatch after " << nsteps << " steps";
}

TEST(SequentialScheduler, NVEConservation) {
  // Zone-walked NVE should conserve energy same as global MD.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorsePair morse(D, alpha, r0, rc);

  scheduler::SequentialSchedulerConfig cfg;
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;

  scheduler::SequentialScheduler sched(state, morse, cfg);

  // Get initial energy.
  neighbors::NeighborList tmp_nlist;
  tmp_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe0 = potentials::compute_pair_forces(state, tmp_nlist, morse);
  real e0 = total_energy(state, pe0);

  // Run 1000 steps.
  sched.run(1000, real{0.001});

  // Recompute energy.
  tmp_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe_final = potentials::compute_pair_forces(state, tmp_nlist, morse);
  real e_final = total_energy(state, pe_final);

  real drift = std::abs((e_final - e0) / e0);
  EXPECT_LT(drift, real{1e-4})
      << "Zone-walked NVE drift |dE/E| = " << drift;
}
