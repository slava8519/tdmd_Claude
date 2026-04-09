// SPDX-License-Identifier: Apache-2.0
// test_fast_pipeline.cu — FastPipelineScheduler tests (ADR 0005 Phase 2).
//
// Tests:
// 1. NVE energy conservation (same tolerance as PipelineScheduler).
// 2. Sharp kernel-launch invariant: exactly 5 launches per step.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "scheduler/fast_pipeline_scheduler.cuh"

using namespace tdmd;

static real compute_ke_host(const std::vector<Vec3>& velocities,
                            const std::vector<i32>& types,
                            const std::vector<real>& masses, i64 natoms) {
  real ke = 0;
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    real mass = masses[static_cast<std::size_t>(types[si])];
    const Vec3& v = velocities[si];
    ke += real{0.5} * mass * kMvv2e * (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  return ke;
}

TEST(FastPipelineScheduler, NVEConservation) {
  // Load same 256-atom Cu FCC system as PipelineNVEConservation.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;

  scheduler::FastPipelineScheduler sched(state.box,
                                         static_cast<i32>(state.natoms),
                                         params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Run 1 step to compute initial forces, then measure E0.
  sched.run_until(1);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  // CPU reference for PE.
  potentials::MorsePair morse(D, alpha, r0, rc);
  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe0 = potentials::compute_pair_forces(state, cpu_nlist, morse);
  real ke0 = compute_ke_host(state.velocities, state.types, state.masses,
                             state.natoms);
  real e0 = pe0 + ke0;

  // Run to step 1000.
  sched.run_until(1000);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe_f = potentials::compute_pair_forces(state, cpu_nlist, morse);
  real ke_f = compute_ke_host(state.velocities, state.types, state.masses,
                              state.natoms);
  real ef = pe_f + ke_f;

  real drift = std::abs((ef - e0) / e0);
  std::printf("  NVEConservation (1k steps): E0=%.6f  Ef=%.6f  |dE/E|=%.2e\n",
              static_cast<double>(e0), static_cast<double>(ef),
              static_cast<double>(drift));
  EXPECT_LT(drift, real{1e-4})
      << "FastPipeline NVE drift |dE/E| = " << drift;
}

TEST(FastPipelineScheduler, LongNVEDrift) {
  // 10 000 steps on 256-atom system — check FP32 doesn't accumulate
  // unacceptable energy drift over long runs.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;

  scheduler::FastPipelineScheduler sched(state.box,
                                         static_cast<i32>(state.natoms),
                                         params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Run 1 step for initial forces, measure E0.
  sched.run_until(1);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  potentials::MorsePair morse(D, alpha, r0, rc);
  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe0 = potentials::compute_pair_forces(state, cpu_nlist, morse);
  real ke0 = compute_ke_host(state.velocities, state.types, state.masses,
                             state.natoms);
  real e0 = pe0 + ke0;

  // Run to step 10000.
  sched.run_until(10000);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                  cfg.r_skin);
  real pe_f = potentials::compute_pair_forces(state, cpu_nlist, morse);
  real ke_f = compute_ke_host(state.velocities, state.types, state.masses,
                              state.natoms);
  real ef = pe_f + ke_f;

  real drift = std::abs((ef - e0) / e0);
  std::printf("  LongNVEDrift (10k steps): E0=%.6f  Ef=%.6f  |dE/E|=%.2e\n",
              static_cast<double>(e0), static_cast<double>(ef),
              static_cast<double>(drift));
  // FP32 accumulates more error; 1e-3 is a reasonable threshold for 10k steps.
  EXPECT_LT(drift, real{1e-3})
      << "FastPipeline long NVE drift |dE/E| = " << drift
      << " (E0=" << e0 << ", Ef=" << ef << ")";
}

TEST(FastPipelineScheduler, KernelLaunchInvariant) {
  // Sharp ADR 0005 invariant: exactly 5 kernel launches per step,
  // plus 2 initial launches (zero_forces + force compute before step 1).
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;

  scheduler::FastPipelineScheduler sched(state.box,
                                         static_cast<i32>(state.natoms),
                                         params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  constexpr i32 nsteps = 100;
  sched.run_until(nsteps);

  auto stats = sched.stats();

  // 2 initial (zero_forces + force) + 5 per step * nsteps.
  i64 expected = 2 + 5 * static_cast<i64>(nsteps);
  EXPECT_EQ(stats.kernel_launches, expected)
      << "Expected " << expected << " kernel launches for " << nsteps
      << " steps, got " << stats.kernel_launches;

  EXPECT_EQ(stats.ticks, static_cast<i64>(nsteps));
}
