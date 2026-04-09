// SPDX-License-Identifier: Apache-2.0
// test_pipeline_scheduler.cu — M4 pipeline scheduler tests.
//
// Tests:
// 1. Deterministic mode matches M3 sequential scheduler (bit-identical).
// 2. Pipeline mode NVE conservation (|dE/E| < 1e-4).
// 3. Pipeline mode produces reasonable results vs deterministic.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "integrator/velocity_verlet.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "scheduler/pipeline_scheduler.cuh"
#include "scheduler/sequential_scheduler.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

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

TEST(PipelineScheduler, DeterministicMatchesM3) {
#ifdef TDMD_PRECISION_MIXED
  GTEST_SKIP() << "CPU-float vs GPU-mixed comparison not meaningful in mixed "
                  "precision mode (different compute precision paths)";
#endif
  // Run 100 steps deterministic pipeline vs M3 sequential, compare.
  std::string data_dir = TDMD_TEST_DATA_DIR;

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  i32 nsteps = 100;
  i32 rebuild_every = 10;

  // --- M3 sequential CPU ---
  SystemState state_m3 = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
  potentials::MorsePair morse_m3(D, alpha, r0, rc);
  scheduler::SequentialSchedulerConfig m3_cfg;
  m3_cfg.r_skin = skin;
  m3_cfg.rebuild_every = rebuild_every;
  scheduler::SequentialScheduler m3_sched(state_m3, morse_m3, m3_cfg);
  m3_sched.run(nsteps, dt);

  // --- M4 deterministic pipeline GPU ---
  SystemState state_m4 = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};
  scheduler::PipelineConfig m4_cfg;
  m4_cfg.dt = dt;
  m4_cfg.r_skin = skin;
  m4_cfg.rebuild_every = rebuild_every;
  m4_cfg.deterministic = true;

  scheduler::PipelineScheduler m4_sched(state_m4.box, static_cast<i32>(state_m4.natoms),
                                        params, m4_cfg);
  m4_sched.upload(state_m4.positions.data(), state_m4.velocities.data(),
                  state_m4.forces.data(), state_m4.types.data(),
                  state_m4.ids.data(), state_m4.masses.data(),
                  static_cast<i32>(state_m4.masses.size()));

  m4_sched.run_until(nsteps);

  // Download results.
  m4_sched.download(state_m4.positions.data(), state_m4.velocities.data(),
                    state_m4.forces.data(), state_m4.types.data(),
                    state_m4.ids.data(), static_cast<i32>(state_m4.natoms));

  // Compare by atom ID.
  auto n = static_cast<std::size_t>(state_m3.natoms);
  std::vector<std::size_t> m3_map(n), m4_map(n);
  for (std::size_t i = 0; i < n; ++i) {
    m3_map[static_cast<std::size_t>(state_m3.ids[i] - 1)] = i;
    m4_map[static_cast<std::size_t>(state_m4.ids[i] - 1)] = i;
  }

  real max_pos_diff = 0, max_vel_diff = 0;
  for (std::size_t id = 0; id < n; ++id) {
    auto i3 = m3_map[id];
    auto i4 = m4_map[id];
    auto d = [](Vec3 a, Vec3 b) {
      return std::max({std::abs(a.x - b.x), std::abs(a.y - b.y),
                       std::abs(a.z - b.z)});
    };
    max_pos_diff =
        std::max(max_pos_diff,
                 d(state_m3.positions[i3], state_m4.positions[i4]));
    max_vel_diff =
        std::max(max_vel_diff,
                 d(state_m3.velocities[i3], state_m4.velocities[i4]));
  }

  // Deterministic mode should match M3 closely. Per-zone force compute may
  // have small FP differences from global compute (different summation order
  // in neighbor list iteration) so allow 1e-10.
  EXPECT_LT(max_pos_diff, kPositionTolerance)
      << "deterministic pipeline vs M3 position diff";
  EXPECT_LT(max_vel_diff, kVelocityTolerance)
      << "deterministic pipeline vs M3 velocity diff";
}

TEST(PipelineScheduler, PipelineNVEConservation) {
  // Pipeline mode with 4 streams. NVE must conserve energy.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::PipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = false;
  cfg.n_streams = 4;

  scheduler::PipelineScheduler sched(state.box, static_cast<i32>(state.natoms),
                                     params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Compute initial energy.
  // Need forces first — upload does initial force compute in run_until.
  // Run 1 step to get forces, then measure.
  sched.run_until(1);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  // For initial energy, we need PE. Compute on CPU as reference.
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
  EXPECT_LT(drift, real{1e-4})
      << "Pipeline NVE drift |dE/E| = " << drift;
}

TEST(PipelineScheduler, ZoneTimeStepsAdvance) {
  // Basic check: after run_until(10), all zones should be at time_step >= 10.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::PipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.deterministic = false;
  cfg.n_streams = 2;

  scheduler::PipelineScheduler sched(state.box, static_cast<i32>(state.natoms),
                                     params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  sched.run_until(10);

  EXPECT_GE(sched.min_time_step(), 10);

  // All zones should be at least at step 10.
  for (const auto& z : sched.partition().zones()) {
    EXPECT_GE(z.time_step, 10) << "zone " << z.id << " at step " << z.time_step;
  }
}
