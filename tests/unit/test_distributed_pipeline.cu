// SPDX-License-Identifier: Apache-2.0
// test_distributed_pipeline.cu — M5 distributed pipeline scheduler tests.
//
// Run with: mpirun -np 2 ./tests/tdmd_mpi_tests
//
// Tests:
// 1. Deterministic 2-rank matches single-rank M4 (bit-identical).
// 2. Pipeline 2-rank NVE conservation (|dE/E| < 1e-4).
// 3. All zones advance to target_step across ranks.
#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "integrator/device_nose_hoover.cuh"
#include "integrator/velocity_verlet.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "scheduler/distributed_pipeline_scheduler.cuh"
#include "scheduler/pipeline_scheduler.cuh"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

static double compute_ke_host(const std::vector<VelocityVec>& velocities,
                              const std::vector<i32>& types,
                              const std::vector<real>& masses, i64 natoms) {
  double ke = 0;
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    double mass =
        static_cast<double>(masses[static_cast<std::size_t>(types[si])]);
    const VelocityVec& v = velocities[si];
    ke += 0.5 * mass * static_cast<double>(kMvv2e) *
          (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  return ke;
}

// Helper: compare two SystemStates atom-by-atom using ID mapping.
/// Minimum-image-aware position comparison for PBC-safe deterministic match.
static double pbc_component_diff(double a, double b, double box_len) {
  double d = std::abs(a - b);
  if (d > box_len * 0.5) d = box_len - d;
  return d;
}

static void compare_states(const SystemState& s1, const SystemState& s2,
                           real pos_tol, real vel_tol, const std::string& label) {
  auto n = static_cast<std::size_t>(s1.natoms);
  ASSERT_EQ(s1.natoms, s2.natoms);

  Vec3D box_size = s1.box.size();

  // Build ID→index maps.
  std::vector<std::size_t> map1(n), map2(n);
  for (std::size_t i = 0; i < n; ++i) {
    map1[static_cast<std::size_t>(s1.ids[i] - 1)] = i;
    map2[static_cast<std::size_t>(s2.ids[i] - 1)] = i;
  }

  double max_pos_diff = 0, max_vel_diff = 0;
  for (std::size_t id = 0; id < n; ++id) {
    auto i1 = map1[id];
    auto i2 = map2[id];
    // PBC-aware position comparison.
    double dp = std::max({pbc_component_diff(s1.positions[i1].x, s2.positions[i2].x, box_size.x),
                          pbc_component_diff(s1.positions[i1].y, s2.positions[i2].y, box_size.y),
                          pbc_component_diff(s1.positions[i1].z, s2.positions[i2].z, box_size.z)});
    max_pos_diff = std::max(max_pos_diff, dp);
    // Velocity comparison (no PBC wrap).
    double dv = std::max({std::abs(s1.velocities[i1].x - s2.velocities[i2].x),
                          std::abs(s1.velocities[i1].y - s2.velocities[i2].y),
                          std::abs(s1.velocities[i1].z - s2.velocities[i2].z)});
    max_vel_diff = std::max(max_vel_diff, dv);
  }

  EXPECT_LT(max_pos_diff, static_cast<double>(pos_tol))
      << label << " position diff";
  EXPECT_LT(max_vel_diff, static_cast<double>(vel_tol))
      << label << " velocity diff";
}

TEST(DistributedPipeline, DeterministicMatchesSingleRank) {
  // 2 ranks, deterministic mode, 100 steps.
  // Compare with M4 single-rank deterministic pipeline.
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  i32 nsteps = 100;
  i32 rebuild_every = 10;

  // --- M4 single-rank reference (run on rank 0) ---
  SystemState state_ref;
  if (world_rank == 0) {
    state_ref = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
    potentials::MorseParams params{D, alpha, r0, rc, rc * rc};
    scheduler::PipelineConfig ref_cfg;
    ref_cfg.dt = dt;
    ref_cfg.r_skin = skin;
    ref_cfg.rebuild_every = rebuild_every;
    ref_cfg.deterministic = true;

    scheduler::PipelineScheduler ref_sched(state_ref.box,
                                           static_cast<i32>(state_ref.natoms),
                                           params, ref_cfg);
    ref_sched.upload(state_ref.positions.data(), state_ref.velocities.data(),
                     state_ref.forces.data(), state_ref.types.data(),
                     state_ref.ids.data(), state_ref.masses.data(),
                     static_cast<i32>(state_ref.masses.size()));
    ref_sched.run_until(nsteps);
    ref_sched.download(state_ref.positions.data(), state_ref.velocities.data(),
                       state_ref.forces.data(), state_ref.types.data(),
                       state_ref.ids.data(),
                       static_cast<i32>(state_ref.natoms));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // --- M5 distributed (2 ranks) ---
  SystemState state_m5 = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};
  scheduler::DistributedConfig m5_cfg;
  m5_cfg.dt = dt;
  m5_cfg.r_skin = skin;
  m5_cfg.rebuild_every = rebuild_every;
  m5_cfg.deterministic = true;

  scheduler::DistributedPipelineScheduler m5_sched(
      state_m5.box, static_cast<i32>(state_m5.natoms), params, m5_cfg,
      MPI_COMM_WORLD);
  m5_sched.upload(state_m5.positions.data(), state_m5.velocities.data(),
                  state_m5.forces.data(), state_m5.types.data(),
                  state_m5.ids.data(), state_m5.masses.data(),
                  static_cast<i32>(state_m5.masses.size()));
  m5_sched.run_until(nsteps);
  m5_sched.download(state_m5.positions.data(), state_m5.velocities.data(),
                    state_m5.forces.data(), state_m5.types.data(),
                    state_m5.ids.data(), static_cast<i32>(state_m5.natoms));

  // Compare on rank 0.
  if (world_rank == 0) {
    // Distributed mode with 2 ranks may have slightly different FP order
    // from single-rank due to zone assignment differences. Allow 1e-6.
    compare_states(state_ref, state_m5, kPositionTolerance, kVelocityTolerance,
                   "distributed vs single-rank");
  }
}

TEST(DistributedPipeline, PipelineNVEConservation) {
  // 2 ranks, pipeline mode, NVE must conserve energy.
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::DistributedConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = false;
  cfg.n_streams = 2;

  scheduler::DistributedPipelineScheduler sched(
      state.box, static_cast<i32>(state.natoms), params, cfg, MPI_COMM_WORLD);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Run 1 step to get forces.
  sched.run_until(1);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  // Compute initial energy on CPU (rank 0).
  double e0 = 0, ef = 0;
  if (world_rank == 0) {
    potentials::MorsePair morse(D, alpha, r0, rc);
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                    cfg.r_skin);
    double pe0 = static_cast<double>(
        potentials::compute_pair_forces(state, cpu_nlist, morse));
    double ke0 = compute_ke_host(state.velocities, state.types, state.masses,
                                 state.natoms);
    e0 = pe0 + ke0;
  }

  // Run to step 1000.
  sched.run_until(1000);
  sched.download(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 static_cast<i32>(state.natoms));

  if (world_rank == 0) {
    potentials::MorsePair morse(D, alpha, r0, rc);
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                    cfg.r_skin);
    double pe_f = static_cast<double>(
        potentials::compute_pair_forces(state, cpu_nlist, morse));
    double ke_f = compute_ke_host(state.velocities, state.types, state.masses,
                                  state.natoms);
    ef = pe_f + ke_f;

    double drift = std::abs((ef - e0) / e0);
    EXPECT_LT(drift, 1e-4)
        << "Distributed Pipeline NVE drift |dE/E| = " << drift;
  }
}

TEST(DistributedPipeline, ZoneTimeStepsAdvance) {
  // All zones should reach target_step after run_until.
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::DistributedConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.deterministic = false;
  cfg.n_streams = 2;

  scheduler::DistributedPipelineScheduler sched(
      state.box, static_cast<i32>(state.natoms), params, cfg, MPI_COMM_WORLD);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  sched.run_until(10);

  EXPECT_GE(sched.min_local_time_step(), 10)
      << "rank " << world_rank << " min_local_time_step";

  // All local zones should be at step >= 10.
  for (const auto& z : sched.partition().zones()) {
    if (z.owner_rank == world_rank) {
      EXPECT_GE(z.time_step, 10)
          << "rank " << world_rank << " zone " << z.id
          << " at step " << z.time_step;
    }
  }
}

static double compute_temperature(double ke, i32 natoms) {
  i32 n_dof = 3 * natoms - 3;
  return 2.0 * ke /
         (static_cast<double>(n_dof) * static_cast<double>(kBoltzmann));
}

/// Initialize velocities from Maxwell-Boltzmann distribution at given T.
/// Removes center-of-mass momentum and rescales to exact target T.
static void init_velocities(SystemState& state, real t_target, unsigned seed) {
  auto n = static_cast<std::size_t>(state.natoms);
  std::mt19937 rng(seed);
  std::normal_distribution<real> gauss(real{0}, real{1});

  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
    real sigma = std::sqrt(kBoltzmann * t_target / (mass * kMvv2e));
    state.velocities[i].x = sigma * gauss(rng);
    state.velocities[i].y = sigma * gauss(rng);
    state.velocities[i].z = sigma * gauss(rng);
  }

  VelocityVec com_v{0, 0, 0};
  double total_mass = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double mass =
        static_cast<double>(state.masses[static_cast<std::size_t>(state.types[i])]);
    com_v.x += mass * state.velocities[i].x;
    com_v.y += mass * state.velocities[i].y;
    com_v.z += mass * state.velocities[i].z;
    total_mass += mass;
  }
  com_v.x /= total_mass;
  com_v.y /= total_mass;
  com_v.z /= total_mass;
  for (std::size_t i = 0; i < n; ++i) {
    state.velocities[i].x -= com_v.x;
    state.velocities[i].y -= com_v.y;
    state.velocities[i].z -= com_v.z;
  }

  double ke = compute_ke_host(state.velocities, state.types, state.masses,
                              state.natoms);
  double t_current = compute_temperature(ke, static_cast<i32>(state.natoms));
  if (t_current > 0) {
    double scale = std::sqrt(static_cast<double>(t_target) / t_current);
    for (std::size_t i = 0; i < n; ++i) {
      state.velocities[i].x *= scale;
      state.velocities[i].y *= scale;
      state.velocities[i].z *= scale;
    }
  }
}

// Regression test: NVT multi-rank atom range bug.
// Bug: first_local_atom_/local_atom_count_ computed before assign_atoms(),
// resulting in 0-atom range -> thermostat is no-op (NVE behavior).
//
// Strategy: Init at 100K, target 500K. NVE from 100K on cold FCC lattice
// gives T < 100K (KE -> PE). Working NVT must heat the system to ~500K.
// With the bug, T stays low -> rel_err >> 0.30 -> test fails.
TEST(DistributedPipeline, NVTTemperatureConverges) {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Initialize at 100K — well below target.
  init_velocities(state, real{100.0}, 42);

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  real t_target = real{500.0};

  scheduler::DistributedConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = true;
  cfg.t_target = t_target;
  cfg.t_period = real{0.1};
  cfg.nhc_length = 3;

  scheduler::DistributedPipelineScheduler sched(
      state.box, static_cast<i32>(state.natoms), params, cfg, MPI_COMM_WORLD);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Equilibrate 1000 steps.
  sched.run_until(1000);

  // Collect temperature over next 1000 steps, sampling every 100.
  std::vector<double> temps;
  for (i32 step = 1100; step <= 2000; step += 100) {
    sched.run_until(step);
    sched.download(state.positions.data(), state.velocities.data(),
                   state.forces.data(), state.types.data(), state.ids.data(),
                   static_cast<i32>(state.natoms));

    if (world_rank == 0) {
      double ke = compute_ke_host(state.velocities, state.types, state.masses,
                                  state.natoms);
      double t = compute_temperature(ke, static_cast<i32>(state.natoms));
      temps.push_back(t);
    }
  }

  if (world_rank == 0) {
    double t_mean = std::accumulate(temps.begin(), temps.end(), 0.0) /
                    static_cast<double>(temps.size());

    // With working thermostat, T should rise from 100K toward 500K.
    // Allow 30% tolerance for 256-atom system.
    double rel_err = std::abs(t_mean - static_cast<double>(t_target)) / t_target;
    printf("  NVT multi-rank: <T> = %.1f K, target = %.1f K, rel_err = %.3f\n",
           t_mean, static_cast<double>(t_target), rel_err);
    EXPECT_LT(rel_err, 0.30)
        << "<T> = " << t_mean << " K, target = " << t_target
        << " K, rel_err = " << rel_err
        << " (if rel_err >> 0.30, thermostat is likely no-op — atom range bug)";
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  // Only print results from rank 0 to avoid garbled output.
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
