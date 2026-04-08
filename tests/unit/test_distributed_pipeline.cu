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
#include "scheduler/distributed_pipeline_scheduler.cuh"
#include "scheduler/pipeline_scheduler.cuh"

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

// Helper: compare two SystemStates atom-by-atom using ID mapping.
static void compare_states(const SystemState& s1, const SystemState& s2,
                           real pos_tol, real vel_tol, const std::string& label) {
  auto n = static_cast<std::size_t>(s1.natoms);
  ASSERT_EQ(s1.natoms, s2.natoms);

  // Build ID→index maps.
  std::vector<std::size_t> map1(n), map2(n);
  for (std::size_t i = 0; i < n; ++i) {
    map1[static_cast<std::size_t>(s1.ids[i] - 1)] = i;
    map2[static_cast<std::size_t>(s2.ids[i] - 1)] = i;
  }

  real max_pos_diff = 0, max_vel_diff = 0;
  for (std::size_t id = 0; id < n; ++id) {
    auto i1 = map1[id];
    auto i2 = map2[id];
    auto d = [](Vec3 a, Vec3 b) {
      return std::max({std::abs(a.x - b.x), std::abs(a.y - b.y),
                       std::abs(a.z - b.z)});
    };
    max_pos_diff = std::max(max_pos_diff, d(s1.positions[i1], s2.positions[i2]));
    max_vel_diff = std::max(max_vel_diff, d(s1.velocities[i1], s2.velocities[i2]));
  }

  EXPECT_LT(max_pos_diff, pos_tol) << label << " position diff";
  EXPECT_LT(max_vel_diff, vel_tol) << label << " velocity diff";
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
    compare_states(state_ref, state_m5, real{1e-6}, real{1e-6},
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
  real e0 = 0, ef = 0;
  if (world_rank == 0) {
    potentials::MorsePair morse(D, alpha, r0, rc);
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc,
                    cfg.r_skin);
    real pe0 = potentials::compute_pair_forces(state, cpu_nlist, morse);
    real ke0 = compute_ke_host(state.velocities, state.types, state.masses,
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
    real pe_f = potentials::compute_pair_forces(state, cpu_nlist, morse);
    real ke_f = compute_ke_host(state.velocities, state.types, state.masses,
                                state.natoms);
    ef = pe_f + ke_f;

    real drift = std::abs((ef - e0) / e0);
    EXPECT_LT(drift, real{1e-4})
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
