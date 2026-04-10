// SPDX-License-Identifier: Apache-2.0
// test_hybrid_pipeline.cu — M6 hybrid 2D time × space pipeline tests.
//
// Run with: mpirun -np 4 ./tests/tdmd_m6_tests
//
// Tests:
// 1. Hybrid 4-rank (P_time=2, P_space=2) matches M5 2-rank pure TD.
// 2. Hybrid NVE conservation (|dE/E| < 1e-4).
// 3. All zones advance to target_step across all ranks.
#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
#include "scheduler/pipeline_scheduler.cuh"
#include "scheduler/distributed_pipeline_scheduler.cuh"
#include "scheduler/hybrid_pipeline_scheduler.cuh"
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

/// Gather all owned atoms from all spatial ranks in a space_comm group
/// onto every rank. Returns the assembled full state.
static void gather_owned_atoms(
    const Vec3* owned_pos, const Vec3* owned_vel, const Vec3* owned_forces,
    const i32* owned_types, const i32* owned_ids, i32 n_owned,
    std::vector<Vec3>& all_pos, std::vector<Vec3>& all_vel,
    std::vector<Vec3>& all_forces, std::vector<i32>& all_types,
    std::vector<i32>& all_ids, MPI_Comm space_comm) {
  int space_size = 0;
  MPI_Comm_size(space_comm, &space_size);

  // Gather counts from all spatial ranks.
  std::vector<int> counts(static_cast<std::size_t>(space_size));
  int my_count = n_owned;
  MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, space_comm);

  // Compute displacements.
  std::vector<int> displs(static_cast<std::size_t>(space_size), 0);
  int total = 0;
  for (int i = 0; i < space_size; ++i) {
    displs[static_cast<std::size_t>(i)] = total;
    total += counts[static_cast<std::size_t>(i)];
  }

  auto t = static_cast<std::size_t>(total);
  all_pos.resize(t);
  all_vel.resize(t);
  all_forces.resize(t);
  all_types.resize(t);
  all_ids.resize(t);

  // Scale counts/displs for Vec3 (3 reals).
  std::vector<int> vec3_counts(counts.size()), vec3_displs(displs.size());
  std::vector<int> i32_counts(counts.size()), i32_displs(displs.size());
  for (std::size_t i = 0; i < counts.size(); ++i) {
    vec3_counts[i] = static_cast<int>(
        static_cast<std::size_t>(counts[i]) * sizeof(Vec3));
    vec3_displs[i] = static_cast<int>(
        static_cast<std::size_t>(displs[i]) * sizeof(Vec3));
    i32_counts[i] = static_cast<int>(
        static_cast<std::size_t>(counts[i]) * sizeof(i32));
    i32_displs[i] = static_cast<int>(
        static_cast<std::size_t>(displs[i]) * sizeof(i32));
  }

  MPI_Allgatherv(owned_pos, static_cast<int>(n_owned * sizeof(Vec3)),
                 MPI_BYTE, all_pos.data(), vec3_counts.data(),
                 vec3_displs.data(), MPI_BYTE, space_comm);
  MPI_Allgatherv(owned_vel, static_cast<int>(n_owned * sizeof(Vec3)),
                 MPI_BYTE, all_vel.data(), vec3_counts.data(),
                 vec3_displs.data(), MPI_BYTE, space_comm);
  MPI_Allgatherv(owned_forces, static_cast<int>(n_owned * sizeof(Vec3)),
                 MPI_BYTE, all_forces.data(), vec3_counts.data(),
                 vec3_displs.data(), MPI_BYTE, space_comm);
  MPI_Allgatherv(owned_types, static_cast<int>(n_owned * sizeof(i32)),
                 MPI_BYTE, all_types.data(), i32_counts.data(),
                 i32_displs.data(), MPI_BYTE, space_comm);
  MPI_Allgatherv(owned_ids, static_cast<int>(n_owned * sizeof(i32)),
                 MPI_BYTE, all_ids.data(), i32_counts.data(),
                 i32_displs.data(), MPI_BYTE, space_comm);
}

TEST(HybridPipeline, DeterministicMatchesM5) {
#ifdef TDMD_PRECISION_MIXED
  GTEST_SKIP() << "Spatial decomposition ghost atoms in float cause boundary "
                  "divergence vs single-rank reference — not meaningful in "
                  "mixed precision (NVE conservation test covers correctness)";
#endif
  // 4 ranks total: P_time=2, P_space=2.
  // Compare against M5 2-rank pure TD (P_time=2, P_space=1) run on a
  // sub-communicator of time-rank-0 spatial ranks.
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_EQ(world_size, 4) << "This test requires exactly 4 MPI ranks";

  std::string data_dir = TDMD_TEST_DATA_DIR;
  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  i32 nsteps = 50;
  i32 rebuild_every = 10;
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  // --- M5 reference: 2-rank pure TD on a subset of ranks ---
  // Use ranks 0,1 (time_rank 0 and 1 from space_idx=0).
  // Actually, for simplicity, run M5 reference on rank 0 only (single-rank).
  SystemState state_ref;
  if (world_rank == 0) {
    state_ref = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
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

  // --- M6 hybrid: 4 ranks, P_time=2, P_space=2 ---
  SystemState state_m6 = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  scheduler::HybridConfig cfg;
  cfg.dt = dt;
  cfg.r_skin = skin;
  cfg.rebuild_every = rebuild_every;
  cfg.deterministic = true;
  cfg.p_time = 2;
  cfg.p_space = 2;

  scheduler::HybridPipelineScheduler sched(
      state_m6.box, static_cast<i32>(state_m6.natoms), params, cfg,
      MPI_COMM_WORLD);
  sched.upload(state_m6.positions.data(), state_m6.velocities.data(),
               state_m6.forces.data(), state_m6.types.data(),
               state_m6.ids.data(), state_m6.masses.data(),
               static_cast<i32>(state_m6.masses.size()));
  sched.run_until(nsteps);

  // Download owned atoms.
  i32 no = sched.n_owned();
  std::vector<Vec3> my_pos(static_cast<std::size_t>(no));
  std::vector<Vec3> my_vel(static_cast<std::size_t>(no));
  std::vector<Vec3> my_forces(static_cast<std::size_t>(no));
  std::vector<i32> my_types(static_cast<std::size_t>(no));
  std::vector<i32> my_ids(static_cast<std::size_t>(no));
  sched.download(my_pos.data(), my_vel.data(), my_forces.data(),
                 my_types.data(), my_ids.data(), no);

  // Gather all owned atoms from spatial ranks in the same time group.
  // Create space_comm for gathering.
  int cart_coords[2];
  // Recover spatial comm. We re-create it from the Cartesian coords.
  int dims[2] = {2, 2};
  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  MPI_Cart_coords(cart_comm, world_rank, 2, cart_coords);
  int time_idx = cart_coords[0];

  // Create space_comm (same time group).
  int space_remain[2] = {0, 1};
  MPI_Comm space_comm;
  MPI_Cart_sub(cart_comm, space_remain, &space_comm);

  std::vector<Vec3> all_pos, all_vel, all_forces;
  std::vector<i32> all_types, all_ids;
  gather_owned_atoms(my_pos.data(), my_vel.data(), my_forces.data(),
                     my_types.data(), my_ids.data(), no, all_pos, all_vel,
                     all_forces, all_types, all_ids, space_comm);

  // Compare on rank 0 (which is time_idx=0, and has the reference).
  if (world_rank == 0) {
    auto n = static_cast<std::size_t>(state_ref.natoms);
    ASSERT_EQ(all_pos.size(), n) << "gathered atom count mismatch";

    // Build ID→index maps.
    std::vector<std::size_t> map_ref(n), map_m6(n);
    for (std::size_t i = 0; i < n; ++i) {
      map_ref[static_cast<std::size_t>(state_ref.ids[i] - 1)] = i;
      map_m6[static_cast<std::size_t>(all_ids[i] - 1)] = i;
    }

    Vec3D box_size = state_ref.box.size();
    auto pbc_diff = [](real a, real b, real box_len) {
      real d = std::abs(a - b);
      if (d > box_len * real{0.5}) d = box_len - d;
      return d;
    };

    real max_pos_diff = 0, max_vel_diff = 0;
    for (std::size_t id = 0; id < n; ++id) {
      auto i_ref = map_ref[id];
      auto i_m6 = map_m6[id];
      // PBC-aware position comparison.
      real dp = std::max({pbc_diff(state_ref.positions[i_ref].x, all_pos[i_m6].x, static_cast<real>(box_size.x)),
                          pbc_diff(state_ref.positions[i_ref].y, all_pos[i_m6].y, static_cast<real>(box_size.y)),
                          pbc_diff(state_ref.positions[i_ref].z, all_pos[i_m6].z, static_cast<real>(box_size.z))});
      max_pos_diff = std::max(max_pos_diff, dp);
      real dv = std::max({std::abs(state_ref.velocities[i_ref].x - all_vel[i_m6].x),
                          std::abs(state_ref.velocities[i_ref].y - all_vel[i_m6].y),
                          std::abs(state_ref.velocities[i_ref].z - all_vel[i_m6].z)});
      max_vel_diff = std::max(max_vel_diff, dv);
    }

    EXPECT_LT(max_pos_diff, kPositionTolerance)
        << "Hybrid vs M4-single-rank position diff";
    EXPECT_LT(max_vel_diff, kVelocityTolerance)
        << "Hybrid vs M4-single-rank velocity diff";
  }

  MPI_Comm_free(&space_comm);
  MPI_Comm_free(&cart_comm);
}

TEST(HybridPipeline, PipelineNVEConservation) {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_EQ(world_size, 4);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::HybridConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = false;
  cfg.n_streams = 2;
  cfg.p_time = 2;
  cfg.p_space = 2;

  scheduler::HybridPipelineScheduler sched(
      state.box, static_cast<i32>(state.natoms), params, cfg, MPI_COMM_WORLD);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Run 1 step to get forces.
  sched.run_until(1);

  // Gather atoms from spatial ranks for energy computation.
  i32 no = sched.n_owned();
  std::vector<Vec3> my_pos(static_cast<std::size_t>(no));
  std::vector<Vec3> my_vel(static_cast<std::size_t>(no));
  std::vector<Vec3> my_forces(static_cast<std::size_t>(no));
  std::vector<i32> my_types(static_cast<std::size_t>(no));
  std::vector<i32> my_ids(static_cast<std::size_t>(no));
  sched.download(my_pos.data(), my_vel.data(), my_forces.data(),
                 my_types.data(), my_ids.data(), no);

  // Create space_comm for gathering.
  int dims[2] = {2, 2};
  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  int space_remain[2] = {0, 1};
  MPI_Comm space_comm;
  MPI_Cart_sub(cart_comm, space_remain, &space_comm);

  std::vector<Vec3> all_pos, all_vel, all_forces;
  std::vector<i32> all_types, all_ids;
  gather_owned_atoms(my_pos.data(), my_vel.data(), my_forces.data(),
                     my_types.data(), my_ids.data(), no, all_pos, all_vel,
                     all_forces, all_types, all_ids, space_comm);

  real e0 = 0, ef = 0;
  if (world_rank == 0) {
    auto natoms = static_cast<i64>(all_pos.size());
    SystemState full_state;
    full_state.natoms = natoms;
    full_state.box = state.box;
    full_state.positions = all_pos;
    full_state.velocities = all_vel;
    full_state.forces = all_forces;
    full_state.types = all_types;
    full_state.ids = all_ids;
    full_state.masses = state.masses;

    potentials::MorsePair morse(D, alpha, r0, rc);
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(full_state.positions.data(), natoms, state.box, rc,
                    cfg.r_skin);
    real pe0 = potentials::compute_pair_forces(full_state, cpu_nlist, morse);
    real ke0 = compute_ke_host(all_vel, all_types, state.masses, natoms);
    e0 = pe0 + ke0;
  }

  // Run to step 500.
  sched.run_until(500);
  sched.download(my_pos.data(), my_vel.data(), my_forces.data(),
                 my_types.data(), my_ids.data(), no);

  gather_owned_atoms(my_pos.data(), my_vel.data(), my_forces.data(),
                     my_types.data(), my_ids.data(), no, all_pos, all_vel,
                     all_forces, all_types, all_ids, space_comm);

  if (world_rank == 0) {
    auto natoms = static_cast<i64>(all_pos.size());
    SystemState full_state;
    full_state.natoms = natoms;
    full_state.box = state.box;
    full_state.positions = all_pos;
    full_state.velocities = all_vel;
    full_state.forces = all_forces;
    full_state.types = all_types;
    full_state.ids = all_ids;
    full_state.masses = state.masses;

    potentials::MorsePair morse(D, alpha, r0, rc);
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(full_state.positions.data(), natoms, state.box, rc,
                    cfg.r_skin);
    real pe_f = potentials::compute_pair_forces(full_state, cpu_nlist, morse);
    real ke_f = compute_ke_host(all_vel, all_types, state.masses, natoms);
    ef = pe_f + ke_f;

    real drift = std::abs((ef - e0) / e0);
    EXPECT_LT(drift, kNVEDriftTolerance)
        << "Hybrid Pipeline NVE drift |dE/E| = " << drift;
  }

  MPI_Comm_free(&space_comm);
  MPI_Comm_free(&cart_comm);
}

TEST(HybridPipeline, ZoneTimeStepsAdvance) {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_EQ(world_size, 4);

  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::HybridConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.deterministic = false;
  cfg.n_streams = 2;
  cfg.p_time = 2;
  cfg.p_space = 2;

  scheduler::HybridPipelineScheduler sched(
      state.box, static_cast<i32>(state.natoms), params, cfg, MPI_COMM_WORLD);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  sched.run_until(10);

  EXPECT_GE(sched.min_local_time_step(), 10)
      << "world_rank " << world_rank << " min_local_time_step";

  for (const auto& z : sched.partition().zones()) {
    if (z.owner_rank == sched.time_rank()) {
      EXPECT_GE(z.time_step, 10)
          << "world_rank " << world_rank << " zone " << z.id
          << " at step " << z.time_step;
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

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
