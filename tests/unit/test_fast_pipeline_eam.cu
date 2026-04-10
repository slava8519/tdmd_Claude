// SPDX-License-Identifier: Apache-2.0
// test_fast_pipeline_eam.cu — FastPipelineScheduler driven by EAM.
//
// FEAT-EAM-Production-Pipeline Phase A. Pins the scheduler's EAM path
// against three properties:
//
//   1. EamStepMatchesDirectCompute — at t=0 (initial force compute), the
//      force array the scheduler produces must match a direct
//      DeviceEam::compute() invocation on the same positions byte-for-byte
//      (same nlist skin, same tables). Guards against the scheduler
//      silently using a different cutoff, nlist, or type array than the
//      direct path.
//
//   2. EamNve100StepsStable — 100 velocity-Verlet steps with dt=0.001 on
//      the Cu FCC 256-atom reference, then compare total energy against
//      the host-side CPU EAM reference. |dE/E| must stay below 1e-3 in
//      mixed mode — EAM accumulates more noise than Morse over the
//      embedding pass, so this bound is a floor for "scheduler wiring
//      did not break the physics", not a precision claim.
//
//   3. EamDeterministicReplay — same initial condition, same seed, two
//      independent scheduler instances: final positions must match to
//      within the force-tolerance noise floor. Does not rely on
//      TDMD_DETERMINISTIC_REDUCE; this is step-level replay, not
//      bit-identical energy reduction.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "neighbors/neighbor_list.hpp"
#include "potentials/device_eam.cuh"
#include "potentials/eam_alloy.hpp"
#include "scheduler/fast_pipeline_scheduler.cuh"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

namespace {

struct EamFixture {
  SystemState state;
  potentials::EamAlloy eam;
};

EamFixture load_cu_fcc_eam() {
  EamFixture s;
  std::string data_dir = TDMD_TEST_DATA_DIR;
  s.state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
  s.eam.read_setfl(data_dir + "/Cu_mishin1.eam.alloy");
  return s;
}

double compute_ke_host(const std::vector<VelocityVec>& velocities,
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

}  // namespace

TEST(FastPipelineSchedulerEam, EamStepMatchesDirectCompute) {
  EamFixture s = load_cu_fcc_eam();

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{0.5};
  cfg.rebuild_every = 10;

  // --- Path A: forces via the scheduler at t=0 (initial compute only). ---
  scheduler::FastPipelineScheduler sched(s.state.box,
                                         static_cast<i32>(s.state.natoms),
                                         s.eam, cfg);
  EXPECT_EQ(sched.potential_kind(), scheduler::PotentialKind::Eam);
  EXPECT_NEAR(static_cast<double>(sched.interaction_cutoff()),
              static_cast<double>(s.eam.cutoff()), 0.0);

  sched.upload(s.state.positions.data(), s.state.velocities.data(),
               s.state.forces.data(), s.state.types.data(),
               s.state.ids.data(), s.state.masses.data(),
               static_cast<i32>(s.state.masses.size()));
  sched.run_until(0);  // computes initial forces, no step taken

  std::vector<ForceVec> sched_forces(
      static_cast<std::size_t>(s.state.natoms));
  std::vector<PositionVec> sched_pos(
      static_cast<std::size_t>(s.state.natoms));
  {
    std::vector<VelocityVec> tmp_v(static_cast<std::size_t>(s.state.natoms));
    std::vector<i32> tmp_t(static_cast<std::size_t>(s.state.natoms));
    std::vector<i32> tmp_i(static_cast<std::size_t>(s.state.natoms));
    sched.download(sched_pos.data(), tmp_v.data(), sched_forces.data(),
                   tmp_t.data(), tmp_i.data(),
                   static_cast<i32>(s.state.natoms));
  }

  // --- Path B: direct DeviceEam::compute on the same positions, same
  // nlist skin. This is exactly how tests/unit/test_device_eam.cu drives
  // EAM, so agreement between Path A and Path B proves the scheduler
  // added no hidden transformation between its inputs and DeviceEam. ---
  real rc = s.eam.cutoff();
  real skin = cfg.r_skin;
  auto n = static_cast<std::size_t>(s.state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<i32> d_types(n);
  d_pos.copy_from_host(s.state.positions.data(), n);
  d_forces.zero();
  d_types.copy_from_host(s.state.types.data(), n);

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), s.state.natoms, s.state.box, rc, skin);

  potentials::DeviceEam gpu_eam;
  gpu_eam.upload_tables(s.eam);

  gpu_eam.compute(d_pos.data(), d_forces.data(), d_types.data(),
                  gpu_nlist.d_neighbors(), gpu_nlist.d_offsets(),
                  gpu_nlist.d_counts(), static_cast<i32>(s.state.natoms),
                  s.state.box, nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ForceVec> direct_forces(n);
  d_forces.copy_to_host(direct_forces.data(), n);

  // Compare. Identical nlist builder + identical EAM tables + identical
  // positions should give bit-identical forces in principle, but two
  // distinct DeviceNeighborList instances can produce slightly different
  // neighbor orderings if internal hashing differs, so we allow the
  // force-tolerance noise floor.
  real max_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real dfx = std::abs(sched_forces[i].x - direct_forces[i].x);
    real dfy = std::abs(sched_forces[i].y - direct_forces[i].y);
    real dfz = std::abs(sched_forces[i].z - direct_forces[i].z);
    max_diff = std::max(max_diff, std::max({dfx, dfy, dfz}));
  }
  EXPECT_LT(max_diff, kForceTolerance)
      << "Scheduler-EAM forces at t=0 do not match direct DeviceEam::compute"
      << " (max |Δf| = " << max_diff << ")";
}

TEST(FastPipelineSchedulerEam, EamNve100StepsStable) {
  EamFixture s = load_cu_fcc_eam();

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{0.5};
  cfg.rebuild_every = 10;

  scheduler::FastPipelineScheduler sched(s.state.box,
                                         static_cast<i32>(s.state.natoms),
                                         s.eam, cfg);
  sched.upload(s.state.positions.data(), s.state.velocities.data(),
               s.state.forces.data(), s.state.types.data(),
               s.state.ids.data(), s.state.masses.data(),
               static_cast<i32>(s.state.masses.size()));

  // Step 1: compute initial forces, take the first Verlet step, measure E0.
  sched.run_until(1);
  sched.download(s.state.positions.data(), s.state.velocities.data(),
                 s.state.forces.data(), s.state.types.data(),
                 s.state.ids.data(), static_cast<i32>(s.state.natoms));

  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(s.state.positions.data(), s.state.natoms, s.state.box,
                  s.eam.cutoff(), cfg.r_skin);
  double pe0 = static_cast<double>(s.eam.compute_forces(s.state, cpu_nlist));
  double ke0 = compute_ke_host(s.state.velocities, s.state.types,
                               s.state.masses, s.state.natoms);
  double e0 = pe0 + ke0;

  // Step 100.
  sched.run_until(100);
  sched.download(s.state.positions.data(), s.state.velocities.data(),
                 s.state.forces.data(), s.state.types.data(),
                 s.state.ids.data(), static_cast<i32>(s.state.natoms));

  cpu_nlist.build(s.state.positions.data(), s.state.natoms, s.state.box,
                  s.eam.cutoff(), cfg.r_skin);
  double pe_f = static_cast<double>(s.eam.compute_forces(s.state, cpu_nlist));
  double ke_f = compute_ke_host(s.state.velocities, s.state.types,
                                s.state.masses, s.state.natoms);
  double ef = pe_f + ke_f;

  double drift = std::abs((ef - e0) / e0);
  std::printf("  EamNve100StepsStable: E0=%.6f Ef=%.6f |dE/E|=%.2e\n", e0, ef,
              drift);
  EXPECT_LT(drift, 1e-3)
      << "EAM FastPipeline NVE drift |dE/E| = " << drift
      << " (E0=" << e0 << ", Ef=" << ef << ")";
}

TEST(FastPipelineSchedulerEam, EamDeterministicReplay) {
  // Two independent scheduler instances, same input, same config, same
  // number of steps. Final positions must agree to the force-tolerance
  // floor. This is scheduler-level replay (same nlist schedule, same
  // kernel launch order) — it does not require TDMD_DETERMINISTIC_REDUCE.
  EamFixture s1 = load_cu_fcc_eam();
  EamFixture s2 = load_cu_fcc_eam();

  scheduler::FastPipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{0.5};
  cfg.rebuild_every = 10;

  auto run = [&](EamFixture& s) {
    scheduler::FastPipelineScheduler sched(s.state.box,
                                           static_cast<i32>(s.state.natoms),
                                           s.eam, cfg);
    sched.upload(s.state.positions.data(), s.state.velocities.data(),
                 s.state.forces.data(), s.state.types.data(),
                 s.state.ids.data(), s.state.masses.data(),
                 static_cast<i32>(s.state.masses.size()));
    sched.run_until(50);
    sched.download(s.state.positions.data(), s.state.velocities.data(),
                   s.state.forces.data(), s.state.types.data(),
                   s.state.ids.data(), static_cast<i32>(s.state.natoms));
  };

  run(s1);
  run(s2);

  auto n = static_cast<std::size_t>(s1.state.natoms);
  real max_pos_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    auto dx = static_cast<real>(std::abs(s1.state.positions[i].x -
                                         s2.state.positions[i].x));
    auto dy = static_cast<real>(std::abs(s1.state.positions[i].y -
                                         s2.state.positions[i].y));
    auto dz = static_cast<real>(std::abs(s1.state.positions[i].z -
                                         s2.state.positions[i].z));
    max_pos_diff = std::max(max_pos_diff, std::max({dx, dy, dz}));
  }
  EXPECT_LT(max_pos_diff, kForceTolerance)
      << "EAM scheduler replay: max |Δx| = " << max_pos_diff
      << " — scheduler is not step-level deterministic on identical input";
}
