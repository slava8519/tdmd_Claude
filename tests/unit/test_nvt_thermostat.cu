// SPDX-License-Identifier: Apache-2.0
// test_nvt_thermostat.cu — M7 NVT thermostat validation tests.
//
// Tests:
// 1. NHC thermostat drives ⟨T⟩ toward T_target (300 K).
// 2. device_compute_ke matches host KE calculation.
// 3. Deterministic NVT pipeline matches re-run (within FP tolerance).
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "integrator/device_nose_hoover.cuh"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"
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

static real compute_temperature(real ke, i32 natoms) {
  i32 n_dof = 3 * natoms - 3;
  return real{2} * ke / (static_cast<real>(n_dof) * kBoltzmann);
}

/// Initialize velocities from Maxwell-Boltzmann distribution at given T.
/// Removes center-of-mass momentum and rescales to exact target T.
static void init_velocities(SystemState& state, real t_target, unsigned seed) {
  auto n = static_cast<std::size_t>(state.natoms);
  std::mt19937 rng(seed);
  std::normal_distribution<real> gauss(0.0, 1.0);

  // v_i = sqrt(kB * T / (mass * kMvv2e)) * gaussian
  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
    real sigma = std::sqrt(kBoltzmann * t_target / (mass * kMvv2e));
    state.velocities[i].x = sigma * gauss(rng);
    state.velocities[i].y = sigma * gauss(rng);
    state.velocities[i].z = sigma * gauss(rng);
  }

  // Remove COM momentum.
  Vec3 com_v{0, 0, 0};
  real total_mass = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
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

  // Rescale to exact target temperature.
  real ke = compute_ke_host(state.velocities, state.types, state.masses,
                            state.natoms);
  real t_current = compute_temperature(ke, static_cast<i32>(state.natoms));
  if (t_current > 0) {
    real scale = std::sqrt(t_target / t_current);
    for (std::size_t i = 0; i < n; ++i) {
      state.velocities[i].x *= scale;
      state.velocities[i].y *= scale;
      state.velocities[i].z *= scale;
    }
  }
}

// Test: NHC thermostat drives ⟨T⟩ toward T_target.
// Initialize at 300K, run 5000 steps NVT, check ⟨T⟩ is within 15% of target
// after equilibration (small 256-atom system has large fluctuations).
TEST(NVTThermostat, TemperatureConvergesToTarget) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  real t_target = real{300.0};

  // Initialize velocities at target temperature.
  init_velocities(state, t_target, 42);

  scheduler::PipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = true;
  cfg.t_target = t_target;
  cfg.t_period = real{0.1};
  cfg.nhc_length = 3;

  auto natoms = static_cast<i32>(state.natoms);
  scheduler::PipelineScheduler sched(state.box, natoms, params, cfg);
  sched.upload(state.positions.data(), state.velocities.data(),
               state.forces.data(), state.types.data(), state.ids.data(),
               state.masses.data(), static_cast<i32>(state.masses.size()));

  // Equilibrate for 1000 steps.
  sched.run_until(1000);

  // Collect temperature over next 4000 steps, sampling every 100.
  std::vector<real> temps;
  for (i32 step = 1100; step <= 5000; step += 100) {
    sched.run_until(step);
    sched.download(state.positions.data(), state.velocities.data(),
                   state.forces.data(), state.types.data(), state.ids.data(),
                   natoms);

    real ke = compute_ke_host(state.velocities, state.types, state.masses,
                              state.natoms);
    real t = compute_temperature(ke, natoms);
    temps.push_back(t);
  }

  // Compute ⟨T⟩.
  real t_mean = std::accumulate(temps.begin(), temps.end(), real{0}) /
                static_cast<real>(temps.size());

  // ⟨T⟩ should be within 15% of target for a 256-atom system.
  real rel_err = std::abs(t_mean - t_target) / t_target;
  EXPECT_LT(rel_err, real{0.15})
      << "⟨T⟩ = " << t_mean << " K, target = " << t_target
      << " K, rel_err = " << rel_err;

  // Temperature should stay in a reasonable range.
  for (std::size_t i = 0; i < temps.size(); ++i) {
    EXPECT_GT(temps[i], real{100.0})
        << "Temperature too low at sample " << i << ": " << temps[i];
    EXPECT_LT(temps[i], real{600.0})
        << "Temperature too high at sample " << i << ": " << temps[i];
  }
}

// Test: device_compute_ke matches host computation.
TEST(NVTThermostat, DeviceKEMatchesHost) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Give atoms non-zero velocities for meaningful KE.
  init_velocities(state, real{300.0}, 123);

  auto natoms = static_cast<i32>(state.natoms);
  auto n = static_cast<std::size_t>(natoms);

  // Upload velocities, types, masses to device.
  DeviceBuffer<Vec3> d_vel(n);
  DeviceBuffer<i32> d_types(n);
  DeviceBuffer<real> d_masses(state.masses.size());

  d_vel.copy_from_host(state.velocities.data(), n);
  d_types.copy_from_host(state.types.data(), n);
  d_masses.copy_from_host(state.masses.data(), state.masses.size());

  real ke_device = integrator::device_compute_ke(d_vel.data(), d_types.data(),
                                                 d_masses.data(), natoms);
  real ke_host = compute_ke_host(state.velocities, state.types, state.masses,
                                 state.natoms);

  // Should match to within floating-point tolerance.
  EXPECT_GT(ke_host, real{0}) << "KE should be non-zero with initialized velocities";
  real rel_diff = std::abs(ke_device - ke_host) / ke_host;
  EXPECT_LT(rel_diff, real{1e-10})
      << "KE device=" << ke_device << " host=" << ke_host;
}

// Test: Deterministic NVT produces identical results on re-run (within FP
// tolerance from GPU reduction order).
TEST(NVTThermostat, DeterministicReproducibility) {
  std::string data_dir = TDMD_TEST_DATA_DIR;

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  scheduler::PipelineConfig cfg;
  cfg.dt = real{0.001};
  cfg.r_skin = real{1.0};
  cfg.rebuild_every = 10;
  cfg.deterministic = true;
  cfg.t_target = real{300.0};
  cfg.t_period = real{0.1};
  cfg.nhc_length = 3;

  auto run_nvt = [&](i32 nsteps) -> std::vector<Vec3> {
    SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");
    init_velocities(state, real{300.0}, 42);  // same seed both runs
    auto natoms = static_cast<i32>(state.natoms);

    scheduler::PipelineScheduler sched(state.box, natoms, params, cfg);
    sched.upload(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 state.masses.data(), static_cast<i32>(state.masses.size()));
    sched.run_until(nsteps);
    sched.download(state.positions.data(), state.velocities.data(),
                   state.forces.data(), state.types.data(), state.ids.data(),
                   natoms);
    return state.positions;
  };

  auto pos1 = run_nvt(500);
  auto pos2 = run_nvt(500);

  ASSERT_EQ(pos1.size(), pos2.size());
  real max_diff = 0;
  for (std::size_t i = 0; i < pos1.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(pos1[i].x - pos2[i].x));
    max_diff = std::max(max_diff, std::abs(pos1[i].y - pos2[i].y));
    max_diff = std::max(max_diff, std::abs(pos1[i].z - pos2[i].z));
  }
  // Allow small FP differences from GPU KE reduction (different block sums
  // may accumulate in slightly different order across runs).
  EXPECT_LT(max_diff, real{1e-10})
      << "Deterministic NVT max position diff: " << max_diff;
}
