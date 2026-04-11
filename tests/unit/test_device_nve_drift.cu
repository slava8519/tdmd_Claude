// SPDX-License-Identifier: Apache-2.0
// test_device_nve_drift.cu — GPU NVE energy conservation test.
//
// Runs 50k steps of Morse NVE on GPU with 256 Cu atoms.
// Checks |dE/E| < 1e-4 (same criterion as CPU M1 test).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "integrator/device_velocity_verlet.cuh"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "potentials/device_morse.cuh"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

static double compute_kinetic_energy_host(
    const std::vector<VelocityVec>& velocities,
    const std::vector<i32>& types,
    const std::vector<real>& masses,
    i64 natoms) {
  double ke = 0;
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    double mass = static_cast<double>(masses[static_cast<std::size_t>(types[si])]);
    const VelocityVec& v = velocities[si];
    ke += 0.5 * mass * static_cast<double>(kMvv2e) *
          (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  return ke;
}

TEST(DeviceNVEDrift, Morse256Atoms50kSteps) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state =
      io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Morse parameters.
  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};  // 1 fs
  i32 nsteps = 50000;
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  auto n = static_cast<std::size_t>(state.natoms);
  i32 ni = static_cast<i32>(state.natoms);

  // Upload positions, velocities, types, masses to device.
  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<VelocityVec> d_vel(n);
  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<i32> d_types(n);
  DeviceBuffer<real> d_masses(state.masses.size());

  d_pos.copy_from_host(state.positions.data(), n);
  d_vel.copy_from_host(state.velocities.data(), n);
  d_types.copy_from_host(state.types.data(), n);
  d_masses.copy_from_host(state.masses.data(), state.masses.size());

  // Build initial neighbor list.
  neighbors::DeviceNeighborList nlist;
  nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  // Initial force compute.
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Compute initial PE on device.
  DeviceBuffer<accum_t> d_pe(1);
  d_pe.zero();
  // Re-zero forces and compute with energy.
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                d_pe.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  accum_t pe0 = 0;
  d_pe.copy_to_host(&pe0, 1);

  // Download velocities for KE.
  std::vector<VelocityVec> h_vel(n);
  d_vel.copy_to_host(h_vel.data(), n);
  double ke0 = compute_kinetic_energy_host(h_vel, state.types, state.masses,
                                           state.natoms);
  double e0 = static_cast<double>(pe0) + ke0;

  // Rebuild interval for neighbor list.
  i32 rebuild_every = 10;

  // MD loop on GPU.
  for (i32 step = 0; step < nsteps; ++step) {
    // Half-kick.
    integrator::device_half_kick(d_vel.data(), d_forces.data(), d_types.data(),
                                 d_masses.data(), ni, dt);

    // Drift.
    integrator::device_drift(d_pos.data(), d_vel.data(), ni, dt, state.box);

    // Rebuild neighbor list periodically.
    if ((step + 1) % rebuild_every == 0) {
      nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);
    }

    // Force compute.
    d_forces.zero();
    potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                  nlist.d_neighbors(), nlist.d_offsets(),
                                  nlist.d_counts(), ni, state.box, params,
                                  nullptr);

    // Half-kick.
    integrator::device_half_kick(d_vel.data(), d_forces.data(), d_types.data(),
                                 d_masses.data(), ni, dt);
  }

  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Compute final energy.
  d_pe.zero();
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                d_pe.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  accum_t pe_final = 0;
  d_pe.copy_to_host(&pe_final, 1);

  d_vel.copy_to_host(h_vel.data(), n);
  double ke_final = compute_kinetic_energy_host(h_vel, state.types,
                                                state.masses, state.natoms);
  double e_final = static_cast<double>(pe_final) + ke_final;

  double drift = std::abs((e_final - e0) / e0);
  // M2 criterion: |dE/E| < 1e-4.
  EXPECT_LT(drift, 1e-4)
      << "NVE drift |dE/E| = " << drift << ", E0=" << e0
      << ", Ef=" << e_final;
}

// ADR 0007 acceptance test: long-run NVE drift per step.
// 100k steps with raw GPU kernel loop. Measures per-step drift against ADR
// targets. This is the gate for mixed precision correctness.
TEST(DeviceNVEDrift, ADR0007AcceptanceTest100k) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  constexpr i32 nsteps = 100000;
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  auto n = static_cast<std::size_t>(state.natoms);
  i32 ni = static_cast<i32>(state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<VelocityVec> d_vel(n);
  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<i32> d_types(n);
  DeviceBuffer<real> d_masses(state.masses.size());

  d_pos.copy_from_host(state.positions.data(), n);
  d_vel.copy_from_host(state.velocities.data(), n);
  d_types.copy_from_host(state.types.data(), n);
  d_masses.copy_from_host(state.masses.data(), state.masses.size());

  neighbors::DeviceNeighborList nlist;
  nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  // Initial force compute with energy.
  DeviceBuffer<accum_t> d_pe(1);
  d_pe.zero();
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                d_pe.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  accum_t pe0 = 0;
  d_pe.copy_to_host(&pe0, 1);

  std::vector<VelocityVec> h_vel(n);
  d_vel.copy_to_host(h_vel.data(), n);
  double ke0 = compute_kinetic_energy_host(h_vel, state.types, state.masses,
                                           state.natoms);
  accum_t e0 = static_cast<accum_t>(pe0) + static_cast<accum_t>(ke0);

  constexpr i32 rebuild_every = 10;

  for (i32 step = 0; step < nsteps; ++step) {
    integrator::device_half_kick(d_vel.data(), d_forces.data(), d_types.data(),
                                 d_masses.data(), ni, dt);
    integrator::device_drift(d_pos.data(), d_vel.data(), ni, dt, state.box);

    if ((step + 1) % rebuild_every == 0) {
      nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);
    }

    d_forces.zero();
    potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                  nlist.d_neighbors(), nlist.d_offsets(),
                                  nlist.d_counts(), ni, state.box, params,
                                  nullptr);
    integrator::device_half_kick(d_vel.data(), d_forces.data(), d_types.data(),
                                 d_masses.data(), ni, dt);
  }
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Final energy.
  d_pe.zero();
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                d_pe.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  accum_t pe_final = 0;
  d_pe.copy_to_host(&pe_final, 1);

  d_vel.copy_to_host(h_vel.data(), n);
  double ke_final = compute_kinetic_energy_host(h_vel, state.types,
                                                state.masses, state.natoms);
  accum_t ef = static_cast<accum_t>(pe_final) + static_cast<accum_t>(ke_final);

  accum_t total_drift = std::abs((ef - e0) / e0);
  accum_t per_step = total_drift / static_cast<accum_t>(nsteps);

  std::printf("  ADR 0007 acceptance (%dk steps):\n", nsteps / 1000);
  std::printf("    E0=%.10f  Ef=%.10f\n", e0, ef);
  std::printf("    |dE/E| total = %.3e\n", total_drift);
  std::printf("    |dE/E|/step  = %.3e\n", per_step);
#ifdef TDMD_PRECISION_FP64
  std::printf("    ADR target   = %.0e (fp64)\n", kEnergyDriftPerStepFP64);
  EXPECT_LT(per_step, kEnergyDriftPerStepFP64)
      << "Per-step drift " << per_step << " exceeds ADR 0007 FP64 target "
      << kEnergyDriftPerStepFP64;
#else
  std::printf("    ADR target   = %.0e (mixed)\n", kEnergyDriftPerStepMixed);
  EXPECT_LT(per_step, kEnergyDriftPerStepMixed)
      << "Per-step drift " << per_step << " exceeds ADR 0007 mixed target "
      << kEnergyDriftPerStepMixed;
#endif
}

// OPT-FUSE-1a bit-equivalence gate. The fused kick+drift kernel must produce
// the exact same velocity and position bytes as the unfused pair — their math
// is identical op-for-op. Any divergence means the fusion broke the math
// contract (ADR 0007). Tested on a realistic force field, not zero forces,
// so PBC wrap + nontrivial velocity scale are both exercised.
TEST(DeviceVelocityVerlet, FusedKickDriftMatchesUnfused) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{1.0};
  real dt = real{0.001};
  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  auto n = static_cast<std::size_t>(state.natoms);
  i32 ni = static_cast<i32>(state.natoms);

  // Shared inputs: real forces from a run-0 Morse compute, not zero, so
  // the half-kick meaningfully changes velocity.
  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<VelocityVec> d_vel(n);
  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<i32> d_types(n);
  DeviceBuffer<real> d_masses(state.masses.size());
  d_pos.copy_from_host(state.positions.data(), n);
  d_vel.copy_from_host(state.velocities.data(), n);
  d_types.copy_from_host(state.types.data(), n);
  d_masses.copy_from_host(state.masses.data(), state.masses.size());
  neighbors::DeviceNeighborList nlist;
  nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);
  d_forces.zero();
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                nlist.d_neighbors(), nlist.d_offsets(),
                                nlist.d_counts(), ni, state.box, params,
                                nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Snapshot pos/vel/forces as ground truth starting point.
  std::vector<PositionVec> pos0(n);
  std::vector<VelocityVec> vel0(n);
  std::vector<ForceVec> f0(n);
  d_pos.copy_to_host(pos0.data(), n);
  d_vel.copy_to_host(vel0.data(), n);
  d_forces.copy_to_host(f0.data(), n);

  // Path A: unfused (half_kick then drift).
  DeviceBuffer<PositionVec> d_pos_a(n);
  DeviceBuffer<VelocityVec> d_vel_a(n);
  DeviceBuffer<ForceVec> d_f_a(n);
  d_pos_a.copy_from_host(pos0.data(), n);
  d_vel_a.copy_from_host(vel0.data(), n);
  d_f_a.copy_from_host(f0.data(), n);
  integrator::device_half_kick(d_vel_a.data(), d_f_a.data(), d_types.data(),
                               d_masses.data(), ni, dt);
  integrator::device_drift(d_pos_a.data(), d_vel_a.data(), ni, dt, state.box);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<PositionVec> pos_a(n);
  std::vector<VelocityVec> vel_a(n);
  d_pos_a.copy_to_host(pos_a.data(), n);
  d_vel_a.copy_to_host(vel_a.data(), n);

  // Path B: fused.
  DeviceBuffer<PositionVec> d_pos_b(n);
  DeviceBuffer<VelocityVec> d_vel_b(n);
  DeviceBuffer<ForceVec> d_f_b(n);
  d_pos_b.copy_from_host(pos0.data(), n);
  d_vel_b.copy_from_host(vel0.data(), n);
  d_f_b.copy_from_host(f0.data(), n);
  integrator::device_fused_half_kick_drift(d_vel_b.data(), d_pos_b.data(),
                                           d_f_b.data(), d_types.data(),
                                           d_masses.data(), ni, dt, state.box);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<PositionVec> pos_b(n);
  std::vector<VelocityVec> vel_b(n);
  d_pos_b.copy_to_host(pos_b.data(), n);
  d_vel_b.copy_to_host(vel_b.data(), n);

  // Bit-exact match required: both paths compute the same double-precision
  // expression tree in the same order. Any difference → math contract broken.
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(vel_a[i].x, vel_b[i].x) << "vx mismatch at atom " << i;
    ASSERT_EQ(vel_a[i].y, vel_b[i].y) << "vy mismatch at atom " << i;
    ASSERT_EQ(vel_a[i].z, vel_b[i].z) << "vz mismatch at atom " << i;
    ASSERT_EQ(pos_a[i].x, pos_b[i].x) << "px mismatch at atom " << i;
    ASSERT_EQ(pos_a[i].y, pos_b[i].y) << "py mismatch at atom " << i;
    ASSERT_EQ(pos_a[i].z, pos_b[i].z) << "pz mismatch at atom " << i;
  }
}
