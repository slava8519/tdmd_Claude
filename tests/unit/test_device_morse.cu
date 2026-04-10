// SPDX-License-Identifier: Apache-2.0
// test_device_morse.cu — GPU Morse kernel tests.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "support/precision_tolerance.hpp"
#include "core/device_buffer.cuh"
#include "core/device_system_state.cuh"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "neighbors/neighbor_list.hpp"
#include "potentials/device_morse.cuh"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"

using namespace tdmd;
using namespace tdmd::testing;

TEST(DeviceMorse, MatchesCPUForces256Atoms) {
  // Load 256-atom Cu FCC structure.
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Morse parameters (same as M1 tests).
  real D = real{0.3429};
  real alpha = real{1.3588};
  real r0 = real{2.866};
  real rc = real{6.0};
  real skin = real{0.5};

  // --- CPU forces ---
  potentials::MorsePair morse(D, alpha, r0, rc);
  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc, skin);
  real cpu_energy = potentials::compute_pair_forces(state, cpu_nlist, morse);

  // Save CPU forces.
  std::vector<ForceVec> cpu_forces = state.forces;

  // --- GPU forces ---
  auto n = static_cast<std::size_t>(state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<ForceVec> d_forces(n);
  d_pos.copy_from_host(state.positions.data(), n);
  d_forces.zero();

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  DeviceBuffer<accum_t> d_energy(1);
  d_energy.zero();

  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                gpu_nlist.d_neighbors(), gpu_nlist.d_offsets(),
                                gpu_nlist.d_counts(),
                                static_cast<i32>(state.natoms), state.box,
                                params, d_energy.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Download GPU forces and energy.
  std::vector<ForceVec> gpu_forces(n);
  d_forces.copy_to_host(gpu_forces.data(), n);

  accum_t gpu_energy = 0;
  d_energy.copy_to_host(&gpu_energy, 1);

  // Compare forces atom-by-atom. Tolerance: 1e-10 for FP64.
  real max_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real dfx = std::abs(gpu_forces[i].x - cpu_forces[i].x);
    real dfy = std::abs(gpu_forces[i].y - cpu_forces[i].y);
    real dfz = std::abs(gpu_forces[i].z - cpu_forces[i].z);
    real diff = std::max({dfx, dfy, dfz});
    max_diff = std::max(max_diff, diff);
  }

  EXPECT_LT(max_diff, kForceTolerance)
      << "Max force component difference between GPU and CPU Morse";

  // Compare energy. GPU accumulates in double, CPU in real.
  accum_t energy_diff = std::abs(gpu_energy - static_cast<accum_t>(cpu_energy));
  EXPECT_LT(energy_diff, kEnergyRelativeTolerance * std::abs(gpu_energy) + 1e-12)
      << "Energy difference: GPU=" << gpu_energy << " CPU=" << cpu_energy;
}
