// SPDX-License-Identifier: Apache-2.0
// test_device_eam.cu — GPU EAM kernel tests.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/device_buffer.cuh"
#include "core/device_system_state.cuh"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "neighbors/neighbor_list.hpp"
#include "potentials/device_eam.cuh"
#include "potentials/eam_alloy.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

TEST(DeviceEam, MatchesCPUForces256Atoms) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state =
      io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Load EAM potential.
  potentials::EamAlloy eam;
  eam.read_setfl(data_dir + "/Cu_mishin1.eam.alloy");

  real rc = eam.cutoff();
  real skin = real{0.5};

  // --- CPU forces ---
  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc, skin);
  real cpu_energy = eam.compute_forces(state, cpu_nlist);
  std::vector<Vec3> cpu_forces = state.forces;

  // --- GPU forces ---
  auto n = static_cast<std::size_t>(state.natoms);

  DeviceBuffer<Vec3> d_pos(n), d_forces(n);
  DeviceBuffer<i32> d_types(n);
  d_pos.copy_from_host(state.positions.data(), n);
  d_forces.zero();
  d_types.copy_from_host(state.types.data(), n);

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  potentials::DeviceEam gpu_eam;
  gpu_eam.upload_tables(eam);

  DeviceBuffer<accum_t> d_energy(1);
  d_energy.zero();

  gpu_eam.compute(d_pos.data(), d_forces.data(), d_types.data(),
                  gpu_nlist.d_neighbors(), gpu_nlist.d_offsets(),
                  gpu_nlist.d_counts(), static_cast<i32>(state.natoms),
                  state.box, d_energy.data());
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  // Download results.
  std::vector<Vec3> gpu_forces(n);
  d_forces.copy_to_host(gpu_forces.data(), n);

  accum_t gpu_energy = 0;
  d_energy.copy_to_host(&gpu_energy, 1);

  // Compare forces. EAM has more numerical noise than Morse, but should
  // still match to ~1e-8 in FP64.
  real max_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real dfx = std::abs(gpu_forces[i].x - cpu_forces[i].x);
    real dfy = std::abs(gpu_forces[i].y - cpu_forces[i].y);
    real dfz = std::abs(gpu_forces[i].z - cpu_forces[i].z);
    real diff = std::max({dfx, dfy, dfz});
    max_diff = std::max(max_diff, diff);
  }

  EXPECT_LT(max_diff, kForceTolerance)
      << "Max force component difference between GPU and CPU EAM";

  // Compare energy.
  real energy_diff = std::abs(gpu_energy - cpu_energy);
  EXPECT_LT(energy_diff, kEnergyRelativeTolerance * std::abs(gpu_energy) + 1e-12)
      << "Energy difference: GPU=" << gpu_energy << " CPU=" << cpu_energy;
}
