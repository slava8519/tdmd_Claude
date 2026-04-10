// SPDX-License-Identifier: Apache-2.0
// test_device_lammps_ab.cu — GPU run-0 force match vs LAMMPS reference dumps.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "io/dump_reader.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "potentials/device_eam.cuh"
#include "potentials/device_morse.cuh"
#include "potentials/eam_alloy.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;
using namespace tdmd::testing;

// Compare GPU forces (indexed by state.ids) against LAMMPS DumpAtom (sorted by id).
static real max_force_diff(const std::vector<ForceVec>& gpu_forces,
                           const std::vector<i32>& gpu_ids,
                           const std::vector<io::DumpAtom>& ref) {
  real max_d = 0;
  for (std::size_t i = 0; i < gpu_ids.size(); ++i) {
    i32 id = gpu_ids[i];
    // ref is sorted by id, so ref[id-1] has matching atom.
    auto ri = static_cast<std::size_t>(id - 1);
    real dx = std::abs(gpu_forces[i].x - ref[ri].force.x);
    real dy = std::abs(gpu_forces[i].y - ref[ri].force.y);
    real dz = std::abs(gpu_forces[i].z - ref[ri].force.z);
    max_d = std::max(max_d, std::max({dx, dy, dz}));
  }
  return max_d;
}

TEST(DeviceLammpsAB, MorseRun0ForceMatch) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state =
      io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  real D = real{0.3429}, alpha = real{1.3588}, r0 = real{2.866}, rc = real{6.0};
  real skin = real{0.5};

  auto n = static_cast<std::size_t>(state.natoms);

  // GPU force compute.
  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<ForceVec> d_forces(n);
  d_pos.copy_from_host(state.positions.data(), n);
  d_forces.zero();

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};
  potentials::compute_morse_gpu(d_pos.data(), d_forces.data(),
                                gpu_nlist.d_neighbors(), gpu_nlist.d_offsets(),
                                gpu_nlist.d_counts(),
                                static_cast<i32>(state.natoms), state.box,
                                params, nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ForceVec> gpu_forces(n);
  d_forces.copy_to_host(gpu_forces.data(), n);

  // Read LAMMPS reference.
  auto ref = io::read_lammps_dump(data_dir + "/reference/forces_morse.dump");
  ASSERT_EQ(ref.size(), n);

  real diff = max_force_diff(gpu_forces, state.ids, ref);
  EXPECT_LT(diff, kForceTolerance)
      << "GPU Morse vs LAMMPS reference max force diff = " << diff;
}

TEST(DeviceLammpsAB, EamRun0ForceMatch) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state =
      io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  potentials::EamAlloy eam;
  eam.read_setfl(data_dir + "/Cu_mishin1.eam.alloy");
  real rc = eam.cutoff();
  real skin = real{0.5};

  auto n = static_cast<std::size_t>(state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<i32> d_types(n);
  d_pos.copy_from_host(state.positions.data(), n);
  d_forces.zero();
  d_types.copy_from_host(state.types.data(), n);

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  potentials::DeviceEam gpu_eam;
  gpu_eam.upload_tables(eam);
  gpu_eam.compute(d_pos.data(), d_forces.data(), d_types.data(),
                  gpu_nlist.d_neighbors(), gpu_nlist.d_offsets(),
                  gpu_nlist.d_counts(), static_cast<i32>(state.natoms),
                  state.box, nullptr);
  TDMD_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ForceVec> gpu_forces(n);
  d_forces.copy_to_host(gpu_forces.data(), n);

  auto ref = io::read_lammps_dump(data_dir + "/reference/forces_eam.dump");
  ASSERT_EQ(ref.size(), n);

  real diff = max_force_diff(gpu_forces, state.ids, ref);
  EXPECT_LT(diff, kForceTolerance)
      << "GPU EAM vs LAMMPS reference max force diff = " << diff;
}
