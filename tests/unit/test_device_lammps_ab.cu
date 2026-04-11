// SPDX-License-Identifier: Apache-2.0
// test_device_lammps_ab.cu — GPU run-0 force match vs LAMMPS reference dumps.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "io/dump_reader.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "neighbors/neighbor_list.hpp"
#include "core/box.hpp"
#include "core/math.hpp"
#include "potentials/device_eam.cuh"
#include "potentials/device_morse.cuh"
#include "potentials/eam_alloy.hpp"
#include "potentials/morse.hpp"
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

// Full per-component statistics for the Phase C A/B report.
// max_abs: L-infinity over all components.
// rms:     sqrt(mean(||ΔF||^2)) across all atoms (vector RMS).
// mean_abs_ref: average |ref force component| — useful to read max_abs
// as "relative to typical force scale", which for an FCC equilibrium
// state is ~0 and not meaningful, but at 300K thermalized it is.
struct ForceStats {
  double max_abs;
  double rms;
  double mean_abs_ref;
};

static ForceStats compute_force_stats(const std::vector<ForceVec>& gpu_forces,
                                       const std::vector<i32>& gpu_ids,
                                       const std::vector<io::DumpAtom>& ref) {
  ForceStats s{0.0, 0.0, 0.0};
  double sumsq = 0.0;
  double sumabs_ref = 0.0;
  const auto n = gpu_ids.size();
  for (std::size_t i = 0; i < n; ++i) {
    i32 id = gpu_ids[i];
    auto ri = static_cast<std::size_t>(id - 1);
    double dx = gpu_forces[i].x - ref[ri].force.x;
    double dy = gpu_forces[i].y - ref[ri].force.y;
    double dz = gpu_forces[i].z - ref[ri].force.z;
    double ax = std::abs(dx), ay = std::abs(dy), az = std::abs(dz);
    s.max_abs = std::max({s.max_abs, ax, ay, az});
    sumsq += dx * dx + dy * dy + dz * dz;
    sumabs_ref += std::abs(ref[ri].force.x) + std::abs(ref[ri].force.y) +
                  std::abs(ref[ri].force.z);
  }
  s.rms = std::sqrt(sumsq / static_cast<double>(n));  // RMS of 3-vector norm
  s.mean_abs_ref = sumabs_ref / static_cast<double>(3 * n);
  return s;
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
  ForceStats fs = compute_force_stats(gpu_forces, state.ids, ref);
  std::printf(
      "  MorseRun0ForceMatch: max_abs=%.3e rms=%.3e mean_abs_ref=%.3e\n",
      fs.max_abs, fs.rms, fs.mean_abs_ref);

  // Host-side Morse PE on the same positions — summed via the same
  // half-list that the EAM reference uses, so it maps 1:1 to LAMMPS
  // step-0 PE on this data file. Energy accumulated in double,
  // per-pair cast at the end (ADR 0007).
  //
  // The energy MorsePair::compute returns is written U(r) - U(inf),
  // where U(inf) = -D. That subtraction is LAMMPS's convention too —
  // so the sum here is directly comparable to LAMMPS's PotEng output.
  {
    neighbors::NeighborList cpu_nlist;
    cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc, skin);
    potentials::MorsePair morse(D, alpha, r0, rc);
    const Vec3D box_size = state.box.size();
    const auto& periodic = state.box.periodic;
    double pe = 0.0;
    for (i64 i = 0; i < state.natoms; ++i) {
      const i32 nc = cpu_nlist.count(i);
      const i32* nbrs = cpu_nlist.neighbors_of(i);
      const PositionVec pi = state.positions[static_cast<std::size_t>(i)];
      for (i32 k = 0; k < nc; ++k) {
        auto j = static_cast<std::size_t>(nbrs[k]);
        Vec3D delta = pi - state.positions[j];
        delta = minimum_image(delta, box_size, periodic);
        real r2 = static_cast<real>(length_sq(delta));
        real e = 0, fp = 0;
        morse.compute(r2, e, fp);
        pe += static_cast<double>(e);  // half-list already handles 1/2
      }
    }
    std::printf("  MorseRun0ForceMatch: PE_TDMD_CPU=%.6f eV\n", pe);
  }

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
  ForceStats fs = compute_force_stats(gpu_forces, state.ids, ref);
  std::printf(
      "  EamRun0ForceMatch: max_abs=%.3e rms=%.3e mean_abs_ref=%.3e\n",
      fs.max_abs, fs.rms, fs.mean_abs_ref);

  // Host-side PE on the same positions (CPU reference path). Comparison
  // against LAMMPS step-0 PE on the same data file lives in
  // docs/05-benchmarks/lammps-ab-results.md — the test just emits the
  // TDMD number; the reference number is scraped from the Phase C sweep.
  neighbors::NeighborList cpu_nlist;
  cpu_nlist.build(state.positions.data(), state.natoms, state.box, rc, skin);
  double pe_tdmd = static_cast<double>(eam.compute_forces(state, cpu_nlist));
  std::printf("  EamRun0ForceMatch: PE_TDMD_CPU=%.6f eV\n", pe_tdmd);

  EXPECT_LT(diff, kForceTolerance)
      << "GPU EAM vs LAMMPS reference max force diff = " << diff;
}
