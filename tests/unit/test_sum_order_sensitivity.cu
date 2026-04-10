// SPDX-License-Identifier: Apache-2.0
// test_sum_order_sensitivity.cu — VL-13 (research doc N2).
//
// Permute each atom's neighbor window and re-run the force kernel. The total
// energy must stay within a precision-dependent tolerance relative to the
// baseline run. This catches regressions where a kernel change would make the
// per-thread accumulation catastrophically order-sensitive (e.g. dropping the
// double accumulator in an EAM reduction — the canonical bug from ADR 0007).
//
// Note on scope: this is a *bounded-variance* sensitivity test, not a
// "variance = 0" test. Even in TDMD_DETERMINISTIC_REDUCE=ON mode the intra-
// thread `pe += ...` accumulation is still performed in neighbor-list order
// (and in force_t = float in mixed mode), so permuting the neighbor window
// perturbs the per-thread partial at the ULP level. ADR 0010 only guarantees
// bit-reproducibility on *identical* input — see `DeviceMorseDeterminism` /
// `DeviceEamDeterminism` for that property. VL-13's job is orthogonal: the
// kernel must not amplify neighbor-order perturbations beyond the float noise
// floor.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "core/device_buffer.cuh"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/device_neighbor_list.cuh"
#include "potentials/device_eam.cuh"
#include "potentials/device_morse.cuh"
#include "potentials/eam_alloy.hpp"
#include "support/precision_tolerance.hpp"

using namespace tdmd;

namespace {

// Precision-aware upper bound on fractional energy change caused by permuting
// the neighbor list order. Derived empirically: in mixed mode the intra-thread
// `pe` is a float, so permuting ~28 neighbors around a per-atom magnitude of
// ~3 eV gives ULP(3) * sqrt(28) ≈ 2e-6 per atom; summed into the double
// accumulator across 256 atoms and then divided by |E_total| ≈ 770 eV this
// lands well below 1e-6. In fp64 the same argument gives ~1e-13.
#ifdef TDMD_PRECISION_FP64
constexpr double kSumOrderRelativeBound = 1e-12;
#else
constexpr double kSumOrderRelativeBound = 1e-5;
#endif

// Download the CSR neighbor arrays (counts, offsets, neighbor indices) from
// the GPU-built list into host vectors we can permute.
struct HostCsr {
  std::vector<i32> counts;
  std::vector<i32> offsets;
  std::vector<i32> neighbors;  // flat, length == sum(counts)
};

HostCsr download_csr(const neighbors::DeviceNeighborList& nlist, i64 natoms) {
  auto un = static_cast<std::size_t>(natoms);
  HostCsr h;
  h.counts.resize(un);
  h.offsets.resize(un);
  cudaMemcpy(h.counts.data(), nlist.d_counts(), un * sizeof(i32),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h.offsets.data(), nlist.d_offsets(), un * sizeof(i32),
             cudaMemcpyDeviceToHost);
  i32 total = h.offsets.back() + h.counts.back();
  h.neighbors.resize(static_cast<std::size_t>(total));
  cudaMemcpy(h.neighbors.data(), nlist.d_neighbors(),
             static_cast<std::size_t>(total) * sizeof(i32),
             cudaMemcpyDeviceToHost);
  return h;
}

// Shuffle each atom's neighbor window [offset, offset+count) in place using a
// deterministic RNG. Per-atom windows are disjoint so there's no aliasing.
void permute_windows(HostCsr& h, std::uint32_t seed) {
  std::mt19937 rng(seed);
  for (std::size_t i = 0; i < h.counts.size(); ++i) {
    i32 off = h.offsets[i];
    i32 cnt = h.counts[i];
    if (cnt > 1) {
      std::shuffle(h.neighbors.begin() + off,
                   h.neighbors.begin() + off + cnt, rng);
    }
  }
}

}  // namespace

TEST(SumOrderSensitivity, MorseEnergyBounded) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  // Morse parameters (mirror test_device_morse.cu).
  real D = real{0.3429};
  real alpha = real{1.3588};
  real r0 = real{2.866};
  real rc = real{6.0};
  real skin = real{0.5};

  auto n = static_cast<std::size_t>(state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  d_pos.copy_from_host(state.positions.data(), n);

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  HostCsr h = download_csr(gpu_nlist, state.natoms);

  // Independent device buffers for the permuted list. Offsets/counts are
  // invariant under permutation inside each atom's window, so we reuse the
  // originals and only re-upload `neighbors`.
  DeviceBuffer<i32> d_neighbors_perm(h.neighbors.size());
  DeviceBuffer<i32> d_offsets_perm(n);
  DeviceBuffer<i32> d_counts_perm(n);
  d_offsets_perm.copy_from_host(h.offsets.data(), n);
  d_counts_perm.copy_from_host(h.counts.data(), n);

  potentials::MorseParams params{D, alpha, r0, rc, rc * rc};

  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<accum_t> d_energy(1);

  auto run = [&](const i32* d_nbrs) {
    d_forces.zero();
    d_energy.zero();
    potentials::compute_morse_gpu(d_pos.data(), d_forces.data(), d_nbrs,
                                  d_offsets_perm.data(), d_counts_perm.data(),
                                  static_cast<i32>(state.natoms), state.box,
                                  params, d_energy.data());
    TDMD_CUDA_CHECK(cudaDeviceSynchronize());
    accum_t e = 0;
    d_energy.copy_to_host(&e, 1);
    return e;
  };

  // Baseline: original (GPU-built) neighbor list order.
  d_neighbors_perm.copy_from_host(h.neighbors.data(), h.neighbors.size());
  accum_t e_base = run(d_neighbors_perm.data());
  ASSERT_GT(std::abs(e_base), 0.0) << "baseline energy should be non-zero";

  // Three permutations with distinct seeds; the worst deviation must still
  // stay under the bound.
  double worst_rel = 0.0;
  for (std::uint32_t seed : {1u, 7u, 42u}) {
    HostCsr h_p = h;
    permute_windows(h_p, seed);
    d_neighbors_perm.copy_from_host(h_p.neighbors.data(), h_p.neighbors.size());
    accum_t e_p = run(d_neighbors_perm.data());
    double rel = std::abs(static_cast<double>(e_p - e_base)) /
                 std::abs(static_cast<double>(e_base));
    worst_rel = std::max(worst_rel, rel);
  }

  EXPECT_LT(worst_rel, kSumOrderRelativeBound)
      << "Morse energy variance under neighbor permutation exceeds bound. "
      << "worst_rel=" << worst_rel
      << " bound=" << kSumOrderRelativeBound
      << " E_base=" << e_base;
}

TEST(SumOrderSensitivity, EamEnergyBounded) {
  std::string data_dir = TDMD_TEST_DATA_DIR;
  SystemState state = io::read_lammps_data(data_dir + "/cu_fcc_256.data");

  potentials::EamAlloy eam;
  eam.read_setfl(data_dir + "/Cu_mishin1.eam.alloy");
  real rc = eam.cutoff();
  real skin = real{0.5};

  auto n = static_cast<std::size_t>(state.natoms);

  DeviceBuffer<PositionVec> d_pos(n);
  DeviceBuffer<i32> d_types(n);
  d_pos.copy_from_host(state.positions.data(), n);
  d_types.copy_from_host(state.types.data(), n);

  neighbors::DeviceNeighborList gpu_nlist;
  gpu_nlist.build(d_pos.data(), state.natoms, state.box, rc, skin);

  HostCsr h = download_csr(gpu_nlist, state.natoms);

  DeviceBuffer<i32> d_neighbors_perm(h.neighbors.size());
  DeviceBuffer<i32> d_offsets_perm(n);
  DeviceBuffer<i32> d_counts_perm(n);
  d_offsets_perm.copy_from_host(h.offsets.data(), n);
  d_counts_perm.copy_from_host(h.counts.data(), n);

  potentials::DeviceEam gpu_eam;
  gpu_eam.upload_tables(eam);

  DeviceBuffer<ForceVec> d_forces(n);
  DeviceBuffer<accum_t> d_energy(1);

  auto run = [&](const i32* d_nbrs) {
    d_forces.zero();
    d_energy.zero();
    gpu_eam.compute(d_pos.data(), d_forces.data(), d_types.data(), d_nbrs,
                    d_offsets_perm.data(), d_counts_perm.data(),
                    static_cast<i32>(state.natoms), state.box,
                    d_energy.data());
    TDMD_CUDA_CHECK(cudaDeviceSynchronize());
    accum_t e = 0;
    d_energy.copy_to_host(&e, 1);
    return e;
  };

  d_neighbors_perm.copy_from_host(h.neighbors.data(), h.neighbors.size());
  accum_t e_base = run(d_neighbors_perm.data());
  ASSERT_GT(std::abs(e_base), 0.0) << "baseline energy should be non-zero";

  double worst_rel = 0.0;
  for (std::uint32_t seed : {1u, 7u, 42u}) {
    HostCsr h_p = h;
    permute_windows(h_p, seed);
    d_neighbors_perm.copy_from_host(h_p.neighbors.data(), h_p.neighbors.size());
    accum_t e_p = run(d_neighbors_perm.data());
    double rel = std::abs(static_cast<double>(e_p - e_base)) /
                 std::abs(static_cast<double>(e_base));
    worst_rel = std::max(worst_rel, rel);
  }

  EXPECT_LT(worst_rel, kSumOrderRelativeBound)
      << "EAM energy variance under neighbor permutation exceeds bound. "
      << "worst_rel=" << worst_rel
      << " bound=" << kSumOrderRelativeBound
      << " E_base=" << e_base;
}
