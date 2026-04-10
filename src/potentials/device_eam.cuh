// SPDX-License-Identifier: Apache-2.0
// device_eam.cuh — GPU EAM/alloy force computation (3-pass).
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"
#include "eam_alloy.hpp"

namespace tdmd::potentials {

/// @brief Flat spline table on device — coefficients a,b,c,d packed contiguously.
struct DeviceSpline {
  real dr;       // spacing
  real rmin;     // starting r
  i32 n;         // number of data points (coefficients have n-1 entries)
  i32 offset;    // offset into the flat coefficient array
};

/// @brief Device-side EAM data and force computation.
///
/// Uploads spline tables from an EamAlloy instance to device global memory,
/// then provides a 3-pass compute method.
class DeviceEam {
 public:
  /// @brief Upload tables from a CPU EamAlloy. Must be called before compute.
  void upload_tables(const EamAlloy& eam);

  /// @brief 3-pass EAM force compute on GPU.
  ///
  /// @param d_positions  Device positions.
  /// @param d_forces     Device forces (must be zeroed before call).
  /// @param d_types      Device atom types (1-based, LAMMPS convention).
  /// @param d_neighbors  Flat CSR neighbor list.
  /// @param d_offsets    Per-atom offsets.
  /// @param d_counts     Per-atom neighbor counts.
  /// @param natoms       Number of atoms.
  /// @param box          Simulation box (host).
  /// @param d_energy     Optional device pointer to accumulate PE (may be nullptr).
  /// @param stream       CUDA stream for all internal launches (default 0 =
  ///                     legacy default stream). Threaded through so the
  ///                     scheduler can share its single compute stream when
  ///                     the EAM production pipeline slot lands (FEAT-EAM-
  ///                     Production-Pipeline).
  void compute(const PositionVec* d_positions, ForceVec* d_forces,
               const i32* d_types, const i32* d_neighbors,
               const i32* d_offsets, const i32* d_counts, i32 natoms,
               const Box& box, accum_t* d_energy, cudaStream_t stream = 0);

  [[nodiscard]] real cutoff() const noexcept { return cutoff_; }

 private:
  i32 ntypes_{0};
  real cutoff_{0};

  // Spline metadata (host-side, uploaded once).
  std::vector<DeviceSpline> h_embedding_;
  std::vector<DeviceSpline> h_density_;
  std::vector<DeviceSpline> h_phi_;

  // Device spline metadata.
  DeviceBuffer<DeviceSpline> d_embedding_meta_;
  DeviceBuffer<DeviceSpline> d_density_meta_;
  DeviceBuffer<DeviceSpline> d_phi_meta_;

  // Flat packed coefficients: a[0..n-2], b[0..n-2], c[0..n-2], d[0..n-2]
  // for each spline contiguously.
  DeviceBuffer<real> d_coeff_;

  // Per-atom buffers (allocated during compute).
  DeviceBuffer<accum_t> d_rho_;
  DeviceBuffer<real> d_fp_;
};

}  // namespace tdmd::potentials
