// SPDX-License-Identifier: Apache-2.0
// device_eam.cu — GPU EAM/alloy force computation (3-pass).

#include "device_eam.cuh"

#include <vector>

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::potentials {

// ---- Device-side spline evaluation ----

/// Evaluate spline at r using flat coefficient array.
__device__ real spline_eval(const DeviceSpline& sp, const real* coeff, real r) {
  if (sp.n < 2) return real{0};
  real x = (r - sp.rmin) / sp.dr;
  auto idx = static_cast<i32>(x);
  if (idx < 0) idx = 0;
  if (idx >= sp.n - 1) idx = sp.n - 2;
  real dx = r - (sp.rmin + static_cast<real>(idx) * sp.dr);

  // Coefficients are packed: a[0..n-2], b[0..n-2], c[0..n-2], d[0..n-2].
  i32 ns = sp.n - 1;
  real a = coeff[sp.offset + idx];
  real b = coeff[sp.offset + ns + idx];
  real c = coeff[sp.offset + 2 * ns + idx];
  real d = coeff[sp.offset + 3 * ns + idx];
  return a + dx * (b + dx * (c + dx * d));
}

/// Evaluate spline derivative at r.
__device__ real spline_eval_deriv(const DeviceSpline& sp, const real* coeff,
                                  real r) {
  if (sp.n < 2) return real{0};
  real x = (r - sp.rmin) / sp.dr;
  auto idx = static_cast<i32>(x);
  if (idx < 0) idx = 0;
  if (idx >= sp.n - 1) idx = sp.n - 2;
  real dx = r - (sp.rmin + static_cast<real>(idx) * sp.dr);

  i32 ns = sp.n - 1;
  real b = coeff[sp.offset + ns + idx];
  real c = coeff[sp.offset + 2 * ns + idx];
  real d = coeff[sp.offset + 3 * ns + idx];
  return b + dx * (real{2} * c + dx * real{3} * d);
}

// ---- Kernel: Pass 1 — density gather (full-list) ----

__global__ void eam_density_kernel(
    const Vec3* __restrict__ positions, const i32* __restrict__ types,
    const i32* __restrict__ neighbors, const i32* __restrict__ offsets,
    const i32* __restrict__ counts, i32 natoms, Vec3 box_lo, Vec3 box_size,
    bool pbc_x, bool pbc_y, bool pbc_z, real rc_sq,
    const DeviceSpline* __restrict__ density_meta,
    const real* __restrict__ coeff, real* __restrict__ rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  Vec3 pi = positions[i];
  i32 offset = offsets[i];
  i32 cnt = counts[i];
  real rho_i = real{0};

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];
    real dx = pi.x - positions[j].x;
    real dy = pi.y - positions[j].y;
    real dz = pi.z - positions[j].z;

    if (pbc_x) {
      if (dx > box_size.x * real{0.5}) dx -= box_size.x;
      else if (dx < -box_size.x * real{0.5}) dx += box_size.x;
    }
    if (pbc_y) {
      if (dy > box_size.y * real{0.5}) dy -= box_size.y;
      else if (dy < -box_size.y * real{0.5}) dy += box_size.y;
    }
    if (pbc_z) {
      if (dz > box_size.z * real{0.5}) dz -= box_size.z;
      else if (dz < -box_size.z * real{0.5}) dz += box_size.z;
    }

    real r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc_sq) continue;
    real r = sqrt(r2);

    // rho_j(r) — density function of atom j's type.
    i32 tj = types[j] - 1;  // 0-based
    rho_i += spline_eval(density_meta[tj], coeff, r);
  }

  rho[i] = rho_i;
}

// ---- Kernel: Pass 2 — embedding derivative ----

__global__ void eam_embedding_kernel(
    const i32* __restrict__ types, i32 natoms,
    const DeviceSpline* __restrict__ embedding_meta,
    const real* __restrict__ coeff, const real* __restrict__ rho,
    real* __restrict__ fp, real* __restrict__ d_energy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  i32 ti = types[i] - 1;
  real rho_i = rho[i];
  fp[i] = spline_eval_deriv(embedding_meta[ti], coeff, rho_i);

  if (d_energy) {
    real e = spline_eval(embedding_meta[ti], coeff, rho_i);
    atomicAdd(d_energy, e);
  }
}

// ---- Kernel: Pass 3 — force scatter (full-list) ----

/// phi_index on device (same logic as CPU).
__device__ i32 phi_index_dev(i32 ti, i32 tj, i32 ntypes) {
  if (ti > tj) {
    i32 tmp = ti;
    ti = tj;
    tj = tmp;
  }
  return ti * ntypes - ti * (ti + 1) / 2 + tj;
}

__global__ void eam_force_kernel(
    const Vec3* __restrict__ positions, Vec3* __restrict__ forces,
    const i32* __restrict__ types, const i32* __restrict__ neighbors,
    const i32* __restrict__ offsets, const i32* __restrict__ counts,
    i32 natoms, i32 ntypes, Vec3 box_lo, Vec3 box_size, bool pbc_x,
    bool pbc_y, bool pbc_z, real rc_sq,
    const DeviceSpline* __restrict__ density_meta,
    const DeviceSpline* __restrict__ phi_meta,
    const real* __restrict__ coeff, const real* __restrict__ fp,
    real* __restrict__ d_energy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  Vec3 pi = positions[i];
  i32 ti = types[i] - 1;
  i32 offset = offsets[i];
  i32 cnt = counts[i];
  real fpi = fp[i];

  real fx = real{0}, fy = real{0}, fz = real{0};
  real pe = real{0};

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];
    real dx = pi.x - positions[j].x;
    real dy = pi.y - positions[j].y;
    real dz = pi.z - positions[j].z;

    if (pbc_x) {
      if (dx > box_size.x * real{0.5}) dx -= box_size.x;
      else if (dx < -box_size.x * real{0.5}) dx += box_size.x;
    }
    if (pbc_y) {
      if (dy > box_size.y * real{0.5}) dy -= box_size.y;
      else if (dy < -box_size.y * real{0.5}) dy += box_size.y;
    }
    if (pbc_z) {
      if (dz > box_size.z * real{0.5}) dz -= box_size.z;
      else if (dz < -box_size.z * real{0.5}) dz += box_size.z;
    }

    real r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc_sq) continue;
    real r = sqrt(r2);

    i32 tj = types[j] - 1;
    i32 pidx = phi_index_dev(ti, tj, ntypes);

    // Pair energy (half for full-list).
    real phi_val = spline_eval(phi_meta[pidx], coeff, r);
    pe += real{0.5} * phi_val;

    // Pair force derivative.
    real dphi_dr = spline_eval_deriv(phi_meta[pidx], coeff, r);

    // Embedding force: fp_i * drho_j/dr + fp_j * drho_i/dr.
    real drho_j_dr = spline_eval_deriv(density_meta[tj], coeff, r);
    real drho_i_dr = spline_eval_deriv(density_meta[ti], coeff, r);

    real fpair = -(dphi_dr + fpi * drho_j_dr + fp[j] * drho_i_dr) / r;

    fx += fpair * dx;
    fy += fpair * dy;
    fz += fpair * dz;
  }

  forces[i].x += fx;
  forces[i].y += fy;
  forces[i].z += fz;

  if (d_energy) {
    atomicAdd(d_energy, pe);
  }
}

// ---- upload_tables ----

static void pack_spline(const SplineTable& src, DeviceSpline& meta,
                        std::vector<real>& flat, i32& offset) {
  meta.dr = src.dr;
  meta.rmin = src.rmin;
  meta.n = src.n;
  meta.offset = offset;

  i32 ns = src.n - 1;
  if (ns <= 0) return;
  // Pack: a[0..ns-1], b[0..ns-1], c[0..ns-1], d[0..ns-1].
  for (i32 i = 0; i < ns; ++i) flat.push_back(src.a[static_cast<std::size_t>(i)]);
  for (i32 i = 0; i < ns; ++i) flat.push_back(src.b[static_cast<std::size_t>(i)]);
  for (i32 i = 0; i < ns; ++i) flat.push_back(src.c[static_cast<std::size_t>(i)]);
  for (i32 i = 0; i < ns; ++i) flat.push_back(src.d[static_cast<std::size_t>(i)]);
  offset += 4 * ns;
}

void DeviceEam::upload_tables(const EamAlloy& eam) {
  ntypes_ = eam.ntypes();
  cutoff_ = eam.cutoff();

  const auto& embedding = eam.embedding();
  const auto& density = eam.density();
  const auto& phi = eam.phi();

  // Pack all spline coefficients into a flat array.
  std::vector<real> flat_coeff;
  i32 offset = 0;

  h_embedding_.resize(static_cast<std::size_t>(ntypes_));
  for (i32 t = 0; t < ntypes_; ++t) {
    pack_spline(embedding[static_cast<std::size_t>(t)],
                h_embedding_[static_cast<std::size_t>(t)], flat_coeff, offset);
  }

  h_density_.resize(static_cast<std::size_t>(ntypes_));
  for (i32 t = 0; t < ntypes_; ++t) {
    pack_spline(density[static_cast<std::size_t>(t)],
                h_density_[static_cast<std::size_t>(t)], flat_coeff, offset);
  }

  h_phi_.resize(phi.size());
  for (std::size_t p = 0; p < phi.size(); ++p) {
    pack_spline(phi[p], h_phi_[p], flat_coeff, offset);
  }

  // Upload to device.
  d_coeff_.resize(flat_coeff.size());
  d_coeff_.copy_from_host(flat_coeff.data(), flat_coeff.size());

  d_embedding_meta_.resize(h_embedding_.size());
  d_embedding_meta_.copy_from_host(h_embedding_.data(), h_embedding_.size());

  d_density_meta_.resize(h_density_.size());
  d_density_meta_.copy_from_host(h_density_.data(), h_density_.size());

  d_phi_meta_.resize(h_phi_.size());
  d_phi_meta_.copy_from_host(h_phi_.data(), h_phi_.size());
}

void DeviceEam::compute(const Vec3* d_positions, Vec3* d_forces,
                        const i32* d_types, const i32* d_neighbors,
                        const i32* d_offsets, const i32* d_counts,
                        i32 natoms, const Box& box, real* d_energy) {
  if (natoms == 0) return;

  Vec3 box_size = box.size();
  real rc_sq = cutoff_ * cutoff_;

  auto un = static_cast<std::size_t>(natoms);
  if (d_rho_.size() != un) {
    d_rho_.resize(un);
    d_fp_.resize(un);
  }

  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;

  // Pass 1: density gather.
  d_rho_.zero();
  eam_density_kernel<<<grid, kBlock>>>(
      d_positions, d_types, d_neighbors, d_offsets, d_counts, natoms, box.lo,
      box_size, box.periodic[0], box.periodic[1], box.periodic[2], rc_sq,
      d_density_meta_.data(), d_coeff_.data(), d_rho_.data());
  TDMD_CUDA_CHECK(cudaGetLastError());

  // Pass 2: embedding.
  eam_embedding_kernel<<<grid, kBlock>>>(
      d_types, natoms, d_embedding_meta_.data(), d_coeff_.data(),
      d_rho_.data(), d_fp_.data(), d_energy);
  TDMD_CUDA_CHECK(cudaGetLastError());

  // Pass 3: forces.
  eam_force_kernel<<<grid, kBlock>>>(
      d_positions, d_forces, d_types, d_neighbors, d_offsets, d_counts, natoms,
      ntypes_, box.lo, box_size, box.periodic[0], box.periodic[1],
      box.periodic[2], rc_sq, d_density_meta_.data(), d_phi_meta_.data(),
      d_coeff_.data(), d_fp_.data(), d_energy);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tdmd::potentials
