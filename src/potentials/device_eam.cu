// SPDX-License-Identifier: Apache-2.0
// device_eam.cu — GPU EAM/alloy force computation (3-pass).

#include "device_eam.cuh"

#include <vector>

#include "../core/determinism.hpp"
#include "../core/device_buffer.cuh"
#include "../core/device_math.cuh"
#include "../core/error.hpp"

namespace tdmd::potentials {

namespace {

/// Single-thread ID-ordered reduction helper. One copy per .cu because the
/// anonymous namespace confines it to this TU (see device_morse.cu for the
/// full rationale behind the sequential-on-device strategy).
__global__ void eam_sum_per_atom_kernel(
    const accum_t* __restrict__ d_e_per_atom, i32 natoms,
    accum_t* __restrict__ d_energy) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  accum_t s = *d_energy;
  for (i32 i = 0; i < natoms; ++i) {
    s += d_e_per_atom[i];
  }
  *d_energy = s;
}

}  // namespace

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
    const PositionVec* __restrict__ positions,
    const i32* __restrict__ types, const i32* __restrict__ neighbors,
    const i32* __restrict__ offsets, const i32* __restrict__ counts,
    i32 natoms, Vec3D box_lo, Vec3D box_size, bool pbc_x, bool pbc_y,
    bool pbc_z, real rc_sq, const DeviceSpline* __restrict__ density_meta,
    const real* __restrict__ coeff, accum_t* __restrict__ rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // Position load in double for relative-coordinate trick (ADR 0007/0008).
  pos_t pix = positions[i].x;
  pos_t piy = positions[i].y;
  pos_t piz = positions[i].z;

  // Box dimensions to force_t once, outside the loop.
  const force_t bsx = static_cast<force_t>(box_size.x);
  const force_t bsy = static_cast<force_t>(box_size.y);
  const force_t bsz = static_cast<force_t>(box_size.z);
  const force_t bsx_half = force_t{0.5} * bsx;
  const force_t bsy_half = force_t{0.5} * bsy;
  const force_t bsz_half = force_t{0.5} * bsz;

  // Cutoff in force_t — no epsilon buffer (skin distance handles boundary).
  const force_t rc_sq_f = static_cast<force_t>(rc_sq);

  i32 offset = offsets[i];
  i32 cnt = counts[i];
  accum_t rho_i = accum_t{0};

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];

    // Relative-coordinate trick: one double subtract, then cast to force_t.
    force_t dx = static_cast<force_t>(pix - positions[j].x);
    force_t dy = static_cast<force_t>(piy - positions[j].y);
    force_t dz = static_cast<force_t>(piz - positions[j].z);

    if (pbc_x) {
      if (dx > bsx_half) dx -= bsx;
      else if (dx < -bsx_half) dx += bsx;
    }
    if (pbc_y) {
      if (dy > bsy_half) dy -= bsy;
      else if (dy < -bsy_half) dy += bsy;
    }
    if (pbc_z) {
      if (dz > bsz_half) dz -= bsz;
      else if (dz < -bsz_half) dz += bsz;
    }

    force_t r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc_sq_f) continue;
    real r = math::sqrt_impl(static_cast<real>(r2));

    // rho_j(r) — density function of atom j's type.
    i32 tj = types[j] - 1;  // 0-based
    rho_i += static_cast<accum_t>(spline_eval(density_meta[tj], coeff, r));
  }

  rho[i] = rho_i;
}

// ---- Kernel: Pass 2 — embedding derivative ----

__global__ void eam_embedding_kernel(
    const i32* __restrict__ types, i32 natoms,
    const DeviceSpline* __restrict__ embedding_meta,
    const real* __restrict__ coeff, const accum_t* __restrict__ rho,
    real* __restrict__ fp, accum_t* __restrict__ d_energy,
    accum_t* __restrict__ d_e_per_atom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  i32 ti = types[i] - 1;
  real rho_i = static_cast<real>(rho[i]);
  fp[i] = spline_eval_deriv(embedding_meta[ti], coeff, rho_i);

  // R5 reduction site: per-atom embedding energy F(rho_i).
  if constexpr (kDeterministicReduce) {
    if (d_e_per_atom) {
      real e = spline_eval(embedding_meta[ti], coeff, rho_i);
      d_e_per_atom[i] = static_cast<accum_t>(e);
    }
  } else {
    if (d_energy) {
      real e = spline_eval(embedding_meta[ti], coeff, rho_i);
      atomicAdd(d_energy, static_cast<accum_t>(e));
    }
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
    const PositionVec* __restrict__ positions,
    ForceVec* __restrict__ forces, const i32* __restrict__ types,
    const i32* __restrict__ neighbors, const i32* __restrict__ offsets,
    const i32* __restrict__ counts, i32 natoms, i32 ntypes, Vec3D box_lo,
    Vec3D box_size, bool pbc_x, bool pbc_y, bool pbc_z, real rc_sq,
    const DeviceSpline* __restrict__ density_meta,
    const DeviceSpline* __restrict__ phi_meta,
    const real* __restrict__ coeff, const real* __restrict__ fp,
    accum_t* __restrict__ d_energy,
    accum_t* __restrict__ d_e_per_atom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // Position load in double for relative-coordinate trick (ADR 0007/0008).
  pos_t pix = positions[i].x;
  pos_t piy = positions[i].y;
  pos_t piz = positions[i].z;

  // Box dimensions to force_t once, outside the loop.
  const force_t bsx = static_cast<force_t>(box_size.x);
  const force_t bsy = static_cast<force_t>(box_size.y);
  const force_t bsz = static_cast<force_t>(box_size.z);
  const force_t bsx_half = force_t{0.5} * bsx;
  const force_t bsy_half = force_t{0.5} * bsy;
  const force_t bsz_half = force_t{0.5} * bsz;

  // Cutoff in force_t — no epsilon buffer (skin distance handles boundary).
  const force_t rc_sq_f = static_cast<force_t>(rc_sq);

  i32 ti = types[i] - 1;
  i32 offset = offsets[i];
  i32 cnt = counts[i];
  force_t fpi = static_cast<force_t>(fp[i]);

  force_t fx = force_t{0}, fy = force_t{0}, fz = force_t{0};
  force_t pe = force_t{0};

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];

    // Relative-coordinate trick: one double subtract, then cast to force_t.
    force_t dx = static_cast<force_t>(pix - positions[j].x);
    force_t dy = static_cast<force_t>(piy - positions[j].y);
    force_t dz = static_cast<force_t>(piz - positions[j].z);

    if (pbc_x) {
      if (dx > bsx_half) dx -= bsx;
      else if (dx < -bsx_half) dx += bsx;
    }
    if (pbc_y) {
      if (dy > bsy_half) dy -= bsy;
      else if (dy < -bsy_half) dy += bsy;
    }
    if (pbc_z) {
      if (dz > bsz_half) dz -= bsz;
      else if (dz < -bsz_half) dz += bsz;
    }

    force_t r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc_sq_f) continue;
    force_t r = math::sqrt_impl(r2);

    i32 tj = types[j] - 1;
    i32 pidx = phi_index_dev(ti, tj, ntypes);

    // Pair energy (half for full-list).
    force_t phi_val = spline_eval(phi_meta[pidx], coeff, r);
    pe += force_t{0.5} * phi_val;

    // Pair force derivative.
    force_t dphi_dr = spline_eval_deriv(phi_meta[pidx], coeff, r);

    // Embedding force: fp_i * drho_j/dr + fp_j * drho_i/dr.
    force_t drho_j_dr = spline_eval_deriv(density_meta[tj], coeff, r);
    force_t drho_i_dr = spline_eval_deriv(density_meta[ti], coeff, r);

    force_t fpair = -(dphi_dr + fpi * drho_j_dr
                      + static_cast<force_t>(fp[j]) * drho_i_dr) / r;

    fx += fpair * dx;
    fy += fpair * dy;
    fz += fpair * dz;
  }

  forces[i].x += fx;
  forces[i].y += fy;
  forces[i].z += fz;

  // R6 reduction site: per-atom pair energy half-sum.
  if constexpr (kDeterministicReduce) {
    if (d_e_per_atom) {
      d_e_per_atom[i] = static_cast<accum_t>(pe);
    }
  } else {
    if (d_energy) {
      atomicAdd(d_energy, static_cast<accum_t>(pe));
    }
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

void DeviceEam::compute(const PositionVec* d_positions, ForceVec* d_forces,
                        const i32* d_types, const i32* d_neighbors,
                        const i32* d_offsets, const i32* d_counts,
                        i32 natoms, const Box& box, accum_t* d_energy) {
  if (natoms == 0) return;

  Vec3D box_size = box.size();
  real rc_sq = cutoff_ * cutoff_;

  auto un = static_cast<std::size_t>(natoms);
  if (d_rho_.size() != un) {
    d_rho_.resize(un);
    d_fp_.resize(un);
  }

  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;

  // Per-atom energy scratch used only in deterministic mode. Reused between
  // the embedding and force passes because neither needs its contents after
  // the post-pass sum_per_atom_kernel is done.
  DeviceBuffer<accum_t> d_e_per_atom;
  accum_t* d_e_per_atom_ptr = nullptr;
  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      d_e_per_atom.resize(un);
      d_e_per_atom.zero();
      d_e_per_atom_ptr = d_e_per_atom.data();
    }
  }

  // Pass 1: density gather.
  d_rho_.zero();
  eam_density_kernel<<<grid, kBlock>>>(
      d_positions, d_types, d_neighbors, d_offsets, d_counts, natoms, box.lo,
      box_size, box.periodic[0], box.periodic[1], box.periodic[2], rc_sq,
      d_density_meta_.data(), d_coeff_.data(), d_rho_.data());
  TDMD_CUDA_CHECK(cudaGetLastError());

  // Pass 2: embedding (writes R5 energy contribution per atom).
  eam_embedding_kernel<<<grid, kBlock>>>(
      d_types, natoms, d_embedding_meta_.data(), d_coeff_.data(),
      d_rho_.data(), d_fp_.data(), d_energy, d_e_per_atom_ptr);
  TDMD_CUDA_CHECK(cudaGetLastError());

  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      eam_sum_per_atom_kernel<<<1, 1>>>(d_e_per_atom_ptr, natoms, d_energy);
      TDMD_CUDA_CHECK(cudaGetLastError());
      // Scratch is about to be reused by the force kernel — zero it so any
      // atoms skipped by the force kernel contribute 0, not stale embedding
      // values. (Currently every thread writes, but zeroing is cheap and
      // defends against a future early-exit path.)
      d_e_per_atom.zero();
    }
  }

  // Pass 3: forces (writes R6 energy contribution per atom).
  eam_force_kernel<<<grid, kBlock>>>(
      d_positions, d_forces, d_types, d_neighbors, d_offsets, d_counts, natoms,
      ntypes_, box.lo, box_size, box.periodic[0], box.periodic[1],
      box.periodic[2], rc_sq, d_density_meta_.data(), d_phi_meta_.data(),
      d_coeff_.data(), d_fp_.data(), d_energy, d_e_per_atom_ptr);
  TDMD_CUDA_CHECK(cudaGetLastError());

  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      eam_sum_per_atom_kernel<<<1, 1>>>(d_e_per_atom_ptr, natoms, d_energy);
      TDMD_CUDA_CHECK(cudaGetLastError());
    }
  }
}

}  // namespace tdmd::potentials
