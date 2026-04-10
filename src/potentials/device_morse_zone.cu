// SPDX-License-Identifier: Apache-2.0
// device_morse_zone.cu — Per-zone Morse force kernel.

#include "device_morse_zone.cuh"

#include "../core/determinism.hpp"
#include "../core/device_buffer.cuh"
#include "../core/device_math.cuh"
#include "../core/error.hpp"

namespace tdmd::potentials {

namespace {

/// Single-thread ID-ordered reduction helper (see device_morse.cu for the
/// extended rationale — repeated here because anonymous-namespace kernels
/// cannot cross translation units).
__global__ void sum_per_atom_zone_kernel(
    const accum_t* __restrict__ d_e_per_atom, i32 natoms,
    accum_t* __restrict__ d_energy) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  accum_t s = *d_energy;
  for (i32 i = 0; i < natoms; ++i) {
    s += d_e_per_atom[i];
  }
  *d_energy = s;
}

__global__ void morse_force_zone_kernel(
    const PositionVec* __restrict__ positions,
    ForceVec* __restrict__ forces, const i32* __restrict__ neighbors,
    const i32* __restrict__ offsets, const i32* __restrict__ counts,
    i32 first_atom, i32 atom_count, Vec3D box_lo, Vec3D box_size, bool pbc_x,
    bool pbc_y, bool pbc_z, MorseParams params,
    accum_t* __restrict__ d_energy,
    accum_t* __restrict__ d_e_per_atom) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;

  int i = first_atom + tid;

  // Position load in double for relative-coordinate trick (ADR 0007/0008).
  pos_t pix = positions[i].x;
  pos_t piy = positions[i].y;
  pos_t piz = positions[i].z;

  force_t fx = 0, fy = 0, fz = 0;
  force_t pe = 0;

  const force_t bsx = static_cast<force_t>(box_size.x);
  const force_t bsy = static_cast<force_t>(box_size.y);
  const force_t bsz = static_cast<force_t>(box_size.z);
  const force_t bsx_half = force_t{0.5} * bsx;
  const force_t bsy_half = force_t{0.5} * bsy;
  const force_t bsz_half = force_t{0.5} * bsz;

  const force_t rc_sq = static_cast<force_t>(params.rc_sq);

  i32 offset = offsets[i];
  i32 cnt = counts[i];

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];

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
    if (r2 < rc_sq) {
      force_t r = math::sqrt_impl(r2);
      force_t dr = r - static_cast<force_t>(params.r0);
      force_t exp_val = math::exp_impl(-static_cast<force_t>(params.alpha) * dr);
      force_t one_minus_exp = force_t{1} - exp_val;

      force_t D = static_cast<force_t>(params.d);
      pe += force_t{0.5} * D * (one_minus_exp * one_minus_exp - force_t{1});

      force_t dudr = force_t{2} * D *
                     static_cast<force_t>(params.alpha) * one_minus_exp * exp_val;
      force_t fpair = -dudr / r;

      fx += fpair * dx;
      fy += fpair * dy;
      fz += fpair * dz;
    }
  }

  forces[i].x += fx;
  forces[i].y += fy;
  forces[i].z += fz;

  // R4 reduction site. See device_morse.cu R3 comment for rationale.
  if constexpr (kDeterministicReduce) {
    if (d_e_per_atom) {
      d_e_per_atom[tid] = static_cast<accum_t>(pe);
    }
  } else {
    if (d_energy) {
      atomicAdd(d_energy, static_cast<accum_t>(pe));
    }
  }
}

}  // namespace

void compute_morse_gpu_zone(const PositionVec* d_positions,
                            ForceVec* d_forces, const i32* d_neighbors,
                            const i32* d_offsets, const i32* d_counts,
                            i32 first_atom, i32 atom_count, const Box& box,
                            const MorseParams& params, accum_t* d_energy,
                            cudaStream_t stream) {
  if (atom_count == 0) return;

  Vec3D box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (atom_count + kBlock - 1) / kBlock;

  DeviceBuffer<accum_t> d_e_per_atom;
  accum_t* d_e_per_atom_ptr = nullptr;
  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      d_e_per_atom.resize(static_cast<std::size_t>(atom_count));
      d_e_per_atom.zero();
      d_e_per_atom_ptr = d_e_per_atom.data();
    }
  }

  morse_force_zone_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, d_forces, d_neighbors, d_offsets, d_counts, first_atom,
      atom_count, box.lo, box_size, box.periodic[0], box.periodic[1],
      box.periodic[2], params, d_energy, d_e_per_atom_ptr);
  TDMD_CUDA_CHECK(cudaGetLastError());

  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      sum_per_atom_zone_kernel<<<1, 1, 0, stream>>>(d_e_per_atom_ptr,
                                                     atom_count, d_energy);
      TDMD_CUDA_CHECK(cudaGetLastError());
    }
  }
}

}  // namespace tdmd::potentials
