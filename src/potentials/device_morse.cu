// SPDX-License-Identifier: Apache-2.0
// device_morse.cu — GPU Morse pair force kernel.

#include "device_morse.cuh"

#include "../core/determinism.hpp"
#include "../core/device_buffer.cuh"
#include "../core/device_math.cuh"
#include "../core/error.hpp"

namespace tdmd::potentials {

namespace {

/// Single-thread ID-ordered reduction: sum d_e_per_atom[0..natoms) into *d_energy.
/// Used only when TDMD_DETERMINISTIC_REDUCE=ON. Deliberately not parallelized —
/// the goal is a bit-reproducible result that depends only on the input, and
/// any parallel (tree) reduction inside a kernel would reintroduce order
/// nondeterminism the moment two blocks tie on scheduling. One thread,
/// sequential accum_t sum, tiny next to the force kernel itself.
__global__ void sum_per_atom_kernel(const accum_t* __restrict__ d_e_per_atom,
                                    i32 natoms,
                                    accum_t* __restrict__ d_energy) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  accum_t s = *d_energy;
  for (i32 i = 0; i < natoms; ++i) {
    s += d_e_per_atom[i];
  }
  *d_energy = s;
}

__global__ void morse_force_kernel(const PositionVec* __restrict__ positions,
                                   ForceVec* __restrict__ forces,
                                   const i32* __restrict__ neighbors,
                                   const i32* __restrict__ offsets,
                                   const i32* __restrict__ counts,
                                   i32 natoms, Vec3D box_lo, Vec3D box_size,
                                   bool pbc_x, bool pbc_y, bool pbc_z,
                                   MorseParams params, accum_t* __restrict__ d_energy,
                                   accum_t* __restrict__ d_e_per_atom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // Position load in double for relative-coordinate trick (ADR 0007/0008).
  pos_t pix = positions[i].x;
  pos_t piy = positions[i].y;
  pos_t piz = positions[i].z;

  force_t fx = 0, fy = 0, fz = 0;
  force_t pe = 0;

  // Box dimensions to force_t once, outside the loop.
  const force_t bsx = static_cast<force_t>(box_size.x);
  const force_t bsy = static_cast<force_t>(box_size.y);
  const force_t bsz = static_cast<force_t>(box_size.z);
  const force_t bsx_half = force_t{0.5} * bsx;
  const force_t bsy_half = force_t{0.5} * bsy;
  const force_t bsz_half = force_t{0.5} * bsz;

  // Cutoff in force_t — no epsilon buffer (skin distance handles boundary).
  const force_t rc_sq = static_cast<force_t>(params.rc_sq);

  i32 offset = offsets[i];
  i32 cnt = counts[i];

  for (i32 k = 0; k < cnt; ++k) {
    i32 j = neighbors[offset + k];

    // Relative-coordinate trick: one double subtract, then cast to force_t.
    // In mixed mode (force_t=float): more accurate than float-float subtract.
    // In fp64 mode (force_t=double): identity cast, full double precision.
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

  // R3 reduction site. Default path: float/double atomicAdd, fast but
  // warp-scheduling-nondeterministic. Deterministic path (ADR 0010): drop
  // the per-atom contribution into a dense buffer that the host will sum
  // in strict ID order after the kernel finishes.
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

}  // namespace

void compute_morse_gpu(const PositionVec* d_positions, ForceVec* d_forces,
                       const i32* d_neighbors, const i32* d_offsets,
                       const i32* d_counts, i32 natoms, const Box& box,
                       const MorseParams& params, accum_t* d_energy,
                       cudaStream_t stream) {
  if (natoms == 0) return;

  Vec3D box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;

  DeviceBuffer<accum_t> d_e_per_atom;
  accum_t* d_e_per_atom_ptr = nullptr;
  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      d_e_per_atom.resize(static_cast<std::size_t>(natoms));
      d_e_per_atom.zero();
      d_e_per_atom_ptr = d_e_per_atom.data();
    }
  }

  morse_force_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, d_forces, d_neighbors, d_offsets, d_counts, natoms, box.lo,
      box_size, box.periodic[0], box.periodic[1], box.periodic[2], params,
      d_energy, d_e_per_atom_ptr);
  TDMD_CUDA_CHECK(cudaGetLastError());

  if constexpr (kDeterministicReduce) {
    if (d_energy) {
      sum_per_atom_kernel<<<1, 1, 0, stream>>>(d_e_per_atom_ptr, natoms,
                                                d_energy);
      TDMD_CUDA_CHECK(cudaGetLastError());
    }
  }
}

}  // namespace tdmd::potentials
