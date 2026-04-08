// SPDX-License-Identifier: Apache-2.0
// device_morse_zone.cu — Per-zone Morse force kernel.

#include "device_morse_zone.cuh"

#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::potentials {

namespace {

__global__ void morse_force_zone_kernel(
    const Vec3* __restrict__ positions, Vec3* __restrict__ forces,
    const i32* __restrict__ neighbors, const i32* __restrict__ offsets,
    const i32* __restrict__ counts, i32 first_atom, i32 atom_count,
    Vec3 box_lo, Vec3 box_size, bool pbc_x, bool pbc_y, bool pbc_z,
    MorseParams params, real* __restrict__ d_energy) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;

  int i = first_atom + tid;
  Vec3 pi = positions[i];
  real fx = real{0}, fy = real{0}, fz = real{0};
  real pe = real{0};

  i32 offset = offsets[i];
  i32 cnt = counts[i];

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
    if (r2 < params.rc_sq) {
      real r = sqrt(r2);
      real dr = r - params.r0;
      real exp_val = exp(-params.alpha * dr);
      real one_minus_exp = real{1} - exp_val;

      pe += real{0.5} * params.d * (one_minus_exp * one_minus_exp - real{1});

      real dudr = real{2} * params.d * params.alpha * one_minus_exp * exp_val;
      real fpair = -dudr / r;

      fx += fpair * dx;
      fy += fpair * dy;
      fz += fpair * dz;
    }
  }

  forces[i].x += fx;
  forces[i].y += fy;
  forces[i].z += fz;

  if (d_energy) {
    atomicAdd(d_energy, pe);
  }
}

}  // namespace

void compute_morse_gpu_zone(const Vec3* d_positions, Vec3* d_forces,
                            const i32* d_neighbors, const i32* d_offsets,
                            const i32* d_counts, i32 first_atom,
                            i32 atom_count, const Box& box,
                            const MorseParams& params, real* d_energy,
                            cudaStream_t stream) {
  if (atom_count == 0) return;

  Vec3 box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (atom_count + kBlock - 1) / kBlock;

  morse_force_zone_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, d_forces, d_neighbors, d_offsets, d_counts, first_atom,
      atom_count, box.lo, box_size, box.periodic[0], box.periodic[1],
      box.periodic[2], params, d_energy);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tdmd::potentials
