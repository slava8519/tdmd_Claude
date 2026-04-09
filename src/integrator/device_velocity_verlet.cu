// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet.cu — GPU velocity-Verlet kernels.

#include "device_velocity_verlet.cuh"

#include "../core/constants.hpp"
#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::integrator {

namespace {

__global__ void half_kick_kernel(Vec3* __restrict__ velocities,
                                 const Vec3* __restrict__ forces,
                                 const i32* __restrict__ types,
                                 const real* __restrict__ masses, i32 natoms,
                                 real half_dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // ADR 0007 Integrator math contract: all arithmetic in double.
  // Forces are promoted from force_t (float in mixed) to double on the fly.
  double d_half_dt = static_cast<double>(half_dt);
  double mass = static_cast<double>(masses[types[i]]);
  double factor = d_half_dt / (mass * kMvv2e);

  double vx = static_cast<double>(velocities[i].x) +
              factor * static_cast<double>(forces[i].x);
  double vy = static_cast<double>(velocities[i].y) +
              factor * static_cast<double>(forces[i].y);
  double vz = static_cast<double>(velocities[i].z) +
              factor * static_cast<double>(forces[i].z);

  velocities[i].x = static_cast<real>(vx);
  velocities[i].y = static_cast<real>(vy);
  velocities[i].z = static_cast<real>(vz);
}

__global__ void drift_kernel(Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities, i32 natoms,
                             real dt, Vec3 box_lo, Vec3 box_size, bool pbc_x,
                             bool pbc_y, bool pbc_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // ADR 0007 Integrator math contract: position update in double.
  double d_dt = static_cast<double>(dt);
  double px = static_cast<double>(positions[i].x) +
              d_dt * static_cast<double>(velocities[i].x);
  double py = static_cast<double>(positions[i].y) +
              d_dt * static_cast<double>(velocities[i].y);
  double pz = static_cast<double>(positions[i].z) +
              d_dt * static_cast<double>(velocities[i].z);

  // Wrap into box (in double for determinism).
  double blx = static_cast<double>(box_lo.x);
  double bly = static_cast<double>(box_lo.y);
  double blz = static_cast<double>(box_lo.z);
  double bsx = static_cast<double>(box_size.x);
  double bsy = static_cast<double>(box_size.y);
  double bsz = static_cast<double>(box_size.z);

  if (pbc_x) {
    if (px < blx) px += bsx;
    else if (px >= blx + bsx) px -= bsx;
  }
  if (pbc_y) {
    if (py < bly) py += bsy;
    else if (py >= bly + bsy) py -= bsy;
  }
  if (pbc_z) {
    if (pz < blz) pz += bsz;
    else if (pz >= blz + bsz) pz -= bsz;
  }

  positions[i].x = static_cast<real>(px);
  positions[i].y = static_cast<real>(py);
  positions[i].z = static_cast<real>(pz);
}

__global__ void zero_forces_kernel(Vec3* __restrict__ forces, i32 natoms) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;
  forces[i] = Vec3{0, 0, 0};
}

}  // namespace

void device_half_kick(Vec3* d_velocities, const Vec3* d_forces,
                      const i32* d_types, const real* d_masses, i32 natoms,
                      real dt, cudaStream_t stream) {
  if (natoms == 0) return;
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  real half_dt = real{0.5} * dt;
  half_kick_kernel<<<grid, kBlock, 0, stream>>>(d_velocities, d_forces,
                                                 d_types, d_masses, natoms,
                                                 half_dt);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_drift(Vec3* d_positions, const Vec3* d_velocities, i32 natoms,
                  real dt, const Box& box, cudaStream_t stream) {
  if (natoms == 0) return;
  Vec3 box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  drift_kernel<<<grid, kBlock, 0, stream>>>(d_positions, d_velocities, natoms,
                                             dt, box.lo, box_size,
                                             box.periodic[0], box.periodic[1],
                                             box.periodic[2]);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_zero_forces(Vec3* d_forces, i32 natoms, cudaStream_t stream) {
  if (natoms == 0) return;
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  zero_forces_kernel<<<grid, kBlock, 0, stream>>>(d_forces, natoms);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tdmd::integrator
