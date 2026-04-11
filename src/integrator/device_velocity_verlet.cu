// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet.cu — GPU velocity-Verlet kernels.

#include "device_velocity_verlet.cuh"

#include "../core/constants.hpp"
#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::integrator {

namespace {

__global__ void half_kick_kernel(VelocityVec* __restrict__ velocities,
                                 const ForceVec* __restrict__ forces,
                                 const i32* __restrict__ types,
                                 const real* __restrict__ masses, i32 natoms,
                                 real half_dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // ADR 0007 Integrator math contract: all arithmetic in double.
  // Velocity storage is double (vel_t=double in both modes); writeback is
  // direct (no static_cast<real>) — eliminates truncation bug.
  double d_half_dt = static_cast<double>(half_dt);
  double mass = static_cast<double>(masses[types[i]]);
  double factor = d_half_dt / (mass * kMvv2e);

  velocities[i].x += factor * static_cast<double>(forces[i].x);
  velocities[i].y += factor * static_cast<double>(forces[i].y);
  velocities[i].z += factor * static_cast<double>(forces[i].z);
}

__global__ void drift_kernel(PositionVec* __restrict__ positions,
                             const VelocityVec* __restrict__ velocities,
                             i32 natoms, real dt, Vec3D box_lo, Vec3D box_size,
                             bool pbc_x, bool pbc_y, bool pbc_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  // ADR 0007 Integrator math contract: position update in double.
  // Position storage is double (PositionVec = Vec3D) — direct write, no
  // static_cast<real> truncation.
  double d_dt = static_cast<double>(dt);
  double px = positions[i].x + d_dt * velocities[i].x;
  double py = positions[i].y + d_dt * velocities[i].y;
  double pz = positions[i].z + d_dt * velocities[i].z;

  // Wrap into box (Box is already double per Stage 1).
  if (pbc_x) {
    if (px < box_lo.x) px += box_size.x;
    else if (px >= box_lo.x + box_size.x) px -= box_size.x;
  }
  if (pbc_y) {
    if (py < box_lo.y) py += box_size.y;
    else if (py >= box_lo.y + box_size.y) py -= box_size.y;
  }
  if (pbc_z) {
    if (pz < box_lo.z) pz += box_size.z;
    else if (pz >= box_lo.z + box_size.z) pz -= box_size.z;
  }

  positions[i].x = px;
  positions[i].y = py;
  positions[i].z = pz;
}

__global__ void zero_forces_kernel(ForceVec* __restrict__ forces, i32 natoms) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;
  forces[i] = ForceVec{0, 0, 0};
}

// Fused half-kick + drift. Register-resident velocity hand-off: read F, kick
// velocity into a local, drift position using the local, then write both.
// Mathematically identical (same op order, same double-precision path) to
// half_kick_kernel followed by drift_kernel.
__global__ void fused_half_kick_drift_kernel(
    VelocityVec* __restrict__ velocities, PositionVec* __restrict__ positions,
    const ForceVec* __restrict__ forces, const i32* __restrict__ types,
    const real* __restrict__ masses, i32 natoms, real dt, Vec3D box_lo,
    Vec3D box_size, bool pbc_x, bool pbc_y, bool pbc_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;

  double d_dt = static_cast<double>(dt);
  double d_half_dt = 0.5 * d_dt;
  double mass = static_cast<double>(masses[types[i]]);
  double factor = d_half_dt / (mass * kMvv2e);

  // Half kick — matches half_kick_kernel exactly.
  double vx = velocities[i].x + factor * static_cast<double>(forces[i].x);
  double vy = velocities[i].y + factor * static_cast<double>(forces[i].y);
  double vz = velocities[i].z + factor * static_cast<double>(forces[i].z);

  // Drift — matches drift_kernel exactly, reading velocity from register.
  double px = positions[i].x + d_dt * vx;
  double py = positions[i].y + d_dt * vy;
  double pz = positions[i].z + d_dt * vz;

  if (pbc_x) {
    if (px < box_lo.x) px += box_size.x;
    else if (px >= box_lo.x + box_size.x) px -= box_size.x;
  }
  if (pbc_y) {
    if (py < box_lo.y) py += box_size.y;
    else if (py >= box_lo.y + box_size.y) py -= box_size.y;
  }
  if (pbc_z) {
    if (pz < box_lo.z) pz += box_size.z;
    else if (pz >= box_lo.z + box_size.z) pz -= box_size.z;
  }

  velocities[i].x = vx;
  velocities[i].y = vy;
  velocities[i].z = vz;
  positions[i].x = px;
  positions[i].y = py;
  positions[i].z = pz;
}

}  // namespace

void device_half_kick(VelocityVec* d_velocities, const ForceVec* d_forces,
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

void device_drift(PositionVec* d_positions, const VelocityVec* d_velocities,
                  i32 natoms, real dt, const Box& box, cudaStream_t stream) {
  if (natoms == 0) return;
  Vec3D box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  drift_kernel<<<grid, kBlock, 0, stream>>>(d_positions, d_velocities, natoms,
                                             dt, box.lo, box_size,
                                             box.periodic[0], box.periodic[1],
                                             box.periodic[2]);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_zero_forces(ForceVec* d_forces, i32 natoms, cudaStream_t stream) {
  if (natoms == 0) return;
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  zero_forces_kernel<<<grid, kBlock, 0, stream>>>(d_forces, natoms);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_fused_half_kick_drift(VelocityVec* d_velocities,
                                  PositionVec* d_positions,
                                  const ForceVec* d_forces,
                                  const i32* d_types, const real* d_masses,
                                  i32 natoms, real dt, const Box& box,
                                  cudaStream_t stream) {
  if (natoms == 0) return;
  Vec3D box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (natoms + kBlock - 1) / kBlock;
  fused_half_kick_drift_kernel<<<grid, kBlock, 0, stream>>>(
      d_velocities, d_positions, d_forces, d_types, d_masses, natoms, dt,
      box.lo, box_size, box.periodic[0], box.periodic[1], box.periodic[2]);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tdmd::integrator
