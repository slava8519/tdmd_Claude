// SPDX-License-Identifier: Apache-2.0
// device_velocity_verlet_zone.cu — Per-zone GPU integration kernels.

#include "device_velocity_verlet_zone.cuh"

#include "../core/constants.hpp"
#include "../core/device_buffer.cuh"
#include "../core/error.hpp"

namespace tdmd::integrator {

namespace {

__global__ void half_kick_zone_kernel(VelocityVec* __restrict__ velocities,
                                      const ForceVec* __restrict__ forces,
                                      const i32* __restrict__ types,
                                      const real* __restrict__ masses,
                                      i32 first_atom, i32 atom_count,
                                      real half_dt) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;
  int i = first_atom + tid;

  // ADR 0007 Integrator math contract: all arithmetic in double.
  // Velocity storage is double — direct accumulate, no truncation cast.
  double d_half_dt = static_cast<double>(half_dt);
  double mass = static_cast<double>(masses[types[i]]);
  double factor = d_half_dt / (mass * kMvv2e);

  velocities[i].x += factor * static_cast<double>(forces[i].x);
  velocities[i].y += factor * static_cast<double>(forces[i].y);
  velocities[i].z += factor * static_cast<double>(forces[i].z);
}

__global__ void drift_zone_kernel(PositionVec* __restrict__ positions,
                                  const VelocityVec* __restrict__ velocities,
                                  i32 first_atom, i32 atom_count, real dt,
                                  Vec3D box_lo, Vec3D box_size, bool pbc_x,
                                  bool pbc_y, bool pbc_z) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;
  int i = first_atom + tid;

  // ADR 0007 Integrator math contract: position update in double.
  // Position storage is double — direct write, no truncation cast.
  double d_dt = static_cast<double>(dt);
  double px = positions[i].x + d_dt * velocities[i].x;
  double py = positions[i].y + d_dt * velocities[i].y;
  double pz = positions[i].z + d_dt * velocities[i].z;

  // Wrap into box (Box is already double per Stage 1).
  if (pbc_x) {
    if (px < box_lo.x) px += box_size.x;
    if (px >= box_lo.x + box_size.x) px -= box_size.x;
  }
  if (pbc_y) {
    if (py < box_lo.y) py += box_size.y;
    if (py >= box_lo.y + box_size.y) py -= box_size.y;
  }
  if (pbc_z) {
    if (pz < box_lo.z) pz += box_size.z;
    if (pz >= box_lo.z + box_size.z) pz -= box_size.z;
  }

  positions[i].x = px;
  positions[i].y = py;
  positions[i].z = pz;
}

__global__ void zero_forces_zone_kernel(ForceVec* __restrict__ forces,
                                        i32 first_atom, i32 atom_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;
  int i = first_atom + tid;
  forces[i].x = force_t{0};
  forces[i].y = force_t{0};
  forces[i].z = force_t{0};
}

}  // namespace

void device_half_kick_zone(VelocityVec* d_velocities, const ForceVec* d_forces,
                           const i32* d_types, const real* d_masses,
                           i32 first_atom, i32 atom_count, real dt,
                           cudaStream_t stream) {
  if (atom_count == 0) return;
  constexpr int kBlock = 256;
  int grid = (atom_count + kBlock - 1) / kBlock;
  real half_dt = real{0.5} * dt;
  half_kick_zone_kernel<<<grid, kBlock, 0, stream>>>(
      d_velocities, d_forces, d_types, d_masses, first_atom, atom_count,
      half_dt);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_drift_zone(PositionVec* d_positions,
                       const VelocityVec* d_velocities, i32 first_atom,
                       i32 atom_count, real dt, const Box& box,
                       cudaStream_t stream) {
  if (atom_count == 0) return;
  Vec3D box_size = box.size();
  constexpr int kBlock = 256;
  int grid = (atom_count + kBlock - 1) / kBlock;
  drift_zone_kernel<<<grid, kBlock, 0, stream>>>(
      d_positions, d_velocities, first_atom, atom_count, dt, box.lo, box_size,
      box.periodic[0], box.periodic[1], box.periodic[2]);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_zero_forces_zone(ForceVec* d_forces, i32 first_atom,
                             i32 atom_count, cudaStream_t stream) {
  if (atom_count == 0) return;
  constexpr int kBlock = 256;
  int grid = (atom_count + kBlock - 1) / kBlock;
  zero_forces_zone_kernel<<<grid, kBlock, 0, stream>>>(d_forces, first_atom,
                                                       atom_count);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tdmd::integrator
