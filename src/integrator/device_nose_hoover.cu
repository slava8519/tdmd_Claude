// SPDX-License-Identifier: Apache-2.0
// device_nose_hoover.cu — GPU Nosé-Hoover NVT thermostat implementation.
//
// KE reduction: two-pass (per-block partial sums, then host reduction).
// Velocity scaling: trivial element-wise kernel.
// Chain update: MTTK scheme on host (chain is O(chain_length), not worth GPU).

#include "device_nose_hoover.cuh"

#include <cmath>
#include <vector>

#include "../core/constants.hpp"
#include "../core/device_buffer.cuh"

namespace tdmd::integrator {

// ---- GPU kernels ----

static constexpr int kBlock = 256;

/// Per-block KE reduction kernel. Each block reduces its chunk and writes
/// one partial sum to d_block_sums. Accumulation always in double (accum_t).
__global__ void ke_reduce_kernel(const Vec3* __restrict__ velocities,
                                 const i32* __restrict__ types,
                                 const real* __restrict__ masses, i32 natoms,
                                 accum_t* __restrict__ d_block_sums) {
  extern __shared__ accum_t sdata[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  accum_t val = 0;
  if (gid < natoms) {
    accum_t mass = static_cast<accum_t>(masses[types[gid]]);
    Vec3 v = velocities[gid];
    val = 0.5 * mass * kMvv2e *
          (static_cast<accum_t>(v.x) * static_cast<accum_t>(v.x) +
           static_cast<accum_t>(v.y) * static_cast<accum_t>(v.y) +
           static_cast<accum_t>(v.z) * static_cast<accum_t>(v.z));
  }
  sdata[tid] = val;
  __syncthreads();

  // Standard parallel reduction.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_block_sums[blockIdx.x] = sdata[0];
  }
}

__global__ void scale_vel_kernel(Vec3* __restrict__ velocities, i32 natoms,
                                 real factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= natoms) return;
  velocities[i].x *= factor;
  velocities[i].y *= factor;
  velocities[i].z *= factor;
}

__global__ void scale_vel_zone_kernel(Vec3* __restrict__ velocities,
                                      i32 first_atom, i32 atom_count,
                                      real factor) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= atom_count) return;
  int i = first_atom + tid;
  velocities[i].x *= factor;
  velocities[i].y *= factor;
  velocities[i].z *= factor;
}

// ---- Device functions ----

accum_t device_compute_ke(const Vec3* d_velocities, const i32* d_types,
                          const real* d_masses, i32 natoms) {
  if (natoms == 0) return 0;

  int grid = (natoms + kBlock - 1) / kBlock;

  // Allocate block sums on device (always double).
  accum_t* d_block_sums = nullptr;
  TDMD_CUDA_CHECK(cudaMalloc(&d_block_sums,
                              static_cast<std::size_t>(grid) * sizeof(accum_t)));

  ke_reduce_kernel<<<grid, kBlock, kBlock * sizeof(accum_t)>>>(
      d_velocities, d_types, d_masses, natoms, d_block_sums);
  TDMD_CUDA_CHECK(cudaGetLastError());

  // Copy block sums to host and sum.
  std::vector<accum_t> h_sums(static_cast<std::size_t>(grid));
  TDMD_CUDA_CHECK(cudaMemcpy(h_sums.data(), d_block_sums,
                              static_cast<std::size_t>(grid) * sizeof(accum_t),
                              cudaMemcpyDeviceToHost));
  cudaFree(d_block_sums);

  accum_t total = 0;
  for (accum_t s : h_sums) total += s;
  return total;
}

void device_scale_velocities(Vec3* d_velocities, i32 natoms, real factor) {
  if (natoms == 0) return;
  int grid = (natoms + kBlock - 1) / kBlock;
  scale_vel_kernel<<<grid, kBlock>>>(d_velocities, natoms, factor);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

void device_scale_velocities_zone(Vec3* d_velocities, i32 first_atom,
                                  i32 atom_count, real factor,
                                  cudaStream_t stream) {
  if (atom_count == 0) return;
  int grid = (atom_count + kBlock - 1) / kBlock;
  scale_vel_zone_kernel<<<grid, kBlock, 0, stream>>>(d_velocities, first_atom,
                                                      atom_count, factor);
  TDMD_CUDA_CHECK(cudaGetLastError());
}

// ---- v_max reduction ----

/// Per-block max speed reduction. Each block computes max |v| for its chunk.
/// Accumulation always in double (accum_t).
__global__ void vmax_reduce_kernel(const Vec3* __restrict__ velocities,
                                   i32 natoms,
                                   accum_t* __restrict__ d_block_max) {
  extern __shared__ accum_t sdata[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  accum_t val = 0;
  if (gid < natoms) {
    Vec3 v = velocities[gid];
    val = static_cast<accum_t>(v.x) * static_cast<accum_t>(v.x) +
          static_cast<accum_t>(v.y) * static_cast<accum_t>(v.y) +
          static_cast<accum_t>(v.z) * static_cast<accum_t>(v.z);
  }
  sdata[tid] = val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_block_max[blockIdx.x] = sdata[0];
  }
}

accum_t device_compute_vmax(const Vec3* d_velocities, i32 natoms) {
  if (natoms == 0) return 0;

  int grid = (natoms + kBlock - 1) / kBlock;

  accum_t* d_block_max = nullptr;
  TDMD_CUDA_CHECK(cudaMalloc(&d_block_max,
                              static_cast<std::size_t>(grid) * sizeof(accum_t)));

  vmax_reduce_kernel<<<grid, kBlock, kBlock * sizeof(accum_t)>>>(
      d_velocities, natoms, d_block_max);
  TDMD_CUDA_CHECK(cudaGetLastError());

  std::vector<accum_t> h_max(static_cast<std::size_t>(grid));
  TDMD_CUDA_CHECK(cudaMemcpy(h_max.data(), d_block_max,
                              static_cast<std::size_t>(grid) * sizeof(accum_t),
                              cudaMemcpyDeviceToHost));
  cudaFree(d_block_max);

  accum_t max_v2 = 0;
  for (accum_t m : h_max) {
    if (m > max_v2) max_v2 = m;
  }
  return std::sqrt(max_v2);
}

// ---- Nosé-Hoover Chain (host-side) ----

NoseHooverChain::NoseHooverChain(const NHCConfig& cfg, real dt, i32 n_dof)
    : cfg_(cfg), dt_(dt), n_dof_(n_dof) {
  auto len = static_cast<std::size_t>(cfg.chain_length);
  eta_.assign(len, real{0});
  eta_v_.assign(len, real{0});
  eta_m_.resize(len);

  // Chain masses following LAMMPS convention:
  // Q_1 = n_dof * kB * T_target * t_period^2
  // Q_k = kB * T_target * t_period^2  (k > 1)
  real kt = kBoltzmann * cfg_.t_target;
  real tau2 = cfg_.t_period * cfg_.t_period;

  eta_m_[0] = static_cast<real>(n_dof_) * kt * tau2;
  for (std::size_t i = 1; i < len; ++i) {
    eta_m_[i] = kt * tau2;
  }
}

real NoseHooverChain::half_step(accum_t ke_current) {
  // MTTK integration of the NHC (half-step).
  // References:
  //   Martyna, Tuckerman, Tobias, Klein, Mol. Phys. 87, 1117 (1996)
  //   LAMMPS src/fix_nh.cpp (nhc_temp_integrate)
  //
  // The chain is integrated from the outermost thermostat inward.
  // The velocity scaling factor is accumulated as exp(-ξ₁·dt/2).

  auto M = static_cast<i32>(eta_v_.size());
  if (M == 0) return real{1};

  real half_dt = real{0.5} * dt_;

  // Target KE from equipartition: KE_target = 0.5 * n_dof * kB * T.
  real ke_target = real{0.5} * static_cast<real>(n_dof_) * kBoltzmann *
                   cfg_.t_target;

  // Update chain masses (in case T_target changed).
  real kt = kBoltzmann * cfg_.t_target;
  real tau2 = cfg_.t_period * cfg_.t_period;
  eta_m_[0] = static_cast<real>(n_dof_) * kt * tau2;
  for (i32 i = 1; i < M; ++i) {
    eta_m_[static_cast<std::size_t>(i)] = kt * tau2;
  }

  // Use Yoshida-Suzuki sub-stepping for accuracy.
  // LAMMPS uses nrespa=1 by default for NHC; we do the same (single sub-step).
  // Each sub-step integrates the full chain once.

  // Force on outermost thermostat (last in chain).
  auto last = static_cast<std::size_t>(M - 1);

  // Integrate chain from outermost → innermost.
  // Step 1: Update outermost thermostat velocity (half-step).
  if (M > 1) {
    // Force on thermostat M-1: G_{M-1} = (Q_{M-2} * v_{M-2}^2 - kB*T) / Q_{M-1}
    real g_last =
        (eta_m_[last - 1] * eta_v_[last - 1] * eta_v_[last - 1] -
         kBoltzmann * cfg_.t_target) /
        eta_m_[last];
    eta_v_[last] += g_last * half_dt;
  }

  // Step 2: Propagate inward.
  for (i32 k = M - 2; k >= 1; --k) {
    auto sk = static_cast<std::size_t>(k);
    real exp_factor = std::exp(-eta_v_[sk + 1] * half_dt / real{4});
    eta_v_[sk] *= exp_factor;

    real g_k = (eta_m_[sk - 1] * eta_v_[sk - 1] * eta_v_[sk - 1] -
                kBoltzmann * cfg_.t_target) /
               eta_m_[sk];
    eta_v_[sk] += g_k * half_dt;
    eta_v_[sk] *= exp_factor;
  }

  // Step 3: Update thermostat 0 (innermost, coupled to particles).
  {
    real exp_factor = (M > 1)
                          ? std::exp(-eta_v_[1] * half_dt / real{4})
                          : real{1};
    eta_v_[0] *= exp_factor;

    // Force on thermostat 0: G_0 = (2*KE - 2*KE_target) / Q_0
    real g0 = (real{2} * ke_current - real{2} * ke_target) / eta_m_[0];
    eta_v_[0] += g0 * half_dt;
    eta_v_[0] *= exp_factor;
  }

  // Step 4: The velocity scale factor is exp(-eta_v_[0] * half_dt).
  real scale = std::exp(-eta_v_[0] * half_dt);

  // Step 5: Update chain positions.
  for (i32 k = 0; k < M; ++k) {
    eta_[static_cast<std::size_t>(k)] +=
        eta_v_[static_cast<std::size_t>(k)] * half_dt;
  }

  // After scaling, the new KE will be ke_current * scale^2.
  // This is needed for the second half-step (called after the VV step).

  return scale;
}

}  // namespace tdmd::integrator
