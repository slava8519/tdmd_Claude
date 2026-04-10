// SPDX-License-Identifier: Apache-2.0
// device_morse.cuh — GPU Morse pair force kernel.
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/device_buffer.cuh"
#include "../core/types.hpp"

namespace tdmd::potentials {

/// @brief Morse parameters for GPU (plain struct, no vtable).
struct MorseParams {
  real d;
  real alpha;
  real r0;
  real rc;
  real rc_sq;
};

/// @brief Compute Morse pair forces on GPU using a full neighbor list.
///
/// Forces are accumulated (not zeroed) — caller must zero first.
/// Energy is halved per pair to compensate for full-list double-counting.
///
/// @param d_positions Atom positions (device).
/// @param d_forces    Atom forces to accumulate into (device).
/// @param d_neighbors Flat CSR neighbor indices (device).
/// @param d_offsets   Per-atom offsets into neighbor array (device).
/// @param d_counts    Per-atom neighbor counts (device).
/// @param natoms      Number of atoms.
/// @param box         Simulation box (host, copied to kernel args).
/// @param params      Morse parameters.
/// @param d_energy    Optional device pointer to accumulate total PE (may be nullptr).
/// @param stream      CUDA stream (default 0 = legacy default stream).
void compute_morse_gpu(const PositionVec* d_positions, ForceVec* d_forces,
                       const i32* d_neighbors, const i32* d_offsets,
                       const i32* d_counts, i32 natoms, const Box& box,
                       const MorseParams& params, accum_t* d_energy,
                       cudaStream_t stream = 0);

}  // namespace tdmd::potentials
