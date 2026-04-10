// SPDX-License-Identifier: Apache-2.0
// device_morse_zone.cuh — Per-zone GPU Morse force kernel.
#pragma once

#include <cuda_runtime.h>

#include "../core/box.hpp"
#include "../core/types.hpp"
#include "device_morse.cuh"

namespace tdmd::potentials {

/// @brief Compute Morse forces for atoms in [first, first+count) only.
///
/// Uses the global neighbor list but only processes the specified atom range.
/// Forces are accumulated (not zeroed) — caller must zero the zone's forces.
void compute_morse_gpu_zone(const PositionVec* d_positions,
                            ForceVec* d_forces, const i32* d_neighbors,
                            const i32* d_offsets, const i32* d_counts,
                            i32 first_atom, i32 atom_count, const Box& box,
                            const MorseParams& params, accum_t* d_energy,
                            cudaStream_t stream);

}  // namespace tdmd::potentials
