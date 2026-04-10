// SPDX-License-Identifier: Apache-2.0
// types.hpp — fundamental scalar and vector types used everywhere in TDMD.
#pragma once

#include <array>
#include <cstdint>

namespace tdmd {

// ---- precision selection ----
#if defined(TDMD_PRECISION_FP64)
using real = double;
#elif defined(TDMD_PRECISION_MIXED)
using real = float;
#else
#error "Define exactly one of TDMD_PRECISION_MIXED or TDMD_PRECISION_FP64"
#endif

// ---- aliases ----
using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ============================================================================
// Type role aliases for mixed precision (ADR 0007)
//
// Mixed mode: pos_t/vel_t = double, force_t = float, accum_t = double
// FP64 mode:  all = double
//
// New code SHOULD prefer role aliases over bare `real`. Old code still
// using `real` will be migrated file by file.
// ============================================================================

// Layer 1: precision-explicit vector type (template).
// Aggregate POD — works on host and device without ceremony.
template<class T>
struct Vec3T {
  T x;
  T y;
  T z;
};

using Vec3D = Vec3T<double>;
using Vec3F = Vec3T<float>;

// ---- minimal vec3 (legacy alias for build-mode precision) ----
// Vec3 is now a template alias over `real`. In FP64 mode Vec3 == Vec3D.
// Aggregate initialization `Vec3{x, y, z}` continues to work because
// Vec3T<T> is itself an aggregate with public x/y/z members.
using Vec3 = Vec3T<real>;

// ---- axis-aligned bounding box ----
// Geometry in double regardless of build mode (ADR 0007).
struct Aabb {
  Vec3D lo;
  Vec3D hi;
};

// Layer 2: role aliases (semantic naming).
// Mixed mode: positions/velocities in double, forces in float (LAMMPS-style).
// FP64 mode: everything double (reference/validation).
#if defined(TDMD_PRECISION_MIXED)
using pos_t      = double;    // position storage type
using vel_t      = double;    // velocity storage type
using force_t    = float;     // force storage type (hot path)
using accum_t    = double;    // reduction accumulators — always double
using PositionVec = Vec3D;
using VelocityVec = Vec3D;
using ForceVec    = Vec3F;
#elif defined(TDMD_PRECISION_FP64)
using pos_t      = double;    // position storage type
using vel_t      = double;    // velocity storage type
using force_t    = double;    // force storage type
using accum_t    = double;    // reduction accumulators — always double
using PositionVec = Vec3D;
using VelocityVec = Vec3D;
using ForceVec    = Vec3D;
#endif

}  // namespace tdmd
