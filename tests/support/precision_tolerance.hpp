// SPDX-License-Identifier: Apache-2.0
// precision_tolerance.hpp — Precision-aware test tolerances.
//
// Provides tolerances and macros that adjust to FP32/FP64 build mode.
// Use in NEW tests only — existing tests keep their explicit tolerances.
//
// Session 3A extends this header with placeholders for additional categories
// needed by the mixed precision contract (ADR 0007). Placeholders are
// defined but not used by any test until session 3B.
#pragma once

#include <cmath>

#include "core/types.hpp"

namespace tdmd::testing {

// =========================================================================
// Session 2 — core constants (kept as-is)
// =========================================================================

#ifdef TDMD_PRECISION_FP64
inline constexpr real kPositionTolerance = 1e-10;
inline constexpr real kVelocityTolerance = 1e-10;
inline constexpr real kForceTolerance = 1e-10;
inline constexpr real kEnergyRelativeTolerance = 1e-10;
#else
inline constexpr real kPositionTolerance = 1e-4;
inline constexpr real kVelocityTolerance = 1e-4;
inline constexpr real kForceTolerance = 1e-4;
inline constexpr real kEnergyRelativeTolerance = 1e-4;
#endif

// =========================================================================
// Session 3B — additional precision-aware constants
// =========================================================================

// Analytic comparison (closed-form solutions, e.g. Morse analytic, harmonic).
#ifdef TDMD_PRECISION_FP64
inline constexpr real kAnalyticTolerance = 1e-12;
#else
inline constexpr real kAnalyticTolerance = 1e-5;
#endif

// Time accumulation tolerance (state.time stored as real).
#ifdef TDMD_PRECISION_FP64
inline constexpr double kTimeTolerance = 1e-15;
#else
inline constexpr double kTimeTolerance = 1e-7;
#endif

// Host-vs-device reduction comparison.
// In mixed mode: host computes in float, device in accum_t (double) → ~1e-7.
// In fp64 mode: both use double → tight tolerance.
#ifdef TDMD_PRECISION_FP64
inline constexpr accum_t kReductionCrossTolerance = 1e-14;
#else
inline constexpr accum_t kReductionCrossTolerance = 1e-6;
#endif

// NVE drift tolerance for multi-rank pipeline tests.
// Mixed mode: float storage causes larger drift than fp64.
#ifdef TDMD_PRECISION_FP64
inline constexpr real kNVEDriftTolerance = real{1e-4};
#else
inline constexpr real kNVEDriftTolerance = real{5e-3};
#endif

// Per-step energy drift bounds (for long-run conservation tests).
inline constexpr double kEnergyDriftPerStepMixed = 1e-9;
inline constexpr double kEnergyDriftPerStepFP64  = 1e-15;

}  // namespace tdmd::testing

// =========================================================================
// Macros — session 2 (unchanged)
// =========================================================================

#define EXPECT_POSITION_NEAR(val1, val2)                                    \
  EXPECT_NEAR((val1), (val2), ::tdmd::testing::kPositionTolerance)

#define EXPECT_VELOCITY_NEAR(val1, val2)                                    \
  EXPECT_NEAR((val1), (val2), ::tdmd::testing::kVelocityTolerance)

#define EXPECT_FORCE_NEAR(val1, val2)                                       \
  EXPECT_NEAR((val1), (val2), ::tdmd::testing::kForceTolerance)

#define EXPECT_ENERGY_REL_NEAR(actual, expected)                            \
  do {                                                                      \
    auto _a = (actual);                                                     \
    auto _e = (expected);                                                   \
    if (std::abs(_e) > 0) {                                                 \
      EXPECT_LT(std::abs((_a - _e) / _e),                                  \
                ::tdmd::testing::kEnergyRelativeTolerance)                   \
          << "actual=" << _a << " expected=" << _e;                         \
    } else {                                                                \
      EXPECT_NEAR(_a, _e, ::tdmd::testing::kEnergyRelativeTolerance);       \
    }                                                                       \
  } while (0)

// =========================================================================
// Macros — session 3A (defined but not yet used by any test)
// =========================================================================

#define EXPECT_ANALYTIC_NEAR(actual, expected)                              \
  EXPECT_NEAR((actual), (expected), ::tdmd::testing::kAnalyticTolerance)
