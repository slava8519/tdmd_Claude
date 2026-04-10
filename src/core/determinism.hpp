// SPDX-License-Identifier: Apache-2.0
// determinism.hpp — compile-time flag for bit-reproducible reductions.
//
// ADR 0010 defines a "deterministic reduction" build mode: cell-list slot
// assignment and every float/double atomicAdd-based energy accumulator are
// replaced by ID-ordered paths so that a given input produces bit-identical
// metrics across runs on the same GPU. This is a developer-only correctness
// aid for debugging precision regressions; default is OFF because the ordered
// paths are ~2x slower on the hot force kernels.
//
// The mode is selected at CMake configure time via -DTDMD_DETERMINISTIC_REDUCE=ON,
// which defines the preprocessor macro of the same name. All call sites consult
// `tdmd::kDeterministicReduce` (constexpr bool) rather than the raw macro so that
// both branches type-check in every build and dead-code elimination removes the
// inactive one.
#pragma once

namespace tdmd {

#if defined(TDMD_DETERMINISTIC_REDUCE) && TDMD_DETERMINISTIC_REDUCE
inline constexpr bool kDeterministicReduce = true;
#else
inline constexpr bool kDeterministicReduce = false;
#endif

}  // namespace tdmd
