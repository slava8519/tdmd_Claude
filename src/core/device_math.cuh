// SPDX-License-Identifier: Apache-2.0
//
// device_math.cuh — precision-aware transcendental math intrinsics for device
// code.
//
// Avoids silent FP32 → FP64 promotion in force kernels. On consumer GPUs
// (RTX 5080 / Blackwell, sm_120), FP64 throughput is 1/32 of FP32. Standard
// C++ <cmath> overloads (sqrt, exp) accept double, so passing a float
// argument triggers implicit promotion — the entire computation runs at FP64
// speed.
//
// This header provides overloaded intrinsics that dispatch on argument type:
// float → sqrtf/expf, double → sqrt/exp. No #ifdef needed — overload
// resolution picks the right variant at compile time.

#pragma once

#include <cuda_runtime.h>

namespace tdmd::math {

__device__ __forceinline__ float sqrt_impl(float x) { return sqrtf(x); }
__device__ __forceinline__ float exp_impl(float x) { return expf(x); }

__device__ __forceinline__ double sqrt_impl(double x) { return sqrt(x); }
__device__ __forceinline__ double exp_impl(double x) { return exp(x); }

}  // namespace tdmd::math
