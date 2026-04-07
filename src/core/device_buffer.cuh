// SPDX-License-Identifier: Apache-2.0
// device_buffer.cuh — RAII wrapper for CUDA device memory.
#pragma once

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "error.hpp"
#include "types.hpp"

namespace tdmd {

/// @brief Check CUDA error and abort on failure.
inline void check_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                 cudaGetErrorString(err));
    std::abort();
  }
}

#define TDMD_CUDA_CHECK(call) ::tdmd::check_cuda((call), __FILE__, __LINE__)

/// @brief RAII wrapper for CUDA device memory.
///
/// Owns its allocation. Move-only. No implicit host<->device transfers.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t n) : size_(n) {
    if (n > 0) {
      TDMD_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  // Move only.
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /// Resize. Discards old data.
  void resize(std::size_t n) {
    if (n == size_) return;
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) {
      TDMD_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
  }

  /// Copy from host to device.
  void copy_from_host(const T* src, std::size_t n) {
    TDMD_CUDA_CHECK(
        cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }

  /// Copy from device to host.
  void copy_to_host(T* dst, std::size_t n) const {
    TDMD_CUDA_CHECK(
        cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
  }

  /// Set all bytes to zero.
  void zero() {
    if (ptr_ && size_ > 0) {
      TDMD_CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }

 private:
  T* ptr_{nullptr};
  std::size_t size_{0};
};

}  // namespace tdmd
