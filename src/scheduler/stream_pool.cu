// SPDX-License-Identifier: Apache-2.0
// stream_pool.cu — CUDA stream pool implementation.

#include "stream_pool.cuh"

#include "../core/device_buffer.cuh"

namespace tdmd::scheduler {

StreamPool::StreamPool(i32 n_streams) {
  auto n = static_cast<std::size_t>(n_streams);
  streams_.resize(n);
  events_.resize(n);
  busy_.assign(n, false);

  for (std::size_t i = 0; i < n; ++i) {
    TDMD_CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    TDMD_CUDA_CHECK(
        cudaEventCreateWithFlags(&events_[i], cudaEventDisableTiming));
  }
}

StreamPool::~StreamPool() {
  for (auto& e : events_) cudaEventDestroy(e);
  for (auto& s : streams_) cudaStreamDestroy(s);
}

i32 StreamPool::try_acquire() {
  // First check if any busy stream has completed.
  for (std::size_t i = 0; i < streams_.size(); ++i) {
    if (busy_[i]) {
      if (cudaEventQuery(events_[i]) == cudaSuccess) {
        busy_[i] = false;
      }
    }
  }
  // Find a free stream.
  for (std::size_t i = 0; i < streams_.size(); ++i) {
    if (!busy_[i]) {
      busy_[i] = true;
      return static_cast<i32>(i);
    }
  }
  return -1;
}

void StreamPool::record_event(i32 stream_id) {
  auto si = static_cast<std::size_t>(stream_id);
  TDMD_CUDA_CHECK(cudaEventRecord(events_[si], streams_[si]));
  busy_[si] = true;
}

bool StreamPool::is_complete(i32 stream_id) const {
  auto si = static_cast<std::size_t>(stream_id);
  return cudaEventQuery(events_[si]) == cudaSuccess;
}

void StreamPool::release(i32 stream_id) {
  busy_[static_cast<std::size_t>(stream_id)] = false;
}

i32 StreamPool::n_busy() const noexcept {
  i32 count = 0;
  for (auto b : busy_) {
    if (b) ++count;
  }
  return count;
}

}  // namespace tdmd::scheduler
