// SPDX-License-Identifier: Apache-2.0
// stream_pool.cuh — CUDA stream pool for the TD pipeline.
#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "../core/device_buffer.cuh"
#include "../core/types.hpp"

namespace tdmd::scheduler {

/// @brief Pool of CUDA streams with event-based completion tracking.
///
/// Each stream has an associated event. A stream is "busy" if its event
/// hasn't completed yet. The pool allows acquiring a free stream and
/// releasing it after recording an event.
class StreamPool {
 public:
  explicit StreamPool(i32 n_streams);
  ~StreamPool();

  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  /// @brief Try to acquire a free stream. Returns -1 if all busy.
  [[nodiscard]] i32 try_acquire();

  /// @brief Record an event on the stream and mark it as in-use.
  void record_event(i32 stream_id);

  /// @brief Check if the event on stream_id has completed.
  [[nodiscard]] bool is_complete(i32 stream_id) const;

  /// @brief Release a stream (mark as free).
  void release(i32 stream_id);

  /// @brief Get the raw CUDA stream.
  [[nodiscard]] cudaStream_t stream(i32 stream_id) const {
    return streams_[static_cast<std::size_t>(stream_id)];
  }

  [[nodiscard]] i32 n_total() const noexcept {
    return static_cast<i32>(streams_.size());
  }
  [[nodiscard]] i32 n_busy() const noexcept;

 private:
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;
  std::vector<bool> busy_;
};

}  // namespace tdmd::scheduler
