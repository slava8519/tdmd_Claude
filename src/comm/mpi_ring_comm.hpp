// SPDX-License-Identifier: Apache-2.0
// mpi_ring_comm.hpp — MPI ring communicator for TD zone exchange.
//
// Ring topology: each rank has prev/next neighbors. Zone atom data
// (positions + velocities) flows between neighbors via non-blocking
// MPI_Isend / MPI_Irecv with host-staged GPU transfers.
#pragma once

#include <mpi.h>

#include <vector>

#include "../core/types.hpp"

namespace tdmd::comm {

/// @brief Packed zone data for MPI transfer.
///
/// Contains positions and velocities for atoms in a zone,
/// plus the zone's time_step for dependency tracking.
struct ZoneMessage {
  i32 zone_id{-1};
  i32 time_step{0};
  i32 natoms{0};
  std::vector<Vec3> positions;
  std::vector<Vec3> velocities;

  /// @brief Total byte size of the payload (excluding header).
  [[nodiscard]] std::size_t payload_bytes() const noexcept {
    return static_cast<std::size_t>(natoms) * 2 * sizeof(Vec3);
  }
};

/// @brief MPI ring communicator for TD zone data exchange.
///
/// Provides async send/recv of zone atom data between neighboring ranks.
/// Uses pinned host buffers for D2H→MPI→H2D staging.
/// Even/odd phase alternation avoids deadlock.
class MpiRingComm {
 public:
  MpiRingComm() = default;

  /// @brief Initialize the ring communicator.
  /// @param comm MPI communicator (typically MPI_COMM_WORLD).
  void init(MPI_Comm comm);

  [[nodiscard]] i32 rank() const noexcept { return rank_; }
  [[nodiscard]] i32 size() const noexcept { return size_; }
  [[nodiscard]] i32 prev_rank() const noexcept { return prev_rank_; }
  [[nodiscard]] i32 next_rank() const noexcept { return next_rank_; }

  /// @brief Begin async send of zone data to next rank.
  /// @param msg Zone message to send.
  void begin_send_to_next(const ZoneMessage& msg);

  /// @brief Begin async send of zone data to prev rank.
  /// @param msg Zone message to send.
  void begin_send_to_prev(const ZoneMessage& msg);

  /// @brief Begin async recv of zone data from prev rank.
  /// @param max_atoms Maximum atoms expected in the message.
  void begin_recv_from_prev(i32 max_atoms);

  /// @brief Begin async recv of zone data from next rank.
  /// @param max_atoms Maximum atoms expected in the message.
  void begin_recv_from_next(i32 max_atoms);

  /// @brief Test if send to next is complete.
  [[nodiscard]] bool test_send_next();

  /// @brief Test if send to prev is complete.
  [[nodiscard]] bool test_send_prev();

  /// @brief Test if recv from prev is complete.
  /// If complete, fills msg and returns true.
  [[nodiscard]] bool test_recv_prev(ZoneMessage& msg);

  /// @brief Test if recv from next is complete.
  /// If complete, fills msg and returns true.
  [[nodiscard]] bool test_recv_next(ZoneMessage& msg);

  /// @brief Barrier synchronization across all ranks.
  void barrier();

  /// @brief True if any send or recv is still in flight.
  [[nodiscard]] bool has_pending() const noexcept;

  /// @brief Wait for all pending operations to complete.
  void wait_all();

 private:
  MPI_Comm comm_{MPI_COMM_NULL};
  i32 rank_{0};
  i32 size_{1};
  i32 prev_rank_{0};
  i32 next_rank_{0};

  // MPI tags for distinguishing message types.
  static constexpr int kTagToNext = 100;
  static constexpr int kTagToPrev = 200;

  // Send/recv state for next rank.
  MPI_Request send_next_req_{MPI_REQUEST_NULL};
  MPI_Request recv_prev_req_{MPI_REQUEST_NULL};
  bool send_next_active_{false};
  bool recv_prev_active_{false};

  // Send/recv state for prev rank.
  MPI_Request send_prev_req_{MPI_REQUEST_NULL};
  MPI_Request recv_next_req_{MPI_REQUEST_NULL};
  bool send_prev_active_{false};
  bool recv_next_active_{false};

  // Flat send/recv buffers: [zone_id, time_step, natoms, pos..., vel...]
  std::vector<char> send_next_buf_;
  std::vector<char> send_prev_buf_;
  std::vector<char> recv_prev_buf_;
  std::vector<char> recv_next_buf_;

  /// Pack a ZoneMessage into a flat buffer.
  static void pack(const ZoneMessage& msg, std::vector<char>& buf);

  /// Unpack a flat buffer into a ZoneMessage.
  static void unpack(const std::vector<char>& buf, i32 nbytes, ZoneMessage& msg);

  /// Required buffer size for a given atom count.
  static std::size_t buf_size(i32 natoms);
};

}  // namespace tdmd::comm
