// SPDX-License-Identifier: Apache-2.0
// mpi_ring_comm.cpp — MPI ring communicator implementation.

#include "mpi_ring_comm.hpp"

#include <cstring>

#include "../core/error.hpp"

namespace tdmd::comm {

// Buffer layout: [i32 zone_id | i32 time_step | i32 natoms | Vec3[] pos | Vec3[] vel]
static constexpr std::size_t kHeaderBytes = 3 * sizeof(i32);

std::size_t MpiRingComm::buf_size(i32 natoms) {
  return kHeaderBytes +
         static_cast<std::size_t>(natoms) * 2 * sizeof(Vec3);
}

void MpiRingComm::pack(const ZoneMessage& msg, std::vector<char>& buf) {
  buf.resize(buf_size(msg.natoms));
  char* p = buf.data();

  std::memcpy(p, &msg.zone_id, sizeof(i32));
  p += sizeof(i32);
  std::memcpy(p, &msg.time_step, sizeof(i32));
  p += sizeof(i32);
  std::memcpy(p, &msg.natoms, sizeof(i32));
  p += sizeof(i32);

  auto n = static_cast<std::size_t>(msg.natoms);
  std::memcpy(p, msg.positions.data(), n * sizeof(Vec3));
  p += static_cast<std::ptrdiff_t>(n * sizeof(Vec3));
  std::memcpy(p, msg.velocities.data(), n * sizeof(Vec3));
}

void MpiRingComm::unpack(const std::vector<char>& buf, i32 /*nbytes*/,
                          ZoneMessage& msg) {
  const char* p = buf.data();

  std::memcpy(&msg.zone_id, p, sizeof(i32));
  p += sizeof(i32);
  std::memcpy(&msg.time_step, p, sizeof(i32));
  p += sizeof(i32);
  std::memcpy(&msg.natoms, p, sizeof(i32));
  p += sizeof(i32);

  auto n = static_cast<std::size_t>(msg.natoms);
  msg.positions.resize(n);
  msg.velocities.resize(n);
  std::memcpy(msg.positions.data(), p, n * sizeof(Vec3));
  p += static_cast<std::ptrdiff_t>(n * sizeof(Vec3));
  std::memcpy(msg.velocities.data(), p, n * sizeof(Vec3));
}

void MpiRingComm::init(MPI_Comm comm) {
  comm_ = comm;
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  // Ring neighbors with wrapping.
  prev_rank_ = (rank_ - 1 + size_) % size_;
  next_rank_ = (rank_ + 1) % size_;
}

void MpiRingComm::begin_send_to_next(const ZoneMessage& msg) {
  TDMD_ASSERT(!send_next_active_, "send to next already in flight");
  pack(msg, send_next_buf_);
  MPI_Isend(send_next_buf_.data(), static_cast<int>(send_next_buf_.size()),
            MPI_BYTE, next_rank_, kTagToNext, comm_, &send_next_req_);
  send_next_active_ = true;
}

void MpiRingComm::begin_send_to_prev(const ZoneMessage& msg) {
  TDMD_ASSERT(!send_prev_active_, "send to prev already in flight");
  pack(msg, send_prev_buf_);
  MPI_Isend(send_prev_buf_.data(), static_cast<int>(send_prev_buf_.size()),
            MPI_BYTE, prev_rank_, kTagToPrev, comm_, &send_prev_req_);
  send_prev_active_ = true;
}

void MpiRingComm::begin_recv_from_prev(i32 max_atoms) {
  TDMD_ASSERT(!recv_prev_active_, "recv from prev already in flight");
  recv_prev_buf_.resize(buf_size(max_atoms));
  MPI_Irecv(recv_prev_buf_.data(), static_cast<int>(recv_prev_buf_.size()),
             MPI_BYTE, prev_rank_, kTagToNext, comm_, &recv_prev_req_);
  recv_prev_active_ = true;
}

void MpiRingComm::begin_recv_from_next(i32 max_atoms) {
  TDMD_ASSERT(!recv_next_active_, "recv from next already in flight");
  recv_next_buf_.resize(buf_size(max_atoms));
  MPI_Irecv(recv_next_buf_.data(), static_cast<int>(recv_next_buf_.size()),
             MPI_BYTE, next_rank_, kTagToPrev, comm_, &recv_next_req_);
  recv_next_active_ = true;
}

bool MpiRingComm::test_send_next() {
  if (!send_next_active_) return true;
  int flag = 0;
  MPI_Test(&send_next_req_, &flag, MPI_STATUS_IGNORE);
  if (flag) send_next_active_ = false;
  return flag != 0;
}

bool MpiRingComm::test_send_prev() {
  if (!send_prev_active_) return true;
  int flag = 0;
  MPI_Test(&send_prev_req_, &flag, MPI_STATUS_IGNORE);
  if (flag) send_prev_active_ = false;
  return flag != 0;
}

bool MpiRingComm::test_recv_prev(ZoneMessage& msg) {
  if (!recv_prev_active_) return false;
  int flag = 0;
  MPI_Status status;
  MPI_Test(&recv_prev_req_, &flag, &status);
  if (flag) {
    int count = 0;
    MPI_Get_count(&status, MPI_BYTE, &count);
    unpack(recv_prev_buf_, count, msg);
    recv_prev_active_ = false;
    return true;
  }
  return false;
}

bool MpiRingComm::test_recv_next(ZoneMessage& msg) {
  if (!recv_next_active_) return false;
  int flag = 0;
  MPI_Status status;
  MPI_Test(&recv_next_req_, &flag, &status);
  if (flag) {
    int count = 0;
    MPI_Get_count(&status, MPI_BYTE, &count);
    unpack(recv_next_buf_, count, msg);
    recv_next_active_ = false;
    return true;
  }
  return false;
}

void MpiRingComm::barrier() { MPI_Barrier(comm_); }

bool MpiRingComm::has_pending() const noexcept {
  return send_next_active_ || send_prev_active_ || recv_prev_active_ ||
         recv_next_active_;
}

void MpiRingComm::wait_all() {
  if (send_next_active_) {
    MPI_Wait(&send_next_req_, MPI_STATUS_IGNORE);
    send_next_active_ = false;
  }
  if (send_prev_active_) {
    MPI_Wait(&send_prev_req_, MPI_STATUS_IGNORE);
    send_prev_active_ = false;
  }
  if (recv_prev_active_) {
    MPI_Wait(&recv_prev_req_, MPI_STATUS_IGNORE);
    recv_prev_active_ = false;
  }
  if (recv_next_active_) {
    MPI_Wait(&recv_next_req_, MPI_STATUS_IGNORE);
    recv_next_active_ = false;
  }
}

}  // namespace tdmd::comm
