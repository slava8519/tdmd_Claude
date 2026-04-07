// SPDX-License-Identifier: Apache-2.0
// test_device_buffer.cu — unit tests for DeviceBuffer and DeviceSystemState.
#include <gtest/gtest.h>

#include <vector>

#include "core/device_buffer.cuh"
#include "core/device_system_state.cuh"
#include "core/types.hpp"

using namespace tdmd;

TEST(DeviceBuffer, AllocAndCopyRoundTrip) {
  constexpr std::size_t N = 128;
  DeviceBuffer<real> buf(N);
  EXPECT_EQ(buf.size(), N);
  EXPECT_NE(buf.data(), nullptr);

  // Fill host array, upload, download, compare.
  std::vector<real> host_in(N);
  for (std::size_t i = 0; i < N; ++i) {
    host_in[i] = static_cast<real>(i) * static_cast<real>(0.1);
  }
  buf.copy_from_host(host_in.data(), N);

  std::vector<real> host_out(N, static_cast<real>(-1));
  buf.copy_to_host(host_out.data(), N);

  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(host_in[i], host_out[i]) << "mismatch at i=" << i;
  }
}

TEST(DeviceBuffer, ZeroFill) {
  constexpr std::size_t N = 64;
  DeviceBuffer<int> buf(N);

  // Upload nonzero data.
  std::vector<int> ones(N, 42);
  buf.copy_from_host(ones.data(), N);

  // Zero.
  buf.zero();

  // Download and check.
  std::vector<int> out(N, -1);
  buf.copy_to_host(out.data(), N);
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(out[i], 0);
  }
}

TEST(DeviceBuffer, MoveSemantics) {
  DeviceBuffer<real> a(32);
  real* ptr = a.data();
  EXPECT_NE(ptr, nullptr);

  // Move construct.
  DeviceBuffer<real> b(std::move(a));
  EXPECT_EQ(b.data(), ptr);
  EXPECT_EQ(b.size(), 32u);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), 0u);

  // Move assign.
  DeviceBuffer<real> c;
  c = std::move(b);
  EXPECT_EQ(c.data(), ptr);
  EXPECT_EQ(c.size(), 32u);
  EXPECT_EQ(b.data(), nullptr);
}

TEST(DeviceBuffer, Resize) {
  DeviceBuffer<real> buf(16);
  EXPECT_EQ(buf.size(), 16u);

  buf.resize(64);
  EXPECT_EQ(buf.size(), 64u);
  EXPECT_NE(buf.data(), nullptr);

  buf.resize(0);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.data(), nullptr);
}

TEST(DeviceSystemState, UploadDownloadRoundTrip) {
  // Build a small host state.
  SystemState host;
  host.natoms = 4;
  host.box.lo = {0, 0, 0};
  host.box.hi = {10, 10, 10};
  host.resize(4);
  host.masses = {63.546};
  host.type_names = {"Cu"};

  for (i32 i = 0; i < 4; ++i) {
    real fi = static_cast<real>(i);
    host.positions[static_cast<std::size_t>(i)] = {fi, fi * 2, fi * 3};
    host.velocities[static_cast<std::size_t>(i)] = {fi * real{0.1}, 0, 0};
    host.forces[static_cast<std::size_t>(i)] = {fi * real{0.5}, fi * real{0.5}, 0};
    host.types[static_cast<std::size_t>(i)] = 0;
    host.ids[static_cast<std::size_t>(i)] = i + 1;
  }

  // Upload.
  DeviceSystemState dss;
  dss.upload(host);
  EXPECT_EQ(dss.natoms, 4);

  // Modify host to prove download overwrites.
  for (auto& p : host.positions) {
    p = {-1, -1, -1};
  }

  // Download.
  dss.download(host);
  EXPECT_EQ(host.natoms, 4);

  // Verify positions came back.
  for (i32 i = 0; i < 4; ++i) {
    real fi = static_cast<real>(i);
    auto& p = host.positions[static_cast<std::size_t>(i)];
    EXPECT_NEAR(p.x, fi, 1e-6);
    EXPECT_NEAR(p.y, fi * 2, 1e-6);
    EXPECT_NEAR(p.z, fi * 3, 1e-6);
  }
}

TEST(DeviceSystemState, ZeroForces) {
  SystemState host;
  host.natoms = 8;
  host.resize(8);
  host.masses = {1.0};
  for (std::size_t i = 0; i < 8; ++i) {
    host.forces[i] = {1, 2, 3};
    host.types[i] = 0;
    host.ids[i] = static_cast<i32>(i + 1);
  }

  DeviceSystemState dss;
  dss.upload(host);
  dss.zero_forces();
  dss.download(host);

  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(host.forces[i].x, 0, 1e-12);
    EXPECT_NEAR(host.forces[i].y, 0, 1e-12);
    EXPECT_NEAR(host.forces[i].z, 0, 1e-12);
  }
}
