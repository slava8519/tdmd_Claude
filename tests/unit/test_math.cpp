// SPDX-License-Identifier: Apache-2.0
// test_math.cpp — unit tests for core/math.hpp
#include <gtest/gtest.h>

#include <cmath>

#include "core/math.hpp"

using namespace tdmd;

TEST(Vec3Ops, AddSubtract) {
  Vec3 a{1, 2, 3};
  Vec3 b{4, 5, 6};
  Vec3 c = a + b;
  EXPECT_DOUBLE_EQ(c.x, 5);
  EXPECT_DOUBLE_EQ(c.y, 7);
  EXPECT_DOUBLE_EQ(c.z, 9);

  Vec3 d = a - b;
  EXPECT_DOUBLE_EQ(d.x, -3);
  EXPECT_DOUBLE_EQ(d.y, -3);
  EXPECT_DOUBLE_EQ(d.z, -3);
}

TEST(Vec3Ops, Scale) {
  Vec3 v{2, 3, 4};
  Vec3 s1 = real{2} * v;
  EXPECT_DOUBLE_EQ(s1.x, 4);
  EXPECT_DOUBLE_EQ(s1.y, 6);
  EXPECT_DOUBLE_EQ(s1.z, 8);

  Vec3 s2 = v * real{0.5};
  EXPECT_DOUBLE_EQ(s2.x, 1);
  EXPECT_DOUBLE_EQ(s2.y, 1.5);
  EXPECT_DOUBLE_EQ(s2.z, 2);
}

TEST(Vec3Ops, DotProduct) {
  Vec3 a{1, 0, 0};
  Vec3 b{0, 1, 0};
  EXPECT_DOUBLE_EQ(dot(a, b), 0);

  Vec3 c{1, 2, 3};
  Vec3 d{4, 5, 6};
  EXPECT_DOUBLE_EQ(dot(c, d), 32);  // 4+10+18
}

TEST(Vec3Ops, Length) {
  Vec3 v{3, 4, 0};
  EXPECT_DOUBLE_EQ(length_sq(v), 25);
  EXPECT_DOUBLE_EQ(length(v), 5);
}

TEST(Vec3Ops, PlusEqualsMinusEquals) {
  Vec3 a{1, 2, 3};
  Vec3 b{10, 20, 30};
  a += b;
  EXPECT_DOUBLE_EQ(a.x, 11);
  EXPECT_DOUBLE_EQ(a.y, 22);
  EXPECT_DOUBLE_EQ(a.z, 33);

  a -= b;
  EXPECT_DOUBLE_EQ(a.x, 1);
  EXPECT_DOUBLE_EQ(a.y, 2);
  EXPECT_DOUBLE_EQ(a.z, 3);
}

TEST(MinimumImage, NoWrap) {
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, true, true};
  Vec3 delta{2, -3, 4};
  Vec3 result = minimum_image(delta, box_size, pbc);
  EXPECT_DOUBLE_EQ(result.x, 2);
  EXPECT_DOUBLE_EQ(result.y, -3);
  EXPECT_DOUBLE_EQ(result.z, 4);
}

TEST(MinimumImage, WrapPositive) {
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, true, true};
  Vec3 delta{7, 0, 0};  // > 5 -> should wrap to -3
  Vec3 result = minimum_image(delta, box_size, pbc);
  EXPECT_DOUBLE_EQ(result.x, -3);
}

TEST(MinimumImage, WrapNegative) {
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, true, true};
  Vec3 delta{-8, 0, 0};  // < -5 -> should wrap to +2
  Vec3 result = minimum_image(delta, box_size, pbc);
  EXPECT_DOUBLE_EQ(result.x, 2);
}

TEST(MinimumImage, NonPeriodicAxis) {
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, false, true};
  Vec3 delta{7, 8, -8};
  Vec3 result = minimum_image(delta, box_size, pbc);
  EXPECT_DOUBLE_EQ(result.x, -3);
  EXPECT_DOUBLE_EQ(result.y, 8);   // not wrapped
  EXPECT_DOUBLE_EQ(result.z, 2);
}

TEST(WrapPosition, InsideBox) {
  Vec3 lo{0, 0, 0};
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, true, true};
  Vec3 pos{3, 5, 7};
  Vec3 result = wrap_position(pos, lo, box_size, pbc);
  EXPECT_NEAR(result.x, 3, 1e-12);
  EXPECT_NEAR(result.y, 5, 1e-12);
  EXPECT_NEAR(result.z, 7, 1e-12);
}

TEST(WrapPosition, OutsidePositive) {
  Vec3 lo{0, 0, 0};
  Vec3 box_size{10, 10, 10};
  std::array<bool, 3> pbc{true, true, true};
  Vec3 pos{12, 25, -3};
  Vec3 result = wrap_position(pos, lo, box_size, pbc);
  EXPECT_NEAR(result.x, 2, 1e-12);
  EXPECT_NEAR(result.y, 5, 1e-12);
  EXPECT_NEAR(result.z, 7, 1e-12);
}
