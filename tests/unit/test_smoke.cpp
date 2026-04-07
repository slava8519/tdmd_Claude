// SPDX-License-Identifier: Apache-2.0
// test_smoke.cpp — minimal smoke test for M0.
//
// At M1 we switch to a real test framework (Catch2 or doctest). This is enough
// to prove the build system actually links and runs a binary.

#include <cstdio>
#include <cstdlib>

#include "core/box.hpp"
#include "core/error.hpp"
#include "core/system_state.hpp"
#include "core/types.hpp"
#include "scheduler/zone.hpp"

// Tiny in-tree test harness. M0 only.
#define TDMD_TEST(cond)                                              \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::fprintf(stderr, "FAIL: %s at %s:%d\n", #cond, __FILE__,   \
                   __LINE__);                                        \
      return EXIT_FAILURE;                                           \
    }                                                                \
  } while (0)

int main() {
  using namespace tdmd;

  // types compile and have expected size relations
  TDMD_TEST(sizeof(Vec3) == 3 * sizeof(real));

  // SystemState resize is consistent
  SystemState s;
  s.resize(10);
  TDMD_TEST(s.natoms == 10);
  TDMD_TEST(s.positions.size() == 10);
  TDMD_TEST(s.velocities.size() == 10);
  TDMD_TEST(s.forces.size() == 10);
  TDMD_TEST(s.ids.size() == 10);

  // Box edge length
  Box b;
  b.lo = {0, 0, 0};
  b.hi = {10, 20, 30};
  const auto sz = b.size();
  TDMD_TEST(sz.x == real{10});
  TDMD_TEST(sz.y == real{20});
  TDMD_TEST(sz.z == real{30});

  // Zone state machine: legal transitions
  using namespace tdmd::scheduler;
  TDMD_TEST(is_legal_transition(ZoneState::Free, ZoneState::Receiving));
  TDMD_TEST(is_legal_transition(ZoneState::Computing, ZoneState::Done));
  TDMD_TEST(is_legal_transition(ZoneState::Sending, ZoneState::Free));

  // Zone state machine: illegal transitions
  TDMD_TEST(!is_legal_transition(ZoneState::Free, ZoneState::Computing));
  TDMD_TEST(!is_legal_transition(ZoneState::Done, ZoneState::Free));
  TDMD_TEST(!is_legal_transition(ZoneState::Ready, ZoneState::Done));

  // Zone transition_to increments time_step on Computing -> Done
  Zone z;
  z.transition_to(ZoneState::Receiving);
  z.transition_to(ZoneState::Received);
  z.transition_to(ZoneState::Ready);
  z.transition_to(ZoneState::Computing);
  TDMD_TEST(z.time_step == 0);
  z.transition_to(ZoneState::Done);
  TDMD_TEST(z.time_step == 1);

  std::printf("test_smoke: OK\n");
  return EXIT_SUCCESS;
}
