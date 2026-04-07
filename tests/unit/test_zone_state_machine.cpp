// SPDX-License-Identifier: Apache-2.0
// test_zone_state_machine.cpp — zone state machine tests per spec.
// Covers: legal transitions, illegal transitions, time_step monotonicity.
#include <gtest/gtest.h>

#include "scheduler/zone.hpp"

using namespace tdmd;
using namespace tdmd::scheduler;

// Test 1: all legal transitions succeed.
TEST(ZoneStateMachine, LegalTransitions) {
  Zone z;
  z.id = 0;

  // Free → Receiving.
  EXPECT_EQ(z.state, ZoneState::Free);
  z.transition_to(ZoneState::Receiving);
  EXPECT_EQ(z.state, ZoneState::Receiving);

  // Receiving → Received.
  z.transition_to(ZoneState::Received);
  EXPECT_EQ(z.state, ZoneState::Received);

  // Received → Ready.
  z.transition_to(ZoneState::Ready);
  EXPECT_EQ(z.state, ZoneState::Ready);

  // Ready → Computing.
  z.transition_to(ZoneState::Computing);
  EXPECT_EQ(z.state, ZoneState::Computing);

  // Computing → Done (increments time_step).
  EXPECT_EQ(z.time_step, 0);
  z.transition_to(ZoneState::Done);
  EXPECT_EQ(z.state, ZoneState::Done);
  EXPECT_EQ(z.time_step, 1);

  // Done → Sending.
  z.transition_to(ZoneState::Sending);
  EXPECT_EQ(z.state, ZoneState::Sending);

  // Sending → Free.
  z.transition_to(ZoneState::Free);
  EXPECT_EQ(z.state, ZoneState::Free);
}

// Test 2: all illegal transitions fail (assert/abort).
TEST(ZoneStateMachine, IllegalTransitions) {
  // Enumerate all 7*7 = 49 pairs. Only 7 are legal.
  constexpr ZoneState states[] = {
      ZoneState::Free,      ZoneState::Receiving, ZoneState::Received,
      ZoneState::Ready,     ZoneState::Computing, ZoneState::Done,
      ZoneState::Sending,
  };
  constexpr int N = 7;

  // Legal transitions as (from, to) pairs.
  constexpr std::pair<ZoneState, ZoneState> legal[] = {
      {ZoneState::Free, ZoneState::Receiving},
      {ZoneState::Receiving, ZoneState::Received},
      {ZoneState::Received, ZoneState::Ready},
      {ZoneState::Ready, ZoneState::Computing},
      {ZoneState::Computing, ZoneState::Done},
      {ZoneState::Done, ZoneState::Sending},
      {ZoneState::Sending, ZoneState::Free},
  };

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      bool expected_legal = false;
      for (const auto& [from, to] : legal) {
        if (from == states[i] && to == states[j]) {
          expected_legal = true;
          break;
        }
      }
      EXPECT_EQ(is_legal_transition(states[i], states[j]), expected_legal)
          << "from=" << static_cast<int>(states[i])
          << " to=" << static_cast<int>(states[j]);
    }
  }
}

// Test 3: time_step is monotonically non-decreasing through full cycles.
TEST(ZoneStateMachine, TimeStepMonotonicity) {
  Zone z;
  z.id = 0;

  for (int cycle = 0; cycle < 100; ++cycle) {
    i32 ts_before = z.time_step;

    z.state = ZoneState::Free;
    z.transition_to(ZoneState::Receiving);
    EXPECT_GE(z.time_step, ts_before);

    z.transition_to(ZoneState::Received);
    EXPECT_GE(z.time_step, ts_before);

    z.transition_to(ZoneState::Ready);
    EXPECT_GE(z.time_step, ts_before);

    z.transition_to(ZoneState::Computing);
    EXPECT_GE(z.time_step, ts_before);

    z.transition_to(ZoneState::Done);  // increments
    EXPECT_EQ(z.time_step, ts_before + 1);

    z.transition_to(ZoneState::Sending);
    EXPECT_EQ(z.time_step, ts_before + 1);

    z.transition_to(ZoneState::Free);
    EXPECT_EQ(z.time_step, ts_before + 1);
  }
  EXPECT_EQ(z.time_step, 100);
}
