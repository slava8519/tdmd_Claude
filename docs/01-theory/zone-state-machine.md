# Zone State Machine — Formal Specification

> This is the **executable contract** for the Zone state machine.
> Any code that touches a zone's state must respect these transitions.
> All assertions in this document MUST be encoded as runtime checks in `src/scheduler/zone.hpp` and as property tests in `tests/unit/test_zone_state_machine.cpp`.

---

## States

```
enum class ZoneState : uint8_t {
  Free,        // 0 — empty, no atom data, available
  Receiving,   // 1 — recv posted, data in flight
  Received,    // 2 — data arrived, dependencies not yet checked
  Ready,       // 3 — dependencies satisfied, can launch work
  Computing,   // 4 — kernel running on GPU
  Done,        // 5 — kernel finished, results valid for time_step T
  Sending,     // 6 — send posted, data in flight to next rank
};
```

## Allowed transitions

```
Free       → Receiving                  : recv_post(zone)
Receiving  → Received                   : recv_complete(zone)
Received   → Ready                      : check_deps(zone) returns satisfied
Ready      → Computing                  : launch_kernel(zone)
Computing  → Done                       : kernel_complete(zone), time_step += 1
Done       → Sending                    : send_post(zone)
Sending    → Free                       : send_complete(zone)
```

**No other transition is legal.** Any attempt to transition outside this set MUST trigger `TDMD_ASSERT(false, "illegal zone transition")`.

## Invariants (must hold at all times)

### I-1: Time monotonicity
For every zone Z, `Z.time_step` is monotonically non-decreasing through the lifecycle. The only state transition that increments `time_step` is `Computing → Done`.

### I-2: Causal dependency
A zone Z can transition to `Computing` at time step T only if every zone Z' such that `dist(Z, Z') ≤ r_c + r_skin + r_buf` has `Z'.time_step ≥ T - 1` and `Z'.state ∈ {Received, Ready, Computing, Done}`.

### I-3: No double computing
At most one zone can be in `Computing` state per CUDA stream. Multiple Computing zones across different streams are allowed if and only if they have disjoint atom sets and disjoint neighbor influence regions.

### I-4: Send safety
A zone can transition to `Sending` only if:
- Its state is `Done`.
- All neighbor zones on this rank have `time_step ≥ Z.time_step`.
- No pending kernel reads from this zone.

### I-5: Free safety
A zone can transition to `Free` only after a `Sending` is acknowledged or the zone was never Receiving (initial state).

### I-6: Single ownership
At any wall-clock moment, a zone is owned by exactly one rank. Ownership transfer happens atomically via send/recv pairing. There is no "borrowed" state.

### I-7: Skin invariant
For every zone Z in any state other than `Free`, no atom in Z has moved more than `r_skin / 2` since the last neighbor list rebuild. Violation triggers an immediate rebuild.

### I-8: Same global order
All ranks walk zones in the same order. The order is fixed at startup and does not change during a run.

## Events that drive transitions

| Event | Source | Effect |
|---|---|---|
| `recv_post(z)` | scheduler | `z.state = Free → Receiving`; posts MPI_Irecv |
| `recv_complete(z)` | comm poll | `z.state = Receiving → Received`; data is now valid |
| `check_deps(z)` | scheduler | if I-2 holds, `z.state = Received → Ready` |
| `launch_kernel(z)` | scheduler | `z.state = Ready → Computing`; cuda_event_record |
| `kernel_complete(z)` | event poll | `z.state = Computing → Done`; `z.time_step++` |
| `send_post(z)` | scheduler | `z.state = Done → Sending`; posts MPI_Isend |
| `send_complete(z)` | comm poll | `z.state = Sending → Free`; data is no longer needed here |
| `force_rebuild()` | neighbor list | invalidates all skin invariants; triggers list rebuild |

## Observable counters

Every state transition increments a per-state counter for telemetry:

```
counters.transitions[from][to] += 1
counters.time_in_state[state] += elapsed
```

These counters are visible to VerifyLab and to the perf-breakdown printer. They are how we observe pipeline health.

## Failure modes and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| Many zones stuck in `Received` | dependency check failing | check `r_c + r_skin + r_buf` calculation |
| Many zones stuck in `Ready` | not enough CUDA streams | increase stream pool size |
| Many zones stuck in `Done` | downstream rank slow | check pipeline balance, K parameter |
| Many zones stuck in `Sending` | comm bandwidth bound | increase K, check GPU-aware MPI |
| Frequent skin violations | `r_skin` too small for `v_max·Δt` | increase `r_skin` or decrease `Δt` |
| Different time_step ordering on different ranks | I-8 violated | bug — order must be deterministic |

## Reference C++ skeleton

```cpp
namespace tdmd::scheduler {

enum class ZoneState : uint8_t { Free, Receiving, Received, Ready, Computing, Done, Sending };

struct Zone {
  int32_t id;
  std::array<int32_t, 3> lattice_index;
  int32_t time_step{0};
  ZoneState state{ZoneState::Free};
  int32_t owner_rank{-1};
  // ...

  void transition_to(ZoneState new_state) {
    TDMD_ASSERT(is_legal_transition(state, new_state),
                "illegal zone transition");
    state = new_state;
    if (new_state == ZoneState::Done) {
      ++time_step;
    }
  }
};

constexpr bool is_legal_transition(ZoneState from, ZoneState to) {
  switch (from) {
    case ZoneState::Free:      return to == ZoneState::Receiving;
    case ZoneState::Receiving: return to == ZoneState::Received;
    case ZoneState::Received:  return to == ZoneState::Ready;
    case ZoneState::Ready:     return to == ZoneState::Computing;
    case ZoneState::Computing: return to == ZoneState::Done;
    case ZoneState::Done:      return to == ZoneState::Sending;
    case ZoneState::Sending:   return to == ZoneState::Free;
  }
  return false;
}

} // namespace tdmd::scheduler
```

## Tests required (M3)

1. `test_legal_transitions` — try all `ZoneState x ZoneState` pairs, assert which succeed.
2. `test_time_step_monotonic` — fuzz with random legal transitions, assert `time_step` never decreases.
3. `test_dependency_check` — set up two zones with various distance/state combos, assert check_deps returns the correct answer.
4. `test_send_safety` — assert I-4.
5. `test_skin_violation_detection` — synthesize an atom moving > r_skin/2 and assert force_rebuild is triggered.
