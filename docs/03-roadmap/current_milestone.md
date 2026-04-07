# Current Milestone: M3 — Zone Decomposition (in-process) ✅ COMPLETE

> **Goal:** Introduce Zone data structure and cell→zone mapping. Scheduler walks zones in order. No time decomposition yet.

## Checklist

- [x] `scheduler/zone.hpp` — Zone struct + ZoneState enum + transition logic
- [x] `domain/ZonePartition` — 1D partition along X-axis, atom-zone assignment
- [x] Zone neighbor mapping (precomputed per r_list)
- [x] `scheduler/SequentialScheduler` — walks zones linearly, global force compute
- [x] Zone state machine tests (legal/illegal transitions, time_step monotonicity)
- [x] Zone partition tests (every atom in one zone, correct bbox)
- [x] Bit-identical results vs M2 global compute
- [x] NVE conservation under zone-walked scheduler

## Exit criteria — ALL MET

- [x] Zone-walked simulation produces identical results to M2 (positions < 1e-12, forces < 1e-10).
- [x] NVE energy conservation: 1000 steps, |dE/E| < 1e-4.
- [x] Zone state machine: all 49 transition pairs tested, only 7 legal ones pass.
- [x] Zone partition: every atom in exactly one zone, atoms in correct zone bbox.
- [x] All 54 tests pass (31 CPU M1 + 15 GPU M2 + 8 M3 zone/scheduler).

## Architecture notes

- 1D zone partition along X-axis (simplest, per theory doc).
- Auto zone count: floor(Lx / r_cut), minimum 3 for PBC.
- Atoms sorted by zone (counting sort) for future data locality.
- Sequential scheduler: all zones at same time_step, global force compute.
- Zone states bypass MPI states (Free→Receiving→Received) in M3 since single-rank.

## Next: M4 — TD scheduler (full pipeline, in-process)
