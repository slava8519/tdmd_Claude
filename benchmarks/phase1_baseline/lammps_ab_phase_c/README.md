# Phase C LAMMPS A/B sweep

One-off comparison between TDMD (`bench_pipeline_scheduler`) and stock
LAMMPS-GPU (stable 2Aug2023u3, GPU package in mixed precision) on the
same Morse and EAM workloads. Report lives at
`docs/05-benchmarks/lammps-ab-results.md`.

## Files

- `morse_{tiny,small,medium}.in` — LAMMPS inputs for Morse, matching
  TDMD's Morse benchmark parameters exactly (D=0.3429 α=1.3588 r0=2.866,
  rc=6.0, skin 1.0, rebuild every 10, seed 42, T=300 K, dt=0.001).
- `eam_{tiny,small,medium}.in` — LAMMPS inputs for EAM/alloy with
  `tests/data/Cu_mishin1.eam.alloy`, same physics settings as Morse.
- `run_ab_sweep.sh` — interleaved sequential driver. Runs TDMD 3× then
  LAMMPS 3× per cell, with 7-second pauses per CLAUDE.md §4. Writes
  everything to `results_<timestamp>/`.
- `parse_results.py` — aggregates a `results_<ts>/` dir into the
  Markdown table used in the report.

## How to reproduce

```bash
./run_ab_sweep.sh
python3 parse_results.py results_<ts>
```

Expect ~6 minutes for all 36 runs (6 cells × 3 TDMD + 3 LAMMPS,
with 7-second pauses between runs).

## Protocol invariants

- TDMD and LAMMPS runs for the same cell are interleaved within the
  same GPU session — this cancels the cross-session thermal/clock
  drift that we flagged in `feat-eam-pipeline-results.md`. The ratios
  in the report are apples-to-apples because the two codes ran
  moments apart on the same silicon state.
- LAMMPS timings use the *second* `Loop time of ... for N steps` line
  in each `.log` — the first is the 10 % warmup block, the second is
  after `reset_timestep 0`.
- TDMD timings use `timesteps_per_s` from each `.json` — already the
  measured-block-only number from `bench_pipeline_scheduler`.
- Each cell is median-of-3; raw per-run numbers are preserved in the
  report so spread can be audited.
