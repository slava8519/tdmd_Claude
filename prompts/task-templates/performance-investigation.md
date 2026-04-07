# Task Template: Performance Investigation

## Title
<one line>

## Baseline
- Benchmark name
- Commit / build mode
- Current `timesteps/s` (or `atom·steps/(s·GPU)`)
- Hardware (GPU model, host CPU, MPI ranks)

## Target
- Desired metric, or "find out what's possible"

## Why
What user-visible problem does this solve?

## Constraints
- Must remain VerifyLab-green
- Must not break determinism in deterministic mode
- Must not sacrifice readability for cleverness (pre-M7)

## Plan
What's the first thing to profile or measure?
