# Architecture: VerifyLab Integration

> How VerifyLab plugs into the rest of TDMD. Read `verifylab/README.md` for the user-facing description.

## VerifyLab is not a library

VerifyLab is a directory of scenarios + a Python runner. It does not link against TDMD code. It treats TDMD as a black-box executable.

This is intentional:

- VerifyLab tests must survive TDMD refactors. If we link, internal changes break VerifyLab — wrong direction.
- VerifyLab tests should run the same way as a user would run TDMD: input file in, dump file + log out.
- VerifyLab tests must be reproducible by anyone with a built TDMD binary, even outside the development environment.

## How a case talks to TDMD

1. Case directory contains an `input/input.in` file.
2. Runner invokes `./build/bin/tdmd input/input.in --log run.log --dump run.dump`.
3. TDMD reads the input, runs, writes log + dump.
4. Runner's `check.py` reads `run.log` / `run.dump` and computes the comparison.

## What TDMD must expose for VerifyLab

- A stable command-line interface (changes go through ADRs).
- A log format that's machine-readable (LAMMPS-style thermo lines work).
- A dump format that's machine-readable (LAMMPS-style `dump custom`).
- A `--deterministic` flag that forces FP64 + single stream + sorted reductions.
- A `--seed N` flag for reproducible velocity initialization.
- A non-zero exit code on any error.

## What VerifyLab does not test

- Internal data structures.
- Private functions.
- Code paths that don't affect physics output.

Those are the job of `tests/unit/`.

## CI integration

A subset of fast VerifyLab cases (< 5 minutes total) runs on every push. Slow cases run nightly via `.github/workflows/verifylab.yml`. The split is configured per-case via `tolerance.toml`'s `slow = true` field.

## When to add a VerifyLab case vs a unit test

| Question | Answer |
|---|---|
| Does it check physics correctness? | VerifyLab |
| Does it check that a function returns the right value? | unit test |
| Does it take more than 30 seconds? | not unit test |
| Does it need LAMMPS as a reference? | VerifyLab |
| Does it need a multi-rank MPI run? | VerifyLab (or `tests/integration/`) |
| Does it check for bit-identical output across modes? | VerifyLab |
| Does it check that an error is thrown for bad input? | unit test |
