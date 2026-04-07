# LAMMPS Baseline

> How we generate and maintain the LAMMPS reference numbers.

## Why LAMMPS

LAMMPS is the de-facto standard for materials-science MD. It's:
- Open-source.
- Well-tested.
- Supported on the same hardware we target.
- Familiar to the materials community.

If TDMD agrees with LAMMPS on physical observables, we have very high confidence we're right.

## Versioning

We pin a specific LAMMPS version per release of TDMD. The version is recorded in:

- `benchmarks/lammps_version.txt` — current pinned version.
- Each `benchmarks/results/*.json` includes the LAMMPS version used for that run.

If LAMMPS releases a new stable version, we pin-bump deliberately, re-run all baselines, and compare. Any change > tolerance triggers an investigation.

## Build instructions

LAMMPS is built separately, not by our CMake. We document the exact configure flags:

```bash
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake ../cmake \
  -DPKG_GPU=on -DGPU_API=cuda -DGPU_ARCH=sm_90 \
  -DPKG_MANYBODY=on -DPKG_EXTRA-DUMP=on \
  -DCMAKE_BUILD_TYPE=Release
make -j
```

This produces `lmp` which we invoke from VerifyLab and benchmarks.

## Reference data

For each benchmark we commit:

- The LAMMPS input script (`benchmarks/<case>/lammps/in.<case>.in`).
- The reference log (`benchmarks/<case>/expected/log.lammps`).
- The reference dump for run-0 force comparison.

These files are binary-as-far-as-git-is-concerned (`.gitattributes` marks them binary) and are updated only with explicit human approval.

## Re-baselining policy

You re-baseline (regenerate the reference outputs) only when:

1. The LAMMPS version is intentionally bumped.
2. The benchmark input is intentionally changed (and the change is in an ADR).
3. A bug in the old reference is identified and proven.

You do NOT re-baseline because:
- "TDMD's numbers look slightly different and I'm sure TDMD is right." — prove it.
- "It would be easier to update the reference." — no.

## Where we deviate from LAMMPS by design

A few intentional deviations are documented in `docs/02-architecture/lammps-compatibility.md`:

- Mixed precision is the TDMD default; LAMMPS-GPU uses FP64.
- Force ordering may differ in non-deterministic mode.
- Thermo output line format is similar but not byte-identical.

These are expected and accounted for in the comparison tolerances.
