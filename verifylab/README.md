# VerifyLab — Physics Validation Subsystem

> **The dedicated physical-correctness test bed for TDMD.**
> If `tests/` answers "did this function do what I told it to?", VerifyLab answers "did the physics come out right?"

---

## Why VerifyLab is separate

Unit tests in `tests/` are fast (< 30 s total), deterministic, and focused on single functions. Physics tests are different:

- They run **whole simulations**, not isolated functions.
- They take **minutes to hours**, not milliseconds.
- They require **statistical interpretation**, not equality checks.
- They compare against **LAMMPS or analytic solutions**, which means external dependencies.
- They are the **ground truth** for "is this code physically correct" — not "does this code compile."

Mixing these into `tests/` would either bloat unit tests or hide physics failures behind compile-time checks. VerifyLab is the dedicated home for "is the science right?"

---

## Layout

```
verifylab/
├── README.md                       # this file
├── specs/                          # what each kind of check verifies
│   ├── energy-conservation.md
│   ├── momentum-conservation.md
│   ├── temperature-stability.md
│   ├── maxwell-distribution.md
│   ├── ab-lammps-match.md
│   └── trajectory-comparison.md
├── runners/                        # Python scripts that run scenarios
│   ├── run_all.py
│   ├── compare_with_lammps.py
│   └── plot_results.py
└── cases/                          # individual scenarios (one folder each)
    ├── two-atoms-morse/            # M1 — analytic 2-body
    ├── run0-force-match/           # M1 — single-step A/B vs LAMMPS
    ├── nve-drift/                  # M1 — energy drift over long run
    ├── nvt-stability/              # M7 — temperature variance
    ├── npt-relaxation/             # M7 — box relaxation
    ├── msd-diffusion/              # M7 — coefficient of diffusion
    ├── fcc16-stress/               # M5+ — multi-type EAM stress test
    └── nicocr-baseline/            # M2+ — 3-component HEA sanity check
```

Each case is a self-contained folder with:

```
case-name/
├── README.md           # what this case checks, expected behavior
├── input/              # TDMD input files
│   └── input.in
├── lammps/             # LAMMPS input + reference data file
│   └── in.case
├── reference/          # baseline outputs (committed)
│   ├── lammps.log
│   └── thermo.dat
├── check.py            # the actual physics-check script
└── tolerance.toml      # numerical tolerances for this case
```

---

## How a VerifyLab run works

1. The runner starts at `verifylab/runners/run_all.py`.
2. For each enabled case:
   - Run TDMD on the input.
   - Capture outputs (thermo log, dump files, end-of-run summary).
   - Run `check.py` to compare against the reference.
   - Record result: **PASS / FAIL / WARN** with metrics.
3. Print a summary table.
4. Exit non-zero if any case failed.

Run from the project root:

```bash
./scripts/run-verifylab.sh                # all enabled cases
./scripts/run-verifylab.sh --case nve-drift   # single case
./scripts/run-verifylab.sh --update-reference # rare; refresh baselines
```

---

## What kinds of checks live here

### 1. Analytic checks (no LAMMPS needed)

- **Two-atom Morse**: forces match the analytic derivative of the Morse formula.
- **Harmonic oscillator**: a single atom in a quadratic potential, period and energy match analytic.
- **Free particle**: zero force, position evolves as `x_0 + v_0 · t`.

### 2. Conservation checks (long runs)

- **Energy in NVE**: relative drift `|ΔE/E|` over 50k–500k steps stays below threshold (typically `1e-4`).
- **Linear momentum**: `|Σ m_i v_i|` stays below `1e-8` per atom in NVE with PBC.
- **Angular momentum**: not enforced under PBC (intentional).

### 3. Statistical checks

- **Maxwell-Boltzmann velocity distribution** matches the target temperature in NVT (KS test).
- **Temperature variance** in NVT: `σ_T / ⟨T⟩ ≈ √(2/(3N))`.
- **Pressure mean and variance** in NPT.

### 4. A/B vs LAMMPS

- **Run-0 force match**: same input → same forces (within FP tolerance) → bug in our code if not.
- **Thermo trajectory match**: identical seed, same `dt`, traj agrees in distribution.
- **MSD slope match**: diffusion coefficient within 5%.

### 5. Cross-configuration checks

- **1 rank vs N ranks**: same seed, same input — trajectories must agree (in deterministic mode) or have controlled FP drift (in fast mode).
- **TD vs SD on same input**: physically equivalent results.

---

## Tolerances

Every case has a `tolerance.toml`:

```toml
[forces]
mode = "max_relative"
threshold = 1e-5

[energy_drift]
mode = "relative_per_atom_per_ps"
threshold = 1e-7

[temperature]
mode = "mean"
threshold = 1.0  # Kelvin
```

Tolerances are committed to git. Loosening a tolerance requires:
1. A clear justification in the commit message.
2. A linked issue or ADR.
3. Human approval.

---

## Failure modes and what they mean

| Failure | Likely meaning |
|---|---|
| `forces` exceeds threshold on run-0 | Bug in potential implementation, or wrong cutoff/skin |
| `energy_drift` too large | Bug in integrator, or `dt` too large for the system |
| `temperature` mean wrong | Thermostat damping wrong, or unit mismatch |
| `lammps_log_diff` too large | Either we're wrong, or LAMMPS version drifted (check) |
| `1_vs_N_ranks` differ | Race condition, non-deterministic ordering, or scheduler bug |

---

## How to add a new case

1. Create folder `verifylab/cases/<your-case>/`.
2. Write `README.md` explaining what it checks and why.
3. Add `input/input.in` for TDMD.
4. If A/B with LAMMPS: add `lammps/in.case` and a `reference/lammps.log` (committed).
5. Write `check.py` that loads outputs and asserts the physics.
6. Add `tolerance.toml`.
7. Register the case in `runners/run_all.py`.
8. Run it locally, ensure it passes.
9. Commit with message: `verifylab: add <case-name>`.

---

## Roadmap fit

| Milestone | New VerifyLab cases |
|---|---|
| M1 | two-atoms-morse, harmonic-oscillator, run0-force-match (Morse + EAM), nve-drift |
| M2 | (same as M1, but executed on GPU) |
| M3 | zone-walking-equivalence (M3 result == M2 result) |
| M4 | td-pipeline-equivalence, zone-state-machine-fuzz |
| M5 | 1-vs-N-ranks, mpi-deadlock-stress |
| M6 | 2d-parallel-equivalence |
| M7 | nvt-stability, npt-relaxation, msd-diffusion, maxwell-distribution |

VerifyLab grows with the code. By M7 it should be the most trusted thing in the repo.

---

## Running VerifyLab in CI

A subset of fast cases (< 5 min total) runs on every PR. The full suite (which can take an hour) runs nightly. Cases marked `slow=true` in their `tolerance.toml` only run nightly.

```yaml
# .github/workflows/verifylab.yml
- name: Run VerifyLab fast suite
  run: ./scripts/run-verifylab.sh --suite fast
```

---

## A note on philosophy

**VerifyLab is what makes TDMD a science project, not just a software project.** Without it, "the code runs" means nothing. With it, we have an objective, automated, continuous answer to the only question that matters: *is the simulation physically correct?*

Treat it as the most important part of the codebase.
