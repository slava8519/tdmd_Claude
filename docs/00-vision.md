# 00 — Vision

> **One-page statement of what TDMD is and why it exists.**

## The problem

Modern molecular dynamics is dominated by a handful of excellent codes — LAMMPS, GROMACS, HOOMD, NAMD, OpenMM. All of them parallelize the same way: they split **space** across processors and exchange ghost atoms on halo boundaries every step. This works well, but it has two structural limits:

1. **Communication is in the critical path of every step.** The halo exchange bandwidth is a hard ceiling, especially on cheap GPU clusters without NVLink.
2. **Many-body potentials make it worse.** EAM, 3-body, and ML potentials spend a lot of work gathering neighbor data; splitting space and then gathering across the split is expensive.

The dissertation that anchors this project proposed a different answer back in the mid-2000s: **Time Decomposition.** Don't split space within a step — split time across steps. Different regions of the model can advance different integration steps simultaneously, as long as they are separated by more than one cutoff distance (which they almost always are, since the cutoff is 5–10 Å and the model is hundreds of Å across).

The idea got a rigorous treatment in the dissertation but never a modern GPU implementation.

## The opportunity

Time Decomposition has properties that map unusually well to modern hardware:

- **Ring topology** means each GPU talks to only two neighbors. That's a near-perfect fit for point-to-point MPI and device-to-device links.
- **Multi-step batching** (parameter K) reduces communication bandwidth linearly without changing the model size. No other MD method has this knob.
- **No intra-step spatial split** means many-body potentials stay local per GPU. This is exactly where the mainstream spatial-decomposition codes struggle most.

On a modern GPU like the RTX 5080, and on clusters built from cheap commodity GPUs, we expect TDMD to beat spatial-decomposition codes on metal/alloy simulations with EAM and ML potentials — especially at the awkward "medium-size, high-bandwidth-pressure" regime where spatial decomposition codes flatline.

## What we are building

A molecular dynamics engine that:

1. **Implements the Time Decomposition method faithfully** to the dissertation, with a clean space-time DAG scheduler, zone state machine, and ring communication.
2. **Runs on NVIDIA GPUs** as the primary target, with CUDA kernels for Morse and EAM, and a plugin interface for ML potentials.
3. **Combines TD with spatial decomposition** in a 2D `time × space` parallel scheme, because pure TD has a ceiling on processor count that we need to scale past.
4. **Is LAMMPS-compatible on inputs and outputs** — same data files, same potential files, same dump format, same thermo breakdown — so validation is A/B-direct.
5. **Has a dedicated physics validation subsystem** (VerifyLab) that tests energy/momentum conservation, distribution statistics, and LAMMPS agreement automatically and continuously.
6. **Is written to be read.** The codebase will be built by a solo developer with AI assistance. Readability, testability, and small modules are non-negotiable.

## What we are not building (yet)

- General-purpose MD with every potential under the sun (bonded, Coulomb long-range, polarizable, etc.). The first focus is **metals and alloys** with EAM.
- Biomolecular MD (GROMACS territory).
- A CPU-first code. CPU support exists as a correctness reference and for small debug runs.
- A replacement for LAMMPS. LAMMPS is our partner, not our target.

## Why now

Three things happened that make this project worth doing in 2026:

1. **GPUs got much faster than their interconnects.** Compute-to-bandwidth ratios now favor methods that reduce communication, which is exactly what TD does.
2. **Many-body and ML potentials became the norm for materials science.** These are the potentials where TD has the biggest advantage.
3. **AI coding agents became capable enough to build something this ambitious with a solo developer.** Five years ago this project would have needed a team of five for three years. Now it can be built by one person with Claude Code.

## The measure of success

We will consider TDMD successful when:

- **Correctness:** VerifyLab shows agreement with LAMMPS on FCC16 and Ni-Co-Cr benchmarks within FP tolerance, across NVE/NVT/NPT ensembles.
- **Speed:** on EAM-alloy benchmarks on an RTX 5080, TDMD is within 1.5× of LAMMPS-GPU on single GPU, and faster on 4+ GPUs when bandwidth-bound.
- **Readability:** a new developer (or the same developer, six months later) can understand any module in under an hour by reading its header and its doc.
- **Usability:** a user can take a LAMMPS input file, change the executable name, and run — for the subset of features we support.

That is the bar. Everything in `docs/03-roadmap/` is there to reach it.
