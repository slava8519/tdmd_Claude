# 0001 — C++20 + CUDA + CMake/Ninja as the implementation stack

- **Status:** Accepted
- **Date:** 2026-04-07
- **Decider:** human + architect
- **Affected milestone(s):** M0–all

## Context

We need to pick a language and tooling stack for TDMD. Constraints:

- Target NVIDIA GPUs (RTX 5080 dev box, future commodity GPU clusters).
- Solo developer who is not a strong C++ programmer.
- Long horizon (months to years), with AI assistance throughout.
- Performance matters eventually, but readability matters more day to day.
- Need to interoperate with LAMMPS for validation.

## Options considered

### Option A — C++20 + CUDA + CMake (chosen)
- Pros: native CUDA support, mature tooling (clangd, Nsight, gdb), matches LAMMPS ecosystem, mainstream.
- Cons: C++ has sharp edges; the developer is not a C++ expert.

### Option B — Rust + cust / cudarc + cargo
- Pros: safer, nicer build system, better error messages.
- Cons: CUDA support in Rust is still immature; LAMMPS interop needs FFI; smaller community for HPC + GPU; higher risk for the solo developer.

### Option C — Julia + CUDA.jl
- Pros: very productive, great GPU story.
- Cons: harder to interop with LAMMPS; harder to tightly control performance; ecosystem mismatch with materials science HPC.

### Option D — Python + JAX / Triton
- Pros: very fast to prototype.
- Cons: not a path to "one of the fastest MD codes in the world." Limited control over memory layout and kernel launches.

## Decision

**C++20 + CUDA + CMake/Ninja.** This is what LAMMPS, GROMACS, HOOMD, NAMD all use. The tradeoff for sharper edges in C++ is mitigated by:

1. Strict style rules in `docs/04-development/code-style.md`.
2. AI agent doing the bulk of the typing.
3. Sanitizers in CI.
4. `TDMD_ASSERT` enabled even in Release builds.

## Consequences

- **Positive:** mainstream tooling, proven path, ecosystem alignment with LAMMPS, clear performance ceiling.
- **Negative:** C++ build complexity, template error messages, manual memory management.
- **Risks:** the solo developer + AI may write bugs the compiler can't catch — mitigation via sanitizers and VerifyLab.
- **Reversibility:** very hard. This decision is foundational and locks in the rest.

## Follow-ups

- [ ] M0: minimal CMake skeleton with C++20 standard.
- [ ] M2: enable CUDA in CMake, set `CMAKE_CUDA_ARCHITECTURES` for sm_120 (Blackwell).
- [ ] Nightly CI runs sanitizer build.
