# OPT session benchmark results (2026-04-10)

Before / after numbers for the OPT-1 + OPT-2 optimization pass on top of
Phase 3. Scheduler: `FastPipelineScheduler`, potential: Morse (unless
noted), mixed-precision preset, RTX 5080, CUDA 12.6.

Protocol: 3 sequential runs per configuration with 6–8 s pauses between
runs (GPU-exclusive per CLAUDE.md §4 benchmark rule). All runs use
`bench_pipeline_scheduler --scheduler fast_pipeline` with default
`rebuild_every=10`, `skin=1.0`, `t_init=300K`, `seed=42`. Median of 3
reported; spread is ≤0.5 % inside each cell unless noted.

## Session baseline (before OPT-1)

Mixed preset rebuilt with `-DTDMD_BUILD_BENCHMARKS=ON`, benchmarks freshly
compiled, same input data files as the Phase 1 / Phase 2 suite.

| System  | N atoms | steps | warmup | ts/s (median of 3) |
|---------|--------:|------:|-------:|-------------------:|
| tiny    |     256 |  5000 |    500 |            10 005  |
| small   |   4 000 |  2000 |    200 |             5 964  |
| medium  |  32 000 |  1000 |    100 |             4 026  |

Note: these numbers are lower than the Phase 2 / Phase 3 historical
figures in `docs/03-roadmap/current_milestone.md` (14 814 / 9 139 / 6 927
ts/s). The delta is not investigated in this session — every comparison
below is apples-to-apples against the same baseline, so the OPT deltas
are unaffected by whatever drifted between the historical runs and this
session's environment.

## After OPT-1 — device-resident neighbor-list prefix sum

Replaces the `D2H counts → CPU prefix sum → H2D offsets` flow inside
`DeviceNeighborList::build()` with three chained CUB operations
(`DeviceScan::ExclusiveSum`, `DeviceReduce::Max`, tiny `pack_meta_kernel`)
and one 8-byte D2H. The old path had **two** `cudaStreamSynchronize`
calls plus ~256 KB PCIe round-trip per rebuild on `medium`. The new path
has one tiny sync and 0 bus traffic for the prefix sum itself.

| System  | ts/s (before) | ts/s (after OPT-1) | Δ      |
|---------|--------------:|-------------------:|-------:|
| tiny    |        10 005 |              9 981 | −0.2 % |
| small   |         5 964 |              5 950 | −0.2 % |
| medium  |         4 026 |              4 070 | +1.1 % |

### Why the delta is smaller than the original estimate

Earlier scouting estimated +10–15 % on small and +5 % on medium based on
ADR 0005's "nlist rebuild = 41.8 % of GPU time on 4K atoms" figure. That
estimate did not hold for two reasons:

1. **The host prefix sum was not on the critical path.** The two
   `cudaStreamSynchronize` calls inside the old build() only blocked the
   *CPU* briefly — the scheduler's hot loop is single-stream, and the GPU
   queue already serializes kernels behind `build()` even without those
   syncs. Removing them frees CPU queuing time, which is mostly hidden
   behind in-stream GPU work. The 256 KB PCIe traffic is also dwarfed by
   the force kernel's shared-memory traffic.
2. **CUB launches eat most of the savings.** The new path issues three
   device-side launches per rebuild (exclusive scan, reduce-max, tiny
   pack-meta), and CUB scan internally uses two small kernels. Each
   launch pays ~2–3 µs. Across 100 rebuilds in a 1000-step run on
   medium, the launch overhead is ~1–1.5 ms out of a ~50 ms savings
   ceiling from eliminating PCIe + CPU scan.

Net effect on `medium` is +1.1 % — positive but modest. `tiny` and
`small` are noise-level. The real value of OPT-1 is architectural, not
throughput:

- **Zero PCIe bus traffic during nlist rebuild** (previously 256 KB on
  medium). Matters more when PCIe is under contention.
- **Removes the last host round-trip from the scheduler hot path**,
  simplifying the scheduler's async story (the comment at
  `fast_pipeline_scheduler.cu:82` that warned about the sync has been
  retired).
- **Unblocks bigger future wins.** A follow-up "grow-on-overflow"
  pattern could eliminate even the 8-byte sync entirely. Pair-list
  form (store dx/dy/dz in CSR entries) becomes cleaner once the scan
  is already device-resident.

Correctness: `ctest --preset mixed` → 85/85 passed (4 deterministic-mode
tests correctly skipped). `ctest --preset fp64` → same. Neighbor-list
specific tests (`DeviceNeighborList.*`, `DeviceCellList.*`) all green.
