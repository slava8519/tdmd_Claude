# Workflow: Investigate performance

Use this only at M7 or later, and only when there's a measured slowdown to investigate.

## Steps

1. **Have a baseline.** What's the current `timesteps/s` on FCC16 medium? Note it.
2. **Have a target.** What do you want it to be? Note it.
3. **Profile.** `nsys profile` on a representative run. Look at:
   - Top kernels by time.
   - Memory throughput vs theoretical.
   - GPU occupancy.
   - CPU↔GPU transfer time.
4. **Identify the bottleneck.** Compute-bound, bandwidth-bound, latency-bound, host-bound?
5. **Form a hypothesis** about why the bottleneck exists.
6. **Try one change.** Just one. Measure.
7. If it helped: keep it, **but** verify VerifyLab is still green. If it didn't: revert.
8. **Document the experiment** in `docs/05-benchmarks/perf-log.md` with: change, before, after, conclusion.
