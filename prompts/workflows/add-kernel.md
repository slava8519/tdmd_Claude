# Workflow: Add a CUDA kernel

Use this when implementing or rewriting a CUDA kernel (potential, neighbor build, etc.).

## Steps

1. **Reference implementation first.** Write the kernel as a CPU function in the same module. Make it pass tests on a small input. This is your oracle.

2. **Port to CUDA.**
   - Use the same algorithm, same data layout (SoA).
   - Mixed precision: FP32 for force computation, FP64 for accumulation.
   - Bounds-check in debug builds with TDMD_ASSERT.
   - One kernel per .cu file. No mega-files.

3. **Numerical comparison.** Run the CPU and GPU versions on the same input. Diff the outputs.
   - FP64 mode: bit-exact or < 1e-12 relative.
   - Mixed precision: < 1e-5 relative on forces, < 1e-7 relative on energy.
   - If the diff is larger, **stop**. Find the bug.

4. **Profile.** `nsys profile` and inspect:
   - Occupancy.
   - Memory throughput vs roofline.
   - Register pressure.
   - Coalescing.

5. **Optimize ONLY if M7 or later.** Until then, leave the readable version alone.

6. **Document.**
   - Header of the .cu file: what it computes, expected layout, complexity.
   - Module doc: list this kernel and its perf characteristics.

7. **Test on the actual GPU.** RTX 5080 in this dev box. The human will help you run it.

8. **Add a VerifyLab case** comparing the GPU result to the CPU reference, and to LAMMPS where applicable.
