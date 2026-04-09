# How to read Phase 1 traces

This guide explains how to extract the metrics from the summary table in CHECKLIST.md.

## Prerequisites

- `nsys` (Nsight Systems CLI) — part of CUDA Toolkit or standalone install.
- `ncu` (Nsight Compute CLI) — same.

## Extracting metrics from nsys traces

### Total kernel launches and durations

```bash
nsys stats results/<size>/trace.nsys-rep --report cuda_gpu_kern_sum
```

This prints a table of all kernels, sorted by total duration. Key columns:
- **Instances** — number of times this kernel was launched.
- **Total Time** — total GPU time across all launches.
- **Avg** — average duration per launch.

Sum the **Instances** column across all kernel rows to get **total kernel launches**.

### Kernel launches per step

```
total_kernel_launches / n_steps
```

Compare with expected value. With per-zone launches, expect `5 * n_zones` launches per step (half_kick, drift, zero_forces, morse, half_kick per zone). The `results.json` from the benchmark already reports this.

### Average kernel duration

Look at the **Avg** column in `cuda_gpu_kern_sum` for the main force kernel (the one with "morse" in the name). If average duration is < 10 us, the kernel is launch-overhead-dominated.

### Launch overhead

```bash
nsys stats results/<size>/trace.nsys-rep --report cuda_api_sum
```

Find the row for `cuLaunchKernel` (or `cudaLaunchKernel`). The **Total Time** column is total CPU time spent launching kernels. Compare to the total GPU kernel time from `cuda_gpu_kern_sum`.

```
launch_overhead_fraction = cuLaunchKernel_total_time / total_gpu_kernel_time
```

If this is > 0.5, launch overhead dominates — the per-zone anti-pattern is significant.

### Wall clock breakdown

```bash
nsys stats results/<size>/trace.nsys-rep --report cuda_api_sum
```

Shows total time in each CUDA API call. The biggest entries will be `cuLaunchKernel`, `cudaMemcpy*`, `cudaDeviceSynchronize`, `cudaEventQuery`.

## Extracting metrics from ncu profiles

### SM occupancy

Open in Nsight Compute GUI:
```bash
ncu --import results/<size>/ncu_profile.ncu-rep
```

Or use CLI:
```bash
ncu --import results/<size>/ncu_profile.ncu-rep --page details --csv
```

Look for the **Achieved Occupancy** metric. For Phase 1 we expect low occupancy due to per-zone launches (1 thread block on 84 SMs = ~1.2%).

### Memory throughput

In the ncu details, look for **Memory Throughput** (% of peak). For Morse force kernel, this should be the bottleneck. Low throughput + low occupancy = under-utilized GPU.

## Quick cheat sheet

| Metric | Command | Where to look |
|---|---|---|
| Total kernel launches | `nsys stats ... --report cuda_gpu_kern_sum` | Sum of Instances column |
| Avg kernel duration | same | Avg column for morse kernel |
| Launch overhead time | `nsys stats ... --report cuda_api_sum` | cuLaunchKernel row, Total Time |
| SM occupancy | `ncu --import ... --page details` | Achieved Occupancy |
| Memory throughput | `ncu --import ... --page details` | Memory Throughput % |

## Tips

- Run benchmarks with no other GPU processes (`nvidia-smi` to check).
- GPU clocks may boost/throttle. For stable measurements, lock clocks:
  ```bash
  sudo nvidia-smi -lgc 2000,2000   # lock at 2000 MHz (adjust for your GPU)
  ```
  Reset after: `sudo nvidia-smi -rgc`
- Run each measurement 3 times and take the median.
- The `results.json` from the benchmark already contains timesteps/s and kernel launch counts — start there before diving into traces.
