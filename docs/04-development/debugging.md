# Debugging Guide

> Tools and techniques for debugging TDMD.

## Quick reference

| Problem | First tool |
|---|---|
| Crash with stack trace | `gdb`, `addr2line` |
| Memory error / leak | `valgrind`, `AddressSanitizer` |
| Race condition | `ThreadSanitizer`, deterministic mode |
| Wrong physics | VerifyLab + LAMMPS A/B |
| Slow code | `nsys profile`, `ncu`, `perf` |
| Wrong CUDA result | `compute-sanitizer`, FP64 mode comparison |
| Hangs in MPI | `gdb` attach to ranks, `MPI_Barrier` audit |

## Sanitizers

Build with sanitizers:

```bash
cmake -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
./build-asan/tdmd_standalone <args>
```

Run the test suite with ASan periodically (and in CI nightly).

## CUDA debugging

```bash
compute-sanitizer ./build/tdmd_standalone <args>           # memory checks
compute-sanitizer --tool racecheck ./build/tdmd_standalone # races
cuda-gdb ./build/tdmd_standalone                            # interactive
```

For numerical issues:
1. Switch to `TDMD_FP64=ON`. If the bug goes away, it's a precision issue.
2. Switch to `TDMD_DETERMINISTIC=ON`. If the bug goes away, it's an FP-order issue.
3. Compare GPU output to the CPU reference one step at a time.

## Profiling

```bash
nsys profile -o tdmd_profile ./build/tdmd_standalone <args>
nsys-ui tdmd_profile.nsys-rep   # interactive
```

For kernel-level detail:

```bash
ncu --set full -o tdmd_kernel ./build/tdmd_standalone <args>
ncu-ui tdmd_kernel.ncu-rep
```

## MPI debugging (M5+)

```bash
mpirun -np 4 xterm -e gdb --args ./build/tdmd_standalone <args>
```

Or attach after launch:

```bash
mpirun -np 4 ./build/tdmd_standalone <args> &
gdb -p $(pgrep -f tdmd_standalone | head -1)
```

## "It works on my machine" — checklist

- [ ] Same compiler version?
- [ ] Same CMake build type (Release vs Debug)?
- [ ] Same `TDMD_FP64` setting?
- [ ] Same `TDMD_DETERMINISTIC` setting?
- [ ] Same input file (md5)?
- [ ] Same number of MPI ranks and GPUs?
- [ ] Same RNG seed?

If all of these are equal and behavior differs — it's an undefined-behavior bug. Run with sanitizers.

## When stuck

1. Reproduce in the smallest possible input.
2. Capture the failing case as a regression test.
3. `git bisect` to find the introducing commit.
4. Read the diff carefully.
5. **Then** start guessing.
