# FCC16 benchmark suite

> A 16-atom-type FCC alloy stress test for EAM potentials. The main TDMD benchmark.

## Why FCC16

A 16-component FCC alloy is the hardest realistic test for an EAM implementation:

- **Lots of pair types** (16×17/2 = 136 unique pair interactions).
- **Lots of embedding functions** (16 unique F(ρ)).
- **Heavy memory traffic** for the EAM density gather.
- **Statistically meaningful** size at thousands of atoms.
- **Reproducible** — public NIST potentials and a published baseline.

If TDMD's EAM is correct on FCC16, it's correct on the easier alloys too.

## Sizes

| Size | Atoms | Approx. memory | Use |
|---|---|---|---|
| Tiny | 256 | < 1 MB | Unit-style smoke test (M2 first run) |
| Small | 4 096 | ~1 MB | CI nightly perf, single-GPU sanity |
| Medium | 32 768 | ~10 MB | Single-GPU production benchmark |
| Large | 262 144 | ~80 MB | Multi-GPU strong scaling |
| Huge | 2 097 152 | ~640 MB | Stretch goal for the RTX 5080 |

## Files

- `generate.py` — generates an FCC16 LAMMPS data file at the requested size.
- `lammps/in.fcc16_suite.in` — LAMMPS input running NVE for N steps.
- `expected/` — committed baseline numbers (timesteps/s, energies, forces) per size.

## Order of operations

1. **Before fcc16:** validate Ni-Co-Cr (3 types) first. It's a smaller test surface and most of the EAM bugs surface there. See `benchmarks/nicocr/`.
2. Once Ni-Co-Cr is green on the smallest size, scale up to FCC16 tiny.
3. Once FCC16 tiny is green, climb the size ladder one step at a time.

## Running

```bash
# Generate the data file (M2+)
python3 benchmarks/fcc16/generate.py --size small --output build/fcc16_small.lmps

# LAMMPS reference
lmp -in benchmarks/fcc16/lammps/in.fcc16_suite.in -var size small

# TDMD
./build/bin/tdmd --input benchmarks/fcc16/tdmd/in.fcc16_small.in --benchmark
```

## Pass criteria

- Forces match LAMMPS within `1e-5` relative (mixed precision).
- Energy drift over 10 000 steps within `1e-7 / atom / ps`.
- Throughput within 1.5× of LAMMPS-GPU on the same hardware (Medium size, single GPU, M7 target).
