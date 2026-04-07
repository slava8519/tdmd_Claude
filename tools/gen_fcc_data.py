#!/usr/bin/env python3
"""Generate an FCC LAMMPS data file (atomic style, metal units).

Usage:
    python3 gen_fcc_data.py --elem Cu --a 3.615 --nx 4 --ny 4 --nz 4 -o cu_fcc_256.data
"""
import argparse
import sys


MASSES = {
    "Cu": 63.546,
    "Ni": 58.6934,
    "Al": 26.9815,
    "Co": 58.9332,
    "Cr": 51.9961,
    "Fe": 55.845,
}

FCC_BASIS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]


def main():
    parser = argparse.ArgumentParser(description="Generate FCC LAMMPS data file")
    parser.add_argument("--elem", default="Cu", help="Element symbol")
    parser.add_argument("--a", type=float, default=3.615, help="Lattice parameter (A)")
    parser.add_argument("--nx", type=int, default=4, help="Unit cells in x")
    parser.add_argument("--ny", type=int, default=4, help="Unit cells in y")
    parser.add_argument("--nz", type=int, default=4, help="Unit cells in z")
    parser.add_argument("-o", "--output", default="fcc.data", help="Output file")
    args = parser.parse_args()

    a = args.a
    nx, ny, nz = args.nx, args.ny, args.nz
    natoms = 4 * nx * ny * nz
    mass = MASSES.get(args.elem, 1.0)

    lx = nx * a
    ly = ny * a
    lz = nz * a

    atoms = []
    atom_id = 1
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                for bx, by, bz in FCC_BASIS:
                    x = (ix + bx) * a
                    y = (iy + by) * a
                    z = (iz + bz) * a
                    atoms.append((atom_id, 1, x, y, z))
                    atom_id += 1

    with open(args.output, "w") as f:
        f.write(f"LAMMPS data file: {natoms} {args.elem} atoms, FCC a={a}, {nx}x{ny}x{nz}\n\n")
        f.write(f"{natoms} atoms\n")
        f.write(f"1 atom types\n\n")
        f.write(f"0.0 {lx:.10f} xlo xhi\n")
        f.write(f"0.0 {ly:.10f} ylo yhi\n")
        f.write(f"0.0 {lz:.10f} zlo zhi\n\n")
        f.write(f"Masses\n\n")
        f.write(f"1 {mass}\n\n")
        f.write(f"Atoms # atomic\n\n")
        for aid, atype, x, y, z in atoms:
            f.write(f"{aid} {atype} {x:.10f} {y:.10f} {z:.10f}\n")

    print(f"Wrote {natoms} atoms to {args.output}")


if __name__ == "__main__":
    main()
