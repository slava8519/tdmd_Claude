# Workflow: Validate against LAMMPS

Use this whenever physics output needs to be compared with LAMMPS.

## Steps

1. **Pick the case.** Either an existing VerifyLab case or a new one. If new: create `verifylab/cases/<name>/`.

2. **Prepare the LAMMPS input.**
   - Same atoms (same data file, ideally).
   - Same potential file.
   - Same `units metal`, same `dt`, same neighbor settings.
   - Same initial velocities (`velocity ... loop geom` for determinism across MPI sizes).
   - Same number of steps.
   - Output `thermo` with the columns we'll compare on.

3. **Run LAMMPS.** Save the log and any dumps as the `reference/` baseline. Commit them.

4. **Prepare the TDMD input.** Translate the LAMMPS input to TDMD format, keeping everything consistent.

5. **Run TDMD.** Save outputs.

6. **Compare** with `verifylab/runners/compare_with_lammps.py`. Tolerances live in `tolerance.toml`.

7. **Interpret.**
   - Equal within tolerance → pass.
   - Different but within statistical noise → check by running multiple seeds.
   - Different in a meaningful way → bug. Investigate.

8. **Document the result** in the case `README.md`:
   - What we compared.
   - The numbers.
   - The conclusion.
