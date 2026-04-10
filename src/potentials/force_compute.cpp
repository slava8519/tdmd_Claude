// SPDX-License-Identifier: Apache-2.0
// force_compute.cpp — pair force evaluation (CPU, half-list with Newton 3rd).

#include "force_compute.hpp"

#include "../core/math.hpp"

namespace tdmd::potentials {

real compute_pair_forces(SystemState& state,
                         const neighbors::NeighborList& nlist,
                         const IPairStyle& pair) {
  const i64 natoms = state.natoms;
  const Vec3D box_size = state.box.size();
  const auto& periodic = state.box.periodic;

  // Zero forces.
  for (i64 i = 0; i < natoms; ++i) {
    state.forces[static_cast<std::size_t>(i)] = {0, 0, 0};
  }

  real total_energy = real{0};

  for (i64 i = 0; i < natoms; ++i) {
    const auto si = static_cast<std::size_t>(i);
    // Position load in double (PositionVec = Vec3D per ADR 0007).
    const PositionVec pi = state.positions[si];
    const i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);

    for (i32 k = 0; k < cnt; ++k) {
      const auto j = static_cast<std::size_t>(nbrs[k]);
      // LAMMPS convention: delta = r_i - r_j (double subtract).
      Vec3D delta = pi - state.positions[j];
      delta = minimum_image(delta, box_size, periodic);
      const double r2_d = length_sq(delta);
      const real r2 = static_cast<real>(r2_d);

      real energy, fpair;
      pair.compute(r2, energy, fpair);

      total_energy += energy;

      // Force in force_t (float in mixed). Cast delta components for the
      // multiplication.
      const force_t fp_f = static_cast<force_t>(fpair);
      ForceVec fij{fp_f * static_cast<force_t>(delta.x),
                   fp_f * static_cast<force_t>(delta.y),
                   fp_f * static_cast<force_t>(delta.z)};
      state.forces[si] += fij;
      state.forces[j] -= fij;  // Newton 3rd
    }
  }

  return total_energy;
}

}  // namespace tdmd::potentials
