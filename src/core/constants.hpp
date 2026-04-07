// SPDX-License-Identifier: Apache-2.0
// constants.hpp — physical constants in LAMMPS "metal" units.
//
// Metal units:
//   distance = Angstrom, time = picosecond, mass = gram/mole,
//   energy = eV, velocity = Angstrom/picosecond,
//   force = eV/Angstrom, temperature = Kelvin.
#pragma once

#include "types.hpp"

namespace tdmd {

/// Boltzmann constant in eV/K.
inline constexpr real kBoltzmann = real{8.617333262e-5};

/// Conversion factor: force (eV/A) * dt (ps) / mass (g/mol) -> velocity (A/ps).
/// In metal units: a = F/m needs a prefactor of 1/(1e-4 * 6.0221408e23) ... but
/// LAMMPS metal units define mvv2e = 1.0364269e-4. We use the LAMMPS value for
/// perfect A/B match.
///
/// v += (dt / mass) * force * kMvv2eInv
/// where kMvv2eInv = 1 / mvv2e = 1 / 1.0364269e-4
///
/// Actually in LAMMPS metal units, the equation of motion is:
///   a_i = F_i / (mass_i * mvv2e)
/// so dt_force = dt / (mass * mvv2e).
inline constexpr real kMvv2e = real{1.0364269e-4};

}  // namespace tdmd
