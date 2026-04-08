// SPDX-License-Identifier: Apache-2.0
// device_nose_hoover.cuh — GPU Nosé-Hoover NVT thermostat.
//
// Implements a Nosé-Hoover chain (NHC) thermostat following the LAMMPS
// "fix nvt" algorithm. The chain update is done on the host; GPU kernels
// handle KE reduction and velocity scaling.
//
// Integration order within one VV step:
//   1. nhc_half_step (host: update chain, get scale factor)
//   2. scale_velocities (GPU)
//   3. half_kick (GPU)
//   4. drift (GPU)
//   5. force compute (GPU)
//   6. half_kick (GPU)
//   7. nhc_half_step (host: update chain, get scale factor)
//   8. scale_velocities (GPU)
#pragma once

#include <vector>

#include "../core/types.hpp"

namespace tdmd::integrator {

/// @brief Configuration for Nosé-Hoover chain thermostat.
struct NHCConfig {
  real t_target{300.0};   ///< Target temperature in Kelvin.
  real t_period{0.1};     ///< Coupling period in picoseconds (LAMMPS Tdamp).
  i32 chain_length{3};    ///< Number of thermostats in the chain.
};

/// @brief Nosé-Hoover chain state and update logic (host-side).
///
/// Manages the chain variables and computes the velocity scaling factor.
/// The chain update follows the Martyna-Tuckerman-Tobias-Klein (MTTK)
/// integration scheme used by LAMMPS.
class NoseHooverChain {
 public:
  /// @param cfg Chain configuration.
  /// @param dt Time step in picoseconds.
  /// @param n_dof Number of translational degrees of freedom (3*N - 3).
  NoseHooverChain(const NHCConfig& cfg, real dt, i32 n_dof);

  /// @brief Perform half-step chain update and return velocity scale factor.
  /// @param ke_current Current total kinetic energy (in eV).
  /// @return Scale factor to multiply all velocities by.
  [[nodiscard]] real half_step(real ke_current);

  [[nodiscard]] real t_target() const noexcept { return cfg_.t_target; }
  void set_t_target(real t) { cfg_.t_target = t; }

 private:
  NHCConfig cfg_;
  real dt_;
  i32 n_dof_;

  std::vector<real> eta_;   ///< Chain positions (dimensionless).
  std::vector<real> eta_v_; ///< Chain velocities (1/ps).
  std::vector<real> eta_m_; ///< Chain masses (eV·ps²).
};

/// @brief Compute total kinetic energy on GPU.
/// @param d_velocities Device array of velocities.
/// @param d_types Device array of atom types (0-based).
/// @param d_masses Device array of per-type masses.
/// @param natoms Number of atoms.
/// @return Total kinetic energy in eV.
[[nodiscard]] real device_compute_ke(const Vec3* d_velocities,
                                    const i32* d_types, const real* d_masses,
                                    i32 natoms);

/// @brief Scale all velocities by a constant factor on GPU.
/// @param d_velocities Device array of velocities.
/// @param natoms Number of atoms.
/// @param factor Scale factor.
void device_scale_velocities(Vec3* d_velocities, i32 natoms, real factor);

/// @brief Scale velocities for a zone range on GPU.
void device_scale_velocities_zone(Vec3* d_velocities, i32 first_atom,
                                  i32 atom_count, real factor,
                                  cudaStream_t stream = nullptr);

/// @brief Compute maximum atomic speed |v| on GPU.
/// @param d_velocities Device array of velocities.
/// @param natoms Number of atoms.
/// @return Maximum speed (Å/ps).
[[nodiscard]] real device_compute_vmax(const Vec3* d_velocities, i32 natoms);

}  // namespace tdmd::integrator
