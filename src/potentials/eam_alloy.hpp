// SPDX-License-Identifier: Apache-2.0
// eam_alloy.hpp — EAM/alloy many-body potential with setfl file reader.
//
// EAM total energy:
//   E = sum_i F(rho_i) + (1/2) sum_i sum_{j!=i} phi(r_ij)
//
// where rho_i = sum_{j!=i} rho(r_ij) is the electron density at atom i.
//
// Three-pass force computation:
//   Pass 1: gather rho_i = sum_j rho(r_ij)
//   Pass 2: compute dF/drho for each atom
//   Pass 3: scatter forces from phi'(r_ij) + [dF/drho_i + dF/drho_j] * rho'(r_ij)
#pragma once

#include <string>
#include <vector>

#include "../core/system_state.hpp"
#include "../core/types.hpp"
#include "../neighbors/neighbor_list.hpp"

namespace tdmd::potentials {

/// @brief Cubic spline table for one function (density, embedding, or pair phi).
///
/// Stores spline coefficients a,b,c,d for intervals [n*dr, (n+1)*dr].
/// f(r) = a + b*(r-r_n) + c*(r-r_n)^2 + d*(r-r_n)^3
struct SplineTable {
  i32 n{0};         // number of data points
  real dr{0};       // spacing
  real rmin{0};     // starting r (usually 0)
  std::vector<real> values;  // raw tabulated values (length n)
  // Spline coefficients (length n-1 each).
  std::vector<real> a, b, c, d;

  /// Build cubic spline coefficients from raw values.
  void build_splines();

  /// Evaluate value at r.
  [[nodiscard]] real eval(real r) const noexcept;

  /// Evaluate derivative df/dr at r.
  [[nodiscard]] real eval_deriv(real r) const noexcept;
};

/// @brief EAM/alloy potential — reads setfl format.
class EamAlloy {
 public:
  /// @brief Read a setfl file (LAMMPS eam/alloy format).
  /// @param filename Path to the setfl file.
  /// @throws tdmd::Error on parse failure.
  void read_setfl(const std::string& filename);

  /// @brief Compute forces and energy (3-pass EAM).
  ///
  /// Uses the neighbor list for pair iteration.
  /// @return Total potential energy.
  [[nodiscard]] real compute_forces(SystemState& state,
                                    const neighbors::NeighborList& nlist) const;

  [[nodiscard]] real cutoff() const noexcept { return cutoff_; }
  [[nodiscard]] i32 ntypes() const noexcept { return ntypes_; }

 private:
  i32 ntypes_{0};
  real cutoff_{0};
  std::vector<std::string> elements_;

  // Per-element tables: F(rho) and rho(r). Indexed by type-1.
  std::vector<SplineTable> embedding_;  // F(rho), length ntypes
  std::vector<SplineTable> density_;    // rho(r), length ntypes

  // Pair interaction phi(r). For ntypes elements, stored as flat upper-triangle:
  // index(i,j) = i*ntypes - i*(i+1)/2 + j, where i <= j (0-based).
  std::vector<SplineTable> phi_;

  [[nodiscard]] std::size_t phi_index(i32 ti, i32 tj) const noexcept {
    // ti, tj are 0-based type indices. Ensure ti <= tj.
    if (ti > tj) std::swap(ti, tj);
    return static_cast<std::size_t>(ti * ntypes_ - ti * (ti + 1) / 2 + tj);
  }
};

}  // namespace tdmd::potentials
