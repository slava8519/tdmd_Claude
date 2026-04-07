// SPDX-License-Identifier: Apache-2.0
// morse.hpp — Morse pair potential.
//
// U(r) = D * [(1 - exp(-alpha*(r - r0)))^2 - 1]
// dU/dr = 2 * D * alpha * (1 - exp(-alpha*(r-r0))) * exp(-alpha*(r-r0))
// fpair = -dU/dr / r  (force-over-distance for avoiding sqrt)
#pragma once

#include <cmath>

#include "../core/types.hpp"
#include "ipair.hpp"

namespace tdmd::potentials {

/// @brief Morse pair potential.
///
/// Parameters are per-type-pair. For M1 we store a single set of parameters
/// (all type pairs use the same Morse). Multi-pair support is straightforward
/// to add later.
class MorsePair : public IPairStyle {
 public:
  /// @param d Well depth (eV).
  /// @param alpha Width parameter (1/Angstrom).
  /// @param r0 Equilibrium distance (Angstrom).
  /// @param rc Cutoff radius (Angstrom).
  MorsePair(real d, real alpha, real r0, real rc)
      : d_(d), alpha_(alpha), r0_(r0), rc_(rc), rc_sq_(rc * rc) {}

  [[nodiscard]] real cutoff() const noexcept override { return rc_; }

  /// @brief Compute energy and force-over-distance for squared distance r2.
  ///
  /// If r2 >= rc^2, sets energy=0, fpair=0.
  /// energy = D * [(1 - exp(-alpha*(r-r0)))^2 - 1]
  /// fpair  = -dU/dr / r  (so F_ij = fpair * delta_ij)
  void compute(real r2, real& energy, real& fpair) const noexcept override {
    if (r2 >= rc_sq_) {
      energy = real{0};
      fpair = real{0};
      return;
    }
    const real r = std::sqrt(r2);
    const real dr = r - r0_;
    const real exp_val = std::exp(-alpha_ * dr);
    const real one_minus_exp = real{1} - exp_val;

    energy = d_ * (one_minus_exp * one_minus_exp - real{1});

    // dU/dr = 2 * D * alpha * one_minus_exp * exp_val
    const real dudr = real{2} * d_ * alpha_ * one_minus_exp * exp_val;

    // fpair = -dU/dr / r
    fpair = -dudr / r;
  }

  [[nodiscard]] real d() const noexcept { return d_; }
  [[nodiscard]] real alpha() const noexcept { return alpha_; }
  [[nodiscard]] real r0() const noexcept { return r0_; }

 private:
  real d_;
  real alpha_;
  real r0_;
  real rc_;
  real rc_sq_;
};

}  // namespace tdmd::potentials
