// SPDX-License-Identifier: Apache-2.0
// ipair.hpp — pair potential interface. Stub at M0.
#pragma once

#include "../core/types.hpp"

namespace tdmd::potentials {

/// Common interface for pair potentials (Morse, Lennard-Jones, etc.).
/// Many-body potentials (EAM, ML) implement a different interface — see eam.hpp at M1.
class IPairStyle {
 public:
  virtual ~IPairStyle() = default;

  /// Cutoff radius for force/energy evaluation.
  virtual real cutoff() const noexcept = 0;

  /// Compute pair energy and (df/dr)/r given squared distance r^2.
  /// Returning energy and force-over-distance avoids one sqrt per pair.
  virtual void compute(real r2, real& energy, real& fpair) const noexcept = 0;
};

}  // namespace tdmd::potentials
