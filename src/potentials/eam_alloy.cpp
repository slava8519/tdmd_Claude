// SPDX-License-Identifier: Apache-2.0
// eam_alloy.cpp — EAM/alloy potential implementation.

#include "eam_alloy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "../core/error.hpp"
#include "../core/math.hpp"

namespace tdmd::potentials {

// ---- SplineTable ----

void SplineTable::build_splines() {
  if (n < 2) return;
  const auto ns = static_cast<std::size_t>(n);
  a.resize(ns - 1);
  b.resize(ns - 1);
  c.resize(ns - 1);
  d.resize(ns - 1);

  // Natural cubic spline.
  // We use a simple approach: cubic Hermite with finite-difference derivatives.
  // This matches LAMMPS's interpolation closely enough for our tolerance.
  for (std::size_t i = 0; i < ns - 1; ++i) {
    a[i] = values[i];
    // Finite difference slope.
    real slope_left = (i > 0) ? (values[i] - values[i - 1]) / dr : real{0};
    real slope_right = (values[i + 1] - values[i]) / dr;
    real slope_next =
        (i + 2 < ns) ? (values[i + 2] - values[i + 1]) / dr : slope_right;

    real m0 = (slope_left + slope_right) * real{0.5};
    real m1 = (slope_right + slope_next) * real{0.5};
    if (i == 0) m0 = slope_right;
    if (i == ns - 2) m1 = slope_right;

    // Hermite basis: f(t) = a + b*t*h + c*t^2*h^2 + d*t^3*h^3 (t in [0,1], h=dr)
    // Actually, let's store coefficients for f(x) = a + b*dx + c*dx^2 + d*dx^3
    // where dx = x - x_i.
    b[i] = m0;
    c[i] = (real{3} * slope_right - real{2} * m0 - m1) / dr;
    d[i] = (m0 + m1 - real{2} * slope_right) / (dr * dr);
  }
}

real SplineTable::eval(real r) const noexcept {
  if (n < 2) return real{0};
  real x = (r - rmin) / dr;
  auto idx = static_cast<i32>(x);
  if (idx < 0) idx = 0;
  if (idx >= n - 1) idx = n - 2;
  real dx = r - (rmin + static_cast<real>(idx) * dr);
  auto si = static_cast<std::size_t>(idx);
  return a[si] + dx * (b[si] + dx * (c[si] + dx * d[si]));
}

real SplineTable::eval_deriv(real r) const noexcept {
  if (n < 2) return real{0};
  real x = (r - rmin) / dr;
  auto idx = static_cast<i32>(x);
  if (idx < 0) idx = 0;
  if (idx >= n - 1) idx = n - 2;
  real dx = r - (rmin + static_cast<real>(idx) * dr);
  auto si = static_cast<std::size_t>(idx);
  return b[si] + dx * (real{2} * c[si] + dx * real{3} * d[si]);
}

// ---- setfl reader ----

namespace {

// Read N values from a stream, one or more per line.
void read_values(std::ifstream& file, i32 count, std::vector<real>& out) {
  out.resize(static_cast<std::size_t>(count));
  for (i32 i = 0; i < count; ++i) {
    if (!(file >> out[static_cast<std::size_t>(i)])) {
      TDMD_THROW("unexpected end of EAM file while reading values");
    }
  }
}

}  // namespace

void EamAlloy::read_setfl(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    TDMD_THROW("cannot open EAM file: " + filename);
  }

  // Lines 1-3: comments.
  std::string line;
  for (int i = 0; i < 3; ++i) {
    if (!std::getline(file, line)) TDMD_THROW("premature end of EAM file");
  }

  // Line 4: ntypes element1 element2 ...
  if (!std::getline(file, line)) TDMD_THROW("premature end of EAM file");
  {
    std::istringstream iss(line);
    iss >> ntypes_;
    elements_.resize(static_cast<std::size_t>(ntypes_));
    for (i32 i = 0; i < ntypes_; ++i) {
      iss >> elements_[static_cast<std::size_t>(i)];
    }
  }

  // Line 5: Nrho drho Nr dr cutoff
  i32 nrho, nr;
  real drho, dr_tab;
  if (!std::getline(file, line)) TDMD_THROW("premature end of EAM file");
  {
    std::istringstream iss(line);
    iss >> nrho >> drho >> nr >> dr_tab >> cutoff_;
  }

  // For each element: read atomic number, mass, lattice constant, lattice type,
  // then Nrho values of F(rho), then Nr values of rho(r).
  embedding_.resize(static_cast<std::size_t>(ntypes_));
  density_.resize(static_cast<std::size_t>(ntypes_));

  for (i32 t = 0; t < ntypes_; ++t) {
    auto st = static_cast<std::size_t>(t);

    // Element header line: atomic_number mass lattice_constant lattice_type.
    // For t==0 the stream is positioned at the start of this line (after the
    // getline on line 5). For t>0 the stream is positioned mid-line: the
    // previous read_values() used operator>> which stops after the last
    // numeric token, leaving the rest of that line unread. A plain getline
    // here would consume only that empty remainder. Guard against this by
    // skipping any line that does not contain a non-space character.
    bool got_header = false;
    while (std::getline(file, line)) {
      for (char c : line) {
        if (c != ' ' && c != '\t' && c != '\r') {
          got_header = true;
          break;
        }
      }
      if (got_header) break;
    }
    if (!got_header) TDMD_THROW("premature end of EAM file");
    // Header content (atomic_number mass lattice_const lattice_type) is
    // intentionally discarded — mass lives in the LAMMPS data file, and the
    // lattice metadata is informational only.

    // F(rho) values.
    embedding_[st].n = nrho;
    embedding_[st].dr = drho;
    embedding_[st].rmin = real{0};
    read_values(file, nrho, embedding_[st].values);
    embedding_[st].build_splines();

    // rho(r) values.
    density_[st].n = nr;
    density_[st].dr = dr_tab;
    density_[st].rmin = real{0};
    read_values(file, nr, density_[st].values);
    density_[st].build_splines();
  }

  // Pair interactions phi(r) * r. Stored as upper triangle.
  // For each pair (i,j) where i <= j, read Nr values of r*phi(r).
  i32 npairs = ntypes_ * (ntypes_ + 1) / 2;
  phi_.resize(static_cast<std::size_t>(npairs));

  for (i32 i = 0; i < ntypes_; ++i) {
    for (i32 j = i; j < ntypes_; ++j) {
      auto pidx = phi_index(i, j);
      phi_[pidx].n = nr;
      phi_[pidx].dr = dr_tab;
      phi_[pidx].rmin = real{0};
      std::vector<real> rphi;
      read_values(file, nr, rphi);

      // Convert r*phi(r) -> phi(r). At r=0, phi is undefined; set to 0.
      phi_[pidx].values.resize(static_cast<std::size_t>(nr));
      phi_[pidx].values[0] = real{0};
      for (i32 k = 1; k < nr; ++k) {
        real r = static_cast<real>(k) * dr_tab;
        phi_[pidx].values[static_cast<std::size_t>(k)] =
            rphi[static_cast<std::size_t>(k)] / r;
      }
      phi_[pidx].build_splines();
    }
  }
}

// ---- 3-pass force compute ----

real EamAlloy::compute_forces(SystemState& state,
                              const neighbors::NeighborList& nlist) const {
  const i64 natoms = state.natoms;
  const Vec3D box_size = state.box.size();
  const auto& periodic = state.box.periodic;
  const real rc_sq = cutoff_ * cutoff_;

  // Zero forces.
  for (i64 i = 0; i < natoms; ++i) {
    state.forces[static_cast<std::size_t>(i)] = {0, 0, 0};
  }

  // Allocate per-atom arrays.
  auto n = static_cast<std::size_t>(natoms);
  std::vector<real> rho(n, real{0});
  std::vector<real> fp(n, real{0});  // dF/drho

  // Pass 1: gather electron density rho_i = sum_j rho_j(r_ij).
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    // Position load in double (PositionVec = Vec3D per ADR 0007).
    const PositionVec pi = state.positions[si];
    const i32 ti = state.types[si] - 1;  // 0-based type
    const i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);

    for (i32 k = 0; k < cnt; ++k) {
      auto j = static_cast<std::size_t>(nbrs[k]);
      Vec3D delta = pi - state.positions[j];
      delta = minimum_image(delta, box_size, periodic);
      real r2 = static_cast<real>(length_sq(delta));
      if (r2 >= rc_sq) continue;

      real r = std::sqrt(r2);
      const i32 tj = state.types[j] - 1;

      // Half-list: both i and j accumulate.
      rho[si] += density_[static_cast<std::size_t>(tj)].eval(r);
      rho[j] += density_[static_cast<std::size_t>(ti)].eval(r);
    }
  }

  // Pass 2: compute embedding energy and derivative.
  real energy_embed = real{0};
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const i32 ti = state.types[si] - 1;
    auto sti = static_cast<std::size_t>(ti);
    energy_embed += embedding_[sti].eval(rho[si]);
    fp[si] = embedding_[sti].eval_deriv(rho[si]);
  }

  // Pass 3: compute pair forces + embedding contribution.
  real energy_pair = real{0};
  for (i64 i = 0; i < natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const PositionVec pi = state.positions[si];
    const i32 ti = state.types[si] - 1;
    const i32 cnt = nlist.count(i);
    const i32* nbrs = nlist.neighbors_of(i);

    for (i32 k = 0; k < cnt; ++k) {
      auto j = static_cast<std::size_t>(nbrs[k]);
      // LAMMPS convention: delta = r_i - r_j (double subtract).
      Vec3D delta = pi - state.positions[j];
      delta = minimum_image(delta, box_size, periodic);
      real r2 = static_cast<real>(length_sq(delta));
      if (r2 >= rc_sq) continue;

      real r = std::sqrt(r2);
      const i32 tj = state.types[j] - 1;
      auto pidx = phi_index(ti, tj);

      // Pair energy (half-list, count once).
      real phi_val = phi_[pidx].eval(r);
      energy_pair += phi_val;

      // Pair force: -dphi/dr / r.
      real dphi_dr = phi_[pidx].eval_deriv(r);

      // Embedding force: -(fp_i * drho_j/dr + fp_j * drho_i/dr) / r.
      real drho_j_dr =
          density_[static_cast<std::size_t>(tj)].eval_deriv(r);
      real drho_i_dr =
          density_[static_cast<std::size_t>(ti)].eval_deriv(r);

      real fpair =
          -(dphi_dr + fp[si] * drho_j_dr + fp[j] * drho_i_dr) / r;

      const force_t fp_f = static_cast<force_t>(fpair);
      ForceVec fij{fp_f * static_cast<force_t>(delta.x),
                   fp_f * static_cast<force_t>(delta.y),
                   fp_f * static_cast<force_t>(delta.z)};
      state.forces[si] += fij;
      state.forces[j] -= fij;
    }
  }

  return energy_embed + energy_pair;
}

}  // namespace tdmd::potentials
