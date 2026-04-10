// SPDX-License-Identifier: Apache-2.0
// tdmd_main.cpp — standalone driver entry point.
//
// M1: reads LAMMPS data file, runs NVE MD with Morse pair potential.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "core/constants.hpp"
#include "core/math.hpp"
#include "core/system_state.hpp"
#include "integrator/velocity_verlet.hpp"
#include "io/lammps_data_reader.hpp"
#include "neighbors/neighbor_list.hpp"
#include "potentials/force_compute.hpp"
#include "potentials/morse.hpp"

namespace {
constexpr const char* kVersion = "0.7.0-dev";

struct RunConfig {
  std::string data_file;
  int nsteps = 1000;
  double dt = 0.001;  // ps (1 fs)
  // Morse parameters.
  double morse_d = 0.3429;
  double morse_alpha = 1.3588;
  double morse_r0 = 2.866;
  double morse_rc = 9.5;
  // Neighbor list.
  double skin = 1.0;
  // Output.
  int thermo_every = 100;
  // VerifyLab force dump (step 0, LAMMPS dump format).
  std::string dump_forces_file;
};

void print_usage() {
  std::printf("Usage: tdmd_standalone [options]\n");
  std::printf("  --data <file>          LAMMPS data file (required)\n");
  std::printf("  --nsteps <N>           Number of steps (default 1000)\n");
  std::printf("  --dt <ps>              Time step in ps (default 0.001)\n");
  std::printf("  --morse D,alpha,r0,rc  Morse parameters (default 0.3429,1.3588,2.866,9.5)\n");
  std::printf("  --skin <A>             Neighbor list skin (default 1.0)\n");
  std::printf("  --thermo <N>           Print thermo every N steps (default 100)\n");
  std::printf("  --dump-forces <file>   Write step-0 forces in LAMMPS dump format\n");
  std::printf("  --version, -v          Print version\n");
  std::printf("  --help, -h             Print this help\n");
}

/// Write the current system state (positions + forces) as a LAMMPS dump.
/// Format matches `dump custom id type x y z fx fy fz` — chosen so that
/// the same parser can compare TDMD output against committed LAMMPS
/// reference dumps in verifylab/cases/*.
void write_lammps_force_dump(const tdmd::SystemState& state,
                             const std::string& path) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Error: cannot open %s for writing\n", path.c_str());
    return;
  }
  out.precision(17);
  out << "ITEM: TIMESTEP\n" << state.step << "\n";
  out << "ITEM: NUMBER OF ATOMS\n" << state.natoms << "\n";
  out << "ITEM: BOX BOUNDS pp pp pp\n";
  out << static_cast<double>(state.box.lo.x) << " "
      << static_cast<double>(state.box.hi.x) << "\n";
  out << static_cast<double>(state.box.lo.y) << " "
      << static_cast<double>(state.box.hi.y) << "\n";
  out << static_cast<double>(state.box.lo.z) << " "
      << static_cast<double>(state.box.hi.z) << "\n";
  out << "ITEM: ATOMS id type x y z fx fy fz\n";
  for (tdmd::i64 i = 0; i < state.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    const auto& p = state.positions[si];
    const auto& f = state.forces[si];
    out << state.ids[si] << " " << state.types[si] << " "
        << static_cast<double>(p.x) << " "
        << static_cast<double>(p.y) << " "
        << static_cast<double>(p.z) << " "
        << static_cast<double>(f.x) << " "
        << static_cast<double>(f.y) << " "
        << static_cast<double>(f.z) << "\n";
  }
}

tdmd::real kinetic_energy(const tdmd::SystemState& s) {
  tdmd::accum_t ke = 0;
  for (tdmd::i64 i = 0; i < s.natoms; ++i) {
    auto si = static_cast<std::size_t>(i);
    tdmd::accum_t mass =
        s.masses[static_cast<std::size_t>(s.types[si])];
    ke += tdmd::accum_t{0.5} * mass *
          static_cast<tdmd::accum_t>(tdmd::kMvv2e) *
          tdmd::length_sq(s.velocities[si]);
  }
  return static_cast<tdmd::real>(ke);
}

void print_thermo(const tdmd::SystemState& s, tdmd::real pe, tdmd::real ke) {
  tdmd::real te = pe + ke;
  // Temperature: T = 2 * KE / (3 * N * kB)
  tdmd::real temp = tdmd::real{0};
  if (s.natoms > 0) {
    temp = tdmd::real{2} * ke /
           (tdmd::real{3} * static_cast<tdmd::real>(s.natoms) * tdmd::kBoltzmann);
  }
  std::printf("Step %8lld  PE %20.14f  KE %20.14f  TE %20.14f  T %10.4f\n",
              static_cast<long long>(s.step), static_cast<double>(pe),
              static_cast<double>(ke), static_cast<double>(te),
              static_cast<double>(temp));
}

}  // namespace

int main(int argc, char** argv) {
  RunConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--version" || arg == "-v") {
      std::printf("tdmd v%s\n", kVersion);
      return 0;
    }
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
    if (arg == "--data" && i + 1 < argc) {
      cfg.data_file = argv[++i];
    } else if (arg == "--nsteps" && i + 1 < argc) {
      cfg.nsteps = std::atoi(argv[++i]);
    } else if (arg == "--dt" && i + 1 < argc) {
      cfg.dt = std::atof(argv[++i]);
    } else if (arg == "--skin" && i + 1 < argc) {
      cfg.skin = std::atof(argv[++i]);
    } else if (arg == "--thermo" && i + 1 < argc) {
      cfg.thermo_every = std::atoi(argv[++i]);
    } else if (arg == "--dump-forces" && i + 1 < argc) {
      cfg.dump_forces_file = argv[++i];
    } else if (arg == "--morse" && i + 1 < argc) {
      // Parse "D,alpha,r0,rc"
      if (std::sscanf(argv[++i], "%lf,%lf,%lf,%lf",
                       &cfg.morse_d, &cfg.morse_alpha,
                       &cfg.morse_r0, &cfg.morse_rc) != 4) {
        std::fprintf(stderr, "Error: --morse expects D,alpha,r0,rc\n");
        return 1;
      }
    }
  }

  if (cfg.data_file.empty()) {
    std::printf("tdmd v%s\n", kVersion);
    std::printf("  Time-Decomposition Molecular Dynamics engine\n");
    std::printf("  Use --data <file> to run a simulation. --help for options.\n");
    return 0;
  }

  // Read input.
  std::printf("tdmd v%s\n", kVersion);
  std::printf("Reading: %s\n", cfg.data_file.c_str());
  auto state = tdmd::io::read_lammps_data(cfg.data_file);
  std::printf("Atoms: %lld  Types: %zu  Box: %.3f x %.3f x %.3f\n",
              static_cast<long long>(state.natoms),
              state.masses.size() - 1,
              static_cast<double>(state.box.size().x),
              static_cast<double>(state.box.size().y),
              static_cast<double>(state.box.size().z));

  // Set up potential and integrator.
  tdmd::potentials::MorsePair pot(
      static_cast<tdmd::real>(cfg.morse_d),
      static_cast<tdmd::real>(cfg.morse_alpha),
      static_cast<tdmd::real>(cfg.morse_r0),
      static_cast<tdmd::real>(cfg.morse_rc));

  tdmd::integrator::VelocityVerlet vv(static_cast<tdmd::real>(cfg.dt));

  tdmd::neighbors::NeighborList nlist;
  auto skin = static_cast<tdmd::real>(cfg.skin);

  // Build neighbor list and compute initial forces.
  nlist.build(state.positions.data(), state.natoms, state.box,
              pot.cutoff(), skin);

  tdmd::real pe = tdmd::potentials::compute_pair_forces(state, nlist, pot);
  tdmd::real ke = kinetic_energy(state);

  std::printf("\nMorse: D=%.4f alpha=%.4f r0=%.3f rc=%.1f\n",
              cfg.morse_d, cfg.morse_alpha, cfg.morse_r0, cfg.morse_rc);
  std::printf("dt=%.4f ps  nsteps=%d  skin=%.1f A\n\n",
              cfg.dt, cfg.nsteps, cfg.skin);

  print_thermo(state, pe, ke);

  if (!cfg.dump_forces_file.empty()) {
    write_lammps_force_dump(state, cfg.dump_forces_file);
    std::printf("Wrote step-0 force dump to %s\n", cfg.dump_forces_file.c_str());
  }

  // Main MD loop.
  int rebuilds = 0;
  for (int step = 0; step < cfg.nsteps; ++step) {
    vv.half_kick(state);
    vv.drift(state);

    if (nlist.needs_rebuild(state.positions.data(), state.natoms)) {
      nlist.build(state.positions.data(), state.natoms, state.box,
                  pot.cutoff(), skin);
      ++rebuilds;
    }

    pe = tdmd::potentials::compute_pair_forces(state, nlist, pot);
    vv.half_kick(state);

    if (cfg.thermo_every > 0 && (step + 1) % cfg.thermo_every == 0) {
      ke = kinetic_energy(state);
      print_thermo(state, pe, ke);
    }
  }

  if (cfg.nsteps > 0) {
    ke = kinetic_energy(state);
    std::printf("\nFinal:\n");
    print_thermo(state, pe, ke);
    std::printf("Neighbor list rebuilds: %d\n", rebuilds);
  }

  return 0;
}
