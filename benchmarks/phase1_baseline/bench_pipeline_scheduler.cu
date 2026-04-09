// SPDX-License-Identifier: Apache-2.0
// bench_pipeline_scheduler.cu — Phase 1 measurement tool for ADR 0005.
//
// Runs PipelineScheduler on a given system, measures wall clock and telemetry,
// outputs JSON. Designed to be wrapped by nsys/ncu from outside.
//
// Usage:
//   bench_pipeline_scheduler --data path.data --steps 1000 [--warmup 100]
//
// No profiling inside — nsys/ncu wrap this binary externally.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/system_state.hpp"
#include "core/types.hpp"
#include "io/lammps_data_reader.hpp"
#include "potentials/device_morse.cuh"
#include "scheduler/fast_pipeline_scheduler.cuh"
#include "scheduler/pipeline_scheduler.cuh"

using namespace tdmd;

namespace {

struct BenchConfig {
  std::string data_file;
  std::string scheduler_name = "pipeline";  // "pipeline" or "fast_pipeline"
  i32 steps = 1000;
  i32 warmup = 100;
  std::string output;  // empty = stdout
  // Morse Cu parameters (same as all M1-M7 tests).
  real morse_d = real{0.3429};
  real morse_alpha = real{1.3588};
  real morse_r0 = real{2.866};
  real morse_rc = real{6.0};
  // Scheduler config.
  real dt = real{0.001};
  real r_skin = real{1.0};
  i32 rebuild_every = 10;
  bool deterministic = false;
  i32 n_streams = 4;
  // Velocity initialization.
  real t_init = real{300.0};
  unsigned seed = 42;
};

void print_usage() {
  std::printf(
      "bench_pipeline_scheduler — Phase 1 measurement tool for ADR 0005\n\n"
      "Usage:\n"
      "  bench_pipeline_scheduler --data <file> [options]\n\n"
      "Options:\n"
      "  --scheduler <name>  pipeline or fast_pipeline (default pipeline)\n"
      "  --data <file>       LAMMPS data file (required)\n"
      "  --steps <N>         Measurement steps (default 1000)\n"
      "  --warmup <N>        Warmup steps, not measured (default 100)\n"
      "  --output <file>     Write JSON to file (default stdout)\n"
      "  --dt <ps>           Time step (default 0.001)\n"
      "  --skin <A>          Verlet skin (default 1.0)\n"
      "  --deterministic     Single stream, sequential walk\n"
      "  --streams <N>       CUDA stream pool size (default 4)\n"
      "  --t-init <K>        Initial temperature for velocity init (default "
      "300)\n"
      "  --seed <N>          RNG seed for velocity init (default 42)\n"
      "  --morse D,a,r0,rc   Morse parameters (default 0.3429,1.3588,2.866,6.0)\n"
      "  --help, -h          Print this help\n");
}

BenchConfig parse_args(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    }
    if (arg == "--scheduler" && i + 1 < argc) {
      cfg.scheduler_name = argv[++i];
    } else if (arg == "--data" && i + 1 < argc) {
      cfg.data_file = argv[++i];
    } else if (arg == "--steps" && i + 1 < argc) {
      cfg.steps = std::atoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup = std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      cfg.output = argv[++i];
    } else if (arg == "--dt" && i + 1 < argc) {
      cfg.dt = static_cast<real>(std::atof(argv[++i]));
    } else if (arg == "--skin" && i + 1 < argc) {
      cfg.r_skin = static_cast<real>(std::atof(argv[++i]));
    } else if (arg == "--deterministic") {
      cfg.deterministic = true;
    } else if (arg == "--streams" && i + 1 < argc) {
      cfg.n_streams = std::atoi(argv[++i]);
    } else if (arg == "--t-init" && i + 1 < argc) {
      cfg.t_init = static_cast<real>(std::atof(argv[++i]));
    } else if (arg == "--seed" && i + 1 < argc) {
      cfg.seed = static_cast<unsigned>(std::atoi(argv[++i]));
    } else if (arg == "--morse" && i + 1 < argc) {
      double d, a, r0, rc;
      if (std::sscanf(argv[++i], "%lf,%lf,%lf,%lf", &d, &a, &r0, &rc) == 4) {
        cfg.morse_d = static_cast<real>(d);
        cfg.morse_alpha = static_cast<real>(a);
        cfg.morse_r0 = static_cast<real>(r0);
        cfg.morse_rc = static_cast<real>(rc);
      } else {
        std::fprintf(stderr, "Error: --morse expects D,alpha,r0,rc\n");
        std::exit(1);
      }
    }
  }
  return cfg;
}

/// Initialize velocities from Maxwell-Boltzmann at target T.
/// Removes COM momentum and rescales to exact T.
/// Same logic as test_nvt_thermostat.cu init_velocities().
void init_velocities(SystemState& state, real t_target, unsigned seed) {
  auto n = static_cast<std::size_t>(state.natoms);
  std::mt19937 rng(seed);
  std::normal_distribution<real> gauss(0.0, 1.0);

  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
    real sigma = std::sqrt(kBoltzmann * t_target / (mass * kMvv2e));
    state.velocities[i].x = sigma * gauss(rng);
    state.velocities[i].y = sigma * gauss(rng);
    state.velocities[i].z = sigma * gauss(rng);
  }

  // Remove COM momentum.
  Vec3 com_v{0, 0, 0};
  real total_mass = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
    com_v.x += mass * state.velocities[i].x;
    com_v.y += mass * state.velocities[i].y;
    com_v.z += mass * state.velocities[i].z;
    total_mass += mass;
  }
  com_v.x /= total_mass;
  com_v.y /= total_mass;
  com_v.z /= total_mass;
  for (std::size_t i = 0; i < n; ++i) {
    state.velocities[i].x -= com_v.x;
    state.velocities[i].y -= com_v.y;
    state.velocities[i].z -= com_v.z;
  }

  // Rescale to exact target T.
  real ke = 0;
  for (std::size_t i = 0; i < n; ++i) {
    real mass = state.masses[static_cast<std::size_t>(state.types[i])];
    const Vec3& v = state.velocities[i];
    ke += real{0.5} * mass * kMvv2e * (v.x * v.x + v.y * v.y + v.z * v.z);
  }
  i32 n_dof = 3 * static_cast<i32>(state.natoms) - 3;
  real t_current = real{2} * ke / (static_cast<real>(n_dof) * kBoltzmann);
  if (t_current > 0) {
    real scale = std::sqrt(t_target / t_current);
    for (std::size_t i = 0; i < n; ++i) {
      state.velocities[i].x *= scale;
      state.velocities[i].y *= scale;
      state.velocities[i].z *= scale;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  BenchConfig cfg = parse_args(argc, argv);

  if (cfg.data_file.empty()) {
    print_usage();
    return 1;
  }

  // Load system.
  std::fprintf(stderr, "Loading: %s\n", cfg.data_file.c_str());
  SystemState state = io::read_lammps_data(cfg.data_file);
  auto natoms = static_cast<i32>(state.natoms);
  std::fprintf(stderr, "Atoms: %d  Box: %.3f x %.3f x %.3f\n", natoms,
               static_cast<double>(state.box.size().x),
               static_cast<double>(state.box.size().y),
               static_cast<double>(state.box.size().z));

  // Initialize velocities (data files start with zero velocities).
  init_velocities(state, cfg.t_init, cfg.seed);
  std::fprintf(stderr, "Velocities initialized at T=%.0f K (seed=%u)\n",
               static_cast<double>(cfg.t_init), cfg.seed);

  // Set up Morse potential.
  potentials::MorseParams params{cfg.morse_d, cfg.morse_alpha, cfg.morse_r0,
                                 cfg.morse_rc, cfg.morse_rc * cfg.morse_rc};

  // Common measurement variables.
  i32 n_zones = 0;
  i64 kernel_launches = 0;
  i64 dep_checks = 0;
  i64 dep_failures = 0;
  i64 ticks = 0;
  double wall_s = 0;

  if (cfg.scheduler_name == "fast_pipeline") {
    // --- FastPipelineScheduler path ---
    scheduler::FastPipelineConfig fcfg;
    fcfg.dt = cfg.dt;
    fcfg.r_skin = cfg.r_skin;
    fcfg.rebuild_every = cfg.rebuild_every;

    std::fprintf(stderr, "Creating FastPipelineScheduler\n");
    scheduler::FastPipelineScheduler sched(state.box, natoms, params, fcfg);
    sched.upload(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 state.masses.data(), static_cast<i32>(state.masses.size()));

    if (cfg.warmup > 0) {
      std::fprintf(stderr, "Warmup: %d steps...\n", cfg.warmup);
      sched.run_until(cfg.warmup);
    }

    auto stats_before = sched.stats();
    std::fprintf(stderr, "Measuring: %d steps...\n", cfg.steps);
    auto t0 = std::chrono::steady_clock::now();
    sched.run_until(cfg.warmup + cfg.steps);
    auto t1 = std::chrono::steady_clock::now();
    wall_s = std::chrono::duration<double>(t1 - t0).count();

    auto stats_after = sched.stats();
    kernel_launches = stats_after.kernel_launches - stats_before.kernel_launches;
    ticks = stats_after.ticks - stats_before.ticks;
  } else {
    // --- PipelineScheduler path (default) ---
    scheduler::PipelineConfig pcfg;
    pcfg.dt = cfg.dt;
    pcfg.r_skin = cfg.r_skin;
    pcfg.rebuild_every = cfg.rebuild_every;
    pcfg.deterministic = cfg.deterministic;
    pcfg.n_streams = cfg.n_streams;

    std::fprintf(stderr, "Creating PipelineScheduler (deterministic=%d, streams=%d)\n",
                 cfg.deterministic ? 1 : 0, cfg.n_streams);
    scheduler::PipelineScheduler sched(state.box, natoms, params, pcfg);
    sched.upload(state.positions.data(), state.velocities.data(),
                 state.forces.data(), state.types.data(), state.ids.data(),
                 state.masses.data(), static_cast<i32>(state.masses.size()));

    n_zones = sched.partition().n_zones();
    std::fprintf(stderr, "Zones: %d\n", n_zones);

    if (cfg.warmup > 0) {
      std::fprintf(stderr, "Warmup: %d steps...\n", cfg.warmup);
      sched.run_until(cfg.warmup);
    }

    auto stats_before = sched.stats();
    std::fprintf(stderr, "Measuring: %d steps...\n", cfg.steps);
    auto t0 = std::chrono::steady_clock::now();
    sched.run_until(cfg.warmup + cfg.steps);
    auto t1 = std::chrono::steady_clock::now();
    wall_s = std::chrono::duration<double>(t1 - t0).count();

    auto stats_after = sched.stats();
    kernel_launches = stats_after.kernel_launches - stats_before.kernel_launches;
    dep_checks = stats_after.dep_check_calls - stats_before.dep_check_calls;
    dep_failures = stats_after.dep_check_failures - stats_before.dep_check_failures;
    ticks = stats_after.ticks - stats_before.ticks;
  }

  double timesteps_per_s = wall_s > 0 ? static_cast<double>(cfg.steps) / wall_s : 0;
  double launches_per_step =
      cfg.steps > 0 ? static_cast<double>(kernel_launches) / cfg.steps : 0;

  // Output JSON.
  FILE* out = stdout;
  FILE* file_out = nullptr;
  if (!cfg.output.empty()) {
    file_out = std::fopen(cfg.output.c_str(), "w");
    if (file_out) {
      out = file_out;
    } else {
      std::fprintf(stderr, "Warning: cannot open %s, writing to stdout\n",
                   cfg.output.c_str());
    }
  }

  std::fprintf(out,
               "{\n"
               "  \"scheduler\": \"%s\",\n"
               "  \"data_file\": \"%s\",\n"
               "  \"n_atoms\": %d,\n"
               "  \"n_zones\": %d,\n"
               "  \"n_steps\": %d,\n"
               "  \"warmup_steps\": %d,\n"
               "  \"deterministic\": %s,\n"
               "  \"n_streams\": %d,\n"
               "  \"dt\": %.6f,\n"
               "  \"wall_clock_s\": %.6f,\n"
               "  \"timesteps_per_s\": %.2f,\n"
               "  \"kernel_launches_total\": %lld,\n"
               "  \"kernel_launches_per_step\": %.2f,\n"
               "  \"dep_check_calls\": %lld,\n"
               "  \"dep_check_failures\": %lld,\n"
               "  \"ticks\": %lld\n"
               "}\n",
               cfg.scheduler_name.c_str(), cfg.data_file.c_str(), natoms,
               n_zones, cfg.steps, cfg.warmup,
               cfg.deterministic ? "true" : "false", cfg.n_streams,
               static_cast<double>(cfg.dt), wall_s, timesteps_per_s,
               static_cast<long long>(kernel_launches), launches_per_step,
               static_cast<long long>(dep_checks),
               static_cast<long long>(dep_failures),
               static_cast<long long>(ticks));

  if (file_out) std::fclose(file_out);

  std::fprintf(stderr,
               "\n=== Measurement Results ===\n"
               "Scheduler: %s  Atoms: %d  Zones: %d  Steps: %d (warmup: %d)\n"
               "Wall clock: %.3f s\n"
               "Timesteps/s: %.2f\n"
               "Kernel launches: %lld (%.1f per step)\n"
               "Ticks: %lld  Dep checks: %lld (failures: %lld)\n"
               "===========================\n",
               cfg.scheduler_name.c_str(), natoms, n_zones, cfg.steps,
               cfg.warmup, wall_s, timesteps_per_s,
               static_cast<long long>(kernel_launches), launches_per_step,
               static_cast<long long>(ticks),
               static_cast<long long>(dep_checks),
               static_cast<long long>(dep_failures));

  return 0;
}
