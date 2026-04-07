// SPDX-License-Identifier: Apache-2.0
// tdmd_main.cpp — standalone driver entry point.
//
// M0: prints version and exits. M1: parses CLI, loads input, runs simulation.

#include <cstdio>
#include <string>

namespace {
constexpr const char* kVersion = "0.0.1-m0";
}

int main(int argc, char** argv) {
  std::printf("tdmd v%s\n", kVersion);
  std::printf("  Time-Decomposition Molecular Dynamics engine\n");
  std::printf("  M0 scaffold — no physics yet. See docs/03-roadmap/milestones.md\n");

  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (arg == "--version" || arg == "-v") {
      // already printed
      return 0;
    }
    if (arg == "--help" || arg == "-h") {
      std::printf("\nUsage: tdmd_standalone [--version] [--help]\n");
      std::printf("  See docs/04-development/build-and-run.md\n");
      return 0;
    }
  }
  return 0;
}
