// SPDX-License-Identifier: Apache-2.0
// log.hpp — minimal structured logging facade. M0 stub, real impl at M1.
#pragma once

#include <cstdio>
#include <string_view>

namespace tdmd::log {

enum class Level { Debug, Info, Warn, Error };

inline void info(std::string_view msg) {
  std::fprintf(stdout, "[info ] %.*s\n", static_cast<int>(msg.size()), msg.data());
}
inline void warn(std::string_view msg) {
  std::fprintf(stderr, "[warn ] %.*s\n", static_cast<int>(msg.size()), msg.data());
}
inline void error(std::string_view msg) {
  std::fprintf(stderr, "[error] %.*s\n", static_cast<int>(msg.size()), msg.data());
}
inline void debug(std::string_view msg) {
  // M1: gate behind TDMD_LOG env var
  std::fprintf(stdout, "[debug] %.*s\n", static_cast<int>(msg.size()), msg.data());
}

}  // namespace tdmd::log
