// SPDX-License-Identifier: Apache-2.0
// error.hpp — TDMD-wide assertion and error macros.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tdmd {

/// Recoverable user error: bad input file, missing potential, etc.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace tdmd

/// Always-on assertion. Stays enabled in Release for the first year of development.
#define TDMD_ASSERT(cond, msg)                                                      \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "TDMD_ASSERT failed: %s\n  at %s:%d\n  message: %s\n",   \
                   #cond, __FILE__, __LINE__, (msg));                               \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

/// Throws tdmd::Error with formatted message.
#define TDMD_THROW(msg)                                                             \
  do {                                                                              \
    throw ::tdmd::Error(std::string("TDMD error: ") + (msg) + "\n  at " +           \
                        __FILE__ + ":" + std::to_string(__LINE__));                 \
  } while (0)
