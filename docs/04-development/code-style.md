# Code Style — C++ / CUDA / Python

> The full version of the rules summarized in `CLAUDE.md` §5.
> When this doc and `CLAUDE.md` disagree, **`CLAUDE.md` wins**.

---

## 1. Language

**C++20** for host code. **CUDA C++** with `--std=c++20` for device code. **Python 3.10+** for VerifyLab runners and benchmark scripts.

No C, no C++23 yet (compiler support too uneven). No Python 2.

---

## 2. File layout

```
src/<module>/
    foo.hpp        # public interface, doc-commented
    foo.cpp        # implementation
    foo_impl.hpp   # private helpers if needed (rare)
    foo.cu         # CUDA kernels (M2+)
    CMakeLists.txt
```

- Headers and sources live **next to each other**. There is no separate `include/` tree.
- One major class or one major free-function family per file.
- Max ~500 lines per file is a soft target. If you're past it, split.

---

## 3. Naming

| Kind | Style | Example |
|---|---|---|
| Type / class / struct / enum | `PascalCase` | `SystemState`, `ZoneState` |
| Function / method / variable | `snake_case` | `compute_forces`, `natoms` |
| Constants (constexpr) | `kPascalCase` | `kCutoffDefault` |
| Macros (rare) | `SCREAMING_SNAKE_CASE` | `TDMD_ASSERT` |
| Namespace | lowercase | `tdmd::scheduler` |
| Template parameters | `PascalCase` | `template <typename Real>` |
| Files | `snake_case` | `system_state.hpp` |

**No Hungarian notation.** No `m_` prefix for members. No `_t` suffix for types.

---

## 4. Includes

Order, separated by blank lines:

```cpp
// 1. The matching header for this .cpp (if any)
#include "foo.hpp"

// 2. C system headers
#include <cstdint>
#include <cstdio>

// 3. C++ standard library
#include <string>
#include <vector>

// 4. Third-party libraries
#include <cuda_runtime.h>

// 5. TDMD project headers
#include "core/error.hpp"
#include "core/types.hpp"
```

`clang-format` enforces this automatically (`IncludeBlocks: Regroup`).

Always `#pragma once`. No include guards.

---

## 5. Memory and ownership

- **No naked `new` / `delete`.** Use `std::unique_ptr`, `std::vector`, RAII wrappers.
- **Owning pointer** → `std::unique_ptr<T>`.
- **Non-owning, non-null** → reference (`T&`).
- **Non-owning, nullable** → raw pointer (`T*`).
- **Shared ownership** is suspicious. Use `std::shared_ptr` only when you really mean it.
- **Device memory** wraps in `DeviceBuffer<T>` (defined in `src/core/device_buffer.hpp` at M2). Never pass raw `cudaMalloc`'d pointers around.

---

## 6. Const-correctness

- **Default to const.** Method that doesn't mutate state → `const`. Parameter that isn't modified → `const T&`.
- **`constexpr`** everywhere it's free.
- **`noexcept`** on functions that genuinely cannot throw (most leaf math functions).

---

## 7. Error handling

| Situation | Mechanism |
|---|---|
| Invariant violation (a bug) | `TDMD_ASSERT(cond, msg)` |
| User error at IO/setup | `throw tdmd::Error(...)` via `TDMD_THROW(msg)` |
| CUDA call failed | `TDMD_CHECK_CUDA(call)` |
| Recoverable runtime condition | return `std::optional<T>` or `std::expected<T, E>` |

**Never** swallow an exception silently. **Never** use `errno`. **Never** use `assert()` from `<cassert>` (it's compiled out in Release).

---

## 8. Logging

```cpp
#include "core/log.hpp"
tdmd::log::info("loaded N=" + std::to_string(natoms));
```

- No `printf`. No `std::cout`. Use `tdmd::log::*`.
- One log line per significant event, not per atom.
- Debug logging is gated by `TDMD_LOG=<module>:debug` env var (M1+).

---

## 9. Comments

- **Why, not what.** "Sort by zone id to keep neighbor list cache-friendly" — yes. "Loop over zones" — no.
- **Doc comments** on public functions/types use `///` before the declaration. Format follows Doxygen-lite:
  ```cpp
  /// Builds a Verlet neighbor list for all atoms within rcut + rskin.
  /// @param state SystemState containing positions and box.
  /// @param rcut  Force cutoff in Angstroms.
  /// @param rskin Skin distance for the Verlet algorithm.
  /// @returns A NeighborList valid until any atom moves more than rskin/2.
  NeighborList build_neighbor_list(const SystemState& state, real rcut, real rskin);
  ```
- **TODO comments** must reference an issue: `// TODO(#42): handle triclinic boxes`.

---

## 10. CUDA-specific rules

- **One kernel per `.cu` file.** Helpers go in `<name>_kernels.cuh`.
- **Always check** with `TDMD_CHECK_CUDA(...)`.
- **No `cudaDeviceSynchronize()`** in the hot loop — use streams and events.
- **Prefer `__restrict__`** on kernel pointer params.
- **SoA, not AoS.** Always.
- **Mixed precision** is the default: FP32 storage, FP64 accumulators where it matters (force sums, energy sums).
- **Coalescing** before cleverness. Verify with `nsys` / `ncu`.
- **No template-heavy CUDA code** unless there's a measured win. Templates make `.cu` files harder to debug.

---

## 11. Python (VerifyLab, scripts)

- Python 3.10+, type hints throughout, formatted with `black`, linted with `ruff` (when we add it).
- Scripts at top of file:
  ```python
  #!/usr/bin/env python3
  """One-line description.

  Longer description.
  """
  from __future__ import annotations
  ```
- Standard library first; no heavy frameworks unless justified.
- VerifyLab cases use only `numpy`, `scipy`, `matplotlib`. Nothing else.

---

## 12. CMake

- **One `CMakeLists.txt` per directory** that produces a target.
- **Targets, not directories.** Use `target_*` everywhere; never globally `include_directories` or `add_definitions`.
- **No globbing of source files.** List them explicitly so the build is reproducible.
- **`tdmd::*` aliases** for libraries: `add_library(tdmd::core ALIAS ...)`.

---

## 13. Things we just don't do

- `using namespace std;` at file scope.
- Macros for anything other than the four `TDMD_*` ones.
- Multiple inheritance.
- Operator overloading except for math types (`Vec3` ops).
- Friend declarations.
- Global mutable state.
- C-style casts. Use `static_cast`, `reinterpret_cast`, `const_cast`.
- Raw arrays. Use `std::array`, `std::vector`, or `std::span`.
- `auto` for variables whose type is not obvious from the right-hand side. (`auto x = make_unique<Foo>()` — fine. `auto y = thing.compute();` — write the type.)

---

## 14. When in doubt

Read the existing code in `src/core/`. It is the reference for style across the project. If your code looks different from `src/core/system_state.hpp`, your code is probably wrong about style.
