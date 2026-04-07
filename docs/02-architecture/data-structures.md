# Architecture: Data Structures

> The fundamental data types in TDMD. If a structure isn't here, it's an internal implementation detail of one module.

## SystemState

The single canonical representation of the simulation. Lives in `src/core/system_state.hpp`.

```cpp
namespace tdmd::core {

struct SystemState {
  int64_t natoms{0};
  Box box;

  // device-resident SoA
  DeviceBuffer<Vec3>    positions;
  DeviceBuffer<Vec3>    velocities;
  DeviceBuffer<Vec3>    forces;
  DeviceBuffer<int32_t> types;
  DeviceBuffer<int32_t> ids;       // stable global atom IDs

  // host-side metadata
  std::vector<Real>        masses;       // indexed by type
  std::vector<std::string> type_names;

  // simulation clock
  int64_t step{0};
  Real    time{Real(0)};
};

} // namespace tdmd::core
```

Key invariants:

- `positions.size() == velocities.size() == forces.size() == types.size() == ids.size() == natoms`.
- `ids[i]` is unique and stable across migrations.
- Force buffer is **zeroed** at the start of every force pass.
- `step` and `time` are global; per-zone time lives in `Zone`.

## Box

Periodic simulation cell. Orthorhombic at M1; triclinic at M7+.

```cpp
struct Box {
  Vec3 lo;     // lower corner
  Vec3 hi;     // upper corner
  Vec3 size;   // hi - lo
  bool periodic[3] = {true, true, true};
};
```

Helper methods: `wrap(Vec3&)`, `min_image(Vec3 a, Vec3 b)`, `volume()`.

## Zone

Defined in `src/scheduler/zone.hpp`. See `docs/01-theory/zone-state-machine.md` for the full state machine.

```cpp
struct Zone {
  int32_t id;
  std::array<int32_t, 3> lattice_index;
  Aabb bbox;
  int32_t natoms_in_zone;
  int32_t atom_offset;
  int32_t time_step;
  ZoneState state;
  int32_t owner_rank;
  cudaEvent_t done_event;
};
```

## NeighborList (CSR)

```cpp
struct NeighborList {
  DeviceBuffer<int32_t> neighbors;
  DeviceBuffer<int32_t> neighbor_offsets;
  DeviceBuffer<int32_t> neighbor_counts;
  Real r_cut;
  Real r_skin;
  int64_t built_at_step;
};
```

CSR layout: `neighbors_of_atom_i = neighbors[neighbor_offsets[i] : neighbor_offsets[i+1]]`.

## Vec3 / Real

```cpp
namespace tdmd::core {
#ifdef TDMD_DOUBLE_PRECISION
using Real = double;
using Vec3 = double3;
#else
using Real = float;
using Vec3 = float3;
#endif
}
```

Selected at compile time. Default is FP32. Deterministic mode forces FP64 via `-DTDMD_DOUBLE_PRECISION=ON`.

## DeviceBuffer<T>

A thin RAII wrapper over `cudaMalloc` / `cudaFree`. Owns its allocation. Copyable only via explicit `clone()`. No implicit host↔device transfers.

```cpp
template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept;

  T* data() noexcept;
  const T* data() const noexcept;
  size_t size() const noexcept;
  void resize(size_t n);
  void copy_from_host(const T* src, size_t n);
  void copy_to_host(T* dst, size_t n) const;
};
```

## What does NOT belong in core data structures

- Force kernel implementations (live in `potentials/`).
- Integration logic (lives in `integrator/`).
- Neighbor list build (lives in `neighbors/`).
- Zone state machine logic (lives in `scheduler/`).
- IO routines (live in `io/`).

`core` is for shared types, not shared logic.
