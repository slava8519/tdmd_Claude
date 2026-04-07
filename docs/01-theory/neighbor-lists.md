# Neighbor Lists

> Verlet lists with skin, plus cell lists for the build.

## Why neighbor lists

In MD, the force on atom `i` depends only on neighbors within `r_c`. A naive `O(N²)` loop over all pairs is wasteful — for `N = 32 000` atoms with `r_c = 5 Å` in a box where atoms have ~50 neighbors each, we should be doing `O(50 N) = 1.6M` pair evaluations, not `1B`.

Two structures together solve this:

1. **Cell list** — partition the box into cubic cells of side `≥ r_list`. Each atom is in exactly one cell. Neighbors of atom `i` are searched only in the cell of `i` and its 26 neighboring cells (3D). This brings the build cost from `O(N²)` to `O(N)`.

2. **Verlet list** — for each atom, store the list of neighbors found within `r_list = r_c + r_skin`. Use this list for force calculations until any atom moves more than `r_skin / 2`, at which point we rebuild.

## The skin trick

A naive Verlet list (without skin) needs to be rebuilt every step, because atoms move every step. The skin idea: store neighbors within `r_c + r_skin`, which is more than we need *right now* but means we can reuse the list for several steps.

The list is valid as long as:

```
max(displacement of any atom since rebuild) < r_skin / 2
```

Why `/2`? Because two atoms can both move toward each other; the worst case is each moves `r_skin/2`, putting them `r_skin` closer, which is exactly the buffer we built in.

Typical `r_skin` for metals: 1.0 Å. Typical rebuild frequency: every 10–50 steps.

## Connection to TD

The skin is also what makes TD's **K parameter** work. If `r_skin` is large enough that the list survives K steps, you can batch K steps before rebuilding (and before sending to the next rank).

Constraint:

```
K · v_max · Δt < r_skin / 2
```

For typical metal MD (`v_max ≈ 10 Å/ps`, `Δt = 1 fs = 0.001 ps`, `r_skin = 1 Å`), this gives:

```
K < 0.5 / (10 × 0.001) = 50
```

So we can batch up to ~50 steps in a single comm cycle. This is the bandwidth-reduction knob that TD uniquely has.

## CSR layout (for GPU)

On GPU we store the neighbor list in CSR (Compressed Sparse Row) format:

```
neighbors[]         : flat array of neighbor indices
neighbor_offsets[i] : start of atom i's neighbor list in neighbors[]
neighbor_counts[i]  : number of neighbors of atom i
```

Then to iterate neighbors of `i`:

```cpp
for (int k = 0; k < neighbor_counts[i]; ++k) {
  int j = neighbors[neighbor_offsets[i] + k];
  // ... compute force from j on i
}
```

Coalesced reads on GPU; cache-friendly on CPU.

## Newton's third law

Two storage modes:

1. **Half-list** — for each pair `(i, j)` store only one direction (`i < j`). Each force is computed once and applied to both atoms (Newton 3rd law). Saves memory and compute.

2. **Full-list** — for each pair `(i, j)` store both directions. Forces are computed twice. Wastes compute but allows lock-free GPU scatter.

TDMD uses **full-list on GPU** (lock-free scatter) and **half-list on CPU** (less memory). Both produce the same physics.

## Build algorithm (cell-list-based)

1. Compute box-relative cell index for each atom: `cell_id = (x / r_list, y / r_list, z / r_list)`.
2. Build a counting sort: how many atoms in each cell?
3. Compute prefix sum to get the start offset for each cell.
4. Place atoms into the sorted array.
5. For each atom `i`, loop over its cell and 26 neighboring cells, test `r² < r_list²`, append to neighbor list.

On GPU all five steps are kernels. Standard technique, well-known in the literature.

## Tests required (M1/M2)

- **Cell consistency** — every atom's cell index is correct given its position.
- **No-loss** — every neighbor within `r_list` appears in the list.
- **No-gain** — every pair in the list is within `r_list`.
- **Rebuild trigger** — synthetic atom moving > `r_skin/2` triggers rebuild.
- **CPU vs GPU agreement** — same input, same neighbor list (modulo ordering).
