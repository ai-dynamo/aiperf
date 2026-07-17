<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: sub-cell partitioning — how each workload shape splits across W threads

**Date:** 2026-07-16
**Author:** Anthony Casagrande (Tech Lead) + Claude
**Status:** built — documents the final partitioning math of the unified thread-per-core execution model
**Grounding:** end-to-end read of `rust/runtime/src/engine/sharded_scheduled.rs`
(`two_level_partition`, `slice_phase_for_thread`, `slice_common`,
`compute_phase_ordinal_bases`), `rust/runtime/src/cellular/partition.rs`
(`ModuloCellPartition`), `rust/runtime/src/engine/cell_launcher.rs`
(`owned_positions`), `rust/runtime/src/dataset/sampler.rs` (`PartitionedSampler`),
and `rust/runtime/src/multiturn.rs` (`NativeDatasetConversationSource::new_with_endpoint`).

One sentence: a run is a two-level `(cell × thread)` modulo partition — this
process is cell `c` of `cells`, and within it thread `t` of `W` owns the nested
residue class `c + cells·t (mod cells·W)`, and every workload shape is split so the
union across all cells and threads is a permutation of the whole run.

Read the companion spec `2026-07-16-unified-thread-per-core-execution.md` first — it
covers WHY there is one execution model; this spec covers HOW the run is divided
across that model's `W` sub-cell threads.

---

## 1. The primitive: `ModuloCellPartition`

A `CellPartition` (`partition.rs:31-50`) is the seam that lets `cell_count` cells
produce the same trace *set* as one cell, with different *ownership*. The only
shipping impl is the round-robin `ModuloCellPartition` (`partition.rs:61-65`):

```text
owns(i)  ≡  i % cell_count == cell_id          (partition.rs:133-136)
```

Its contract (`partition.rs:26-30`): the `u64` instance-index space is split into
`cell_count` disjoint classes whose union is complete — exactly one `cell_id` owns
each `i`. Construction validates `cell_count >= 1` and `cell_id < cell_count`
(`partition.rs:100-114`) so it can never silently drop or double-own an instance.
The key algebraic property the whole design leans on
(`partition.rs:55-60`, test `partition.rs:231-242`): a cell's `n`-th owned instance
in ascending order is exactly `n·cell_count + cell_id`, so a cell-local counter
reconstructs a dense global dispatch ordinal.

The identity partition `(0, 1)` (`ModuloCellPartition::direct`, `partition.rs:77-82`)
owns the entire space and is the single-process default; process-level cells read
`(cell_id, cell_count)` from `AIPERF_CELL_ID` / `AIPERF_CELL_COUNT`
(`partition.rs:88-92`).

## 2. The two-level `(cell × thread)` grid

A process is cell `c` of `cells`; within it, thread `t` of `W` owns a nested slice.
`two_level_partition` (`sharded_scheduled.rs:105-125`) builds it:

```text
index   = cell_id + cells * thread_id
modulus = cells * workers
partition_t = ModuloCellPartition::new(index, modulus)     (sharded_scheduled.rs:113-123)
```

The design note originally wrote the flat `c·W + t`, but that index does **not**
nest inside a controller child's already-cell-sliced envelope
(`sharded_scheduled.rs:20-45`). When this process is a controller child,
`build_cell_envelope` has *already* sliced each phase to cell `c`'s round-robin share
of the global stream (`i % cells == c`), and each thread must draw a subset of *that*
share, not of the whole global stream. The unique modulo family that (a) is a modulo
partition, (b) nests inside cell `c`'s `(c, cells)` ownership, and (c) tiles the
global `cells·W` grid is `c + cells·t`, because:

```text
c + cells·t  ≡  c   (mod cells)             ⟹ nests inside cell c
{c, c+cells, …, c+(W-1)·cells}              ⟹ the W threads partition cell c exactly
```

The flat `c·W + t` fails (b): its residue `≡ c·W + t (mod cells)` is generally not
`c`, so a thread would draw instances belonging to *other* cells and the merge would
overflow its residue class. For a single process (`cells == 1`) the two formulas
coincide (`0 + 1·t == 0·W + t == t`, modulus `W`), so the correction only matters for
the multi-process grid. Both properties are machine-checked in
`two_level_partition_nests_and_tiles` (`sharded_scheduled.rs:428-455`).

```text
                     THE TWO-LEVEL (cell × thread) GRID
       global instance space  {0, 1, 2, …, total-1}   (modulus cells·W)
       ─────────────────────────────────────────────────────────────────

  cells = 3   (this process = cell c),   W = 2 threads per cell
  ═══════════════════════════════════════════════════════════════════════

  controller slices GLOBAL stream by cells:   owns_cell(i) = i % 3 == c
  ┌──────────────┬──────────────┬──────────────┐
  │  cell 0      │  cell 1      │  cell 2      │   ← process-level (AIPERF_CELL_*)
  │  i%3==0      │  i%3==1      │  i%3==2      │
  └──────┬───────┴──────┬───────┴──────┬───────┘
         │ each cell then splits its share across W=2 threads
         ▼              ▼              ▼         ← thread-level (sub-cell, in-process)
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ residues  │  │ residues  │  │ residues  │   modulus = cells·W = 6
   │ within    │  │ within    │  │ within    │
   │ cell 0:   │  │ cell 1:   │  │ cell 2:   │
   │           │  │           │  │           │
   │ t0: idx 0 │  │ t0: idx 1 │  │ t0: idx 2 │   index = c + cells·t
   │  (0 mod 6)│  │  (1 mod 6)│  │  (2 mod 6)│         = c + 3·t
   │ t1: idx 3 │  │ t1: idx 4 │  │ t1: idx 5 │
   │  (3 mod 6)│  │  (4 mod 6)│  │  (5 mod 6)│
   └───────────┘  └───────────┘  └───────────┘
        └──────────────┴──────────────┘
        the 6 residues {0..5} tile the cells·W grid EXACTLY once
        ⟹ union over all (cell, thread) = permutation of 0..total
```

Both the sampler and the issuer receive the SAME `partition` object inside one shard
(`execute.rs:2347-2356` builds it; `execute.rs:2425` feeds the issuer via
`issuance_authority_for(partition)`; `execute.rs:2437-2439` feeds the sampler via
`cell_partition: Some(partition)`), so "which instance it draws" and "which global
ordinal it stamps" stay in lockstep and the ordinals tile `0..total`.

## 3. Per-workload partition strategy

Each workload shape is partitioned by whichever axis is natural to it. This is
`slice_phase_for_thread` (`sharded_scheduled.rs:140-218`), which mirrors the
controller's per-cell arithmetic one level down, using the *cell's* budget divided by
`W` (never `cells·W` — the envelope is already cell-sliced;
`sharded_scheduled.rs:46-58`).

The two request-budget helpers (`sharded_scheduled.rs:147-151`):

```text
owned_budget(v) = owned_positions(v, t, W)              round-robin request share
owned_cap(v)    = owned_positions(v, t, W).max(1)       admission cap, floored to 1
scaled_rate(r)  = r / W                                  even rate split
```

where `owned_positions(total, id, count) = (total - id).div_ceil(count)` when
`id < total`, else `0` (`cell_launcher.rs:205-212`) — the count of indices `id`
owns under round-robin, which sums to `total` across ids.

| Workload shape (`PhaseSpec`) | Split axis | What is sliced | Source |
|---|---|---|---|
| `Concurrency` | request budget | `requests → owned_budget`; `concurrency → owned_cap`; `prefill_concurrency → owned_cap` | `sharded_scheduled.rs:155-161`, `:221-232` |
| `Poisson` / `Constant` | request budget + rate | `requests → owned_budget`; `rate → rate/W`; optional `concurrency → owned_cap` | `sharded_scheduled.rs:162-177` |
| `Gamma` | request budget + rate | same as Poisson/Constant (shape param untouched) | `sharded_scheduled.rs:178-189` |
| `UserCentric` | request budget + rate + users | `requests → owned_budget`; `rate → rate/W`; `users → owned_cap`; optional `concurrency → owned_cap`; **draws its 1/W conversation subset from the partitioned sampler** | `sharded_scheduled.rs:197-209` |
| `FixedSchedule` | **per conversation** | phase returned **unchanged** — no budget/rate to slice; the trace is split by per-conversation ownership in the dataset source instead | `sharded_scheduled.rs:210-216` |

Two families, two mechanisms:

- **Budget/rate-driven** (concurrency, poisson, constant, gamma, and the
  open-loop knobs of user_centric): the *phase spec itself* is sliced. Slicing the
  cell's already-cell-sliced `requests` by `owned_positions(·, t, W)` yields
  per-thread dispatch counts that **equal** the flat global two-level share
  `owned_positions(global, c + cells·t, cells·W)` — proven in
  `per_thread_slice_counts_match_global_two_level` (`sharded_scheduled.rs:461-495`)
  — so each thread stamps exactly its residue class and the ordinals tile `0..total`.
  Concurrency and rate examples are pinned in the tests at
  `sharded_scheduled.rs:497-540` (e.g. thread 0 of 4 of a 100-request/8-concurrency
  phase → 25 requests, 2 concurrency; rate 10.0 → 2.5 per thread).

- **Trace-driven** (`user_centric` and `fixed_schedule`): the *conversation set* is
  sliced, per §4. `slice_phase_for_thread` leaves `fixed_schedule` untouched
  because it has no budget/rate — it replays one authored schedule per conversation,
  and each sub-cell already owns a disjoint conversation subset
  (`sharded_scheduled.rs:210-216`).

## 4. Trace-driven shapes: per-conversation ownership (the enumerate = sample proof)

Trace-driven workloads select conversations two different ways, and the design's
correctness hinges on those two ways selecting the **identical** subset:

1. **Sample-based** (`user_centric`, and any random/sequential sampler): the sampler
   is wrapped in a `PartitionedSampler` (`sampler.rs:279-333`). On each `.next()` it
   pulls from the inner sampler and skips any draw whose *position* the partition does
   not own (`sampler.rs:322-332`): `owns(position)` over an incrementing counter. A
   multi-cell partition (`cell_count > 1`) applies the filter; identity /`None`
   returns the inner sampler unchanged so single-process sampling is byte-identical
   (`sampler.rs:311-319`).

2. **Enumerate-based** (`fixed_schedule`, and `user_centric`'s conversation
   enumeration): these workloads iterate `conversations()` directly rather than
   drawing through the sampler. So `NativeDatasetConversationSource::new_with_endpoint`
   restricts the enumerated metadata to the partition-owned authored indices
   (`multiturn.rs:1259-1297`): it enumerates `dataset.sampleable_metadata()`, and
   `filter`s each `(index, _)` by `enumeration_partition.owns(index)`
   (`multiturn.rs:1263-1270`). The enumeration partition is applied only for a
   real multi-cell partition (`cell_count > 1`); identity keeps every conversation
   (`multiturn.rs:1259-1262`).

The two must agree, and they do **by construction**: both use the *same*
`ModuloCellPartition::owns` ownership test over the *same* authored root order
(`multiturn.rs:1249-1258` states this invariant explicitly — the enumeration filter
is "the same `owns(position)` ownership the sampler applies to `.next()` over the same
authored root order, so the enumerate-based and sample-based partitions select the
identical subset and the W threads tile the conversation space exactly"). A sharded
sub-cell injects its per-thread partition through the
`*_for_partition` constructors (`multiturn.rs:1116-1150`, `:1195-1223`) — a
`Some(partition)` the process-global `AIPERF_CELL_ID`/`_COUNT` env cannot express;
`None` reads the env (byte-unchanged single-process default). The partition flows into
`new_with_endpoint`, where it wraps the sampler (`multiturn.rs:1243-1248`,
`PartitionedSampler::for_partition`) AND filters the enumeration
(`multiturn.rs:1259-1297`).

```text
      TRACE-DRIVEN: two selection routes, ONE owned subset
      ════════════════════════════════════════════════════

  authored conversations (root order):  [ c0  c1  c2  c3  c4  c5  c6  … ]
                                           0   1   2   3   4   5   6   position

  this shard's partition owns positions where owns(position) == true
  (e.g. partition (index=1, modulus=3):    owns(1),owns(4),owns(7)… )

   route 1: SAMPLE                         route 2: ENUMERATE
   PartitionedSampler.next()               new_with_endpoint filter
   (sampler.rs:322-332)                    (multiturn.rs:1263-1270)
        │                                       │
        │ inner.next() → id                     │ sampleable_metadata()
        │ if owns(position): yield              │   .enumerate()
        │ else: skip, position++                │   .filter(owns(index))
        ▼                                       ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │ owned draws:         │   IDENTICAL  │ owned metadata:      │
   │  c1, c4, c7, …       │  ◀════════▶  │  c1, c4, c7, …       │
   └──────────────────────┘   subset     └──────────────────────┘
   (user_centric draws)                  (fixed_schedule replays,
                                          user_centric enumerates users)

   ⟹ across W threads the owned subsets are disjoint and tile the full
     conversation space exactly (no W× duplication, no dropped trace)
```

Without this, every thread would replay the whole trace — a `W×` duplication bug the
enumeration filter exists specifically to prevent (`multiturn.rs:1249-1258`).

## 5. Phase-ordinal-base tiling: keeping global ordinals dense across phases

Partitioning divides *within* a phase; phase-ordinal bases keep ordinals dense
*across* phases. The autonomous issuer stamps
(`sharded_scheduled.rs:67-72`):

```text
ordinal = phase_base + within · (cells·W) + (c + cells·t)
```

`within` is the thread's cell-local per-phase counter; `(c + cells·t)` is the
thread's residue; `cells·W` is the stride; and `phase_base` is the number of turns
the run's prior phases dispatched globally. Without `phase_base`, profiling's
ordinals would collide with warmup's `[0, W)` block. The base for each phase is the
running sum of prior phases' `requests`.

Two sources for the base map (`execute.rs:3138-3146`):

- **Controller child** — reads the already-global, already-correct bases from
  `AIPERF_CELL_PHASE_ORDINAL_BASES` (`execute.rs:3141`,
  `phase_ordinal_bases_from_env`). Recomputing them from a cell's *local* sliced
  `requests` would understate them, so the env map is preferred when present.
- **Lone process** — `compute_phase_ordinal_bases(&request.phases)`
  (`sharded_scheduled.rs:244-257`) derives them from the phase `requests` budgets:
  it walks the phases, records `base` for each metric phase, then advances
  `base += phase.common().requests.unwrap_or(0)`. So warmup → base 0, profiling →
  base = warmup's request count. Pinned in
  `phase_ordinal_bases_are_cumulative_prior_requests` (`sharded_scheduled.rs:542-552`,
  e.g. warmup=8 → profiling base=8).

The resolved map is stored once in `ShardedShared.phase_ordinal_bases`
(`execute.rs:2228-2230`, `execute.rs:3199`) and injected *identically* into every
thread's issuer (`execute.rs:2426`) — it is partition-independent, so the phase
tiling is orthogonal to the per-thread residue tiling.

```text
   PHASE-ORDINAL-BASE TILING  (single process, cells=1, W threads)
   ═══════════════════════════════════════════════════════════════
   phases:  [ warmup: 8 reqs ] [ profiling: 200 reqs ]

   compute_phase_ordinal_bases →  { Warmup: 0, Profiling: 8 }
                                                    ▲
                            base = Σ prior phases' requests

   global ordinal = phase_base + within·(cells·W) + (c + cells·t)
                    ▲            ▲                   ▲
                    │            │                   └ thread residue (0..cells·W)
                    │            └ per-thread cell-local counter × stride
                    └ 0 for warmup, 8 for profiling  → NO cross-phase collision

   warmup block:     ordinals  0 .. 7      (dense, tiled across W threads)
   profiling block:  ordinals  8 .. 207    (dense, tiled across W threads)
```

## 6. Why merge needs only a sort, not a renumber

Because (a) the residues `c + cells·t` tile the `cells·W` grid exactly
(`sharded_scheduled.rs:428-455`), (b) each thread's sliced request count equals its
flat global two-level share (`sharded_scheduled.rs:461-495`), and (c) phase bases
keep phases from overlapping (`sharded_scheduled.rs:542-552`), the union of every
shard's stamped ordinals is a **permutation of `0..total`**. So `merge_shards`
(`sharded_scheduled.rs:388-417`) concatenates the retained record Vecs and simply
`sort_by_key(request_index)` (`sharded_scheduled.rs:410-412`); the sorted records
tile the store's `insert_record_at` slots with no collision and no renumber, and the
row order is topology-deterministic (independent of racy thread completion order).
Fold-and-drop shards keep only errored records, whose order is irrelevant to error
grouping, so there is nothing to sort there.

---

## Appendix — primary source map

| Claim | Source |
|---|---|
| `ModuloCellPartition::owns(i) = i % count == id` | `partition.rs:133-136` |
| n-th owned instance = `n·count + id` | `partition.rs:55-60`, test `:231-242` |
| identity `(0,1)` default / `from_env` | `partition.rs:77-92` |
| two-level index = `c + cells·t`, modulus `cells·W` | `sharded_scheduled.rs:105-125` |
| why flat `c·W + t` fails to nest | `sharded_scheduled.rs:20-45` |
| nest + tile property (test) | `sharded_scheduled.rs:428-455` |
| `slice_phase_for_thread` per-shape slicing | `sharded_scheduled.rs:140-218` |
| `owned_positions` round-robin count | `cell_launcher.rs:205-212` |
| nested slice = flat two-level share (test) | `sharded_scheduled.rs:461-495` |
| concurrency/rate slice examples (tests) | `sharded_scheduled.rs:497-540` |
| `PartitionedSampler` ownership filter on `.next()` | `sampler.rs:279-333` |
| enumerate filter = same `owns` subset | `multiturn.rs:1249-1297` |
| per-thread partition injection (`*_for_partition`) | `multiturn.rs:1116-1150, 1195-1223` |
| sampler wrap + enumeration filter in `new_with_endpoint` | `multiturn.rs:1243-1297` |
| issuer ordinal formula w/ phase_base | `sharded_scheduled.rs:67-72` |
| `compute_phase_ordinal_bases` (lone process) | `sharded_scheduled.rs:244-257`, test `:542-552` |
| controller child reads env bases | `execute.rs:3138-3146` |
| partition feeds both sampler and issuer in a shard | `execute.rs:2347-2356, 2425, 2437-2439` |
| `merge_shards` sort-by global ordinal | `sharded_scheduled.rs:388-417` |
