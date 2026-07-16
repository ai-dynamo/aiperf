<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cellular horizontal scale — reaching k6-class scaling without giving up measurement fidelity

**Status:** Design / analysis + stepwise plan (forward; not built)
**Date:** 2026-07-15
**Repo:** `aiperf` (branch `velo-connect`, on top of `ajc/rust`)
**Companion to:** `2026-07-15-ultimate-cellular-velo-runtime-design.md` (built substrate + forward plane).
This spec is the **scaling gap analysis** against Grafana k6 and the **ordered, tiered step plan**
to close it. It reuses the ultimate spec's forward items (sketch-cellular, streaming finalize, SPMC
fan-out) and sequences them into a fidelity ladder.

> **Grounding.** k6 claims are verified against `grafana/k6` source (`lib/execution_segment.go`,
> `output/`). AIPerf claims cite `rust/aiperf/src/**`. This is designed-but-not-built; verify against
> `rust/` before relying on any step.

---

## 1. The core finding — AIPerf already owns k6's hard part

k6's horizontal scaling rests on **execution segments** (`lib/execution_segment.go:11-24`):

> *"if work is split between multiple k6 instances, each k6 instance can precisely and reproducibly
> calculate its share of the work, just by knowing its own segment. There won't be a need to schedule
> the execution from a master node, or to even know how many other k6 instances are running!"*

This is **exactly AIPerf's model**: k6 uses rational `(from,to]` segments; AIPerf uses modulo
round-robin (`owned_positions`, `CellularAutonomousIssuer` = `phase_base + within*cell_count + cell_id`,
`cellular/{partition,issuance}.rs`) — both static, reproducible, per-instance-local, **no master
scheduler during load**. AIPerf's `!Send` thread-per-core generator is, if anything, a tighter
per-instance engine than k6's goroutine-VU model.

**So the gap is not in load generation.** During the load window AIPerf is already k6-class (no
per-request chokepoint — see the ultimate spec §0 and the "hands-off controller" analysis). The gap
is entirely in the **results / lifecycle plane**, where the two tools differ in topology:

| Axis | k6 | AIPerf today |
|---|---|---|
| Execution partition | static segments, master-less | static round-robin, master-less ✓ (parity) |
| Result aggregation | **stream to external sink**; histogram/approx aggregation at ingest | **collect all → merge in one controller** (byte-exact on retain) |
| Per-instance memory | discard samples after emit (memory-efficient sinks) | **retain every record until end-of-run drain** |
| Cross-instance percentiles | approximate (histogram) | exact (retain) or approximate (sketch, cellular-blocked) |
| Start coordination | none — instances just start | **synchronized START barrier** (O(N) rendezvous) |
| Topology | N → scalable ingest (fan-out / tree) | **N → 1 controller (star)** |

Every AIPerf scaling ceiling (ultimate spec §11-adjacent, and the scaling analysis) is a consequence
of the **star topology + byte-exact retain merge**. k6 trades exactness for a streamed, bounded,
sink-based model. The design question is therefore **not** "become k6" but **"add k6's scalable path
as the top of a fidelity ladder, without losing the byte-exact path that makes AIPerf a *measurement*
tool rather than a *load* tool."**

---

## 2. Invariants — what must never degrade, at any scale

These are AIPerf's differentiators over k6; the plan must preserve them on the paths that keep them:

1. **Exact counts always.** `request_count`, `records_count`, error counts, per-metric sample counts,
   sums, min, max, mean, and std must stay **exact at every tier** (Welford + exact aggregates already
   give this in sketch mode — `metrics_core::store.rs` `TagSketch`).
2. **Exact throughput/rate derivations.** Derived from min/max timestamp aggregates, which stay exact
   even under sketching.
3. **Byte-exact reproducibility remains available.** The retain path's global-order merge byte-parity
   (`merge_records_in_global_order`) is a genuine value for regression/CI and small runs; it stays the
   **default** and is never removed — only *not selected* when scale forbids it.
4. **Timing-breakdown fidelity.** TTFT / ITL / TPOT / e2e per-record semantics are unchanged; only
   their *aggregation* may become approximate (percentiles), never their *measurement*.
5. **Determinism-at-topology.** Any approximate aggregate (t-digest) must stay associative and
   deterministic for a fixed topology (already proven for the sketch merge).

The single thing that legitimately degrades with scale: **percentile exactness** (→ t-digest
approximation) and **per-record artifact availability** (→ dropped on the streaming path) — and only
when a run opts into (or is auto-promoted to) a higher tier.

---

## 3. The design principle — a scale-adaptive fidelity ladder, not a wholesale switch

Do **not** force every run onto k6's approximate, sink-based model. Gate each compromise behind the
scale that requires it, so small/CI runs keep full fidelity and only the largest runs pay k6's price.
Four tiers, each a superset of the prior's scalability and a subset of its fidelity:

| Tier | Aggregation | Per-cell memory | Topology | Start | % exactness | Ceiling |
|---|---|---|---|---|---|---|
| **T0 — Exact** (default today) | retain + global-order merge (byte-exact) | O(records) | star | synchronized | exact | ~100M records / single controller |
| **T1 — Bounded** | sketch `StorePartition` (assoc. t-digest) | O(1) (streaming finalize) | star | synchronized | approx | ~1024 cells, unbounded duration |
| **T2 — Hierarchical** | + tree-merge aggregator tiers | O(1) | tree | synchronized | approx | past single-controller fan-in |
| **T3 — k6-class** | + external streaming sink, no central merge | O(1) | N → scalable ingest | barrier-free (opt) | approx | horizontal / SUT-bound |

Selection: explicit flag per tier, or auto-promote by projected scale (cells × projected records). T0
stays the default; T1 is the workhorse for large runs; T2/T3 are for extreme N. **Counts/sums/rates
stay exact in every tier** (§2).

---

## 4. The steps (ordered by leverage ÷ compromise)

Each step: what, which barrier/tier it serves, built-vs-new, the compromise, effort, dependency.

### Step 1 — Unblock sketch-cellular: ship the merged sketch as a `StorePartition` (→ T1)
> **BUILT — 2026-07-15.** The `ensure!(!sketch_mode)` cell-ship guard is removed; the sketch branch
> ships the folded store via `ship_store` exactly like exact-fold. A monotonic
> `ColumnStore::ingested_total` (serialized, summed on `append_store`, exposed as `ingested_count()`)
> carries the true record total through ship+merge since a sketch store retains no rows; the
> controller's outcome count reads it. Unit-tested by
> `cellular::shard::tests::sketch_store_partitions_merge_matches_single_sketch_and_carry_the_count`;
> e2e by `test_cellular_sketch_matches_single_cell`.

**What.** Remove the `ensure!(!sketch_mode)` cell guard; wire the per-`(phase,tag)` `TagSketch` store
into the existing `CellMessage::StorePartition` ship; the controller already merges stores by append
(`merge_store_partitions`) and t-digest merge is associative/deterministic.
**Why it's #1.** Highest leverage, lowest compromise, **mostly already built** — it collapses the
controller merge from O(total_records) to **O(cells × tags × phases)** and is the precondition for
every higher tier. Lifts scaling barrier #1 (central merge).
**Compromise.** Percentiles approximate; per-record artifacts unavailable on the sketch path — both
already the documented sketch tradeoff, and **opt-in** (`--sketch-metrics`). Counts/sums/min/max/mean/std
stay exact.
**Effort.** Small (one guard + one ship wiring; the sketch, the store, the merge, the t-digest all
exist). **Dependency.** None. **Built today:** `metrics_core::{store,accumulator}` sketch mode,
`cellular::sketch::TDigest`, `merge_store_partitions`. **New:** the cell-side sketch→StorePartition ship.

### Step 2 — Streaming per-worker finalize: fold-and-drop per record (→ T1, per-cell memory)
> **ALREADY BUILT (verified 2026-07-15).** The prior exact-fold/sketch work already fold-and-drops
> per-worker on BOTH paths: `folds_records() = metrics_only(sketch) || exact_fold` gates it; the
> single-thread path folds via `fold_record` at completion and skips the end-of-run drain, and the
> thread-per-core sharded path returns `ShardRecords::Folded { accumulator, errored }` (each shard
> folds into its own bounded accumulator, retaining only errored records). So per-cell peak RSS is
> O(shards × sketch + concurrency), not O(records). No new code was needed — Step 1's unblock is what
> made tier T1 reachable, because Step 2 was already in place.

**What.** Fold each record into the worker-local accumulator **at completion** and drop it (retain only
errored records for grouping), keyed by phase — extending the existing `RunCapture::finish_fold_into`
from finalize-time to per-record-streaming.
**Why.** Bounds the **per-cell peak RSS**, which is set upstream by per-worker observers retaining every
record until drain (the documented caveat that sketch mode does *not* fix). Without this, a single cell
is O(its records) regardless of cell count — the k6 "no instance retains its stream" property. Lifts
barrier #2; enables unbounded per-cell duration.
**Compromise.** Per-record artifacts (records/raw/outputs JSONL, per-record OTLP) unavailable on the
streaming path — same tradeoff as Step 1; keep them on T0. Orthogonal win: benefits single-process runs too.
**Effort.** Medium (per-record worker finalization keyed by phase — the streaming-finalize follow-up
already scoped in CLAUDE.md). **Dependency.** Composes with Step 1. **Built:** `finish_fold_into`
fold-and-drop skeleton. **New:** per-record (not end-of-run) invocation keyed by phase.

### Step 3 — Tree / hierarchical sketch merge: star → tree (→ T2)
**What.** Insert optional intermediate aggregator tiers: cells → sub-collectors → controller, each an
aggregator that **merges its children's `StorePartition` sketches** and forwards one merged partition
up. An aggregator is just a cell that also runs the controller's merge on a fan-in of `fanout` children.
**Why.** For very large N the controller's O(N) fan-in (register + collect + NIC) is the star ceiling.
A tree of associative t-digest merges is O(log N) depth, each node O(fanout × tags × phases). Lifts
barrier #4.
**Compromise.** Minimal — t-digest merge is associative and deterministic-at-topology (already proven);
extra merge levels add negligible t-digest error. Adds topology/orchestration complexity (the JobSet/
operator must lay out the tier tree; cell↔aggregator↔controller addressing over velo connect-by-endpoint).
**Effort.** Medium. **Dependency.** Step 1 (needs bounded partitions to merge in a tree; you cannot
tree-merge O(records) retain partitions). **Reuses:** the exact same `StorePartition`/sketch merge.

### Step 4 — External streaming sink: drop the internal central merge as the scaling path (→ T3)
**What.** A mode where each cell streams its bounded aggregates to a **scalable backend** (OTLP →
collector, Prometheus remote-write, object store for records) instead of shipping a terminal partition
to the controller. AIPerf already emits per-record OTLP + W&B/MLflow.
**Why.** The fullest k6 model — the controller stops being the aggregation point; the TSDB/ingest
aggregates. Lifts barrier #3 (controller fan-in) entirely at the cost of the single internal report.
**Compromise.** The single merged `native-v2.json` becomes optional at extreme scale; you rely on the
external backend for cross-cell aggregation (which must itself do bounded/histogram aggregation — the
same approximation). Biggest divergence from AIPerf's "one authoritative report," so it is the **top
rung**, opt-in for the largest runs only.
**Effort.** Medium (reuse existing OTLP/W&B sinks per-cell; add a "no central merge / stream-only" mode
that skips the collect+merge phase). **Dependency.** Steps 1–2 (bounded aggregates to stream).

### Step 5 — Dataset fan-out: routed SPMC broadcast or shared object store (→ any tier, file datasets)
**What.** Replace controller-serves-file (O(N × size) egress from one NIC) with **routed fan-out** (each
cell pulls only its shard) — the velo SPMC broadcast anchor from the DEP — or, interim, shared object
storage. Synthetic already scales (each cell composes only its owned instances locally, k6-style).
**Why.** File datasets are the one distribution path that is O(N) at the controller. Lifts barrier #5.
**Compromise.** Routed fan-out needs the velo SPMC anchor (external dependency — the filed DEP); until
it lands, shared object store (S3/RWX) is the interim, which reintroduces a shared-storage dependency
AIPerf otherwise avoids. Synthetic runs need nothing.
**Effort.** Large (velo SPMC anchor, external) or Medium (object-store interim). **Dependency.** velo
DEP (§ references) for the hot-RAM routed path.

### Step 6 — Barrier-optional START: scale-adaptive start coordination (→ T3)
**What.** Keep the synchronized START barrier for normal N (cheap, a fidelity win). For unbounded N,
offer a **barrier-free start**: each cell derives `(segment, cell_count)` from config and starts on a
shared wall-clock **epoch** (operator-injected timestamp, NTP-class) or a **fan-out-only phaser with no
fan-in ack** — the k6 "don't need to know N" property.
**Why.** The synchronized START requires the controller to **know N and gather all N registrations**
(O(N) unary calls converging on one process — a thundering herd at extreme N). This is the one place
AIPerf deliberately diverges from k6; making it a **knob** keeps both: tight coordination at normal
scale, master-less start at extreme scale.
**Compromise.** Looser start correlation across cells (arrival-epoch jitter) — a *fidelity* knob, not a
correctness loss (already the aggregate-equivalent model for rate/ramp). Opt-in for T3.
**Effort.** Small–Medium (epoch injection + a start-mode selector; the fan-out phaser is the DEP's
control plane). **Dependency.** Independent of 1–5; the fan-out phaser variant depends on the velo DEP.

---

## 5. Sequencing & recommended first cut

```
Step 1 (sketch StorePartition) ──┬──> T1 reachable
Step 2 (streaming finalize) ─────┘        (bounded controller + bounded per-cell)
        │
        └──> Step 3 (tree merge) ──> T2 (past single-controller fan-in)
        └──> Step 4 (external sink) ─┐
        └──> Step 6 (barrier-free) ──┴──> T3 (k6-class horizontal)
Step 5 (dataset fan-out) ── parallel track, gated on the velo DEP (file datasets only)
```

- **Do first (biggest win, least compromise, mostly built): Steps 1 + 2.** Together they deliver T1 —
  bounded controller merge *and* bounded per-cell memory — which is the single largest jump in reach
  (retain's ~100M-record single-node ceiling → ~1024 cells at unbounded duration) while keeping counts/
  sums/rates exact and leaving T0 byte-exact as the default. This is ~80% of "k6 scale" for ~20% of the
  work, and neither step needs velo changes.
- **Then Step 3** when a real run exceeds one controller's fan-in; it reuses the Step-1 merge in a tree.
- **Steps 4 + 6** are the extreme-scale rungs (external sink + master-less start) — the true k6 model,
  opt-in, accepting the "no single authoritative report / looser start" divergence only where scale
  demands it.
- **Step 5** proceeds independently on the velo DEP timeline; synthetic runs don't need it.

**Net: T1 (Steps 1–2) is the target "reach k6 without compromising too much."** It matches k6's
scalability class for the common case, keeps every exact aggregate AIPerf is trusted for, and preserves
the byte-exact T0 path for CI/regression — the compromises (approximate percentiles, no per-record
artifacts) are opt-in and reversible per run. T2/T3 are there for when the SUT itself can absorb more
than one controller can currently coordinate.

---

## 6. What stays better than k6 (do not regress)

Even at T3, AIPerf keeps advantages k6 lacks, and the plan must not trade them away:
- **Exact counts/sums/rates at every tier** (k6 trend metrics are also approximate, but AIPerf keeps the
  exact-aggregate half via Welford).
- **A byte-exact tier (T0)** for regression/CI — k6 has no equivalent.
- **Inference-native measurement** (TTFT/ITL/TPOT, token accounting, goodput) — unchanged.
- **Deterministic-at-topology aggregates** — reproducible sketches, not just a fire-hose to a TSDB.

---

## 7. Testing

- **Step 1**: sketch-cellular merged report matches a single-cell sketch run on counts/sums/min/max
  exactly and percentiles within t-digest tolerance; N-cell sketch merge is order-independent.
- **Step 2**: a long/high-rate single cell holds bounded RSS (assert peak RSS flat vs record count);
  errored-record grouping still correct.
- **Step 3**: a 3-tier tree (cells → 2 aggregators → controller) reproduces a flat-star sketch merge
  within tolerance; associativity across tiers.
- **Step 4**: stream-only mode emits per-cell OTLP that a collector aggregates to the same exact counts;
  no central `native-v2.json` required.
- **Step 6**: barrier-free start over a shared epoch yields aggregate-equivalent arrival distribution vs
  the synchronized-start run.
- **Regression**: every T0 byte-parity assertion (existing `test_cellular*.rs`) stays green — the ladder
  must not perturb the exact path.

---

## 8. References

- **Built substrate + forward plane**: `2026-07-15-ultimate-cellular-velo-runtime-design.md` (S1–S5
  seams, velo transport, sketch, the SPMC/phaser north-star).
- **velo primitive DEP** (SPMC broadcast anchor + Session; unblocks Steps 5/6 fan-out): the drafted DEP
  for `ai-dynamo/dynamo`.
- **k6 model** (verified): `grafana/k6` `lib/execution_segment.go` (master-less segments), `output/`
  (external streaming sinks).
- **AIPerf code**: `rust/aiperf/src/cellular/{partition,issuance,shard,sketch}.rs`,
  `rust/aiperf/src/metrics_core/{store,accumulator}.rs` (sketch + Welford + `finish_fold_into`),
  `rust/aiperf/src/runner_protocol/cellular_controller.rs` (merge orchestration).
