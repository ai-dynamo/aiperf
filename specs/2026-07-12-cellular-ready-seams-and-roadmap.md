<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cellular-ready seams + roadmap — encode the extension points now, ship only the basic case

**Status:** Draft / decided direction

**Relationship:** This is the umbrella for the Track-A specs
(`2026-07-12-scheduled-worker-local-accumulation.md` A1,
`…-http-connector-seam-uds-duplex.md` A3, `…-lean-per-request-hotpath.md` A4,
`…-single-observer-compat-projection.md` A2). It aligns them with the
already-written **aiperf-v2 cellular runtime** design
(`ajc/dag-v3:docs/deps/aiperf-v2-cellular-runtime.md`).

## Decision

Encode the extensibility the cellular architecture needs **as traits, now**, but
implement **only the basic ("Direct", single-process, single-issuer, in-process
sharded) case today**. The fully autonomous cellular runtime (per-cell issuers,
records-shard partitions moved over a cross-node communication seam, cross-node
heartbeat aggregation, multi-node cell topology) becomes a **drop-in** behind those
seams — not a rearchitecture. This spec stays transport- and deployment-neutral: a
"cell" is a unit of autonomous scale (a thread, a process, or a node); how cells
talk across a node boundary is itself an abstracted seam, not a fixed wire.

Design discipline (repo `CLAUDE.md`): every one of these is a `trait` with exactly
one concrete impl today, taken by trait (not concrete) in signatures, with the
deferred extension noted in `//!`/`///`. "One model at two scales" — the single
process is a **cell of one**; the cluster is N cells. Nothing on the hot path
changes shape between them (aiperf-v2 REQ 10, REQ 5).

## The five seams (trait today → cellular impl later)

### S1 — `IssuanceAuthority` (the issuance tier ladder)
- **Contract:** a `VirtualTraceRunner`/worker acquires a credit + admission slot and
  releases it on terminal, through one object-safe seam. Carries the **global
  dispatch ordinal** assignment (see A1: single, dense, deterministic → byte-parity).
- **Today (Tier 0 "Direct"):** in-process — one coordinator-owned `SlotPool`
  (`aiperf-timing/slots.rs`) + the sequential `record_index` counter
  (`scheduled.rs:868`). **Exact** global concurrency; deterministic assignment.
  Implement as a **single central assignment point**, NOT a shared-atomic self-issue
  (shared-atomic breaks run-to-run float reproducibility — A1).
- **Later:** a `CellularAutonomousIssuer` — an in-cell issuer with **zero
  coordinator hop**, exact-per-cell / global-bounded-by-partition. Same
  `acquire/release/ordinal` contract; only the impl and the exactness guarantee
  change. (The seam is generic enough to admit other issuance strategies later, but
  none are on the roadmap.)
- **Freeze now:** the `acquire → (credit_id, global_ordinal, slot_guard)` shape and
  the RAII slot-release, so a distributed issuer is a pure swap.

### S2 — `RecordsShard` + `ColumnStorePartition` (sharded accumulation)
- **Contract:** a shard ingests per-record metrics locally, can emit a
  **mergeable snapshot summary** on demand (for live), and exports a **mergeable,
  wire-serializable partition** at phase boundary. The final report is the merge of
  all partitions (aiperf-v2 REQ 6).
- **Today:** one `NativeMetricsObserver` + `MetricsAccumulator` **per worker thread**
  (in-process shard); merge once at the join via the verified
  `MetricsAccumulator::merge` (`accumulator.rs:485-514`) / `ColumnStore` `append_store`
  (`store.rs:569-656`). Records-first re-ingest in global-ordinal (dispatch) order →
  exact, byte-parity (A1).
- **Later:** the shard is a **per-cell `records-shard`** owning a `ColumnStorePartition`,
  moved to the controller across a **cross-node communication seam** (transport
  abstracted — the concept is "get a serialized partition from cell A to the
  aggregator," not any specific wire); the controller merges partitions **at export
  only** (concatenate in `cell_id` order). Same ingest/merge/export contract — the
  `ColumnStore` already IS the partition; make it **serializable** now even though
  today's "transport" is an in-process move.
- **Freeze now:** (a) `ColumnStorePartition` merge is associative + deterministic at a
  fixed topology; (b) the partition is serializable (so a cross-node transfer is a
  transport swap, not a data-model change); (c) the **records-manager finalization
  predicate** and export filenames stay byte-stable (aiperf-v2 REQ 4).

### S3 — `MetricsHeartbeat` (live, sketch-based)
- **Contract:** a bounded-cadence (≤1 s) snapshot of **counters + associatively-
  mergeable distribution sketches** + saturation, aggregated across shards. Live
  percentiles are **sketch-derived; the final report is exact** from S2 partitions
  (aiperf-v2 REQ 7, line 399).
- **Today:** a single cheap live lane — one `WindowSampler`-style consumer
  (`aiperf-adaptive/window.rs`) of the drained record stream computing counts + a
  **t-digest** of TTFT/ITL/latency, plus per-record live streaming to Python
  (`live_streaming.rs`, already per-record). In-process merge of shard sketches on a
  timer. Live counts from the monotonic issuer (S1), not summed shards (avoids the
  wobble noted in `project_cr_progress_source_of_truth`).
- **Later:** cell heartbeats carry the same t-digest + counters to the controller's
  `TrafficCoordinator`, which merges by sum / t-digest-merge / max (aiperf-v2 line
  399). Identical sketch type + merge; only the transport (in-process → heartbeat)
  changes.
- **Freeze now:** the **sketch type is t-digest** (aiperf-v2 ref [17]) and its
  serialized, associatively-mergeable form — so an in-process merge today and a
  cross-cell merge later are the same operation. (Report percentiles stay **exact**
  from S2; the sketch is live-only — this supersedes the earlier
  "HdrHistogram-for-the-report" idea.)

### S4 — deterministic partition / seed derivation
- **Contract:** identical `(workload_seed, cell_count, partition_assignment)` →
  byte-stable artifacts; per-shard RNG derivation composable so different cell counts
  produce the same trace **set** with different **ownership/order** (aiperf-v2 REQ 3).
- **Today:** **already most of the way there** — `aiperf-rng::RngRoot::derive`
  (order-independent `blake3(root:id)` per trace/hash id) is composable by
  construction; trace identity does not depend on who runs it. Basic case:
  `cell_count = 1`, `partition = identity`. Take `(cell_id, cell_count)` in the
  partition/derivation API **now** even though both are fixed at `(0, 1)`.
- **Later:** the controller assigns each cell a static budget partition + deterministic
  remainder distribution; per-cell derivation selects its owned trace instances from
  the same seed space. No new RNG; just a partition function over the existing
  hash-derived streams.
- **Freeze now:** the derivation signature carries `(cell_id, cell_count)`; trace
  identity is ownership-independent (it already is).

### S5 — executor / runner (related; mostly out of Track-A scope)
- **Contract:** a pooled long-lived per-trace runner over the graph-IR fire-on-ready
  loop, with a **flat-graph fast path** whose overhead matches non-graph dispatch
  (aiperf-v2 REQ 8/9).
- **Today:** the scheduled dispatch path *is* the flat/degenerate case; the graph
  executor (`aiperf-graph`) already exists for DAG workloads. No new work for Track-A
  beyond keeping the scheduled path lean (A4).
- **Later:** unify scheduled + graph under the one `VirtualTraceRunner`/`FlatGraphActor`
  pool (aiperf-v2 "unified substrate"). Noted, not scheduled here.
- **Freeze now:** nothing new — just don't couple measurement (S2/S3) to the execution
  model, so either runner feeds the same shard/heartbeat seams.

## What ships today (Phase 0 — the "basic" case)

- Single **Direct** issuer (S1): one `SlotPool` + sequential deterministic ordinal.
- **In-process per-worker** `RecordsShard`s (S2), merged once at the join; final report
  exact via records-first dispatch-order re-ingest.
- One cheap **live lane** (S3): t-digest sketches + monotonic counters, in-process
  merge on a timer; per-record streaming to Python as today.
- `cell_count = 1`, identity partition (S4).
- Scheduled dispatch (S5), lean (A4).

No cross-node transport, no controller/cell split, no distributed issuance — but
every seam is the one the cellular runtime consumes, so none of it is throwaway.

## Roadmap to fully-autonomous cellular

- **Phase 0 (now):** basic case above (= Track-A PRs 1–5).
- **Phase 1 — seam extraction (no behavior change):** land `IssuanceAuthority`,
  `RecordsShard`, `MetricsHeartbeat`, and the `(cell_id, cell_count)` derivation
  signature as traits with the Direct/in-process impls. Make `ColumnStorePartition`
  serializable. Gate: byte-identical to Phase 0 (the Track-A parity harnesses).
- **Phase 2 — Cellular Autonomous:** a per-cell autonomous issuer (S1), per-cell
  `records-shard` whose partition + heartbeat cross a **cross-node communication seam**
  (transport abstracted — S2/S3 transport swap), a controller that aggregates
  heartbeats live and merges partitions at export, and partitioned seed derivation
  (S4). Reuses every Phase-1 seam; the new work is the **cross-node transport +
  controller + topology**, not the measurement model. (The cellular runtime doc,
  `ajc/dag-v3`, is the reference for the concrete deployment shape; this spec stays
  transport- and deployment-neutral.)

## Contracts frozen now so Phase 2 is a swap, not a rewrite

1. `IssuanceAuthority::acquire → (credit_id, global_ordinal, RAII slot)`; single
   deterministic ordinal (S1).
2. `ColumnStorePartition`: associatively + deterministically mergeable, serializable;
   report = merge of partitions (S2).
3. `MetricsHeartbeat`: t-digest sketch + counters, associatively mergeable; live =
   sketch, final = exact-merge (S3).
4. `(cell_id, cell_count)`-parameterized deterministic derivation; ownership-independent
   trace identity (S4).
5. Preserve the public contracts aiperf-v2 REQ 4 pins: records-manager finalization
   predicate, exporter formats/filenames, CLI flag semantics, `CreditPhaseStats` schema.
