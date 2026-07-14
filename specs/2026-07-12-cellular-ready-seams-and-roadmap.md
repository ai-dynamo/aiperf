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
  (`aiperf::timing`, `slots.rs`) + the sequential `record_index` counter
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
  (`aiperf::adaptive_core`, `window.rs`) of the drained record stream computing counts + a
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
- **Today:** **already most of the way there** — `aiperf::rng::RngRoot::derive`
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
  executor (`aiperf::graph`) already exists for DAG workloads. No new work for Track-A
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

## Addendum — 2026-07-14 — Phase 1 seams built (S1, S2, S4)

Phase 1 seam extraction is **built** as the always-on `aiperf::cellular` module
(`rust/aiperf/src/cellular/`), each seam a trait with its Direct in-process impl.
Implementation design: `~/.aiperf/docs/superpowers/specs/2026-07-14-rust-cellular-runtime-implementation-design.md`.

- **S1 `cellular::issuance`** — `IssuanceAuthority` (map a cell-local dispatch index
  to the dense global dispatch ordinal); `DirectIssuanceAuthority` (identity, the
  shipping cell-of-one) and `CellularAutonomousIssuer` (`local*cell_count+cell_id`,
  = the round-robin trace instance index; tiles the dense `0..total` ordinal space,
  zero coordinator hop). Threaded through `RunCapture::finish` in the runner: the
  default Direct authority is identity, so single-process output is byte-unchanged
  (confirmed by the 1-vs-4-worker parity harness and a real end-to-end mock run).
- **S2 `cellular::shard`** — `RecordsShard` + two serializable partitions on a
  MessagePack wire: `RecordsShardPartition` (raw records; the controller re-ingests
  every cell's records in global-ordinal order via `merge_records_in_global_order`,
  which is **byte-identical to a single-cell run** — the same worker-count-independent
  mechanism the product path already uses — and validates the ordinals are a
  permutation of `0..total`, returning `RecordsMergeError` rather than panicking on
  malformed wire input), and `ColumnStorePartition` (the roadmap's serializable
  store form; `append_store` merge, deterministic at a fixed topology). `ColumnStore`
  and `RecordIngest` (+ their nested types and `MetricValue`) are now serde-serializable.
- **S4 `cellular::partition`** — `CellPartition` + `ModuloCellPartition` (round-robin
  ownership, identity `(0,1)`; disjoint+complete instance coverage; per-cell seed
  derivation via `RngRoot::derive_indexed_root`).

Superseded here: the earlier "MetricsHeartbeat carries the report percentiles"
framing is unaffected — S3 remains the live-only sketch. **Still designed, not
built:** S3 `MetricsHeartbeat` (t-digest live lane) and the `CellTransport` +
controller/cell multi-process topology (Phase 2). Section 124's "What ships today
(Phase 0)" is superseded for S1/S2/S4 by this addendum; the hot path is unchanged.

## Addendum — 2026-07-14 — S3 heartbeat + t-digest live lane built

S3 is **built** (supersedes the "still designed, not built: S3" note in the prior
addendum): `aiperf::cellular::sketch::TDigest` (a deterministic, serde-serializable,
associatively-mergeable merging t-digest — the frozen sketch type), and
`aiperf::cellular::heartbeat::{MetricsHeartbeat, HeartbeatAccumulator}` (counters +
saturation + TTFT/ITL/latency sketches; `MetricsHeartbeat::merge` folds cells by
sum + t-digest-merge). The single-process **live lane** is built in the runner
(`heartbeat_lane.rs`): env-gated `AIPERF_CELLULAR_HEARTBEAT_LOG` feeds the
accumulator per record and writes a percentile-projected heartbeat NDJSON line on
the phase-progress cadence, composed alongside the Python live sink. An end-to-end
mock run confirms the live sketch percentiles converge to the exact `native-v2.json`
report (TTFT/latency to ~0.01%, ITL to sub-percent at p50). Report percentiles stay
exact from S2; the sketch is live-only.

**Still designed, not built:** the `CellTransport` cross-node seam + the Phase-2
controller/cell multi-process topology (cross-cell heartbeat aggregation and
records-shard partition transfer). The heartbeat/partition types are already
serde-wire-ready for it.

## Addendum — 2026-07-14 — Phase 2 CellTransport + controller/cell topology built

Phase 2 is **built** (supersedes the "still designed, not built: `CellTransport` +
controller/cell topology" note in the prior addendum). This completes the roadmap's
"fully-autonomous cellular" phase for the single-host multi-process case; the seam
is transport-neutral, so a cross-host impl is a `CellClient`/`ControllerTransport`
swap, not a rewrite.

- **`CellTransport` seam** (`aiperf::cellular::transport`): a length-prefixed
  MessagePack frame (`u32` BE length + `rmp-serde` body) carrying `CellMessage`
  (`Heartbeat` / `Partition` — the initial `Hello`/`Done` framing was later dropped as
  unused; see the 2026-07-14 addenda). MessagePack because it is
  self-describing (round-trips the untagged `MetricValue`) and preserves the
  NaN/`+inf` sketch sentinels JSON cannot. Two traits: `CellClient` (the cell,
  blocking `std::net::TcpStream` — off its hot path) and `ControllerTransport` (the
  controller, a Tokio listener that accepts `expected_cells` connections and merges
  their framed streams into one channel). `TcpCellClient` / `TcpControllerTransport`
  are the process impls; a thread-cell would implement the same two traits over an
  in-process channel.
- **Cell mode** (`aiperf-runner --cell`, `runner::cellular_cell`): a child runs the
  ordinary single-process execute path over its budget slice, made cell-aware purely
  by three controller-set env vars — `AIPERF_CELL_ID` / `AIPERF_CELL_COUNT` (select
  the `CellularAutonomousIssuer`'s partition, so its dense global dispatch ordinals
  and the `PartitionedSampler`'s instance selection reproduce the single-process
  trace set) and `AIPERF_CELL_CONTROLLER_ADDR` (the records shipper's target). After
  the run it ships one final `RecordsShardPartition` + its merged heartbeat, never
  writing a report.
- **Controller** (`runner::cellular_controller`): a non-cell execute request with
  `cfg.runtime.cells > 1` becomes the controller. It slices each phase's request
  budget and concurrency cap by the `owned_share` round-robin (shares tile `0..total`
  exactly), spawns one `--cell` child per cell (`current_exe`, spec piped to stdin),
  serves the transport, and — critically — watches every child so a cell that dies
  **before** connecting aborts the run (`select!` of the partition-collect loop
  against a child-exit channel) rather than hanging the accept loop. It then merges
  every cell's records in global dispatch-ordinal order (S2 records-first re-ingest)
  into the single authoritative `native-v2.json`, and aggregates the cells' live
  heartbeats (counter sum + t-digest merge) into a `cellular-heartbeat.json` sidecar.
  To Python this is still one run behind one v2 request.

**Verified** (multi-process, OS tools): a 4-cell mock run emits `success:true` with
240 merged records; every dataset-deterministic metric (request/token counts,
ISL/OSL full distributions) is **byte-identical** to the 1-cell run, confirming the
S1+S4 autonomous partition reproduces the single-cell trace set and the S2 merge is
order-exact; only wall-clock timing metrics differ (independent live-server
variance). The cross-cell heartbeat aggregates all 240 requests. A crashed-cell run
(cells fault during preparation) aborts the controller in <1s with a `success:false`
execution envelope — not a hang. `ps`/`ss` confirm the controller + N cell processes
and their TCP sockets during a run and no leaks after.

**Not on the product path:** Python emits no `cells > 1`, so a stock `aiperf profile`
run is byte-unchanged (single process, `DirectIssuanceAuthority`, no transport). The
multi-process topology is reachable only by an authored `cfg.runtime.cells` envelope
— a developer/experimental capability, not yet a product surface. **Out of scope
still:** cross-host deployment (the seam is ready; only TCP-loopback impls exist),
S5 executor changes, and any Python orchestration of cells.

**Phase constraint — request-bounded only.** The dense-ordinal tiling requires every
phase's *actual* dispatch count to equal its sliced `requests` budget, so the
controller fails a cellular run closed (a clear pre-spawn error, not a cryptic merge
`OrdinalOutOfRange` **or a silent N× replay**) if a phase's `type` is not one of the
request-bounded arrival-pattern types (`concurrency`/`poisson`/`gamma`/`constant`) —
a trace-driven `fixed_schedule`/`user_centric` phase sets `enforce_stop=false` and
builds its schedule from the *full, unpartitioned* conversation list, so every cell
would replay the whole trace — or if any phase lacks `requests` or carries a
`duration` / `sessions` / `adaptive_scale` bound whose real count can diverge (e.g.
`ramp_until_fail` stopping on an SLA breach). Pacing-only knobs (concurrency/rate ramps) and post-send
cancellation are allowed — they change *when* turns are sent or mark them cancelled
after dispatch, not *how many*. The merged report reproduces a 1-cell run's metric
data (profiling + warmup sections byte-identical, run mode/model, configured
endpoints); two 1-cell blocks are intentionally **not** reproduced — the coordinator's
`finalize_run` provenance (`distribution_id` / alias-resolved `endpoint_profiles`,
carried in the terminal envelope instead) and the grouped per-error message list
(cells ship metric records with error/cancel flags, so error *counts* are in the
metrics, but not the messages a cross-cell regroup would need).

## Addendum — 2026-07-14 — multi-phase absolute-slot issuance (content parity)

A full-review workflow caught a multi-phase byte-parity defect an earlier
uniform-metric e2e had masked (fixed hidden in a uniform ISL/OSL dataset: a *wrong
instance set* yields identical metrics). The runner rebuilds each cell's sampler
fresh per phase — the dataset RNG re-seeds as `runner.phase.{i}.dataset` — so a cell
draws its owned instances of **each phase from position 0**. The issuer, however,
stamped a cumulative ordinal; the first fix made a *base-aware per-phase count* so the
ordinals still tiled `0..total`, which meant the merge never failed — but each cell
drew a *different instance set* than a 1-cell run whenever a warmup phase's length was
not a multiple of `cell_count` (e.g. w=3/p=3/c=2 → `{inst0,inst1,inst3}` vs
`{inst0,inst1,inst2}`). The report was silently wrong.

**Fix — the cell stamps the single-cell absolute slot directly.** `IssuanceAuthority`
now maps `(flat_local, phase_ordinal_base, within_phase_local)`: the identity issuer
returns `flat_local` (non-cell path byte-unchanged); the cellular issuer returns
`phase_ordinal_base + within_phase_local*cell_count + cell_id`, where `phase_ordinal_base`
is the turns the run's prior phases dispatched globally (0 for warmup, `W` for
profiling). That equals the slot a 1-cell run assigns the same instance, so the
cumulative `0..total` merge reproduces it byte-for-byte. `build_cell_envelope` slices
each phase by its own `owned_positions` (base 0), matching the per-phase-reset sampler;
`RunCapture::finish` tracks a per-phase dispatch counter and looks the base up from a
phase→base map; the controller computes the bases and threads them to each cell via
`AIPERF_CELL_PHASE_ORDINAL_BASES`.

**Verified at the content level:** a warmup=10/profiling=500/cells=3 run over a
*varying*-ISL dataset (`isl ~ N(256, 64)`, std ≈ 61.5) is byte-identical to the 1-cell
run on the **full** ISL distribution (avg/std/min/max/all percentiles) and every
dataset-deterministic metric, for both the profiling and warmup sections. A cell's own
(discarded) report accumulator holds sparse global slots — bounded by the 1-cell row
count, transient, and never serialized — the merged report is the authoritative one.
Also fail-closed now (`validate_cellular_run_shape`): `cells>1` is accepted only for
the exact shape the partition/issuance seam is sound for — the scheduled **HTTP**
transport over **synthetic, single-turn** datasets. A non-HTTP transport
(gRPC/dynosim) or a non-synthetic (`file`/`public`, incl. graph-program) dataset runs
a different executor that never ships a partition; and a **multi-turn** dataset
diverges the sampler's per-*draw* partition from the issuer's per-*turn* ordinal (the
same silent-wrong-report class as the multi-phase fix above — continuations advance
the turn count but not the draw position), so only `turns == 1` is sound. Each is
rejected up front with a clear error rather than a silent divergence.
