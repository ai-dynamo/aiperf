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
`ramp_until_fail` stopping on an SLA breach), or drives a concurrency/prefill/rate
**ramp** (which the controller cannot slice per cell, so every cell would ramp to the
full target and N× the aggregate). The static `concurrency`/`prefill_concurrency`/
`rate` caps ARE sliced per cell (`rate → rate/cell_count`, caps by round-robin share)
so the cells' aggregate offered load matches the 1-cell run; post-send cancellation
stays allowed (it marks turns cancelled after dispatch, not *how many* are sent). The
merged report reproduces a 1-cell run's metric
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

## Addendum — 2026-07-14 — audit-closed divergences + cancellation policy

A targeted controller-vs-single-cell audit (enumerating every config axis that a cell
reads locally) surfaced four more silent-divergence classes past the multi-phase fix.
Three are now rejected up front; the fourth (cancellation) is **allowed as a documented
statistical approximation**, per the run owner's preference to keep the feature over
enforcing byte-parity.

- **Run seed now required** (`validate_cellular_run_shape`). Each cell is a separate OS
  process; without `run.random_seed` every cell entropy-seeds an *independent* random
  synthetic dataset (different prompts / ISL / OSL), so the cells no longer partition
  one shared instance space — the core invariant the whole design rests on. A missing
  seed is rejected, not silently divergent.
- **Single endpoint URL required.** Multiple `endpoint.urls` round-robin in *cell-local*
  order (each cell starts at index 0), so on a heterogeneous backend pool the per-request
  URL assignment diverges from the 1-cell interleave. `urls.len() > 1` is rejected.
- **Static caps must be ≥ `cell_count`.** `concurrency` / `prefill_concurrency` are sliced
  round-robin with a `.max(1)` floor; a cap below `cell_count` floors to 1 per cell and
  the aggregate in-flight over-subscribes to `cell_count` (e.g. `concurrency=2`, 4 cells →
  aggregate 4, not 2). Every phase's `requests` must likewise be `≥ cell_count` (no cell
  owns zero). Both are rejected below the floor.
- **Cancellation — allowed, approximate.** This *supersedes* the "post-send cancellation
  stays allowed (it marks turns cancelled after dispatch, not how many are sent)"
  justification in the Phase-2 addendum's constraint paragraph: that framing was wrong —
  a cancelled request truncates its own output, so cancellation *does* move
  dataset-deterministic metrics (OSL, goodput), not just liveness. The accurate statement:
  each cell applies the *same* per-request cancel probability to its slice, so the
  **aggregate** cancellation rate matches the 1-cell run, but the exact cancelled *subset*
  differs (cell-local RNG draw order is not the 1-cell global order). Byte-parity is
  therefore exact only for a seeded phase **without** cancellation; `rate`-based arrival
  pacing and cancellation are intentional statistical approximations — the feature is kept
  rather than disabled. A run mixing them is *not* claimed byte-identical, only
  distribution-equivalent in the aggregate.

**Byte-parity scope, restated.** Exact, byte-for-byte 1-cell reproduction is claimed for:
seeded `concurrency`-bounded phases, synthetic single-turn HTTP, single URL, no ramps, no
cancellation. Everything the guards *allow past* that (Poisson/Gamma/constant arrival
pacing via `rate/cell_count`, cancellation) is aggregate-equivalent, not byte-identical,
and documented as such rather than rejected.

## Addendum — 2026-07-14 — cellular is product-reachable from the Python frontend

**Supersedes the "Not on the product path" caveat above** (the paragraph beginning
"Python emits no `cells > 1`"). That is no longer true: cellular is now reachable and
e2e-tested through the ordinary `aiperf profile` frontend.

- **`runtime.cells`** — a new `RuntimeConfig.cells` field (int, `ge=1`, default 1;
  `src/aiperf/config/runtime.py`) is dumped verbatim into the protocol-v2 execute
  envelope by `dump_benchmark_run`, so `cfg.runtime.cells` reaches the runner. `cells=1`
  (default) keeps the single-process path byte-for-byte unchanged.
- **`--cells N`** — a CLI flag (`cli_config.py`) mapped to `runtime_dict["cells"]` in
  `_converter_runtime.py`, mirroring `--workers-max`. `aiperf profile --cells N` drives
  the controller/cell topology; Python still launches exactly one `aiperf-runner` (which
  becomes the controller and spawns the cells) and reads the one merged `native-v2.json`,
  so the orchestrator flow is unchanged.
- **E2e from the frontend** — `rust/e2e/tests/test_cellular.rs`:
  `test_cellular_run_from_python_frontend` (`--cells 3` runs end-to-end and reports the
  full budget) and `test_cellular_matches_single_cell` (a 3-cell run reproduces the
  1-cell run's input/output sequence-length distributions byte-for-byte through the full
  presentation pipeline, varying ISL).

**Not a bespoke HTTP layer.** The earlier "scheduled HTTP transport" wording overstated
the boundary. A cell runs the *same* `execute.rs` path as any single-process run,
differing only by an injected `IssuanceAuthority` (`Direct` vs `CellularAutonomousIssuer`)
and an env-gated records sink (`CellRecordsShipper` ships over the transport instead of
writing a report). The `transport == "http"` whitelist reflects **wiring coverage** — only
the online-scheduled executor injects the cell issuer and ships partitions today; the
gRPC/graph/offline executors are separate paths not yet threaded. The
partition/issuance/transport seam is transport-neutral by design, so extending cellular to
those executors is a wiring task, not new HTTP code. **Still out of scope:** cross-host
transport (TCP loopback only) and the S5 executor change.

## Addendum — 2026-07-14 — controller panic guard + loud sidecar-drop warning

Two robustness gaps a full-review pass surfaced against the product-reachable path:

- **Controller panic guard.** `run_controller` now wraps `run_cellular` in
  `std::panic::catch_unwind` (mirroring `handle_v2`'s guard on the single-process path).
  The controller runs the records merge, native-v2 serialization, and the newly-added
  export plane inline; a panic in any of them previously aborted the controller
  (exit 101) with no `run_terminal` envelope, so Python saw a crashed subprocess instead
  of a typed execution failure. A caught panic (or a returned error) is now emitted as a
  `success:false` execution-stage envelope carrying the message as a typed diagnostic.
- **Loud sidecar-drop warning.** A cellular run scrapes any side-channel telemetry
  sidecars (`server_metrics` / `gpu_telemetry` / `network_latency`) into each cell's
  discarded scratch tree, so they are omitted from the merged report (the documented
  report-fidelity gap) — whereas a single-process run emits them. This was a *silent*
  divergence; the controller now logs a startup `WARN` naming the dropped sidecars and
  pointing at the non-`--cells` path. It is not fail-closed: `gpu_telemetry` and
  `server_metrics` default *on*, so rejecting any present sidecar would refuse nearly
  every cellular run. Cross-cell sidecar aggregation remains future wiring.

## Addendum — 2026-07-14 — relax byte-parity-only guards to allow-with-warning

Cellular mode's purpose is multi-node **scale with acknowledged precision loss**, so
several guards that existed *only* to protect byte-parity were over-restrictions that
contradicted that bargain. Following the cancellation precedent (allow the feature; warn;
don't disable), these flip from fail-closed **reject** to **allow-with-warning**:

- **Multiple endpoint URLs.** Cells round-robin the URL pool in cell-local order, so the
  exact per-request URL assignment differs from a 1-cell run, but the aggregate load
  across the pool matches. Hitting a backend pool from N nodes is a first-class multi-node
  workload. `validate_cellular_run_shape` no longer rejects `urls.len() > 1`;
  `warn_cellular_approximations` logs the aggregate-equivalence.
- **No run seed.** Instead of requiring `run.random_seed`, `resolve_cellular_seed` derives
  one shared seed from the run identity (hash of `benchmark_id`) and `build_cell_envelope`
  injects the *same* value into every cell — coherent partition, reproducible per
  `benchmark_id`, no flag required. A present authored seed is still inherited verbatim.
- **Concurrency/prefill/rate ramps.** A `RampSpec` is only `{duration, strategy}`; it
  ramps *to* the phase's `concurrency`/`rate` target, which `build_cell_envelope` already
  slices per cell. So each cell ramps to its sliced target and the aggregate reaches the
  full authored target — aggregate-equivalent (the aggregate starts near `cell_count`
  rather than 1). `validate_cellular_phase_budgets` no longer rejects ramps.

**Still fail-closed** (real requirements, not byte-parity): non-HTTP transport,
non-synthetic / multi-turn datasets (scheduled path), `duration`/`sessions` bounds
(need the ragged-count merge — see graph-mode cellular below), `adaptive_scale` (needs
cross-cell scaling consensus), and caps below `cell_count`. Byte-parity remains *exact*
for a seeded `concurrency` phase with none of the approximate knobs.

**Next: graph-mode cellular.** The scheduled path's multi-turn / trace restrictions are
artifacts of partitioning a shared linear sampler *by draw* while numbering *by turn*.
The graph-IR path (`aiperf::graph::TraceExecutor` over `GraphSink`) already models a run
as independent whole traces selected by a `RootPolicy::next_trace()` seam, each with a
run-unique instance ordinal — so partitioning is `instance_ordinal % cell_count == cell_id`
with no sampler surgery, and duration/trace/multi-turn distribution falls out cleanly.
Wiring cellular through the graph path (partitioned `RootPolicy` + graph-records shipping)
is the honest home for those, tracked as the next cellular effort.

## Addendum — 2026-07-14 — graph-mode cellular is built (the trace-partition path)

The "next: graph-mode cellular" note above is now **built and proven end-to-end**. Graph
programs (`dag_jsonl` / `weka_trace` / `dynamo_trace`) distribute across cells cleanly —
each trace is a self-contained unit, so there is no shared-sampler / per-turn-ordinal
problem, and the whole multi-turn/trace restriction class the scheduled path fails on
simply does not arise. Realized as the design's *deterministic-per-topology* contract
(same trace set across topologies, different cell ownership + merge order), not
byte-parity.

Four steps, each reviewed:
- **`PartitionedGraphTraceSource`** (`aiperf::graph::workload`): cell `k` of `C` owns the
  interleaved global session ordinals `k, k+C, …` and cycles templates by the global
  ordinal, so the union across cells reproduces the 1-cell trace set. One formula covers
  the finite (`--num-conversations`) and unbounded (duration) modes; `cell_count == 1`
  reproduces `CyclingGraphTraceSource` exactly. (Weighted/mix sampling will derive a
  per-cell RNG stream here — the skip-N composability — when that source lands.)
- **Runner selection** (`graph_phase_runtime::prepare_graph_phase`): a cell with
  `cell_count > 1` (and no static-node `request_limit`) selects the partitioned source
  from `ModuloCellPartition::from_env`; the non-cell path is byte-unchanged.
- **Cell records ship** (`execute_graph_native`): a graph cell ships its captured
  `RecordIngest` records + a terminal heartbeat exactly like the scheduled path, additive
  and env-gated, its own report going to the throwaway scratch dir.
- **Controller admit + merge** (`run_cellular`): `is_graph_dataset` detects the graph
  format; the controller admits it past the synthetic/single-turn guard, skips the
  scheduled-only `requests`-budget validators, and merges via
  **`merge_records_by_concatenation`** — graph cells carry LOCAL per-cell `request_index`
  (wall-clock start order), so the controller concatenates by `cell_id` and re-numbers
  densely (numerically correct — `request_index` only selects a column slot; phase
  separation rides the `phase` field — matching a 1-cell run up to float summation order).

**Proven** (`rust/e2e/tests/test_graph_cellular.rs`): `aiperf profile --cells 3` over a
`dag_jsonl` dataset runs end-to-end, emits the controller sidecar, and reproduces the
1-cell run's total record count and input-token distribution (partition covers the full
trace set). Still out of scope: cross-host transport; weighted/mix RNG sampling; the
graph static-node `request_limit` partition (falls back to the single-cell cycler).

---

## Addendum — 2026-07-15 — Phase 2 cross-host + multi-turn + k8s built; Phase 3+ north-star split out

Since these addenda, the cross-host transport shipped (velo connect-by-endpoint, §Phase-2 CellTransport
now realized over velo — see `2026-07-15-velo-cell-transport-design.md` and its 2026-07-15 addendum),
multi-turn cellular landed on the **exact-fold** path (conversation-level partition; retain-path
multi-turn stays fail-closed), graph cellular is built, and the Kubernetes launch (operator + JobSet +
headless-DNS zero-discovery + emptyDir/no-RWX) is built on `rust-operator`. Bounded-memory metrics
(t-digest sketch) are built standalone but **cellular+sketch is still blocked** (`ensure!(!sketch_mode)`
on the cell ship path).

The Phase-3+ north-star — turning the static-partition-then-collect model into a live coordinated
fabric — is specified separately in **`2026-07-15-ultimate-cellular-velo-runtime-design.md`**: the
dataset **data plane** (producer-owned SPMC add-only broadcast with replay-on-attach, modeled on the
kvbm p2p `Session`; velo must add the anchor), the **monotonic phaser** control plane (synchronized
START generalized to every phase transition, cyclic-by-monotonic-counter), the per-request dispatch
state machine with a counted `DistributionMiss`, the end-of-warmup barrier (drain-barrier pattern) that
unblocks cross-cell adaptive consensus, and bounded-memory record collection (ship the merged sketch as
a `StorePartition`). That spec is authoritative for the S1–S5 seams' next era.
