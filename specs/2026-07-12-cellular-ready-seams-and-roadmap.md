<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cellular seams — one measurement/execution model at two scales

**Status:** Built (single-host multi-process, cross-host velo, and k8s)

**Relationship:** This is the umbrella for the cellular measurement seams. The velo
cross-node transport that carries them is
`2026-07-15-velo-cell-transport-design.md`; the Phase-3+ live-fabric north-star
(dataset fan-out, monotonic phaser, per-request dispatch state machine,
bounded-memory collection) is `2026-07-15-ultimate-cellular-velo-runtime-design.md`,
authoritative for the seams' next era. The
`2026-07-12-scheduled-worker-local-accumulation.md` perf lever feeds the
single-process impls below. (The earlier Track-A connector-seam / lean-hotpath /
single-observer planning specs were abandoned and removed.)

## The idea

AIPerf's cellular runtime partitions one benchmark across N processes/nodes so a
single run can generate 1M+ concurrency and hold a dataset too large for one process's
RAM. The design encodes the extensibility the distributed runtime needs **as traits**,
each with a **Direct** single-process/in-process-sharded impl and a **cellular**
distributed impl — both built. "One model at two scales": the single process is a
**cell of one** (`DirectIssuanceAuthority`, identity partition); the cluster is N
cells. Nothing on the hot path changes shape between them.

A *cell* is a unit of autonomous scale — a thread, a process, or a node. How cells
talk across a node boundary is itself an abstracted seam (velo today), not a fixed
wire. The seams live in the always-on `aiperf_runtime::cellular` module
(`rust/runtime/src/cellular/`), object-safe where they cross a `dyn` boundary and
generic where hot-path monomorphized. `cells = 1` (the default; Python's default) is
**byte-for-byte unchanged** — `ModuloCellPartition::from_env()` returns `None` off-cell
→ `DirectIssuanceAuthority` → single-process output.

## The five seams

### S1 — `IssuanceAuthority` (`cellular/issuance.rs`)

Maps a cell-local dispatch position to the **dense global dispatch ordinal** stamped on
each issued turn, through one object-safe seam:
`global_ordinal(flat_local, phase_base, within_phase_local)`. The ordinal is the single,
dense, deterministic assignment that gives the merge its byte-parity — a single central
assignment point, never a shared-atomic self-issue (which breaks run-to-run float
reproducibility).

- **`DirectIssuanceAuthority`** (the cell-of-one): returns `flat_local` — identity, so
  the non-cell path is byte-unchanged. It composes with the coordinator-owned `SlotPool`
  (`timing::slots`) for exact global concurrency and the sequential `record_index`
  counter for deterministic assignment.
- **`CellularAutonomousIssuer`**: returns
  `phase_base + within_phase_local * cell_count + cell_id` — each cell self-assigns the
  **absolute single-cell slot with zero coordinator hop**. `phase_base` is the turns the
  run's prior phases dispatched globally (0 for warmup, `W` for profiling), so a cell
  draws its owned instances of *each* phase from position 0 (the sampler re-seeds
  per phase as `runner.phase.{i}.dataset`) yet stamps exactly the slot a 1-cell run
  assigns the same instance. That absolute-slot mapping is what makes the cumulative
  `0..total` merge reproduce a 1-cell run byte-for-byte — an earlier cumulative-ordinal
  form tiled `0..total` (so the merge never *failed*) but drew a different instance set
  whenever a warmup phase's length was not a multiple of `cell_count`, silently
  corrupting the report. The controller computes the bases and threads them to each cell
  via `AIPERF_CELL_PHASE_ORDINAL_BASES`.

### S2 — `RecordsShard` + partitions (`cellular/shard.rs`)

A shard ingests per-record metrics locally and exports a **mergeable, wire-serializable
partition** at the phase boundary; the final report is the merge of all partitions.
`ColumnStore`, `RecordIngest`, and their nested types (`MetricValue`) are serde-
serializable, carried as MessagePack over the transport (JSON cannot round-trip NaN
record values). Three partition/merge forms:

- **`RecordsShardPartition`** (raw records) merged by `merge_records_in_global_order`
  (scheduled): validate the ordinal union is a permutation of `0..total`, stable-sort,
  re-ingest in global-ordinal (dispatch) order → **byte-identical to a single-cell run**
  (the same worker-count-independent mechanism the single-process path already uses),
  returning `RecordsMergeError` rather than panicking on malformed wire input.
- **`RecordsShardPartition`** merged by `merge_records_by_concatenation` (graph): graph
  cells carry LOCAL per-cell `request_index` (wall-clock start order), so the controller
  concatenates by `cell_id` and re-numbers densely — deterministic-per-topology
  (numerically correct: `request_index` only selects a column slot, phase separation
  rides the `phase` field), matching a 1-cell run up to float summation order.
- **`ColumnStorePartition`** (the folded store) merged by `merge_store_partitions` →
  `ColumnStore::append_store` (exact-fold / sketch): associative, ULP-tolerant. Used
  when a cell folds records into its own accumulator and drops them (metrics-only), so
  it ships the folded store instead of a record `Vec`.

The single-process shard is one `NativeMetricsObserver` + `MetricsAccumulator` **per
worker thread** (`DirectRecordsShard`), merged once at the join. The finalization
predicate and export filenames stay byte-stable.

### S3 — `MetricsHeartbeat` + t-digest (`cellular/heartbeat.rs`, `cellular/sketch.rs`)

A bounded-cadence (≤1 s) live snapshot of **counters + saturation + associatively-
mergeable distribution sketches** (TTFT / ITL / latency), aggregated across shards by
`HeartbeatAccumulator` (`MetricsHeartbeat::merge` folds cells by counter-sum +
t-digest-merge). The frozen sketch type is `TDigest` — deterministic (Dunning K1 scale),
serde-serializable, associative/order-independent `merge`, exact min/max/count/sum,
approximate percentiles. **Live percentiles are sketch-derived; the final report is
exact from S2.** The single-process live lane is built in the runner (env-gated
`AIPERF_CELLULAR_HEARTBEAT_LOG` writes a percentile-projected NDJSON line on the
phase-progress cadence); an end-to-end mock run confirms the live sketch percentiles
converge to the exact `native-v2.json` report (TTFT/latency to ~0.01%, ITL sub-percent
at p50). The Phase-2 controller aggregates every cell's shipped heartbeat into a
`cellular-heartbeat.json` sidecar.

### S4 — `CellPartition` / `ModuloCellPartition` (`cellular/partition.rs`)

The deterministic `(cell_id, cell_count)` work partition: round-robin ownership
(`i % cell_count == cell_id`), disjoint + complete, identity `(0, 1)` off-cell. Per-cell
seed derivation composes via `RngRoot::derive_indexed_root`, and trace identity is
ownership-independent by construction (order-independent `blake3(root:id)` per
trace/hash id), so identical `(workload_seed, cell_count, partition_assignment)` yields
byte-stable artifacts and different cell counts produce the same trace **set** with
different ownership/order. The per-cell share is
`owned_positions(total, k, C) = ceil((total - k) / C)`.

**Reused by the single-process sharded runtime.** The same `ModuloCellPartition` and
`owned_positions` primitive now shards the in-process thread-per-core scheduled runtime
(`engine::sharded_scheduled`): when `runtime.workers > 1`, thread `t` of `W` is a
*sub-cell* that owns the nested two-level partition
`ModuloCellPartition::new(cell_id + cells*t, cells*W)` — the unique modulo family that
nests inside cell `c`'s `(c, cells)` ownership *and* tiles the global `cells*W` grid, so
each `(cell, thread)` stamps exactly its residue class and the union is a permutation of
`0..total`. `owned_positions` and `cell_count_from_envelope` deliberately live in
`engine::cell_launcher` (not the velo-gated controller) so this sharded path reuses the
identical share **without** the `velo` feature. Cellular's partition machinery is thus
shared by both the multi-process cellular path and the single-process sharded path.

### S5 — executor / runner

The scheduled dispatch path *is* the flat/degenerate case of the fire-on-ready loop; the
graph executor (`aiperf_runtime::graph`) exists for DAG workloads. Measurement (S2/S3)
is not coupled to the execution model, so either runner feeds the same shard/heartbeat
seams. Unifying scheduled + graph under one `VirtualTraceRunner` pool is noted, not
built.

## Multi-process controller/cell topology (built)

A non-cell execute request with `cfg.runtime.cells > 1` becomes the **controller**
(`engine::cellular_controller::run_cellular`); each child launched as `aiperf --cell`
runs the ordinary single-process execute path (`engine::cellular_cell`), made cell-aware
purely by env. To Python this is still one run behind one v2 request. The controller:

- Detects the run kind (`CellularRunKind::{Scheduled, Graph}`, `engine::cellular_kind`),
  which answers the four ways the paths differ — phase validation, per-phase ordinal
  bases, record merge, and per-cell session-budget slicing. Transport (`http`/`grpc`) is
  orthogonal to the kind.
- Slices each phase's request budget and static caps by the `owned_positions`
  round-robin (`build_cell_envelope`): `requests` and `concurrency`/`prefill_concurrency`
  by round-robin share (`.max(1)` floor), `rate → rate / cell_count`, so the cells'
  aggregate offered load matches the 1-cell run. Multi-turn slices the `sessions` budget
  per conversation (`slices_session_budget`).
- Selects the `CellLauncher` (`Local`/`K8s`), serves the velo transport, and watches
  every child so a cell that dies before connecting **aborts** the run (`select!` of the
  collect loop against a child-exit channel) rather than hanging.
- Merges every cell's partition (global-order / concatenation / store-append per kind)
  into the single authoritative `native-v2.json`, runs the native export plane, and
  writes the merged heartbeat sidecar. Wrapped in `catch_unwind` so a merge/export panic
  becomes a typed `success:false` v2 envelope rather than a crashed subprocess.

The transport, launch, connection, START barrier, and cross-host / k8s details are in
`2026-07-15-velo-cell-transport-design.md`. A **tier-T2 hierarchical merge** is wired
(`engine::cellular_aggregator`, `CellLaunchContext::aggregator_count`): cells ship to a
round-robin aggregator (`cell_id % M`) that pre-merges before the controller, for
fan-in at very large N.

### Product reachability

Cellular is reachable through the ordinary `aiperf profile` frontend. `RuntimeConfig.cells`
(int, `ge=1`, default 1) is dumped into the protocol-v2 execute envelope, and the CLI
`--cells N` flag maps to it. Python still launches exactly one `aiperf` (which becomes
the controller and spawns the cells) and reads the one merged `native-v2.json`, so the
orchestrator flow is unchanged. `cells = 1` keeps the single-process path byte-for-byte
unchanged. E2e from the frontend: `rust/e2e/tests/test_cellular.rs` (a 3-cell run
reproduces the 1-cell input/output sequence-length distributions byte-for-byte through
the full presentation pipeline, over a varying-ISL dataset — verified for both the
profiling and warmup sections) and `test_graph_cellular.rs`.

A cell runs the *same* `execute.rs` path as any single-process run, differing only by an
injected `IssuanceAuthority` (`Direct` vs `CellularAutonomousIssuer`) and an env-gated
records sink (`CellRecordsShipper` ships over the transport instead of writing a report).
The cell's own (discarded) report accumulator holds sparse global slots — bounded by the
1-cell row count, transient, never serialized; the merged report is authoritative.

## Graph-mode cellular (built)

Graph programs (`dag_jsonl` / `weka_trace` / `dynamo_trace`) distribute cleanly because
each trace is a self-contained unit — there is no shared-sampler / per-turn-ordinal
problem, and the multi-turn restriction class the scheduled path fails on does not arise.
`PartitionedGraphTraceSource` (`graph::workload`) gives cell `k` of `C` the interleaved
global session ordinals `k, k+C, …` and cycles templates by the global ordinal, so the
union across cells reproduces the 1-cell trace set (`cell_count == 1` reproduces
`CyclingGraphTraceSource` exactly). One formula covers the finite (`--num-conversations`)
and unbounded (duration) modes. The runner selects the partitioned source when
`cell_count > 1` (and no static-node `request_limit`); the graph cell ships its captured
`RecordIngest` + heartbeat like the scheduled path; the controller admits it past the
synthetic/single-turn guard and concatenation-merges. Realized as the design's
*deterministic-per-topology* contract (same trace set across topologies, different
ownership + merge order), not byte-parity. Proven in `test_graph_cellular.rs`. Still
falls back to the single-cell cycler for a graph static-node `request_limit`; weighted/mix
RNG sampling will derive a per-cell stream here when that source lands.

## Multi-turn cellular (built, exact-fold only)

Conversation-level partition falls out of `PartitionedSampler` (filters on a
per-conversation draw counter: cell `k` owns draws `{k, k+C, …}`), so whole
conversations are cell-local and turns never split. A parallel `sessions` budget slice
(`owned_positions(sessions, k, C)`, non-graph only) makes each cell single-pass its owned
slice. Admitted for synthetic multi-turn + the known multi-turn file formats **only on
the exact-fold (store) merge** and **only with sequential/shuffle sampling**. It rides
the existing `StorePartition` variant — no new `CellMessage`. Still fails closed:
multi-turn on the retain path (per-turn dispatch ordinal ≠ per-conversation draw index —
fundamentally unavailable, the same silent-wrong-report class the S1 multi-phase fix
addressed), random sampling, a live-reply `inputs.json`, and duration/adaptive bounds.

## Bounded-memory cellular metrics (sketch, built)

`MetricsStorageMode::Sketch` streams each record into a per-`(phase, tag)` t-digest +
exact Welford stats then clears the row (accumulator memory O(1) in record count);
`RunCapture::finish_fold_into` folds each finalized record and drops it, retaining only
errored records (`ShardRecords::Folded`, on both the single-thread and sharded paths). A
sketch cell ships its folded sketch store as `CellMessage::StorePartition` (the same wire
form exact-fold uses) and the controller merges the per-cell t-digests associatively
(`merge_store_partitions`) — an O(cells × #tags × #phases × centroids) cross-cell merge,
counts/sums/rates/extrema exact, percentiles approximate. The record total travels with
the store (`ColumnStore::ingested_count`) since a sketch store retains no rows.

## Byte-parity scope

Exact, byte-for-byte 1-cell reproduction is claimed for: a seeded `concurrency`-bounded
phase, synthetic single-turn HTTP, single URL, no ramps, no cancellation. Everything the
guards *allow past* that is aggregate-equivalent, not byte-identical, and warned:
Poisson/Gamma/constant arrival pacing (`rate / cell_count`), post-send cancellation (each
cell applies the same per-request cancel probability to its slice so the aggregate rate
matches, but the exact cancelled subset — and thus the OSL/goodput it truncates — differs
because cell-local RNG draw order is not the 1-cell global order), multi-URL round-robin,
ramps, and an auto-derived seed. Graph/store merges are deterministic-per-topology
(within float-summation order), not byte-exact.

**Merged-report fidelity gaps** (deliberate): the coordinator's `finalize_run`
provenance (`distribution_id` / alias-resolved `endpoint_profiles`) rides the terminal
envelope rather than being reproduced as two 1-cell blocks; the grouped per-error message
list is absent (cells ship metric records with error/cancel flags, so error *counts*
survive but not the messages a cross-cell regroup would need); side-channel telemetry
sidecars (`server_metrics` / `gpu_telemetry` / `network_latency`) are scraped into each
cell's discarded scratch tree and omitted from the merged report — a loud startup `WARN`
names the dropped sidecars (not fail-closed, since those default on). Record-derived
distributions stay byte-identical (or deterministic-per-topology).

## Kubernetes launch (built)

The operator renders one JobSet with two replicatedJobs (`controller` ×1, `cells` ×N),
`enableDNSHostnames: true`. Discovery is **JobSet headless DNS**: the controller is
deterministically addressable and injected as `AIPERF_CELL_CONTROLLER_ADDR`; cell
identity is the downward-API job-index → `AIPERF_CELL_ID`. `K8sLauncher` spawns nothing
(pods already exist); failure is caught by the controller timeout ladder (register →
collect → artifact-upload, each env-overridable, each bailing loudly and drop-poisoning
the START event). **Zero RWX PVC** — all volumes `emptyDir`; cross-pod movement is velo
(metrics) + HTTP+zstd (artifacts/dataset) + a results sidecar.

## Fail-closed matrix

`validate_cellular_run_shape` + the kind validators reject: non-`http` transport
(gRPC/offline/dynosim); non-`{synthetic, file, public}` dataset; file/public formats
outside the single-turn allowlist; multi-turn on retain; scheduled phases outside
`{concurrency, poisson, gamma, constant}` or with `duration` / retain-path `sessions` /
`adaptive_scale`; caps `< cell_count`; graph phases with a static `requests` budget; mixed
store+record partitions; a no-`velo` build with `cells > 1`. Allowed-but-warned as above.

## What remains — the Phase-3+ north-star

The single-host-multi-process, cross-host velo, and k8s cases are built. The leap from
*static partition + collect at the end* to a *live coordinated fabric* — the dataset
**data plane** (producer-owned SPMC add-only broadcast with replay-on-attach), the
**monotonic phaser** control plane (synchronized START generalized to every phase
transition, cyclic-by-monotonic-counter), the per-request dispatch state machine with a
counted `DistributionMiss`, the end-of-warmup barrier that unblocks cross-cell adaptive
consensus, and records over a velo data plane — is specified in
`2026-07-15-ultimate-cellular-velo-runtime-design.md`, authoritative for the seams' next
era. Cross-host telemetry aggregation and gRPC/offline cellular remain the two standing
fail-closed lifts.
