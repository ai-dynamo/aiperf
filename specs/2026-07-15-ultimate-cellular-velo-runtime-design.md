<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The ultimate cellular runtime — velo transport, dataset fan-out, phaser control plane

**Status:** Built (substrate + forward plane, AIPerf-side over velo primitives); remaining items are velo-native optimizations
**Date:** 2026-07-15
**Repo:** `aiperf`
**Relationship:** Extends `2026-07-12-cellular-ready-seams-and-roadmap.md` (the S1–S5
measurement seams) and `2026-07-15-velo-cell-transport-design.md` (the velo transport
swap). This spec is authoritative where it extends them.

> **Grounding.** §1 and §3–§4 cite shipped code (`crate/src/file.rs`). §2 is a
> source-grounded inventory of what velo provides vs what AIPerf builds on top. §5–§6
> mark the few remaining velo-native optimizations explicitly. Verify against `rust/`.

---

## 0. Thesis

AIPerf's cellular runtime partitions one benchmark across N processes/nodes so a single
run can generate 1M+ concurrency and hold a dataset too large for one process's RAM. The
**measurement seams are byte-exact** (S1 issuance, S2 shard partitions, S3
heartbeat/sketch, S4 deterministic partition, S5 controller/cell topology) and the
**transport is done** (velo, connect-by-endpoint, zero discovery). This spec covers the
leap from *static partition + collect at the end* to a *live coordinated fabric*: the
controller generates a dataset once and **fans it out** to cells, drives them through a
**monotonic phaser** (synchronized START generalized to every phase transition), and
collects records in **bounded memory** without a shared RWX filesystem. The organizing
idea: **one add-only broadcast stream for the data plane, one monotonic phaser for the
control plane, and a finalize-then-barrier discipline that makes "issue request R before
R arrived" impossible in the common case and a counted, visible `DistributionMiss` in
the streaming case.** Both the data plane and the phaser are built AIPerf-side as a layer
over velo's MPSC + event primitives (velo lacks a native producer-owned broadcast anchor
and a monotonic phaser; §2, §6).

---

## 1. The built substrate

### 1.1 Topology & lifecycle

One `aiperf` process is the **controller**; N are **cells**; Python sends one protocol-v2
`execute` and reads one merged `native-v2.json`. Mode dispatch reads
`/run/cfg/runtime/cells` (`cell_launcher::cell_count_from_envelope`, clamp `[1, 1024]`);
`> 1` diverts to `run_controller` → `cellular_controller::run_cellular`; `--cell` →
`run_cell`; else the ordinary single-process `run_v2`. `cells = 1` is byte-for-byte
unchanged.

Controller sequence (`cellular_controller::run_cellular`, `#[cfg(feature = "velo")]`):
validate shape → `CellularRunKind::detect` (Scheduled|Graph) → resolve seed + metrics
config → tokio runtime → scratch `temp_root` (RAII `ScratchTreeGuard`) → optional
`ArtifactUploadServer` (started **before** launch so an early k8s pod upload isn't lost)
→ bind velo → create the START event → precompute per-cell sliced envelopes
(`build_cell_envelope`) → bind the four control handlers → `select_launcher().launch()` →
per-cell `wait_failure` watchers → **synchronized START barrier** → **collect loop** (one
terminal partition + latest heartbeat per cell) → **merge** → `NativeReport` + native
export plane → merged heartbeat sidecar → artifact barrier + concat → scratch cleanup.
Wrapped in `catch_unwind` so a merge/export panic becomes a typed v2 failure envelope.

Cell sequence (`cellular_cell`): `fetch_cell_envelope()` (build velo → `connect_controller`
→ `register(cell_id)` → **`await_start`**) → download dataset if needed → ordinary
`run_v2`, made cell-aware purely by env → at finalize, `CellRecordsShipper` ships one
terminal partition + heartbeat over a **fresh, isolated velo instance on a dedicated
thread**.

### 1.2 The five seams (S1–S5)

Built and detailed in `2026-07-12-cellular-ready-seams-and-roadmap.md`. In brief:
`IssuanceAuthority` (`cellular/issuance.rs`) self-assigns the absolute single-cell slot
with zero coordinator hop; `RecordsShardPartition`/`ColumnStorePartition`
(`cellular/shard.rs`) with three merges (global-order byte-identical, concatenation
deterministic-per-topology, store-append ULP-tolerant); `MetricsHeartbeat` + t-digest
(`cellular/heartbeat.rs`, `cellular/sketch.rs`); `ModuloCellPartition`
(`cellular/partition.rs`, `owned_positions(total, k, C) = ceil((total - k) / C)`, reused
by the sharded thread-per-core runtime); and the controller/cell drivers (`engine/`).

### 1.3 Transport — velo, connect-by-endpoint, zero discovery

`cellular/transport/`: two async traits `CellClient::send` / `ControllerTransport::recv`
over a `CellMessage` enum (`Heartbeat`, `Partition(RecordsShardPartition)`,
`StorePartition(Box<ColumnStorePartition>)`), all **MessagePack via velo raw payloads**.
Four named handlers (`register` unary → envelope + START event; `heartbeat`
fire-and-forget; `partition` / `store_partition` unary → `CellAck`). The velo impl is
`velo_transport.rs` (`VeloControllerTransport` / `VeloCellClient`), gated on the `velo`
feature. Connection is address-first (`connect.rs::connect_controller` over the fork's
`Velo::connect(Endpoint)`). Full detail: `2026-07-15-velo-cell-transport-design.md`.

**Fresh-instance-ack pattern**: a cell touches velo twice on separate short-lived
runtimes (fetch, then a fresh ship instance the controller never registered), so every
ship DTO carries the shipping instance's own `PeerInfo`; the partition handler
`register_peer`s it before replying so the `CellAck` routes home.

### 1.4 Synchronized START — and the velo correction load-bearing for §4

The controller creates a velo **distributed event**, threads its `EventHandle` into every
`RegisterReply`; the register handler counts registrations in an `AtomicU32` and the Nth
notifies an `all_registered` `Notify`; the controller `select!`s on that (bounded by
`register_timeout`, default 5 min) then triggers. Cells block in `await_start`. A bail
before trigger drop-poisons the event so waiters error rather than hang.

**Velo correction (load-bearing for §4):** velo distributed events are **single-shot**
and have **no register-count→trigger barrier primitive** — the only aggregation is
`merge_events` (AND-join). The count barrier is **AIPerf's own** `AtomicU32` + `Notify`.
`EventAwaiter` *is* a real `Future` with a completed-event cache (local + LRU-1000), so a
late awaiter resolves immediately. This matters because the **phaser (§4) is a sequence
of such events/streams, and the count/threshold logic is built on the AIPerf side.**

### 1.5 Barrier-synchronized timing origin (opt-in)

The START barrier releases cells together, but each cell then captured its run origin
(`start_ns = clock.now_ns()`) inside `execute` **after** its own per-cell setup (tokenizer
load, dataset compile, connect). Cells with a larger shard / slower setup therefore zeroed
their record timeline at a *later* instant than their peers, so the merged report's
cross-cell absolute timestamps referenced a different `t0` per cell (all latency/throughput
metrics are *differences* and so were unaffected — only absolute per-record timestamps
drifted).

Built behind `AIPERF_CELL_SHARED_ORIGIN` (default off): the cell captures a
`RealClockAnchor` the instant its velo START barrier releases — inside `fetch_cell_envelope`,
the shared logical instant every cell reaches together, *before* its per-cell setup
(`engine::cell_origin::capture_cell_shared_origin`). At run start, `execute` derives its
origin from `cell_origin::run_origin_now_ns(&clock)`: when a barrier anchor was captured it
returns `clock.now_ns() - barrier.now_ns()` (read at one instant so the shared wall-`now`
cancels), i.e. the barrier's reading on the execute clock's own timeline, shifting every
record's timestamp forward so it is measured from the barrier. Default off ⇒
`run_origin_now_ns` returns `clock.now_ns()` unchanged (single-process and existing cellular
runs byte-unchanged). **Cross-host:** each cell zeroes at its *own* clock reading of the
barrier-release instant (not an absolute controller `t0`, which would import clock skew);
the barrier guarantees those instants coincide within network latency. Proven by unit
`cell_origin::tests` and e2e `test_cellular_shared_origin_zeroes_at_the_barrier`. Not yet
default-on (baking); the controller's own report provenance still records its local finalize
time.

### 1.6 Multi-turn, graph, sketch cellular; artifact/dataset shipping; k8s

All built and detailed in the seams spec. In brief:

- **Multi-turn** (exact-fold only): conversation-level partition via `PartitionedSampler`
  + a `sessions` budget slice; sequential/shuffle sampling; rides `StorePartition`.
  Retain-path multi-turn, random sampling, live-reply, duration/adaptive stay fail-closed.
- **Graph**: `PartitionedGraphTraceSource` gives cell `k` the interleaved global session
  ordinals `k, k+C, …`; concatenation merge; rejects a static `requests` budget.
- **Sketch** (bounded memory): `MetricsStorageMode::Sketch` fold-and-drop
  (`ShardRecords::Folded`) on both single-thread and sharded paths; a sketch cell ships
  its folded sketch store as `CellMessage::StorePartition` merged associatively
  (`merge_store_partitions`) — O(cells × #tags × #phases × centroids), counts/sums/rates/
  extrema exact, percentiles approximate; record total via `ColumnStore::ingested_count`.
- **Artifact & dataset shipping (Stages D–G):** per-record file artifacts ship on a
  separate HTTP + streaming-zstd plane (`engine::artifact_shipping`), distinct from the
  velo metrics partition. Same-host = Stage D concat (`concatenate_cell_artifacts`);
  cross-host = Stage E `ArtifactUploadServer` (axum, streaming zstd, `.part`+atomic
  rename, path-traversal rejection, `watch`-based barrier); Stage G serves a `file`/`path`
  dataset over `GET /dataset/{name}` + zstd. Bulk per-record bytes are **not** shipped over
  the velo control plane (velo buffers whole payloads in RAM both ends); this bounded-memory
  HTTP plane (or shared object storage) is the cross-host mechanism, the seam §5.4 targets
  replacing with a streaming velo data plane. Knobs:
  `AIPERF_CELL_HTTP_ARTIFACT_SHIPPING` (default on), `AIPERF_CELL_ARTIFACT_HTTP_FORCE`,
  `AIPERF_CONTROLLER_ARTIFACT_BIND`.
- **Kubernetes**: one JobSet, two replicatedJobs, `enableDNSHostnames`, zero discovery
  (headless DNS + downward-API job-index), `K8sLauncher` spawns nothing, emptyDir/no-RWX,
  controller timeout ladder.

### 1.7 The fail-closed matrix

`validate_cellular_run_shape` + kind validators reject: non-`http` transport; non-
`{synthetic, file, public}` dataset; file/public formats outside the single-turn
allowlist; multi-turn on retain; scheduled phases outside `{concurrency, poisson, gamma,
constant}` or with `duration`/retain-`sessions`/`adaptive_scale`; caps `< cell_count`;
graph phases with a static `requests` budget; mixed store+record partitions; a no-`velo`
build with `cells > 1`. Allowed-but-warned (aggregate-equivalent): multi-URL, ramps,
post-send cancellation, rate pacing (`rate / cell_count`), auto-derived seed.

---

## 2. Velo primitive inventory (source-grounded)

From velo `main` + `feat/connect-by-endpoint`. Every load-bearing claim was verified
against source across the velo (`ai-dynamo` + fork, 27+ branches) and dynamo/kvbm
(200+ branches) repos, not just `main`.

- **Connect-by-endpoint — landed** (fork `feat/connect-by-endpoint`): `Velo::connect(Endpoint)
  -> PeerInfo`, `Endpoint::{Tcp, Uds}`, `ActiveMessageClient::connect(WorkerAddress)`
  (registers a provisional peer, `_hello` handshake with `peer_info` in the response,
  re-registers under the responder's real id). `WorkerAddressBuilder` stays `pub(crate)`;
  you reach a peer by address only through `Endpoint`. The bootstrap/`serve_bootstrap`
  mechanism is superseded.
- **Streaming anchors — SPSC + MPSC only, both consumer-owned, no replay.** There is **NO
  producer-owned fan-out (SPMC/MPMC, 1→N with per-consumer replay-on-attach)**;
  `AnchorKind{Spsc, Mpsc}` is a 1-bit wire discriminator with no third kind, and a
  tree-wide grep for `replay|broadcast|spmc|mpmc` returns zero streaming hits.
  **AIPerf builds this itself** (§3.1, §6.1).
- **Rendezvous** (`register_data(Bytes) -> DataHandle` / `get -> (Bytes, lease)`): on
  `main`, transfer is 512 KiB chunks over AM and RDMA paths `bail!` as unimplemented; on
  the unmerged `feat/rendezvous-nixl` branch it is a **real** NIXL+UCX one-sided RDMA READ
  path (feature-gated). Either way it stays **put-once / pull-by-handle, single-owner-slot**
  — it upgrades the *transfer mechanism*, not the producer→N-consumer broadcast shape (the
  handle **fan-out** still must be built on top).
- **Distributed events — single-shot, real `Future`, completed-cache; no countdown.** No
  register-count barrier, no monotonic multi-generation "await ≥ N" event. The phaser (§4)
  builds this AIPerf-side.
- **Transports & addressing:** `main` has TCP/UDS/NATS/gRPC/ZMQ; QUIC/TIPC/NIXL are real
  but feature-gated on unmerged branches; only HTTP is universally absent. `WorkerAddress`
  is an opaque MessagePack `HashMap<transport-key → endpoint-bytes>`, not a `scheme://` URL
  (routing is map-key self-selection); `tcp://`/`uds://` are prefixes *inside* the value.

**Net:** the two central conclusions — *build a producer-owned SPMC broadcast anchor with
per-consumer replay-on-attach*, and *build a monotonic phaser* — hold against every branch;
nothing in velo or kvbm provides either. NIXL rendezvous and QUIC/TIPC are built-but-unmerged
(branch-gated), which shortens the "velo must add" list for the *transfer* layer but leaves
the *fan-out* and *phaser* layers genuinely greenfield. Reusable to port back into velo: the
kvbm `ReplayStream` snapshot-then-live pattern and the `PeerCommitted`/`PeerAvailable` seal
state machine (§6).

---

## 3. The dataset **data plane** (SPMC add-only broadcast) — built

**Goal:** the controller generates a large dataset once and streams it hot-in-RAM to each
cell's shard, replacing "every cell regenerates from a shared seed" and "ship files over
HTTP."

### 3.1 The primitive: a producer-owned broadcast anchor with replay-on-attach

Built as `cellular::broadcast` (an AIPerf-side layer over velo's MPSC + rendezvous, since
velo has no native SPMC anchor — §2, §6.1). Modeled on the kvbm `ReplayStream`
(snapshot-and-go-live atomically under one lock), generalized to N connectors:

- Owns `{ history: Vec<Chunk> (append-only, producer call order, incl. terminal Finalized),
  senders: Vec<Sender> }` under **one lock**.
- `attach(consumer)`: lock → clone `history` into this consumer's replay buffer → push a
  fresh sender → unlock.
- `add(chunk)`: lock → `history.push` → fan-out to every sender → unlock.
- **The single invariant:** membership (attach/detach) serializes on the *same* lock as
  delta-append, so each consumer's `(replay snapshot ⊎ live tail)` reconstructs the full
  producer order with no gap/dup at *its* attach seam, by construction. Proven by 4 tests
  (full-order reconstruction for every attach time incl. post-finalize; no gap/dup at the
  seam across 50 interleaved attach/add; add-after-finalize rejected; a slow consumer does
  not stall the producer).

**Collapsed for AIPerf:** add-only + finalize (drop kvbm's separate commit/available split
— a chunk's bytes exist at `add` time, so committed ≡ available). One monotonic add-only
stream, terminal finalize.

### 3.2 Decouple pull/index from issue

Two planes at two cadences. The **data plane** (`add` = "pull this next chunk") is bulk,
async, order-insensitive: cells pull and build a **local index keyed by stable
`request_id`**, never by arrival position (sidestepping the arrival-ordered ≠
position-ordered gotcha). The **control plane** (§4 phaser: "issue request R now") is
per-request, timed, dispatch-driven.

### 3.3 Routed fan-out (the memory fork)

Cellular exists for memory scaling, so a plain uniform broadcast (every cell holds the full
dataset) defeats the point at large N. The built v1 is uniform fan-out + **consumer-side
owned-filter**: each cell sees every frame header but only *indexes* the requests it owns
(round-robin owned positions), so RAM is 1/N even if wire bandwidth isn't. Server-side
routing (a `target cell_id` on the broadcast frame) is a later optimization.

### 3.4 Built end-to-end

`cellular::dataset_session` + `transport::dataset_velo` + runner (`AIPERF_CELL_DATASET_FANOUT`,
default off): the controller broadcasts the dataset request-ids (phaser `ShardsAvailable` per
chunk), every cell builds its owned shard over velo and dispatches it exactly-once. 4 unit
tests (owned-filter tiling, late-attach replay, arrival-order independence, disjoint owned
indexes over velo) + e2e `test_cellular_dataset_fanout_matches_baseline` (fan-out reproduces
the baseline deterministic metrics exactly). The remaining last mile: a `ControlledIssuer`
*workload* that materializes the request **body** from the owned index instead of the sampler
(today the fan-out delivers + owned-filters + runs the dispatch state machine over each cell's
shard, proving the path). Until velo RDMA lands, chunk bytes move over chunked-AM rendezvous
or the HTTP+zstd Stage-G plane; the broadcast anchor carries handles/notifications, not
megabytes — keep the control notification (small) and the bulk pull (large) on separate
transports.

---

## 4. The **control plane** (monotonic phaser) — built

**Goal:** generalize the one-shot synchronized START (§1.4) into a **monotonic generation
counter the controller increments as the benchmark progresses**, observed with
replay-on-attach.

### 4.1 What it is

Built as `cellular::phaser` + its velo distribution `transport::phaser_velo`: an SPMC
monotonic stream whose payload is `{generation, transition}` where transition ∈ `{Started,
ShardsAvailable(k), PhaseAdvance(Warmup|Profiling|Drain), Done}`. Monotonic +
replay-on-attach ⇒ a late-joining cell reads "current generation = G" atomically then
live-follows, missing no transition. It subsumes three things: **START** (generation 1 =
"go"), **dataset progression** (`ShardsAvailable(k)` ⇒ shards `[0, k)` pullable, driving
§3's `add`/`finalize` from the cell's view), and **phase transitions** (warmup→profiling→drain
as generation steps). Proven by 5 phaser unit tests (monotonicity, replay for passed targets,
block-then-wake on live advance, cyclic gate-on-≥ with no ABA, await-after-finalize) + 1 velo
test (a cell subscribing over two in-process velo instances reaches pre-subscribe generations
via replay, then observes live advances).

**Cyclic from the start:** never reset the counter; a looping run (ramp steps, multi-round
sweeps) keeps incrementing and cells gate on `generation >= threshold_for_this_round`, never
equality → no ABA, cyclic for free (the java `Phaser` monotonic-generation discipline; the
completed-event cache §1.4 gives late/repeat awaiters immediate resolution).

### 4.2 Phaser-driven START, integrated

The control plane drives a real run behind `AIPERF_CELL_PHASER_START` (default off): the
controller binds a `PhaserServer` + `advance(Started)`; cells subscribe + await generation 1.
E2e `test_cellular_phaser_start_matches_event_start` — phaser-START reproduces event-START's
deterministic metrics exactly (byte-identical request_count/ISL/OSL). The event START stays
byte-unchanged by default.

### 4.3 One-way vs barrier

Pure progress broadcast (cells react, controller never waits) is SPMC-only; a true barrier
(controller waits until *all* cells reach G before G+1) needs the fan-in arrival half too.
START and end-of-phase drain *are* barriers (built via the AIPerf-side count over MPSC, as
START does); dataset progression is one-way. The **end-of-warmup barrier** is the
precedent-backed one (the mocker drain-barrier pattern, a counted one-ack-per-event gate) —
"every cell finishes warmup before any begins profiling," which is what unblocks **cross-cell
adaptive-scale consensus** (today fail-closed). That barrier is the standing forward item on
the control plane.

### 4.4 Where it hooks into dispatch (built seams)

The dispatch landing points exist: a `ControlledIssuer` `Workload` impl awaiting "issue id R"
off a channel and calling `ScheduledRuntime::issue_turn_with_hooks_and_cancellation`;
`TurnLifecycleObserver::on_issue` (the synchronous "this turn is admitted, before backend
dispatch" seam); `IssuanceAuthority::global_ordinal` (binds the assigned global id to the
turn); and `SlotPool` admission as the backpressure gate whether arrivals are paced or
externally commanded.

### 4.5 The readiness interlock — "issue R before R arrived" — built

Built as `cellular::dispatch_state`: a per-request **state machine** on the cell
`Unknown → Indexed → InFlight → Done`. The "issue R" handler dispatches on state:
`Indexed` → issue → `InFlight`; `InFlight`/`Done` → no-op (dedup; exactly-once-issue per id);
`Unknown` → the race, handled in two layers (mirroring kvbm's bilateral seal enforcement):

- **Source side (make it rare/impossible):** *bounded runs* → §3's finalize→all-drained
  barrier before START, so every owned request is `Indexed` before any issue (Unknown
  impossible). *Streaming runs* → keep the control plane causally downstream of the data
  plane (the controller doesn't publish the generation authorizing window `k` until
  `ShardsAvailable(k)`).
- **Sink side (guard the residual — reorder, cell reattach with cold index, controller bug):**
  demand-pull R with a bounded deadline; if it lands, issue and tag the record with the
  incurred wait (measured, not hidden); if not, classify a **`DistributionMiss`** — a
  distinct, counted, surfaced error class, never a silent skip.

Proven by 3 unit tests (issue-once-then-dedup, unknown-is-a-counted-miss, in-flight
accounting) + run over the fan-out index in the §3.4 e2e (every 4-cell run dispatched its
owned slice `issued=owned completed=owned misses=0`, fail-closed on any miss).

**Two regimes:** *bounded (request-count)* distributes all → barrier → dispatch (race-free,
the common case); *continuous (rate/duration/streaming)* streams the dataset concurrently with
the phaser generation as the backpressure interlock gating dispatch behind availability.

---

## 5. Bounded-memory record collection (no RWX PVC)

### 5.1 Sketch StorePartition ship — built

Cellular high-request-rate memory is bounded: a sketch cell ships its per-`(phase, tag)`
`ColumnStorePartition`-of-sketches as `CellMessage::StorePartition`; the controller
`merge_store_partitions` appends stores → **O(cells × #tags × #phases × centroids)**, no
per-record retention, no shared FS (§1.6). Per-worker fold-and-drop (`ShardRecords::Folded`,
`finish_fold_into`) already bounds per-cell RSS on both the single-thread and sharded paths.

### 5.2 Streaming per-worker finalization (deeper fix, forward)

Bound the *worker observer* peak, not just the accumulator: per-record worker finalization
keyed by phase, so a worker folds and drops each record at completion instead of retaining
until end-of-run drain. This is the ceiling that sets exact-collection peak RSS at 1M
concurrency; orthogonal to cellular but compounds with it. Not yet built for the retain path.

### 5.3 Records over a velo data plane (forward)

Record collection is N independent 1:1 sessions (each cell → controller), *not*
multi-connector — the kvbm session verbatim, no fan-out machinery. When velo gains a streaming
data plane with real backpressure (and eventually RDMA), the per-record file artifacts can
ship over it instead of the bespoke HTTP+zstd server, collapsing two transports into one.
Until then the HTTP+zstd plane (§1.6) stays the cross-host mechanism.

---

## 6. What velo could add (upstream asks, optimizations)

The forward plane is built AIPerf-side over velo primitives, so these are **bandwidth/latency
optimizations, not blockers**:

1. **Producer-owned broadcast anchor (SPMC), replay-on-attach, producer-owned finalize** —
   a native velo third `AnchorKind` would replace the AIPerf-side `cellular::broadcast` layer
   (§3.1). Smallest form: a producer-side registry `{append-only replay log, DashMap<ConsumerId,
   Sender>}` + `attach_broadcast` (replay-then-live) + producer finalize, reusing the existing
   `AnchorManager`/`flume`/`FrameTransport` control plane.
2. **Monotonic / countdown event** — a native "fire after N arrivals" or "await ≥ N generation"
   event would make the AIPerf-side phaser count logic (§1.4, §4) a thin wrapper.
3. **Merge/enable NIXL rendezvous** — already built on `feat/rendezvous-nixl`; merging + a `nixl`
   feature enable gives zero-copy shard *bytes*. Not on the critical path (the handle fan-out,
   ask #1, is).
4. **Optional routing key on broadcast frames** — for server-side sharded fan-out (§3.3).

The connect-by-endpoint ask is delivered on the fork; upstreaming to `ai-dynamo/main` is the
remaining PR. Reusable to port back: the kvbm `ReplayStream` snapshot-then-live pattern and the
seal state machine.

---

## 7. Standing forward items (in order)

| Item | Unlocks | Status |
|---|---|---|
| Sketch StorePartition ship (§5.1) | bounded-memory cellular metrics at 1M rate | built |
| SPMC broadcast + phaser + dataset fan-out (§3, §4) | hot-dataset fan-out, every-phase progression, cyclic runs | built (AIPerf-side) |
| `ControlledIssuer` body materialization (§3.4) | dispatch the request body from the owned index, not the sampler | forward (last mile) |
| End-of-warmup barrier (§4.3) | **cross-cell adaptive-scale consensus** (lifts a fail-closed) | forward |
| Streaming per-worker finalize (§5.2) | multi-turn retain, duration-bounded cellular (ragged counts) | forward |
| Records over velo data plane (§5.3) | one transport, drop HTTP+zstd server | forward (velo streaming data plane) |
| Side-channel telemetry aggregation | server_metrics/GPU/network in the merged report | forward |
| gRPC/offline cellular | non-HTTP cellular | forward (cell issuer/ship on those executors) |
| Native velo SPMC/countdown/NIXL (§6) | bandwidth/latency optimization | velo change |

---

## 8. Known defects & fidelity gaps

- **Operator `"workers"` vs `"cells"`** (`_classify_jobset_failure`): a cell-job failure is
  not matched as the restartable case → run fails unclassified. Fix to the `"cells"`
  replicatedJob name.
- **Merged-report fidelity gaps** (deliberate): coordinator finalize provenance, grouped
  per-error detail (counts survive), and side-channel sidecars — record-derived distributions
  stay byte-identical.
- **Aggregate-equivalent (not byte-exact), warned:** rate pacing, post-send cancellation
  subset, multi-URL round-robin, ramps. Byte-parity holds only for a seeded pure `concurrency`
  phase.
- **`was_cancelled` left false** (no cross-cell cancellation consensus).

---

## 9. Testing

- **Unit:** SPMC broadcast replay-on-attach under N concurrent attach/add (the racy-state
  invariant §3.1); phaser generation monotonicity + cyclic gating; per-request state machine
  (dedup on InFlight/Done, bounded-await + `DistributionMiss` on Unknown); sketch StorePartition
  merge; `cell_origin` origin math.
- **E2e** (`rust/e2e/tests/test_cellular*.rs` / `test_graph_cellular.rs`): dataset fan-out
  reproduces the baseline dataset SET; phaser-driven START matches event START; shared origin
  zeroes at the barrier; sketch cellular memory bound. A streaming late-reattach test exercising
  the `Unknown`→demand-pull `DistributionMiss` path is the forward addition.
- **Regression:** every §1 byte-parity assertion stays green — the forward work must not perturb
  the built exact-merge path for the bounded `concurrency` case.
