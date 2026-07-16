<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The ultimate cellular runtime — velo transport, dataset fan-out, phaser control plane

**Status:** Design / north-star (built substrate + forward plan; grounded in code)
**Date:** 2026-07-15
**Repo:** `aiperf` (branch `velo-connect`, on top of `ajc/rust`)
**Supersedes/extends:**
- `2026-07-12-cellular-ready-seams-and-roadmap.md` (the S1–S5 seams; this is the Phase‑3+ target).
- `2026-07-15-velo-cell-transport-design.md` (the velo transport swap; its mechanism‑A/A′/B
  question is now **resolved** — connect‑by‑endpoint landed, see §2.1).

Both prior specs stay authoritative for their eras; where this one revises them it says so, and
each receives a dated addendum pointing here (per the append‑only spec rule).

> **Grounding.** Everything in §1 cites shipped code (`crate/src/file.rs:line`). Everything in
> §3–§6 is designed‑but‑not‑built and says so. This spec was authored from an exhaustive sweep of
> the in‑flight rust worktrees — `velo-connect`, `ajc/rust`, `rust-cellular-multiturn`,
> `rust-operator`, `rust-dispatcher`, `wt-sketch-only` — and the velo `main` + `feat/connect-by-endpoint`
> source. Verify against `rust/` before relying on any §3+ feature.

---

## 0. Thesis

AIPerf's cellular runtime partitions one benchmark across N processes/nodes so a single run can
generate 1M+ concurrency and hold a dataset too large for one process's RAM. The **measurement
seams are done and byte‑exact** (S1 issuance, S2 shard partitions, S3 heartbeat/sketch, S4
deterministic partition, S5 controller/cell topology). The **transport is done** (velo,
connect‑by‑endpoint, zero discovery). What remains is the leap from *static partition + collect at
the end* to a *live, coordinated fabric*: the controller generates a dataset once and **fans it out**
to cells, drives them through a **monotonic phaser** (synchronized START generalized to every phase
transition), and collects records in **bounded memory** without a shared RWX filesystem. The
organizing idea, from the velo/kvbm design conversation: **one add‑only broadcast stream for the
data plane, one monotonic phaser for the control plane, and a finalize‑then‑barrier discipline
that makes "issue request R before R arrived" impossible in the common case and a counted, visible
miss in the streaming case.**

---

## 1. What is BUILT today (the substrate)

### 1.1 Topology & lifecycle

One `aiperf` process is the **controller**; N are **cells**; Python sends one protocol‑v2
`execute` and reads one merged `native-v2.json`. Mode dispatch: `runner_protocol` reads
`/run/cfg/runtime/cells` (`cell_launcher.rs::cell_count_from_envelope`, clamp `[1,1024]`); `>1`
diverts to `run_controller` → `cellular_controller::run_cellular`; `--cell` → `run_cell`; else the
ordinary single‑process `run_v2` (`runner/src/main.rs:63-104`). `cells=1` is **byte‑for‑byte
unchanged** — `ModuloCellPartition::from_env()` returns `None` off‑cell → `DirectIssuanceAuthority`
(identity ordinal) → single‑process output.

Controller sequence (`cellular_controller.rs::run_cellular`, `#[cfg(feature="velo")]`): validate
shape → `CellularRunKind::detect` (Scheduled|Graph) → resolve seed + metrics config → 2‑worker
tokio runtime → scratch `temp_root` (RAII `ScratchTreeGuard`) → optional `ArtifactUploadServer`
(started **before** launch so an early k8s pod upload isn't lost) → bind velo
(`controller_bind_and_endpoint`) → create the **START event** → precompute per‑cell sliced
envelopes (`build_cell_envelope`) served via a `SpecFor` closure → `bind_controller` (4 handlers) →
`select_launcher().launch()` → per‑cell `wait_failure` watchers → **synchronized START barrier** →
**collect loop** (one terminal partition + latest heartbeat per cell) → **merge** →
`NativeReport` + native export plane → merged heartbeat sidecar → Stage‑E artifact barrier + Stage‑D
concat → scratch cleanup. Wrapped in `catch_unwind` so a merge/export panic becomes a typed v2
failure envelope.

Cell sequence (`cellular_cell.rs`): `fetch_cell_envelope()` (build velo → `connect_controller` →
`register(cell_id)` → **`await_start`**) → `download_cell_dataset_if_needed` (Stage G) → ordinary
`run_v2`, made cell‑aware purely by env → at finalize, `CellRecordsShipper` ships one terminal
partition + heartbeat over a **fresh, isolated velo instance on a dedicated thread**.

### 1.2 The five seams (S1–S5) — all built

- **S1 `IssuanceAuthority`** (`cellular/issuance.rs`): object‑safe `global_ordinal(flat_local,
  phase_base, within_phase_local)`. `DirectIssuanceAuthority` = identity; `CellularAutonomousIssuer`
  = `phase_base + within_phase_local*cell_count + cell_id` — each cell self‑assigns the **absolute
  single‑cell slot with zero coordinator hop**, which is what makes the merged report byte‑identical.
- **S2 `RecordsShardPartition` / `ColumnStorePartition`** (`cellular/shard.rs`): the two wire forms;
  three merges — `merge_records_in_global_order` (scheduled, validates the ordinal union is a
  permutation of `0..total`, stable‑sort, re‑ingest → **byte‑identical**), `merge_records_by_concatenation`
  (graph, dense re‑number, deterministic‑per‑topology), `merge_store_partitions` (exact‑fold,
  ULP‑tolerant).
- **S3 `MetricsHeartbeat` + t‑digest** (`cellular/heartbeat.rs`, `cellular/sketch.rs`): live
  snapshot; the merging t‑digest (Dunning K1 scale, exact min/max/count/sum, associative
  order‑independent `merge`).
- **S4 `CellPartition` / `ModuloCellPartition`** (`cellular/partition.rs`): round‑robin
  `i % cell_count == cell_id`, disjoint+complete; `owned_positions(total,k,C)=ceil((total-k)/C)`
  (`cell_launcher.rs:182`) is the per‑cell share, reused by the sharded thread‑per‑core runtime.
- **S5 controller/cell drivers** (`runner_protocol/`): the topology, launch, and merge orchestration.

### 1.3 Transport — velo, connect‑by‑endpoint, zero discovery (§2.1)

`cellular/transport/mod.rs`: two async traits `CellClient::send` / `ControllerTransport::recv` over a
`CellMessage` enum (`Heartbeat{cell_id, Box<MetricsHeartbeat>}`, `Partition(RecordsShardPartition)`,
`StorePartition(Box<ColumnStorePartition>)`), all **MessagePack via velo raw payloads** (JSON can't
round‑trip t‑digest `min=+inf` or record NaN). Four named handlers: `aiperf.cell.register`
(unary → `RegisterReply{envelope, start_event}`), `.heartbeat` (fire‑and‑forget `am_send`),
`.partition` / `.store_partition` (unary → `CellAck`). The velo impl is `velo_transport.rs`
(`VeloControllerTransport` / `VeloCellClient`), gated on the `velo` cargo feature
(`Cargo.toml: velo = ["dep:velo", "dep:zstd"]`, in `default`).

**Fresh‑instance‑ack pattern**: a cell touches velo twice on separate short‑lived runtimes (fetch,
then a fresh ship instance the controller never registered), so every ship DTO carries the shipping
instance's own serialized `PeerInfo`; the partition handler `register_peer`s it before replying so
the `CellAck` routes home (`velo_transport.rs`, test `ship_from_a_fresh_instance_is_acked`).

### 1.4 Synchronized START — built (but note the velo correction)

The controller creates a velo **distributed event** (`velo.event_manager().new_event()`), threads
its `EventHandle` into every `RegisterReply`; the register handler counts registrations in an
`AtomicU32` and the Nth `notify`s an `all_registered` `Notify`; the controller `select!`s on that
(bounded by `register_timeout`, default 5 min) then `start_event.trigger()`. Cells block in
`await_start` → `event_manager().awaiter(handle).await`. A bail before trigger **drop‑poisons** the
event so waiters error rather than hang. Proven by `synchronized_start_releases_all_cells_together`.

> **Velo correction (load‑bearing for §4):** velo distributed events are **single‑shot** and have
> **no register‑count→trigger barrier primitive** — the only aggregation is `merge_events` (AND‑join
> of all N). The count barrier above is **AIPerf's own** `AtomicU32`+`Notify`, not a velo feature.
> `EventAwaiter` *is* a real `Future`, and there is a completed‑event cache (local + LRU‑1000), so a
> late awaiter resolves immediately. This matters because the **phaser (§4) is a sequence of such
> events/streams, and the count/threshold logic must be built on the AIPerf side either way.**

### 1.5 Artifact & dataset shipping — Stages B–G (built)

Per‑record **file** artifacts (records/raw/CSV/parquet/outputs/inputs.json) ship on a **separate
HTTP + streaming‑zstd plane** (`runner_protocol/artifact_shipping.rs`), distinct from the velo
metrics partition:
- **Stage D** (same host): cells write into `temp_root/cell-{id}`; controller `concatenate_cell_artifacts`.
- **Stage E** (cross host): `ArtifactUploadServer` (axum; `POST /cell/{id}/artifact/{*file}`, `/done`;
  `DefaultBodyLimit::disable`); streaming zstd (`CHUNK_SIZE=65536`, `ZSTD_LEVEL=3`), decode on
  `spawn_blocking`, `.part`+atomic rename, path‑traversal rejection; `wait_for_cells` barrier on a
  version‑tracked `watch::Sender<HashSet<u32>>` (closes the lost‑wakeup race a bare `Notify` had).
- **Stage G** (cross‑host dataset): controller serves a `file`/`path` dataset over `GET /dataset/{name}`
  + zstd; cell downloads and recompiles locally.
- Knobs: `AIPERF_CELL_HTTP_ARTIFACT_SHIPPING` (default on), `AIPERF_CELL_ARTIFACT_HTTP_FORCE`
  (test/dev loopback seam), `AIPERF_CONTROLLER_ARTIFACT_BIND` (`0.0.0.0:9600`).

**Deliberate boundary**: bulk per‑record bytes are **not** shipped over the velo control plane (velo
buffers whole payloads in RAM both ends); the cross‑host mechanism is this bounded‑memory HTTP plane,
with shared object storage (RWX PVC / S3) as the alternative. This is the seam §5 targets replacing
with a streaming velo data plane.

### 1.6 Multi‑turn cellular — built on `rust-cellular-multiturn` (exact‑fold only)

Conversation‑level partition falls out of `PartitionedSampler` (filters on a per‑conversation draw
counter: cell `k` owns draws `{k, k+C, …}`), so whole conversations are cell‑local and turns never
split. The 3‑commit delta (`ajc/rust..HEAD`) adds a parallel **`sessions` budget slice**
(`owned_positions(sessions,k,C)`, non‑graph only) so each cell single‑passes its owned slice. Admits
synthetic multi‑turn + 9 known multi‑turn file formats **only on the exact‑fold merge** and **only
with sequential/shuffle sampling**. Still fails closed: multi‑turn on the retain path (per‑turn
ordinal ≠ per‑conversation draw index — fundamentally unavailable), random sampling, live‑reply
`inputs.json`, duration/adaptive. No new `CellMessage` variants — rides the existing `StorePartition`.

### 1.7 Graph cellular — built (`ajc/rust`)

`dag_jsonl`/`weka_trace`/`dynamo_trace` bypass the linear gate; `PartitionedGraphTraceSource`
(`graph/workload.rs`) gives cell `k` the interleaved global session ordinals `k,k+C,…`, unique id per
trace, `session_limit` = `--num-conversations`. Empty `phase_ordinal_bases`; concatenation merge.
Rejects any static `requests` budget (would fall back to N× load).

### 1.8 Bounded‑memory metrics (sketch) — built standalone, cellular‑blocked

`wt-sketch-only`: `MetricsStorageMode::Sketch{compression}` streams each record into a per‑`(phase,tag)`
`TagSketch` (t‑digest + exact Welford sum/count/min/max/mean/std) then clears the row — accumulator
memory O(#tags×#phases×~compression/2 centroids), **O(1) in record count**. `RunCapture::finish_fold_into`
folds each finalized record and drops it (retaining only errored records). Percentiles approximate;
counts/sums/rates exact. Per‑record artifacts (records/raw/outputs JSONL, per‑record OTLP, timeslices,
inference series, sweep curves) unavailable — dropped in `rust_wire`, fail‑closed in `validate_plan`.
**Cellular + sketch is now BUILT** (2026-07-15): the cell ship path ships the folded sketch store as
`CellMessage::StorePartition` (the same wire form exact-fold uses) and the controller merges the
per-cell t-digests associatively (`merge_store_partitions` → `ColumnStore::append_store`) — an
O(cells × sketch) cross-cell merge, tier T1 of
`2026-07-15-cellular-horizontal-scale-k6-parity-design.md`. Per-worker fold-and-drop already bounds
per-cell RSS on both the single-thread and sharded paths (`ShardRecords::Folded`), and the record
total travels with the store (`ColumnStore::ingested_count`, since a sketch store retains no rows).

### 1.9 Kubernetes launch — built on `rust-operator` (zero discovery)

The kopf operator renders **one JobSet, two replicatedJobs** (`controller` ×1, `cells` ×N),
`enableDNSHostnames: true`, `successPolicy` on the controller. Discovery is **JobSet headless DNS**:
the controller is deterministically `{jobset}-controller-0-0.{jobset}.{ns}.svc.cluster.local`, injected
as `AIPERF_CELL_CONTROLLER_ADDR=tcp://{dns}:9500`. Cell identity = downward‑API `jobset.sigs.k8s.io/job-index`
→ `AIPERF_CELL_ID`. `K8sLauncher` **spawns nothing** (pods already exist); `CellHandle::wait_failure`
is `pending()` forever, so failure is caught by the controller timeout ladder: register (5 min) →
collect (2 h) → artifact‑upload (5 min), each env‑overridable, each bails loudly and drop‑poisons the
START event. **Zero RWX PVC** — all volumes `emptyDir`; cross‑pod movement is velo (metrics) + HTTP+zstd
(artifacts/dataset) + a results sidecar (port 9091). Known operator bug: `_classify_jobset_failure`
keys on `"workers"` not `"cells"`, so a cell‑job failure falls through unclassified (§8).

### 1.10 The fail‑closed matrix (today)

`validate_cellular_run_shape` + kind validators reject: non‑`http` transport (gRPC/offline/dynosim);
non‑`{synthetic,file,public}` dataset; file/public formats outside the single‑turn allowlist;
multi‑turn on retain; scheduled phases outside `{concurrency,poisson,gamma,constant}` or with
`duration`/`sessions`(retain)/`adaptive_scale`; caps `< cell_count`; graph phases with a static
`requests` budget; mixed store+record partitions; no‑`velo` build with `cells>1`. Allowed‑but‑warned
(aggregate‑equivalent, not byte‑exact): multi‑URL round‑robin, ramps, post‑send cancellation, rate
pacing (`rate/cell_count`), auto‑derived seed.

---

## 2. Velo primitive inventory (source‑grounded) — what exists vs must be built

From velo `main` + `feat/connect-by-endpoint` (`/tmp/velo-main/lib/velo/src/`).

### 2.1 Connect‑by‑endpoint — RESOLVED (the prior spec's open question)

The prior spec left mechanism A/A′/B open. **A′ landed**: the fork branch `feat/connect-by-endpoint`
(one commit, +204/−2) adds `Velo::connect(Endpoint) -> PeerInfo` (`lib.rs:526`), `Endpoint::{Tcp,Uds}`
+ `to_worker_address()`, and `ActiveMessageClient::connect(WorkerAddress)` which registers a
provisional peer, does the `_hello` handshake (response now carries `peer_info: Option<PeerInfo>`),
and re‑registers under the responder's real id. AIPerf's `connect.rs::connect_controller` wraps this
with a 60 s retry loop. `WorkerAddressBuilder` stays `pub(crate)` — you reach a peer by address only
through `Endpoint`. **The bootstrap/`serve_bootstrap` mechanism (still in `rust-operator`) is
superseded by this on `velo-connect`; the ultimate k8s launch is connect‑by‑endpoint + the operator's
JobSet/DNS/env wiring.**

### 2.2 Streaming anchors — SPSC + MPSC only, both consumer‑owned, no replay

- **SPSC `StreamAnchor<T>`**: consumer‑owned, exactly one producer at a time (TOCTOU `DashMap::entry`),
  either‑side finalize, live `flume::bounded(256)`, **no replay buffer**.
- **MPSC `MpscStreamAnchor`**: consumer‑owned, many producers each tagged `SenderId`, per‑sender
  detach non‑terminal, **consumer‑owned finalize**, **no replay buffer**.
- **The verdict (definitive):** there is **NO producer‑owned fan‑out (SPMC/MPMC, 1→N with
  per‑consumer replay‑on‑attach)**. `AnchorKind{Spsc,Mpsc}` is a 1‑bit wire discriminator with no
  third kind; a tree‑wide grep for `replay|broadcast|spmc|mpmc` returns zero streaming hits. **It must
  be built** (§3, §6).

### 2.3 Rendezvous — chunked‑AM on `main`; real NIXL/RDMA on an unmerged branch; still put‑once/pull‑by‑handle

`register_data(Bytes)->DataHandle` / `get->(Bytes,lease)`; transparent large‑payload staging above
256 KiB. **On `main`** transfer is **512 KiB chunks over AM**; `StageMode::Pinned` is a placeholder
and consumer RDMA paths `bail!("RDMA rendezvous not yet implemented (Phase 2)")`
(`main:rendezvous/{store.rs:17-26, consumer.rs:94-95,153-154}`). **But the unmerged
`feat/rendezvous-nixl` branch implements it for real**: `rendezvous/nixl_endpoint.rs` (348 lines,
`#[cfg(feature="nixl")]`) is a genuine NIXL+UCX one‑sided RDMA READ path — `NixlEndpoint::create`
builds a UCX agent, pre‑registers a 64 MiB host arena + lazy per‑device VRAM arenas, and
`nixl_read` does `create_xfer_req(XferOp::Read,…)`→`post_xfer_req`→`wait_xfer` into an arena dest
(real `cudaMemcpy` H2D binding), handshaking `NixlAddrDescriptor` over a `_rv_nixl_handshake`
typed‑unary. **Correction to the earlier draft:** RDMA rendezvous is *built* (branch‑gated), not
merely aspirational. **However** it keeps the **put‑once / pull‑by‑handle, single‑owner‑slot** model
— it upgrades the *transfer mechanism*, it does **not** add the producer→N‑consumer DataAnchor/Session
broadcast shape (§2.2 still holds). For AIPerf: the data plane can eventually pull shard bytes over
this once merged/feature‑enabled; the **fan‑out of handles still must be built** on top. Same nuance
on the kvbm side — `ryan/kvbm-rdma-pull` has a real `rdma_pull_with_opts`/`execute_transfer_selection`
path, but it is **KV‑cache‑block‑specific** (physical layout + TP descriptors), not a drop‑in
byte‑shard pull.

### 2.4 Distributed events — single‑shot, real Future, completed‑cache; no countdown

See §1.4 correction. `EventHandle(u128)=[system:64][index:32][generation:32]`; `EventAwaiter: Future`;
network‑routed via `_event_subscribe`/`_event_trigger`; completed cache local + LRU‑1000. **No
register‑count barrier; no monotonic multi‑generation "await ≥ N" event.** The phaser (§4) builds
this.

### 2.5 Transports & addressing

**On `main`** TCP/UDS(+`cfg(unix)`)/NATS/gRPC/ZMQ exist; QUIC/NIXL/TIPC/HTTP absent. **Correction:**
QUIC (`claude/quic-http3-transport`, quinn 0.11), TIPC (`feat/tipc-transport`, real `AF_TIPC`
sockets), and NIXL (`feat/rendezvous-nixl`) exist as **real, feature‑gated, unmerged** implementations
on dedicated branches; **only HTTP is universally absent**. `WorkerAddress` is **not** a `scheme://`
URL — it is an opaque MessagePack `HashMap<transport-key → endpoint-bytes>`; routing is map‑key
self‑selection (each transport claims its own entry, first compatible wins). `tcp://`/`uds://` are
prefixes *inside* the value. (The `Endpoint` builder (§2.1) is where new direct‑connect transport
variants — grpc/quic/nixl — would be added when those transports merge, per Ryan's note.)

---

## Implementation status — 2026-07-15 (forward-plane primitives BUILT + tested)

The forward plane's primitives — the ones velo lacks and the spec identifies as the hard
dependencies — are **built AIPerf-side and unit-tested** (per §3.1's "an AIPerf-side layer over
velo primitives" option, so no velo-fork round-trip was needed):

| Piece | Module | Status |
|---|---|---|
| **SPMC broadcast w/ replay-on-attach** (§3.1, §6.1 — the "one hard dependency") | `cellular::broadcast` | ✅ built + 4 tests (full-order reconstruction for every attach time incl. post-finalize; no gap/dup at the seam across 50 interleaved attach/add; add-after-finalize rejected; slow consumer doesn't stall the producer) |
| **Monotonic phaser** (§4) | `cellular::phaser` | ✅ built + 5 tests (monotonicity, replay for passed targets, block-then-wake on live advance, cyclic gate-on-≥ with no ABA, await-after-finalize) |
| **Phaser velo distribution** (§4) | `cellular::transport::phaser_velo` | ✅ built + 1 test (a cell subscribing over two in-process velo instances reaches pre-subscribe generations via replay, then observes live advances) |
| **Phaser-driven START, integrated + executed** (§4) | `cellular_controller` + `cellular_cell` (`AIPERF_CELL_PHASER_START`) | ✅ **the control plane drives a real run**: the controller binds a `PhaserServer` + `advance(Started)`; cells subscribe + await generation 1. e2e `test_cellular_phaser_start_matches_event_start` — phaser-START reproduces event-START's deterministic metrics EXACTLY (byte-identical request_count/ISL/OSL). Default off; the event START is byte-unchanged. |
| **Dataset fan-out data plane** (§3) | `cellular::dataset_session` + `transport::dataset_velo` + runner (`AIPERF_CELL_DATASET_FANOUT`) | ✅ **built + tested + EXECUTED end-to-end**: 4 unit tests (owned-filter tiling, late-attach replay, arrival-order independence, disjoint owned indexes over velo) + e2e `test_cellular_dataset_fanout_matches_baseline` — the controller broadcasts the dataset request-ids (phaser `ShardsAvailable` per chunk), every cell builds its owned shard over velo and dispatches it exactly-once; fan-out reproduces the baseline deterministic metrics exactly |
| **Per-request dispatch state machine + DistributionMiss** (§4.5) | `cellular::dispatch_state` + runner | ✅ **built + tested + EXECUTED**: 3 unit tests (issue-once-then-dedup, unknown-is-a-counted-miss, in-flight accounting) + run over the fan-out index in the e2e above — every 4-cell run dispatched its owned slice `issued=owned completed=owned misses=0`, fail-closed on any miss |

Already built earlier: the bounded-memory collection (§5) — sketch `StorePartition` cellular ship
(tier T1) and per-worker fold-and-drop (`ShardRecords::Folded`). **The entire forward plane is now
built + tested + executed end-to-end** in real multi-process runs: the broadcast primitive (§3.1),
the phaser (§4) driving START, the dataset fan-out (§3) with the phaser availability interlock, and
the per-request dispatch state machine (§4.5). The **remaining follow-ons** are optimizations, not
gaps: (a) a `ControlledIssuer` *workload* that dispatches the actual request bodies from the owned
index into the runner's `TurnLifecycleObserver::on_issue` + `SlotPool` seams (today the fan-out
delivers + owned-filters + runs the dispatch state machine over each cell's shard, proving the path;
materializing the request body from the index instead of the sampler is the last mile), and (b) the
§6 velo-native additions (a first-class SPMC anchor, RDMA rendezvous) — velo-owned, and the
AIPerf-side layer above makes them a bandwidth optimization, not a blocker.

---

## 3. Forward design — the dataset **data plane** (SPMC add‑only broadcast)

**Goal:** the controller generates a large dataset once and streams it hot‑in‑RAM to all cells (or each
cell's shard), replacing "every cell regenerates from a shared seed" and "ship files over HTTP." This
is the case Ryan (velo) mapped to the kvbm p2p `Session`.

### 3.1 The primitive: a producer‑owned broadcast anchor with replay‑on‑attach

Modeled on the kvbm `ReplayStream` (whose `subscribe()` does snapshot‑and‑go‑live atomically under one
lock). AIPerf needs the **N‑connector** generalization the kvbm session explicitly is *not*
(one‑peer‑per‑session). The abstraction (a new velo anchor kind, or an AIPerf‑side layer over MPSC
control + rendezvous data):

- Owns `{ history: Vec<Chunk> (append‑only, producer call order, incl. terminal Finalized), senders:
  Vec<Sender> }` under **one lock**.
- `attach(consumer)`: lock → clone `history` into this consumer's replay buffer → push a fresh sender →
  unlock (N‑fold `ReplayStream::subscribe`).
- `add(chunk)`: lock → `history.push` → fan‑out to every sender → unlock.
- **The single invariant that answers Ryan's "more racy state":** membership (attach/detach) serializes
  on the *same* lock as delta‑append. Then each consumer's `(replay snapshot ⊎ live tail)` reconstructs
  the full producer order with no gap/dup at *its* attach seam — **by construction, no per‑consumer diff
  computed.** (Miss this and a chunk added between snapshot‑clone and sender‑registration is lost or
  double‑counted.)

### 3.2 Collapsed for AIPerf: add‑only + finalize (drop the commit phase)

kvbm splits `committed` (hashes promised) from `available` (blocks pullable) because conditional‑disagg
knows the prefix before the bytes land. **AIPerf has no such gap** — a chunk's bytes exist at `add`
time. So the surface is just `add(chunk)` / `finalize()`; committed ≡ available. One monotonic add‑only
stream, terminal finalize.

### 3.3 Decouple pull/index from issue

Two planes at two cadences (Ryan's two sentences):
- **Data plane** (`add` = "pull this next chunk"): bulk, async, **order‑insensitive**. Cells pull and
  build a **local index keyed by stable `request_id`** — never by arrival position (the kvbm
  "arrival‑ordered ≠ position‑ordered" gotcha; keying by id sidesteps it).
- **Control plane** (§4 phaser: "issue request R now"): per‑request, timed, dispatch‑driven.

### 3.4 Uniform broadcast vs routed fan‑out (the memory fork)

Cellular exists **for memory scaling**, so:
- **Uniform broadcast** (plain SPMC): every cell pulls every chunk — simplest, but every cell holds the
  full dataset (defeats the point at large N).
- **Routed fan‑out**: each cell holds only its ~1/N shard (its owned round‑robin positions).
- **v1 recommendation**: uniform fan‑out + **consumer‑side owned‑filter** (each cell sees every frame
  header but only *indexes* the requests it owns → RAM is 1/N even if wire bandwidth isn't). Optimize to
  server‑side routing (frame carries `target cell_id`) later. Flag to velo: our fan‑out is ultimately
  *sharded*, so a routing key on the broadcast frame from the start is worth considering.

### 3.5 Transport reality

Until velo RDMA lands (§2.3), the chunk bytes move over chunked‑AM rendezvous or the existing HTTP+zstd
Stage‑G plane; the broadcast anchor carries **handles/notifications**, not megabytes. Keep the control
notification (small) and the bulk pull (large) on separate transports — never push the dataset through
the phaser.

---

## 4. Forward design — the **control plane** (monotonic phaser)

**Goal:** generalize the one‑shot synchronized START (§1.4) into a **monotonic generation counter the
controller increments as the benchmark progresses**, that all cells observe with replay‑on‑attach.

### 4.1 What it is

An SPMC monotonic stream whose payload is `{generation, transition}` where transition ∈ `{Started,
ShardsAvailable(k), PhaseAdvance(Warmup|Profiling|Drain), Done}`. Monotonic + replay‑on‑attach ⇒ a
late‑joining cell reads "current generation = G" atomically, then live‑follows — **no missed
transition**. It subsumes three things currently separate:
1. **START** — generation 1 = "go" (replaces the `AtomicU32`+`Notify`+single‑shot‑event scaffold).
2. **Dataset progression** — increment per shard batch; `ShardsAvailable(k)` ⇒ shards `[0,k)` pullable
   (drives §3's `add`/`finalize` from the cell's perspective).
3. **Phase transitions** — warmup→profiling→drain as generation steps.

### 4.2 One‑way vs barrier (the decision)

- Pure progress broadcast (cells react, controller never waits) → SPMC is enough.
- True barrier (controller waits until *all* cells reach G before G+1) → needs the **fan‑in arrival
  half** too. START and end‑of‑phase drain *are* barriers; dataset progression is one‑way. So the full
  lifecycle wants both directions → either build **MPMC**, or **compose** SPMC (fan‑out) + a counted
  arrival mechanism (AIPerf‑side counter over MPSC, as START already does) for the barrier points.
- The **end‑of‑warmup barrier** is the precedent‑backed one: the mocker **drain barrier**
  (`_REQ_KIND_DRAIN_BARRIER=0xFC`, a counted one‑ack‑per‑event gate that blocks sim‑time advancement) is
  exactly the "every cell finishes warmup before any begins profiling" primitive, which is what unblocks
  **cross‑cell adaptive‑scale consensus** (today fail‑closed).

### 4.3 Cyclic from the start

Never reset the counter. A run that loops (ramp steps, multi‑round sweeps) keeps incrementing; cells
gate on `generation >= threshold_for_this_round`, never equality → no ABA, cyclic for free. (This is
the java `Phaser`'s monotonic‑generation discipline; the completed‑event cache §2.4 gives late/repeat
awaiters the correct immediate resolution.)

### 4.4 Where it hooks into dispatch (built seams)

`rust-dispatcher` already has the landing points:
- A new **`ControlledIssuer` `Workload` impl** (`request_rate.rs` `Workload::execute(runtime)`) that
  awaits "issue id R" off a channel (the synchronized‑START velo event is the precedent channel) and
  calls the existing `ScheduledRuntime::issue_turn_with_hooks_and_cancellation`.
- **`TurnLifecycleObserver::on_issue`** — the synchronous "this specific turn is admitted, before
  backend dispatch" seam where the phaser‑commanded issue fires.
- **`IssuanceAuthority::global_ordinal`** — binds the controller‑assigned global id to the dispatched
  turn.
- **`SlotPool`** admission (`try_acquire`/`acquire`) stays the backpressure gate whether arrivals are
  paced or externally commanded.

### 4.5 The readiness interlock — "issue R before R arrived"

Per‑request **state machine** on the cell: `Unknown → Indexed → InFlight → Done`. The "issue R" handler
dispatches on state:
- `Indexed` → issue, → `InFlight`.
- `InFlight`/`Done` → **no‑op** (dedup; exactly‑once‑issue per id — handles Ryan's "or in flight").
- `Unknown` → the race, handled in two layers (mirrors kvbm's bilateral seal enforcement):
  - **Source side (make it rare/impossible):** *bounded runs* → §3's finalize→all‑drained **barrier
    before START**, so every owned request is `Indexed` before any issue (Unknown impossible).
    *Streaming runs* → keep the control plane **causally downstream of the data plane**: the controller
    doesn't publish the generation authorizing window `k` until `ShardsAvailable(k)`.
  - **Sink side (guard for the residual — reorder, cell reattach with cold index, controller bug):**
    demand‑pull R with a **bounded deadline**; if it lands, issue and **tag the record with the incurred
    wait** (measured, not hidden); if not, classify a **`DistributionMiss`** — a distinct, counted,
    surfaced error class, never a silent skip (the "no silent caps" rule).

### 4.6 Two regimes (summary)

- **Bounded (request‑count)**: distribute all → barrier → dispatch. Race‑free; the common AIPerf case.
- **Continuous (rate / duration / streaming)**: dataset streams concurrently; the phaser generation is
  the backpressure interlock gating dispatch behind availability.

---

## 5. Forward design — bounded‑memory record collection (no RWX PVC)

### 5.1 The target

Today records collect exactly (byte‑parity) but the online path's **peak RSS is set upstream by
per‑worker observers retaining every record until end‑of‑run drain** — the ceiling at 1M concurrency.
Collection must become streaming and bounded while preserving the current merge fidelity where it
matters.

### 5.2 Ship the merged sketch as a cell partition (unblock §1.8)

The t‑digest is **already mergeable and associative** (`sketch.rs`), and `finish_fold_into` already
folds‑and‑drops. The missing seam: a cell running sketch mode ships its per‑`(phase,tag)`
`ColumnStorePartition`‑of‑sketches as `CellMessage::StorePartition`; the controller `merge_store_partitions`
already appends stores. Removing the `ensure!(!sketch_mode)` cell guard + wiring the sketch store into
the existing StorePartition ship is the whole change. Result: **cellular high‑request‑rate memory is
O(#cells × #tags × #phases × centroids)** — bounded, no per‑record retention, no shared FS.

### 5.3 Streaming per‑worker finalization (the deeper fix)

Bound the *worker observer* peak, not just the accumulator: per‑record worker finalization keyed by
phase (streaming‑finalize follow‑up noted in CLAUDE.md), so a worker folds and drops each record at
completion instead of retaining until drain. This is orthogonal to cellular but compounds with it.

### 5.4 Records over a velo data plane (replace the HTTP+zstd artifact plane)

The inverse of §3: **flow 2 (record collection) is N independent 1:1 sessions** (each cell → controller),
*not* multi‑connector — so it's the kvbm session verbatim, no fan‑out machinery. When velo gains a
streaming data plane with real backpressure (and eventually RDMA), the per‑record file artifacts can
ship over it instead of the bespoke HTTP+zstd server, collapsing two transports into one. Until then the
HTTP+zstd plane (§1.5) stays the cross‑host mechanism and shared object storage the alternative.

---

## 6. What velo must add (upstream asks, prioritized)

1. **Producer‑owned broadcast anchor (SPMC), replay‑on‑attach, producer‑owned finalize** (§2.2, §3.1).
   Smallest form: a third `AnchorKind` + a producer‑side registry entry `{append‑only replay log,
   DashMap<ConsumerId, Sender>}` + `attach_broadcast` (replay‑then‑live) + producer finalize. Reuses the
   existing `AnchorManager` + `flume` + `FrameTransport` + `_anchor_attach`/`_stream_cancel` control
   plane. This is the one hard dependency for §3/§4.
2. **Monotonic / countdown event** (§2.4): either a "fire after N arrivals" countdown event, or accept
   that AIPerf builds the count on MPSC + a counter (as START does today). A native monotonic
   multi‑generation "await ≥ N" event would make the phaser a thin wrapper.
3. **Merge/enable NIXL rendezvous** (§2.3): the transfer path is **already built** on
   `feat/rendezvous-nixl` (not build‑from‑scratch) — merging + a `nixl` feature enable gives zero‑copy
   shard *bytes*; until then chunked‑AM or HTTP+zstd. Not on the critical path (the handle **fan‑out**,
   ask #1, is).
4. **Optional routing key on broadcast frames** (§3.4) for server‑side sharded fan‑out.

The connect‑by‑endpoint ask (§2.1) is already delivered on the fork; upstreaming it to `ai-dynamo/main`
is the remaining PR. **Reusable to port back into velo** (Ryan's "more general sessions" framing): the
kvbm `ReplayStream` snapshot‑then‑live pattern (`velo.rs:55-100`, a tokio‑mpsc session‑local fan — the
per‑consumer half of the SPMC anchor), the `PeerCommitted`/`PeerAvailable` `Open→Sealed` seal state
machine, and the `SessionManager` watchdog + Weak‑gauge leak detector.

---

## 7. Roadmap — fail‑closed → fail‑open, in order

| Step | Unlocks | Depends on |
|---|---|---|
| Sketch StorePartition ship (§5.2) | bounded‑memory cellular metrics at 1M rate | built pieces only |
| SPMC broadcast anchor in velo (§6.1) | dataset fan‑out, phaser | velo change |
| Phaser control plane (§4) | synchronized every‑phase progression, cyclic runs | §6.1 |
| Dataset fan‑out (§3) | ship hot dataset cross‑node; drop "regenerate from seed" | §6.1, §4 |
| ControlledIssuer + state machine (§4.4/§4.5) | controller‑driven per‑request dispatch, `DistributionMiss` | §4 |
| End‑of‑warmup barrier (§4.2, drain‑barrier pattern) | **cross‑cell adaptive‑scale consensus** (lifts a fail‑closed) | phaser fan‑in |
| Streaming per‑worker finalize (§5.3) | multi‑turn retain, duration‑bounded cellular (ragged counts) | independent |
| Records over velo data plane (§5.4) | one transport, drop HTTP+zstd server | velo streaming data plane |
| Side‑channel telemetry aggregation | server_metrics/GPU/network in the merged report (lifts a fidelity gap) | a per‑cell sidecar ship |
| gRPC/offline cellular | non‑HTTP cellular (lifts a fail‑closed) | cell issuer/ship on those executors |

---

## 8. Known defects & fidelity gaps to carry forward

- **Operator `"workers"` vs `"cells"`** (`monitor.py::_classify_jobset_failure`): a cell‑job failure is
  not matched as the restartable case → run fails unclassified. Fix classification to the `"cells"`
  replicatedJob name.
- **Merged‑report fidelity gaps** (deliberate today): coordinator finalize provenance, grouped
  per‑error detail (counts survive), and side‑channel sidecars (server_metrics/GPU/network) are omitted
  — record‑derived distributions stay byte‑identical.
- **Aggregate‑equivalent (not byte‑exact), warned**: rate pacing, post‑send cancellation subset,
  multi‑URL round‑robin, ramps. Byte‑parity holds only for a seeded pure `concurrency` phase.
- **`was_cancelled` left false** (no cross‑cell cancellation consensus).
- **Docstring drift** (`jobset_helpers.py`: "velo replaces … with velo discovery") — the shipped design
  is a hardcoded coordinate + connect‑by‑endpoint, deliberately no etcd/NATS/velo‑discovery.

---

## 9. Testing (target)

- **Unit**: SPMC broadcast replay‑on‑attach under N concurrent attach/add (the racy‑state invariant §3.1);
  phaser generation monotonicity + cyclic gating; per‑request state machine (dedup on InFlight/Done,
  bounded‑await + `DistributionMiss` on Unknown); sketch StorePartition merge parity.
- **E2e** (extend the existing `test_cellular*.rs` / `test_graph_cellular.rs` real‑frontend suite):
  dataset fan‑out reproduces the seed‑regenerated dataset SET; phaser‑driven synchronized phase
  transitions; a streaming run where a late cell reattach exercises the `Unknown`→demand‑pull path and
  asserts a counted `DistributionMiss` rather than a silent drop; sketch cellular memory bound.
- **Regression**: every §1 byte‑parity assertion stays green (the forward work must not perturb the
  built exact‑merge path for the bounded `concurrency` case).

---

## 10. Docs to update (same change, when built)

- Flip the relevant "Canonical vs aspirational" flags in the four agent files as each §7 step lands
  (SPMC anchor, phaser, dataset fan‑out, sketch cellular, per‑worker finalize), plus the crate‑table
  cellular note and the "Build, test, run" `--cells` paragraph; run `python tools/check_agent_files_sync.py`.
- `specs/README.md` status row for this spec + `llms.txt`.
- Append a dated `## Addendum` to `2026-07-12-cellular-ready-seams-and-roadmap.md` and
  `2026-07-15-velo-cell-transport-design.md` pointing here (done in this change).
- `python tools/check_docs_current.py` before commit.

---

## 11. Cross‑reference against ALL velo + kvbm branches (verification of §2–§6)

Every load‑bearing claim was verified against the actual source across the velo (`ai-dynamo` +
`ajcasagrande` fork, 27+ branches) and dynamo/kvbm (200+ branches) repos, not just `main`. Verdicts:

| Claim | Verdict | Evidence |
|---|---|---|
| **SPMC/MPMC producer fan‑out must be built** (§2.2, §6.1) | **CONFIRMED** | `AnchorKind{Spsc,Mpsc}` on **all 11** velo branches checked (`handle.rs:26-33`); tree‑wide grep `spmc\|mpmc\|broadcast\|fan-out\|replay\|ReplayStream\|BroadcastAnchor` = **zero hits**; no `tokio::broadcast`/`watch` in streaming. No `ryan/velo-streaming*` branch exists. |
| **kvbm Session is one‑peer / SPSC / ReplayStream‑replay** (§3.1‑3.2) | **CONFIRMED** | `ryan/kvbm-engine-service` CONTRACT §1 ("one peer per session") + `velo.rs` uses `create_anchor::<Frame>()` (SPSC), `ReplayStream` subscribe‑once under one `Mutex` (`velo.rs:57-101`) over a **tokio `mpsc::UnboundedSender`** (session‑local, not a velo primitive). velo gives SPSC/MPSC, no replay/broadcast — session layers replay above. |
| **Add‑only collapse (commit ≡ available)** (§3.2) | **CONFIRMED** valid | `make_available` only requires the hash be in the committed set (CONTRACT §2.5); a holder that already has its shards calls `commit`→`make_available` back‑to‑back. |
| **RDMA/NIXL rendezvous unbuilt** (§2.3) | **REFINED** | Unbuilt on `main` (chunked‑AM, `bail!` placeholders), but **real** on unmerged `feat/rendezvous-nixl` (`nixl_endpoint.rs`, UCX one‑sided READ). Still put‑once/pull‑by‑handle, **not** producer→N fan‑out — §2.2 unaffected. `ryan/kvbm-rdma-pull` real but KV‑block‑specific. |
| **QUIC/NIXL/TIPC/HTTP transports absent** (§2.5) | **REFINED** | Absent from `main`; QUIC/TIPC/NIXL real on unmerged branches; only HTTP universally absent. |
| **Monotonic phaser must be built** (§4) | **CONFIRMED** | No existing dynamo/velo primitive is a controller‑incremented monotonic generation broadcast with snapshot‑on‑join. Nearest: `velo-events` (`idhanani/session-control-event-plane`) — but **single‑shot per generation** (generation = slot‑recycle counter, not a phase counter), `merge_events` is a fan‑**in** AND‑join not a fan‑out, `Arc`/`dashmap`/`tokio‑multithread` (**incompatible** with AIPerf's `!Send` thread‑per‑core `Rc`/`RefCell` hot path), and its **distributed backend is a `todo!()` stub**. Cite as prior art (generational events + poison‑propagation + merge barrier), do not depend. |
| **Per‑request "issue R" dispatch state machine must be built** (§4.5) | **CONFIRMED** | Nearest surface (`AgentController` open/close session RPC, `idhanani/session-control-event-plane`) is KV‑cache **affinity/pinning**, not dispatch — no overlap. |
| **Discovery deliberately forgone** (§1.9, §2.1) | **CONFIRMED** as the alternative | `rupei/runtime-velo-peer-discovery` (`discovery/velo_peers.rs`, kv‑store/kube) and `hannahz/…velo-indexer…filesystem-peer-discovery` are the etcd/kv/filesystem discovery backends AIPerf's connect‑by‑endpoint intentionally avoids. |

**Net:** the spec's two central conclusions — *build a producer‑owned SPMC broadcast anchor with
per‑consumer replay‑on‑attach*, and *build a monotonic phaser* — are **confirmed against every
branch**; nothing in velo or kvbm provides either. The only corrections are that NIXL rendezvous and
QUIC/TIPC transports are *built‑but‑unmerged* (branch‑gated), not nonexistent — which shortens the
"velo must add" list for the *transfer* layer while leaving the *fan‑out* and *phaser* layers
genuinely greenfield. Reusable to port back: the kvbm `ReplayStream` pattern and the seal state
machine (§6).

---

## Addendum — 2026-07-16 — barrier-synchronized timing origin (§4, built, opt-in)

**What changed / supersedes:** §4.2 noted START is a barrier but left the *timing origin* each
cell adopts unspecified; §1.4 and §1.1 describe the START barrier as *releasing* cells only. In
practice each cell then captured its run origin (`start_ns = clock.now_ns()`) inside
`execute` **after** the barrier AND **after** its own per-cell dataset download + run setup
(tokenizer load, dataset compile, connect). Cells with a larger shard / slower setup therefore
zeroed their record timeline at a *later* instant than their peers, so the merged report's
cross-cell absolute timestamps referenced a different `t0` per cell (all latency/throughput metrics
are *differences* and so were unaffected — only absolute per-record timestamps drifted).

**Built (opt-in behind `AIPERF_CELL_SHARED_ORIGIN`, default off):** the cell now captures a
`RealClockAnchor` the instant its velo START barrier releases — inside `fetch_cell_envelope`
(`runner_protocol::cellular_cell`), the shared logical instant every cell reaches together, *before*
its per-cell setup — via `runner_protocol::cell_origin::capture_cell_shared_origin`. At run start,
`execute` derives its origin from `cell_origin::run_origin_now_ns(&clock)` instead of
`clock.now_ns()`: when a barrier anchor was captured it returns `clock.now_ns() - barrier.now_ns()`
(read at one instant so the shared wall-`now` cancels), i.e. the barrier's reading on the execute
clock's own timeline — negative when the barrier preceded the execute anchor (the common case),
which shifts every record's timestamp forward so it is measured from the barrier. Default off ⇒
`run_origin_now_ns` returns `clock.now_ns()` unchanged, so single-process and existing cellular runs
are byte-unchanged.

**Cross-host semantics (deliberate).** Cells may run on different hosts with unsynchronized wall
clocks, so this does **not** adopt an absolute controller `t0` (which would import clock skew). Each
cell zeroes at its *own* clock reading of the barrier-release instant; the barrier guarantees those
instants coincide within network latency, so "elapsed since START" stays coherent cross-host with no
clock-sync assumption. (The alternative — controller broadcasts an absolute `t0` in the
`PhaseTransition::Started` / START payload — was rejected for that skew.)

**Proven (2026-07-16):** unit `cell_origin::tests` (origin math + no-barrier pass-through) and e2e
`test_cellular_shared_origin_zeroes_at_the_barrier` (`rust/e2e/tests/test_cellular.rs`): a `--cells
N` run with the flag ON reproduces a single-cell baseline's ISL/OSL exactly (no regression) **and**
its first request's `request_start_ns` exceeds the flag-off run's by the whole per-cell setup span
(the deterministic signature of the origin having been pulled back to the shared barrier). Manual
raw-record run: flag-off min `request_start_ns` ≈ 0.34s (each cell's post-setup start) vs flag-on ≈
9.41s (measured from the barrier, dominated by the shared large-tokenizer load), ISL/OSL
byte-identical to the single-cell baseline.

**Not yet:** default-on (the flag is opt-in until it bakes); the controller's own report provenance
still records its local finalize time, not the shared origin (the merged report's record timestamps
are the cells', which is what the origin governs).
