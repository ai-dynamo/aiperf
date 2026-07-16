<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Velo as the core transport for AIPerf cell mode

**Status:** Design / awaiting approval
**Date:** 2026-07-15
**Author:** (brainstorm session)
**Repo:** `aiperf` (worktree `velo-cell-transport`, branched from `ajc/rust` HEAD `c019db53e`)
**Relationship:** Implements the "Phase 2 cross-node transport" swap left open by
`rust/specs/2026-07-12-cellular-ready-seams-and-roadmap.md` (see its addenda: "the seam is
transport-neutral, so a cross-host impl is a `CellClient`/`ControllerTransport` swap, not a
rewrite"). This spec is that swap plus the launch/discovery rework it enables.

---

## 1. Goal

Replace the loopback-only `TcpCellClient` / `TcpControllerTransport` cell↔controller transport
with a **velo-backed** transport, and rework cell launch + registration so cells can be
Kubernetes pods reached by **operator-hardcoded DNS with zero discovery infrastructure**
(mirroring the existing Python k8s model). Every measurement seam the cellular runtime already
ships — static `owned_positions` partition, `IssuanceAuthority`, `RecordsShard`,
`MetricsHeartbeat`/t-digest, global-ordinal / concatenation merge — stays **byte-for-byte
unchanged**. Only the *wire*, the *launch mechanism*, and the *registration/barrier* change.

### Decisions locked in brainstorming

1. **Replace TCP entirely** with velo (no dual TCP+velo seam; velo is the only cell transport).
2. **Static partition retained** (not the Python dynamic credit router) — smallest delta,
   preserves byte-parity, dynamic routing can come later behind the same seam.
3. **Zero discovery.** No etcd / NATS / velo-discovery backend. Cells reach the controller from
   an operator-hardcoded DNS:port only, exactly like the Python `AIPERF_K8S_ZMQ_CONTROLLER_HOST`
   model.
4. **Full distributed control plane, launch = k8s.** Remote cell launch is Kubernetes (operator
   + JobSet); velo owns discovery-less registration, management, and transport. This spec does
   **not** build the operator/JobSet/CRD (the Python operator already models it) — it builds the
   Rust transport + registration + launch seam that a k8s deployment consumes.
5. **Local subprocess launcher survives, over velo** — `--cells N` on one host and the existing
   e2e tests keep spawning subprocesses, but they speak velo (UDS on Unix, TCP-loopback on
   Windows), so the same velo code runs locally and in-cluster.
6. **Feature-gated** behind a `velo` cargo feature. **`ai-dynamo/main` lineage only — never the
   legacy fork.** A small upstream patch is acceptable *only* as a PR against `ai-dynamo/main`
   (see §4.2): if the connection mechanism needs one, author it upstream and temporarily pin Cargo
   to the PR `rev` until it merges. The zero-patch bootstrap mechanism (§4.2 mechanism B) is the
   guaranteed fallback that needs no velo change at all.

### Non-goals

- The k8s operator / JobSet / CRD / Helm chart (follow-on; Python operator is the reference).
- Dynamic credit routing (least-loaded / session-sticky pull).
- Cross-cell side-channel telemetry aggregation (`server_metrics` / `gpu_telemetry` /
  `network_latency`) — remains the documented report-fidelity gap.
- Any change to the S1/S2/S3/S4 measurement seams or the merged-report byte-parity contract.

---

## 2. Background — what exists today

`rust/runtime/src/cellular/` is the always-on cellular module. The relevant pieces:

- **`transport.rs`** — the seam being replaced. Two traits: `CellClient` (cell side, blocking
  send) and `ControllerTransport` (controller side, merged inbound stream), carrying a
  `CellMessage` enum (`Heartbeat { cell_id, Box<MetricsHeartbeat> }`, `Partition(RecordsShardPartition)`)
  as length-prefixed `rmp-serde` frames. Concrete impls `TcpCellClient` (blocking
  `std::net::TcpStream`) / `TcpControllerTransport` (Tokio listener accepting `expected_cells`
  connections, merging into one channel). **This file gains a velo impl; the TCP impls are
  removed.**
- **`runtime/src/runner_protocol/cellular_controller.rs`** — `run_cellular(envelope, cell_count, report_path)`:
  binds the transport, spawns `aiperf --cell` subprocesses (`spawn_cell`, stdin-piped
  `CellLaunchSpec`), watches child exits, collects one partition + heartbeats per cell, merges
  in global-ordinal (scheduled) or concatenation (graph) order, writes `native-v2.json`, runs
  the export plane, writes the heartbeat sidecar. **The spawn + stdin-pipe + accept-N-connections
  parts are reworked; the slice/merge/report parts are unchanged.**
- **`runtime/src/runner_protocol/cellular_cell.rs`** — `CellLaunchSpec` (piped to a cell's stdin), env vars
  (`AIPERF_CELL_CONTROLLER_ADDR`, `AIPERF_CELL_PHASE_ORDINAL_BASES`), `CellRecordsShipper`
  (connects, ships heartbeat then partition). **The stdin path is replaced by a velo `register`
  fetch; the shipper sends over velo.**
- **`cli/src/main.rs`** — dispatches `--cell` (stdin `CellLaunchSpec` → env → single-process
  execute) vs controller (`execute` + `cells>1` → `run_controller` → `run_cellular`). **Cell
  bootstrap changes from "stdin spec" to "env coordinate + velo register".**

### The Python k8s model this mirrors (reference only, not modified)

`../new-config-kube`: a kopf **operator** reacts to an **AIPerfJob** CR → writes a ConfigMap
(`run_config.json`) + a **JobSet** (`enableDNSHostnames`) with 1 controller pod + N worker pods.
Discovery is **JobSet headless-service DNS**: the controller's DNS is deterministic
(`{jobset}-controller-0-0.{jobset}.{ns}.svc.cluster.local`) and injected into every worker pod as
`AIPERF_K8S_ZMQ_CONTROLLER_HOST`. Identity is the downward-API `AIPERF_POD_INDEX` (JobSet
job-index label). Transport is ZMQ dual-bind; the controller **binds**, workers **connect**; the
readiness **barrier** waits for a *count* of registered services == num worker pods. Our velo
design reproduces exactly these facts (self-index + controller DNS via env; count barrier) with
velo as the wire.

---

## 3. Velo facts this design relies on (official `ai-dynamo/main`, tag `v0.5.0`, commit `c53fea2`)

Verified against `/tmp/velo-main` (a checkout of `ai-dynamo/main`). Zero velo change is the
baseline (mechanism B); any change is a PR against `ai-dynamo/main`, never a fork.

- **One crate.** `velo = { git = "https://github.com/ai-dynamo/velo.git", tag = "v0.5.0",
  default-features = false }` yields TCP + UDS transports (never feature-gated), filesystem
  discovery (unused here), and all four active-message patterns. `velo-ext` (identity/address/
  trait surface) is pulled transitively.
- **Build + messaging API** (`lib/velo/src/lib.rs`, `messenger/`):
  `Velo::builder().add_transport(Arc<dyn Transport>).build().await -> Arc<Velo>`;
  `Velo::register_handler(Handler)`; `Handler::typed_unary(_async)("name", |ctx: TypedContext<I>| -> Result<O>)`,
  `am_handler(_async)`; send builders `typed_unary::<R>("name")?.payload(&req)?.instance(id).send().await`,
  `am_send("name")?.payload(&e)?.instance(id).send().await`. Typed payloads are **JSON**
  (`serde_json`) on the wire.
- **Addressing.** Peers targeted by `InstanceId`. `PeerInfo::new(InstanceId, WorkerAddress)` is a
  public manual constructor; `Velo::register_peer(PeerInfo)` wires a peer with **no discovery
  backend**. `peer_info()` returns this instance's `PeerInfo`. `InstanceId` is a **random
  v4 UUID generated inside `VeloBackend::new`** (`transports.rs:139`) — there is **no** public
  hook to inject a fixed identity (hence the zero-discovery mechanism below cannot rely on a
  known controller id).
- **Discovery-free reachability (the crux, verified):**
  - Inbound frames dispatch **purely by handler name** — `DispatcherHub::dispatch_message(handler_name, ctx)`
    (`server/dispatcher.rs:231`) does **not** validate the frame's destination `InstanceId`.
  - A reply routes back by the **`WorkerId` embedded in the request's `ResponseId`** —
    `backend.send_message_to_worker(WorkerId::from_u64(response_id.worker_id()), …)`
    (`server/dispatcher.rs:316`). The requester's real identity travels *in the message*.
  - Therefore a caller may address a peer by its **known address with a placeholder
    `InstanceId`**, and the callee replies to the caller's real id — provided the callee can
    resolve the caller's `WorkerId` to an address (achieved by handing the caller's real
    `peer_info()` in the request payload and `register_peer`-ing it callee-side).
- **Large payloads.** Rendezvous: `register_data(Bytes) -> DataHandle` (producer) / `get(handle).await -> (Bytes, lease)`
  (consumer); `DataHandle` is a `u128` shippable in any message field. Used for the records-shard
  partition so a large shard never inflates an AM frame.
- **Streaming default (v0.5.0 #43).** With no `stream_config`, `build()` binds a TCP streaming
  listener on `0.0.0.0:0` and `peer_info()` includes a `tcp-stream` endpoint. Harmless for our
  messenger-only use; pin/relocate with `stream_bind_addr(ip)` if a pod must not advertise extra
  endpoints.

---

## 4. Architecture

Unchanged topology shapes: one `aiperf` **controller**, N **cells**, one merged
`native-v2.json`, one v2 request from Python. New wire (velo), new launch/registration.

```
            (operator injects controller DNS:port + cell_id via env — ZERO discovery)
   ┌──────────────────────── controller pod / process ────────────────────────┐
   │ aiperf (controller)                                                │
   │   Velo(bind known port) ── register handler ──┐  barrier: count==N       │
   │   run_cellular: slice budget, phase bases      │  heartbeat handler       │
   │   merge partitions → native-v2.json + exports  │  ship_partition handler  │
   └───────────────▲───────────────────────────────┴──────────▲───────────────┘
                   │ register(cell peer_info, cell_id) → CellLaunchSpec        │ heartbeat / partition
   ┌───────────────┴──────── cell pod / subprocess ───────────┴───────────────┐
   │ aiperf --cell                                                      │
   │   Velo(bind ephemeral) ── register_peer(controller placeholder@DNS)       │
   │   fetch CellLaunchSpec via register → set env → ordinary execute path     │
   │   CellRecordsShipper: heartbeat (am_send) + partition (rendezvous)        │
   └──────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Transport: velo behind the existing seam (`cellular/transport.rs`)

Keep the `CellClient` / `ControllerTransport` traits and the `CellMessage` enum. Add velo impls;
remove `TcpCellClient` / `TcpControllerTransport`. The control surface is a small set of named
velo handlers on the **controller**:

| Handler | Pattern | Direction | Payload → reply |
|---|---|---|---|
| `aiperf.cell.register` | `typed_unary` | cell → controller | `CellRegister { cell_id, cell_peer_info }` → `CellLaunchSpec` |
| `aiperf.cell.heartbeat` | `am_send` | cell → controller | `CellMessage::Heartbeat` |
| `aiperf.cell.partition` | `typed_unary` | cell → controller | `CellPartitionShip { cell_id, data_handle }` → `Ack` |

- **`register`** replaces the stdin-piped `CellLaunchSpec` **and** the "accept N connections"
  barrier: the controller's handler `register_peer`s the cell (from `cell_peer_info`), records the
  `cell_id` (barrier tick), and returns that cell's `CellLaunchSpec`. Idempotent on re-send
  (same `cell_id` → same spec, no double count).
- **`heartbeat`** carries the periodic `MetricsHeartbeat` (fire-and-forget; loss is acceptable —
  the report is exact from partitions).
- **`partition`** ships the final `RecordsShardPartition`: the cell stages the msgpack partition
  via `register_data`, sends the `DataHandle`; the controller `get`s it, decodes, and treats it
  exactly as a received `CellMessage::Partition` today. Ack lets the cell exit cleanly.

`ControllerTransport::recv()` is reimplemented as a merged stream fed by these handlers (an
internal `mpsc` the handlers push into), so `run_cellular`'s existing
`while partitions.len() < cell_count { select! recv ... }` loop is **unchanged**. `CellClient`
becomes a thin async velo client the cell drives after `register`.

The bespoke `encode_frame` / `read_frame` / length-prefix / `MAX_FRAME_LEN` plumbing is deleted;
velo owns framing.

### 4.2 Identity & connection — zero discovery (`ai-dynamo/main` only)

The **only** a-priori fact is the controller's DNS:port (operator-injected env
`AIPERF_CELL_CONTROLLER_ADDR`, unchanged name). velo targets peers by a random per-run
`InstanceId`, so reaching the controller from DNS:port alone needs one of three mechanisms,
resolved by a first-implementation spike behind a single `resolve_controller_peer()` seam. **All
three are zero-discovery** (no etcd/NATS/velo-discovery backend); a velo change, if any, is a PR
against `ai-dynamo/main`, never the legacy fork:

- **A — placeholder-address, zero patch.** The cell constructs the controller `PeerInfo` from the
  hardcoded DNS:port with a **placeholder `InstanceId`** using existing public velo API,
  `register_peer`s it, and calls `register` with its real `peer_info()` in the payload. Verified
  viable by velo's dispatch (by handler name, dst id ignored — `dispatcher.rs:231`) and reply
  routing (by the request's `ResponseId` worker id — `dispatcher.rs:316`); the only open question
  is whether a public constructor for a `WorkerAddress` from `host:port` exists
  (`WorkerAddressBuilder` is currently `crate`-internal).
- **A′ — placeholder-address, small upstream patch.** Same as A, enabled by a **small PR against
  `ai-dynamo/main`** (e.g. re-export `WorkerAddressBuilder` / add `WorkerAddress::tcp(addr)`, or an
  injectable `VeloBuilder::instance_id`). Cargo pins to the PR `rev` until it merges. Chosen over B
  only if genuinely small and useful upstream.
- **B — bootstrap-PeerInfo fetch, zero patch (guaranteed fallback).** The controller serves its
  real, fully-public, serde `peer_info()` bytes at a bootstrap port on the hardcoded DNS; the cell
  fetches + `register_peer`s them. Needs no velo change at all.

Flow (identical regardless of mechanism):

1. Controller builds `Velo` bound to the known port (fixed `bind_addr` for k8s; UDS/loopback for
   local); knows only its own bind address.
2. Each cell builds `Velo` (ephemeral bind) and obtains the controller `PeerInfo` via
   `resolve_controller_peer()` (mechanism A/A′/B), then `register_peer`s it.
3. Cell calls `aiperf.cell.register` with its **real `peer_info()`** + `cell_id`; the controller
   `register_peer`s the cell (so it can reach back) and replies with the `CellLaunchSpec`.
4. Cell applies the spec (sets the same `AIPERF_CELL_*` env the stdin path set today) and runs the
   ordinary single-process execute path.
5. Controller **barrier**: distinct registered `cell_id`s reach `cell_count`. A cell that never
   registers is caught by the launcher's failure watcher (local child exit) or a registration
   timeout (k8s) — fail loud, never hang.

**Verification spike (implementation step 1):** two `Velo` instances where the "controller" knows
only its own bind addr and the "cell" knows only the controller addr — land the cleanest of
A/A′/B end-to-end. B (bootstrap-PeerInfo fetch) is the guaranteed fallback and needs no velo
change; A/A′ are preferred when viable. The dispatch/reply code inspected above (dispatch by
handler name, reply by request `ResponseId`) makes A workable if a public `WorkerAddress`-from-addr
constructor exists, and A′ makes it so with a small `ai-dynamo/main` PR.

### 4.3 Launch abstraction (`CellLauncher` trait)

New object-safe seam in `runner/`; `run_cellular` is otherwise unchanged. The launcher's only job
is to *start N cell contexts* and *know how to detect a dead one* — the transport/registration is
uniform velo.

```rust
/// Starts a run's cells and reports hard failures; the transport is always velo.
trait CellLauncher {
    /// Start the cells; return handles the controller watches for hard failure.
    fn launch(&self, ctx: &CellLaunchContext) -> Result<Vec<CellHandle>>;
}
```

- **`LocalLauncher`** (default; dev/test): spawns `aiperf --cell` subprocesses as today, but
  passes only the **controller velo coordinate + `cell_id` via env** (no stdin `CellLaunchSpec`);
  the cell fetches its full spec over velo `register`. Transport: **UDS on Unix, TCP-loopback on
  Windows** (`#[cfg(unix)]` UDS, else `127.0.0.1:0`). `kill_on_drop` + child-exit watcher retained.
- **`K8sLauncher`**: **does not spawn** — the operator/JobSet already created the cell pods. It
  reads `cell_count` (from `runtime.cells`) and waits on the velo registration barrier. Cell pods
  get `cell_id` (JobSet job-index downward-API label → `AIPERF_CELL_ID`) and the controller
  coordinate from env.

Selection: `AIPERF_CELL_LAUNCHER=local|k8s` (default `local`), or a `runtime` config signal.

### 4.4 CellLaunchSpec delivery

`CellLaunchSpec` (cell_id, cell_count, phase_ordinal_bases, sliced envelope) is unchanged as a
type but is now **returned from the `register` reply** instead of piped to stdin. `spawn_cell`'s
stdin serialization is removed; `run_cell` in `main.rs` no longer reads a spec from stdin — it
reads the controller coordinate + `cell_id` from env, builds its velo client, and `register`s to
fetch the spec. This unifies local and k8s: both deliver the spec over velo.

### 4.5 Feature gating

All velo code (the `cellular/transport.rs` velo impl, `CellLauncher`, the cell register/ship
paths) sits behind a **`velo` cargo feature** on `aiperf-runtime` + `aiperf-cli`. Because TCP is fully
replaced:

- **Without `velo`:** `cells=1` (default) is byte-for-byte unchanged (never touches the
  transport). `cells>1` **fails closed** at controller detection with a clear diagnostic
  ("aiperf built without the `velo` feature; multi-cell runs require it").
- **With `velo`:** the full controller/cell/velo path. Mirrors how `dynosim` / `dynamo-full` gate
  optional runtime surface.

`Cargo.toml`: `velo = { git = "https://github.com/ai-dynamo/velo.git", tag = "v0.5.0",
default-features = false, optional = true }`; `velo = ["dep:velo"]` feature.

---

## 5. Data flow (end to end)

1. Python sends one v2 `execute` with `cfg.runtime.cells = N`. The receiving `aiperf`
   becomes the **controller** (`main.rs`).
2. Controller: validate cellular run shape (unchanged guards), compute per-phase ordinal bases +
   per-cell budget slices (unchanged), build `Velo` bound to the known port, register the three
   handlers, select the `CellLauncher`, and `launch` N cell contexts.
3. Each cell: build `Velo`, register the controller placeholder peer, `register` → receive
   `CellLaunchSpec`, set `AIPERF_CELL_*` env, run the ordinary execute path with the
   `CellularAutonomousIssuer` + `PartitionedSampler` (unchanged).
4. Cells stream `heartbeat` (am_send) during the run; on completion each ships its
   `RecordsShardPartition` via rendezvous → `partition` handler.
5. Controller collects one partition per cell (barrier on `cell_count`), merges in global-ordinal
   (scheduled) / concatenation (graph) order (unchanged), writes `native-v2.json`, runs the export
   plane, writes the merged heartbeat sidecar (unchanged), emits the terminal v2 envelope.

Byte-parity vs a 1-cell run is preserved exactly where it is today (seeded `concurrency` phase,
synthetic single-turn HTTP, no ramps/cancellation); the aggregate-equivalent knobs (rate pacing,
cancellation, ramps, multi-URL) keep their existing warnings.

---

## 6. Failure & lifecycle

- **Cell dies before registering:** local — child-exit watcher fires (unchanged `select!` against
  a failure channel); k8s — pod failure surfaces via the launcher's watch (JobSet/pod status is
  the operator's concern; the controller sees a missing registration past a timeout). Fail loud,
  never hang.
- **Cell dies after shipping, before others:** same accepted race as today (documented in
  `cellular_controller.rs`) — the biased collect takes a ready partition before a failure.
- **Heartbeat loss:** tolerated (report is exact from partitions).
- **Controller panic:** unchanged `catch_unwind` → typed `success:false` execution envelope.
- **Velo shutdown:** drop `Arc<Velo>` tears down transports; no graceful-drain requirement on the
  cellular control plane (a handful of messages, not a hot path).

---

## 7. Testing

- **Unit:** velo `CellClient`/`ControllerTransport` round-trip; the placeholder-address +
  peer-handoff spike (§4.2); register/barrier handshake reaching `cell_count`; rendezvous partition
  transfer decode-equals-input; `CellMessage` variants over velo.
- **E2e (kept, now over velo):** `rust/e2e/tests/test_cellular.rs` and `test_graph_cellular.rs`
  run via `LocalLauncher` over velo (UDS on Unix / loopback on Windows). The existing byte-parity
  assertions (3-cell reproduces 1-cell ISL/OSL distributions; graph total record count + token
  distribution) are unchanged — they now exercise the velo path.
- **Gate:** these tests build with the `velo` feature; a no-`velo` build asserts `cells>1` fails
  closed with the diagnostic.

---

## 8. Docs to update (same change)

- `rust/AGENTS.md` / `CLAUDE.md` / `.github/copilot-instructions.md` / `.cursor/rules/python.mdc`
  (identical body): the crate-table `aiperf-cli`/`aiperf-runtime` cellular note (velo transport,
  `velo` feature) and the "Build, test, run" cellular paragraph; run
  `python tools/check_agent_files_sync.py`.
- `rust/specs/README.md` + root `llms.txt`: reference this design.
- Append a dated `## Addendum — 2026-07-15 — velo cell transport + zero-discovery k8s topology`
  to `rust/specs/2026-07-12-cellular-ready-seams-and-roadmap.md` (never edit its body) recording:
  TCP transport replaced by velo behind the unchanged `CellClient`/`ControllerTransport` seam;
  zero-discovery placeholder-address registration; `CellLauncher` (Local-over-velo / K8s-pod);
  `velo` feature gate; measurement seams unchanged.
- `python tools/check_docs_current.py` before commit.

---

## 9. Risks & open questions

1. **Placeholder-addressed send (primary risk).** Mitigated by the step-1 spike choosing among
   A / A′ / B; mechanism B (bootstrap-PeerInfo fetch) is the guaranteed zero-velo-change fallback.
   A′ (a small `ai-dynamo/main` PR) is acceptable but adds a temporary Cargo `rev` pin until it
   merges. All zero-discovery; none touch a fork.
2. **Streaming endpoint advertisement (v0.5.0 default TCP stream listener).** Each velo instance
   binds an extra `0.0.0.0:0` stream listener by default. Benign for messenger-only use; if a pod
   must not open it, pin via `stream_bind_addr`. Confirm no port-exhaustion concern at large N.
3. **Rendezvous vs inline for partitions.** Start with rendezvous for the (potentially large)
   partition; if a small-partition run shows rendezvous overhead, inline AM is a trivial swap
   behind the same handler.
4. **k8s cell identity source.** This spec assumes the operator injects `AIPERF_CELL_ID` (JobSet
   job-index) + `AIPERF_CELL_CONTROLLER_ADDR`. The operator/JobSet that does so is the follow-on
   effort; the Rust side only consumes these two env facts.
5. **velo as a build dependency.** New git dependency (experimental crate). Feature-gated so the
   default runner build is unaffected; CI builds the `velo` feature separately.
6. **Windows CI.** UDS is `#[cfg(unix)]`; the loopback path must be exercised on Windows CI (or at
   least compile-checked) since that is the Windows transport.

---

## 10. Implementation phases (for the plan)

1. **Spike:** placeholder-address + peer-handoff round-trip against velo v0.5.0 (gate the whole
   design; pick placeholder vs bootstrap-file).
2. **Cargo + feature:** add `velo` optional dep + `velo` feature to `aiperf-runtime` / `aiperf-cli`.
3. **Transport:** velo `CellClient` / `ControllerTransport` impls behind the existing traits;
   remove the TCP impls; the three handlers; rendezvous partition ship.
4. **Registration/barrier:** `register` handler + controller barrier; `CellLaunchSpec` over the
   reply; cell fetches spec from env coordinate (remove stdin path).
5. **Launcher seam:** `CellLauncher` trait + `LocalLauncher` (UDS/loopback) + `K8sLauncher`
   (no-spawn, barrier-wait); wire selection.
6. **Fail-closed:** `cells>1` without the `velo` feature.
7. **Tests:** unit + port `test_cellular.rs` / `test_graph_cellular.rs` onto the velo `LocalLauncher`.
8. **Docs:** the §8 updates in the same change.

---

## Addendum — 2026-07-15 — built, and mechanism resolved; superseded going forward by the ultimate spec

This design is now **built** on `velo-connect` and the connection question is **resolved**: mechanism
A/A′/B is settled by **connect-by-endpoint** — the fork branch `feat/connect-by-endpoint` adds
`Velo::connect(Endpoint) -> PeerInfo` (address-first `_hello` handshake, response carries
`peer_info`), and AIPerf's `connect.rs::connect_controller` wraps it with a retry loop. The
bootstrap-PeerInfo fallback (mechanism B, `serve_bootstrap`) — still present on the older
`rust-operator` branch — is **superseded** by connect-by-endpoint; do not build new work on it. The
four-handler velo protocol (`register`/`heartbeat`/`partition`/`store_partition` over rmp raw
payloads), synchronized START via a velo `EventHandle` (with an AIPerf-side `AtomicU32` count barrier
— velo has no count-trigger primitive), the `CellLauncher` Local/K8s split, and Stages B–G
artifact/dataset shipping are all shipped.

The forward north-star — dataset **fan-out** (producer-owned SPMC broadcast, which velo does **not**
have and must add), the **monotonic phaser** control plane (START generalized to every phase
transition), the per-request dispatch state machine + `DistributionMiss`, and bounded-memory record
collection (sketch `StorePartition`) — is specified in
`2026-07-15-ultimate-cellular-velo-runtime-design.md`, which is authoritative where it revises this
spec.
