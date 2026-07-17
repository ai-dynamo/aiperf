<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Velo as the core transport for AIPerf cell mode

**Status:** Built
**Date:** 2026-07-15
**Repo:** `aiperf`
**Relationship:** Realizes the "Phase 2 cross-node transport" swap that
`rust/specs/2026-07-12-cellular-ready-seams-and-roadmap.md` left as an abstract
`CellClient`/`ControllerTransport` seam. The forward north-star that builds on this
transport — dataset fan-out (producer-owned SPMC broadcast), the monotonic phaser
control plane, the per-request dispatch state machine + `DistributionMiss`, and
bounded-memory record collection — is specified in
`2026-07-15-ultimate-cellular-velo-runtime-design.md`, which is authoritative where
it extends this spec.

---

## 1. What this is

The cell↔controller wire is **velo**, behind the existing `CellClient` /
`ControllerTransport` traits (`rust/runtime/src/cellular/transport/`). Cells reach
the controller with **zero discovery infrastructure** (no etcd / NATS /
velo-discovery backend) — from a single operator-hardcoded coordinate, mirroring the
Python k8s model (`AIPERF_K8S_ZMQ_CONTROLLER_HOST`). Every measurement seam the
cellular runtime ships — the static `owned_positions` partition, `IssuanceAuthority`,
`RecordsShard`, `MetricsHeartbeat`/t-digest, and the global-ordinal / concatenation /
store merges — is **byte-for-byte unchanged**. Only the wire, the launch mechanism,
and the registration/barrier are velo's.

The static partition is retained (not a Python-style dynamic credit router): it is
the smallest delta and preserves byte-parity; dynamic routing can come later behind
the same seam. All velo code sits behind a **`velo` cargo feature** (on by default;
`velo = ["dep:velo", "dep:zstd"]`). Without it, `cells = 1` (the default) is
byte-for-byte unchanged (it never touches the transport) and `cells > 1` **fails
closed** at controller detection with a clear diagnostic.

Velo is `ai-dynamo/main` lineage plus the `feat/connect-by-endpoint` fork branch
(§2.1) — never the legacy fork.

### Non-goals

- Dynamic credit routing (least-loaded / session-sticky pull).
- Cross-cell side-channel telemetry aggregation (`server_metrics` / `gpu_telemetry` /
  `network_latency`) — the documented report-fidelity gap.
- Any change to the S1/S2/S3/S4 measurement seams or the merged-report byte-parity
  contract.

---

## 2. The velo transport (`cellular/transport/`)

The seam is two async traits over a `CellMessage` enum
(`Heartbeat { cell_id, Box<MetricsHeartbeat> }`, `Partition(RecordsShardPartition)`,
`StorePartition(Box<ColumnStorePartition>)`), all encoded as **MessagePack
(`rmp-serde`) carried as velo *raw* payloads** — not velo's typed (JSON) payloads —
because the t-digest sketches anchor `min = +inf` and records carry NaN metric
values, neither of which JSON round-trips. Velo owns framing and transparent
large-payload staging, so the module does no length-prefixing of its own. A cell that
was a thread rather than a process would implement the same two traits over an
in-process channel with the controller and merge logic unchanged.

The velo impls are `VeloControllerTransport` / `VeloCellClient` (`velo_transport.rs`),
gated on the `velo` feature. The control surface is four named handlers on the
controller:

| Handler (`HANDLER_*`) | Pattern | Direction | Payload → reply |
|---|---|---|---|
| `aiperf.cell.register` | typed unary | cell → controller | `CellRegister { cell_id, cell_peer }` → `RegisterReply { envelope, start_event }` |
| `aiperf.cell.heartbeat` | `am_send` (fire-and-forget) | cell → controller | `CellMessage::Heartbeat` |
| `aiperf.cell.partition` | unary | cell → controller | `CellPartitionShip { cell_peer, partition }` → `CellAck` |
| `aiperf.cell.store_partition` | unary | cell → controller | `CellStorePartitionShip { cell_peer, partition }` → `CellAck` |

- **`register`** replaces both the stdin-piped `CellLaunchSpec` and the "accept N
  connections" barrier: the handler `register_peer`s the cell (from its own serialized
  `PeerInfo` in `cell_peer`), counts the registration (barrier tick), and returns that
  cell's sliced envelope plus the START `EventHandle`.
- **`heartbeat`** carries the periodic `MetricsHeartbeat` fire-and-forget; loss is
  acceptable because the report is exact from partitions.
- **`partition`** / **`store_partition`** ship the cell's final records-shard
  partition or, for a metrics-only exact-fold cell, its folded `ColumnStorePartition`;
  the controller decodes and treats each exactly as a received `CellMessage`. The
  `CellAck` lets the cell exit cleanly.

`ControllerTransport::recv()` is a merged stream fed by these handlers (an internal
channel the handlers push into), so `run_cellular`'s collect loop is unchanged.

**Fresh-instance-ack pattern.** A cell touches velo twice on separate short-lived
runtimes — once to fetch its envelope, then a *fresh* ship instance the controller
never registered — so every ship DTO (`CellPartitionShip` / `CellStorePartitionShip`)
carries the shipping instance's own serialized `PeerInfo`; the partition handler
`register_peer`s it before replying so the `CellAck` routes home (test
`ship_from_a_fresh_instance_is_acked`).

### 2.1 Identity & connection — zero discovery via connect-by-endpoint

Velo targets peers by a random per-run `InstanceId`, so reaching the controller from
a DNS:port alone needs an address-first connect. This is provided by the fork branch
`feat/connect-by-endpoint`, which adds `Velo::connect(Endpoint) -> PeerInfo` (an
address-first `_hello` handshake whose response carries the responder's real
`peer_info`) and `Endpoint::{Tcp, Uds}`. AIPerf's `connect.rs::connect_controller`
wraps this with a retry loop (`build_velo` / `parse_endpoint` / `connect_controller`).
This supersedes the earlier bootstrap-PeerInfo-fetch fallback; do not build new work
on it.

The only a-priori fact is the controller's coordinate, injected as
`AIPERF_CELL_CONTROLLER_ADDR` (`file:PATH` locally, `tcp://HOST:PORT` in k8s). Flow:

1. Controller builds `Velo` bound to a known coordinate and publishes it.
2. Each cell builds `Velo` (ephemeral bind) and `connect`s the controller by endpoint.
3. Cell calls `aiperf.cell.register` with its `cell_id` and its own `PeerInfo`; the
   controller `register_peer`s it and replies with the sliced envelope + START event.
4. Cell sets the `AIPERF_CELL_*` env the launch spec dictates and runs the ordinary
   single-process execute path (`CellularAutonomousIssuer` + `PartitionedSampler`).
5. Controller **barrier**: distinct registered `cell_id`s reach `cell_count`, then it
   triggers START. A cell that never registers is caught by the launcher's failure
   watcher (local child exit) or the controller's registration timeout — fail loud,
   never hang.

### 2.2 Synchronized START

The controller creates a velo **distributed event**
(`velo.event_manager().new_event()`) and threads its `EventHandle` into every
`RegisterReply`. Velo distributed events are **single-shot** and have **no
register-count→trigger barrier primitive** (the only native aggregation is
`merge_events`, an AND-join), so the count barrier is **AIPerf's own** `AtomicU32` +
`Notify`: the register handler counts registrations and the Nth notifies an
`all_registered` `Notify`; the controller `select!`s on that (bounded by
`register_timeout`, default 5 min) then calls `start_event.trigger()`. Cells block in
`await_start` → `event_manager().awaiter(handle).await`. A velo `EventAwaiter` is a
real `Future` with a completed-event cache, so a late awaiter resolves immediately. A
controller bail before trigger **drop-poisons** the event so waiters error rather than
hang (test `synchronized_start_releases_all_cells_together`).

---

## 3. Launch (`CellLauncher` trait, `engine/cell_launcher.rs`)

The velo transport is uniform across deployments; only *how the cell processes come to
exist* differs, so that is the one launch seam. It is object-safe; `run_cellular` is
otherwise unchanged.

- **`LocalLauncher`** (default; dev/test): spawns `aiperf --cell` subprocesses on the
  same host, passing only `cell_id`, `cell_count`, the controller coordinate, the
  per-phase ordinal bases, and (Stage E) the artifact upload authority — all via env,
  no stdin pipe. `kill_on_drop(true)` SIGKILLs cells if the controller aborts, so a
  failed run never leaves cells generating load; a `CellHandle` wraps each child and
  `wait_failure` surfaces a non-zero exit.
- **`K8sLauncher`**: spawns nothing — the operator/JobSet already created the cell
  pods. It only reports how many cells to expect; the pods discover the controller
  from the operator-injected env (`AIPERF_CELL_ID` from the JobSet job-index label,
  `AIPERF_CELL_CONTROLLER_ADDR` from the deterministic controller DNS). `CellHandle`
  has no child, so `wait_failure` is `pending()` forever and a dead pod is caught by
  the controller's registration timeout.

Selection is `AIPERF_CELL_LAUNCHER=local|k8s` (default `local`).

The cell learns its full run shape (`cell_id`, `cell_count`, `phase_ordinal_bases`,
sliced envelope) from the `register` reply, not stdin: `run_cell` reads the controller
coordinate + `cell_id` from env, builds its velo client, connects, and `register`s to
fetch its envelope. This unifies local and k8s: both deliver the spec over velo.

---

## 4. Data flow (end to end)

1. Python sends one v2 `execute` with `cfg.runtime.cells = N`. The receiving `aiperf`
   becomes the **controller** (mode dispatch reads `/run/cfg/runtime/cells` via
   `cell_count_from_envelope`).
2. Controller: validate the cellular run shape (§5), compute per-phase ordinal bases +
   per-cell budget slices, build `Velo` bound to the known coordinate, register the
   handlers + START event, select the `CellLauncher`, and `launch` N cell contexts.
3. Each cell: build `Velo`, connect the controller by endpoint, `register` → receive
   its envelope + START event, set `AIPERF_CELL_*` env, await START, run the ordinary
   execute path.
4. Cells stream `heartbeat` (`am_send`) during the run; on completion each ships its
   `RecordsShardPartition` (or folded `ColumnStorePartition`) over the transport.
5. Controller collects one partition per cell (barrier on `cell_count`), merges in
   global-ordinal (scheduled) / concatenation (graph) / store-append (exact-fold)
   order, writes `native-v2.json`, runs the export plane, writes the merged heartbeat
   sidecar, and emits the terminal v2 envelope.

Byte-parity vs a 1-cell run is preserved exactly where it is today (seeded
`concurrency` phase, synthetic single-turn HTTP, no ramps/cancellation); the
aggregate-equivalent knobs (rate pacing, cancellation, ramps, multi-URL) keep their
warnings (§5).

---

## 5. Fail-closed matrix and allowed-with-warning knobs

`validate_cellular_run_shape` + the per-kind validators fail closed on shapes the
partition/issuance seam is not sound for, and allow-with-warning the ones that trade
byte-parity for scale (cellular's purpose is multi-node scale with acknowledged
precision loss).

- **Fail closed:** non-`http` transport (gRPC/offline/dynosim); non-`{synthetic, file,
  public}` dataset; file/public formats outside the single-turn allowlist; multi-turn
  on the retain path; scheduled phases outside `{concurrency, poisson, gamma,
  constant}` or with a `duration` / retain-path `sessions` / `adaptive_scale` bound;
  caps `< cell_count`; graph phases with a static `requests` budget; mixed
  store+record partitions; a no-`velo` build with `cells > 1`.
- **Allowed but warned (aggregate-equivalent, not byte-exact):** multi-URL
  round-robin, concurrency/prefill/rate ramps, post-send cancellation, rate pacing
  (`rate / cell_count`), and an auto-derived seed (a missing `run.random_seed` is
  hashed from the run identity and injected identically into every cell, so the
  partition stays coherent and reproducible per `benchmark_id`).

---

## 6. Failure & lifecycle

- **Cell dies before registering:** local — the child-exit watcher fires (`select!`
  against a failure channel); k8s — a missing registration past the timeout. Fail
  loud, never hang.
- **Cell dies after shipping, before others:** the accepted biased-collect race — the
  collect loop takes a ready partition before a failure.
- **Heartbeat loss:** tolerated (report is exact from partitions).
- **Controller panic:** wrapped in `catch_unwind` → a typed `success:false` execution
  envelope.
- **Velo shutdown:** dropping `Arc<Velo>` tears down transports; no graceful-drain
  requirement on the control plane (a handful of messages, not a hot path).

---

## 7. Testing

- **Unit:** velo `CellClient`/`ControllerTransport` round-trip; the connect-by-endpoint
  handshake; register/barrier reaching `cell_count`; fresh-instance-ack partition
  transfer; `CellMessage` variants over velo; synchronized-START release.
- **E2e (over velo):** `rust/e2e/tests/test_cellular.rs` and `test_graph_cellular.rs`
  run via `LocalLauncher` (UDS on Unix / loopback on Windows). A 3-cell run reproduces
  the 1-cell ISL/OSL distributions byte-for-byte (scheduled) and the graph total
  record count + input-token distribution (graph).
- **Gate:** a no-`velo` build asserts `cells > 1` fails closed with the diagnostic.
