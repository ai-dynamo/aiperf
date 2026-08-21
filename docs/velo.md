# Velo in AIPerf

This document describes exactly how the [`velo`](https://github.com/ajcasagrande/velo)
messaging framework is used inside AIPerf, traced from the Rust source rather than
from design intent. Every claim here maps to code under `rust/runtime/src/cellular/`
and `rust/runtime/src/engine/`.

## Scope: what velo is and is not

`velo` is an external async messaging framework. In AIPerf it is used for exactly one
thing: the **cross-process / cross-host cellular control plane** — the small control
and metrics-summary messages exchanged between a controller and its cells.

It is **never** on the per-request or per-token hot path. It has no involvement in the
HTTP/gRPC request transports, the metrics computation, scheduling, or single-process
runs. When AIPerf runs without cellular execution (`--cells` unset), velo is not
constructed at all.

- Git dependency: `github.com/ajcasagrande/velo`, branch `feat/connect-by-endpoint`,
  `default-features = false` (`rust/runtime/Cargo.toml`).
- Gated behind the `cellular` Cargo feature: `cellular = ["dep:velo", "dep:zstd"]`.
  Every velo-touching module is `#[cfg(feature = "cellular")]`.
- The default CLI feature set includes `cellular`.

## The velo API surface AIPerf calls

All direct velo usage lives in `rust/runtime/src/cellular/transport/`:

| Concern        | velo API used |
|----------------|---------------|
| Construction   | `Velo::builder().add_transport(..).build()` with `TcpTransportBuilder` (loopback / fixed bind / `from_listener`) or `UdsTransportBuilder` |
| Bootstrap      | `velo.connect(Endpoint::Tcp \| Uds)` — address-first `_hello` handshake returning the peer's real `PeerInfo` |
| Addressing     | `velo.register_peer(PeerInfo)`, `velo.peer_info()`, `peer.instance_id()` |
| Unary RPC      | `velo.unary(NAME).raw_payload(Bytes).instance(id).send()` |
| Fire-and-forget| `velo.am_send(NAME).raw_payload(Bytes).instance(id).send()` |
| Handlers       | `velo.register_handler(Handler::unary_handler_async(NAME, ..))` / `Handler::am_handler_async(NAME, ..)` |
| Sync events    | `velo.event_manager().new_event()` → `EventHandle`; `trigger()` / `awaiter(handle).await` |

### Discovery-free bootstrap

A cell knows exactly one fact a priori: the controller coordinate string in
`AIPERF_CELL_CONTROLLER_ADDR` (`tcp://HOST:PORT`, or `uds://PATH` for a pure-local run).
There is no discovery backend and no bootstrap side channel. The cell calls
`velo.connect(endpoint)`; velo's `_hello` handshake learns the controller's real
`PeerInfo` and mutually registers the two peers (`transport/connect.rs`).

Kubernetes coordinates are headless-service DNS names, so `parse_endpoint` resolves
them through `to_socket_addrs` (getaddrinfo) before handing velo a concrete
`SocketAddr`. `connect_controller` retries for up to 60s (200ms interval) because a
cell pod may start before the controller has bound its listener.

### Wire encoding

All message bodies are **`rmp-serde` (MessagePack) carried as velo _raw_ payloads**,
not velo's typed JSON payloads. The reason is numeric fidelity: t-digest sketches in
`MetricsHeartbeat` anchor `min = +inf`, and records carry `NaN` metric values —
neither survives a JSON round-trip. The dataset fan-out plane additionally
zstd-compresses (level 3) via `zpack` / `zunpack`, since it replays identical-shaped
request bodies to every cell.

## The three velo planes

All three are a cell → controller star, defined in `rust/runtime/src/cellular/transport/`.

### 1. Cell ↔ controller transport (`velo_transport.rs`)

Four named handlers carry the whole protocol:

- `aiperf.cell.register` (unary) — a cell sends `CellRegister { cell_id, cell_peer }`.
  The controller `register_peer`s it, ticks the synchronized-start barrier, and replies
  with `RegisterReply { envelope, start_event }`. The envelope is the cell's sliced
  protocol-v2 execute envelope — this **replaces the stdin spec pipe** used by
  non-cellular self-execution.
- `aiperf.cell.heartbeat` (fire-and-forget) — the cell's periodic `MetricsHeartbeat`.
- `aiperf.cell.partition` (unary, ack'd) — the cell's final `RecordsShardPartition`
  (raw records, for the byte-exact global-order merge / retain path).
- `aiperf.cell.store_partition` (unary, ack'd) — the cell's final `ColumnStorePartition`
  (a pre-folded exact/sketch store, shipped in place of raw records on the fold path).

Because a cell ships its terminal partition from a **fresh** velo instance the
controller has not yet seen, each ship (`CellPartitionShip` / `CellStorePartitionShip`)
carries the shipper's own serialized `PeerInfo` so the controller can `register_peer`
it and route the ack back.

### 2. Phaser control plane (`phaser_velo.rs`, opt-in `AIPERF_CELL_PHASER_START`)

Distributes the monotonic `Phaser` as a control plane:

- `aiperf.phaser.subscribe` (unary) — cell subscribes with its `PeerInfo`; the
  controller attaches a broadcast consumer, returns the replay snapshot, and spawns a
  per-cell pump task.
- `aiperf.phaser.event` (fire-and-forget push) — the pump forwards each live
  generation to the subscribed cell.

Replay-on-attach + live-tail semantics guarantee a generation advanced concurrently
with a subscribe lands in exactly one of {reply snapshot, pushed live}.

### 3. Dataset fan-out data plane (`dataset_velo.rs`, opt-in `AIPERF_CELL_DATASET_FANOUT`)

Distributes the dataset so cells do not each regenerate it:

- `aiperf.dataset.subscribe` (unary) — cell subscribes; controller returns the replay
  snapshot and spawns a pump for the live tail.
- `aiperf.dataset.chunk` (fire-and-forget push) — one broadcast chunk (zstd + rmp).

The controller builds each request's endpoint-ready body once, broadcasts them in
16-request chunks, and finalizes. Each cell subscribes and builds an owned index
filtered to its round-robin slice, using O(1/N) RAM even though it observes every
chunk.

## Engine orchestration wiring

Under `rust/runtime/src/engine/`:

- **Controller** (`cellular_controller.rs`): binds one velo instance at a known
  coordinate (`controller_bind_and_endpoint` — k8s uses a fixed `0.0.0.0:PORT`,
  same-host uses an OS-assigned loopback port); optionally binds `PhaserServer` and
  `DatasetServer` on the same instance; precomputes each cell's sliced envelope into a
  `SpecFor` lookup; binds `VeloControllerTransport`; awaits `await_all_registered()`
  (unless `AIPERF_CELL_BARRIER_FREE=1` triggers START immediately); `trigger()`s the
  run-wide START event; then collects exactly one partition (or store) per cell.
- **Cell** (`cellular_cell.rs`): builds a **separate short-lived velo instance per
  phase** — `fetch_cell_envelope` (register + await START), `verify_dataset_fanout`
  (dataset subscribe), and `ship` (final heartbeat + terminal partition). The ship runs
  on a dedicated thread with its own multi-thread runtime so velo never touches the
  cell's current-thread execute runtime.
- **Hierarchy refusal** (`cellular_aggregator.rs`): a requested cellular fanout is
  rejected before controller startup. Supported cellular runs use only the flat
  controller-to-cell Velo topology.

## What velo does NOT carry: the HTTP artifact plane

Large per-record artifacts are **not** shipped over velo. They travel on a separate raw
[`hyper`](https://hyper.rs) HTTP/1 plane (`engine/artifact_shipping.rs`) with zstd
content-encoding, whose authority is derived by port-swapping the same velo coordinate.

So the division is:

- **velo** — small control messages and metrics summaries (register, START, heartbeats,
  folded/record partitions, phaser events, dataset chunks).
- **HTTP (hyper)** — bulk per-record artifact bytes.

### Future direction: fold the bulk plane onto velo (not yet implemented)

The HTTP+zstd artifact plane is a second transport and a second bootstrap sitting next
to velo. The intended future consolidation is to carry the bulk over velo as well,
removing the separate axum server and its extra exposed port. Two candidate mechanisms:

- **velo-streaming** — ship artifacts as a `StreamSender` of chunks, preserving the
  bounded-memory O(chunk) property AIPerf already relies on (a cell never buffers a whole
  multi-hundred-MB artifact).
- **nixl-backed rendezvous / descriptor-pull** — the cell has *already written the
  artifacts to disk*, so instead of read+compress+push it registers a descriptor over the
  file/mmap and the controller **pulls** only the ranges it needs. With nixl descriptors
  this is zero-copy cross-host (RDMA / GPU-direct), avoids CPU-bound zstd + socket copies,
  and lets the controller fetch lazily, skip, or dedup across N cells.

Both collapse artifact shipping onto the one velo transport and its connect-by-endpoint
bootstrap. Until this lands, bulk artifacts remain on the HTTP plane described above, and
cross-host artifact runs additionally require the operator to expose the controller
artifact port (default `9600`, `AIPERF_CONTROLLER_ARTIFACT_BIND`) alongside velo's
bootstrap port — a wiring gap that does not affect synthetic / no-artifact runs.

## Synchronized start

The run-wide START is a velo `EventHandle` created on the controller before its velo
instance moves into the transport. Cells `awaiter(start_event).await` after registering.
The controller `trigger()`s it once all `cell_count` cells have registered (or
immediately under barrier-free mode), so all cells begin dispatching together. If the
controller bails before triggering, dropping the untriggered event **poisons** it,
unblocking every waiting cell with an error rather than a hang.

## Diagram

```
                         AIPERF_CELL_CONTROLLER_ADDR = tcp://HOST:PORT
                                          (one a-priori fact per cell)
                                                    │
   ┌────────────────────────────────────────────────────────────────────────────┐
   │                              CONTROLLER (one velo instance)                  │
   │                                                                              │
   │   velo handlers:                        events:                             │
   │     aiperf.cell.register     (unary)      START EventHandle                  │
   │     aiperf.cell.heartbeat    (am)          trigger() once all registered     │
   │     aiperf.cell.partition    (unary/ack)                                     │
   │     aiperf.cell.store_partition (unary/ack)                                  │
   │   optional servers on same velo:                                            │
   │     PhaserServer  : aiperf.phaser.subscribe / .event                        │
   │     DatasetServer : aiperf.dataset.subscribe / .chunk                       │
   └───────▲───────────────▲───────────────▲──────────────────────┬─────────────┘
           │ rmp raw        │ rmp raw        │ rmp raw              │ push (am)
           │ over velo      │ over velo      │ over velo            │ phaser/dataset
           │                │                │                      ▼
   ┌───────┴──────┐  ┌──────┴───────┐  ┌─────┴────────┐    replay-on-attach
   │   CELL 0     │  │   CELL 1     │  │   CELL N-1   │    + live tail
   │  (per-phase  │  │              │  │              │
   │   velo inst) │  │              │  │              │
   │              │  │              │  │              │
   │ 1 connect()  │  │ ...          │  │ ...          │   ── velo _hello handshake
   │ 2 register   │  │              │  │              │      -> RegisterReply
   │   -> envelope│  │              │  │              │        { sliced envelope,
   │ 3 await START│  │              │  │              │          start_event }
   │ 4 [dataset]  │  │              │  │              │
   │ 5 dispatch   │  │              │  │              │
   │ 6 ship:      │  │              │  │              │
   │   heartbeat  │  │              │  │              │
   │   + partition│  │              │  │              │
   └──────┬───────┘  └──────────────┘  └──────────────┘
          │
          │  bulk per-record artifacts do NOT use velo:
          └────────────►  HTTP/1 (hyper) artifact plane, zstd content-encoding
                          authority = velo coordinate with port swapped
                          (engine/artifact_shipping.rs)

   Hierarchical aggregation request:

     AIPERF_CELL_AGG_FANOUT ──► refusal before controller startup
```
