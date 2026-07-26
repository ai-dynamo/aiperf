<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Velo hub

## Purpose

The velo hub is a **per-experiment** control-plane host: one hub per benchmark
run, co-binding a single [`velo`](https://github.com/ajcasagrande/velo) messaging
instance and a single `axum` HTTP service, and exposing a set of *plugins* that
each contribute both an HTTP router and a set of velo handlers. A plugin's
behavior is reachable identically over an HTTP route **or** a velo message: the
same handler logic backs both surfaces.

The hub exists to unify the two control/transport planes AIPerf runs today into
one composable host:

- the **velo cell↔controller plane** (`cellular::transport`, anchored by the
  controller at tcp `:9500`), and
- the **raw-hyper HTTP + zstd artifact plane** (`engine::artifact_shipping`, a
  second server at tcp `:9600`).

Both planes are per-run already; the hub is the shape they converge on — one
address anchor, one plugin registry, one dual-surface handler contract — so a run
has a single well-known service rather than an ad-hoc pair of ports.

### Per-experiment, not cluster-wide

The hub is scoped to a single run, deliberately:

- AIPerf runs are **independent benchmarks**; results are never aggregated across
  runs. A per-run hub bounds blast radius (a crash or overload touches one run)
  and scales naturally with the number of concurrent jobs.
- It unifies the three launch topologies behind one anchor: the hub is in-process
  in the **primary rank** (CLI single-host / slurm rank 0) or its own **service**
  (k8s), with the same code path either way.

The cluster-wide "list all runs" role is intentionally **out of scope for the
hub**. That index stays with the Kubernetes operator (a thin catalog of
AIPerfJob CRs); the hub never enumerates peer runs.

## Built

### Location and gating

The hub is a module, `aiperf_runtime::hub`, gated behind the existing `cellular`
feature (`#[cfg(feature = "cellular")]`). It was placed in `aiperf-runtime`
rather than a new crate because every dependency it needs is already a runtime
dependency under `cellular` — `velo`, `axum`, `hyper`, `tokio` — and it reuses
`cellular::transport::connect::{build_velo, BindSpec, parse_endpoint,
connect_controller}` directly. A separate crate would duplicate those deps and
the velo git pin, and would trip the crate-topology doc guard for no isolation
benefit; the module sits beside `cellular` and shares its feature flag.

### The plugin trait

```rust
pub trait HubPlugin: Send + Sync {
    fn prefix(&self) -> &str;                                   // e.g. "/discovery"
    fn required_abi(&self) -> HubAbiRequirement {               // ABI negotiation
        HubAbiRequirement::current()
    }
    fn router(&self) -> axum::Router;                           // HTTP surface
    fn register_velo_handlers(&self, velo: &Arc<Velo>)          // velo surface
        -> Result<(), HubError>;
}
```

Each plugin owns a stable `prefix` (its HTTP mount point and its diagnostic
identity), an `axum::Router` nested under that prefix, and a
`register_velo_handlers` hook that installs its velo handlers on the shared
instance. A plugin is expected to route both surfaces into **one shared handler
function** so the HTTP and velo paths cannot diverge.

### ABI/version negotiation

The plugin surface carries a version contract. `HUB_ABI_VERSION` is the hub's
current contract version, and `required_abi` returns the inclusive
`HubAbiRequirement { min, max }` range a plugin was built against. The default
(`HubAbiRequirement::current`) pins a plugin to exactly the ABI it compiled
against — the safe default — so existing plugins compile unchanged; a plugin that
has verified it tolerates a wider window opts into `HubAbiRequirement::range(min,
max)` explicitly. `Hub::register` checks the requirement **first**, before it
touches any hub state, and rejects a plugin whose range excludes
`HUB_ABI_VERSION` with `HubError::IncompatibleAbi { prefix, required, supported }`
(a `Display` message naming the mismatch). Because the check precedes the prefix
insert and handler install, an incompatible plugin needs no rollback — the hub is
left exactly as before, alongside the existing duplicate-prefix and
velo-handler-failure rollbacks. `DiscoveryPlugin` declares `current()` explicitly
as the worked example.

### The hub host

`Hub` composes plugins over one velo instance and serves both surfaces:

- `Hub::new(velo: Arc<Velo>)` takes an already-bound velo instance (built with
  the shared `build_velo(BindSpec::…)`, so the hub does not duplicate transport
  construction).
- `Hub::register(plugin)` is **transactional-style**: it rejects a duplicate
  `prefix` (mirroring `AIPerfRegistry`/`TransactionalRegistry` duplicate-name
  rejection) and, only after that check passes, calls the plugin's
  `register_velo_handlers`. On success the plugin is retained for HTTP mounting.
- `Hub::router()` merges every registered plugin's router, each nested under its
  `prefix`, into one `axum::Router`.
- `Hub::serve(http_listener)` spawns the axum server on a caller-provided
  `TcpListener` (bound `127.0.0.1:0` in tests, `0.0.0.0:PORT` in k8s), returning a
  `HubServer` handle with graceful shutdown. The velo instance is already serving
  from `Hub::new`; the returned handle keeps both alive.

The velo and HTTP surfaces bind distinct sockets (as the controller and artifact
planes do today); "co-bound" means one `Hub` owns and lifecycle-manages both, not
that they share a port.

### The discovery plugin (first plugin, dual-surface proof)

`DiscoveryPlugin` is the hub's connect-by-endpoint anchor and the concrete proof
of the dual-handler property. It holds a `DiscoveryState` (the hub's own endpoint
coordinate string, its velo instance id, and the list of registered plugin
prefixes) and answers a `DiscoveryRequest { client }` with a `DiscoveryReply
{ hub_instance, endpoint, plugins, greeting }`.

One function, `handle_discovery(&DiscoveryState, DiscoveryRequest) ->
DiscoveryReply`, is the single source of truth. Both surfaces call it:

- **velo**: a unary handler on `HUB_DISCOVERY` decodes the request with
  `rmp_serde` (raw payload, matching the cellular numeric-fidelity convention),
  calls `handle_discovery`, and returns the `rmp`-encoded reply.
- **HTTP**: `POST /discovery/hello` decodes a JSON `DiscoveryRequest`, calls the
  same `handle_discovery`, and returns the JSON `DiscoveryReply`.

A client reaches the hub by endpoint alone — `connect_controller(&client_velo,
"tcp://HOST:PORT")` performs velo's `_hello` bootstrap and yields the hub's
`PeerInfo` — then sends the discovery unary. The equivalent HTTP client POSTs to
`/discovery/hello`. The test asserts the two decoded `DiscoveryReply` values are
identical, demonstrating that the HTTP and velo surfaces are the same handler.

This is the anchor role the cellular controller plays today (a cell learns the
controller's identity by dialing one injected endpoint); the discovery plugin
generalizes it — the hub is the endpoint a client dials, and discovery is the
first thing it can ask.

### Discovery matrix (CLI / slurm / k8s)

| Topology | Hub placement | Endpoint a client dials |
|---|---|---|
| CLI single-host | in-process in the primary rank | `tcp://127.0.0.1:PORT` (or `uds://…`) |
| slurm multi-node | in-process in rank 0 | `tcp://<rank0-host>:PORT` (operator/launcher-injected) |
| k8s | its own service (or the controller pod) | `tcp://<pod>.<svc>.<ns>.svc.cluster.local:PORT` |

The coordinate stays a `tcp://HOST:PORT` (or `uds://PATH`) string parsed by the
shared `parse_endpoint`, matching the operator injection already used for the
controller (`CELL_CONTROLLER_ADDR_ENV`, `controller_dns_name` in
`src/aiperf/kubernetes/`).

### The `/artifact` plugin (velo artifact plane fold-in)

`ArtifactHubPlugin` (prefix `/artifact`, gated on the `engine` feature) re-homes the
velo artifact-streaming plane onto the hub. Its `register_velo_handlers` installs the
exact OPEN/CLOSE/DONE streaming handlers via
`engine::artifact_stream_velo::ArtifactVeloReceiver::register` — the per-file
`velo::StreamAnchor` consumers and the streaming-zstd bounded-memory machinery are
reused verbatim; only the mount point moves onto the hub velo instance. The bound
`ArtifactVeloReceiver` (which owns the cell-completion barrier) is captured into a
take-once slot (`ReceiverSlot`) the bootstrap owns after `Hub::register` returns, so
the controller still awaits `wait_for_cells`. The HTTP surface is a dual-surface
diagnostic: `GET /artifact/allowed` returns the same fail-closed allowlist the velo
OPEN handler enforces, from the same plugin state. The bulk byte movement stays on the
velo stream primitive (ordered + backpressured); the streaming OPEN/CLOSE/DONE protocol
has no faithful plain-axum mirror and is not duplicated. `engine::artifact_shipping`
(the raw-hyper HTTP `:9600` plane) and the standalone `ArtifactVeloReceiver` are
retained unchanged — the fold-in is additive, not a replacement.

### The cell↔controller plugin (control plane fold-in)

`CellControllerHubPlugin` (prefix `/cell`) re-homes the register / heartbeat /
partition / store-partition velo handlers onto the hub. Its `register_velo_handlers`
calls `cellular::transport::velo_transport::VeloControllerTransport::bind_controller`
on the hub velo — the exact handlers, unchanged — making a `Hub` the connect anchor the
standalone controller is today (the `:9500` role). Because `VeloControllerTransport`
carries a live `ControllerTransport::recv` stream the controller must own, the bound
transport is captured into a take-once slot (`TransportSlot`) the bootstrap takes back
out after registration. Its velo surface is inherently peer-registration + streaming
coordination with no faithful plain-HTTP mirror, so the HTTP surface is a diagnostic
`GET /cell/status`; the full protocol stays on velo.

### The `/dataset` plugin (dataset fan-out plane fold-in)

`DatasetHubPlugin` (prefix `/dataset`) re-homes the velo dataset fan-out data plane onto
the hub. Its `register_velo_handlers` installs the exact `aiperf.dataset.subscribe` /
`aiperf.dataset.chunk` handlers via `cellular::transport::dataset_velo::DatasetServer::
bind`, serving the same `DatasetPublisher` the bootstrap fills and finalizes — cells
subscribe over the one hub anchor and build their owned index unchanged; only the mount
point moves. The bound `DatasetServer` is retained inside the plugin, but the handlers
and their per-cell pump tasks ride the hub velo instance the `HubServer` holds, so they
survive the plugin's drop at `Hub::serve`. The HTTP surface is a dual-surface diagnostic:
`GET /dataset/status` reports the two handler names plus the publisher's live
`chunk_count`, so the same publisher state the velo subscribe handler replays is
observable over HTTP. The bulk fan-out (replay + live chunks) stays on velo; the broadcast
protocol has no faithful plain-axum mirror and is not duplicated.

### The `/phaser` plugin (phaser control plane fold-in)

`PhaserHubPlugin` (prefix `/phaser`) re-homes the velo monotonic-phaser control plane onto
the hub. Its `register_velo_handlers` installs the exact `aiperf.phaser.subscribe` /
`aiperf.phaser.event` handlers via `cellular::transport::phaser_velo::PhaserServer::bind`,
serving the same `Phaser` the bootstrap `advance`s — cells subscribe over the hub anchor
and observe replay-then-live generations unchanged. As with `/dataset`, the bound
`PhaserServer` is retained inside the plugin while the handlers and pump tasks ride the hub
velo the `HubServer` holds. `GET /phaser/status` is the dual-surface diagnostic: the two
handler names plus the phaser's live `current_generation`, the same state the velo
subscribe handler replays.

### Hub bootstrap wiring and the `AIPERF_CELLULAR_HUB` toggle

`engine::cellular_controller` gates the hub path on `AIPERF_CELLULAR_HUB`
(`1`/`true`/`on`/`yes`; default **off**). When off, behavior is byte-identical to the
standalone planes: the `VeloControllerTransport` binds directly on the control-plane
velo and the velo artifact receiver (when per-record artifacts ride velo) registers
directly. When on, `build_cellular_hub` stands up ONE `Hub` over the same control-plane
velo instance, mounting the cell↔controller plugin, the `/artifact` plugin (only when
`http_shipping && velo_artifacts`), the `/phaser` plugin (only when `AIPERF_CELL_PHASER_
START` is set), the `/dataset` plugin (only when `AIPERF_CELL_DATASET_FANOUT` is set), and
the discovery plugin (advertising the mounted prefixes and the hub's dial-able endpoint),
then serves the co-bound axum diagnostic surface (`AIPERF_CELLULAR_HUB_HTTP_BIND`, default
loopback `:0`). In hub mode the standalone `PhaserServer::bind` / `DatasetServer::bind`
calls are skipped (the plugins bind the identical servers on the hub velo); the phaser is
still `advance`d and the publisher still filled + finalized by the bootstrap, unchanged.
The captured transport + artifact receiver flow back into the unchanged collect/barrier/
merge loop, and the served `HubServer` is held for the run. With the phaser and dataset
planes mounted, a hub-mode run is a complete replacement of the standalone control/data
planes on the one anchor. Cells reach the hub by the identical `tcp://HOST:PORT` velo
coordinate either way — the hub IS the connect anchor the controller already is, so no
cell-side change is needed. The `test_cellular_hub_mode_matches_default_velo_path` e2e
proves a hub-mode 3-cell velo run is wire- and data-equivalent (byte-identical
`inputs.json`, identical records/raw/outputs row sets, per-cell velo observables) to the
default standalone velo path, and `test_cellular_hub_mode_dataset_fanout_and_phaser_
matches_standalone` proves the same equivalence with the dataset fan-out + phaser planes
active on both anchors.

## Future requirements

The vertical slice, all three original fold-ins, and the dataset fan-out + phaser control
planes are delivered — under `AIPERF_CELLULAR_HUB` every cellular control/data plane rides
the one hub anchor. Remaining additive work (not blocking):

- **Retire the raw-hyper `:9600` artifact server.** The velo artifact plane is folded
  into the hub, but the HTTP-transport artifact path (`engine::artifact_shipping`) still
  binds its own server when `http_upload` is selected. Folding those upload/dataset
  routes onto the hub's axum surface would retire the second port entirely.

## Source anchors

- `rust/runtime/src/hub/mod.rs` — the `Hub` host, `HubServer` handle, and module
  docs; re-exports the plugin and discovery types.
- `rust/runtime/src/hub/plugin.rs` — the `HubPlugin` trait and `HubError`.
- `rust/runtime/src/hub/discovery.rs` — `DiscoveryPlugin`, `DiscoveryState`,
  `DiscoveryRequest`/`DiscoveryReply`, `handle_discovery`, and the dual-surface
  tests.
- `rust/runtime/src/hub/artifact.rs` — `ArtifactHubPlugin` (prefix `/artifact`,
  `engine`-gated), the `ReceiverSlot` capture, and the hub-anchor streaming test.
- `rust/runtime/src/hub/cell_controller.rs` — `CellControllerHubPlugin` (prefix
  `/cell`), the `TransportSlot` capture, and the hub-anchor register test.
- `rust/runtime/src/hub/dataset.rs` — `DatasetHubPlugin` (prefix `/dataset`), wrapping
  `DatasetServer`, its `GET /dataset/status` diagnostic, and the hub-anchor fan-out test.
- `rust/runtime/src/hub/phaser.rs` — `PhaserHubPlugin` (prefix `/phaser`), wrapping
  `PhaserServer`, its `GET /phaser/status` diagnostic, and the hub-anchor phaser test.
- `rust/runtime/src/engine/cellular_controller.rs` — the `AIPERF_CELLULAR_HUB`
  toggle (`CELLULAR_HUB_ENV`), `build_cellular_hub` (mounting the cell↔controller,
  `/artifact`, `/phaser`, `/dataset`, and discovery plugins), and the hub-mode bootstrap
  round-trip test.
- `rust/e2e-tests/tests/test_cellular_velo_shipping.rs` —
  `test_cellular_hub_mode_matches_default_velo_path`, the hub-vs-default parity e2e, and
  `test_cellular_hub_mode_dataset_fanout_and_phaser_matches_standalone`, the complete
  hub-mode (dataset fan-out + phaser) parity e2e.
- `rust/runtime/src/cellular/transport/connect.rs` — reused `build_velo`,
  `BindSpec`, `parse_endpoint`, `connect_controller`.
- `rust/runtime/src/engine/artifact_shipping.rs` — the HTTP+zstd artifact plane
  the hub folds in later.
- `rust/runtime/src/extensions/mod.rs` — the `AIPerfExtension` transactional
  registry the plugin registration mirrors.
