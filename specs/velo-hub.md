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

## Future requirements

This pass delivers a reviewable vertical slice: the plugin trait, the hub host,
the discovery plugin, and the dual-surface tests. The following are explicitly
**deferred** and are additive — the existing controller and artifact planes are
untouched and keep working as-is:

- **Fold the artifact plane in as a plugin.** `engine::artifact_shipping`'s
  upload/dataset routes become a `HubPlugin` (prefix `/artifact`) mounted on the
  hub's axum service, retiring the separate `:9600` server. Its velo surface (if
  any) registers alongside. The streaming-zstd bounded-memory machinery is reused
  verbatim; only the mount point moves.
- **Fold the cell↔controller plane in as a plugin.** The register / heartbeat /
  partition / store-partition handlers (`cellular::transport::velo_transport`)
  become a `HubPlugin` that registers exactly those velo handlers, making the hub
  the connect anchor the controller is today (the `:9500` role).
- **Compat/version negotiation** on `register` (beyond duplicate-prefix
  rejection), so a plugin can declare a required hub ABI.
- **Wiring the hub into the engine bootstrap** (`engine::cellular_controller`) so
  a real run stands up a hub instead of the two standalone servers.

## Source anchors

- `rust/runtime/src/hub/mod.rs` — the `Hub` host, `HubServer` handle, and module
  docs; re-exports the plugin and discovery types.
- `rust/runtime/src/hub/plugin.rs` — the `HubPlugin` trait and `HubError`.
- `rust/runtime/src/hub/discovery.rs` — `DiscoveryPlugin`, `DiscoveryState`,
  `DiscoveryRequest`/`DiscoveryReply`, `handle_discovery`, and the dual-surface
  tests.
- `rust/runtime/src/cellular/transport/connect.rs` — reused `build_velo`,
  `BindSpec`, `parse_endpoint`, `connect_controller`.
- `rust/runtime/src/engine/artifact_shipping.rs` — the HTTP+zstd artifact plane
  the hub folds in later.
- `rust/runtime/src/extensions/mod.rs` — the `AIPerfExtension` transactional
  registry the plugin registration mirrors.
