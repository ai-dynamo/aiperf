<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cellular execution

## Purpose

Cellular execution partitions one run across multiple cell processes and merges
their records or folded metric stores, so offered load scales past a single
process while measurement fidelity is preserved. It is selected with `--cells N`
or `runtime.cells`; `cells = 1` (the default) is byte-unchanged. The front end
puts `runtime.cells` in the execute envelope and launches one runner as the
controller.

## Built

### Seams

`aiperf_runtime::cellular` provides the partitioning and measurement seams:

- **Issuance** — `IssuanceAuthority` with a `Direct` impl and a
  `CellularAutonomousIssuer`, threaded byte-unchanged through `RunCapture::finish`.
- **Partition** — `CellPartition`, whose shipping impl is the round-robin
  `ModuloCellPartition` (`owns(i) ≡ i % cell_count == cell_id`; a cell's `n`-th
  owned instance is `n·cell_count + cell_id`, so a cell-local counter reconstructs
  a dense global ordinal). This is the same primitive the sub-cell thread grid
  uses (see [execution-model.md](execution-model.md)).
- **Records** — a serializable `RecordsShardPartition` (byte-exact global-order
  re-ingest merge) and `ColumnStorePartition`.
- **Heartbeat** — a `MetricsHeartbeat` t-digest sketch and an env-gated live lane
  (`cellular::{sketch,heartbeat}` plus the runner `heartbeat_lane`).

### Multi-process topology

A `CellTransport` framed-MessagePack seam (`cellular::transport`) plus the
`aiperf --cell` / controller topology (`engine::cellular_cell` /
`engine::cellular_controller`) run cells as separate processes: the issuer and
partition are env-selected, records ship over the transport and merge in global
order, cross-cell heartbeats aggregate, and a crashed cell aborts the run. Each
cell stamps the single-cell absolute slot (`phase_ordinal_base + within·cells +
cell_id`, per-phase-reset sampler). A warmup+profiling `cells > 1` run over a
varying-ISL dataset is byte-identical to a 1-cell run on the full metric
distributions.

Graph-mode cellular is built: graph programs partition at the trace level
(`PartitionedGraphTraceSource`, cell `k` owning interleaved global session
ordinals); each cell ships its graph records and the controller
concatenation-merges by `cell_id`, re-numbering local `request_index` densely.

### Cross-host transport (velo)

Behind the `cellular` Cargo feature, which links velo, a velo-backed
`CellClient`/`ControllerTransport` provides cross-host transport with
zero-discovery connect-by-endpoint (cells reach the controller from one
operator-hardcoded DNS:port). A `CellLauncher` seam splits
local-subprocess-over-velo from k8s-pod launch. Without the feature, `cells = 1`
is byte-unchanged and `cells > 1` fails closed; the loopback transport serves
single-host runs.

### Cross-host session trust

Deployment addressing and application identity are separate. The configured
controller coordinate and Velo `_hello` supply an unauthenticated routing fact,
not application admission. `_hello` can install a transport peer before AIPerf
has accepted that peer; AIPerf neither authenticates `_hello` nor claims to
prevent that Velo peer-table insertion. Cell-originated admission handlers
decode no request DTO and install no AIPerf route or state until their
authenticated frame is accepted.

Each process receives fixed binary role material for one run and one role. A
deployment controller and each remote cell read separate regular, no-follow
files with exact `0600` permissions. For same-host children, the controller
mints the roster in memory, drains each non-cloneable role entry once, and sends
it over a launcher-owned pipe at an inherited file descriptor; environment and
argv contain only the non-secret descriptor number, never key bytes or bootstrap
JSON. Acquisition atomically installs one opaque, process-owned
`CellSecurityContext`; later Velo registration and control clients borrow that
same context rather than rereading or cloning private role material. The
cross-host HTTP artifact uploader instead uses the exact bearer authorized by
that registration over the pinned-TLS artifact channel; it does not borrow the
context or use `AuthenticatedFrame` for each upload.

Cell registration signs the run nonce, exact role, cell peer bytes, artifact
capability digest, and the controller binding. That binding covers the exact
Velo instance bytes, the messenger worker-address bytes published by `_hello`,
and the resolved TCP/UDS dial target. The controller's reply attestation covers
the same binding, the exact encoded registration frame, and the exact reply
payload. Registration and reply validation therefore bind the application
session to the connection that was actually dialed without treating `_hello`
itself as authenticated.

After registration, every controller-side Velo admission handler accepts a
bounded `AuthenticatedFrame`. Its signature transcript binds protocol version,
run nonce, role, purpose, process session nonce, per-purpose sequence, peer-info
digest, and payload digest. Controller-owned fixed role slots keep a 64-sequence
replay window per purpose plus fixed rejection counters; malformed, oversized,
wrong-role, wrong-session, invalid-signature, and replay traffic is rejected as
one bounded `AdmissionRejected` class without per-frame logging.
Controller-to-cell dataset/phaser pushes follow routes established by authenticated
subscriptions, but their individual payloads are not per-push authenticated
frames. Adding that direction to the frame protocol is a separate follow-up.
These controls protect their stated application boundaries, not transport
confidentiality.

Registration is transactional across the application ledger, artifact bearer
authorization, reverse-route installation, and start barrier. Planning and
reply attestation finish before the last fallible route installation; commit is
then infallible and exact retries return the cached reply bytes. RAII rollback
returns incomplete slots and artifact reservations to vacant without publishing
partial admission state. Hierarchical aggregation remains refused before source
acquisition, scratch creation, transport bind, or launch because controller-
planned role security for every tree edge is not implemented.

### Fidelity guards

Byte-parity is exact only for a seeded `concurrency` phase with no approximating
knobs. The guards otherwise:

- Fail closed on non-request-bounded phase types (only `concurrency`, `poisson`,
  `gamma`, `constant`; `fixed_schedule`/`user_centric` would N×-replay the trace),
  on `duration`/`adaptive_scale` bounds, and on budgets or caps below
  `cell_count`. A `sessions` budget is not blanket-refused: it is allowed on the
  exact-fold merge path and rejected only on the retain path, where a multi-turn
  conversation's per-turn dispatch ordinal diverges from the sampler's
  per-conversation draw index.
- Allow-with-warning (aggregate-equivalent, not byte-identical, warned via
  `warn_cellular_approximations`): a seedless run auto-derives one shared seed from
  the run identity; multiple endpoint URLs are round-robined cell-locally;
  concurrency/prefill/rate ramps are allowed (each cell ramps to its sliced
  target); and `rate` pacing and post-send cancellation are approximate. Static
  rate/concurrency/prefill caps are sliced per cell so aggregate offered load
  matches a 1-cell run.

The merged report reproduces 1-cell metric data (profiling and warmup) but not the
coordinator `finalize_run` provenance or the grouped error-message list.

## Future requirements

- A dataset data plane: producer-owned SPMC/MPMC add-only broadcast with
  per-consumer replay-on-attach. Velo provides only SPSC and MPSC consumer-owned
  anchors with no replay, so producer-owned fan-out must be built.
- A monotonic phaser control plane: START generalized to `{generation, transition}`
  broadcast (one-way vs barrier, cyclic-by-monotonic-counter) over the built
  issuer/`SlotPool` seams.
- A per-request dispatch state machine (`Unknown → Indexed → InFlight → Done`) with
  dedup and bounded-await-then-counted misses.
- A scale-adaptive fidelity ladder (exact/byte-parity default → bounded sketch →
  external streaming sink). Hierarchical tree-merge is unavailable and refused
  before cellular startup; counts/sums/rates remain exact in supported modes.
- Additional cross-host launcher integrations beyond Kubernetes and SLURM,
  gRPC/offline cell wiring, and graph weighted-sampling plus static-node
  `request_limit` partition.

## Source anchors

- `rust/runtime/src/cellular/` (`issuance.rs`, `partition.rs`, `shard.rs`,
  `heartbeat.rs`, `sketch.rs`, `transport/`, and the forward-plane `broadcast.rs`,
  `phaser.rs`, `dispatch_state.rs`, `dataset_session.rs`).
- `rust/runtime/src/engine/{cellular_bootstrap.rs,cellular_registration.rs,cellular_cell.rs,cellular_controller.rs,cellular_aggregator.rs,cell_launcher.rs,heartbeat_lane.rs,record_lane.rs}`.
- `rust/e2e-tests/tests/{test_cellular.rs,test_graph_cellular.rs,test_grpc_cellular.rs,test_cellular_multiturn.rs}`.
