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

### Fidelity guards

Byte-parity is exact only for a seeded `concurrency` phase with no approximating
knobs. The guards otherwise:

- Fail closed on non-request-bounded phase types (only `concurrency`, `poisson`,
  `gamma`, `constant`; `fixed_schedule`/`user_centric` would N×-replay the trace),
  on `duration`/`sessions`/`adaptive_scale` bounds, on non-HTTP transports
  (gRPC/offline cell wiring is not wired), and on non-synthetic/multi-turn
  datasets and caps below `cell_count`.
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
- Cross-host beyond loopback, gRPC/offline cell wiring, and graph weighted-sampling
  plus static-node `request_limit` partition.
- Kubernetes control-plane isolation: the planned `native-k8s/v1` deployment
  has only controller, cell, and results-sidecar roles. Rust mints fixed binary
  role material into named immutable mounts; the independent operator validates
  reference metadata and never reads Secret bytes. Hierarchy remains refused.

## Source anchors

- `rust/runtime/src/cellular/` (`issuance.rs`, `partition.rs`, `shard.rs`,
  `heartbeat.rs`, `sketch.rs`, `transport/`, and the forward-plane `broadcast.rs`,
  `phaser.rs`, `dispatch_state.rs`, `dataset_session.rs`).
- `rust/runtime/src/engine/{cellular_cell.rs,cellular_controller.rs,cellular_aggregator.rs,cell_launcher.rs,heartbeat_lane.rs,record_lane.rs}`.
- `rust/e2e-tests/tests/{test_cellular.rs,test_graph_cellular.rs,test_grpc_cellular.rs,test_cellular_multiturn.rs}`.
