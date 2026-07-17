<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `specs/` — AIPerf design records

This folder is the design record for AIPerf. Each spec describes one subsystem or
seam: what it is, the contract it holds, and where the code lives. A spec states
current built behavior; a `## Future requirements` section, where present, states
explicitly planned but unbuilt work. The code in `rust/` is authoritative — when a
spec and the code disagree, fix the spec.

Every spec follows the same shape: `## Purpose`, `## Built`, an optional
`## Future requirements`, and `## Source anchors` that point at the files that
realize it.

Start with [architecture.md](architecture.md) for the whole-system picture, then
read the record for the subsystem you are touching.

## Index

### Whole system

| Spec | Purpose |
|---|---|
| [architecture.md](architecture.md) | Process model, crate topology, and the three orthogonal seams (time, transport, workload) every subsystem composes over. |
| [repository-layout.md](repository-layout.md) | Cargo workspace topology, package identity, and the naming rules for any new package, enforced by `tools/check_crate_layout.py`. |
| [extension-registry.md](extension-registry.md) | Static link-time extensibility: the `AIPerfRegistry`/`AIPerfExtension` composition seam, its capability categories, and the frozen bootstrap object graph. |
| [runner-protocol.md](runner-protocol.md) | The Config-v2 front end ↔ execution boundary: the protocol-v2 stdio envelope, the `BenchmarkRun` vocabulary, path selection, and `--capabilities` discovery. |

### Execution and scheduling

| Spec | Purpose |
|---|---|
| [execution-model.md](execution-model.md) | The single thread-per-core hot path, the two-trait transport seam, worker-local accumulation, and the shared reduce/measure layers. |
| [flatgraph-fast-path.md](flatgraph-fast-path.md) | Built `FlatGraphActor` fast path: eligible local and worker-backed production graph placement routes one-node/no-fan-in traces through the shared sink without the general graph context, proven byte-identical to the general executor through the real `aiperf` binary; later scheduled-workload and multi-node work remain future. |
| [scheduling.md](scheduling.md) | The scheduled workload shapes (request-rate, concurrency, user-centric, fixed-schedule) over one `Clock`-backed runtime, and how each partitions across sub-cells. |
| [phase-orchestration.md](phase-orchestration.md) | One `Clock`-native lifecycle for warmup→profiling phases: the escalation ladder, cancellation latch, and the shared seam scheduled and graph runs both use. |
| [ancillary-timing.md](ancillary-timing.md) | The three knobs that ride on a running phase: ramping, seeded request cancellation, and sticky round-robin URL selection. |
| [adaptive-scale.md](adaptive-scale.md) | The closed-loop SLA controller (`ramp_until_fail`) layered over a running load phase, its actuators, and its schema-v2 artifacts. |
| [cellular.md](cellular.md) | Partitioning one run across cell processes and merging records or folded metric stores, the multi-process and velo cross-host topologies, and the fidelity guards. |

### Transports

| Spec | Purpose |
|---|---|
| [http-transport.md](http-transport.md) | The Clock-injected hyper HTTP stack: wire/protocol support, SSE streaming, endpoint binding, and post-send cancellation. |
| [grpc-transport.md](grpc-transport.md) | The Clock-injected Tonic gRPC stack: the binding registry, the KServe OIP v2 and Riva families, the protoc-free codec, and the worker-local sink. |
| [websocket-transport.md](websocket-transport.md) | Built reusable content, SSE, bidirectional framing, measurement, placement, and registration prerequisites plus the future Clock-injected WebSocket transport contract. |
| [offline-cosimulation.md](offline-cosimulation.md) | Socket-free Dynamo co-simulation behind the `dynosim` feature: the steppable clocked engine boundary and the observer contract feeding AIPerf's own measurement. |

### Inputs, endpoints, and graph

| Spec | Purpose |
|---|---|
| [dataset.md](dataset.md) | The input-resolution plane: the content-addressed segment store and the loader→compose→store→sampler→materializer pipeline. |
| [endpoint-body-construction.md](endpoint-body-construction.md) | How an endpoint declares its request shape (`format_payload → BodyPlan`) and how the two shared materializers turn segment handles into wire bytes. |
| [endpoints.md](endpoints.md) | The `Endpoint` dialect adapter: the trait, every native dialect, endpoint identity, and the registry consumed by validation and execution. |
| [content-server.md](content-server.md) | The run-owned HTTP delivery sidecar that serves generated media by URL, and its publication seam. |
| [rng.md](rng.md) | The hash-derived randomness substrate: order-independent BLAKE3 stream derivation, generators, and sampling distributions. |
| [graph-runtime.md](graph-runtime.md) | The Graph-IR runtime: deterministic async dataflow, the `dag_jsonl`/`weka_trace`/`dynamo_trace` compilers, and the trajectory-snapshot/warmup-priming subsystem. |

### Measurement and output

| Spec | Purpose |
|---|---|
| [metrics.md](metrics.md) | The IO-free metrics engine: the column-store accumulator, the metric catalog, sweep curves, and the typed report; exact vs sketch modes. |
| [telemetry.md](telemetry.md) | Side-channel measurement: GPU telemetry, server metrics, and network latency, feeding values into the metrics seam. |
| [exporters.md](exporters.md) | The native output plane: the typed report core and the static set of `Exporter` sinks behind one trait. |
| [accuracy.md](accuracy.md) | The Rust dispatch/capture vs pinned-Python grading split, the injected evaluator seam, and sharded capture with a single grade. |

### Targets

| Spec | Purpose |
|---|---|
| [mock-server.md](mock-server.md) | `aiperf-mock-server`: the standalone HTTP/gRPC inference target with deterministic generation, latency and error models, telemetry, and request recording. |
