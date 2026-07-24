<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf architecture

## Purpose

AIPerf is a load generator and measurement front end for inference servers. This
record states the whole-system architecture: the process model, the crate
topology, and the orthogonal seams every subsystem composes over. Subsystem
contracts live in their own records; this file links to them rather than
duplicating their detail.

## Built

### Process model

The native product entry point is the single `aiperf` binary from crate
`aiperf-cli`. It is both the public CLI and the execution engine. `aiperf
profile` resolves Config v2, projects a protocol-v2 benchmark request, and
re-executes the same binary in an internal `--execute` mode over stdio; that
child process is the sole Rust composition root for one run. Internal `--cell`
and `--aggregator` modes support cellular execution and are intercepted before
clap parsing.

Native commands include `profile`, `config`, `controller`, `cell`,
`aggregator`, `results-sidecar`, `analyze-trace`, `chat`, `validate`,
`speed-bench-report`, and `synthesize`. Commands not owned natively dispatch to
`aiperf.entrypoint.main`: `pyo3-embed` builds invoke it in-process, lean builds
invoke `python -m aiperf`.

`aiperf-mock-server` is a separately launched benchmark and test target; it is
not supervised by a profile run. See [mock-server.md](mock-server.md).

### Three orthogonal axes

The runtime separates three independent concerns so real, mock, and offline
execution are dependency injection rather than code paths:

- **Time** — `aiperf_runtime::clock::Clock`. `RealClock` supplies wall time;
  `SimClock` supplies integer-nanosecond virtual time with deterministic
  `(at_ns, seq_no)` ordering. Execution selects the real-reactor or simulation
  driver through `Clock::is_virtual()`. See [execution-model.md](execution-model.md)
  and [offline-cosimulation.md](offline-cosimulation.md).
- **Transport** — where a request goes (HTTP, gRPC, offline co-simulation,
  dry-run), realized as a `RequestSink<R>` over a `Dispatchable` request. See
  [http-transport.md](http-transport.md) and [grpc-transport.md](grpc-transport.md).
- **Workload** — what requests fire and in what pattern (request-rate,
  concurrency, user-centric, fixed-schedule, graph). See
  [scheduling.md](scheduling.md) and [graph-runtime.md](graph-runtime.md).

### Dispatch and observation seam

`aiperf_runtime::dispatch` is the transport-neutral nucleus: `Dispatchable`,
`RequestSink<R>`, `RequestObserver`, `ObservedUsage`, endpoint observations,
`TraceCollector`, and `CollectorObserver`. A `RequestSink<R>` drives one request
to terminal and emits arrival, admission, token, classified-token,
batched-output-token, usage, endpoint-metric, and terminal events through a
`RequestObserver`. TTFT is the first token observation; there is no separate
first-token event. `RequestObserver` has no `Send`/`Sync` supertrait, so each
thread-per-core worker owns a local `Rc<RefCell<_>>` observer graph. HTTP, gRPC,
mock HTTP, and offline co-simulation all feed this one seam.

### Registration

`AIPerfExtension` registers implementations in `AIPerfRegistry` at startup;
registration is transactional and duplicate identifiers are rejected. Capability
categories (endpoints, datasets, samplers, transports, workloads, exporters,
actuators) are frozen in the application at bootstrap; unknown identifiers fail
closed. Structurally selected seams (`Clock`, `RequestSink<R>`,
`RequestObserver`, `SegmentStore`, graph sinks, accuracy evaluators) are
constructor-injected, not registry entries. See
[extension-registry.md](extension-registry.md).

### Execution model

The native execution model is thread-per-core. `workers == 1` uses one
co-located worker sink on the coordinator's current-thread runtime and
`LocalSet`; `workers > 1` tiles that same sink across self-contained sub-cells on
OS threads. HTTP and gRPC share response reduction (`transport::reduce`) and
worker measurement (`transport::measure`); a transport contributes only wire
decode and terminal mapping. Metrics accumulate per worker and merge at a
boundary. Cellular execution (`--cells N` or `runtime.cells`) partitions work
across cell processes and merges records or folded metric stores. See
[execution-model.md](execution-model.md) and [cellular.md](cellular.md).

### Measurement and output

The metrics plane computes record, aggregate, derived, phase-window, and sweep
metrics; exact mode retains records, sketch mode uses mergeable t-digests. The
exporter plane writes configured JSON, CSV, Parquet, console, timeslice,
server-metrics, accuracy, OTLP, MLflow, and W&B outputs. Side-channel
measurement covers GPU telemetry, server metrics, and network latency. See
[metrics.md](metrics.md), [accuracy.md](accuracy.md), [telemetry.md](telemetry.md),
and [exporters.md](exporters.md).

## Source anchors

- Workspace manifest: `Cargo.toml`; crate topology in [repository-layout.md](repository-layout.md).
- Process entry and command routing: `rust/cli/src/{main.rs,dispatch.rs,execute.rs,execute_mode.rs}`.
- Neutral seam: `rust/runtime/src/dispatch/`.
- Runtime composition: `rust/runtime/src/lib.rs` and `rust/runtime/src/engine/`.
- Module map: `docs/module-organization.md`.
