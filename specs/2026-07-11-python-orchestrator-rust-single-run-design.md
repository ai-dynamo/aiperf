<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python orchestrator / Rust single-run architecture

**Status:** built and canonical on `ajc/rust`.

## Decision

Python owns Config v2 and every outer control loop. Rust owns exactly one
fully-resolved benchmark execution. The boundary is a fresh
`aiperf-runner` subprocess with a strict, versioned JSON request on stdin and
one terminal JSON response on stdout.

This is not a temporary compatibility path. It is the intended architecture:

- Python remains the right home for YAML/Jinja/environment expansion, plugin
  discovery, sweep/search libraries, trial convergence, report generators,
  uploads, plots, and canonical Python evaluation libraries.
- Rust remains the only home for arrival timing, admission, request dispatch,
  HTTP/SSE, cancellation, phase boundaries, per-request measurement, metric
  accumulation, and native results.
- Python libraries needed during a run execute as supervised stdio workers.
  They receive completed facts; they never take ownership of the request hot
  path or recompute authoritative performance measurements.

The old Python controller/service/ZMQ topology and the old Rust mechanical port
are not alternate implementations of this design.

## Concrete ownership

| Concern | Owner | Current implementation |
|---|---|---|
| Config-v2 YAML, CLI overlay, aliases, environment interpolation | Python | `aiperf.config` loader/converter |
| Grid, zip, scenario, QMC, Bayesian and monotonic search | Python | `aiperf.orchestrator` planners |
| Trials, iteration order, cooldown, convergence, confidence, sweep aggregation | Python | `MultiRunOrchestrator` and aggregation/convergence packages |
| Per-run artifact directory, user-file rendering, tokenizer/public-dataset resolution | Python | Config-v2 resolver chain |
| One run's models, dataset, phases, ramps, cancellation, adaptive policy | Rust | `aiperf-runner::execute` over scheduled runtime traits |
| HTTP, TLS/UDS/h2c, request bodies, SSE, usage, raw exchanges | Rust | `aiperf-transport` and `aiperf-endpoints` |
| Clock, phase lifecycle, arrivals, slots, TTFT release, stop/drain | Rust | `aiperf-clock`, `aiperf-timing`, `aiperf::phase_runtime` |
| Request metrics, sweeps, SLO goodput, timeslices, native-v2 | Rust | `aiperf-metrics` |
| GPU, server Prometheus, network RTT phase sidecars | Rust | native telemetry crates and runner adapters |
| Canonical accuracy prompt/task/grader libraries | Python worker | `aiperf.accuracy.worker`, supervised by Rust |
| OTel SDK and live MLflow implementation | Python worker | `native_streaming_worker` using the existing `OTelMetricsResultsProcessor` |
| JSON/CSV/Parquet/console, W&B, MLflow upload, user exporter plugins | Python | canonical `ExporterManager` over Rust-owned `ProfileResults` |
| Auto-plot and cross-run presentation | Python | CLI completion callbacks and aggregate exporters |

## Parent/child contract

`src/aiperf/orchestrator/rust_wire.py` is the sole Config-v2-to-native
projection. It writes every accepted field explicitly; it does not serialize a
raw Pydantic object as an opaque implementation request.

`crates/aiperf-runner/src/protocol.rs` is the matching Rust DTO. Every object
uses `deny_unknown_fields`. Before launching a run, Python calls
`aiperf-runner --capabilities` and verifies:

- runner protocol version;
- native report schema version;
- endpoint, dataset and phase inventories;
- optional phase and run features;
- Python-worker and artifact format inventories.

The normal process sequence is:

1. Python loads and validates Config v2.
2. Python expands the complete plan and chooses one variation/trial.
3. Python resolves artifact, tokenizer, dataset and extension inputs.
4. Python negotiates runner capabilities and writes one protocol-v1 request.
5. Rust constructs the dataset, clock, transport, policies and phase plans.
6. Rust runs all warmup/profiling traffic and writes `native-v2.json` plus
   requested native record/telemetry artifacts.
7. Rust writes one `run_terminal` response and exits.
8. Python validates the terminal/report path, projects native results, invokes
   canonical report/export plugins, and returns metrics to the outer loop.
9. The outer loop decides convergence or proposes the next search coordinate.

No benchmark process is reused between trials. That isolates allocator,
connection-pool, RNG and extension state exactly at the run boundary.

## Live Python extensions without a Python hot path

OTel and live MLflow require data before run completion, so post-run replay is
not faithful. The built path is a second strict stdio relationship owned by the
Rust child:

```text
Python orchestrator
    -> aiperf-runner
        -> Python native_streaming_worker
            -> canonical OTel/MLflow fanout process
```

Rust emits two fact types:

- terminal request records serialized through the same compatibility row
  builder used by persisted JSONL; and
- exact `PhaseStats` start/progress/sending-complete/complete snapshots from
  the injected `Clock` and `PhaseObserver` seam.

The Rust-to-worker queue is bounded, local to the current-thread runtime, and
drop-oldest. Emission is synchronous only through serialization/enqueue; pipe
I/O runs in a detached local pump. A slow collector therefore cannot
backpressure HTTP dispatch. The Python worker validates every line, converts
facts into the canonical `MetricRecordsData` / `CreditPhaseStats` models, and
invokes the existing strategies and SDK fanout unchanged.

Telemetry remains best effort, matching the existing processor contract.
Protocol/configuration errors are diagnostic; missing optional SDKs may disable
only the affected sink. Rust's native report is never derived from or mutated
by the side channel.

## Config-v2 YAML parity

OTLP URL normalization now lives on `OTelConfig`, so YAML, CLI flags and
programmatic Config-v2 construction share one rule:

- bare `host[:port]` gains `http://`;
- only HTTP(S) is accepted;
- host and port are validated;
- `/v1/metrics` is appended exactly once.

The CLI converter delegates to the same function. This prevents the previous
split where `--otel-url host:4318` worked but equivalent YAML posted to `/`.

## Evidence

The architecture is pinned at three levels:

- `crates/aiperf-runner/tests/stdio_e2e.rs`: real child process, HTTP/SSE,
  native report, records/raw outputs, Python accuracy, telemetry and adaptive
  actuators.
- `tests/integration/test_rust_executor_e2e.py`: Config v2 through a fresh Rust
  child, including decoded OTLP protobuf batches received while the child is
  still executing and with persisted record JSONL disabled.
- `tests/integration/test_rust_orchestrator_outer_loops.py`: real HTTP/SSE
  executions for Cartesian grid, zip, scenarios, QMC, both trial orders, every
  convergence mode, and a two-parameter adaptive search. Every cell writes a
  native-v2 report and outer decisions consume Rust-produced samples.

These are process proofs, not mocked executor tests.

## Deleted alternatives

- Keeping Python's multiprocess timing manager, credit protocol or ZMQ bus.
- Reimplementing Lighteval, Harbor, OTel, MLflow, report or user-extension
  libraries in Rust merely to avoid a Python dependency.
- Allowing Python extensions to issue inference requests outside Rust's
  transport/clock path.
- Post-run replay presented as live streaming.
- Passing arbitrary Python objects, pickles or an unversioned config dump to
  Rust.
