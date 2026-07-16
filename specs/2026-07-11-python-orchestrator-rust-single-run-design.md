<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python orchestrator / Rust single-run architecture

**Status:** built protocol-v2-only on `ajc/rust`. The original authored-projection
wire and discovery contract are superseded by the BenchmarkRun wire + runner
catalog design (`specs/2026-07-13-benchmarkrun-wire-and-runner-catalog-design.md`),
whose reality is folded in below.

## Decision

Python owns Config v2 and every outer control loop. Rust owns exactly one
fully-resolved benchmark execution. The boundary is a fresh
`aiperf --execute` subprocess — the same `aiperf` binary (crate `aiperf-cli`)
re-execing itself into an internal hidden execution mode intercepted before clap —
with a strict, versioned protocol-v2 JSON request on stdin and one terminal JSON
response on stdout. The one `aiperf` binary is the only Rust executable on the
product path; Python is only on that path when `AIPERF_NATIVE=0`, where the
pure-Python frontend spawns the same `aiperf --execute` child.

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
| Config resolution product (`BenchmarkRun.resolved`): artifact target, tokenizer/public-dataset policy | Python | Config-v2 resolver chain, sent on the wire as `resolved` |
| Per-run dataset/tokenizer/endpoint preparation from that resolution | Rust | `aiperf-cli` frozen registries and pair adapters |
| One run's models, dataset, phases, ramps, cancellation, adaptive policy | Rust | `aiperf-cli::execute` over scheduled runtime traits |
| HTTP, TLS/UDS/h2c, request bodies, SSE, usage, raw exchanges | Rust | `aiperf_runtime::transport_http` and `aiperf_runtime::endpoints` |
| Clock, phase lifecycle, arrivals, slots, TTFT release, stop/drain | Rust | `aiperf_runtime::clock`, `aiperf_runtime::timing`, runner phase runtime |
| Request metrics, sweeps, SLO goodput, timeslices, native-v2 | Rust | `aiperf_runtime::metrics_core` |
| GPU, server Prometheus, network RTT phase sidecars | Rust | native side-channel modules and runner adapters |
| Canonical accuracy prompt/task/grader libraries | Python worker | supervised accuracy worker over stdio |
| OTel SDK and live MLflow implementation | Python worker | native streaming worker using the existing `OTelMetricsResultsProcessor` |
| JSON/CSV/Parquet/console, W&B, MLflow upload, user exporter plugins | Python | canonical `ExporterManager` over Rust-owned `ProfileResults` |
| Auto-plot and cross-run presentation | Python | CLI completion callbacks and aggregate exporters |

Python selects an artifact target and authors dataset/tokenizer/endpoint policy
into the resolution facts; the runner performs the actual protocol-v2
preparation. Python-only presentation, outer-loop, user-template, or external
library work that Rust does not implement remains Python-owned and runs after the
appropriate validation gate.

## Parent/child contract

The request body is exact `BenchmarkRun` JSON. Python sends a thin envelope:

```json
{ "protocol_version": 2, "operation": "validate | execute", "run": { …BenchmarkRun… } }
```

`run` is the outer `BenchmarkRun` (`benchmark_id`, `artifact_dir`, canonical
`cfg`, and `resolved`), not a bespoke projection dialect. `resolved` is the
Config-resolution product Python already computed — a first-class wire field, not
a private cache. The runner does not reinterpret it; frozen factories own the
strict decode of component config during registry validation.
`rust/runtime/src/protocol_v2.rs` is the matching Rust DTO: `RunnerEnvelopeV2` and
`BenchmarkRunWireV2`, every object `deny_unknown_fields`. `aiperf-cli`
advertises `protocol_versions: [2]` only and rejects any non-v2 request as a
protocol-v2 failure envelope; there is no runner protocol v1.

Discovery is a linked-inventory catalog, not a hand-built capability
struct. There is no `--capabilities` argv mode: the catalog is produced
in-process by `aiperf_cli::execute_mode::capabilities_catalog()`, which composes
the stock application and returns the categories and inventories built into that
exact `aiperf` binary. The run is bound to that binary's executable-content
identity and requires the linked image to advertise the requested pair.

Execution path selection is derived, not authored as a separate axis: the runner
chooses scheduled vs graph execution from the dataset format and binds the
transport from `cfg.transport.type`. There is no separate `workload` selection
axis on the wire.

The normal process sequence is:

1. Python loads and validates Config v2 and expands the plan to one
   variation/trial.
2. Python selects and verifies one exact `aiperf` binary against its in-process
   linked capabilities catalog (`aiperf_cli::execute_mode::capabilities_catalog()`).
3. Python resolves artifact, tokenizer, dataset and extension inputs into
   `BenchmarkRun.resolved`.
4. Python writes one protocol-v2 envelope: `{protocol_version: 2, operation,
   run}`. The runner performs static validation and reports deferred checks.
5. On execution, Rust resolves datasets/tokenizers/endpoints from `cfg` +
   `resolved`, completes deferred validation, and constructs the clock,
   transport, policies and phase plans.
6. Rust runs all warmup/profiling traffic and writes `native-v2.json` plus
   requested native record/telemetry artifacts.
7. Rust writes one terminal response and exits.
8. Python validates the terminal/report path, projects native results, invokes
   canonical report/export plugins, and returns metrics to the outer loop.
9. The outer loop decides convergence or proposes the next search coordinate.

No benchmark process is reused between trials. That isolates allocator,
connection-pool, RNG and extension state exactly at the run boundary.

The pair adapter is the single preparation boundary. Component config stays
factory-owned until its registered pair strictly validates and prepares a typed
harness; no request is serialized or converted through a second protocol shape.
In particular, `dag_jsonl`/`weka_trace`/`dynamo_trace` remain authored graph
input until the runner-owned graph-input resolver parses each once and returns
canonical trace plans plus one frozen segment store. Graph input never passes
through Python dataset resolution or a linear Rust `Dataset` intermediate.

Explicit open workloads own their own phase stop semantics: the runner does not
force an extension to author an inert second scheduling contract before its Rust
factory can validate the phase. This is required for the harness-owned agentic
lifecycle.

## Live Python extensions without a Python hot path

OTel and live MLflow require data before run completion, so post-run replay is
not faithful. The built path is a second strict stdio relationship owned by the
Rust child:

```text
Python orchestrator
    -> aiperf --execute
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

## Transport vocabulary and registered pairs

The execution axis is `transport`, not `backend`. The runner reads
`cfg.transport.type` / `cfg.transport.config`. The built transport IDs are
`http`, `grpc`, `dynosim_offline`, and `dynosim_online`:

- `http` — native online HTTP/SSE, including HTTP-only KServe dialects such as
  `kserve_v1_predict`, over prepared open-registry execution.
- `grpc` — native gRPC (KServe OIP + Riva) with `grpc://`/`grpcs://` targets.
  Config v2 rejects HTTP/gRPC mismatches, mixed gRPC schemes, and the legacy
  Python `endpoint.transport` selector.
- `dynosim_offline` / `dynosim_online` — in-process Dynamo replay under a
  virtual or wall clock, split across two transport IDs with no `replay_mode`
  field; available only in a `dynosim`-built runner.

The base runner advertises HTTP scheduled, graph, static-accuracy, and agentic
execution plus native gRPC scheduled execution. Exact deployment-owned evaluator
roots conditionally add the stock `http + evaluation` GSM8K-canary pair. A
`dynosim`-built runner additionally advertises scheduled and graph execution over
the in-process simulator. An unadvertised pair is a preflight error: Python does
not resolve, convert, or fall back to another protocol. The `RustSubprocessExecutor`
projects exactly one protocol-v2 request and never selects between protocol
generations.

## Config-v2 YAML parity

OTLP URL normalization lives on `OTelConfig`, so YAML, CLI flags and
programmatic Config-v2 construction share one rule:

- bare `host[:port]` gains `http://`;
- only HTTP(S) is accepted;
- host and port are validated;
- `/v1/metrics` is appended exactly once.

The CLI converter delegates to the same function. This prevents the previous
split where `--otel-url host:4318` worked but equivalent YAML posted to `/`.

## Evidence

The architecture is pinned at three levels:

- runner subprocess tests: real child process, HTTP/SSE and native gRPC, native
  report, records/raw outputs, Python accuracy, telemetry and adaptive actuators,
  including exact unsupported-pair rejection with no legacy resolution.
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
- A `src/main.rs` or `cargo run -p aiperf-runtime` execution surface inside the
  runtime library. The `aiperf-runtime` Rust package is a library only; the sole
  native product executable is the `aiperf` binary (crate `aiperf-cli`).
- Runner protocol v1: the v1 request dispatch, `execute_v1`/`execute_run*`
  chain, the `RunRequest`/`RunSpec`/`RunTerminal`/`EndpointSpec`/`DatasetSpec`/
  `AccuracySpec` DTOs, the v1 graph-input adapters, and the `Legacy` variants
  are deleted, not dormant.
- A bespoke authored-projection dialect, `expected_distribution_id` pinning,
  `transport`/`workload` `{type, config}` framing, a "no resolved" projection
  rule, and pair-matrix preflight against `supported_pairs`. The wire is
  BenchmarkRun JSON including `resolved`; discovery is the `plugins.yaml`-shaped
  catalog.
- Reimplementing Lighteval, Harbor, OTel, MLflow, report or user-extension
  libraries in Rust merely to avoid a Python dependency.
- Allowing Python extensions to issue inference requests outside Rust's
  transport/clock path.
- Post-run replay presented as live streaming.
- Passing arbitrary Python objects, pickles or an unversioned config dump to
  Rust.
