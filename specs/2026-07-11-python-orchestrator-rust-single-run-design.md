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
| HTTP, TLS/UDS/h2c, request bodies, SSE, usage, raw exchanges | Rust | `aiperf-transport-http` and `aiperf-endpoints` |
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

## Addendum — 2026-07-11 (the runner is the only Rust product executable)

The process boundary described above is now structural. The `aiperf` Rust
package is a library only: its `[[bin]]` target, `src/main.rs`, Clap schema,
console logger, and native-CLI acceptance suites are deleted. The human-facing
entry point is the Python `aiperf` command, and the only native executable it
may launch for an AIPerf benchmark is the strict `aiperf-runner` child. Direct
`cargo run -p aiperf` commands are intentionally invalid.

This removes a second configuration and orchestration surface; it does not move
hot-path work into Python. The runner still owns online scheduled execution,
all four adaptive actuators, datasets and endpoint dialects, static Python-
evaluated accuracy, telemetry, request artifacts, metrics, and native-v2 output.
The `aiperf` library retains the underlying runtime implementations and tests.
Human-readable tables, accuracy CSV, plots, and other presentation/export work
belong to the Python parent after it reads Rust-owned results.

Two formerly CLI-reachable library families are not yet product-reachable
through runner protocol v1: stateful agentic evaluation and feature-gated
Dynamo offline co-simulation. Their Rust implementations and focused library
tests remain, but the removed Harbor/BrowserGym/MCPMark live CLI canaries and
the exhaustive offline CLI suite no longer prove an end-user path. They must be
reintroduced as runner requests and runner subprocess tests before the product
can claim those modes. The Python `aiperf dynosim` facade remains a separate
canonical Dynamo-owned product path and does not make the AIPerf runner offline.

This addendum supersedes every older reference in the design record to a native
`aiperf` CLI, its flags, `src/main.rs`, or `CARGO_BIN_EXE_aiperf`. The runtime,
transport, scheduling, metrics, evaluator, and backend algorithm decisions in
those records are unchanged.

## Addendum — 2026-07-11 (protocol-v2 authored projection and complete runner reachability)

`2026-07-11-aiperf-runner-only-execution-surface-design.md` is now authoritative
for making every built native backend/workload product-reachable through the only
native executable. It defines the common v2 operation envelope, online/offline
backend selection, scheduled/Graph/static-accuracy/agentic workload selection,
feature-bearing runner distributions, capabilities, preparation, reports, and
subprocess gates. The endpoint-specific companion is
`2026-07-11-aiperf-runner-owned-endpoint-registry-design.md`.

The protocol-v1 process sequence in this spec remains code truth only for the
compatibility path. Protocol v2 changes the permanent sequence to:

1. Python performs structural Config-v2 validation and outer-loop expansion.
2. Python selects and verifies one exact `RunnerInstallation`.
3. Python projects a side-effect-free authored request without consuming
   `BenchmarkRun.resolved`.
4. The runner performs static validation and reports deferred checks.
5. On execution, Rust resolves datasets/tokenizers/endpoints/backends and completes
   deferred validation.
6. Only then are run artifacts, supervised workers, scheduling, and traffic started.
7. Python consumes Rust-owned results for outer-loop decisions and presentation.

Accordingly, the earlier ownership row assigning per-run artifact creation,
tokenizer localization, and public-dataset resolution permanently to Python is
superseded where Rust already owns those capabilities. Python may select an
artifact target and author dataset/tokenizer policy, but protocol-v2 preparation
belongs to Rust. Python-only presentation, outer-loop, user-template, or external
library work that Rust does not implement remains Python-owned and must run after
the appropriate validation gate.

Stateful agentic, online Graph-IR, and Dynamo offline are no longer left as
unbounded “future runner additions”: the runner-only execution-surface spec owns
their explicit migration increments and acceptance matrix. Until each pair is
advertised and proven by the exact runner, it remains library-built but not
product-reachable.

## Addendum — 2026-07-12 (native gRPC authored selection)

Config v2 now accepts explicit `grpc://` and `grpcs://` targets only when
`benchmark.backend.type` is `online_grpc`. It rejects HTTP/gRPC backend
mismatches, mixed gRPC schemes, and the legacy Python
`endpoint.transport` selector. The authored projector preserves
`online_grpc`, exact-image capability preflight selects the registered
`online_grpc + scheduled` pair, and protocol-v1 projection fails rather than
resolving or falling back.

`online_http + scheduled` is also registered through v2 so HTTP-only KServe
dialects such as `kserve_v1_predict` use prepared open-registry execution.
Runner subprocess tests prove both paths and native-v2 reporting. This
supersedes this spec's earlier “authored v2 static validation only” status for
those registered pairs; the compatibility sequence remains only for exact
pairs not yet advertised by the selected runner.

## Addendum — 2026-07-12 (protocol-v2-only product execution)

The Python product executor no longer selects between protocol generations.
`RustSubprocessExecutor` always projects exactly one authored protocol-v2
request, binds it to the selected runner's executable-content identity, and
requires that exact image to advertise the requested backend/workload pair.
An absent pair is a preflight error. Python does not invoke the resolver chain,
construct a protocol-v1 request, or reinterpret the run through a fallback.
Runner discovery consequently requires protocol v2 and accepts a v2-only
runner image. The Rust protocol-v1 decoder may remain as an isolated
compatibility surface, but it is not reachable from the canonical Python
executor.

The pair adapter is the single preparation boundary. Authored backend and
workload objects remain factory-owned until their registered pair strictly
validates and prepares a typed harness; no request is serialized or converted
through a second protocol shape. In particular, `dag_jsonl` remains authored
graph input until the selected `GraphInputAdapter` parses it once and returns
canonical `GraphTracePlan`s plus one frozen segment store. It never passes
through Python dataset resolution or a linear Rust `Dataset` intermediate.

Explicit open workloads also own their phase stop semantics. Python retains
the generic request/duration/session requirement when it implicitly selects a
built-in workload, but does not force an extension workload to author an inert
second scheduling contract before its Rust factory can validate the phase.
This is required for the harness-owned agentic lifecycle.

The base runner advertises direct protocol-v2 execution for online HTTP
scheduled, graph, static-accuracy, and agentic workloads plus native gRPC
scheduled execution. A runner built with `dynosim` additionally
advertises scheduled and graph execution over the in-process simulator. Python
Config-v2 subprocess proofs cover every one of those canonical mode families,
including exact unsupported-pair rejection with no legacy resolution.

This addendum supersedes this spec's protocol-v1 process sequence, its
conditional v2 selection/fallback language, and the final sentence of the
native-gRPC addendum that retained compatibility execution for unadvertised
pairs. Unadvertised pairs now fail closed; they do not select another protocol.
