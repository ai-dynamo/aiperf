<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python-to-Rust parity gaps

## Purpose

This document consolidates the major confirmed behavior gaps between AIPerf's
Python execution stack and the native Rust stack. It is a migration backlog,
not an argument that Rust should reproduce Python's internal architecture.
Where the implementations deliberately use different designs, the required
parity boundary is the public configuration, request, result, artifact, and
failure contract.

The inventory is current as of 2026-08-26. It is based on executable code and
tests. Design records, plans, tutorials, and architecture prose may explain
intent, but they are not evidence for a behavior claim in this document.

The intended readers are:

- owners deciding whether a Python path can be retired;
- implementers closing native execution gaps;
- reviewers assessing whether a change preserves public behavior;
- test owners building cross-engine acceptance gates;
- downstream consumers depending on AIPerf artifacts.

## How to read the backlog

Priorities mean:

- **P0**: blocks migration or can silently produce invalid benchmark results. A
  gap that fails loudly at parse, validation, or run time does not qualify on
  disruption alone; a gap that changes a reported number on the stock path with
  no error does.
- **P1**: materially changes workload semantics, measurements, artifacts, or
  operational behavior.
- **P2**: affects narrower modes, edge cases, diagnostics, or compatibility,
  including gaps reachable only outside the default dispatch mode, workload
  type, or Cargo feature set. Defaults are read from code: an unset
  `runtime.dispatch` resolves to `global` when `runtime.cells <= 1` and to
  `sharded` when `cells > 1`
  (`parse_dispatch_mode`, `rust/runtime/src/engine/protocol_v2.rs:265-278`),
  unset `runtime.workers` auto-selects the machine's parallelism, and `grpc`,
  `cellular`, `parquet`, and `websocket` are in the default CLI feature set.
- **P3**: intentional native-only capability or low-risk polish. P3 has no
  chapter of its own: a native-only capability belongs in
  `# Intentional architecture and capability differences`, and a limitation both
  engines share belongs in
  `# Shared defects and documentation-sensitive traps`.

Directions mean:

- **Python to Rust**: behavior available in Python is absent or different in
  Rust.
- **Rust to Python**: native behavior has no Python equivalent. This often
  indicates that Python should become a configuration/orchestration client
  rather than duplicate the implementation.
- **Bidirectional**: neither side is an unambiguous authority; a shared contract
  must be chosen.
- **Shared defect**: both paths are incomplete, the parity gate itself is
  missing, or two native paths disagree with each other.
- **Native-only defect**: the capability exists only natively and the Python
  counterpart has been deleted from the tree, so the item is a native
  correctness or lifecycle defect with no parity dimension.

Every numbered section carries a `**Direction:**` field. Section IDs are stable
and are not renumbered: a gap in the sequence means a section was deleted or
re-banded, a re-banded section is appended at the end of its destination band,
and a retired number is never reused. Retired: P0.6, P1.1, P1.3, P1.5, P1.9,
P1.10, P1.18, P1.20, P1.25, P1.29, P1.50, P1.51, P1.52, P2.11, P2.12.

## Executive findings

The migration is not blocked by one isolated subsystem. The concentrated risk is
the set of gaps that change a reported number on the stock path without raising
an error:

1. The native authoring boundary accepts and then discards per-model and
   per-transport keys, so an authored tokenizer, weight, LoRA, or modality can
   vanish with no parse error.
2. Client-visible token counts can mean tokenizer tokens in Python and response
   events in Rust.
3. Request latency ends at a different response per engine, so the headline
   latency and every token-derived timing can be inflated on native runs.
4. A user-centric phase floors its authored `users` and `concurrency` per worker
   thread in every dispatch mode, and unset `runtime.workers` auto-selects the
   machine's parallelism, so a cap below the core count runs at the core count.
5. The Python programmatic API still executes the service-mesh engine while the
   shipped CLI executes the native one.
6. Per-request records retain familiar field names while changing field
   meanings.
7. Configurations accepted by one frontend can be rejected, silently truncated,
   or assigned different defaults by the other, and several native profile flags
   are accepted without affecting execution.
8. Workload timing and admission also differ in fixed-schedule, graph, and
   `--dispatch sharded` modes.
9. The default dispatch mode reproduces a rate-paced run's mean rate but not its
   authored arrival process, so a stock `--request-rate` run under the default
   `poisson` distribution measures roughly half the authored burstiness and
   every latency tail derived from it.
10. A run with zero successful requests — a duration-bounded phase that ends
    before any request completes, or a run whose every request was cancelled by
    policy — reports success and exits zero natively while Python exits nonzero.
11. Native accuracy grading contains substantial implementation but is not
    reachable from the stock profile path.
12. Native exporters often implement a narrower operational contract than their
    Python counterparts.
13. Existing “parity” tests frequently compare one implementation with its own
    golden or compare only tolerant aggregate projections, and the cross-engine
    latency suite passes while comparing the native run against itself, because
    its Python leg executes nothing and both legs write to one artifact
    directory.

The Python `kubernetes` and `operator` packages are absent from the tree, so the
remaining Kubernetes items are native-only defects rather than parity gaps.

The first migration milestone should therefore be a deterministic cross-engine
acceptance gate. Without it, closing individual gaps cannot establish that a
Python execution path is safe to retire.

---

# P0 migration blockers

## P0.1 Native accuracy is not reachable from the stock profile path

**Direction:** Python to Rust

Python accuracy execution loads benchmark problems, dispatches requests, grades
records, and writes accuracy outputs. Rust contains evaluator protocols,
subprocess supervision, grading capture, accumulation, and CSV support, but the
normal CLI projection does not select the `static_accuracy` workload and the
stock HTTP workload registration does not expose the complete path.

**Impact**

- `--accuracy-benchmark` and its companion flags resolve into `cfg.accuracy`
  without executing the intended native accuracy workload.
- The native YAML authoring path exposes no `accuracy` key at all, so an
  accuracy run is not expressible as a config file.
- Python and native runs have different grading timing and association rules.
- Native report capabilities cannot compensate for an unreachable entry path.

**Executable evidence**

- Python: `src/aiperf/dataset/dataset_manager.py`
- Python: `src/aiperf/dataset/loader/accuracy_dataset_loader.py`
- Python: `src/aiperf/accuracy/accuracy_record_processor.py`
- Python tests: `tests/unit/accuracy/`
- Rust CLI: `rust/cli/src/load.rs`
- Rust projection: `rust/runtime/src/engine/protocol_v2.rs`
- Rust execution: `rust/runtime/src/accuracy.rs`
- Rust registration: `rust/runtime/src/engine/online_execution.rs`
- Rust E2E: `rust/e2e-tests/tests/test_accuracy_mock.rs`

**Convergence target**

Lower `cfg.accuracy` into a registered native accuracy workload, expose the
required evaluator configuration, and add a stock `aiperf profile` E2E that
asserts graded per-record output, aggregate accuracy, evaluator provenance, and
CSV output.

## P0.2 Endpoint transport policy can be silently lost

**Direction:** Bidirectional

The Python endpoint model declares HTTP/2, TLS verification, UDS, connection
limits, keepalive, templates, and response selectors. The Python aiohttp path
does not consume several of these fields. Rust's lower transport layers support
many of them, but the native YAML and CLI projection omits or hardcodes part of
the same surface.

The protocol substitution is silent. The ordinary non-mTLS TLS client advertises
ALPN `h2` before `http/1.1`
(`rust/runtime/src/transport/http/client/connection.rs:360`), so a native run
against any h2-capable HTTPS endpoint negotiates HTTP/2 where Python's aiohttp
used HTTP/1.1 — and because the `http2` key is unauthorable, the author cannot
even express the choice.

**Impact**

- A valid-looking profile can execute with the wrong protocol, pool size,
  keepalive, or endpoint formatter.
- `http2`, `connection_limit`, `keepalive_timeout`, `template`, and
  `response_field` reach execution only over the internal protocol-v2 wire;
  neither CLI nor YAML authoring can set them, and resolution pins them to
  constants.
- Load shape and measured latency change materially: connection reuse, pool
  limits, and DNS resolver behavior differ, and Python is HTTP/1.1 with global
  environment defaults while Rust multiplexes over h2c or negotiated HTTP/2
  under endpoint-scoped limits.
- The runtime's advanced transport support is not reliably available through
  the public product entry point.

**Executable evidence**

- Python model: `src/aiperf/config/endpoint.py`
- Python projection: `src/aiperf/common/models/model_endpoint_info.py`
- Python transport: `src/aiperf/transports/aiohttp_transport.py`
- Python transport defaults: `src/aiperf/transports/http_defaults.py`
- Rust YAML: `rust/cli/src/yaml.rs`
- Rust CLI projection: `rust/cli/src/load.rs`
- Rust endpoint model: `rust/runtime/src/config/model/endpoint.rs`
- Rust endpoint construction: `rust/runtime/src/config/resolve.rs`
- Rust runner profiles: `rust/runtime/src/engine/registry.rs`
- Rust client: `rust/runtime/src/transport/http/client/`
- Rust policy tests: `rust/cli/tests/http_policy_v2_stdio.rs`

**Convergence target**

Use one strict endpoint-profile DTO from authoring through worker-local client
construction. Every accepted field must either reach execution or fail during
validation. `rust/cli/src/yaml.rs`'s
`every_authored_endpoint_field_reaches_protocol_v2` is the YAML leg of that
gate; extend the same whole-surface assertion to `ProfileFlags` and to the
protocol-v2 request, and give every runtime-honored endpoint field an authoring
key. The negotiated HTTP protocol must be authorable and reported, so a run
never substitutes a protocol the author did not select.

## P0.3 Visible output-token counts have different meanings

**Direction:** Bidirectional

Python generally tokenizes reconstructed output text with the configured
tokenizer. Rust's observer path commonly increments output count once per
parsed response event. For non-streaming text, a complete response can count as
one Rust output token. For streaming text, the count can depend on server chunk
boundaries.

**Impact**

- OSL, token throughput, ITL denominators, goodput, and SLA decisions are not
  comparable between engines.
- TGI, tool-heavy streams, and servers that batch multiple tokens per SSE event
  are especially affected.
- Aggregate latency parity can look acceptable while token-derived metrics are
  wrong.

**Executable evidence**

- Python reduction: `src/aiperf/records/inference_result_parser.py`
- Python TGI endpoint: `src/aiperf/endpoints/huggingface_generate.py`
- Rust observer: `rust/runtime/src/metrics.rs`
- Rust reduction: `rust/runtime/src/transport/reduce.rs`
- Rust TGI endpoint: `rust/runtime/src/endpoints/tier2.rs`
- Rust E2E: `rust/e2e-tests/tests/test_huggingface_generate_endpoint.rs`

**Convergence target**

Define visible OSL as tokenizer output over reconstructed response text in both
engines. Retain response-event count separately for token-arrival timing and
diagnostics.

## P0.4 Usage normalization changes provider totals

**Direction:** Bidirectional

Both engines re-totalize Anthropic/Bedrock input tokens with cache-read and
cache-write fields over the same synonym order and the same disjoint-key gate.
The streaming and derived-total rules still diverge: Python selects the last
non-empty streaming usage object wholesale, Rust merges fields across usage
events with per-field carry-forward, and Rust derives a total from prompt plus
completion when the provider omits one while Python preserves absence.

**Impact**

- A provider that changes usage shape mid-stream, or nulls a field it set
  earlier, yields different `usage_*` values per engine.
- `usage_total_tokens` is a provider fact in Python and a derived value in Rust.
- Explicit absence is lost in one engine but preserved in the other.

**Executable evidence**

- Python: `src/aiperf/common/models/usage_models.py`
- Python streaming selection:
  `src/aiperf/common/models/record_models.py`
- Python adversarial tests:
  `tests/unit/common/models/test_usage_models_adversarial.py`
- Rust: `rust/runtime/src/endpoints/usage.rs`
- Rust reduction: `rust/runtime/src/transport/reduce.rs`
- Rust metrics: `rust/runtime/src/metrics.rs`
- Rust E2E: `rust/e2e-tests/tests/test_use_server_token_counts.rs`
- Rust E2E: `rust/e2e-tests/tests/test_usage_fields.rs`

**Convergence target**

Represent verbatim provider fields separately from derived normalized totals.
Specify identical synonym precedence, null handling, streaming merge, and
derived-total rules.

## P0.5 Mixed prose and tool calls undercount Python output

**Direction:** Rust to Python

Both chat parsers retain prose and tool-call text from mixed chunks.
`ToolCallResponseData.get_text()` combines them, but Python's result parser
tokenizes only `tool_call_text`. Rust reduction includes both.

**Impact**

- Python undercounts OSL for assistant turns that explain an action while
  dispatching a tool.
- Tool-heavy agentic workloads produce engine-dependent output counts.

**Executable evidence**

- `src/aiperf/endpoints/openai_chat.py`
- `src/aiperf/common/models/record_models.py`
- `src/aiperf/records/inference_result_parser.py`
- `tests/unit/endpoints/test_openai_chat_tool_call_reassembly.py`
- `rust/runtime/src/endpoints/chat_chunk.rs`
- `rust/runtime/src/endpoints/implementation.rs`
- `rust/runtime/src/transport/reduce.rs`
- `rust/e2e-tests/tests/test_tool_calls.rs`

**Convergence target**

Tokenize `ToolCallResponseData.get_text()` and add paired mixed-prose/tool-call
fixtures at both parser and product-E2E layers.

## P0.7 No cross-engine per-record acceptance gate

**Direction:** Shared defect

Most Python and Rust tests exercise separate engines and separate fixtures.
Three further suites are written as cross-engine comparisons but do not execute
Python at all, because the harness defaults its Python lane to an inert module
(P0.14): the narrow latency test compares aggregate averages with tolerance
against artifacts the native run itself wrote, and the random-range and seeded
Poisson tests fail their record-count assertions instead of comparing anything.

Two suites do run both engines and compare per-record facts, and each is scoped
to one feature: the raw-export and per-chunk-usage suites compare deterministic
`profile_export_raw.jsonl` projections. They also disagree on what "Python"
means — the raw-export suite exercises the current tree, while the
per-chunk-usage suite compares against a detached worktree pinned to a
hard-coded commit. No gate runs an arbitrary resolved request through both
current engines.

**Impact**

- Payload, response, usage, error, conversation metadata, and artifact drift can
  ship while each local suite remains green.
- Feature-scoped projections and pinned Python oracles cannot detect drift
  outside the field they select.
- Closing individual backlog items cannot prove migration readiness.

**Executable evidence**

- Python harness: `tests/integration/conftest.py`
- Rust harness: `rust/e2e-tests/tests/common/mod.rs`
- Existing latency comparison:
  `rust/e2e-tests/tests/test_rust_python_latency_parity.rs`
- Existing RNG comparison:
  `rust/e2e-tests/tests/test_seeded_poisson_parity.rs`
- Existing request-body comparison:
  `rust/e2e-tests/tests/test_random_range_e2e_parity.rs`
- Existing pinned-oracle raw-record comparisons:
  `rust/e2e-tests/tests/test_port_raw_parity.rs`,
  `rust/e2e-tests/tests/test_per_chunk_usage_parity.rs`

**Convergence target**

Run one deterministic resolved request through both engines and compare:

- materialized request bodies and headers;
- raw and metric JSONL records;
- response text and usage fields;
- errors and terminal status;
- conversation, turn, and branch identity;
- complete summary distributions and artifacts.

## P0.8 Authored per-model and per-transport fields are accepted and discarded

**Direction:** Python to Rust

(was P1.1 and P1.3)

Python Pydantic models declare `extra="forbid"` per model and reject unknown
fields. The Rust YAML root and every section struct set `deny_unknown_fields`,
so a Python-only key such as `benchmark.accuracy`, `endpointProfiles`,
`failurePolicy`, `logging`, `metrics`, or `endpoint.template` is a hard parse
error rather than a silent truncation. Three spots stay permissive and carry the
loss: the inline `Full` structs in the `ModelsSection` and `ModelItem` visitors,
and `TransportSection`'s flattened `options`, whose unknown keys are discarded
for `transport.type: http` and `grpc`. `ModelItem`'s `visit_map` decodes
`full.name` only, so every other key in a model mapping is read and dropped. Two
request fields are lost a second way: the resolver hardcodes
`endpoint_profiles` and `failure_policy` empty
(`rust/runtime/src/config/resolve.rs:1682-1683`) even though protocol v2 carries
them.

This is a blocker rather than an inconvenience because a dropped per-model
tokenizer silently changes every token count in the run.

**Impact**

- Per-model tokenizer, weights, LoRA, and modalities are accepted and then
  discarded, so the run benchmarks a different model configuration than the one
  authored.
- A typo or an unsupported per-model or per-transport key produces a different
  benchmark with no parse error.
- Seamless policy and ramp strategy are hardcoded at lowering
  (`seamless: false`, `strategy: "linear"`) regardless of what was authored.
- `endpoint_profiles` and `failure_policy` reach execution empty, so endpoint
  policy and failure policy authored anywhere upstream have no effect.

**Executable evidence**

- Python: `src/aiperf/config/config.py`
- Python: `src/aiperf/config/models.py`
- Python authoring surface: `src/aiperf/config/`
- Rust YAML: `rust/cli/src/yaml.rs`
- Rust CLI projection: `rust/cli/src/load.rs`
- Rust resolution: `rust/runtime/src/config/resolve.rs`
- Rust config models: `rust/runtime/src/config/model/`

**Convergence target**

Fail closed at the public authoring boundary: every accepted field must reach
the protocol request or be rejected before execution, and no decode site may
silently drop a key. Reserve permissive forward compatibility for an explicitly
named extension bag so a future key is opt-in rather than indistinguishable
from a typo.

## P0.9 Python execution paths still enter the service-mesh engine

**Direction:** Python to Rust

(was P1.9 and P1.18)

Python subprocess orchestration drives native runs through config files,
inherited output, and artifact heuristics; native execution expects a strict
stdin request and a terminal JSON envelope, and no Python consumer of that
protocol-v2 stdio contract exists. The consequence is that callers of
`build_benchmark_plan`, `run_benchmark`, `MultiRunOrchestrator`, and
`RunExecutor` still enter the Python service mesh while the shipped
`aiperf profile` enters the native runner, so the same public API silently
returns numbers from the engine being retired. While that holds, the
service-mesh engine cannot be deleted.

**Impact**

- Programmatic and CLI consumers do not exercise the same product, and neither
  path reports that it selected a different engine.
- Every outer loop written against the Python API keeps the mesh alive, so no
  amount of native gap closure retires it.
- Orchestration cannot observe typed native progress or the authoritative
  report path, because it reads artifacts instead of the terminal envelope.

**Executable evidence**

- Python plan: `src/aiperf/config/loader/plan.py`
- Python runner: `src/aiperf/cli_runner/`
- Python orchestration: `src/aiperf/orchestrator/`
- Python subprocess execution:
  `src/aiperf/orchestrator/local_executor.py`,
  `src/aiperf/orchestrator/subprocess_runner.py`
- Rust single run: `rust/cli/src/profile.rs`
- Rust stdio execution: `rust/cli/src/execute.rs`
- Rust stdio contract tests: `rust/cli/tests/protocol_v2_stdio.rs`

**Convergence target**

Build one Python protocol-v2 executor adapter that sends the resolved request,
parses typed terminal responses, forwards progress, and loads the authoritative
report path, then route `run_benchmark` and `MultiRunOrchestrator` through it
and explicitly deprecate the service-mesh execution API.

## P0.10 User-centric caps are over-subscribed to the worker count

**Direction:** Python to Rust

(was part of P1.14)

The `PhaseSpec::UserCentric` arm applies `owned_cap` —
`owned_positions(..).max(1)` — to `users` and to `concurrency` in every dispatch
mode, not only `sharded`
(`rust/runtime/src/engine/sharded_scheduled.rs:217-233`), while unset
`runtime.workers` auto-selects machine parallelism. So an authored `users: 8` on
a 144-core host admits one user per thread and runs roughly 144 concurrent
users. Python's global scheduler holds the authored cap exactly.

**Impact**

- The load shape, every latency percentile, and throughput are wrong, with no
  error and no warning.
- The over-subscription factor is the host's core count, so the same config
  measures differently on every machine.
- A user-centric result cannot be compared against a Python baseline or against
  another native host.

**Executable evidence**

- Python: `src/aiperf/timing/concurrency.py`
- Rust slicing: `rust/runtime/src/engine/sharded_scheduled.rs`
- Rust sharding: `rust/runtime/src/engine/execute/sharding.rs`
- Rust product tests: `rust/cli/tests/thread_per_core_product.rs`

**Convergence target**

Floor per-thread caps at zero rather than one, and prove exact global `users`,
`concurrency`, and prefill totals at every worker count and in every dispatch
mode.

## P0.11 Request latency ends at a different response per engine

**Direction:** Bidirectional

(was P1.29)

Python ends request latency at the last meaningful content response. Rust can
use transport terminal time, which includes a trailing usage frame or `[DONE]`.
Both boundaries are individually defensible; the defect is that neither is
declared, so the same named metric means two things.

**Impact**

- Trailing usage and `[DONE]` frames are standard on the default streaming chat
  path, so native request latency and every token-derived timing can be
  inflated on every run.
- A native-versus-Python latency comparison is invalid without either engine
  reporting a problem.
- Downstream SLA and goodput decisions inherit the inflation, because request
  latency is the headline metric they read.

**Executable evidence**

- Python metric: `src/aiperf/metrics/types/request_latency_metric.py`
- Python worker: `src/aiperf/workers/worker.py`
- Rust dispatch: `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs`
- Rust metric store: `rust/runtime/src/metrics_core/store.rs`

**Convergence target**

Record the last meaningful response separately from transport terminal time, and
document one boundary for request latency that both engines apply.

## P0.12 A zero-success native run reports success and exits zero

**Direction:** Python to Rust

(was P1.52)

The coordinator's run-outcome guard fails a run only when the native report
carries a positive `error_request_count` and no `request_count`, and it first
subtracts every `RequestCancellationError` from that error count
(`rust/runtime/src/engine/coordinator.rs:440-475`). Two zero-success outcomes
therefore return `success: true` with `exit_code: 0`: a phase whose
`benchmark_duration` elapses before any request completes, and a run whose every
request was cancelled by policy. Python exits nonzero for both — the first
through `No profile results to export`
(`src/aiperf/controller/system_controller.py:1115-1133`), the second through the
all-failed guard at `:1136`, which does not exempt cancellations because
`RecordsTracker.update_from_request` counts any record carrying an error as an
error (`src/aiperf/records/records_tracker.py:348-353`) and a policy cancellation
carries one (`src/aiperf/transports/aiohttp_client.py:290-295`).

The guard's doc comment claims it "mirrors the python engine's
`system_controller` guard" and names `--dry-run` as the case the zero-request
pass-through protects. Neither holds: Python's zero-record guard has no native
counterpart, and a normal `--dry-run` reports `request_count >= 1`, so it exits
zero through the success arm and never reaches the exemption. Rust already
contains a Python-faithful classifier — `classify_summary` in
`rust/cli/src/sweep/confidence.rs:139-155` reproduces
`local_executor._build_result_from_metrics` including its `No requests completed`
and `All N requests failed` strings — but the single-run coordinator does not use
it, so the same run passes as a single run and fails as a sweep or search cell.

**Impact**

- A misconfigured duration-bounded run writes ten artifact files, an empty
  `profile_export.jsonl`, an empty console export, and a `native-v2.json` whose
  `metrics` array is empty, and exits zero. Any CI gate or wrapper script that
  keys on the exit code records a passing benchmark that measured nothing.
- An all-cancelled run persists a full report and exits zero, so a cancellation
  policy that accidentally cancels everything is indistinguishable from a healthy
  run at the process boundary.
- The cancellation subtraction cannot distinguish an authored `cancellation:`
  workload from a deadline cancellation, a router-side cancel, or a server that
  closed every connection, so it suppresses genuine transport failures too.
- Single-run and sweep-cell classification disagree on the same artifacts, so a
  cancellation or zero-request workload is unreportable in a sweep while passing
  as a single run.
- The guard has no test anywhere in the Rust tree: its only references are its
  definition and its one call site.

**Executable evidence**

- Python single-run guards: `src/aiperf/controller/system_controller.py`
- Python sweep classifier: `src/aiperf/orchestrator/local_executor.py`
- Python success/error split: `src/aiperf/records/records_tracker.py`
- Python cancellation record: `src/aiperf/transports/aiohttp_client.py`
- Python exit-code contract test: `tests/integration/test_startup_failures.py`
- Rust guard: `rust/runtime/src/engine/coordinator.rs`
- Rust cancellation classification: `rust/runtime/src/engine/records.rs`
- Rust faithful classifier: `rust/cli/src/sweep/confidence.rs`,
  `rust/cli/src/sweep/aggregate.rs`
- Rust cancellation product test:
  `rust/e2e-tests/tests/test_request_cancellation.rs`
- Rust dry-run product test: `rust/dry-run-tests/tests/dry_run.rs`

**Convergence target**

Classify every run — single, repeated trial, sweep cell, and search probe — from
the native report through one coordinator-owned three-way classifier
(`Succeeded` / `AllRequestsFailed` / `NoRequests`), exit nonzero on both failure
arms with the sweep path's existing diagnostic strings, and delete the
summary-reading duplicate. Drop the cancellation subtraction so cancellations
count as errors exactly as Python counts them; a cancellation-terminal run that
must exit zero opts in through the resolved config, never through an absent
counter. Add product tests asserting exit codes for a duration-truncated run, an
all-cancelled run, an all-failed run, and a `--dry-run`, on both the single-run
and sweep-cell paths.

## P0.13 Default dispatch does not reproduce the authored arrival process

**Direction:** Python to Rust

Python's request-rate issuer is a single-issuer renewal process: each tick draws
one interval from the arrival-pattern generator and adds it to the running
target (`src/aiperf/timing/strategies/request_rate.py:284`), so `poisson`
inter-arrivals are exponential and `gamma` carries its authored shape. The
default `global` dispatch mode instead fires on a fixed base grid claimed from
`GlobalRateGate` and adds a mean-zero per-thread offset
(`rust/runtime/src/request_rate.rs:605-646`,
`rust/runtime/src/timing/rate_gate.rs:98-116`). Mean rate is exact — that is what
the gate is for — but the process is a differenced grid, not a renewal process.
`rate_gate.rs:16-24` states this and points at `global-hop` for arrival-pattern
parity; `Global` is nonetheless the default
(`rust/runtime/src/config/model/dispatch.rs:81-82`) and nothing warns when a
jittered pattern is authored under it.

Measured at 200 req/s, `--request-rate-mode poisson`, `--random-seed 42`, 400
requests against `aiperf-mock-server --fast`, coefficient of variation of
observed inter-arrivals (`request_start_ns` from `profile_export.jsonl`): Python
1.061, Rust `global-hop` 1.074, Rust `global` 0.546. A true Poisson process has
CV = 1.

**Impact**

- `poisson` is the default arrival distribution
  (`rust/cli/src/flags.rs:831-832`), so this is the stock rate-paced path rather
  than an opt-in mode: a plain `--request-rate` run silently measures roughly
  half the authored burstiness.
- Queue-sensitive results — TTFT and ITL tails, p95/p99 request latency, and any
  SLA or goodput decision derived from them — differ from Python and from
  `global-hop` on the same config, with no error and no warning.
- The same config also cannot be compared across dispatch modes, so a native
  rate-paced baseline is not portable between `global` and `global-hop`.
- The divergence is admitted by the implementation's own module doc rather than
  caught by a test, so it is a rationalized difference: no committed reference
  number pins the native inter-arrival distribution against Python's.

**Executable evidence**

- Python issuer: `src/aiperf/timing/strategies/request_rate.py`
- Rust shared rate gate: `rust/runtime/src/timing/rate_gate.rs`
- Rust request-rate workload: `rust/runtime/src/request_rate.rs`
- Rust sharding: `rust/runtime/src/engine/execute/sharding.rs`
- Rust dispatch default: `rust/runtime/src/config/model/dispatch.rs`
- Rust arrival-distribution default: `rust/cli/src/flags.rs`
- Rust timing tests: `rust/runtime/tests/request_rate_real.rs`

**Convergence target**

Make the shared gate claim jittered fire times from one cell-wide renewal
sequence rather than a grid plus per-thread offsets, so `global` matches a single
issuer in distribution as well as in mean. Until then, warn when a
non-`constant` arrival pattern is authored under `global` and state the exact
guarantee in the flag help. Pin the observed inter-arrival distribution against a
Python reference in a product test, not only the mean rate.

## P0.14 Cross-engine parity suites compare the native run against itself

**Direction:** Shared defect

`src/aiperf/cli.py` constructs the cyclopts `app` at module scope and has no
`if __name__ == "__main__"` guard; the guard lives in `src/aiperf/__main__.py`.
`python -m aiperf.cli profile <args>` therefore imports, runs nothing, writes
nothing, and exits 0 — including for unknown flags and for `--help`. The e2e
harness defaults its Python lane to exactly that module
(`python_module.unwrap_or("aiperf.cli")`,
`rust/e2e-tests/tests/common/mod.rs:456`), so only the two suites that override
`AIPERF_E2E_PYTHON_MODULE` to `aiperf` execute Python at all.

`test_rust_python_latency_parity.rs` does not merely lose its oracle: it passes
while comparing the native run against itself. `harness.run` and
`harness.run_env` both target the single `artifact_path()`
(`rust/e2e-tests/tests/common/mod.rs:322`, `:327-328`, `:372-373`), and the
harness documents `run_in` as the way to keep "A/B outputs disjoint when both
products target the same mock" (`:331-332`), which this suite does not use. The
Rust leg runs first and writes `profile_export_aiperf.json`; the inert Python leg
exits 0 and therefore satisfies the `python.success()` assertion;
`metric_avg(&python, …)` then globs `**/*aiperf.json` out of that same directory
and reads the Rust run's numbers. The TTFT, ITL, and request-latency deltas are
exactly zero, all three parity assertions pass, and the suite carries no
`#[ignore]`.

**Impact**

- The suite this backlog cites as its cross-engine latency comparison (P0.7,
  `# Parity test blind spots`) is green and vacuous, so its passing status is
  evidence about the native engine only.
- `test_random_range_e2e_parity.rs` is the cited byte-exact request-body oracle.
  Its inert Python leg satisfies `python.success()` and then fails the
  `python_captures.len() == REQUESTS` assertion, so the suite yields no
  cross-engine signal in either direction.
- `test_seeded_poisson_parity.rs` is the only evidence offered for the
  `AIPERF_RNG_BACKEND` parity lane. Both of its legs run the inert module, both
  satisfy their `exit_code == 0` assertion, and both then fail the record-count
  assertion in `observed_arrival_offsets`.
- The harness fails open at the boundary that matters: a Python leg that created
  no artifact directory, no records, and no captured requests is
  indistinguishable from a completed run at the `success()` check, and a shared
  artifact directory converts that silence into the other engine's numbers.
- Cross-engine coverage is therefore two suites, and they do not agree on what
  they compare against: `test_port_raw_parity.rs` overrides the module to
  `aiperf` and exercises the current tree, while `test_per_chunk_usage_parity.rs`
  builds a detached worktree at the hard-coded `PYTHON_ORACLE_COMMIT` and
  compares against that frozen commit instead (P0.7).

**Executable evidence**

- Inert Python entry point: `src/aiperf/cli.py`
- Working entry point: `src/aiperf/__main__.py`
- Harness module selection, shared artifact directory, and the disjoint-output
  alternative: `rust/e2e-tests/tests/common/mod.rs`
- Self-comparing suite:
  `rust/e2e-tests/tests/test_rust_python_latency_parity.rs`
- Inert Python leg: `rust/e2e-tests/tests/test_random_range_e2e_parity.rs`
- Inert Python leg: `rust/e2e-tests/tests/test_seeded_poisson_parity.rs`
- Overriding suites that do run Python:
  `rust/e2e-tests/tests/test_port_raw_parity.rs`,
  `rust/e2e-tests/tests/test_per_chunk_usage_parity.rs`

**Convergence target**

Give `src/aiperf/cli.py` a `__main__` guard or point the harness default at
`aiperf`, and run each engine into its own artifact directory so one engine's
artifacts can never answer for the other. Make the harness fail closed when a
Python leg produces no records, no artifact directory, and no captured requests,
then re-establish all three suites' assertions against a Python run that
actually executes.

---

# P1 configuration and CLI gaps

## P1.2 The Rust authoring root and the Pydantic model accept disjoint field sets

**Direction:** Bidirectional

The committed schema is generated from the Pydantic model and gated against it,
so those two agree. The Rust root disagrees with both in each direction:
`benchmark.transport`, `benchmark.metadata`, and the
`trajectoryStart{Min,Max}Ratio` axes are Rust-only, while `endpointProfiles`,
`failurePolicy`, `accuracy`, `logging`, and `metrics` are Python-only. Each side
rejects the other's keys as unknown.

**Risk:** IDE validation approves a file the native binary rejects, and a file
the native binary accepts is rejected by Python.

**Evidence:** `src/aiperf/config/config.py`,
`src/aiperf/config/schema/aiperf-config.schema.json`,
`tools/generate_config_schema.py`,
`tests/unit/config/test_config_schema_generator_integration.py`,
`rust/runtime/src/config/model/config.rs`, `rust/cli/src/yaml.rs`.

**Target:** generate the schema and both frontends from one typed capability
model.

## P1.4 Native profile accepts dead or differently defined flags

**Direction:** Bidirectional

Wired in a dedicated pass (CLI clap + `Inputs` / resolve projection), with
runtime backends for the former fail-closed trio:

- `--export-outputs-json`, `--allow-dataset-wrap` / `--no-allow-dataset-wrap`,
  `--cache-bust`, `--max-context-length`, `--use-think-time-only`,
  `--trace-idle-gap-cap-seconds`, `--burst-phase-starts`, `--hf-weka-dataset`
- `-vv` / `--extra-verbose`, wired by the pre-clap argv scan in
  `rust/cli/src/logging.rs` rather than through `Inputs`/resolve. It remains
  listed in `UNIMPLEMENTED_FLAGS`, so authoring it selects TRACE and also warns
  that it has no effect.
- `--vary-seed-per-trial`, `--no-fixed-schedule`, `--profile-export-prefix`,
  `--show-trace-timing`
- `--trace-session-sample-ratio` (baseten_trace whole-session subsample)
- `--agentic-warmup-grace-period` (synthesized agentic warmup barrier grace)
- `--failed-request-threshold` (profiling soft-cancel on error ratio)

Still dead / differently defined (intentionally retired or unfinished):

- `--stream` as Python OTel telemetry-domain selection
  (`metrics`/`timing`/`default`) versus a native boolean that no call site
  reads and that `UNIMPLEMENTED_FLAGS` does not even warn about;
- `--auto-plot` and `--plot-required` without the Python completion callback;
- `--stats-interval` without native runtime wiring;
- API/UI/ZMQ/record-processor options that belong to the retired service mesh;
- `--workers-max` not affecting native worker resolution;
- `--num-conversations` and `--num-sessions` represented as independent Rust
  fields rather than aliases.

**Evidence:** `src/aiperf/config/flags/`,
`rust/cli/src/flags.rs`, `rust/cli/src/load.rs`,
`rust/cli/src/profile.rs`.

**Target:** remove obsolete flags, wire remaining supported flags, and generate a
flag-to-resolved-field parity test.

## P1.6 Environment-variable contracts are split

**Direction:** Bidirectional

Examples:

- Python uses `AIPERF_UI_REALTIME_METRICS_INTERVAL`; native runtime uses
  `AIPERF_STATS_INTERVAL`.
- Rust hardcodes discrepancy thresholds that Python reads from environment.
- The flatgraph toggle (`AIPERF_DISABLE_FLATGRAPH`) has no Python declaration
  at all, and four separate Rust truthy parsers disagree on their accepted
  spellings: `0/false/off/no`, `1`/`true`, `1/true/t/yes/y/on`, and
  `1/true/yes/on`.
- The runtime engine selector is largely a harness choice, not an in-process
  runtime switch: no code under `src/aiperf/` reads
  `Environment.RUNTIME.ENGINE`, and `AIPERF_RUNTIME_ENGINE` is set only by the
  Rust e2e harness.

**Evidence:** `src/aiperf/common/environment.py`,
`rust/cli/src/logging.rs`, `rust/runtime/src/realtime.rs`,
`rust/runtime/src/metrics_core/accumulator.rs`.

**Target:** one typed environment catalog with common names, precedence,
parsing, defaults, and generated documentation.

## P1.7 General adaptive search remains Python-only

**Direction:** Python to Rust

Python owns multidimensional adaptive sweeps and QMC expansion, and native trial
convergence is absent because `--convergence-mode`, `--convergence-stat`, and
`--convergence-threshold` sit in `UNIMPLEMENTED_FLAGS`. Native search is
centered on named recipes and feature-gated planner implementations, but it does
implement Pareto behavior (a `pareto-sweep` recipe plus its own frontier
aggregation), an adaptive replicate budget, and `--scenario` submission locks.

**Evidence:** `src/aiperf/orchestrator/`,
`src/aiperf/config/sweep/expand_qmc.py`, `rust/cli/src/search.rs`,
`rust/cli/src/isotonic.rs`, `rust/cli/src/profile.rs`,
`rust/cli/src/sweep/aggregate.rs`.

**Target:** adopt one outer-loop plan model. Native recipes should be
projections of it rather than a separate orchestration contract.

## P1.8 Trial seeds, cooldowns, and convergence differ

**Direction:** Python to Rust

Both frontends default to reusing the seed across trials and both opt into
per-trial variation with `--vary-seed-per-trial`. Python still distinguishes
trial from variation cooldowns and can stop repetitions early; native execution
collapses `--profile-run-cooldown-seconds` and
`--parameter-sweep-cooldown-seconds` into one inter-cell duration and runs every
planned cell because the `--convergence-*` flags are unimplemented.

**Evidence:** `src/aiperf/orchestrator/orchestrator.py`,
`src/aiperf/orchestrator/strategies.py`,
`rust/cli/src/profile.rs`, `rust/cli/src/sweep/run.rs`.

**Target:** one trial scheduler covering ordering, seed derivation, cooldowns,
convergence, and failure policy.

---

# P1 workload and execution gaps

## P1.11 Native phase projection collapses authored phase semantics

**Direction:** Python to Rust

Python supports an ordered phase list with validated common fields. Native YAML
preserves an explicit `phases:` list verbatim through `phases_override`, but
collapses the `warmup:`/`profiling:` shorthand to one profiling phase and
hardcodes several advanced fields at lowering.

**Affected behavior**

- seamless transitions, hardcoded `false` at every lowering site even though the
  runtime implements the handoff;
- cache-pressure warmup duration, hardcoded `None` in YAML and reachable only
  from `--agentic-cache-warmup-duration`;
- non-linear ramps: a ramp accepts only a scalar duration and is lowered with
  `strategy: "linear"`, against Python's `RampConfig{duration, strategy}`;
- grace validation;
- native validation rejects `excludeFromResults: true` on a profiling phase,
  which Python accepts.

**Evidence:** `src/aiperf/config/phases.py`,
`src/aiperf/config/ramp.py`, `rust/cli/src/yaml.rs`,
`rust/cli/src/load.rs`.

**Target:** preserve the ordered phase union and all common fields end to end.

## P1.12 Fixed-schedule admission and stop behavior differs

**Direction:** Bidirectional

Python applies request/duration stop conditions and concurrency/prefill
admission around fixed replay. Rust generally treats the trace as the plan,
rejects prefill in this mode, and handles offset filtering separately from the
workload config.

**Impact:** the same trace can dispatch a different request set and concurrency
shape.

The native prefill rejection fires at dataset build rather than at validation,
so `aiperf config validate` reports such a config as valid and the run fails
later.

**Evidence:** `src/aiperf/timing/strategies/fixed_schedule.py`,
`src/aiperf/timing/phase/stop_conditions.py`,
`rust/runtime/src/fixed_schedule.rs`,
`rust/runtime/src/engine/execute/dataset_build.rs`.

**Target:** choose one public rule—trace-authoritative or admission-limited—and
apply it in both engines.

## P1.13 User-centric prefill and budget validation are incomplete

**Direction:** Python to Rust

Python applies prefill admission and validates request/session budgets against
the user count. Rust user-centric scheduling lacks an equivalent prefill pool
and does not mirror all cross-field checks.

**Evidence:** `src/aiperf/config/phases.py`,
`src/aiperf/timing/strategies/user_centric_rate.py`,
`rust/runtime/src/user_centric.rs`,
`rust/runtime/src/engine/execute/dataset_build.rs`.

**Target:** align budget validation and prefill admission while preserving
Rust's interruptible adaptive wake-up.

## P1.14 Sharded-mode cap slicing can exceed global caps

**Direction:** Python to Rust

Python uses a global scheduler. Rust partitions work per thread and forces some
shard-local limits to at least one. The `Concurrency`, `Poisson`, `Constant`,
and `Gamma` slicing that does so is gated on `DispatchMode::Sharded`, so it is
reachable only under `--dispatch sharded` or `--cells N`, where `Sharded` is the
deliberate default; the default `global` mode admits concurrency and prefill
from one shared per-cell gate built on the unsliced authored cap. P0.10 owns the
`user_centric` leg, which is floored per thread in every mode.

**Evidence:** `src/aiperf/timing/concurrency.py`,
`rust/runtime/src/engine/sharded_scheduled.rs`,
`rust/runtime/src/engine/execute/sharding.rs`,
`rust/cli/tests/thread_per_core_product.rs`.

**Target:** allow zero-cap shards and prove exact global concurrency,
prefill, and request totals for every worker count.

## P1.15 URL selection restarts per Rust worker

**Direction:** Python to Rust

Python's round-robin selector is global. Rust worker-local selectors can each
start from URL zero.

**Impact:** endpoint balance and request-to-endpoint assignment change with
worker count.

**Evidence:** `src/aiperf/timing/url_samplers.py`,
`rust/runtime/src/timing/url_selection.rs`,
`rust/runtime/src/engine/execute/dataset_build.rs`.

**Target:** derive endpoint selection from a global dispatch ordinal or
partition one logical sequence, as the graph path already does with
`session_url_index` in `rust/runtime/src/engine/graph_execution.rs`.

## P1.16 Graph records omit Python branch lineage

**Direction:** Python to Rust

Python records include parent correlation, depth, branch mode, and rich branch
statistics. Native graph records and summaries do not preserve the complete
compatibility shape.

**Impact:** downstream tools cannot reconstruct branch topology from native
artifacts.

**Evidence:** `src/aiperf/common/models/record_models.py`,
`src/aiperf/common/models/branch_stats.py`,
`rust/runtime/src/engine/records.rs`,
`rust/e2e-tests/tests/test_dag_spawn.rs`,
`rust/e2e-tests/tests/test_dag_full_topology.rs`.

**Target:** emit stable branch correlation IDs and a compatibility branch-stat
rollup. Unignore the topology E2E tests.

## P1.17 Failure policy is not authoritative end to end

**Direction:** Shared defect

The Rust runtime supports abort-on-failure and protocol v2 carries the policy
end to end, but the native frontend cannot author it: `benchmark.failurePolicy`
is rejected as an unknown key, no `--failure-policy` flag exists (only the
graph-only `--graph-stop-on-failure`), and the resolver hardcodes
`failure_policy: None`. Python exposes the setting in `cfg` but no Python
execution path reads it, so scheduled HTTP execution behaves as continue.

**Evidence:** `src/aiperf/config/config.py`,
`rust/runtime/src/config/resolve.rs`, `rust/runtime/src/failure.rs`,
`rust/runtime/src/engine/execute/compose_sidecars.rs`,
`rust/runtime/src/request_rate.rs`.

**Target:** one typed policy carried through both frontends and tested with
identical partial and total failures.

---

# P1 transport and endpoint gaps

## P1.19 TLS trust semantics differ

**Direction:** Shared defect

Python aiohttp uses OpenSSL/system trust and a global verification environment
toggle. Rust HTTP uses rustls/WebPKI and endpoint policy. Neither exposes one
shared inference custom-CA/mTLS contract.

**Evidence:** `src/aiperf/transports/http_defaults.py`,
`tests/unit/transports/test_tcp_connector.py`,
`rust/runtime/src/transport/http/config/defaults.rs`,
`rust/runtime/tests/transport_http/tls.rs`.

**Target:** endpoint-scoped verification, system/custom trust selection, and
client certificate fields with matching semantics.

## P1.21 Redirect and non-2xx behavior differs

**Direction:** Bidirectional

Python inference requests follow redirects by aiohttp default. Rust reports
3xx as errors. Error-body retention also differs.

**Evidence:** `src/aiperf/transports/aiohttp_client.py`,
`rust/runtime/src/transport/http/client/http_client.rs`,
`rust/e2e-tests/tests/test_error_fidelity.rs`.

**Target:** disable inference redirects and preserve the same status, headers,
body, and error fields.

## P1.22 Timeout and error taxonomies differ

**Direction:** Python to Rust

Python exports exception-derived names and cause chains and has a send-timeout
state. Rust uses normalized error kinds but does not retain the same nested
cause information.

**Evidence:** `src/aiperf/common/models/error_models.py`,
`src/aiperf/transports/aiohttp_client.py`,
`rust/runtime/src/transport/core/error.rs`,
`rust/runtime/src/engine/records.rs`.

**Target:** a transport-neutral exported error enum with stable names/codes and
equivalent cause/body retention.

## P1.23 SSE semantics differ on repeated data and case variants

**Direction:** Python to Rust

Python joins repeated `data:` lines and performs case-insensitive field/error
matching. Rust's parser behavior is narrower. Both share a mixed-delimiter edge
case.

**Evidence:** `src/aiperf/transports/sse_utils.py`,
`src/aiperf/common/models/record_models.py`,
`tests/unit/transports/test_sse_utils.py`,
`rust/runtime/src/transport/http/sse/`.

**Target:** one conformance corpus for line endings, repeated data, UTF-8
splits, comments, case variants, `[DONE]`, malformed JSON, and error events.

## P1.24 Embeddings validation differs and usage is dropped

**Direction:** Shared defect

Python and Rust disagree on zero-dimensional vectors and mixed-validity
`chat_embeddings` arrays. Both fail to preserve non-empty response usage in
the complete metrics path.

**Evidence:** `src/aiperf/endpoints/openai_embeddings.py`,
`src/aiperf/endpoints/chat_embeddings.py`,
`rust/runtime/src/endpoints/implementation.rs`,
`rust/e2e-tests/tests/test_embeddings_endpoint.rs`.

**Target:** one vector validity contract and usage capture for both dialects.

## P1.26 Raw and template endpoint contracts are not portable

**Direction:** Bidirectional

Python uses runtime plugins and Jinja2, accepts arbitrary JSON values for raw
payloads, and authors nested template config. Rust uses linked registration and
MiniJinja, requires an object in key paths, and expects a different template
projection.

**Evidence:** `src/aiperf/endpoints/template_endpoint.py`,
`src/aiperf/endpoints/raw_endpoint.py`,
`rust/runtime/src/endpoints/tier2/flexible.rs`,
`rust/cli/src/yaml.rs`.

**Target:** canonical template/selector shape, a published common template
subset, arbitrary valid JSON raw bodies, and an explicit native extension seam.

## P1.27 Content-server configuration is not a public end-to-end feature

**Direction:** Shared defect

Python exposes environment settings but no server implementation or reliable
projection. Rust implements secure serving and publishing, but activation is
primarily protocol-side rather than normal profile authoring.

**Evidence:** `src/aiperf/common/environment.py`,
`rust/runtime/src/content_server/`,
`rust/cli/tests/online_v2_stdio.rs`.

**Target:** a typed `contentServer` Config v2 section lowered through the stock
profile path; remove inert environment settings.

## P1.28 Multimodal URL materialization occurs at different stages

**Direction:** Bidirectional

Python fetches required media during dataset setup. Rust fetches after payload
rendering at dispatch.

**Impact:** failure timing, deduplication scope, cache behavior, and measured
overhead differ. Synthetic media bytes can also differ by encoder settings.

**Evidence:** `src/aiperf/dataset/dataset_manager.py`,
`src/aiperf/dataset/generator/image.py`,
`rust/runtime/src/dataset/media.rs`,
`rust/runtime/src/transport/http/transport/inline_media.rs`.

**Target:** one materialization phase, timeout/concurrency policy, dedup scope,
and deterministic media fixture.

---

# P1 metrics, records, artifacts, and exporters

## P1.30 Aggregate record eligibility differs

**Direction:** Bidirectional

Canceled, failed, and partially measured records enter distributions
differently. Credit/effective latency can be populated outside Rust's normal
valid-record gate while Python includes other numeric canceled rows.

**Evidence:** `src/aiperf/metrics/accumulator.py`,
`src/aiperf/metrics/derived_latency.py`,
`rust/runtime/src/metrics_core/store.rs`,
`rust/runtime/src/metrics_core/accumulator.rs`.

**Target:** one metric-by-metric eligibility matrix applied before all
distribution accumulation.

## P1.31 Per-request metadata changed meaning

**Direction:** Python to Rust

Examples:

- `session_num`: Python credit index versus Rust conversation index;
- `request_ack_ns`: Python response acknowledgment versus Rust first token;
- cancellation time: dedicated instant versus request end;
- record-processor ID: a live service identity versus the constant
  `aiperf runner`;
- DAG parent/depth: omitted or hardcoded in Rust.

**Evidence:** `src/aiperf/common/models/record_models.py`,
`src/aiperf/records/record_processor_service.py`,
`rust/runtime/src/engine/records.rs`,
`rust/runtime/src/scheduled.rs`.

**Target:** restore legacy meanings or introduce a versioned schema with
unambiguous renamed fields.

## P1.32 Native raw traces and errors are lossy

**Direction:** Python to Rust

Rust replaces the rich Python HTTP trace with a compact transport summary,
drops nested error causes/details, lacks the same binary response variant, and
uses inconsistent synthetic error names between artifacts.

**Evidence:** `src/aiperf/common/models/trace_models.py`,
`src/aiperf/common/models/error_models.py`,
`rust/runtime/src/engine/records.rs`,
`rust/runtime/src/transport/core/response.rs`.

**Target:** one versioned raw-record DTO with equivalent trace, response, and
error fidelity.

## P1.33 Summary schemas differ at missing and non-finite values

**Direction:** Bidirectional

Python can retain a key with `null`; Rust may omit it. Distribution sums,
top-level times, and branch statistics are also not consistently projected.

**Evidence:** `src/aiperf/exporters/metrics_json_exporter.py`,
`tests/unit/exporters/test_metrics_json_exporter.py`,
`rust/runtime/src/export/genai_perf.rs`.

**Target:** one schema version with explicit missing/non-finite policy and
complete run metadata.

## P1.34 Artifact policy is not fully projected

**Direction:** Python to Rust

Native gaps include summary disabling, templated user-file content, trace
display options, prefixing of the server-metrics and network-latency sidecar
filenames, and some overwrite rules.

**Evidence:** `src/aiperf/config/artifacts.py`,
`tests/unit/config/test_profile_export_prefix_scope.py`,
`rust/cli/src/flags.rs`, `rust/cli/src/load.rs`,
`rust/cli/src/yaml.rs`.

**Target:** type and validate every artifact policy field and require expected
artifacts to exist in E2E tests.

## P1.35 OTLP export is post-run and narrower in Rust

**Direction:** Python to Rust

Python periodically exports through the OTel SDK and includes additional
AIPerf/timing metrics. Rust performs a single post-run request carrying only
the four GenAI semantic-convention client histograms, with different
resource/provider inference and no equivalent custom header support.

**Evidence:** `src/aiperf/post_processors/otel_streaming_fanout.py`,
`src/aiperf/post_processors/otel_metrics_results_processor.py`,
`rust/runtime/src/export/otel.rs`.

**Target:** either keep the live sidecar authoritative or implement periodic,
authenticated, full-surface native export.

## P1.36 MLflow lifecycle and metadata are narrower in Rust

**Direction:** Python to Rust

Native MLflow lacks Python's live-phase run reuse, authentication and
non-HTTP/non-file URI handling, the metadata sidecar artifact, and projected
run parameters.

**Evidence:** `src/aiperf/exporters/mlflow_data_exporter.py`,
`rust/runtime/src/export/mlflow.rs`,
`rust/runtime/src/config/model/export.rs`.

**Target:** one run across live/final phases, full redacted parameters,
standard authentication, and the metadata sidecar artifact.

## P1.37 W&B is offline-only and metadata-poor in Rust

**Direction:** Python to Rust

Python creates a cloud run and artifact bundle. Rust writes an offline datastore
requiring later synchronization, defers the versioned artifact bundle, and
projects an empty redacted config and invocation command.

**Evidence:** `src/aiperf/exporters/wandb_data_exporter.py`,
`rust/runtime/src/export/wandb/`,
`rust/runtime/src/config/model/export.rs`.

**Target:** make online/offline mode explicit and preserve the same reproducible
artifact and metadata contract.

## P1.38 Telemetry configuration is only partially lowered

**Direction:** Python to Rust

Native projection refuses dashboard mode, hardcodes the collection interval,
selects local GPU collectors only from Config v2 rather than the
`--gpu-telemetry` keyword list, and omits the flag-level enable/disable mutex
check.

**Evidence:** `src/aiperf/config/flags/_converter_telemetry.py`,
`src/aiperf/config/gpu_telemetry.py`,
`rust/runtime/src/config/model/telemetry.rs`,
`rust/runtime/src/config/resolve.rs`, `rust/cli/src/load.rs`.

**Target:** carry the complete typed policy into sidecar specs and resolve
defaults in one place.

## P1.39 Server-metrics phase and histogram rules differ

**Direction:** Bidirectional

Python uses continuous scrape history plus pre-phase references. Rust uses
forced phase-boundary snapshots. Python invalidates a histogram after a
non-finite bucket; Rust can keep remaining buckets.

**Evidence:** `src/aiperf/server_metrics/manager.py`,
`src/aiperf/server_metrics/accumulator.py`,
`rust/runtime/src/server_metrics/accumulator.rs`,
`rust/runtime/src/server_metrics/parser.rs`.

**Target:** shared pathological scrape fixtures and one boundary/reset/
histogram-validity contract.

## P1.40 GPU telemetry windows and failure observability differ

**Direction:** Python to Rust

Python includes grace behavior around energy counters and publishes structured
status/errors. Rust uses exact boundaries and often logs warnings before
producing empty telemetry.

**Evidence:** `src/aiperf/gpu_telemetry/manager.py`,
`src/aiperf/gpu_telemetry/accumulator.py`,
`rust/runtime/src/gpu_telemetry/accumulator.rs`,
`rust/runtime/src/engine/gpu_telemetry.rs`.

**Target:** align windows and emit structured native telemetry status and
failures.

---

# P1 datasets, synthesis, tokenizers, and operational contracts

## P1.41 Dataset load and sampling semantics differ

**Direction:** Bidirectional

Differences include:

- sampling preference bound at config resolution in Python versus at dataset
  construction in Rust, so the projected request differs even when the
  effective strategy agrees;
- unpaired per-loader sampling overrides;
- trace offsets accepted but not always propagated;
- Rust-only per-turn endpoint/model/streaming/token-ID fields;
- generic CSV support only in Rust;
- coercion and row-validation differences.

**Evidence:** `src/aiperf/config/dataset/`,
`src/aiperf/dataset/loader/`,
`rust/runtime/src/dataset/loader/`,
`rust/runtime/src/dataset/sampler.rs`,
`rust/runtime/src/engine/execute/dataset_build.rs`.

**Target:** one row schema, loader-preference rule, trace window contract, and
paired loader corpus.

## P1.42 Public datasets have two unsynchronized catalogs

**Direction:** Bidirectional

Python uses plugin metadata with alias normalization (lowercase, `-`→`_`) and
runtime extension. Rust embeds a YAML catalog with exact-key lookup. The key
sets have diverged: Python lists 43 entries, Rust 32, and only 26 are shared.
Python-only keys are `weka_hf` plus the 16-key `semianalysis_cc_traces_weka*`
family, which Rust replaces with the single renamed `weka_cc_traces_062126`
and a `weka_hf` special case outside the catalog. Rust-only keys are `alpaca`,
`alpaca_cleaned`, `dolly`, `mmlu`, `sharegpt_vicuna`, and `ultrachat`.
`--hf-dataset` bypasses the catalog with the `hf` loader rather than
reconciling it, and is mutually exclusive with `--public-dataset`.

**Evidence:** `src/aiperf/plugin/plugins.yaml`,
`src/aiperf/plugin/extensible_enums.py`,
`rust/runtime/resources/public_datasets.yaml`,
`rust/runtime/src/config/model/public_catalog.rs`,
`rust/runtime/src/config/resolve.rs`,
`rust/cli/tests/public_catalog_formats.rs`.

**Target:** generate both representations from one catalog and test key parity,
metadata, aliases, errors, and capability checks.

## P1.43 Public dataset bounds and timing validation differ

**Direction:** Bidirectional

Both engines leave a non-streaming dataset unbounded when no entry count is
authored, but only two Rust loaders (`exgentic`, `asr`) consume the
`max_conversations` bound the catalog computes, so a streaming catalog entry
bound to any other loader loads its whole split. Python validates
fixed-schedule timing metadata before execution; Rust threads a
`fixed_schedule` flag into loader options and discovers the metadata after
loading.

**Evidence:** `src/aiperf/dataset/loader/base_hf_dataset.py`,
`src/aiperf/config/dataset/resolver.py`,
`rust/runtime/src/config/model/public_catalog.rs`,
`rust/runtime/src/config/resolve.rs`,
`rust/runtime/src/dataset/loader/`.

**Target:** centralize bound calculation and catalog capability validation.

## P1.44 Seeded synthetic datasets are not reproducible across engines

**Direction:** Bidirectional

The RNG namespace registries agree, and Rust carries a Python-compatible RNG
core, but draw order, rounding, prefix generation, audio bounds, video
defaults, and ranking length generation differ, and no committed golden pins a
synthetic conversation or media payload across both engines.

One corpus is exempt by construction. The `random` corpus bypasses the swappable
RNG on both sides for a fixed vLLM/SGLang reference stream — Python's
`RangeRatioDistribution` draws from `numpy.random.default_rng` (PCG64) and its
SGLang subclass from `numpy.random.RandomState` (MT19937), and Rust's
`ReferenceRandomStream` mirrors both with `NumpyGenerator` and
`NumpyRandomState` — so its bodies are byte-exact across engines under the
default backends. The named corpora are not.

**Evidence:** `src/aiperf/dataset/composer/synthetic.py`,
`src/aiperf/common/models/sequence_distribution.py`,
`src/aiperf/dataset/generator/`,
`rust/runtime/src/dataset/loader/synthetic.rs`,
`rust/runtime/src/dataset/random_range.rs`,
`rust/runtime/src/dataset/generator/`.

**Target:** one namespace/draw-order specification and a full conversation/media
golden produced by both engines.

## P1.45 Native synthesize has a narrower artifact and validation contract

**Direction:** Python to Rust

Python agentic-code synthesis writes quality and visualization artifacts and
enforces stricter turn/reset/restart feasibility. Rust writes the core dataset
and manifest but accepts combinations Python rejects.

**Evidence:** `src/aiperf/dataset/agentic_code_gen/`,
`rust/cli/src/synthesize/`,
`tests/unit/dataset/agentic_code_gen/test_writer.py`.

**Target:** choose one artifact set and validation contract, then enforce it in
both commands.

## P1.46 Synthesize goldens are native-authored, not cross-engine oracles

**Direction:** Shared defect

Three committed goldens under `tools/parity/synthesize/` pin
`aiperf synthesize agentic-code` byte-exactly, and the native tests are active.
No in-tree generator produces them: they are regenerated from the Rust
implementation whenever it changes, so they are a native regression gate rather
than a Python-authoritative oracle, and the Python command never produces or
checks them.

**Evidence:** `rust/cli/tests/synthesize_parity.rs`,
`tools/parity/synthesize/`,
`src/aiperf/dataset/agentic_code_gen/cli.py`.

**Target:** regenerate the goldens from the Python command, pin them as the
cross-engine oracle, and require the native parity tests in CI.

## P1.47 Tokenizer resolution policies differ

**Direction:** Python to Rust

Python performs Hub alias search, ambiguity handling, prefetch, and broader
remote tokenizer behavior. Rust resolves repository IDs directly, ignores
`trust_remote_code` and refuses only repositories that ship neither a
`tokenizer.json` nor a native tiktoken vocab, and pins the artifact set to a
fixed predicate list rather than to the repository revision: the remote path
fetches whatever matches `is_downloadable_tokenizer_file` — `tokenizer.json` or
a tiktoken vocab, `tokenizer_config.json`, the special-tokens and vocab/merges
files, and chat templates — plus `config.json` and `generation_config.json` by
exact name, so any artifact a revision adds outside that list is absent.

**Evidence:** `src/aiperf/common/tokenizer.py`,
`src/aiperf/common/tokenizer_validator.py`,
`src/aiperf/dataset/_tokenizer_preload.py`,
`rust/runtime/src/dataset/tokenizer.rs`,
`rust/runtime/src/dataset/hf_hub.rs`,
`rust/runtime/src/engine/online_execution.rs`.

**Target:** implement or explicitly reject alias resolution, define the
revision-complete artifact set, and test recorded Hub fixtures.

## P1.48 Comprehensive RNG parity silently skips

**Direction:** Shared defect

The Python parity test points to a stale path and skips when it cannot find the
file. The committed vectors live under the runtime crate, and no Rust test
consumes the complete JSON vector set.

**Evidence:** `tests/unit/common/test_rng_parity.py`,
`rust/runtime/tests/data/rng_parity_vectors.json`,
`rust/runtime/examples/rng_parity_vectors.rs`.

**Target:** fix the path, fail instead of skipping in CI, and replay the same
vectors from both languages.

## P1.49 Path resolution and cache policy differ

**Direction:** Bidirectional

Relative paths, config-directory anchoring, tilde expansion, symlink handling,
artifact creation timing, cache roots, offline behavior, and revision keys are
not consistent.

**Evidence:** `src/aiperf/config/resolution/`,
`src/aiperf/dataset/loader/base_public_dataset.py`,
`rust/cli/src/load.rs`, `rust/runtime/src/dataset/fetch.rs`,
`rust/runtime/src/engine/online_execution.rs`.

**Target:** specify path base/expansion per field and a shared cache root,
namespace, revision, and offline policy.

## P1.53 The session-affinity header cannot be disabled and its env var is renamed

**Direction:** Python to Rust

Both engines derive `X-Session-Affinity` from the stable correlation ID by
default. Python exposes an off switch,
`AIPERF_HTTP_X_SESSION_AFFINITY_FROM_CORRELATION_ID=0`
(`src/aiperf/common/environment.py:748, 846`, consumed at
`src/aiperf/transports/base_transports.py:150`). Rust reads a differently named
variable, `AIPERF_HTTP_X_SESSION_AFFINITY`
(`rust/runtime/src/transport/http/transport/headers.rs:63-67`), whose doc comment
claims to mirror a Python field `Environment.HTTP.X_SESSION_AFFINITY` that does
not exist. The switch also has no wire effect. Header composition honors it
(`rust/runtime/src/transport/http/sink.rs:568, 858`,
`rust/runtime/src/transport/http/transport/endpoint_binding.rs:370-373`), but the
transport facade then re-inserts the canonical value unconditionally whenever a
correlation ID is present
(`rust/runtime/src/transport/http/transport/http_transport.rs:127-131`), which is
the one header path in that file not gated on its toggle. The header reaches the
wire under no env var, under Python's name set to `0`, and under Rust's own
documented name set to `0`.

**Risk:** `X-Session-Affinity` pins every turn of a session to one replica on an
affinity-aware router (a Dynamo frontend or the SGLang Model Gateway), so a user
who disables it in Python to measure unpinned routing measures pinned routing
natively — different KV-cache hit rate, different TTFT, different throughput,
with no error and no warning. The dead switch also means the three sibling
opt-in toggles in the same module are gated correctly through the facade while
this one is not, so the module reads as if all four are live.

**Evidence:** `src/aiperf/common/environment.py`,
`src/aiperf/transports/base_transports.py`,
`rust/runtime/src/transport/http/transport/headers.rs`,
`rust/runtime/src/transport/http/transport/http_transport.rs`,
`rust/runtime/src/transport/http/sink.rs`,
`rust/e2e-tests/tests/test_port_raw_parity.rs`.

**Target:** route the facade's header insertion through
`session_affinity_header_enabled()`, accept Python's variable name (keeping the
short name as an alias), and add a product test asserting the header is absent
from the wire when the switch is off. Fold the naming half into P1.6.

## P1.54 The `runtime` section is disjoint in both directions

**Direction:** Bidirectional

P1.2 lists the Rust-only and Python-only keys at the authoring root; the nested
`runtime` block diverges too, and neither side's keys appear in that list.
`RuntimeSection` in `rust/cli/src/yaml.rs` sets `deny_unknown_fields` and accepts
exactly `ui`, `workers`, `workersMin`/`workers_min`, `cells`, `dispatch`, and
`hopRouting`/`hop_routing`. Python's `RuntimeConfig` is
`ConfigDict(extra="forbid")` and declares `ui`, `workers`, `record_processors`,
`service_run_type`, `communication`, `api_port`, `api_host`,
`dataset_api_base_url`, `workers_per_pod`, `record_processors_per_pod`,
`workers_min`, and `stats_interval`. Only `ui`, `workers`, and `workers_min` are
shared. The committed schema is generated from the Pydantic model with
`additionalProperties: false`, so it rejects `cells`, `dispatch`, and
`hopRouting`; the native binary rejects `statsInterval`, `recordProcessors`,
`serviceRunType`, and `apiPort` by name, and `workersMax` reaches neither
frontend.

**Risk:** the two knobs that decide whether a run's aggregate caps are exact —
`runtime.dispatch` and `runtime.cells` (see P1.14 and P0.10) — and the knob that
decides whether it reproduces the authored arrival process (P0.13) cannot be
authored in a schema-validated config at all, so an operator cannot pin the
admission regime their measurement depends on, and IDE validation flags a file
the native binary accepts.

**Evidence:** `src/aiperf/config/runtime.py`,
`src/aiperf/config/schema/aiperf-config.schema.json`,
`tools/generate_config_schema.py`,
`rust/runtime/src/config/model/runtime.rs`, `rust/cli/src/yaml.rs`.

**Target:** fold `runtime` into P1.2's one typed capability model. Every key
either reaches both frontends or is rejected with an explicit migration error
naming the owning engine.

---

# P2 compatibility and edge gaps

## P2.1 Config interpolation and normalization are narrower in Rust

**Direction:** Python to Rust

Python supports richer normalization, named-list Jinja lookup, and scenario
merging. Rust expansion is narrower.

**Target:** shared interpolation and normalization fixtures executed by both
frontends.

## P2.2 `config` command interfaces differ

**Direction:** Bidirectional

The native and Python commands use different positional/flag forms and expose
different expand/init options.

**Target:** make the shipped native command a compatible superset.

## P2.3 Dry-run is native-only

**Direction:** Rust to Python

Rust has an analytical dry-run transport and dedicated tuning flags. Python has
no dry-run mode at all, and its Config-v2 `TransportType` schema admits only
`http`, so the mode is not authorable through the shared schema.

**Target:** expose dry-run in shared Config v2 and reserve distinct names for
manifest preview.

## P2.4 Proxy scope and loopback policy differ

**Direction:** Bidirectional

Benchmark traffic is direct by default on both sides: Python's aiohttp sessions
set `trust_env=False`, and Rust leaves the client proxy unset unless
`--proxy`/`--proxy-from-env` or Config-v2 `endpoint.proxy` opts in. Dataset and
tokenizer downloads diverge: Rust resolves them from the ambient proxy
environment automatically, while Python keeps them direct. Loopback policy also
differs by opt-in shape. Rust's environment-derived proxying always excludes
loopback, but an explicit `--proxy` is applied as authored including loopback,
and Python's trust-env lane has no loopback carve-out, so inference, telemetry,
or media requests can traverse an ambient proxy once it is enabled.

**Target:** one typed proxy policy with the same opt-in surface, the same
download scope, and a mandatory loopback bypass on both engines.

## P2.5 Fixed-schedule filtering and continuation error policies differ

**Direction:** Bidirectional

One engine can fail setup where the other omits a conversation, and continuation
delay anchoring or invalid metadata can produce different outcomes.

**Target:** shared offset/filter/continuation fixtures.

## P2.6 Cancellation reason and stuck-slot reporting differ

**Direction:** Bidirectional

Rust has richer lifecycle completion reasons and RAII slot release. Python
synthesizes some recoveries and exports a smaller reason vocabulary.

**Target:** common exported lifecycle reasons while preserving internal
ownership designs.

## P2.7 Graph input DTOs accept different fields

**Direction:** Bidirectional

The shared `dag_jsonl` name masks parser differences in endpoint overrides,
headers, roles, joins, non-finite delays, and empty roots.

**Target:** one strict DTO and pathological fixture corpus. That corpus must
cover:

- Mooncake traces, whose discriminator and numeric coercion rules are not
  identical between the engines.

## P2.8 Per-record Parquet lacks a cross-engine contract

**Direction:** Rust to Python

Per-record Parquet is native-only in practice — Python's `RecordsExportFormat`
accepts `parquet` but all emission is native Rust — and its hand-built flat
Arrow schema inherits record metadata gaps. Python's separate server-metrics
Parquet exporter is a different artifact and not part of this contract.

**Target:** derive Parquet from the versioned record DTO and fail configuration
when the selected engine cannot emit it.

## P2.9 Native console lacks Python's live operational surface

**Direction:** Python to Rust

Native output does not mirror Python realtime rows, server snapshots, GPU and
accuracy sections, quiet behavior, or cache hints.

**Target:** a common progress projection and explicit UI ownership.

## P2.10 Cache and media encoders are not byte-compatible

**Direction:** Bidirectional

Python and Rust can produce different JPEG bytes and use different cache keys
for otherwise identical media or datasets.

**Target:** only require byte parity where artifacts promise it; otherwise
require semantic media equivalence and stable cache identity.

## P2.13 Feature-blind validation can approve unavailable capabilities

**Direction:** Native-only defect

(was P1.5)

Native behavior depends on Cargo features for gRPC, cellular, DynoSim,
Parquet, and search integrations. Validation is not consistently tied to the
frozen application capability registry. `grpc`, `cellular`, `parquet`, and
`websocket` are in the default CLI feature set (`rust/cli/Cargo.toml:34`), so
the reachable gap is `dynosim` and `search-pyo3`/`pyo3-embed` builds, and the
failure is loud at execution.

**Risk:** validation succeeds, then execution fails or an artifact is absent.

**Evidence:** `rust/cli/Cargo.toml`, `rust/cli/src/config/mod.rs`,
`rust/runtime/src/extensions/mod.rs`, `rust/runtime/src/export/mod.rs`.

**Target:** validate against the exact running binary's capability set and
hard-fail requested unavailable outputs.

## P2.14 Python image edit can miss multipart selection

**Direction:** Rust to Python

(was P1.25)

Python endpoint metadata declares form-data requirements, but transport
selection can still send JSON unless content type is explicitly set. Rust
automatically selects and validates multipart, so Python is the defective side.
The reach is narrow: one endpoint (`openai_image_edit`), and the server rejects
the wrong content type loudly.

**Evidence:** `src/aiperf/endpoints/openai_image_edit.py`,
`src/aiperf/transports/aiohttp_transport.py`,
`rust/runtime/src/endpoints/tier2.rs`,
`rust/e2e-tests/tests/test_image_edit_endpoint.rs`.

**Target:** endpoint metadata must select multipart in every execution path,
with wire-level tests for file bytes and malformed input.

## P2.15 Kubernetes native-only submission and lifecycle gaps

**Direction:** Native-only defect

(was P1.50 and P1.51)

No Python Kubernetes or operator package remains in the tree, so these are
native defects rather than parity gaps. Native Kubernetes execution invokes
`aiperf controller`, `cell`, `aggregator`, and `results-sidecar`, and requires
an `--image-capabilities` document whose `imageDigest` matches the envelope
digest and which declares `cellular: true`, `resultsSidecar: true`, and
`hierarchicalAggregation: false` before any cluster effect. Confirmed issues
are:

- the capability document is caller-authored, and nothing inspects the
  referenced image to confirm the declared commands are present, so a
  mismatched declaration is discovered only when a pod fails;
- progress body construction without production patch calls, so a running
  benchmark never publishes phase progress;
- best-effort CR writes with swallowed failures;
- cancellation hardcoded `false` at native completion, and the sidecar
  discarding the manifest's `wasCancelled` field;
- no native gzip response negotiation.

**Evidence:** `rust/cli/src/kube/contract.rs`,
`rust/cli/src/kube/submission.rs`,
`rust/cli/src/kube/command.rs`,
`contracts/native-k8s/v1/image-capabilities.schema.json`,
`rust/cli/src/cellular_role.rs`, `rust/cli/src/k8s.rs`,
`rust/cli/src/results_sidecar.rs`, `rust/cli/src/kube/results.rs`.

**Target:** verify declared image capabilities against the referenced image
digest before submission, then one marker/status lifecycle, periodic progress,
cancellation propagation, explicit CR durability policy, and one sidecar HTTP
contract.

## P2.16 Native-only transports and endpoint families are unauthorable in the shared schema

**Direction:** Rust to Python

P2.3 records this for `dry_run`; the same closed enums exclude every other
native-only capability. The shared `TransportType` schema admits only `http`,
while `Transport::canonical_id` in
`rust/runtime/src/config/model/transport.rs:40-48` emits `http`, `grpc`,
`dynosim_offline`, `dynosim_online`, `dry_run`, and `websocket`, and the schema
rejects the `benchmark.transport` key outright. The shared `EndpointType` enum is
closed over 19 HTTP dialects and rejects all 18 native gRPC endpoint ids
registered in `rust/runtime/src/endpoints/kserve.rs` (`kserve_chat`,
`kserve_completions`, `kserve_embeddings`, `kserve_v1_predict`,
`kserve_v2_infer`, `kserve_v2_embeddings`, `kserve_v2_rankings`,
`kserve_v2_vlm`, `kserve_v2_images`) and `rust/runtime/src/endpoints/riva.rs`
(`riva_asr`, `riva_tts`, `riva_text_classify`, `riva_token_classify`,
`riva_transform_text`, `riva_punctuate_text`, `riva_natural_query`,
`riva_analyze_intent`, `riva_analyze_entities`); none of them declares an alias,
so no spelling is accepted. A config the native binary reports as valid fails
schema validation.

**Risk:** the `# Intentional architecture and capability differences` chapter
declares gRPC, KServe, and Riva as owned native capabilities, but no
schema-validated config can select them, so "native-only capability" and
"unreachable through the shared authoring contract" are indistinguishable to a
downstream consumer or an IDE.

**Evidence:** `src/aiperf/config/schema/aiperf-config.schema.json`,
`rust/runtime/src/config/model/transport.rs`,
`rust/runtime/src/endpoints/kserve.rs`,
`rust/runtime/src/endpoints/riva.rs`, `rust/cli/src/yaml.rs`.

**Target:** extend the shared `TransportType` and `EndpointType` enumerations to
the native registry's frozen capability set, generated from one source, and have
Python reject an unimplementable selection with an explicit native-only error
instead of an unknown-value error.

## P2.17 Warmup-to-profiling handoff exists in both engines and is compared in neither

**Direction:** Bidirectional

The `# Intentional architecture and capability differences` chapter labels the
graph warmup handoff native-only. Python implements the same transfer:
`AgenticReplayStrategy.finalize_phase` builds per-lane handoff state from drained
credits and join annotations (`_build_handoff_states`), derives replay resume
boundaries (`_build_handoff_replay_boundaries`), rewrites the shared trajectory
list (`_build_handoff_trajectories`), and carries the `CacheBustLedger` across
the phase boundary so a continued session reuses its warmup marker. Rust's
`GraphWarmupHandoff` carries the parallel facts — a `LaneHandoff` per live lane
with `template_trace_id`, `instance_id`, `t_star_us`, `executed_node_ids`, and
`return_wall_us`, plus `drain_end_wall_us`, the next corpus draw index, and the
pressure lane count — and `graph_phase_runtime.rs` builds it at warmup finalize
and pops it once at profiling start. Neither engine's handoff is checked against
the other.

**Risk:** the resume frontier, the residual re-root delay, and the cache-bust
marker lineage are what an accelerated-warmup run measures. Two independent
implementations of the same transfer can resume at different turns or
cold-prefill behind a fresh marker, changing TTFT and cache-hit behavior with no
error, and the native-only label means no gap tracks the divergence.

**Evidence:** `src/aiperf/timing/strategies/agentic_replay.py`,
`src/aiperf/timing/trajectory_source.py`,
`src/aiperf/timing/replay_dependencies.py`,
`tests/unit/timing/strategies/test_agentic_replay.py`,
`rust/runtime/src/graph/warmup_handoff.rs`,
`rust/runtime/src/engine/graph_phase_runtime.rs`,
`rust/runtime/src/agentic_replay.rs`.

**Target:** define the handoff's public outcome — resumed turn per lane, residual
delay, and marker continuity — and assert it identically from both engines over
one recorded trace with a fixed warmup cutoff. Then either keep one
implementation authoritative or state explicitly which fields are not promised to
match.

## P2.18 Sharded dispatch renders a constant arrival pattern as a bursty one

**Direction:** Python to Rust

`--dispatch sharded` slices the authored rate across `W` threads
(`scaled_rate` in `rust/runtime/src/engine/sharded_scheduled.rs:137-238`), each
pacing its own independent sub-grid. The union has the right mean but not the
right spacing. Measured at 200 req/s, `--request-rate-mode constant`, 8 workers,
400 requests: effective rate 207.8/s, inter-arrival CV 5.6, maximum gap 160 ms
against a 5 ms authored interval; `global` gives CV 0.21 and `global-hop` 0.033
on the same config. Python's single issuer paces one grid, so `constant` is
effectively CV 0.

**Risk:** a `constant` arrival pattern under `sharded` measures a bursty
workload, so latency tails and any concurrency-derived metric differ from Python
and from the other native modes. Reachable only outside the default dispatch
mode, so it is narrower than P0.13.

**Evidence:** `src/aiperf/timing/strategies/request_rate.py`,
`rust/runtime/src/engine/sharded_scheduled.rs`,
`rust/runtime/src/engine/execute/sharding.rs`,
`rust/runtime/tests/request_rate_real.rs`.

**Target:** pace sharded threads against one cell-wide interleaved grid (offset
thread `k` by `k * interval / W`) and gate the mode on a measured inter-arrival
CV bound.

---

# Intentional architecture and capability differences

Each difference here is an internal design or an additive native capability that
Python should not reimplement. Two obligations survive that decision and are not
satisfied by the label alone.

First, a native-only capability must be selectable through the shared authoring
contract, or the backlog must carry the gap that says it is not.
`runtime.cells`, `runtime.dispatch`, `transport.type: grpc`, and every
`kserve*`/`riva*` endpoint id are accepted by the native binary and rejected by
the generated Config-v2 schema (P1.54, P2.16); `dry_run` has the same shape and
is tracked as P2.3.

Second, “native-only” is a claim about the current Python tree and must be
re-derived from it, not inherited. The graph warmup handoff is a live parity
question rather than an ownership statement, because Python carries its own
tested warmup-to-profiling transfer (P2.17).

An invariant listed in this chapter is a requirement, not a finding. Where one
does not hold in code it carries a numbered gap, and the entry names it.

## Native thread-per-core versus Python central credit scheduling

Python dynamically routes credits through central services. Rust selects one of
four `runtime.dispatch` admission strategies: `sharded` statically partitions
budget, concurrency, and rate per worker thread; `global` (the default) admits
from one shared per-cell slot pool and rate gate that is byte-exact against a
single global limiter; `global-hop` adds one coordinator-owned dispatcher for
exact issuance order; and `global-push` keeps that order while routing
identity-only credits the worker materializes, after Python's
`StickyCreditRouter`. `sharded` slices
`Concurrency`/`Poisson`/`Constant`/`Gamma` budgets, concurrency, and rate per
thread with a per-shard floor of one; `global`, `global-hop`, and `global-push`
hold the authored cap in one shared or single-coordinator gate, so exact global
request/session budgets and concurrency and prefill caps hold for those phase
shapes outside `sharded`. They do not hold for `user_centric`, whose `users` and
`concurrency` are floored per thread in every mode, and `sharded` is the default
whenever `runtime.cells > 1`. The remaining shared invariants are:

- deterministic endpoint assignment under a fixed seed (P1.15);
- equivalent records and aggregate metrics (P0.7);
- identical aggregate caps under `sharded` (P1.14) and, in every mode, for
  `user_centric` phases (P0.10);
- the authored arrival process, not only its mean rate, under a jittered
  `--request-rate-mode` (P0.13).

Those are gaps, not chosen differences. Internal issue order does not need to be
identical unless the product promises it.

## Native Graph-IR versus Python runtime branch sessions

Python creates branch sessions at runtime and tracks explicit joins. Rust
lowers branches to static nodes/channels and executes whole traces. Required
shared invariants are authored input, payloads, failure policy, lineage,
records, and summaries—not internal state machines. The warmup-to-profiling
handoff is not one of them: both engines implement that transfer, so its resume
frontier, residual delay, and cache-bust marker lineage are a shared contract
tracked as P2.17.

## Cellular execution is native-only

Python should validate and orchestrate cellular runs, not implement a second
cell transport. The parity target is consistent configuration, errors,
artifacts, and aggregate invariants with single-process execution.

## gRPC is native-only

Python has no current gRPC transport. KServe OIP and Riva should remain native.
The parity target is normalized endpoint behavior, records, metrics, and
errors—not a duplicate Python channel stack. Those endpoint families are
reachable only over the native gRPC transport, are additional native
capabilities rather than regressions from current Python shared endpoints, and
require strong byte-level codecs and product E2E tests.

## Default RNGs intentionally differ

Python defaults to its historical generators (MT plus NumPy with SHA-256
derivation). Rust defaults to BLAKE3-derived PCG streams. `AIPERF_RNG_BACKEND`
is honored symmetrically: `=rust` moves Python onto the native streams and
`=python` moves the native runtime onto the Python-compatible ones, so either
engine can host the parity lane.

`AIPERF_RNG_BACKEND` swaps the generator substrate and the seed algebra, not the
order in which a workload consumes draws. Setting `=python` natively therefore
produces a third distinct workload rather than convergence, because draw order,
rounding, prefix generation, and media bounds still differ (P1.44). No test
exercises `=python` against the native runtime, so the reverse direction is
unverified. One corpus is byte-exact across engines under the default backends
because it bypasses the swappable RNG for a fixed vLLM/SGLang reference stream
(`rust/runtime/src/dataset/random_range.rs`). The migration contract must name
that corpus as the only surface where cross-engine byte equivalence holds, and
must not describe `AIPERF_RNG_BACKEND` as a parity mode until draw order is
specified and a cross-engine golden pins it.

---

# Shared defects and documentation-sensitive traps

The following are not simply “Rust missing Python”:

- Generic raw/template paths on both sides can drop usage.
- Image-generation output count is not the same as input-image metrics.
- Streaming image tutorial extraction assumptions do not match top-level
  streaming chunks.
- Neither engine retries a request whose bytes reached the server: Rust hands
  every post-send outcome back to the caller unchanged, and Python's transport
  has no retry path. Native `max_connect_retries` (default 0) covers only
  pre-send DNS/TCP/TLS/handshake failures and is not an attempt model. Codify
  “no inference retries” or add an explicit idempotency-aware attempt policy
  with per-attempt records, because hidden retries invalidate benchmark
  semantics.
- Both contain configuration fields whose runtime meaning is incomplete.
- The Python tree contains no Riva or KServe implementation; its only mention is
  a prose comment in the RTFX metric. Any Riva behavior attributed to Python
  must be re-derived from the current tree before it is treated as product
  behavior.
- Some docs describe capabilities that only lower-level Rust code, not the
  stock CLI path, can currently reach.

Migration decisions must use executable current-tree behavior.

---

# Parity test blind spots

## Rust config goldens are Rust-only self-checks

`rust/cli/tests/parity.rs` validates native resolution against committed
requests. It does not run Python configuration resolution over the same
fixtures, and it excludes some exporter-critical dynamic fields.

## RNG parity can skip silently

Tracked with evidence and a target as P1.48.

## Search parity disappears without `search-pyo3`

Isotonic and Bayes parity tests are feature-gated. The default build therefore
provides no signal for those planners.

## Latency parity is aggregate and tolerant

`rust/e2e-tests/tests/test_rust_python_latency_parity.rs` compares a narrow set of
averages. It does not protect per-record boundaries, percentiles, delayed usage,
reasoning, batching, or client-token semantics.

## Several parity-named tests compare only Rust paths

Fold, flatgraph, and worker accumulation tests compare two native paths against
each other and name no Python reference at all, so they provide useful native
invariants rather than Python-to-Rust parity. Sweep aggregation asserts byte
equality against a committed golden with no provenance metadata and no
generator, so its Python attribution is unchecked. Five test files reference
`AIPERF_RUNTIME_ENGINE` and only two drive both engines; the other three drive a
Python module that executes nothing, and one of them passes by comparing the
native run against itself (P0.14).

## Tokenizer alias tests provide no live signal

The native alias-resolution E2E matrix is ignored and contains empty assertion
helpers.

## Synthesize goldens carry no provenance

Tracked with evidence and a target as P1.46.

## Graph parity expectations remain ignored

Native DAG spawn/topology E2E tests are ignored while waiting for branch
identity and statistics.

## Telemetry tests run in separate universes

Both implementations have substantial local tests, but no shared timestamped
scrape stream proves equivalent clipping, reset, malformed-histogram, cadence,
and error behavior.

---

# Migration roadmap

## Phase 0: establish the acceptance boundary

Before retiring any Python execution path:

1. Build a deterministic cross-engine harness.
2. Resolve one config through both frontends and compare normalized requests.
3. Run against the same deterministic mock-server configuration.
4. Compare materialized request bytes where the wire contract requires it.
5. Compare raw/per-record JSONL after removing only declared nondeterminism.
6. Compare complete metric distributions, not only averages.
7. Compare errors, exit status, and expected artifacts.
8. Run with client token counts and server token counts.
9. Run with one worker and multiple workers.
10. Make missing fixtures and skipped required gates fail CI.

The existing cross-engine suites cannot carry this gate. Three of them drive a
Python module that executes nothing, and the harness treats that as a completed
run, so the gate must be built on a Python leg that actually executes and a
harness that fails closed when it does not (P0.14).

## Phase 1: close correctness blockers

Recommended order:

1. Fix output-token semantics and mixed tool-call content.
2. Fix the request-latency terminal boundary.
3. Reproduce the authored arrival process in the default dispatch mode, not only
   its mean rate.
4. Enforce strict authoring validation that stops discarding accepted per-model
   and per-transport keys.
5. Prove exact global `users`, `concurrency`, prefill, and request totals at
   every worker count.
6. Adopt one run-outcome classifier that exits nonzero for zero-success and
   zero-request runs on the single-run, repeated-trial, sweep-cell, and
   search-probe paths.
7. Give the cross-engine parity suites a Python leg that executes, a per-engine
   artifact directory, and a harness that fails closed on an empty leg.
8. Align record metadata meanings and graph lineage.
9. Version and align usage accounting.
10. Complete endpoint policy projection, including the negotiated HTTP protocol.
11. Wire native accuracy.
12. Move programmatic Python execution onto the native runner over the
    protocol-v2 stdio adapter.

These items directly affect whether a benchmark result can be trusted.

## Phase 2: close workload and artifact compatibility

1. Preserve complete phase semantics.
2. Resolve fixed-schedule and user-centric admission contracts.
3. Make endpoint selection deterministic across workers.
4. Align record eligibility.
5. Version raw record and CSV schemas, and hold the existing per-record Parquet
   (`aiperf.schema_version`) and outputs-document versions stable.
6. Project full artifact policy.
7. Align telemetry windows and structured failures.

## Phase 3: migrate orchestration and outer loops

1. Adopt one sweep/search/trial plan.
2. Align seeds, cooldowns, convergence, ranking, and post-processing.
3. Retire service-mesh-only flags and environment variables.

## Phase 4: operational convergence

1. Decide native versus sidecar ownership for OTLP, MLflow, and W&B.
2. Align console/progress projection.
3. Complete the native-only Kubernetes image-capability, progress,
   cancellation, marker, and sidecar contracts, which are native defects rather
   than parity gaps.
4. Unify public dataset catalogs, caches, paths, and tokenizer resolution.
5. Declare native-only gRPC, cellular, KServe, Riva, and graph execution
   boundaries.

---

# Definition of migration-ready

The Python execution path can be considered replaceable only when all of the
following are true:

- every supported Python config either executes equivalently in Rust or fails
  with an explicit migration error;
- no accepted CLI/YAML field is silently ignored, including per-model and
  per-transport keys nested inside the `models` and `transport` sections;
- authored concurrency, prefill, user, and request caps hold exactly at every
  worker count and in every dispatch mode;
- a rate-paced run reproduces the authored arrival process, not only its mean
  rate, in every dispatch mode;
- a zero-success or zero-request run exits nonzero on every execution path;
- the Python programmatic API executes the native runner rather than the
  service mesh;
- deterministic shared workloads produce equivalent request bodies, records,
  usage, errors, and summaries;
- output-token and request-latency semantics are engine-independent;
- single-run, multi-run, search, and cellular failures use a consistent exit
  contract;
- expected artifacts are required by tests, not conditionally inspected;
- every cross-engine parity assertion runs against a Python process that
  actually executed, with the harness failing closed when it did not;
- accuracy grading is reachable from the stock profile path;
- graph records carry sufficient lineage for downstream reconstruction;
- exporter ownership and reduced native feature sets are explicit;
- required cross-language gates run in the default release CI configuration;
- remaining differences are listed either as intentional native-only
  architecture or as native-only defects with no Python counterpart, not as
  implied parity.

Until those conditions hold, “native default” and “Python-compatible” should be
treated as separate claims.
