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

The inventory is current as of 2026-07-17. It is based on executable code and
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

- **P0**: blocks migration or can silently produce invalid benchmark results.
- **P1**: materially changes workload semantics, measurements, artifacts, or
  operational behavior.
- **P2**: affects narrower modes, edge cases, diagnostics, or compatibility.
- **P3**: intentional native-only capability or low-risk polish.

Directions mean:

- **Python to Rust**: behavior available in Python is absent or different in
  Rust.
- **Rust to Python**: native behavior has no Python equivalent. This often
  indicates that Python should become a configuration/orchestration client
  rather than duplicate the implementation.
- **Bidirectional**: neither side is an unambiguous authority; a shared contract
  must be chosen.
- **Shared defect**: both paths are incomplete or the parity gate itself is
  missing.

## Executive findings

The migration is not blocked by one isolated subsystem. The major risk is that
several independent compatibility boundaries are incomplete at once:

1. Configurations accepted by one frontend can be rejected, silently truncated,
   or assigned different defaults by the other.
2. Several native profile flags and YAML fields are accepted without affecting
   execution.
3. Workload timing and admission differ in fixed-schedule, user-centric,
   multi-worker, and graph modes.
4. Client-visible token counts can mean tokenizer tokens in Python and response
   events in Rust.
5. Per-request records retain familiar field names while changing field
   meanings.
6. Native accuracy grading contains substantial implementation but is not
   reachable from the stock profile path.
7. Native exporters often implement a narrower operational contract than their
   Python counterparts.
8. Existing “parity” tests frequently compare one implementation with its own
   golden or compare only tolerant aggregate projections.

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

- A configuration can carry accuracy settings without executing the intended
  native accuracy workload.
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

**Impact**

- A valid-looking profile can execute with the wrong protocol, trust policy,
  pool size, keepalive, or endpoint formatter.
- Unsupported keys can disappear instead of producing a validation error.
- The runtime's advanced transport support is not reliably available through
  the public product entry point.

**Executable evidence**

- Python model: `src/aiperf/config/endpoint.py`
- Python projection: `src/aiperf/common/models/model_endpoint_info.py`
- Python transport: `src/aiperf/transports/aiohttp_transport.py`
- Rust YAML: `rust/cli/src/yaml.rs`
- Rust CLI projection: `rust/cli/src/load.rs`
- Rust endpoint model: `rust/cli/src/model/endpoint.rs`
- Rust runner profiles: `rust/runtime/src/engine/registry.rs`
- Rust policy tests: `rust/cli/tests/http_policy_v2_stdio.rs`

**Convergence target**

Use one strict endpoint-profile DTO from authoring through worker-local client
construction. Every accepted field must either reach execution or fail during
validation. Add field-by-field projection tests for CLI, YAML, and protocol-v2
input.

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

Python returns the first prompt-token synonym as reported. Rust re-totalizes
Anthropic/Bedrock input tokens with cache-read and cache-write fields. Python
selects the last non-empty streaming usage object; Rust merges fields across
usage events. Rust can also derive a total when Python preserves absence.

**Impact**

- `usage_prompt_tokens`, server-count ISL, cache percentages, and total usage
  differ on the same response.
- Raw provider accounting and normalized accounting are conflated.
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

**Direction:** Python to Rust

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
- `rust/runtime/src/endpoints/endpoints.rs`
- `rust/runtime/src/transport/reduce.rs`
- `rust/e2e-tests/tests/test_tool_calls.rs`

**Convergence target**

Tokenize `ToolCallResponseData.get_text()` and add paired mixed-prose/tool-call
fixtures at both parser and product-E2E layers.

## P0.6 A single native run can succeed when every request fails

**Direction:** Python to Rust

Python records an exit error when no requests succeed and one or more requests
fail. Rust's multi-run aggregation has a summary classifier, but a single
native profile can commit a report and return success based only on terminal
execution status.

**Impact**

- CI and automation can accept a benchmark that produced zero successful
  inference requests.
- Single-run and sweep semantics disagree within the native CLI.

**Executable evidence**

- Python: `src/aiperf/controller/system_controller.py`
- Python tests: `tests/integration/test_startup_failures.py`
- Rust single run: `rust/cli/src/profile.rs`
- Rust classifier: `rust/cli/src/sweep/confidence.rs`
- Rust coordinator: `rust/runtime/src/engine/coordinator.rs`
- Rust E2E: `rust/e2e-tests/tests/test_error_fidelity.rs`

**Convergence target**

Apply one post-report success classifier to single runs, repeated trials,
sweeps, and search probes. Exit nonzero when the run attempted requests and all
of them failed.

## P0.7 No cross-engine per-record acceptance gate

**Direction:** Shared defect

Most Python and Rust tests exercise separate engines and separate fixtures.
The narrow cross-engine latency test compares aggregate averages with
tolerance. The seeded Poisson test compares Python with a Rust reference
generator, not complete Python and Rust product runs.

**Impact**

- Payload, response, usage, error, conversation metadata, and artifact drift can
  ship while each local suite remains green.
- Closing individual backlog items cannot prove migration readiness.

**Executable evidence**

- Python harness: `tests/integration/conftest.py`
- Rust harness: `rust/e2e-tests/tests/common/mod.rs`
- Existing latency comparison:
  `rust/e2e-tests/tests/test_rust_python_latency_parity.rs`
- Existing RNG comparison:
  `rust/e2e-tests/tests/test_seeded_poisson_parity.rs`

**Convergence target**

Run one deterministic resolved request through both engines and compare:

- materialized request bodies and headers;
- raw and metric JSONL records;
- response text and usage fields;
- errors and terminal status;
- conversation, turn, and branch identity;
- complete summary distributions and artifacts.

---

# P1 configuration and CLI gaps

## P1.1 Rust Config v2 authoring is permissive where Python is strict

Python Pydantic models reject unknown fields. Rust YAML parsing intentionally
ignores unknown keys in several authoring structs.

**Risk:** typos or unsupported sections silently produce a different benchmark.

**Evidence:** `src/aiperf/config/base.py`,
`src/aiperf/config/config.py`, `rust/cli/src/yaml.rs`.

**Target:** fail closed at the public authoring boundary. Reserve permissive
forward compatibility for an explicitly named extension bag.

## P1.2 The JSON schema, Pydantic model, and Rust model disagree

The committed schema exposes fields such as transport/workload that Python's
strict model can reject, while Rust uses those axes for native transports.

**Risk:** IDE validation can approve a file that Python rejects; a file valid
for Python may omit fields required by native execution.

**Evidence:** `src/aiperf/config/config.py`,
`src/aiperf/config/schema/aiperf-config.schema.json`,
`rust/cli/src/model/config.rs`, `rust/cli/src/yaml.rs`.

**Target:** generate the schema and both frontends from one typed capability
model.

## P1.3 Native YAML projection is lossy

Confirmed losses or hardcoded values include parts of:

- endpoint profiles and failure policy;
- model weights, LoRA, modalities, and per-model tokenizer;
- template body and response selector;
- phase order, seamless policy, and ramp strategy;
- accuracy;
- artifact prefix, summary policy, and user files;
- telemetry collector and mode;
- richer runtime settings.

**Evidence:** `src/aiperf/config/`, `rust/cli/src/yaml.rs`,
`rust/cli/src/load.rs`, and the corresponding models under
`rust/cli/src/model/`.

**Target:** every accepted field reaches the protocol request or is rejected
before execution.

## P1.4 Native profile accepts dead or differently defined flags

Wired in a dedicated pass (CLI clap + `Inputs` / resolve projection):

- `--export-outputs-json`, `--allow-dataset-wrap` / `--no-allow-dataset-wrap`,
  `--cache-bust`, `--max-context-length`, `--use-think-time-only`,
  `--trace-idle-gap-cap-seconds`, `--burst-phase-starts`, `--hf-weka-dataset`,
  `-vv` / `--extra-verbose`
- `--vary-seed-per-trial`, `--no-fixed-schedule`, `--profile-export-prefix`,
  `--show-trace-timing`

Fail-closed until runtime exists (clear error, no silent ignore):

- `--trace-session-sample-ratio`
- `--agentic-warmup-grace-period`
- `--failed-request-threshold`

Still dead / differently defined (intentionally retired or unfinished):

- `--stream` as Python domain selection versus a native boolean with no
  equivalent live-streaming projection;
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

## P1.5 Feature-blind validation can approve unavailable capabilities

Native behavior depends on Cargo features for gRPC, cellular, DynoSim,
Parquet, and search integrations. Validation is not consistently tied to the
frozen application capability registry.

**Risk:** validation succeeds, then execution fails or an artifact is absent.

**Evidence:** `rust/cli/Cargo.toml`, `rust/cli/src/config/mod.rs`,
`rust/runtime/src/extensions/mod.rs`, `rust/runtime/src/export/mod.rs`.

**Target:** validate against the exact running binary's capability set and
hard-fail requested unavailable outputs.

## P1.6 Environment-variable contracts are split

Examples:

- Python uses `AIPERF_UI_REALTIME_METRICS_INTERVAL`; native runtime uses
  `AIPERF_STATS_INTERVAL`.
- Rust hardcodes discrepancy thresholds that Python reads from environment.
- Exact-fold and flatgraph toggles use Rust-only readers and inconsistent
  truthy parsing.
- Content-server environment settings exist in Python but are not projected.
- The runtime engine selector is largely a harness choice, not an in-process
  runtime switch.

**Evidence:** `src/aiperf/common/environment.py`,
`rust/cli/src/logging.rs`, `rust/runtime/src/realtime.rs`,
`rust/runtime/src/metrics_core/accumulator.rs`.

**Target:** one typed environment catalog with common names, precedence,
parsing, defaults, and generated documentation.

## P1.7 General adaptive search remains Python-only

Python owns multidimensional adaptive sweeps, multi-objective/Pareto behavior,
trial convergence, replicates, scenarios, and QMC expansion. Native search is
centered on named recipes and feature-gated planner implementations.

**Evidence:** `src/aiperf/orchestrator/`,
`src/aiperf/config/sweep/`, `rust/cli/src/search.rs`,
`rust/cli/src/profile.rs`, `rust/cli/src/sweep/`.

**Target:** adopt one outer-loop plan model. Native recipes should be
projections of it rather than a separate orchestration contract.

## P1.8 Trial seeds, cooldowns, and convergence differ

Python can vary seeds per trial, distinguish trial and variation cooldowns, and
stop repetitions early. Native execution can reuse seeds across trials,
collapse cooldown layers, and run a fixed number of cells despite convergence
flags.

**Evidence:** `src/aiperf/orchestrator/orchestrator.py`,
`src/aiperf/orchestrator/strategies.py`,
`rust/cli/src/profile.rs`, `rust/cli/src/sweep/run.rs`.

**Target:** one trial scheduler covering ordering, seed derivation, cooldowns,
convergence, and failure policy.

## P1.9 The Python programmatic API still executes a different engine

Callers of `build_benchmark_plan`, `run_benchmark`, `MultiRunOrchestrator`, and
`RunExecutor` enter the Python service mesh. The shipped `aiperf profile`
enters the native runner.

**Impact:** programmatic and CLI consumers do not exercise the same product.

**Evidence:** `src/aiperf/cli_runner/`, `src/aiperf/orchestrator/`,
`rust/cli/src/profile.rs`, `rust/cli/src/execute.rs`.

**Target:** provide a Python adapter for protocol-v2 native execution and
explicitly deprecate the service-mesh execution API.

## P1.10 The Kubernetes command tree is not registered on the Python root

The `kube` sub-app exists, but the Python root command list does not expose it.
The native binary delegates unknown commands to that root.

**Evidence:** `src/aiperf/cli.py`,
`src/aiperf/cli_commands/kube/_app.py`,
`rust/cli/src/dispatch.rs`.

**Target:** register `kube` and test it through the installed native entry
point.

---

# P1 workload and execution gaps

## P1.11 Native phase projection collapses authored phase semantics

Python supports an ordered phase list with validated common fields. Native YAML
reduces this to one warmup and one profiling phase in several paths and omits
advanced fields.

**Affected behavior**

- seamless transitions;
- explicit result exclusion;
- cache-pressure warmup duration;
- non-linear ramps;
- multiple authored profiling phases;
- grace validation.

**Evidence:** `src/aiperf/config/phases.py`,
`src/aiperf/config/ramp.py`, `rust/cli/src/yaml.rs`,
`rust/cli/src/load.rs`.

**Target:** preserve the ordered phase union and all common fields end to end.

## P1.12 Fixed-schedule admission and stop behavior differs

Python applies request/duration stop conditions and concurrency/prefill
admission around fixed replay. Rust generally treats the trace as the plan,
rejects prefill in this mode, and handles offset filtering separately from the
workload config.

**Impact:** the same trace can dispatch a different request set and concurrency
shape.

**Evidence:** `src/aiperf/timing/strategies/fixed_schedule.py`,
`src/aiperf/timing/phase/stop_conditions.py`,
`rust/runtime/src/fixed_schedule.rs`,
`rust/runtime/src/engine/execute.rs`.

**Target:** choose one public rule—trace-authoritative or admission-limited—and
apply it in both engines.

## P1.13 User-centric prefill and budget validation are incomplete

Python applies prefill admission and validates request/session budgets against
the user count. Rust user-centric scheduling lacks an equivalent prefill pool
and does not mirror all cross-field checks.

**Evidence:** `src/aiperf/config/phases.py`,
`src/aiperf/timing/strategies/user_centric_rate.py`,
`rust/runtime/src/user_centric.rs`.

**Target:** align budget validation and prefill admission while preserving
Rust's interruptible adaptive wake-up.

## P1.14 Worker-local cap slicing can exceed global caps

Python uses a global scheduler. Rust partitions work per thread and forces
some shard-local limits to at least one. If configured concurrency is below
worker count, aggregate concurrency can exceed the authored cap.

**Evidence:** `src/aiperf/timing/concurrency.py`,
`rust/runtime/src/engine/sharded_scheduled.rs`,
`rust/cli/tests/thread_per_core_product.rs`.

**Target:** allow zero-cap shards and prove exact global concurrency,
prefill, and request totals for every worker count.

## P1.15 URL selection restarts per Rust worker

Python's round-robin selector is global. Rust worker-local selectors can each
start from URL zero.

**Impact:** endpoint balance and request-to-endpoint assignment change with
worker count.

**Evidence:** `src/aiperf/timing/url_samplers.py`,
`rust/runtime/src/timing/url_selection.rs`,
`rust/runtime/src/ancillary.rs`.

**Target:** derive endpoint selection from a global dispatch ordinal or
partition one logical sequence.

## P1.16 Graph records omit Python branch lineage

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

Rust runtime supports abort-on-failure, but native projection does not always
carry the authored policy. Python exposes the setting but scheduled HTTP
execution largely behaves as continue.

**Evidence:** `src/aiperf/config/config.py`,
`rust/cli/src/load.rs`, `rust/runtime/src/failure.rs`,
`rust/runtime/src/request_rate.rs`.

**Target:** one typed policy carried through both frontends and tested with
identical partial and total failures.

## P1.18 Python outer loops do not consume protocol-v2 stdio

Python subprocess orchestration uses config files, inherited output, and
artifact heuristics. Native execution uses a strict stdin request and terminal
JSON envelope.

**Evidence:** `src/aiperf/orchestrator/local_executor.py`,
`src/aiperf/orchestrator/subprocess_runner.py`,
`rust/cli/src/execute.rs`,
`rust/cli/tests/protocol_v2_stdio.rs`.

**Target:** a Python executor adapter that sends the resolved request, parses
typed terminal responses, forwards progress, and loads the authoritative report
path.

---

# P1 transport and endpoint gaps

## P1.19 TLS trust semantics differ

Python aiohttp uses OpenSSL/system trust and a global verification environment
toggle. Rust HTTP uses rustls/WebPKI and endpoint policy. Neither exposes one
shared inference custom-CA/mTLS contract.

**Evidence:** `src/aiperf/transports/http_defaults.py`,
`tests/unit/transports/test_tcp_connector.py`,
`rust/runtime/src/transport/http/config/defaults.rs`,
`rust/runtime/tests/transport_http/tls.rs`.

**Target:** endpoint-scoped verification, system/custom trust selection, and
client certificate fields with matching semantics.

## P1.20 HTTP protocol, pool, and DNS behavior differs

Python is HTTP/1.1 with global environment defaults. Rust supports ALPN, h2c,
multiplexing, endpoint-scoped limits, and a different resolver strategy.

**Impact:** load shape and measured latency can change materially.

**Evidence:** `src/aiperf/transports/aiohttp_transport.py`,
`src/aiperf/transports/http_defaults.py`,
`rust/runtime/src/transport/http/client/`.

**Target:** make native endpoint-scoped policy canonical and document the
Python path as compatibility-only.

## P1.21 Redirect and non-2xx behavior differs

Python inference requests follow redirects by aiohttp default. Rust reports
3xx as errors. Error-body retention also differs.

**Evidence:** `src/aiperf/transports/aiohttp_client.py`,
`rust/runtime/src/transport/http/client/http_client.rs`,
`rust/e2e-tests/tests/test_error_fidelity.rs`.

**Target:** disable inference redirects and preserve the same status, headers,
body, and error fields.

## P1.22 Timeout and error taxonomies differ

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

Python and Rust disagree on zero-dimensional vectors and mixed-validity
`chat_embeddings` arrays. Both fail to preserve non-empty response usage in
the complete metrics path.

**Evidence:** `src/aiperf/endpoints/openai_embeddings.py`,
`src/aiperf/endpoints/chat_embeddings.py`,
`rust/runtime/src/endpoints/endpoints.rs`,
`rust/e2e-tests/tests/test_embeddings_endpoint.rs`.

**Target:** one vector validity contract and usage capture for both dialects.

## P1.25 Python image edit can miss multipart selection

Python endpoint metadata declares form-data requirements, but transport
selection can still send JSON unless content type is explicitly set. Rust
automatically selects and validates multipart.

**Evidence:** `src/aiperf/endpoints/openai_image_edit.py`,
`src/aiperf/transports/aiohttp_transport.py`,
`rust/runtime/src/endpoints/tier2.rs`,
`rust/e2e-tests/tests/test_image_edit_endpoint.rs`.

**Target:** endpoint metadata must select multipart in every execution path,
with wire-level tests for file bytes and malformed input.

## P1.26 Raw and template endpoint contracts are not portable

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

Python exposes environment settings but no server implementation or reliable
projection. Rust implements secure serving and publishing, but activation is
primarily protocol-side rather than normal profile authoring.

**Evidence:** `src/aiperf/common/environment.py`,
`rust/runtime/src/content_server/`,
`rust/cli/tests/online_v2_stdio.rs`.

**Target:** a typed `contentServer` Config v2 section lowered through the stock
profile path; remove inert environment settings.

## P1.28 Multimodal URL materialization occurs at different stages

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

## P1.29 Request-latency terminal boundary differs

Python ends request latency at the last meaningful content response. Rust can
use transport terminal time, including trailing usage or `[DONE]`.

**Impact:** latency and token-derived timing can be inflated on native runs.

**Evidence:** `src/aiperf/metrics/types/request_latency_metric.py`,
`src/aiperf/workers/worker.py`,
`rust/runtime/src/transport/http/sink/endpoint_dispatch.rs`,
`rust/runtime/src/metrics_core/store.rs`.

**Target:** record last meaningful response separately and use one documented
boundary for request latency.

## P1.30 Aggregate record eligibility differs

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

Examples:

- `session_num`: Python credit index versus Rust conversation index;
- `request_ack_ns`: Python response acknowledgment versus Rust first token;
- cancellation time: dedicated instant versus request end;
- worker and processor IDs: dynamic values versus constants;
- DAG parent/depth: omitted or hardcoded in Rust.

**Evidence:** `src/aiperf/common/models/record_models.py`,
`src/aiperf/records/record_processor_service.py`,
`rust/runtime/src/engine/records.rs`,
`rust/runtime/src/scheduled.rs`.

**Target:** restore legacy meanings or introduce a versioned schema with
unambiguous renamed fields.

## P1.32 Native raw traces and errors are lossy

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

Python can retain a key with `null`; Rust may omit it. Distribution sums,
top-level times, and branch statistics are also not consistently projected.

**Evidence:** `src/aiperf/exporters/metrics_json_exporter.py`,
`tests/unit/exporters/test_metrics_json_exporter.py`,
`rust/runtime/src/export/genai_perf.rs`.

**Target:** one schema version with explicit missing/non-finite policy and
complete run metadata.

## P1.34 Artifact policy is not fully projected

Native gaps include filename prefixing, summary disabling, user-file handling,
trace display options, and some overwrite rules.

**Evidence:** `src/aiperf/config/artifacts.py`,
`tests/unit/config/test_profile_export_prefix_scope.py`,
`rust/cli/src/flags.rs`, `rust/cli/src/load.rs`,
`rust/cli/src/yaml.rs`.

**Target:** type and validate every artifact policy field and require expected
artifacts to exist in E2E tests.

## P1.35 OTLP is deferred and narrower in Rust

Python periodically exports through the OTel SDK and includes additional
AIPerf/timing metrics. Rust performs a post-run request with a smaller GenAI
surface, different resource/provider inference, and no equivalent custom
header support.

**Evidence:** `src/aiperf/post_processors/otel_streaming_fanout.py`,
`src/aiperf/post_processors/otel_metrics_results_processor.py`,
`rust/runtime/src/export/otel.rs`.

**Target:** either keep the live sidecar authoritative or implement periodic,
authenticated, full-surface native export.

## P1.36 MLflow lifecycle and metadata are narrower in Rust

Native MLflow lacks parts of Python's run reuse, authentication/URI handling,
metadata sidecar, artifact breadth, and projected parameters.

**Evidence:** `src/aiperf/exporters/mlflow_data_exporter.py`,
`rust/runtime/src/export/mlflow.rs`,
`rust/cli/src/model/export.rs`.

**Target:** one run across live/final phases, full redacted parameters,
standard authentication, and the same artifact bundle.

## P1.37 W&B is offline-only and metadata-poor in Rust

Python creates a cloud run and artifact bundle. Rust writes an offline datastore
requiring later synchronization and can omit the redacted config and invocation
metadata.

**Evidence:** `src/aiperf/exporters/wandb_data_exporter.py`,
`rust/runtime/src/export/wandb/`,
`rust/cli/src/model/export.rs`.

**Target:** make online/offline mode explicit and preserve the same reproducible
artifact and metadata contract.

## P1.38 Telemetry configuration is only partially lowered

Native projection drops or hardcodes parts of local GPU collectors, dashboard
mode, metrics files, environment-tuned intervals, profile prefixes, sidecar
suppression, and mutex validation.

**Evidence:** `src/aiperf/config/flags/_converter_telemetry.py`,
`src/aiperf/config/gpu_telemetry.py`,
`rust/cli/src/model/telemetry.rs`,
`rust/cli/src/load.rs`, `rust/cli/src/yaml.rs`.

**Target:** carry the complete typed policy into sidecar specs and resolve
defaults in one place.

## P1.39 Server-metrics phase and histogram rules differ

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

Differences include:

- loader-preferred shuffle versus retained sequential sampling;
- trace offsets accepted but not always propagated;
- Rust-only per-turn endpoint/model/streaming/token-ID fields;
- generic CSV support only in Rust;
- coercion and row-validation differences.

**Evidence:** `src/aiperf/config/dataset/`,
`src/aiperf/dataset/loader/`,
`rust/runtime/src/dataset/loader/`,
`rust/runtime/src/dataset/sampler.rs`,
`rust/runtime/src/engine/execute.rs`.

**Target:** one row schema, loader-preference rule, trace window contract, and
paired loader corpus.

## P1.42 Public datasets have two unsynchronized catalogs

Python uses plugin metadata with alias normalization and runtime extension.
Rust embeds a JSON catalog with exact lookup. Keys currently align, but
capabilities, defaults, subsets, and future additions can drift.

**Evidence:** `src/aiperf/plugin/plugins.yaml`,
`src/aiperf/plugin/extensible_enums.py`,
`rust/cli/resources/public_datasets.json`,
`rust/cli/src/model/public_catalog.rs`.

**Target:** generate both representations from one catalog and test metadata,
aliases, errors, and capability checks.

## P1.43 Public dataset bounds and timing validation differ

Rust can inject a default 100-conversation limit where Python leaves a
non-streaming dataset unbounded, and only some Rust loaders consume that option.
Python validates fixed-schedule timing metadata before execution; Rust often
discovers it after loading.

**Evidence:** `src/aiperf/dataset/loader/base_hf_dataset.py`,
`src/aiperf/config/dataset/resolver.py`,
`rust/cli/src/model/public_catalog.rs`,
`rust/runtime/src/dataset/loader/`.

**Target:** centralize bound calculation and catalog capability validation.

## P1.44 Seeded synthetic datasets are not reproducible across engines

Namespaces, draw order, rounding, prefix generation, audio bounds, video
defaults, and ranking length generation differ.

**Evidence:** `src/aiperf/dataset/composer/synthetic.py`,
`src/aiperf/dataset/generator/`,
`rust/runtime/src/dataset/loader/synthetic.rs`,
`rust/runtime/src/dataset/generator/`.

**Target:** one namespace/draw-order specification and a full conversation/media
golden produced by both engines.

## P1.45 Native synthesize has a narrower artifact and validation contract

Python agentic-code synthesis writes quality and visualization artifacts and
enforces stricter turn/reset/restart feasibility. Rust writes the core dataset
and manifest but accepts combinations Python rejects.

**Evidence:** `src/aiperf/dataset/agentic_code_gen/`,
`rust/cli/src/synthesize/`,
`tests/unit/dataset/agentic_code_gen/test_writer.py`.

**Target:** choose one artifact set and validation contract, then enforce it in
both commands.

## P1.46 Synthesize parity goldens are absent

Rust tests reference expected files under `tools/parity/synthesize/`, but the
oracle files are not committed.

**Evidence:** `rust/cli/tests/synthesize_parity.rs`.

**Target:** generate Python-authoritative goldens, commit them, and require the
native parity tests in CI.

## P1.47 Tokenizer resolution policies differ

Python performs Hub alias search, ambiguity handling, prefetch, and broader
remote tokenizer behavior. Rust resolves repository IDs directly, rejects
remote-code tokenizers, and fetches a different pinned artifact set.

**Evidence:** `src/aiperf/common/tokenizer.py`,
`src/aiperf/common/tokenizer_validator.py`,
`rust/runtime/src/dataset/tokenizer.rs`,
`rust/runtime/src/engine/online_execution.rs`.

**Target:** implement or explicitly reject alias resolution, define the
revision-complete artifact set, and test recorded Hub fixtures.

## P1.48 Comprehensive RNG parity silently skips

The Python parity test points to a stale path. The committed vectors live under
the runtime crate, and no Rust test consumes the complete JSON vector set.

**Evidence:** `tests/unit/common/test_rng_parity.py`,
`rust/runtime/tests/data/rng_parity_vectors.json`,
`rust/runtime/examples/rng_parity_vectors.rs`.

**Target:** fix the path, fail instead of skipping in CI, and replay the same
vectors from both languages.

## P1.49 Path resolution and cache policy differ

Relative paths, config-directory anchoring, tilde expansion, symlink handling,
artifact creation timing, cache roots, offline behavior, and revision keys are
not consistent.

**Evidence:** `src/aiperf/config/resolution/`,
`src/aiperf/dataset/loader/base_public_dataset.py`,
`rust/cli/src/load.rs`, `rust/runtime/src/dataset/fetch.rs`,
`rust/runtime/src/engine/execute.rs`.

**Target:** specify path base/expansion per field and a shared cache root,
namespace, revision, and offline policy.

## P1.50 Kubernetes manifests require the native binary

Current manifests invoke `aiperf controller`, `cell`, `aggregator`, and
`results-sidecar`. Python-only images cannot provide these commands, while
several tests still expect the retired service-mesh topology.

**Evidence:** `src/aiperf/kubernetes/jobset.py`,
`src/aiperf/kubernetes/jobset_builder.py`,
`rust/cli/src/dispatch.rs`,
`rust/cli/src/cellular_role.rs`,
`tests/unit/kubernetes/test_jobset.py`.

**Target:** make native binary availability an explicit image preflight and
replace mesh-era test expectations.

## P1.51 Kubernetes progress and result lifecycle is incomplete

Confirmed issues include:

- progress body construction without production patch calls;
- best-effort CR writes with swallowed failures;
- processing markers read but not written natively;
- cancellation hardcoded false in native completion;
- no native gzip response negotiation;
- a retired shutdown endpoint still called by operator code.

**Evidence:** `rust/cli/src/k8s.rs`,
`rust/cli/src/cellular_role.rs`,
`rust/cli/src/results_sidecar.rs`,
`src/aiperf/kubernetes/results_sidecar.py`,
`src/aiperf/operator/handlers/monitor.py`.

**Target:** one marker/status lifecycle, periodic progress, cancellation
propagation, explicit CR durability policy, and one sidecar HTTP contract.

---

# P2 compatibility and edge gaps

## P2.1 Config interpolation and normalization are narrower in Rust

Python supports richer normalization, named-list Jinja lookup, and scenario
merging. Rust expansion is narrower.

**Target:** shared interpolation and normalization fixtures executed by both
frontends.

## P2.2 `config` command interfaces differ

The native and Python commands use different positional/flag forms and expose
different expand/init options.

**Target:** make the shipped native command a compatible superset.

## P2.3 Dry-run is native-only

Rust has an analytical dry-run transport and dedicated tuning flags. Python's
similarly named Kubernetes behavior is not the same product mode.

**Target:** expose dry-run in shared Config v2 and reserve distinct names for
manifest preview.

## P2.4 Proxy behavior differs when explicitly enabled

Defaults align: Python disables ambient proxies and Rust has no proxy layer.
With Python trust-env enabled, inference, telemetry, or media requests can use
ambient proxies while Rust always connects directly.

**Target:** standardize that benchmark and loopback traffic never use ambient
proxies, or define a typed proxy policy with mandatory loopback bypass.

## P2.5 Fixed-schedule filtering and continuation error policies differ

One engine can fail setup where the other omits a conversation, and continuation
delay anchoring or invalid metadata can produce different outcomes.

**Target:** shared offset/filter/continuation fixtures.

## P2.6 Cancellation reason and stuck-slot reporting differ

Rust has richer lifecycle completion reasons and RAII slot release. Python
synthesizes some recoveries and exports a smaller reason vocabulary.

**Target:** common exported lifecycle reasons while preserving internal
ownership designs.

## P2.7 Graph input DTOs accept different fields

The shared `dag_jsonl` name masks parser differences in endpoint overrides,
headers, roles, joins, non-finite delays, and empty roots.

**Target:** one strict DTO and pathological fixture corpus.

## P2.8 Per-record Parquet lacks a cross-engine contract

Parquet is native-only in practice, and its flat schema inherits record
metadata gaps.

**Target:** derive Parquet from the versioned record DTO and fail configuration
when the selected engine cannot emit it.

## P2.9 Native console lacks Python's live operational surface

Native output does not mirror Python realtime rows, server snapshots, GPU and
accuracy sections, quiet behavior, or cache hints.

**Target:** a common progress projection and explicit UI ownership.

## P2.10 Cache and media encoders are not byte-compatible

Python and Rust can produce different JPEG bytes and use different cache keys
for otherwise identical media or datasets.

**Target:** only require byte parity where artifacts promise it; otherwise
require semantic media equivalence and stable cache identity.

## P2.11 Mooncake validation strictness differs

Discriminator and numeric coercion rules are not identical.

**Target:** one shared strict trace schema and test corpus.

## P2.12 Inference retries are absent in both engines

This is a shared limitation rather than a migration regression. Retrying can
invalidate benchmark semantics if hidden.

**Target:** codify “no inference retries” or add explicit idempotency-aware
attempt policy with per-attempt records.

---

# Intentional architecture and capability differences

These differences should not be “fixed” by duplicating native systems in
Python. They require explicit ownership and stable compatibility boundaries.

## Native thread-per-core versus Python central credit scheduling

Python dynamically routes credits through central services. Rust partitions
budgets and runs worker-local schedulers. Required shared invariants are:

- exact global request/session budgets;
- global concurrency and prefill caps;
- deterministic endpoint assignment under a fixed seed;
- equivalent records and aggregate metrics.

Internal issue order does not need to be identical unless the product promises
it.

## Native Graph-IR versus Python runtime branch sessions

Python creates branch sessions at runtime and tracks explicit joins. Rust
lowers branches to static nodes/channels and executes whole traces. Required
shared invariants are authored input, payloads, failure policy, lineage,
records, and summaries—not internal state machines.

## Cellular execution is native-only

Python should validate and orchestrate cellular runs, not implement a second
cell transport. The parity target is consistent configuration, errors,
artifacts, and aggregate invariants with single-process execution.

## gRPC is native-only

Python has no current gRPC transport. KServe OIP and Riva should remain native.
The parity target is normalized endpoint behavior, records, metrics, and
errors—not a duplicate Python channel stack.

## KServe and Riva endpoint families are native-only

These are additional native capabilities, not regressions from current Python
shared endpoints. They require strong byte-level codecs and product E2E tests.

## Rust graph warmup handoff is native-only

Python configuration and reporting must understand the native outcome, but
Python need not reproduce the warmup state-transfer implementation.

## Default RNGs intentionally differ

Python defaults to its historical generators. Rust uses BLAKE3-derived PCG
streams. `AIPERF_RNG_BACKEND=rust` is the compatibility lane.

The migration contract must state which surfaces require parity mode. Default
cross-engine byte equivalence should not be claimed without enabling it.

---

# Shared defects and documentation-sensitive traps

The following are not simply “Rust missing Python”:

- Generic raw/template paths on both sides can drop usage.
- Image-generation output count is not the same as input-image metrics.
- Streaming image tutorial extraction assumptions do not match top-level
  streaming chunks.
- Both paths omit a complete retry-attempt model.
- Both contain configuration fields whose runtime meaning is incomplete.
- Some historical Python Riva implementations exist only on unmerged history
  and must not be treated as current product behavior.
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

`tests/unit/common/test_rng_parity.py` references the wrong committed vector
path and skips when it cannot find the file.

## Search parity disappears without `search-pyo3`

Isotonic and Bayes parity tests are feature-gated. The default build therefore
provides no signal for those planners.

## Latency parity is aggregate and tolerant

`rust/e2e-tests/tests/test_rust_python_latency_parity.rs` compares a narrow set of
averages. It does not protect per-record boundaries, percentiles, delayed usage,
reasoning, batching, or client-token semantics.

## Several parity-named tests compare only Rust paths

Fold, flatgraph, worker accumulation, and sweep aggregation tests provide useful
native invariants, but they do not establish Python-to-Rust parity.

## Tokenizer alias tests provide no live signal

The native alias-resolution E2E matrix is ignored and contains empty assertion
helpers.

## Synthesize goldens are missing

The byte-exact native test names expected Python-generated files that are not
present.

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

## Phase 1: close correctness blockers

Recommended order:

1. Fix output-token semantics and mixed tool-call content.
2. Version and align usage accounting.
3. Fix all-request-failed single-run status.
4. Complete endpoint policy projection.
5. Wire native accuracy.
6. Align record metadata meanings and graph lineage.
7. Enforce strict authoring validation.

These items directly affect whether a benchmark result can be trusted.

## Phase 2: close workload and artifact compatibility

1. Preserve complete phase semantics.
2. Resolve fixed-schedule and user-centric admission contracts.
3. Enforce global caps and deterministic endpoint selection across workers.
4. Align record eligibility and request-latency boundaries.
5. Version raw record, summary, CSV, and Parquet schemas.
6. Project full artifact policy.
7. Align telemetry windows and structured failures.

## Phase 3: migrate orchestration and outer loops

1. Implement the Python protocol-v2 executor adapter.
2. Move programmatic profile execution onto the native runner.
3. Adopt one sweep/search/trial plan.
4. Align seeds, cooldowns, convergence, ranking, and post-processing.
5. Retire service-mesh-only flags and environment variables.

## Phase 4: operational convergence

1. Decide native versus sidecar ownership for OTLP, MLflow, and W&B.
2. Align console/progress projection.
3. Complete Kubernetes progress, cancellation, marker, and sidecar contracts.
4. Unify public dataset catalogs, caches, paths, and tokenizer resolution.
5. Declare native-only gRPC, cellular, KServe, Riva, and graph execution
   boundaries.

---

# Definition of migration-ready

The Python execution path can be considered replaceable only when all of the
following are true:

- every supported Python config either executes equivalently in Rust or fails
  with an explicit migration error;
- no accepted CLI/YAML field is silently ignored;
- deterministic shared workloads produce equivalent request bodies, records,
  usage, errors, and summaries;
- output-token and request-latency semantics are engine-independent;
- single-run, multi-run, search, and cellular failures use a consistent exit
  contract;
- expected artifacts are required by tests, not conditionally inspected;
- accuracy grading is reachable from the stock profile path;
- graph records carry sufficient lineage for downstream reconstruction;
- exporter ownership and reduced native feature sets are explicit;
- required cross-language gates run in the default release CI configuration;
- remaining differences are listed as intentional native-only architecture,
  not implied parity.

Until those conditions hold, “native default” and “Python-compatible” should be
treated as separate claims.
