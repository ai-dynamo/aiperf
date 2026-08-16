<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recorded agent replay — native Rust implementation

## Purpose

This record specifies the native Rust implementation derived from a pinned
experimental branch in the sibling `dynamo-graph-ir` checkout.
The public AIPerf capability is named **recorded agent replay**. It makes a
mini-swe-agent performance-replay recording a first-class Graph-IR input and
preserves the source behavior: request-body replay, optional real tool
execution in local or Docker sandboxes, per-recording warmup, task-family
sampling, run-scoped cache isolation, scenario locks, timing artifacts, and an
independent reference/AIPerf A/B parity harness.

For the comparable default workload, the workload manifest—not directory
discovery—is the benchmark definition. It fixes ordered task selection,
adapter/environment family, request defaults, normalization targets, expected
corpus shape, and attribution. Raw recording files and directories remain
useful lower-level inputs, but they do not by themselves define a
`recorded-agent-default` result.

The design keeps the runtime's existing composition rules. Graph input,
lowering, tool dispatch, sandbox construction, time, placement, result folding,
and artifact export each have an injectable boundary. Tool nodes do not enter
the inference request-credit or request-record planes, and no per-command path
adds cross-thread shared-state contention.

The parity baseline is:

- Experimental AIPerf Python source at
  `244222b5999f48d89799f25ee946eedd81831117`.
- Experimental reference checkout `main` at
  `b8897f5de1664ad6de9cd669a96c3ba5d379e81e`.

Supplemental architecture research used Harbor Framework at
`a27e9c2ae10a31c40b2dcef33ef5486bce36e185`. Harbor is not a parity authority
or runtime dependency. Its relevant patterns are capability preflight before
environment spend; separate agent, environment, and trial lifecycles;
fresh/load/resume execution; provider-neutral action/observation records;
tool-call/result correlation; copied-context and continuation markers; and
distinct run, trajectory, and delegated-agent identities. This design adopts
those ideas through AIPerf-native registered traits, clocked dispatch, and
controller-owned folding rather than copying Harbor's Python object model.

Claims in this record were derived from executable source and tests at those
revisions. Prose in the source branch's handoff record is supporting context,
not authority where it disagrees with code.

The Rust design deliberately corrects source-branch limitations found in that
audit: it applies cache isolation to the first wire message as the reference
does; measures only after task-environment setup and warmup; carries a complete
per-task environment recipe instead of treating a Docker image as the whole
environment; folds supplements through workers and cells instead of letting
phase instances write a shared file; derives backend identity from actual
per-trace selection and reports heterogeneous runs as `mixed`; and makes
artifact/export failure explicit rather than warning and silently losing the
measurement. These are parity and runtime-integrity fixes, not optional scope
reductions.

## Built

The Rust runtime already supplies the prerequisites the port composes over:

- `Application` freezes a registered `GraphInputAdapterResolver`; the built-in
  resolver selects one strict adapter and produces a `GraphInputBundle`.
- `GraphInputBundle` carries complete `GraphTracePlan`s, one immutable
  content-addressed segment store, and static input metadata. This port widens
  the owned placement command to a `GraphTraceProgram` so task setup, excluded
  warmup, profiling, and teardown cannot be reordered across traces.
- `GraphTracePlan` is serde-ready and crosses local, thread-per-core, and
  cellular trace-placement boundaries as one owned command.
- `TracePlacement`, `GraphSink`, `PromptMaterializer`, node dispatch/failure
  policies, and `Clock` are injected, object-safe seams. Worker-local request
  execution uses `Rc`, `RefCell`, current-thread Tokio runtimes, and `LocalSet`.
- `GraphExecutionEvent` carries ordered worker facts to phase-local coordinator
  accounting. The graph execution path already folds request records before
  report and artifact finalization.
- The graph phase runtime provides warmup/profiling orchestration, cancellation,
  trace ownership, dataset sampling, t-star snapshot priming, and trace-level
  cellular partitioning.
- Raw recorded message arrays, tool definitions, extra request-body fields, and
  node-specific generation limits can be interned in the segment store and
  materialized without rebuilding their JSON shape.
- Config v2 models every cache-bust target, while the current graph execution
  projection recognizes only `none` and `first_turn_prefix`. The AgentX path
  already demonstrates system-prefix and system-suffix placement semantics.
- Cellular execution merges retained records or associative folded metric
  stores at the controller, and artifact shipping uses an allowlist derived
  from `ArtifactSpec`.

The native implementation supplies `LlmNode` and ordered `ToolNode` execution,
owned `GraphTraceProgram` lifecycle, a trace-local tool dispatcher and sandbox,
controller-folded replay supplements and metrics, and the strict
`agent_recording` input adapter. Recorded programs retain their environment,
workspace, response, and replay metadata through graph placement rather than
being reduced to a static projection.

## Supported behavior and non-goals

### Product behavior and non-goals

The native port implements the following behavior at the pinned Python branch
HEAD:

1. Read one mini-swe-agent recording, a shallow directory corpus, or an ordered
recorded-agent replay manifest.
2. Lower every usable model call into an exact Graph-IR LLM request.
3. Lower completed recorded commands into ordered tool nodes when real tool
   execution is enabled; otherwise represent their elapsed gap as edge timing.
4. Resolve and provision each task's complete environment recipe, including
   workspace fixtures, image, mount, working directory, interpreter, network,
   timeout, and command policy.
5. Issue one authored warmup after that task's environment is ready and
   immediately before that same task's profiling graph.
6. Apply task-family wire sampling and one run-scoped profiling cache-isolation
   prefix while leaving upstream-equivalent warmup unmodified.
7. Enforce the `recorded-agent-default` scenario as resolved configuration locks.
8. Emit normalized replay metrics, tool-time, trace-summary, provenance, and
   failure artifacts from merged profiling results.
9. Persist run identity for safe resume and out-of-process container cleanup.
10. Prove wire, lifecycle, environment, and report parity with unit, product
    E2E, and A/B tests.

The first delivery replays predetermined requests and recorded commands. It
does not expose a live-agent scenario, substitute live model replies into the
comparable workload, or grade SWE-bench or PinchBench tasks. It does, however,
freeze the trace-driver, response-selection, tool-call-decoding, and observation
formatting seams required for a future full agentic loop. A sandbox-resident
Rust executor binary is a separate optimization; the parity implementation is
host-resident and drives a persistent local shell or `docker exec` session.

Real tool execution is a real-clock online capability. Validation shall reject
it with `SimClock`, `dry_run`, `dynosim_offline`, or another virtual transport.
Recorded-delay replay without tool execution remains compatible with the
existing real and simulated graph paths.

### User and protocol configuration

The native CLI, Config v2 model, resolved `BenchmarkRun`, and strict protocol-v2
request shall expose one consistent vocabulary:

| CLI | Config v2 | Resolved meaning | Default |
|---|---|---|---|
| `--graph-format agent_recording` | `dataset.format: agent_recording` | Select the native recording/manifest adapter. | Format discovery where already supported; explicit under the scenario. |
| `--graph-replay-root <path>` | `dataset.graph.replay_root` | Root for manifest-relative recording paths and task-pack metadata/assets. | absent; required when a selected manifest contains relative external references |
| `--graph-execute-tools` | `dataset.graph.execute_tools` | Lower and execute completed recorded commands. | `false` |
| `--graph-tool-image <image>` | `dataset.graph.tool_image` | Fallback Docker image for a low-level trace recipe that has no adapter-resolved image; absent selects local execution only outside the scenario. | absent |
| `--graph-pinch-image <image>` | `dataset.graph.pinch_image` | Product-neutral PinchBench task image used by the stock PinchBench environment recipe. | `aiperf-recorded-agent-pinchbench:v1` under the default scenario; absent otherwise |
| `--graph-tool-command-timeout <seconds>` | `dataset.graph.command_timeout_seconds` | Per-command wall-clock ceiling when a node has no authored override. | `900.0` |
| `--graph-tool-container-stop-timeout <seconds>` | `dataset.graph.container_stop_timeout_seconds` | Bound Docker force-removal during recycle or close. | `5.0` |
| `--graph-tool-session-close-grace <seconds>` | `dataset.graph.session_close_grace_seconds` | Grace for a session shell to exit before its process group is killed. | `1.0` |
| `--graph-use-family-sampling` / `--no-graph-use-family-sampling` | `dataset.graph.use_family_sampling` | Overlay the recorded-agent replay profile. | `true` for this adapter |
| `--graph-emit-warmup` | `dataset.graph.emit_warmup` | Attach one excluded warmup plan to each trace program. | `false` |
| `--graph-resume` | `dataset.graph.resume` | Resume an interrupted manifest run from the same artifact root and run identity. | `false` |
| `--graph-stop-on-failure` | `dataset.graph.stop_on_failure` | Stop after the first failed manifest task instead of retaining later independent measurements. | `false` |
| `--hardware-description <text>` | `metadata.hardware` | Free-form endpoint hardware provenance; `unknown` is valid. | absent; required by the default scenario |
| `--endpoint-placement <mode>` | `metadata.endpoint_placement` | Declare `co_located`, `remote`, or `unknown` placement relative to tool execution. | `unknown` |

Existing graph input options continue to control delay suppression/capping,
record limits, dataset wrapping, recorded model selection, and recorded sampling
selection. The recorded-agent adapter shall reject unknown option keys and invalid
combinations rather than silently ignore them. All fields must project through
the native CLI and YAML surfaces into the same typed runner input.

`replay_root` is canonicalized once at input preparation. Relative recording,
task-manifest, task-file, and asset paths must remain beneath it; no current
working-directory fallback or ancestor-search heuristic is permitted. A
standalone recording that has no external workspace dependency does not require
the root.

`--graph-tool-image` without tool execution is rejected as inert configuration.
Tool execution plus open-loop graph replay is rejected: replaying recorded
tool gaps while also measuring real tool time would double-pace the trajectory.
All three timeout fields are positive finite seconds and project to integer
nanoseconds before worker construction. An authored `ToolNode.timeout_ns` wins
over `command_timeout_seconds`; manifest task-family recipes author 30 seconds
for PinchBench and 60 seconds for SWE-Bench, matching the pinned reference
runner, while a
raw recording without a recipe uses the 900-second low-level fallback.
Container-stop and session-close bounds apply only to cleanup/recycle and never
inflate the recorded command duration.
The values are injected into the sandbox factory rather than read from global
environment variables in the execution hot path.

The strict protocol shall add artifact paths for recorded-agent replay results:

- `graph_tool_time_path`, default
  `profile_export_graph_tool_time.json` when real tools are active.
- `graph_trace_summary_path`, default
  `profile_export_graph_trace_summary.json` for recorded-agent profiling.
- `graph_replay_metrics_path`, default `metrics.json` under the recorded-agent
  scenario, plus an optional `graph_replay_metrics_csv_path`.
- `graph_replay_failures_path`, default `failures.tsv`.
- `graph_replay_provenance_path`, default `replay-provenance.json`.
- `graph_replay_backend_metadata_path`, default `backend-metadata.json`.

Every path participates in path validation, cellular allowlisting, same-host
merge rules, sweep collection, and artifact manifests. The controller is the
sole writer of final merged files.

### `recorded-agent-default` scenario

The native scenario registry shall add `recorded-agent-default`. Resolution
auto-fills an unset value and reports every explicit conflict through the
existing scenario-outcome contract. Its locks are:

- graph workload and `dataset.format == "agent_recording"`;
- the versioned built-in `recorded-agent-eight-v1` workload identity, its exact
  manifest and decompressed-recording digests, explicit `replay_root`, manifest
  order, sequential sampling, one worker, one cell, one active trace, no wrap,
  no shuffle, and one pass;
- streaming enabled;
- server token counts enabled;
- no client input truncation;
- caller-selected model required; recorded model and recorded sampling disabled;
- standard resolved request profiles, empty debug extra-body override, positive
  recorded completion caps with fallback `32768`, per-inference timeout 300
  seconds, and one inference attempt;
- open-loop replay disabled;
- real tool execution enabled;
- recorded-replay trace driver and recorded assistant-response selection fixed;
- an environment recipe resolved for every task, Docker selected for every
  standard task, and the product-neutral PinchBench image fixed to
  `aiperf-recorded-agent-pinchbench:v1` with the canonical OCI digest;
- per-recording warmup enabled; and
- cache bust target `first_message_prefix` with run scope;
- fail-fast inference with one attempt and the raw OpenAI-compatible streaming
  timing transport;
- exact per-call records required for normalization and timing validation; and
- a non-empty free-form hardware description required; and
- default-run completeness required for a comparable/submission-valid result.

`recorded-agent-eight-v1` is registry-owned, not inferred from whatever
manifest the caller supplies. The implementation vendors the canonical
manifest plus a generated index of its BLAKE3 manifest-byte digest and each
selected decompressed-recording digest, the neutral PinchBench build-context
digest, and the expected OCI image digest. The scenario requires that index and
this exact ordered task identity vector:

```text
1. (pinchbench, pinchbench-openclaw, task_meeting_council_budget)
2. (pinchbench, pinchbench-openclaw, task_meeting_council_votes)
3. (pinchbench, pinchbench-openclaw, task_k8s_debugging)
4. (pinchbench, pinchbench-openclaw, task_meeting_searchable_index)
5. (pinchbench, pinchbench-openclaw, task_skill_search)
6. (swebench, swe-sample, django__django-15851)
7. (swebench, swe-corpus, django__django-14500)
8. (swebench, swe-corpus, sphinx-doc__sphinx-10614)
```

Its exact source shape is `total_isl = 2_499_441`, `isl_delta = 192_314`,
`peak_isl = 56_000`, `total_osl = 30_883`, `model_calls = 168`,
`tool_calls = 172`, `tool_duration_ms = 35_923.59589`,
`max_tool_call_duration_ms = 4_312.283731`, and
`timed_out_tool_calls = 0`. A different task, order, manifest, recording
payload, or source shape may run through the low-level adapter, but it cannot
resolve as this scenario or become submission-valid. The checked-in fixture is
the digest authority, so the design contains no hand-copied hash that can drift
from its bytes.

Default validation fails closed. If the existing `unsafe_override` contract is
used, bypassable conflicts may proceed only with
`submission_valid == false`, the complete violation list, and stable invalid
reason tags. Capability incompatibilities such as a virtual clock with real
tools remain hard failures and cannot be downgraded.

Debug overrides for request fields, timing transport, replay step/call caps,
fallback generation cap, resume, multi-worker/cellular placement, or task
container networking must be explicit, appear in provenance, and make the
result non-comparable even when execution is otherwise valid. Diagnostic-only
raw exchange dumps and additional raw timing columns do not change the workload
but are labeled sensitive or nonstandard output.

`ScenarioSpec` and its lock input DTO shall be widened generically rather than
special-casing recorded-agent replay in CLI resolution. Existing scenarios
retain their current defaults and behavior. Cache-bust projection shall add the
semantically exact `first_message_prefix` target; it must not misuse
`system_prefix`, silently map the new target to `none`, or change existing
targets.

### Manifest and recording input contract

The preferred comparable input is a recorded-agent replay manifest. Its strict
core is:

```text
RecordedAgentReplayManifest
  name: non-empty string
  mode: "replay"
  defaults: ReplayDefaults
  aggregate: ExpectedCorpusShape
  tasks: non-empty ordered Vec<ManifestTask>
  attribution: JSON object

ManifestTask
  adapter: "pinchbench" | "swebench"
  family: non-empty string
  task_id: non-empty string
  recording: path beneath replay_root
  primary_role: optional string

ReplayDefaults
  config: non-empty string
  step_limit: positive integer
  cost_limit: finite non-negative number
  environment_class: non-empty string
  docker_network: "none"
  per_inference_timeout: positive finite seconds
  fallback_max_output_tokens: positive integer
  temperature: finite non-negative number
  top_p: finite number in [0, 1]
  top_k: positive integer
  min_p: finite number in [0, 1]
  stream_for_timing: bool
  raw_openai_stream_for_replay_timing: bool
  replay_max_tokens_from_recording: bool
  replay_max_tokens_margin: non-negative integer
  extra_request_body: JSON object
  cross_run_cache_isolation: bool
  warmup: bool
  measurement_scope: "agent_run_only"

ExpectedCorpusShape
  total_isl: non-negative integer
  isl_delta: non-negative integer
  peak_isl: non-negative integer
  total_osl: non-negative integer
  model_calls: non-negative integer
  tool_calls: non-negative integer
  tool_duration_ms: finite non-negative number
  max_tool_call_duration_ms: finite non-negative number
  timed_out_tool_calls: non-negative integer
```

Unknown descriptive manifest fields are retained as provenance. Unknown task
adapters, duplicate `(adapter, task_id)` identities, duplicate resolved
recording paths, missing recordings, path escapes, and an empty task set fail
before warmup. Manifest task order is normative and survives lowering,
placement, reporting, and resume.

All fields above are required in the preferred manifest; the default scenario
also requires their canonical values: `config = "mixed"`, `step_limit = 150`,
`cost_limit = 0`, `environment_class = "mixed"`, `docker_network = "none"`,
`per_inference_timeout = 300`, `fallback_max_output_tokens = 32768`,
`temperature = 0.7`, `top_p = 0.8`, `top_k = 20`, `min_p = 0`, both stream
flags true, recorded caps enabled with margin zero, an empty extra body, cache
isolation and warmup enabled, and measurement scope `agent_run_only`.

The loader recomputes source-recording model-call count, eligible tool-call
count, total ISL, ISL delta, peak ISL, total OSL, total recorded tool duration,
maximum recorded tool duration, and timed-out tool-call count. Integer fields
compare exactly. Duration fields compare in integer nanoseconds after decoding
where the recording supplies nanoseconds; decimal manifest milliseconds compare
with an absolute `1e-6 ms` tolerance solely for JSON decimal conversion. These
are source-corpus integrity checks, not expectations for the new runtime's live
durations. A BLAKE3 digest of the manifest bytes and every selected decompressed
recording is stored in run provenance.

The adapter id is `agent_recording`. It accepts:

- a `.json` recording;
- a `.json.gz` recording; or
- a directory whose direct children are inspected in stable sorted order.

Directory traversal is intentionally one level deep. A candidate is accepted
only when content sniffing finds a `format` beginning with
`mini-swe-agent-recording-`; unrelated JSON such as a corpus manifest is
skipped. An explicitly named malformed or non-recording file is an error. An
empty corpus is an error. Duplicate resolved trace ids are an error.

The external DTO layer shall model the replay-relevant envelope while allowing
unknown upstream fields for compatible 1.x evolution:

```text
RecordedAgentRecording
  format: String
  metadata:
    instance_id: Option<String>
    task_id: Option<String>
    benchmark: Option<String>
    model_name: Option<String>
    docker_image: Option<String>
    manifest: Option<String>
    instance: Option<JSON object>
  events: Vec<RecordedEvent>

RecordedEvent
  id: non-negative integer
  type: String
  timestamp: positive finite seconds
  step: Option<non-negative integer>
  duration_ns: Option<non-negative integer>
  provider_request: Option<ProviderRequest>
  response_message: Option<ResponseMessage>
  action: Option<JSON object>
  error: Option<JSON object>

ProviderRequest
  messages: Vec<JSON object>
  tools: Option<Vec<JSON object>>
  model: Option<String>
  temperature: Option<finite non-negative number>
  top_p: Option<finite number in [0, 1]>
  max_tokens: Option<positive integer>
```

The nested response metadata shall extract optional non-negative
`usage.completion_tokens`. Recorded endpoint credentials, base URLs, retries,
and timeouts are never modeled or forwarded.

An event timestamp is the event end time. Its start is
`timestamp - duration_ns / 1e9`; all derived values must remain finite. A model
call is usable only when its event is successful, `provider_request` is present,
and the exact message array is available. A missing or failed model call causes
the recording to fail validation rather than producing a shortened plausible
trajectory.

A tool command is eligible when the event is a `tool_call`, its `action.command`
is a string, and it completed. Absence of `error` means completion. The upstream
agent control-flow types `InterruptAgentFlow`, `Submitted`, `LimitsExceeded`,
`ReplayExhausted`, `UserInterruption`, and `FormatError` also mean the command
ran to completion and must be retained. Other recorded errors exclude that
command from real replay.

Trace id is the first non-empty value in `metadata.instance_id`,
`metadata.task_id`, `metadata.instance.instance_id`, and the file name with
`.json` or `.json.gz` removed. The SWE image precedence is a non-empty
`metadata.docker_image`, then `metadata.instance.image_name`, then
`metadata.instance.docker_image`, then the deterministic image derived from
`metadata.instance.instance_id`; a low-level run-level fallback follows only
when none of those are available. A PinchBench manifest task uses
`dataset.graph.pinch_image`. File and decode errors must include the candidate
path; event errors must include trace id, event id, and event type.

### Pure lowering and segment identity

Decoding/discovery and lowering shall be separate modules. The lowerer consumes
validated recording DTOs and returns a deterministic bundle without filesystem,
network, process, clock, or artifact side effects. It is unit-testable with an
in-memory `SegmentPool`.

Every model call becomes one `LlmNode` in recorded order. Its
`provider_request.messages` objects are interned verbatim as a parent-chained
path. BLAKE3 prefix-dependent identifiers deduplicate common message prefixes
within a trace and across the full corpus while materializing from the stored
serialized bytes. Message key order, extra keys, structured content, assistant
tool calls, and tool-call ids survive unchanged.

The node carries:

- the interned message path;
- an optional strict `LlmRequestSpec` containing the recorded `tools` array
  handle, optional recorded-model override, and a handle to validated additional
  body fields;
- streaming from resolved run configuration;
- a per-call generation cap equal to positive recorded completion tokens, or
  the resolved fallback cap (`32768` for the default workload) when completion
  usage is absent or zero;
- the recorded model only when `use_recorded_model` is enabled; and
- recorded `temperature`/`top_p` only when `use_recorded_sampling` is enabled.

The stock SWE request profile also retains its pinned
`repeat_penalty: 1.05` additional-body field alongside the standard tool and
sampling fields; exact replay tests reject a missing or changed value.

Recorded sampling is provenance, not recorded-agent playback configuration. An
injectable `ReplayRequestProfileResolver` resolves one typed profile from the
manifest adapter/family, scenario, endpoint dialect, and explicit debug
overrides. The pinned reference executable runner is the parity authority:

- `swebench` sends `temperature = 0.7`, `top_p = 0.8`, `top_k = 20`,
  `min_p = 0`, and `parallel_tool_calls = true`;
- `pinchbench` sends no explicit server-side sampling fields from its standard
  adapter configuration.

The reference manifest currently advertises the numeric sampling defaults at
manifest scope while its runner applies them explicitly only to SWE-Bench. The
Rust port records this source inconsistency in provenance and follows executable
wire behavior until the upstream runner changes. `drop_params` remains a
LiteLLM-only client option and is never sent.

When `use_recorded_sampling` is explicitly enabled outside the standard
scenario, recorded sampling replaces corresponding profile fields. A validated
`extra_request_body` is merged last and may replace non-reserved request fields.
It may not contain `api_base`, `api_key`, `custom_llm_provider`, `max_tokens`,
`messages`, `model`, `stream`, `stream_options`, `timeout`, or `tools`. Unknown
families outside a manifest emit one structured warning and send no inferred
family fields.

`LlmRequestSpec` is a generic Graph-IR request extension, not a
recorded-agent-specific metadata bag. Its additional-body object cannot contain
messages, tools, model, generation-cap, streaming, transport, credential, or
routing keys, which retain typed ownership. Existing graph fixtures decode with
an empty/default request spec. Request fields are serialized once into the
segment store and materialized from those bytes without JSON round-tripping.

The existing message-only `PromptMaterializer` remains responsible for dynamic
message assembly. A registered `GraphRequestMaterializer` composes those
messages with `LlmRequestSpec` into a `MaterializedGraphRequest`; `GraphSink`
dispatch is widened to accept that request. Endpoint dialect code remains the
only layer that maps typed core fields to wire spellings. This seam makes tools
and backend fields injectable/testable without teaching the executor or
transport about this recording format.

Recordings are independent task runs and may span months. The lowerer therefore
discards absolute Unix start times across files. It retains only non-negative
relative node offsets and within-trace gaps. This prevents corpus-wide open-loop
pacing from parking later recordings far in the future.

Without real tools, the positive gap between one model-call end and the next
model-call start becomes the dependency edge delay, subject to existing ignore
and cap options. With real tools, completed tool calls between two model calls
become one ordered `ToolNode`; the recorded gap is omitted so host execution
time is not counted twice. Completed calls after the final model call become a
terminal tool node. Every node output channel is declared, including an output
that no successor reads.

### Heterogeneous Graph-IR model

The canonical node map shall become:

```rust
pub enum ExecutableGraphNode {
    Llm(LlmNode),
    Tool(ToolNode),
}

pub struct ToolNode {
    pub output: String,
    pub commands: Vec<String>,
    pub timeout_ns: Option<u64>,
}

pub struct LlmRequestSpec {
    pub tools: Option<Handle>,
    pub model: Option<String>,
    pub additional_body: Option<Handle>,
}
```

The owned placement unit shall be widened from a bare plan to:

```rust
pub struct GraphTraceProgram {
    pub profiling: GraphTracePlan,
    pub warmup: Option<GraphTracePlan>,
    pub environment: Option<TraceEnvironmentSpec>,
    pub replay: Option<ReplayTraceMetadata>,
    pub driver: TraceDriverSpec,
}
```

`ReplayTraceMetadata` contains the manifest ordinal and identities, source and
normalization-target digests, per-call target OSL, expected call counts, request
profile identity, and comparability annotations. It contains no credentials.
Generic graph adapters populate `warmup`, `environment`, and `replay` with
`None` and select the built-in static-graph driver, preserving their behavior.
The recorded adapter selects the built-in recorded-replay driver.

`TraceEnvironmentSpec` is strict and transportable across worker/cell
boundaries. It names a registered environment-recipe kind plus validated data;
it does not contain an open process, host-only temporary path, or trait object.
SWE image resolution follows the full top-level, nested image-name, nested
Docker-image, derived-name, then low-level-fallback precedence defined by the
input contract. The manifest PinchBench recipe uses its configured fixed image.
Resolution must finish before placement preflight.

The serde representation must be explicitly tagged and reject an unknown node
kind. Existing LLM-only serialized fixtures remain readable through a deliberate
compatibility rule rather than an untagged ambiguous enum.

`ExecutableGraphNode` shall provide kind-specific accessors for read channels, write
channels, static request count, and topology validation. Tool nodes write one
observation channel and consume no request credit. The first implementation
uses predetermined recorded commands; future live-reply tool dispatch can add
read dependencies without changing the dispatcher seam.

Every consumer of `GraphRecord.nodes` must match exhaustively. The audit includes
topology validation, entry detection, producer counts, scheduler construction,
executor firing, graph merge, snapshot/t-star rewrite, warmup handoff,
prefix-cache analysis, dataset analysis, metadata stripping, graph sidecars,
flat-graph eligibility, node counting, placement progress, dynosim conversion,
and serde tests. The flat fast path is LLM-only; the presence of a tool node
selects the general executor.

Request counts and progress accounting use LLM-node count, not total graph-node
count. A tool-only terminal does not create a missing-record infrastructure
failure. Separate `llm_node_count()` and `total_node_count()` APIs make this
distinction explicit.

### Trace driver and future live-agent loop seam

Placement owns lifecycle and measurement, while an injected trace driver owns
how turns are selected. This prevents a future live loop from being embedded in
the scheduler, endpoint transport, or sandbox:

```rust
#[async_trait(?Send)]
pub trait TraceProgramDriver {
    async fn run(
        &mut self,
        program: &GraphTraceProgram,
        context: &TraceDriverContext<'_>,
    ) -> Result<TraceTerminalSupplement, TraceDriverError>;
}

pub trait TraceProgramDriverFactory: Send + Sync {
    fn capabilities(
        &self,
        spec: &TraceDriverSpec,
    ) -> Result<TraceDriverCapabilities, TraceDriverError>;

    fn create(
        &self,
        worker: WorkerIdentity,
        trace: &TraceIdentity,
        spec: &TraceDriverSpec,
    ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError>;
}

pub trait AgentTurnCoordinator {
    fn on_assistant_response(
        &mut self,
        context: AgentTurnContext<'_>,
        live: LiveAssistantResponse,
        recorded: Option<RecordedAssistantTurn>,
    ) -> Result<AgentTurnDecision, AgentLoopError>;

    fn on_tool_results(
        &mut self,
        context: AgentTurnContext<'_>,
        results: &[ToolDispatchResult],
    ) -> Result<AgentTurnDecision, AgentLoopError>;
}

pub trait AgentTurnCoordinatorFactory: Send + Sync {
    fn create(
        &self,
        worker: WorkerIdentity,
        invocation: &AgentInvocationIdentity,
        spec: &AgentTurnCoordinatorSpec,
    ) -> Result<Box<dyn AgentTurnCoordinator>, AgentLoopError>;
}

pub trait AgentToolCallDecoder {
    fn decode(
        &self,
        response: &LiveAssistantResponse,
    ) -> Result<Vec<AgentToolCall>, AgentLoopError>;
}

pub trait AgentObservationFormatter {
    fn format(
        &self,
        calls: &[AgentToolCall],
        results: &[ToolDispatchResult],
    ) -> Result<Vec<Bytes>, AgentLoopError>;
}

pub trait AgentResponseStore {
    fn intern(
        &mut self,
        source: AgentResponseSource,
        wire: Bytes,
    ) -> Result<AgentResponseHandle, AgentLoopError>;

    fn get(&self, handle: &AgentResponseHandle) -> Result<Bytes, AgentLoopError>;
}

pub trait AgentResponseStoreFactory: Send + Sync {
    fn create(
        &self,
        trace: &TraceIdentity,
    ) -> Result<Box<dyn AgentResponseStore>, AgentLoopError>;
}

pub trait AgentTrajectorySink {
    fn record(&mut self, event: AgentTrajectoryEvent) -> Result<(), AgentLoopError>;
}

pub trait AgentTrajectorySinkFactory: Send + Sync {
    fn create(
        &self,
        trace: &TraceIdentity,
    ) -> Result<Box<dyn AgentTrajectorySink>, AgentLoopError>;
}

pub trait AgentTrajectoryCodec: Send + Sync {
    fn decode(
        &self,
        bytes: Bytes,
    ) -> Result<NormalizedAgentTrajectory, AgentLoopError>;
}

#[async_trait(?Send)]
pub trait AgentInvocationLease {
    fn dispatcher(&self) -> Rc<dyn ToolDispatcher>;
    async fn close(&mut self) -> Result<(), AgentLoopError>;
}

#[async_trait(?Send)]
pub trait AgentInvocationLeaseFactory {
    async fn open(
        &self,
        request: &AgentInvocationRequest,
        parent: Option<&dyn AgentInvocationLease>,
    ) -> Result<Box<dyn AgentInvocationLease>, AgentLoopError>;
}

pub struct TraceDriverCapabilities {
    pub has_live_turns: bool,
    pub has_load: bool,
    pub has_resume: bool,
    pub has_response_reuse: bool,
    pub has_branching: bool,
    pub has_delegation: bool,
}

pub enum AgentContinuationSpec {
    Fresh,
    Load { trajectory: Handle },
    Resume { checkpoint: Handle },
}

pub enum AssistantSelection {
    Inline {
        source: AgentResponseSource,
        wire: Bytes,
    },
    Reuse(AgentResponseHandle),
}
```

`TraceDriverSpec` and `AgentTurnCoordinatorSpec` are strict serializable
registry ids plus validated data, not trait objects. Driver, coordinator,
tool-call decoder, observation formatter, response-store, trajectory-sink, and
trajectory-codec registrations are frozen in `Application`. Each admitted trace
gets a fresh driver, response store, and trajectory sink; each root or delegated
invocation gets a fresh coordinator. They are never shared among concurrent
`execute_trace` futures on one worker and may be `!Send`.
`TraceDriverContext` borrows the normal request materializer, `GraphSink`,
root invocation lease, invocation-lease factory, response store, trajectory
sink, `Clock`, cancellation token, observer, step/cost/time budgets, and segment
store. A driver cannot open its own HTTP client, process, timer, metrics sink, or
workspace, so static replay and a live loop retain the same operational seams.

`TraceDriverCapabilities` declares support for live turns, loading an authored
trajectory, resuming a driver checkpoint, branching/reusing responses, and
delegating subagents. `TraceDriverSpec` carries an `AgentContinuationSpec` of
`Fresh`, `Load { trajectory }`, or `Resume { checkpoint }`. Composition checks
the requested mode and every sandbox/driver capability before workspace or
container creation; unsupported combinations fail rather than silently start a
fresh loop. Recorded replay is always `Fresh` at runtime even though its source
trajectory is input data, not resumable agent state.

Loading first normalizes the selected authored or native format through a
registered `AgentTrajectoryCodec`; the driver never parses another agent's
private session files. The normalized form preserves assistant responses,
tool-call ids, observations, copied-context markers, and continuation identity,
then validates it before any bytes enter `AgentLoopState`.

The worker-local `AgentResponseStore` interns response bytes by BLAKE3 and
returns typed handles. Reuse is always an explicit turn decision with source
`Live`, `Recorded`, `Loaded`, or `Reused { original_turn }`; it is never an
implicit endpoint cache. Provenance records the source handle and logical turn,
while copied context is marked so the reused response is not attributed as a
new generation, model call, completion cost, training target, or replay target.
Every later dispatch still measures and counts the complete actual wire prompt,
including reused/copied bytes, in observed ISL, server usage, request cost, and
cache metrics. Cross-run response reuse is disabled unless a future scenario
explicitly authors and validates it.

`AgentTurnContext` gives the per-invocation coordinator mutable access to that
trace's response store and trajectory sink plus read-only budgets, invocation
identity, and logical-turn identity. The coordinator interns a new live or
recorded response before returning `AssistantSelection::Reuse(handle)`, or
returns `AssistantSelection::Inline { source, wire }` for a new value. Before
append, the driver interns an inline value through the same store; the sink then
records its resulting handle/source exactly as it does for reuse. A reuse
decision therefore expresses its original turn directly and cannot be confused
with newly received bytes.

`AgentTrajectorySink` receives incremental typed events before the next turn so
a timeout or cancellation retains the last complete decision. Events
distinguish run id, trajectory-document id, agent-invocation id, parent
invocation, logical turn, model-call count, response handle/source, decoded
tool-call ids, correlated observation results, metrics, terminal reason, and
copied-context status. Tool results must reference a tool-call id from the same
logical turn. Continuations and delegated agents use trajectory ids or explicit
artifact references as resolution keys; shared run ids are correlation metadata
and never ambiguous lookup keys. The stock sink folds bounded summaries through
normal graph events; an optional trajectory artifact exporter owns full event
serialization.

`LiveAssistantResponse` owns the endpoint's parsed envelope and canonical
assistant-message bytes serialized once by the endpoint dialect reducer.
`RecordedAssistantTurn` likewise references the recording's interned
`response_message` bytes. An
`AgentTurnDecision` is one of `Continue { assistant, tool_calls }`,
`Retry { feedback }`, `Delegate { invocations, join_policy }`,
`Complete { assistant }`, or `Fail`; `assistant` is an `AssistantSelection`.
Appending or reusing a response therefore does not require JSON
re-serialization and cannot reorder keys or alter tool-call ids. Delegated
invocations carry distinct invocation and trajectory ids plus an
explicit shared-versus-isolated environment choice. The initial implementation
may reject delegation through its capability declaration; the driver contract
does not need to change when a registered coordinator later implements
sequential or bounded parallel delegation and deterministic join ordering.

Delegation is executed only through `AgentInvocationLeaseFactory`. A shared
request returns a scoped lease over the parent's already-open dispatcher and
sandbox; an isolated request asks lifecycle composition to provision a child
workspace and sandbox through the registered provisioner and sandbox factory.
The driver cannot construct either itself. Preflight validates the selected
join policy, sandbox sharing/isolation capability, and child resource bound.
Every child lease closes before its parent and before trace cleanup; failure to
open or close is represented in the parent trajectory with the child invocation
identity. This makes one-sandbox recorded replay unchanged while leaving an
implementable ownership path for future sequential or bounded-parallel agents.

The recorded-replay driver measures every live endpoint response but uses the
pre-lowered successor request, matching recorded assistant bytes when present,
and predetermined recorded commands for all downstream state. A recorded
assistant message is interned once and reusable by the turn seam; successor
request bytes remain authoritative if an older recording lacks that optional
response field. Live semantic content cannot change request count, commands, or
prompts, preserving comparable replay. It still retains the live response for
normal measurements and optional sensitive diagnostics. The standard scenario
locks this response policy and rejects any live selection.

A future live-agent driver can instead select the live bytes, use the
endpoint-dialect `AgentToolCallDecoder`, dispatch decoded calls through the same
`ToolDispatcher`, format tool observations through the injected
`AgentObservationFormatter`, append assistant and observation segments, and
iterate until the coordinator returns complete or a shared budget/cancellation
policy terminates it. The seam must support zero or multiple tool calls,
parallel-tool declarations with deterministic observation ordering, direct
final answers, response reuse across retries or branches, and tool/sandbox
failure feedback. No public live-agent scenario is part of this port; a fake
driver/coordinator test proves the complete response-to-tool-to-observation-to-
next-request cycle and exact byte reuse before the seam is considered stable.

Each live invocation owns an `AgentLoopState`: an ordered parent-chained message
path, response handles, tool-call/result correlations, budgets, terminal state,
and child invocation references. The driver creates subsequent requests by
passing that path plus a typed LLM request template through the existing
`GraphRequestMaterializer`; it does not mutate a static `GraphTracePlan` or
invent a second wire builder. Dynamic turns still consume ordinary request
credits, observer events, records, and cancellation gates.

### Injectable request, environment, tool, and sandbox seams

Manifest interpretation must not hard-code Docker or benchmark-family logic in
the graph executor. Product composition registers these seams:

```rust
pub trait ReplayRequestProfileResolver: Send + Sync {
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<ReplayRequestProfile, RecordedAgentLoweringError>;
}

pub trait GraphRequestMaterializer {
    fn materialize(
        &self,
        node: &LlmNode,
        messages: Vec<Bytes>,
    ) -> Result<MaterializedGraphRequest, GraphRequestMaterializationError>;
}

pub trait TraceEnvironmentResolver: Send + Sync {
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<TraceEnvironmentSpec, TraceEnvironmentError>;
}

#[async_trait(?Send)]
pub trait WorkspaceProvisioner {
    async fn provision(
        &self,
        spec: &WorkspaceSpec,
    ) -> Result<ProvisionedWorkspace, TraceEnvironmentError>;
}

pub trait ToolCommandPolicy {
    fn evaluate(
        &self,
        command: &str,
    ) -> Result<CommandDisposition, TraceEnvironmentError>;
}
```

Names may adapt to established module vocabulary. Resolution and provisioning
are separate: the resolver is pure and serializable; the worker-local
provisioner performs filesystem work before measurement. `CommandDisposition`
either executes the command or returns a specified synthetic command result.
It is not a boolean hook whose rejection semantics vary by caller.
`GraphRequestMaterializer` is worker-local and receives already assembled
message bytes; it resolves node-owned handles without rebuilding their JSON.

Tool dispatch mirrors LLM dispatch but remains a distinct plane:

```rust
#[async_trait(?Send)]
pub trait ToolDispatcher {
    async fn open_trace(&self, context: TraceOpenContext<'_>) -> Result<(), ToolDispatchError>;
    async fn dispatch(
        &self,
        request: ToolDispatchRequest,
        context: &ToolDispatchContext,
    ) -> Result<ToolDispatchResult, ToolDispatchError>;
    async fn close_trace(&self, trace: &TraceIdentity) -> Result<(), ToolDispatchError>;
}

pub trait ToolDispatcherFactory: Send + Sync {
    fn create(&self, worker: WorkerIdentity) -> Result<Rc<dyn ToolDispatcher>, ToolDispatchError>;
}

#[async_trait(?Send)]
pub trait ToolSandbox {
    async fn open(&self) -> Result<(), ToolSandboxError>;
    async fn run(
        &self,
        command: &str,
        timeout_ns: Option<u64>,
    ) -> Result<ToolCommandResult, ToolSandboxError>;
    async fn close(&self) -> Result<(), ToolSandboxError>;
}

pub trait ToolSandboxFactory {
    fn capabilities(&self) -> ToolSandboxCapabilities;

    fn create(
        &self,
        context: SandboxCreateContext<'_>,
    ) -> Result<Rc<dyn ToolSandbox>, ToolSandboxError>;
}
```

Names may adapt to established module vocabulary, but the ownership and
injection boundaries are normative. `ToolDispatcherFactory` is application or
phase composition state and may be `Send + Sync`; the created dispatcher and
sandbox are worker-local `Rc` handles and need no `Send`/`Sync` supertraits.
Constructors accept unwrapped owners; they do not impose `Arc<Mutex<_>>`.
`TraceOpenContext` and `SandboxCreateContext` borrow the trace identity,
resolved environment recipe, provisioned workspace, and opaque run label; they
do not re-resolve metadata or read global environment variables.

`ToolDispatchRequest` is a strict enum with a recorded batch carrying static
`ToolNode` provenance and a live batch carrying decoded `AgentToolCall`s. The
dispatcher, rather than the graph executor, lowers either form through the
selected tool registry and command policy. The current shell implementation
accepts only the recorded command form and advertises that capability; an
unsupported live function name becomes a correlated tool observation when the
policy classifies it as an agent-visible error, or a `ToolDispatchError` when
dispatch infrastructure itself is unavailable.

`ToolSandboxCapabilities` states whether a backend provides persistent
workspace state, file upload/materialization, network disablement, network
allowlists, dynamic phase policy, command timeout/descendant termination, and
shared or isolated delegated-agent sessions. Environment resolution computes
requirements from the recipe and selected trace driver, then validates them
against capabilities before provisioning. A provider that cannot enforce a
required network or isolation policy rejects the run; it never approximates the
policy. This keeps a future remote sandbox factory compatible without changing
the tool or agent-loop contracts.

`ToolDispatchResult` contains the combined observation, one
`ToolCommandResult` per attempted command in execution order, and a timeout
summary. Each command result contains combined stdout/stderr, return code,
finite Clock-derived duration, and `is_timed_out`. The node-level timeout is
applied independently to every command. Nonzero exit and timeout are successful
dispatch outcomes. Failure to create, communicate with, or recover a sandbox is
a typed infrastructure error that aborts the trace.

The stock dispatcher owns one sandbox per unique trace instance, not per trace
template. Concurrent repetitions of the same recording therefore receive
different workspaces and process/container identities. It executes commands in
node order and concatenates observations in command order. After a command
timeout, a successful session/container recycle permits the next recorded
command to run. The batch stops only when recycle fails, which is a sandbox
infrastructure error rather than a timeout outcome.

Tests inject a deterministic fake dispatcher or fake sandbox directly. Product
composition registers local and Docker factories without teaching the graph
executor which backend is selected. This seam is also sufficient for a future
remote dispatcher or live-agentic dispatcher.

### Trace lifecycle, measurement, and cleanup

For each admitted `GraphTraceProgram`, one placement target executes this
indivisible lifecycle before the next manifest task is admitted:

1. Resolve the unique execution identity and provision its workspace.
2. Create and open its dispatcher/sandbox outside measurement.
3. Dispatch its one timing-only warmup through the normal endpoint transport
   while the task sandbox exists; retain warmup diagnostics but exclude its
   records from profiling.
4. Capture `trace_start_ns` from the injected `Clock` immediately before the
   first profiling graph node can fire.
5. Execute profiling LLM and tool nodes through their independent dispatch
   seams.
6. Capture `trace_end_ns` after the terminal profiling node drains.
7. Close the sandbox outside measurement on every terminal path.
8. Run any explicitly configured post-run diagnostic/grading hook outside
   measurement; grading is not required by the Rust replay port.
9. Emit one terminal result supplement and atomic resume checkpoint after
   cleanup has been attempted.

Phase identity is carried on each dispatch rather than fixed when an
`EngineGraphSink` is constructed:

```rust
pub enum TraceSubphase {
    Warmup,
    Profiling,
}

pub struct GraphDispatchContext {
    pub phase: Phase,
    pub trace_subphase: TraceSubphase,
    pub trace_instance: TraceInstanceId,
}
```

`GraphSink` accepts `&GraphDispatchContext` for every request, and the observer
copies it into the resulting `Record` and terminal supplement. The phase
coordinator may add enclosing run identity, but it must preserve this authored
dispatch phase instead of overwriting every record with the outer profiling
phase. Warmup and profiling may use distinct sink views over one transport, but
they must implement this same context contract. Fold-time assertions reject a
warmup record in a profiling metric store or profiling record in a warmup
store. This is the implementable seam for trace-local subphases in the existing
fixed-phase sink path.

Warmup is a trace-local excluded subphase, not a phase-wide corpus pass. Running
all warmups before all profiling traces is forbidden under `recorded-agent-default`.
Environment setup failure or warmup failure fails that task before its measured
interval exists. The standard runner may continue later manifest tasks, but the
final run is incomplete and non-comparable.

The reference task summary excludes model/environment setup and replay warmup; the
Rust interval follows the same boundary. Tool-command duration is measured from
the same injected real `Clock`, ending before timeout cleanup begins. LLM
durations come from completed request records rather than a second wall-clock
measurement, preserving transport measurement authority.

Cleanup is RAII-backed and idempotent across success, trace failure,
cancellation, phase grace escalation, timeout, placement shutdown, and a
partially opened sandbox. A close failure is recorded as infrastructure detail.
It becomes the trace error only when no earlier trace error exists; it never
masks the primary cause. Cancellation waits for bounded cleanup before the
placement reports terminal. Driver provisioning remains cancellable until open
succeeds. After that boundary, cancellation stops graph work but does not abort
or detach the driver-close future. Each resource-owning layer owns its deadline:
Docker container removal has a 20-second injected-`Clock` fence that preserves
the Docker command's 10-second execution budget plus its 10-second reap budget,
and an armed removal guard launches fallback removal if the async operation is
dropped, fails, or exceeds that fence. Only after dispatcher/sandbox cleanup
returns does the recorded driver apply a 10-second injected-`Clock` bound to
lifecycle-lease close; lease RAII fences an expired close exactly once. The
placement itself has no generic close deadline that could preempt either resource
owner. The original cancellation or trace failure remains primary, with cleanup
errors retained as infrastructure detail.

Workspace paths are rooted in a run/cell-owned tool directory and use a
sanitized execution-instance slug. They never derive a filesystem path directly
from an untrusted trace id. The cleanup policy is explicit: successful
workspaces may be removed, while a configured diagnostic-retention mode may
keep failed workspaces and reports their paths in structured diagnostics.

### Environment recipes and workspace staging

The stock recorded-agent resolver emits adapter-specific recipes:

The repository shall own a neutral
`containers/recorded-agent-pinchbench/` build context derived from the pinned
reference task image definition and required assets. Its complete
content-and-mode BLAKE3 digest and built OCI digest live in the canonical
workload fixture. A documented build target produces
`aiperf-recorded-agent-pinchbench:v1`; preflight inspects the resolved image and
requires the expected OCI digest. The tag is operator-friendly metadata, not
the parity authority. Any edit to the Dockerfile, base-image digest, packages,
entrypoint, or task runtime assets requires a workload-version and fixture
digest change.

- **PinchBench:** load the task-pack manifest named by recording metadata beneath
  `replay_root`, select the manifest task by `task_id`, parse its task file, and
  stage every `workspace_files` entry. Literal `{path, content}` entries are
  written as UTF-8; `{source, dest}` entries copy a file or directory from the
  task pack's `assets` root. During input preparation, every source and
  destination is canonicalized beneath its declared root, directories expand in
  stable relative-path order, symlinks are rejected, and fixture bytes plus
  executable mode are interned in the shared segment store. The serialized
  `WorkspaceSpec` therefore contains only safe relative destinations and
  content handles, not controller-local source paths. The worker provisioner
  materializes those exact bytes. The resulting host workspace is mounted at
  `/workspace`; working directory is `/workspace`; interpreter is `bash -lc`;
  network is `none`; default command timeout is 30 seconds; and image is the
  resolved `dataset.graph.pinch_image` (the default scenario pins
  `aiperf-recorded-agent-pinchbench:v1`).
- **SWE-Bench:** resolve the task-specific image in this order:
  `metadata.docker_image`, `metadata.instance.image_name`,
  `metadata.instance.docker_image`, then the deterministic image name derived
  from `metadata.instance.instance_id`. Only a low-level non-scenario run may
  fall back to `dataset.graph.tool_image`. Do not mount a blank workspace over
  the image; use `/testbed` as working directory, use `bash -c`, disable
  network, and apply a 60-second default command timeout. The guarded command
  policy returns code `127` and
  the standard explanatory observation for package-manager commands (`pip`,
  `python -m pip`, Conda/Mamba, Apt, Yum/DNF, or Apk) without invoking the
  sandbox.

The guarded policy parses top-level `&&`, `||`, and `;` segments while honoring
shell quoting and escaping, strips leading environment assignments plus
`sudo`/`env`, and blocks the whole authored command if any segment starts a
recognized installer. Parse failure uses a conservative token split and remains
deterministic. The returned synthetic result is timed and counted as an
attempted tool command, exactly like a fast code-127 sandbox result.

Unsupported task-manifest syntax, missing assets, missing `/testbed`, a missing
image, path escapes, duplicate mounts, or an environment recipe inconsistent
with its adapter fail during preflight/provisioning. No implementation may
silently substitute an empty workspace: recorded requests do not depend on live
tool observations, so such a mistake can otherwise produce plausible request
parity with meaningless near-zero tool timings.

Fixture handles remain part of the existing segment-store contract. A local
worker receives the prepared graph and its immutable segments. For a cross-host
cellular run, the controller instead ships the validated `agent_recording`
`replay_root` tree with safe relative paths, including the selected recording,
task-pack manifest/task files, and nested assets. The cell reconstructs that tree
without following symlinks, rewrites both `dataset.path` and
`dataset.graph.replay_root`, and then performs the normal input preparation so
its graph programs and content-addressed workspace handles are cell-local.
Materialization verifies each content digest before opening the sandbox.

### Local sandbox

`LocalSessionSandbox` starts one persistent shell session per provisioned trace
workspace at the recipe's working directory.
Every recorded command executes through a fresh invocation of the recipe's
interpreter (`bash -lc` for PinchBench and `bash -c` for SWE-Bench), preserving
the authored shell semantics without leaking command-local options or traps
into the session.
Commands share the workspace filesystem and environment changes represented by
files, as recorded agent tasks expect.

The framing protocol shall:

- generate a cryptographically or RNG-seam unique sentinel for every command;
- combine stdout and stderr in arrival order;
- append an unambiguous terminal frame carrying the exit status;
- treat sentinel-like user output as ordinary bytes;
- bound captured output and report truncation explicitly; and
- detect EOF, malformed frames, and a dead session as infrastructure errors.

On timeout, the backend first records the command end time, then terminates the
command process group and descendants, waits a bounded interval, escalates if
needed, and recycles the persistent session. A following command never consumes
stale output from the timed-out command. `open()` is idempotent and does not leak
an earlier session; `close()` is idempotent and reaps children.

Process launching uses an injected process-spawner seam in unit tests. Product
code uses Tokio process primitives and never invokes a blocking wait on the
worker reactor.

### Docker sandbox

`DockerSessionSandbox` creates one detached container per trace instance from
its resolved recipe with:

- the configured image;
- network disabled;
- the recipe's mounts, which include the PinchBench workspace at `/workspace`
  and deliberately include no blank workspace mount for SWE-Bench;
- the recipe's `/workspace` or `/testbed` working directory and interpreter;
- a deterministic sanitized container-name prefix plus unique suffix; and
- a run-identity label independent of the in-memory sandbox handle; and
- no ambient endpoint credentials forwarded by default.

Commands use a persistent `docker exec -i` session and the same framed command
protocol as the local backend, but framing always invokes the resolved recipe
interpreter: `bash -lc` for the PinchBench recipe and `bash -c` for SWE-Bench.
It never hard-codes a login shell. A command timeout recycles the entire
container so descendants and stale exec output cannot leak into the next
command.

Container create/start/exec/inspect/remove operations sit behind an injectable
`ContainerRuntime` trait. The stock implementation invokes the Docker CLI with
an argv vector, never a constructed shell string. Startup validates image and
mount errors before measurement begins. Close force-removes the container with
a bounded timeout and is idempotent. The run reports cleanup failures with the
container id/name for operator recovery.

RAII is the normal cleanup path, but it is not the only cleanup authority. On
signal escalation or process restart, the controller queries and force-removes
containers bearing the exact run label. Labels are resolved from a persisted
opaque run id rather than an unvalidated trace id. Cleanup never selects an
empty or broad label.

The default scenario requires Docker because local execution cannot reproduce
the task environments. Preflight validates every distinct resolved image and
recipe. Local execution remains
available outside the scenario for host-device benchmarks and deterministic E2E
tests.

### Trace-local warmup

`GraphInputBundle` shall carry ordered `GraphTraceProgram`s rather than parallel
profiling and warmup vectors:

```rust
pub struct GraphInputBundle {
    pub programs: Vec<GraphTraceProgram>,
    pub segments: Arc<dyn SegmentStore>,
    pub metadata: GraphInputMetadata,
}
```

Each recording contributes exactly one warmup plan when enabled. Its stable id
is `warmup-<profiling-trace-id>` and it contains one LLM node with:

- one user message: `Reply with exactly: ok`;
- generation cap `8`, marked as authored so generic warmup cap rewriting cannot
  replace it;
- the resolved live model profile's configured bash-tool schema and sampling,
  not the first recorded call's arbitrary tool array;
- the same endpoint and raw streaming transport as profiling;
- fail-fast retry policy with one attempt; and
- no tool-execution node.

Placement dispatches the warmup with `Phase::Warmup` identity after the
environment opens and immediately before the associated profiling plan. Warmup
records and diagnostics are retained separately and never contribute to
profiling budgets, summaries, normalization, or artifacts. A program cannot be
wrapped or independently sampled into a warmup/profiling mismatch.

Trace-local warmup and the existing t-star/cache-pressure warmup are different
concepts. Preparation represents the source explicitly, rejects an unsupported
combination, and never applies a generic snapshot rewrite over tool nodes.
Recorded-agent warmup is cache-isolation exempt: it uses the unmodified user
message, while the run namespace applies only to profiling. The warmup primes
the endpoint and client path, not the measured trajectory's isolated prefix.

### Run-scoped cache isolation

The reference `isolate_replay_messages` behavior prefixes the first wire
message once per replay invocation with a random namespace. Native parity shall mint one marker
per benchmark run from the existing deterministic RNG namespace plus the run's
unique benchmark identity, then reuse it across every profiling trace instance
in that run. A different benchmark run receives a different marker. The
corpus-authored warmup plan carries an explicit cache-isolation exemption; this
phase-aware choice is part of plan metadata or prepared phase policy rather than
an id-string convention.

The transform targets the first wire message, not only the first message whose
role is `system`. This is the upstream contract and supports recordings without
a system role. It shall preserve the message object and all non-content keys.
String content receives a string prefix; structured/multimodal content receives
a leading text part; absent or null content becomes a valid prefixed text
content according to the endpoint dialect. The marker is applied once during
materialization and is never persisted back into the immutable segment store.

The marker uses the reference token-shape-stable template: 32 independently
generated decimal digits separated by single spaces, followed by
`Performance replay cache namespace. Ignore the digits above.` and two trailing
newlines. The RNG seam supplies the digits. Cross-run profiling prefixes differ
and all profiling requests within one run share the exact namespace. Warmup
requests have no marker. The A/B harness normalizes only the 32 digit values
while asserting template, scope, placement, and unmodified warmup.

The controller allocates a distinct opaque run identity for every new invocation
and persists it with the cache namespace before any warmup. The exact identity
scopes container labels and cleanup; concurrent runs of the same manifest never
share it. `--graph-resume` restores that persisted identity and namespace and
requires the same artifact root, manifest digest, recording digests, and resolved
request/environment profiles. A mismatch fails closed rather than creating a
mixed-namespace run. The namespace is sensitive benchmark state: normal reports
include its digest and mode, while the raw value is stored only in the protected
resume checkpoint.

### Resume, completeness, and provenance

After each task reaches terminal cleanup, the controller atomically writes a
bounded checkpoint containing task identity, manifest ordinal, source digest,
resolved environment/profile digests, successful profiling-call count, terminal
classification, and artifact-record offsets. A task is skipped on resume only
when its checkpoint is successful and its model-call count equals the source
recording. Failed, partial, differently configured, or unverifiable tasks rerun.
Resume is supported first for the scenario's single-worker/single-cell mode;
distributed resume is rejected until the cellular protocol has an equivalent
controller-owned checkpoint contract.

`replay-provenance.json` records the manifest/source digests, source format and
revision, ordered task identities, resolved request and environment profiles,
measurement boundary, cache-isolation mode/digest, endpoint placement, user
hardware description, debug overrides, timing-policy version, and whether the
result is comparable. Credentials, raw cache namespace, full prompts, and full
commands are excluded.

A run may continue after an individual task failure to preserve later
measurements. It still exits with an incomplete-result status, lists the task in
`failures.tsv`, and sets `submission_valid = false`; a partial aggregate is
clearly labeled and cannot masquerade as the manifest-defined workload.

Tool execution always runs on the benchmark host. Reports may describe
normalized end-to-end time as a whole-device local result only when endpoint
placement is explicitly `co_located`; `remote` and `unknown` results keep model
and tool timing separate and carry that qualification into comparisons.

### Result propagation and artifact folding

`TracePlacement::execute_trace` shall return a typed terminal value rather than
discarding the executor's trace result. Its graph supplement contains only
bounded trace-level data plus command measurements required by configured
artifacts. Combined command stdout/stderr remains in the worker-local graph
observation channel and is not copied into the coordinator event. A command
measurement carries duration, return code, timeout flag, and resolved backend;
it does not contain request bodies, command text, output, or duplicate captured
records.

`GraphExecutionEvent::TraceComplete` shall carry:

- unique trace-instance id and template id;
- total graph-node and LLM-node counts;
- terminal classification;
- trace start/end or finite wall duration;
- ordered bounded LLM measurements: request duration, TTFT, generation and
  stream-total duration, observed ISL/OSL, target OSL, SSE-event count,
  meaningful-output evidence, completion-sentinel evidence, and recorded-prompt
  ISL when available;
- ordered bounded tool-command measurements; and
- cleanup diagnostics.

The phase coordinator folds events in manifest/completion order into a
`GraphPhaseSupplement`. Warmup and profiling supplements remain separate. A
multi-worker run merges worker supplements at the coordinator. A cellular run
serializes one associative `GraphCellSupplement` with the existing cell
partition, and the controller merges cells in stable cell-id order. Supplement
schema versions are explicit and mixed versions fail closed.

Every terminal classification is retained for progress, failure, cancellation,
and cleanup diagnostics. Timing artifact folds include only traces whose graph
executor completed successfully, matching the Python branch's
`_record_trace_timing(await executor.run(...))` boundary. Failed, refused, and
cancelled traces—and their partial LLM/tool measurements—do not contribute to
either artifact. Their terminal diagnostics remain available through the normal
error/report path. A successful trace with attempted tool commands increments
the tool artifact's `trace_count` once.

No worker, trace, or cell writes the final shared JSON paths. This avoids races,
works with exact-fold/sketch record modes, and gives cellular execution one
authoritative result. Artifact emission occurs after the profiling supplement
is complete and before the cell/controller artifact barrier reports success.

### Injectable replay metrics and timing validity

Normalization and validity are policy, not graph-executor behavior:

```rust
pub trait ReplayMetricsPolicy: Send + Sync {
    fn analyze_call(
        &self,
        measurement: &ReplayCallMeasurement,
    ) -> Result<ReplayCallMetrics, ReplayMetricsError>;

    fn fold_trace(
        &self,
        calls: &[ReplayCallMetrics],
        trace: &TraceTerminalSupplement,
    ) -> Result<ReplayTraceMetrics, ReplayMetricsError>;
}
```

The stock policy derives each call's normalized generation from its source
recording target:

```text
observed_decode_tokens = max(observed_osl - 1, 1)
target_decode_tokens = max(target_osl - 1, 0)
generation_ms_per_token = raw_generation_ms / observed_decode_tokens
normalized_generation_ms = generation_ms_per_token * target_decode_tokens
normalized_stream_total_ms = ttft_ms + normalized_generation_ms
normalized_inference_ms = max(0, raw_inference_ms - raw_generation_ms
                                 + normalized_generation_ms)
normalized_end_to_end_ms = max(0, raw_end_to_end_ms - raw_inference_ms
                                  + normalized_inference_ms)
```

Actual observed OSL is always reported separately from target/normalized OSL.
ISL delta is the first prompt ISL plus positive prompt-token growth between
successive source calls; it is an ideal-prefix workload-shape metric, not a
claim about observed endpoint cache hits.

TTFT is the first meaningful streamed output: non-empty content, reasoning, or
a substantive tool-call delta. Role-only, empty scaffolding, finish, usage, and
completion-sentinel events do not start TTFT. The raw OpenAI-compatible timing
path fails the call on a non-object data payload, a stream error, missing
meaningful output, missing required positive completion usage after observed
output, or termination without `data: [DONE]`.

Post-hoc validation rejects or annotates:

- non-finite or negative durations;
- non-positive TTFT/stream total, or zero generation time for multi-token
  output;
- TTFT, generation, or offsets outside the enclosing stream/model duration;
- TTFT plus generation inconsistent with stream total beyond 1 ms;
- implied generation above 10,000 tokens/second;
- at least 16 completion tokens delivered at more than 32 tokens per SSE event;
  and
- server-reported ISL more than `max(128, 2% of recorded ISL)` below the source
  prompt size.

An anomalous call retains safe raw request/stream totals, raw trace wall time,
tool time, token counts, and structured reasons, but its decomposed and
normalized timing becomes absent. One anomalous call makes aggregate TTFT,
generation, and normalized totals absent so invalid timing cannot hide in a
sum. Observed OSL below 50% of target emits a non-fatal extrapolation warning;
exactly 50% does not.

The scenario requires the observations needed by this policy and rejects
sketch-only execution. General recorded-agent replay may use other registered
policies or ordinary AIPerf metrics, but only the stock policy produces a
comparable default result. `metrics.json` and optional CSV contain per-call,
per-trace, and aggregate normalized metrics plus actual token counts and
validity diagnostics.

`profile_export_graph_tool_time.json` has this exact schema:

```json
{
  "command_count": 0,
  "trace_count": 0,
  "backend": "local",
  "total_s": 0.0,
  "mean_s": 0.0,
  "median_s": 0.0,
  "max_s": 0.0,
  "durations_s": []
}
```

`backend` is `local` or `docker:<image>` when every included command used one
resolved backend, and `mixed` when a corpus used more than one local/image
identity. This keeps the pinned artifact schema while accurately representing
per-trace `TraceEnvironmentSpec` selection. `durations_s` preserves execution order in
single-placement runs and stable cell/worker merge order otherwise; median is
computed over a sorted copy. The artifact is omitted when no successful
profiling trace attempted a tool command.

`profile_export_graph_trace_summary.json` has:

```json
{
  "trace_count": 1,
  "aggregate": {
    "total_s": 1.0,
    "model_s": 0.6,
    "tool_s": 0.3,
    "model_time_fraction": 0.6,
    "tool_time_fraction": 0.3,
    "model_calls": 2,
    "tool_calls": 1
  },
  "traces": [
    {
      "trace_id": "example::0",
      "total_s": 1.0,
      "model_s": 0.6,
      "tool_s": 0.3,
      "model_time_fraction": 0.6,
      "tool_time_fraction": 0.3,
      "model_calls": 2,
      "tool_calls": 1
    }
  ]
}
```

Aggregate fractions are aggregate duration divided by aggregate total, not the
mean of per-trace fractions. A zero total produces `0.0`. Every serialized
number is finite. `tool_calls` means attempted shell commands, not tool nodes.
Trace order is deterministic and documented; completion order is retained for
single placement, while distributed merges use `(cell_id, worker_id,
completion_ordinal)`.

The trace summary is emitted for successful profiling traces even when tool
execution is disabled; the tool-time artifact requires attempted commands on a
successful profiling trace. Neither artifact contains warmup or partial
failed/cancelled trace data.

### Error taxonomy and observability

Library layers shall use explicit errors with `Display` implementations:

- `RecordedAgentInputError`: discovery, decode, schema, duplicate id, empty corpus;
- `RecordedAgentLoweringError`: missing replay fields, invalid event sequence,
  topology, segment interning;
- `GraphRequestMaterializationError`: missing/invalid request handles, reserved
  field collision, dialect projection;
- `TraceDriverError`: unknown driver, driver lifecycle, invalid turn decision,
  dispatch-context, or terminal-supplement failure;
- `AgentLoopError`: response selection/reuse, tool-call decoding, observation
  formatting, or step/cost/time budget termination;
- `TraceEnvironmentError`: task-pack resolution, workspace staging, recipe
  validation, command-policy configuration;
- `ReplayMetricsError`: missing observations, invalid normalization target,
  timing anomaly, fold/schema/export failure;
- `ToolDispatchError`: dispatcher lifecycle or dispatch infrastructure;
- `ToolSandboxError`: local/container session startup, framing, process/runtime,
  cleanup; and
- `GraphSupplementError`: event/fold/schema/artifact failures.

Application and CLI boundaries add context with `anyhow`. Error messages identify
the operation, trace instance, node, command ordinal, and backend without
including endpoint keys, recorded credentials, or full commands by default.

Routine lifecycle events are `debug!`; per-command detail is `trace!`; run-level
tool summary and retained cleanup instructions may be `info!`/`warn!` as
appropriate. Logging is structured (`trace_id`, `node_id`, `command_index`,
`backend`, `duration_s`, `is_timed_out`, `return_code`, `error = %error`) and
never formats full command output on the hot path.

Endpoint and host provenance uses a registered `RunMetadataCollector` seam.
Collectors run concurrently with short bounds before the first task and after
the last cleanup, outside measurement. The stock collector may probe portable
model metadata and common serving-stack version/metrics endpoints, but every
probe is best-effort and cannot make benchmark execution depend on a particular
server. Start/end counter deltas are labeled server-wide and may include warmup
or unrelated clients. User-provided hardware is a required free-form string for
the default scenario; `unknown` is valid. Credentials, URL userinfo/query data,
authorization headers, and secret-like response fields are redacted before
serialization.

Raw request/response exchange dumping is diagnostic-only, disabled by default,
and marked potentially sensitive. Enabling it does not alter wire behavior but
is recorded in provenance. Request debug output must never be required to
calculate the standard metrics.

### Placement and cellular requirements

One placement target owns an entire trace program and its sandbox. Thread-per-core
workers build their own tool dispatcher from the injected factory. Tool state
never crosses worker threads, and command execution never holds a lock across
`.await`.

Cellular partitioning remains trace-level. Every cell host must have access to
every distinct Docker image its partition may resolve and to the Docker runtime,
or to local execution prerequisites when the scenario is not active. Preflight
capability validation occurs on every cell before the phase barrier. A partial
preflight fails the run before warmup or profiling begins.

Cross-host dataset shipping treats `agent_recording` as a graph format. When a
replay root is authored, the controller allowlists and streams the rooted file
tree with its relative hierarchy intact; absolute paths, parent traversal,
duplicate destinations, symlinks, and non-file entries fail closed. Cells land
the tree in a unique process-owned directory and rewrite the recording path and
replay root together before recipe preflight.

Cell workspaces live under the cell's exclusive artifact root. Final graph
supplements travel through the cellular protocol rather than artifact-file
concatenation. Controller merge checks that all expected cells supplied a
compatible supplement. Missing cells, duplicate trace-instance ids, unknown
backend identities, or non-finite durations fail closed. Valid heterogeneous
backend/image identities merge normally and produce the scalar artifact label
`mixed`.

Cancellation and force escalation broadcast through existing placement control.
Each worker stops admitting traces, terminates active command trees/containers,
closes sandboxes, and emits a terminal cancellation supplement before its
bounded shutdown completes.

### A/B parity harness

The integration harness launches the pinned experimental reference runner and
the native
`aiperf` binary as separate subprocesses against the same deterministic mock or
live inference endpoint. Each path uses its own capture proxy so the harness
records ordered request bodies without conflating clients.

The comparison includes each recording's warmup and profiling calls. It
normalizes only deliberate client differences:

- the randomized cache-isolation marker text, while checking placement and
  scope;
- the reference runner's legacy `max_tokens` versus an endpoint's equivalent
  modern
  `max_completion_tokens` spelling where dialect selection requires it;
- the reference runner's `stream_options.include_usage`; and
- LiteLLM's client-only `drop_params`.

It does not normalize messages, roles, content, tools, tool-call ids, model,
temperature, top-p, parallel-tool setting, generation cap value, call count, or
call order. The harness must exercise the `recorded-agent-default` scenario path
so scenario resolution itself is under test. It uses fresh workspaces because
task containers may leave root-owned files.

The harness separately compares task lifecycle order, resolved environment
recipes, fixture-visible command effects, terminal classifications, trace
summaries, per-call targets/observations, timing-validity decisions, and
normalized metrics within documented transport overhead tolerances. It does not
require tool-duration equality across different hosts; it requires the same
command count/order, command-policy disposition, and valid local measurements.
The test corpus includes a Pinch fixture read/write and a SWE relative command
that succeeds only when the image-native `/testbed` working directory is
preserved.

### Verification matrix

Implementation is complete only when these behaviors are covered:

| Layer | Required proof |
|---|---|
| Manifest/DTO | Canonical eight-task identity/order and fixture-derived digests; fully typed defaults and all nine aggregate fields; root-contained path resolution; source-shape recomputation; JSON/gzip recordings; sorted shallow low-level directory; nested trace-id/image precedence; malformed explicit file; finite timestamps; duplicate/empty errors. |
| Request profile/lowering | Exact messages/tools; prefix interning; positive recorded cap or `32768` fallback; standard SWE and Pinch wire profiles; protected extra-body fields; relative timing; trailing tools; control-flow versus genuine tool failures. |
| Graph model | `GraphTraceProgram` and driver-spec serde/cellular compatibility; tagged heterogeneous nodes; topology/read/write channels; LLM versus total counts; flat-path refusal; every consumer audited. |
| Trace/agent drivers | Static and recorded driver selection; factory-owned capability refusal before environment spend; one driver/store/sink per concurrent trace and one coordinator per invocation; fresh/load/resume validation; live responses measured but recorded bytes selected; fake live response -> decoded calls -> tool results -> formatted observations -> next request -> completion; inline-versus-handle response selection, original-turn reuse, and copied-context accounting; retry/branch reuse; shared/isolated fake invocation leases and deterministic join order; budgets/cancellation; no driver-owned transport, timer, process, or metrics sink. |
| Agent trajectory | Incremental events survive cancellation; run/trajectory/invocation identities remain distinct; sequential turns; tool-call/result correlation; deterministic non-LLM steps; copied context excluded from double counting; continuation and delegated-trajectory references resolve only by document id or explicit artifact reference. |
| Environment resolution | Pinch task-manifest parsing, neutral build-context/OCI digest pin, image inspection, and fixture staging; containment failures; full SWE image precedence; `/workspace` versus `/testbed`; recipe-specific `bash -lc` versus `bash -c`, mounts/network/timeouts; capability match/refusal for isolation and network policy; guarded install-command synthetic result; fake resolvers/provisioners/policies. |
| Warmup/lifecycle | Environment open before warmup; one warmup immediately before its task; exact prompt/cap/live tool profile/sampling; one attempt; per-dispatch warmup context retained in records; fold refuses phase leakage; no executable tool node; cleanup after profiling; no phase-wide warmup reorder. |
| Cache isolation | first-message string, structured, null, and no-system cases; one profiling marker within a run; a different marker across runs; unmodified warmup; no segment-store mutation. |
| Dispatcher | fake injection; lifecycle order; sequential command results; nonzero and timeout outcomes; continuation after successful timeout recycle; recovery failure abort; close-error precedence. |
| Local sandbox | persistent workspace; fresh shell semantics; combined output; sentinel collision; output bound; timeout descendant kill and clean recycle; idempotent open/close. |
| Docker sandbox | Recipe-specific argv, mounts/workdir/interpreter, network none, unique names and exact run labels, startup error, timeout recycle, RAII plus label-based force cleanup, fake runtime injection. |
| Phase/placement | Manifest-order one-active-trace scenario; setup/warmup/cleanup excluded; cancellation cleanup; LLM-only request accounting; multi-worker fold determinism outside the scenario; no shared worker writes. |
| Metrics | Meaningful-output TTFT; required usage and `[DONE]`; exact normalization formulas including non-negative clamps; ISL delta; anomaly invalidation; 50% OSL warning; raw-value retention; per-call/trace/aggregate JSON and CSV. |
| Resume/provenance | Persisted namespace before warmup; atomic successful-task checkpoint; call-count/digest/config verification; partial rerun; incomplete status; secret redaction; debug non-comparability. |
| Cellular | Per-cell recipe/image preflight; trace-program ownership; supplement serialization/fold; missing/duplicate/unknown-backend refusal; valid mixed-backend merge; controller-only artifacts; distributed resume refusal. |
| Artifacts | Exact tool/summary schemas; normalized metrics; successful-terminal-only inclusion; finite zero guards; warmup/partial-terminal exclusion; deterministic ordering; single/`mixed` backend labels; artifact allowlist and sweep retention. |
| Product E2E | Native binary against deterministic `aiperf-mock-server`, raw records and request bodies inspected; Pinch fixture and SWE `/testbed` commands produce expected effects; local and Docker tool execution. |
| A/B | Pinned reference and AIPerf subprocesses; per-task setup/warmup/profile order; request bodies; command order/effects; normalized metrics and validity; scenario locks; only documented normalization differences. |

At minimum, repository verification includes:

```bash
source .venv/bin/activate
cd rust
cargo fmt --check
cargo clippy -p aiperf-runtime --all-targets --features engine
cargo test -p aiperf-runtime
cargo test -p aiperf-runtime --features engine
cargo test -p aiperf-cli
cargo test -p aiperf-e2e-tests --test test_recorded_agent_replay
```

Docker tests are opt-in when the daemon or task image is unavailable, but their
fake-runtime unit coverage is unconditional. The local product E2E and A/B mock
traffic test are unconditional in the supported development environment.

### Implementation sequence

The port shall be delivered in dependency order, with each slice retaining a
green workspace:

1. Add strict recording and manifest DTOs, the canonical eight-task fixture and
   digest index, root-contained discovery, complete source-shape validation,
   and pure fixtures; register `RecordedAgentInputAdapter` without tool
   execution.
2. Add the injectable request-profile resolver and exact standard request-body
   lowering, including fallback caps and protected extra-body fields.
3. Introduce tagged `ExecutableGraphNode`, serializable `GraphTraceProgram`, registered
   `TraceProgramDriver`, and per-dispatch phase context; audit every node/plan
   consumer while preserving generic LLM-only compatibility.
4. Implement the static and recorded-replay drivers, trace-local warmup, and
   run-scoped first-message cache isolation; prove environment/warmup/profiling
   ordering, record-level phase accounting, and recorded response-byte reuse.
5. Add a fake live driver/coordinator, tool-call decoder, and observation
   formatter test that executes a dynamic full turn cycle, response reuse,
   capability refusal, continuation validation, trajectory correlation, and a
   delegated invocation without exposing a public live scenario; then add
   environment resolver, workspace provisioner, command-policy,
   dispatcher, sandbox, process, and container traits with deterministic fakes.
6. Implement Pinch fixture staging plus the neutral digest-pinned image build
   and inspection path, full SWE image precedence and image-native `/testbed`,
   guarded command policy, recipe-specific interpreters, local sandbox, and
   Docker sandbox behind those seams.
7. Return typed trace supplements through placement and implement the injected
   normalization/timing-validity policy plus JSON/CSV exporters.
8. Add controller-owned provenance, backend metadata, run-label cleanup,
   failure ledger, and atomic single-placement resume checkpoints.
9. Extend cellular protocol/folding and artifact allowlists for trace programs
   and supplements; keep distributed resume explicitly rejected.
10. Add `recorded-agent-default` canonical workload/image/response-policy locks
    across CLI, Config v2, strict protocol, and scenario outcome reporting.
11. Add product E2E and strengthen the A/B harness to cover lifecycle,
    environment effects, wire bodies, normalized metrics, and artifacts.

Each slice updates this record and the architecture index with the code it
lands. Public DTOs and traits receive `///` documentation; new modules receive
`//!` documentation; every Rust source carries the NVIDIA SPDX header.

## Source anchors

The following experimental paths explain behavioral provenance only. Their
internal names must not appear in public Rust module, type, trait, scenario,
configuration, artifact, CLI, documentation-title, or test-target identifiers.

### Native Rust prerequisites

- `rust/runtime/src/graph/{model.rs,input.rs,execution.rs,executor.rs,placement.rs,workload.rs}`.
- `rust/runtime/src/engine/{application.rs,graph_input.rs,graph_execution.rs,graph_phase_runtime.rs,protocol.rs,artifact_shipping.rs}`.
- `rust/runtime/src/agentx/{cache_bust.rs,scenario.rs}`.
- `rust/runtime/src/config/{model/dataset.rs,resolve.rs,validate.rs}`.
- `rust/cli/src/{flags.rs,load.rs,yaml.rs}`.
- `rust/e2e-tests/tests/test_graph_cellular.rs`.

### Experimental Python source at the pinned branch revision

- Sibling-checkout recording reader/lowerer and its unit fixtures.
- `../dynamo-graph-ir/src/aiperf/graph/{tool_dispatch,sandbox}/` and
  `../dynamo-graph-ir/src/aiperf/graph/dispatch/tool.py`.
- Sibling-checkout scenario policy and
  `src/aiperf/timing/strategies/{agent_graph_replay.py,graph_warmup.py}`.
- Sibling-checkout unit/integration tests for recording replay, tool dispatch,
  sandboxes, timing artifacts, and scenario resolution.
- Sibling-checkout parity handoff, agent-graph tool-execution, and
  sandbox-resident-executor design records.

### Experimental reference authority at the pinned revision

- Experimental reference checkout
  `src/minisweagent/recording/{recorder.py,replay.py,cache_isolation.py,validation.py,warmup.py,wrappers.py}`.
- Experimental reference checkout
  `src/minisweagent/config/benchmarks/{swebench.yaml,pinchbench.yaml}` and
  `src/minisweagent/run/{record_swebench.py,record_pinchbench.py}`.
- Experimental reference checkout `docker/pinchbench/Dockerfile` and its build
  assets for the neutral parity image context and OCI-digest lock.

### Supplemental Harbor architecture research

The Harbor checkout at `a27e9c2ae10a31c40b2dcef33ef5486bce36e185` informed
future-facing seams only:

- `src/harbor/agents/base.py` and `agents/factory.py` for agent construction,
  setup/run, capability flags, and fail-fast load/resume validation;
- `src/harbor/environments/{base.py,capabilities.py,factory.py}` for
  provider-neutral execution, explicit capability negotiation, network policy,
  and environment identity;
- `src/harbor/trial/{trial.py,single_step.py,multi_step.py,hooks.py}` for
  lifecycle ownership, incremental recovery, step continuation, verification
  separation, and typed hooks;
- `src/harbor/agents/computer_1/{computer_1.py,providers/base.py,runtime.py}` for
  normalized turn decisions, retry feedback, action execution, observations,
  terminal decisions, and provider-dialect separation; and
- `src/harbor/models/trajectories/` for sequential turns, tool-call/result
  correlation, copied context, continuation, and delegated-agent trajectory
  identity.

## Revision notes: future evaluation-platform compatibility

This revision preserves the recorded-agent replay port's delivery sequence and
its concrete runtime contracts while reserving compatible seams for a later
agentic SWE evaluation platform. It does not add a public benchmark registry,
task grader, or live-agent scenario to this port.

1. **Keep the execution node union narrow.** For this port,
   `ExecutableGraphNode::{Llm, Tool}` is the lowered runtime IR. It is not the
   canonical source-semantic model; a future `SemanticGraph` lowers into it
   through an explicit fidelity and capability report. Spawn/join/gate behavior stays
   expressed through graph topology, channels, and the trace driver;
   checkpoints, evaluators, and experiment control remain lifecycle or
   controller services rather than new executor node variants. This preserves
   the exhaustive-consumer audit and the LLM-only flat-graph fast path.

2. **Keep terminal checkpoints authoritative.** The initial resume contract is
   the controller-owned atomic checkpoint written after terminal trace cleanup,
   with distributed resume explicitly rejected. A later platform may retain an
   append-only node/tool event journal for diagnostics or recovery research, but
   it must not claim resumability beyond this terminal checkpoint without a
   controller-owned distributed protocol.

3. **Preserve both sandbox-sharing modes.** Sequential recorded replay may use
   the planned shared sandbox lease. Isolation is required for concurrently
   mutating implementation branches: each branch receives an overlay or cloned
   workspace and returns an immutable candidate patch/artifact; an explicit
   selector or merge step alone may update a canonical workspace snapshot.

4. **Treat verification as an external, post-run concern initially.** The
   replay port's diagnostic/grading hook remains outside the profiling interval
   and is not required for parity. A later `TaskSpec`/dataset/trial layer may
   pin sandbox and verifier digests, run a verifier in a separately restored
   workspace or sandbox, and record task-health verdicts. It must not delay the
   current manifest/lowering/driver/sandbox implementation sequence.

5. **Make evidence extensible without adding hot-path contention.** The typed
   terminal supplement and optional trajectory artifact exporter are the first
   provenance boundary. A later versioned event schema may add run, sample,
   attempt, span, tool, sandbox, evaluator, and security evidence, but worker
   state remains local and cells return bounded serializable supplements for
   controller-owned artifact finalization and folding.

6. **Reserve graph comparison for an experiment layer.** A later experiment
   controller may run paired graph variants with task, model, seed, sandbox
   recipe, policy, and budget fixed. It reports quality, cost, latency, and
   failure-mode deltas independently; it does not alter the replay driver's
   deterministic response-selection contract.
