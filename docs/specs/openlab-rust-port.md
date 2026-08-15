<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Open LAB recording replay — native Rust port

## Purpose

This record specifies the native Rust port of the `ajc/open-lab` implementation
from the sibling `dynamo-graph-ir` checkout. The port makes an Open LAB
performance-replay recording a first-class Graph-IR input and preserves the
source branch's complete behavior: request-body replay, optional real tool
execution in local or Docker sandboxes, per-recording warmup, task-family
sampling, run-scoped cache isolation, scenario locks, timing artifacts, and an
Open LAB/AIPerf A/B parity harness.

The design keeps the runtime's existing composition rules. Graph input,
lowering, tool dispatch, sandbox construction, time, placement, result folding,
and artifact export each have an injectable boundary. Tool nodes do not enter
the inference request-credit or request-record planes, and no per-command path
adds cross-thread shared-state contention.

The parity baseline is:

- AIPerf Python branch `ajc/open-lab` at
  `244222b5999f48d89799f25ee946eedd81831117`.
- Open LAB `main` at
  `b8897f5de1664ad6de9cd669a96c3ba5d379e81e`.

Claims in this record were derived from executable source and tests at those
revisions. Prose in the source branch's handoff record is supporting context,
not authority where it disagrees with code.

The Rust design deliberately corrects five source-branch limitations found in
that audit: it applies cache isolation to the first wire message as upstream
Open LAB does (the Python branch searches only for a system role); starts trace
wall measurement after sandbox open (the Python implementation currently starts
its timer before `open_trace` despite the dispatch seam documenting setup as
excluded); folds supplements through workers and cells instead of letting phase
instances write a shared file; derives artifact backend identity from actual
per-trace selection and reports heterogeneous runs as `mixed` instead of
mislabeling them from the run-level fallback; and makes artifact/export failure
explicit rather than warning and silently losing the measurement. These are
parity and runtime integrity fixes, not optional scope reductions.

## Built

The Rust runtime already supplies the prerequisites the port composes over:

- `Application` freezes a registered `GraphInputAdapterResolver`; the built-in
  resolver selects one strict adapter and produces a `GraphInputBundle`.
- `GraphInputBundle` carries complete `GraphTracePlan`s, one immutable
  content-addressed segment store, and static input metadata.
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

The current graph node map contains only `LlmNode`; `GraphInputBundle` contains
only profiling plans; trace placement returns only `Result<(), TraceError>`;
and graph execution has no tool dispatcher, sandbox, graph supplement fold, or
Open LAB input adapter. The following section is the planned port.

## Future requirements

### Product behavior and non-goals

The native port shall implement all behavior at the pinned Python branch HEAD:

1. Read one Open LAB recording file or a shallow directory corpus.
2. Lower every usable model call into an exact Graph-IR LLM request.
3. Lower completed recorded commands into ordered tool nodes when real tool
   execution is enabled; otherwise represent their elapsed gap as edge timing.
4. Execute tools in a per-trace local or Docker sandbox.
5. Issue one authored warmup request per recording before profiling.
6. Apply task-family wire sampling and one run-scoped profiling cache-isolation
   prefix while leaving upstream-equivalent warmup unmodified.
7. Enforce the `openlab-default` scenario as resolved configuration locks.
8. Emit tool-time and trace-summary artifacts from merged profiling results.
9. Prove wire and lifecycle parity with unit, product E2E, and A/B tests.

This port replays predetermined requests and recorded commands. It does not run
the mini-swe-agent control loop, substitute live model replies into subsequent
prompts, grade SWE-bench or PinchBench tasks, or dynamically derive tool calls
from a live response. A sandbox-resident Rust executor binary is a separate
optimization; the parity implementation is host-resident and drives a
persistent local shell or `docker exec` session.

Real tool execution is a real-clock online capability. Validation shall reject
it with `SimClock`, `dry_run`, `dynosim_offline`, or another virtual transport.
Recorded-delay replay without tool execution remains compatible with the
existing real and simulated graph paths.

### User and protocol configuration

The native CLI, Config v2 model, resolved `BenchmarkRun`, and strict protocol-v2
request shall expose one consistent vocabulary:

| CLI | Config v2 | Resolved meaning | Default |
|---|---|---|---|
| `--graph-format openlab_recording` | `dataset.format: openlab_recording` | Select the native adapter. | Format discovery where already supported; explicit under the scenario. |
| `--graph-execute-tools` | `dataset.graph.execute_tools` | Lower and execute completed recorded commands. | `false` |
| `--graph-tool-image <image>` | `dataset.graph.tool_image` | Use one Docker sandbox per trace; absent selects local execution outside the scenario. | absent |
| `--graph-tool-command-timeout <seconds>` | `dataset.graph.command_timeout_seconds` | Per-command wall-clock ceiling when a node has no authored override. | `900.0` |
| `--graph-tool-container-stop-timeout <seconds>` | `dataset.graph.container_stop_timeout_seconds` | Bound Docker force-removal during recycle or close. | `5.0` |
| `--graph-tool-session-close-grace <seconds>` | `dataset.graph.session_close_grace_seconds` | Grace for a session shell to exit before its process group is killed. | `1.0` |
| `--graph-use-family-sampling` / `--no-graph-use-family-sampling` | `dataset.graph.use_family_sampling` | Overlay Open LAB family defaults. | `true` for this adapter |
| `--graph-emit-warmup` | `dataset.graph.emit_warmup` | Compile one corpus-authored warmup plan per recording. | `false` |

Existing graph input options continue to control delay suppression/capping,
record limits, dataset wrapping, recorded model selection, and recorded sampling
selection. The Open LAB adapter shall reject unknown option keys and invalid
combinations rather than silently ignore them. All fields must project through
the native CLI and YAML surfaces into the same typed runner input.

`--graph-tool-image` without tool execution is rejected as inert configuration.
Tool execution plus open-loop graph replay is rejected: replaying recorded
tool gaps while also measuring real tool time would double-pace the trajectory.
All three timeout fields are positive finite seconds and project to integer
nanoseconds before worker construction. An authored `ToolNode.timeout_ns` wins
over `command_timeout_seconds`; Open LAB lowering authors no override, so its
commands use the 900-second default. Container-stop and session-close bounds
apply only to cleanup/recycle and never inflate the recorded command duration.
The values are injected into the sandbox factory rather than read from global
environment variables in the execution hot path.

The strict protocol shall add artifact paths for the two graph supplements:

- `graph_tool_time_path`, default
  `profile_export_graph_tool_time.json` when real tools are active.
- `graph_trace_summary_path`, default
  `profile_export_graph_trace_summary.json` for Open LAB profiling.

Both paths participate in path validation, cellular allowlisting, same-host
merge rules, sweep collection, and artifact manifests. The controller is the
sole writer of final merged files.

### `openlab-default` scenario

The native scenario registry shall add `openlab-default`. Resolution auto-fills
an unset value and reports every explicit conflict through the existing
scenario-outcome contract. Its locks are:

- graph workload and `dataset.format == "openlab_recording"`;
- streaming enabled;
- server token counts enabled;
- no client input truncation;
- open-loop replay disabled;
- real tool execution enabled;
- a non-empty Docker image supplied;
- per-recording warmup enabled; and
- cache bust target `system_prefix` with run scope.

Default validation fails closed. If the existing `unsafe_override` contract is
used, bypassable conflicts may proceed only with
`submission_valid == false`, the complete violation list, and stable invalid
reason tags. Capability incompatibilities such as a virtual clock with real
tools remain hard failures and cannot be downgraded.

`ScenarioSpec` and its lock input DTO shall be widened generically rather than
special-casing Open LAB in CLI resolution. Existing scenarios retain their
current defaults and behavior. Cache-bust projection shall accept every typed
target it claims to model; it must not silently map `system_prefix` to `none`.

### Recording input contract

The adapter id is `openlab_recording`. It accepts:

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
OpenLabRecording
  format: String
  metadata:
    instance_id: Option<String>
    benchmark: Option<String>
    model_name: Option<String>
    docker_image: Option<String>
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

Trace id is `metadata.instance_id` when non-empty, otherwise the file name with
`.json` or `.json.gz` removed. A non-empty `metadata.docker_image` becomes the
trace's sandbox image and takes precedence over the run-level image; the
run-level image is the fallback for recordings without one. File and decode
errors must include the candidate path; event errors must include trace id,
event id, and event type.

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
- the recorded `tools` array through a segment handle;
- streaming from resolved run configuration;
- a per-call generation cap equal to recorded completion tokens, with absent or
  zero promoted to `1`;
- the recorded model only when `use_recorded_model` is enabled; and
- recorded `temperature`/`top_p` only when `use_recorded_sampling` is enabled.

Recorded sampling is provenance, not Open LAB playback configuration. When
family sampling is enabled, the lowerer overlays:

- `swebench`: `temperature = 0.0`, `parallel_tool_calls = true`;
- `pinchbench`: no server-side fields.

`drop_params` is a LiteLLM client option and is never sent. Explicit recorded
sampling wins over family defaults. An unknown non-empty family emits one
structured warning and sends no family fields.

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
pub enum GraphNode {
    Llm(LlmNode),
    Tool(ToolNode),
}

pub struct ToolNode {
    pub output: String,
    pub commands: Vec<String>,
    pub timeout_ns: Option<u64>,
}
```

`TraceRecord` shall additionally carry an optional strict `ToolSandboxSpec`
whose `container` selects that trace's Docker image. Open LAB lowering populates
it from `metadata.docker_image`. A non-empty per-trace container wins over the
run-level `dataset.graph.tool_image`; the latter is the fallback for PinchBench
or another recording without task-image metadata.

The serde representation must be explicitly tagged and reject an unknown node
kind. Existing LLM-only serialized fixtures remain readable through a deliberate
compatibility rule rather than an untagged ambiguous enum.

`GraphNode` shall provide kind-specific accessors for read channels, write
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

### Injectable tool and sandbox seams

Tool dispatch mirrors LLM dispatch but remains a distinct plane:

```rust
#[async_trait(?Send)]
pub trait ToolDispatcher {
    async fn open_trace(&self, trace: &TraceIdentity) -> Result<(), ToolDispatchError>;
    async fn dispatch(
        &self,
        node: &ToolNode,
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
    fn create(&self, trace: &TraceIdentity) -> Result<Rc<dyn ToolSandbox>, ToolSandboxError>;
}
```

Names may adapt to established module vocabulary, but the ownership and
injection boundaries are normative. `ToolDispatcherFactory` is application or
phase composition state and may be `Send + Sync`; the created dispatcher and
sandbox are worker-local `Rc` handles and need no `Send`/`Sync` supertraits.
Constructors accept unwrapped owners; they do not impose `Arc<Mutex<_>>`.

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

For each admitted trace instance:

1. Create its dispatcher/sandbox and unique workspace.
2. Open the sandbox outside the measured trace interval.
3. Capture `trace_start_ns` from the injected `Clock` immediately before the
   first graph node can fire.
4. Execute LLM and tool nodes through their independent dispatch seams.
5. Capture `trace_end_ns` after the terminal graph node drains.
6. Close the sandbox outside the measured interval on every terminal path.
7. Emit one terminal result supplement after cleanup has been attempted.

Open LAB's task summary excludes model/environment setup and replay warmup; the
Rust interval follows the same boundary. Tool-command duration is measured from
the same injected real `Clock`, ending before timeout cleanup begins. LLM
durations come from completed request records rather than a second wall-clock
measurement, preserving transport measurement authority.

Cleanup is RAII-backed and idempotent across success, trace failure,
cancellation, phase grace escalation, timeout, placement shutdown, and a
partially opened sandbox. A close failure is recorded as infrastructure detail.
It becomes the trace error only when no earlier trace error exists; it never
masks the primary cause. Cancellation waits for bounded cleanup before the
placement reports terminal.

Workspace paths are rooted in a run/cell-owned tool directory and use a
sanitized execution-instance slug. They never derive a filesystem path directly
from an untrusted trace id. The cleanup policy is explicit: successful
workspaces may be removed, while a configured diagnostic-retention mode may
keep failed workspaces and reports their paths in structured diagnostics.

### Local sandbox

`LocalSessionSandbox` starts one persistent shell session per trace workspace.
Every recorded command executes through a fresh `bash -lc`, preserving shell
semantics without leaking command-local options or traps into the session.
Commands share the workspace filesystem and environment changes represented by
files, as Open LAB tasks expect.

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

`DockerSessionSandbox` creates one detached container per trace instance with:

- the configured image;
- network disabled;
- the trace workspace bind-mounted at `/workspace`;
- `/workspace` as the working directory;
- a deterministic sanitized container-name prefix plus unique suffix; and
- no ambient endpoint credentials forwarded by default.

Commands use a persistent `docker exec -i ... bash` session and the same framed
fresh-`bash -lc` command protocol as the local backend. A command timeout
recycles the entire container so descendants and stale exec output cannot leak
into the next command.

Container create/start/exec/inspect/remove operations sit behind an injectable
`ContainerRuntime` trait. The stock implementation invokes the Docker CLI with
an argv vector, never a constructed shell string. Startup validates image and
mount errors before measurement begins. Close force-removes the container with
a bounded timeout and is idempotent. The run reports cleanup failures with the
container id/name for operator recovery.

The default scenario requires Docker because local execution cannot reproduce a
recording's task image. A per-recording image wins over the run-level fallback;
preflight validates every distinct resolved image. Local execution remains
available outside the scenario for host-device benchmarks and deterministic E2E
tests.

### Warmup plans and phase selection

`GraphInputBundle` shall carry profiling and corpus-authored warmup plans as
separate typed collections. A suggested shape is:

```rust
pub struct GraphInputBundle {
    pub plans: Vec<GraphTracePlan>,
    pub warmup_plans: Vec<GraphTracePlan>,
    pub segments: Arc<dyn SegmentStore>,
    pub metadata: GraphInputMetadata,
}
```

Each recording contributes exactly one warmup plan when enabled. Its stable id
is `warmup-<profiling-trace-id>` and it contains one LLM node with:

- one user message: `Reply with exactly: ok`;
- generation cap `8`, marked as an authored node cap so generic warmup cap
  rewriting cannot replace it;
- the first recorded call's tools;
- the same family sampling as profiling; and
- no tool node.

The WARMUP phase selects `warmup_plans` when non-empty; PROFILING selects only
`plans`. Corpus warmups do not wrap or cycle unless a phase explicitly defines
that behavior. Their request records retain normal warmup phase identity but do
not contribute to profiling artifacts or metrics.

Corpus-authored warmup and the existing t-star/cache-pressure warmup are
different concepts. Phase preparation shall represent their source explicitly,
reject an unsupported ambiguous combination, and never run a generic snapshot
rewrite over tool nodes. Corpus-authored Open LAB warmups are cache-isolation
exempt: upstream sends the user-only warmup through the unwrapped live model,
then applies the run namespace only to profiling replay. The warmup primes the
endpoint and client path, not the measured trajectory's isolated token prefix.

### Run-scoped cache isolation

Open LAB's `isolate_replay_messages` prefixes the first wire message once per
replay invocation with a random namespace. Native parity shall mint one marker
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

The marker text need not reproduce Open LAB's random digits byte for byte, but
the cache semantics are normative: cross-run profiling prefixes differ and all
profiling requests within one run share the namespace. Warmup requests have no
marker. The A/B harness normalizes profiling marker text while asserting its
scope and placement, and asserts that warmup stays unmodified.

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
- ordered LLM durations;
- ordered bounded tool-command measurements; and
- cleanup diagnostics.

The phase coordinator folds events in completion order into a
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
per-trace `ToolSandboxSpec` selection. `durations_s` preserves execution order in
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

- `OpenLabInputError`: discovery, decode, schema, duplicate id, empty corpus;
- `OpenLabLoweringError`: missing replay fields, invalid event sequence,
  topology, segment interning;
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

### Placement and cellular requirements

One placement target owns an entire trace and its sandbox. Thread-per-core
workers build their own tool dispatcher from the injected factory. Tool state
never crosses worker threads, and command execution never holds a lock across
`.await`.

Cellular partitioning remains trace-level. Every cell host must have access to
every distinct Docker image its partition may resolve and to the Docker runtime,
or to local execution prerequisites when the scenario is not active. Preflight
capability validation occurs on every cell before the phase barrier. A partial
preflight fails the run before warmup or profiling begins.

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

The integration harness launches the real Open LAB runner and the native
`aiperf` binary as separate subprocesses against the same deterministic mock or
live inference endpoint. Each path uses its own capture proxy so the harness
records ordered request bodies without conflating clients.

The comparison includes each recording's warmup and profiling calls. It
normalizes only deliberate client differences:

- the randomized cache-isolation marker text, while checking placement and
  scope;
- Open LAB's legacy `max_tokens` versus an endpoint's equivalent modern
  `max_completion_tokens` spelling where dialect selection requires it;
- Open LAB's `stream_options.include_usage`; and
- LiteLLM's client-only `drop_params`.

It does not normalize messages, roles, content, tools, tool-call ids, model,
temperature, top-p, parallel-tool setting, generation cap value, call count, or
call order. The harness must exercise the `openlab-default` scenario path so
scenario resolution itself is under test. It uses fresh workspaces because Open
LAB Docker tasks may leave root-owned files.

The harness separately compares trace summaries within a documented transport
overhead tolerance. It does not require tool duration equality across different
hosts; it requires the same command count/order and valid local measurements.

### Verification matrix

Implementation is complete only when these behaviors are covered:

| Layer | Required proof |
|---|---|
| DTO/discovery | JSON and gzip; sorted shallow directory; sniffed manifest skip; malformed explicit file; finite timestamps; duplicate/empty errors. |
| Lowering | exact message/tool bytes; prefix interning across calls/traces; output-cap floor; family and recorded-sampling precedence; relative timing; trailing tools; control-flow versus genuine tool failures. |
| Graph model | serde compatibility; topology/read/write channels; LLM versus total counts; flat-path refusal; every heterogeneous-node consumer audited. |
| Warmup | one plan per recording; stable id; exact prompt/cap/tools/sampling; WARMUP-only dispatch; no tools; explicit cache-isolation exemption. |
| Cache isolation | first-message string, structured, null, and no-system cases; one profiling marker within a run; a different marker across runs; unmodified warmup; no segment-store mutation. |
| Dispatcher | fake injection; lifecycle order; sequential command results; nonzero and timeout outcomes; continuation after successful timeout recycle; recovery failure abort; close-error precedence. |
| Local sandbox | persistent workspace; fresh shell semantics; combined output; sentinel collision; output bound; timeout descendant kill and clean recycle; idempotent open/close. |
| Docker sandbox | argv construction; network none; mount/workdir; unique names; startup error; timeout container recycle; bounded force cleanup; fake runtime injection. |
| Phase/placement | setup and cleanup excluded from trace wall time; cancellation cleanup; LLM-only request accounting; multi-worker fold determinism; no shared-file writes. |
| Cellular | per-cell distinct-image preflight; trace ownership; supplement serialization/fold; missing/duplicate/unknown-backend refusal; valid mixed-backend merge; controller-only artifacts. |
| Artifacts | exact schemas; successful-terminal-only inclusion; finite zero guards; warmup/partial-terminal exclusion; deterministic ordering; single and `mixed` backend labels; median/fractions/counts; artifact allowlist and sweep retention. |
| Product E2E | native binary against deterministic `aiperf-mock-server`, raw records and request bodies inspected; local and opt-in Docker tool execution. |
| A/B | real Open LAB and AIPerf subprocesses, warmup plus profiling body parity, scenario locks, normalized differences limited to the explicit list. |

At minimum, repository verification includes:

```bash
source .venv/bin/activate
cd rust
cargo fmt --check
cargo clippy -p aiperf-runtime --all-targets --features engine
cargo test -p aiperf-runtime
cargo test -p aiperf-runtime --features engine
cargo test -p aiperf-cli
cargo test -p aiperf-e2e-tests --test test_openlab
```

Docker tests are opt-in when the daemon or task image is unavailable, but their
fake-runtime unit coverage is unconditional. The local product E2E and A/B mock
traffic test are unconditional in the supported development environment.

### Implementation sequence

The port shall be delivered in dependency order, with each slice retaining a
green workspace:

1. Add strict recording DTOs, discovery, pure lowering fixtures, and register
   `OpenLabRecordingInputAdapter` without tool execution.
2. Introduce tagged `GraphNode`, audit every node consumer, and preserve LLM-only
   wire compatibility and fast-path behavior.
3. Add separate corpus warmup plans and phase selection; prove warmup/profiling
   isolation.
4. Complete graph cache-bust target projection and run-scoped first-message
   prefix materialization.
5. Add tool dispatcher/sandbox/process/container traits and deterministic fakes.
6. Implement local sandbox and then Docker sandbox behind the same seam.
7. Return typed trace supplements through local/threaded placement and fold them
   in the phase coordinator.
8. Extend cellular protocol/folding and artifact allowlists; add controller-only
   exporters for both JSON schemas.
9. Add `openlab-default` locks across CLI, Config v2, strict protocol, and
   scenario outcome reporting.
10. Add local/Docker product E2E and strengthen the A/B harness to include
    warmups, scenario resolution, and artifact checks.

Each slice updates this record and the architecture index with the code it
lands. Public DTOs and traits receive `///` documentation; new modules receive
`//!` documentation; every Rust source carries the NVIDIA SPDX header.

## Source anchors

### Native Rust prerequisites

- `rust/runtime/src/graph/{model.rs,input.rs,execution.rs,executor.rs,placement.rs,workload.rs}`.
- `rust/runtime/src/engine/{application.rs,graph_input.rs,graph_execution.rs,graph_phase_runtime.rs,protocol.rs,artifact_shipping.rs}`.
- `rust/runtime/src/agentx/{cache_bust.rs,scenario.rs}`.
- `rust/runtime/src/config/{model/dataset.rs,resolve.rs,validate.rs}`.
- `rust/cli/src/{flags.rs,load.rs,yaml.rs}`.
- `rust/e2e-tests/tests/test_graph_cellular.rs`.

### Python parity source at the pinned branch revision

- `../dynamo-graph-ir/src/aiperf/dataset/graph/adapters/openlab/{recording_reader.py,recording.py}`.
- `../dynamo-graph-ir/src/aiperf/graph/tool_dispatch/`.
- `../dynamo-graph-ir/src/aiperf/graph/sandbox/`.
- `../dynamo-graph-ir/src/aiperf/graph/dispatch/tool.py`.
- `../dynamo-graph-ir/src/aiperf/common/scenario/openlab_default.py`.
- `../dynamo-graph-ir/src/aiperf/timing/strategies/{agent_graph_replay.py,graph_warmup.py}`.
- `../dynamo-graph-ir/tests/{unit,integration}` paths containing `openlab`,
  `tool_dispatch`, `sandbox`, or `graph_tool`.
- `../dynamo-graph-ir/docs/specs/{openlab-parity-handoff.md,agent-graph-tool-execution.md,sandbox-resident-executor.md}`.

### Upstream Open LAB authority at the pinned revision

- External checkout `open-lab-benchmark`,
  `src/minisweagent/recording/{recorder.py,replay.py,cache_isolation.py,validation.py,warmup.py,wrappers.py}`.
- External checkout `open-lab-benchmark`,
  `src/minisweagent/config/benchmarks/{swebench.yaml,pinchbench.yaml}` and
  `src/minisweagent/run/{record_swebench.py,record_pinchbench.py}`.
