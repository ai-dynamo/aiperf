<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent Trace Replay

Replays recorded agent traces as an agent graph workload. The current importer
supports [Agent Trace Replay](https://github.com/NVIDIA/agent-trace-benchmark) performance-replay
recordings, an internal mini-swe-agent format that
benchmarks a local endpoint by replaying a recorded agent trajectory; this
adapter consumes the same recordings and drives the endpoint from AIPerf's graph
runtime instead of from an agent loop.

Select the Agent Trace Replay-format importer with `--graph-format mini_swe_agent_trace`, or let auto-detection claim
the file. For a parity run against the source runner, use `--scenario swe-mini-agent`
(see [Mandatory flags](#mandatory-flags)) to catch mis-configuration early.

```bash
aiperf profile \
  --scenario swe-mini-agent \
  --graph-format mini_swe_agent_trace \
  --no-open-loop-replay \
  --input-file /path/to/agent-trace-benchmark/benchmark/recordings/default/recordings \
  --model my-model \
  --url http://localhost:8000
```

Without `--scenario`, the equivalent explicit invocation is:

```bash
aiperf profile \
  --graph-format mini_swe_agent_trace \
  --no-open-loop-replay \
  --input-file /path/to/agent-trace-benchmark/benchmark/recordings/default/recordings \
  --model my-model \
  --url http://localhost:8000
```

The input path is either a single recording (`*.json` or `*.json.gz`) or a
directory of them. A directory is scanned one level deep; non-recording JSON
sitting alongside the trajectories, such as the set's `manifest.json`, is
rejected by a content sniff rather than by filename.

## Why no agent loop is needed

Agent Trace Replay's replay sends the recorded request body, not one the agent computed.
`ReplayedModelResponses.query` reads `provider_request["messages"]` straight out
of the recording, and the recorder stamps a `provider_request` on every model
call. The live response is timed and discarded; the recorded assistant message
is substituted before tools run.

The consequence is that the entire sequence of requests a replay puts on the
wire is fixed before the run starts. The agent loop, the Docker workspace, and
the tool execution reproduce a message list that is already stored in the file.
AIPerf replays that list directly.

What this does not reproduce is Agent Trace Replay's task grade and its whole-device
end-to-end time, both of which depend on tools actually executing. In
performance replay the grade is largely determined by the recording anyway,
since recorded responses drive tool execution regardless of what the live
endpoint returns. Whole-device end-to-end time is recoverable by running the
recorded commands for real; see [Tool execution](#tool-execution).

## Lowering

Each recording becomes one graph, keyed by trace id, so a directory lowers as a
multi-graph workload. Every `model_call` event becomes one `LlmNode` in recorded
order, chained by static edges.

| Recording field | Node field | Notes |
|---|---|---|
| `provider_request.messages` | interned prompt segments | Verbatim, via `SegmentPool.add_raw_message`: key order and extra keys such as `tool_calls` are preserved, so the replayed prompt tokenizes identically. |
| `response_message.extra.response.usage.completion_tokens` | `max_tokens` | The wire generation cap, matching Agent Trace Replay's `replay_max_tokens_from_recording`. A recorded length of 0 upgrades to 1 (see `wire_output_cap`). |
| `provider_request.tools` | `raw_tools` | Carried for its prompt-token footprint. |
| `timestamp - duration_ns` | `recorded_start_unix_ms`, `arrival_offset_us` | See the timestamp note below. |
| gap between consecutive calls | edge `delay_after_predecessor_us` | The recorded tool-execution window. Replaced by a real `ToolNode` when tool execution is on. |
| `metadata.instance_id`, else file stem | trace id | |

The segment pool is shared across every recording in a corpus, so identical
system prompts and task preambles intern once. On the shipped 8-trace default
set this collapses 5248 message slots into 444 segments — both a large memory
win on histories that grow quadratically with call count, and a faithful model
of the prefix sharing a real deployment sees.

### Timestamps are event ends, not starts

`Recorder.record_event` stamps `time.time()` when the event is recorded, which
is after the call returned. A call's start is therefore `timestamp - duration_ns`.
Treating the stamp as a start would shift every node late by its own duration
and make inter-call gaps negative.

### What is deliberately not carried

* **The recorded model string.** It is a LiteLLM identifier such as
  `openai/qwen3.6:27b`, usually not the endpoint model id of the system under
  test, and replaying one trajectory against a different model is the point.
  Opt in with `use_recorded_model=True`.
* **Recorded sampling parameters.** The run's own settings win. Opt in with
  `use_recorded_sampling=True`.
* **Agent Trace Replay's cache-isolation namespace.** Agent Trace Replay prefixes every live prompt
  with a per-invocation random namespace to defeat cross-run prefix-cache reuse.
  AIPerf measures cache behavior rather than suppressing it, and the graph plane
  already owns per-instance cache-bust marking (`stamp_cache_bust_marker`) for
  runs that want it. This means AIPerf numbers are not directly comparable to
  Agent Trace Replay numbers produced with cache isolation on.

## Mandatory flags

The Agent Trace Replay parity configuration requires the following settings. Omitting any
of them produces plausible but wrong numbers:

| Flag | Why mandatory |
|---|---|
| `--graph-format mini_swe_agent_trace` | Selects the Agent Trace Replay recording adapter. The scenario rejects an absent or different graph format rather than guessing from an input file. |
| `--no-open-loop-replay` | Open-loop pacing anchors every trace to a corpus-wide schedule zero. The default set spans 95 days, so five of eight traces sit 65–95 days ahead of the anchor and never dispatch. A full run without this flag sends 80 of 168 requests and looks like a hang. |
| `--use-server-token-count` | Without this flag AIPerf tokenises locally while Agent Trace Replay reads ISL/OSL from the server's `usage` field. Token counts are apples-to-oranges even when request bodies are byte-identical. |
| `--streaming` | Required for per-token latency metrics (TTFT, ITL). |
| `--graph-execute-tools` | Re-executes recorded tool commands instead of reporting the capture host's recorded tool delays. |

Pass `--scenario swe-mini-agent` to have the scenario lock auto-fill
`--use-server-token-count`, `--streaming`, and `--graph-execute-tools`; it also
requires `--graph-format mini_swe_agent_trace` and `--no-open-loop-replay`. An
explicit conflict raises a `ScenarioLockError` before the run starts. The
optional `--graph-tool-image <task-image>` selects Docker execution with a
network-isolated task image; without it, AIPerf executes commands in the host
sandbox.

## Replay knobs

The adapter consumes the shared trace-replay options; it adds no flags of its
own. `--inter-turn-delay-cap-seconds` caps the recorded tool gaps,
`--ignore-trace-delays` drops them, and `--num-dataset-entries` caps how many
recordings are lowered. Open-loop pacing, `--replay-speedup`, and t\* snapshot
windows work as described in
[Agent Graph timestamp replay](./graph-timestamp-replay-spec.md), since the
adapter preserves per-node recorded start times.

## Tool execution

By default nothing executes: the interval between two model calls collapses
into an edge delay replayed from the recording. That delay is the tool duration
measured on the CAPTURE host, so the default mode reports the capture host's
tool cost no matter what machine is under test. On the shipped `task_files`
PinchBench trace the same six commands took 74-103 ms in the recording and
34-40 ms on the test host.

With tool execution enabled (`--graph-execute-tools`), those gaps become
`ToolNode` steps that run the recorded commands for real on the machine under
test, in a sandbox owned by the trace.

The flag requires closed-loop dispatch and the adapter refuses the pairing
otherwise, so a tool-execution run reads:

```bash
aiperf profile \
  --graph-format mini_swe_agent_trace \
  --input-file /path/to/recordings \
  --graph-execute-tools \
  --no-open-loop-replay \
  --model my-model \
  --url http://localhost:8000
```

Agent Trace Replay recordings can select a task image through their metadata. PinchBench
recordings that omit one use `agent-trace-pinchbench:latest`, matching Agent Trace Replay's
own runner. To provide an image for a trace with no recording-level assignment,
add `--graph-tool-image <image>`:

```bash
aiperf profile \
  --graph-format mini_swe_agent_trace \
  --input-file /path/to/recordings \
  --graph-execute-tools \
  --graph-tool-image agent-trace-pinchbench:latest \
  --no-open-loop-replay \
  --model my-model \
  --url http://localhost:8000
```

Open-loop replay paces node arrival against the recorded timeline, which
already contains the recorded tool durations. A host slower than the capture
host degrades gracefully (paced targets fall into the past, so dispatch
proceeds on readiness), but a FASTER one is held back to the recorded schedule
— flooring end-to-end time at the capture host's wall clock and measuring the
recording instead of the device. That is the case the benchmark exists to
distinguish, so the combination is rejected up front rather than reported.

When a `ToolNode` is emitted, the recorded delay for that gap is **not** also
replayed. The tool now costs real time; replaying its recorded duration on top
would double-count it.

### Cache-isolation scope

`--scenario swe-mini-agent` uses `--cache-bust system-prefix` with
`--cache-bust-scope trace`: each trace instance gets one marker shared by all
of its own turns, but never by another trace. This prevents an earlier trace
from warming the KV cache for a later one. Use `--cache-bust-scope run` only
when intentional cross-trace cache sharing is part of the workload being
measured. The matching Agent Trace Replay mixed launcher uses
`MSWEA_REPLAY_CACHE_SCOPE=trace` by default; set it to `run` for that same
shared-cache experiment.

### OSL-normalized model time

The graph trace-summary artifact reports `normalized_model_s` when the server
returns OSL and first-token timing. It matches Agent Trace Replay: for each model call,
AIPerf keeps TTFT unchanged and rescales only the post-first-token generation
interval from `max(observed_osl - 1, 1)` decode tokens to
`max(recorded_osl - 1, 0)` decode tokens. Missing OSL or TTFT leaves that call's
raw model duration unchanged. `low_osl_model_calls` counts calls whose observed
OSL is strictly below half their recorded OSL.

### Batching and the terminal step

Every tool call recorded between two model calls becomes ONE `ToolNode` whose
`commands` run sequentially in recorded order, matching how the agent ran them.
Tool calls recorded after the FINAL model call become a terminal `ToolNode`
chained before `END`: an agent trajectory typically finishes with a submit or
finalize command, and dropping it for lack of a successor model call would lose
real measured work. Verified end to end, `task_files` lowers 6 of 6 recorded
commands and the whole `pinchbench-sample` corpus lowers 273 of 273.

### Agent control flow is not a failure

mini-swe-agent stores any raised exception in a tool call's `error` field, but
`Submitted` and its other `InterruptAgentFlow` siblings are control-flow
signals raised on INSPECTING a command's output, after it ran to completion.
Treating them as failures would silently drop the terminal command of every
graded trace — 43 of 849 tool calls across the corpus. The adapter excludes
only genuine command failures.

### Sandbox backends

The trace-level assignment takes precedence over `--graph-tool-image`; the
flag is a fallback. Both are read only when the graph carries tool nodes, so a
run without `--graph-execute-tools` never constructs a sandbox.

| Trace image | Backend | Shape |
|---|---|---|
| Recording metadata, PinchBench default, or `--graph-tool-image` | `DockerSessionSandbox` | One detached container per trace instance from that image, workspace bind-mounted at `/workspace`, `--network none`. |
| No assignment | `LocalSessionSandbox` | One persistent shell per trace instance, rooted at that instance's workspace. The host IS the device under test. |

Networking is off in the container case and is not configurable: a recorded
trajectory's commands were captured against a prepared workspace, and letting
them reach the network makes the measurement depend on whatever they find
there. Workspaces are one directory per trace INSTANCE, under
`<artifact-dir>/graph-tool-workspaces/`, so two concurrent replays of the same
recording cannot write into each other's files.

The per-command ceiling defaults to 15 minutes and is tunable with
`AIPERF_GRAPH_TOOL_COMMAND_TIMEOUT` (alongside
`AIPERF_GRAPH_TOOL_CONTAINER_STOP_TIMEOUT` and
`AIPERF_GRAPH_TOOL_SESSION_CLOSE_GRACE`). A command that trips it is reported
as timed out and its session is recycled — the ceiling FABRICATES a
measurement rather than truncating one, which is why the default is generous.

The persistent session is a TRANSPORT, not a shell the trajectory shares. It
pays the container round trip once for the whole trace instead of once per
command: roughly 37 ms per `docker exec` versus about 2.4 ms per command
through a held session. Each recorded command still runs as a fresh `bash -lc`
inside the session, so a `cd` or an export cannot leak into the next step — it
never did in the recording, where every call was its own `bash -lc`.

### Where the tool time comes out

Tool wall time is accumulated per trace and logged as a per-phase aggregate at
phase teardown — command count, total, mean, median, max, and the backend that
produced them. It is not an exported metric: a tool step issues no credit and
emits no request record, so it has nothing to ride on, and folding it into
request latency would corrupt the latency series.

### Open-loop pacing is refused

Tool execution cannot be combined with open-loop replay, and open-loop replay
is the DEFAULT, so a tool-execution run must pass `--no-open-loop-replay`.

Open-loop pacing releases each node against the recorded timeline, and that
timeline already contains the recorded tool durations. A host SLOWER than the
capture host degrades gracefully — its paced targets fall into the past and
dispatch proceeds on readiness — but a FASTER host is held back to the recorded
schedule, flooring its end-to-end time at the capture host's wall clock. That
is the exact host the benchmark exists to distinguish, and the failure mode is
a plausible wrong number rather than a crash, so the combination is refused
loudly at parse time.

### Comparability

Tool-execution numbers are not comparable to recorded-delay numbers: one
measures the machine under test, the other replays the capture host. Tool time
is reported as its own series. A tool step issues no credit and emits no
request record, so it never enters request latency.

## Refusals

A recording containing a failed model call is refused rather than lowered with
that call skipped, mirroring Agent Trace Replay's own `require_successful_model_calls`:
dropping a call would silently drop its prompt growth from the workload. A
recording whose model call carries no `provider_request` is refused for the same
reason. Two recordings in one corpus that resolve to the same trace id are a
hard error.

## Workload character

The shipped default set is prefill-dominated: roughly 2.5M input tokens against
31k output tokens, since a tool-calling step emits only a short tool-call JSON
body. Read the set's `manifest.json` `aggregate` block for the authoritative
counts rather than the prose in Agent Trace Replay's docs, which can lag a corpus refresh.
