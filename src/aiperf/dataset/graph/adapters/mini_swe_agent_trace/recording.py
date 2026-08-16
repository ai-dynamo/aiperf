# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower Agent Trace Replay performance-replay recordings into a graph ParsedGraph.

Agent Trace Replay (`agent-trace-benchmark`, a mini-swe-agent fork) replays a recorded agent
trajectory against a live endpoint: the live response is timed and discarded,
and the RECORDED assistant response is substituted before the agent executes
tools. Because `replay.py::query` re-sends the recorded
`provider_request["messages"]`, the sequence of requests a replay puts on the
wire is fully determined by the recording file. Reproducing it needs no agent
loop, no tool sandbox, and no response parsing -- only a faithful replay of a
predetermined request chain, which is what the graph runtime already is.

The lowering is therefore direct:

* each `model_call` becomes one :class:`LlmNode` whose prompt is the recorded
  message array interned VERBATIM (`SegmentPool.add_raw_message` preserves key
  order and every extra key, so `tool_calls` in assistant turns survive);
* the recorded output length becomes the wire generation cap, matching Agent
  Trace Replay's `replay_max_tokens_from_recording` behavior;
* recorded tool execution becomes an edge delay between consecutive nodes,
  so prompt growth and inter-request gaps replay without anything executing;
  with `execute_tools`, each gap's recorded commands instead become one
  :class:`ToolNode` that runs them for real, and the recorded gap is dropped
  so the tool's real cost is not double-counted;
* recorded wall-clock start times are preserved for open-loop pacing.

What is deliberately NOT carried: the recorded model string (a LiteLLM
identifier such as `openai/qwen3.6:27b`, which is usually not the endpoint
model id of the system under test) and the recorded sampling parameters. Both
default to the run's own settings so a recording can be replayed against a
different model, which is the entire point of the comparison. Opt back in with
`use_recorded_model` / `use_recorded_sampling`.

Agent Trace Replay additionally prefixes every live prompt with a per-invocation random
namespace to defeat cross-run prefix-cache reuse
(`recording/cache_isolation.py`). That is intentionally not replicated here:
AIPerf measures cache behavior rather than suppressing it, and the graph plane
already owns per-instance cache-bust marking
(:func:`~aiperf.graph.worker_materialize.stamp_cache_bust_marker`) for runs
that want it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording_reader import (
    AgentTraceRecording,
    AgentTraceRecordingError,
    EmptyAgentTraceRecordingError,
    RecordedEvent,
    discover_recordings,
    is_recording_file,
    iter_recordings,
)
from aiperf.dataset.graph.adapters.shared.output_cap import wire_output_cap
from aiperf.dataset.graph.models import (
    ChannelSpec,
    ExpectedTokens,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ProvenanceSpec,
    StaticEdge,
    ToolNode,
    ToolSandboxSpec,
    TraceRecord,
)
from aiperf.dataset.graph.parse_context import GraphParseContext
from aiperf.dataset.graph.segment_trie.envelope import stamp_prompt_segment_ids
from aiperf.dataset.graph.segment_trie.pool import SegmentPool

_logger = AIPerfLogger(__name__)

END = "END"
START = "START"

# The sampling Agent Trace Replay's own default playback puts on the wire, per task
# family.  These are RUN CONFIGURATION, not recorded workload: Agent Trace Replay
# rebuilds each live request from its own config plus the recorded messages,
# so the recording's sampling fields are provenance, not what gets sent.
#
# The A/B harness drives Agent Trace Replay via `record_swebench` / `record_pinchbench`
# with only the base benchmark yaml (not run-mixed-playback.sh's extra -c
# overrides), so the effective per-family wire params are:
#
#   swebench  — `src/minisweagent/config/benchmarks/swebench.yaml`
#               replay launcher: temperature=0.7, top_p=0.8, top_k=20, min_p=0,
#               parallel_tool_calls=true
#   pinchbench — `src/minisweagent/config/benchmarks/pinchbench.yaml`
#               model_kwargs: drop_params=true  (client flag, not sent)
#
# `drop_params` is deliberately absent: it is a LiteLLM client flag, not a
# server-side generation parameter.
#
# Expose via --[no-]use-family-sampling; disable to replay without any
# family-default injection (e.g. when matching a different playback script).
AGENT_TRACE_FAMILY_SAMPLING: dict[str, dict[str, Any]] = {
    "swebench": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "parallel_tool_calls": True,
    },
    "pinchbench": {},
}

WARMUP_PROMPT = "Reply with exactly: ok"
WARMUP_MAX_TOKENS = 8
AGENT_TRACE_PINCHBENCH_IMAGE = "agent-trace-pinchbench:latest"
SWE_BENCH_CWD = "/testbed"
SWE_BENCH_INTERPRETER = ("bash", "-c")


def _trace_id(path: Path, recording: AgentTraceRecording) -> str:
    """Stable trace id: the recorded instance id, else the file stem.

    `.json.gz` stems keep a trailing `.json`, so it is stripped -- the id ends
    up in metric labels and artifact paths, where a `foo.json` trace reads as a
    filename rather than a task.
    """
    if recording.metadata.instance_id:
        return recording.metadata.instance_id
    stem = path.name
    for suffix in (".json.gz", ".json"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _intern_prompt(pool: SegmentPool, messages: list[dict[str, Any]]) -> list[str]:
    """Intern one call's message array as a parent-chained segment path.

    Ids are prefix-dependent, so messages shared with an earlier call (the
    system prompt, the task statement, every settled turn) collapse to the same
    segments across calls AND across traces. The recorded histories grow by
    append, so the common prefix dedups and only each call's tail is new.
    """
    path: list[str] = []
    parent: str | None = None
    for message in messages:
        parent = pool.add_raw_message(message=message, parent_id=parent)
        path.append(parent)
    return path


def _edge_delay_us(
    previous: RecordedEvent,
    current: RecordedEvent,
    *,
    delay_cap_seconds: float | None,
    ignore_delays: bool,
) -> float | None:
    """Idle microseconds between the end of ``previous`` and the start of ``current``.

    This gap is where the recorded agent executed its tools, so replaying it as
    an edge delay reproduces the request cadence without running anything. The
    gap is clamped at zero: `timestamp` is wall-clock from the recording host,
    and a step whose events were stamped out of order would otherwise pull the
    successor backwards.
    """
    if ignore_delays:
        return None
    gap_s = current.start_unix_s - previous.timestamp
    if gap_s <= 0:
        return None
    if delay_cap_seconds is not None:
        gap_s = min(gap_s, delay_cap_seconds)
    return gap_s * 1e6


def _build_node(
    call: RecordedEvent,
    *,
    node_id: str,
    segment_ids: list[str],
    trace_start_unix_s: float,
    use_recorded_model: bool,
    use_recorded_sampling: bool,
    streaming: bool,
    family_sampling: dict[str, Any] | None = None,
) -> LlmNode:
    """One recorded model call as a dispatchable node."""
    request = call.provider_request
    assert request is not None  # guaranteed by AgentTraceRecording.model_calls()

    extra_body: dict[str, Any] = dict(family_sampling or {})
    if use_recorded_sampling:
        # An explicit opt-in to the RECORDED values wins over the family
        # defaults: the caller asked for this recording's own sampling.
        extra_body.update(
            {
                key: value
                for key, value in (
                    ("temperature", request.temperature),
                    ("top_p", request.top_p),
                )
                if value is not None
            }
        )
    extra_body = extra_body or None

    recorded_out = call.response_message.completion_tokens or 0
    node = LlmNode(
        prompt=[],
        output=f"{node_id}_out",
        streaming=streaming,
        model=request.model if use_recorded_model else None,
        max_tokens=wire_output_cap(recorded_out, node_id=node_id),
        raw_tools=request.tools,
        extra_body=extra_body,
        expected=ExpectedTokens(output_tokens=recorded_out)
        if recorded_out > 0
        else None,
        # RELATIVE offset only. `recorded_start_unix_ms` is deliberately NOT
        # stamped: open-loop replay paces every trace against one corpus-wide
        # `schedule_zero = min(recorded_start_unix_ms)`, which is meaningful for a
        # co-recorded trace stream but a category error here. Each Agent Trace Replay
        # recording is an INDEPENDENT task run, and Agent Trace Replay itself executes them
        # sequentially -- it never paces one task against another's wall clock.
        # The shipped default set spans 95 days (2026-05-05 to 2026-08-08), so
        # absolute pacing parks five of its eight traces 65-95 days into the
        # future and they never dispatch. Leaving every node unstamped puts the
        # corpus in the spec's "wholly untimestamped" case, where the graph falls
        # back to relative timing and replays each trace's own edge delays --
        # which is exactly Agent Trace Replay's behaviour.
        arrival_offset_us=max(0, int((call.start_unix_s - trace_start_unix_s) * 1e6)),
    )
    return stamp_prompt_segment_ids(node, segment_ids)


def _add_tool_node(
    *,
    commands: list[str],
    nodes: dict[str, LlmNode | ToolNode],
    tool_index: int,
    arrival_offset_us: int,
) -> tuple[str | None, int]:
    """Add one executable tool node when a recording gap contains commands."""
    if not commands:
        return None, tool_index

    tool_node_id = f"t{tool_index}"
    nodes[tool_node_id] = ToolNode(
        commands=commands,
        output=f"{tool_node_id}_out",
        arrival_offset_us=arrival_offset_us,
    )
    return tool_node_id, tool_index + 1


def _recorded_commands(
    recording: AgentTraceRecording, before_id: int, after_id: int | None
) -> list[str]:
    """Return non-empty recorded commands between two model calls."""
    return [
        command
        for event in recording.tool_calls_between(before_id, after_id)
        if (command := (event.action or {}).get("command", ""))
    ]


def _lower_recording(
    recording: AgentTraceRecording,
    *,
    pool: SegmentPool,
    delay_cap_seconds: float | None,
    ignore_delays: bool,
    use_recorded_model: bool,
    use_recorded_sampling: bool,
    streaming: bool,
    execute_tools: bool,
    family_sampling: bool = True,
    emit_warmup: bool = False,
) -> tuple[GraphRecord, GraphRecord | None]:
    """Lower one recording's model calls into a (profiling, warmup) graph pair.

    When ``emit_warmup=True``, returns a second ``GraphRecord`` containing a
    single LlmNode that mirrors Agent Trace Replay's per-recording warmup call ("Reply
    with exactly: ok", max 8 tokens, first call's tools, family sampling).
    The profiling record is unchanged.  When ``emit_warmup=False``, the second
    element is ``None``.
    """
    calls = recording.model_calls()
    if not calls:
        raise EmptyAgentTraceRecordingError("recording contains no model calls")

    # Agent Trace Replay puts different sampling on the wire per task family, so resolve
    # it from THIS recording rather than applying one setting to a mixed corpus.
    sampling: dict[str, Any] | None = None
    if family_sampling:
        family = recording.metadata.benchmark
        sampling = AGENT_TRACE_FAMILY_SAMPLING.get(family or "")
        if sampling is None and family:
            _logger.warning(
                lambda: f"unknown Agent Trace Replay benchmark family {family!r}; sending no "
                "family sampling. Replayed requests may differ from Agent Trace Replay's."
            )

    trace_start_unix_s = calls[0].start_unix_s
    nodes: dict[str, LlmNode | ToolNode] = {}
    edges: list[StaticEdge] = []
    tool_index = 0
    # The id the NEXT llm node hangs off: the previous llm node, or the tool
    # node standing between them when tools execute for real.
    previous_id: str | None = None

    for index, call in enumerate(calls):
        node_id = f"n{index}"
        nodes[node_id] = _build_node(
            call,
            node_id=node_id,
            segment_ids=_intern_prompt(pool, call.provider_request.messages),
            trace_start_unix_s=trace_start_unix_s,
            use_recorded_model=use_recorded_model,
            use_recorded_sampling=use_recorded_sampling,
            streaming=streaming,
            family_sampling=sampling,
        )

        if previous_id is None:
            edges.append(StaticEdge(source=START, target=node_id))
            previous_id = node_id
            continue

        tool_node_id, tool_index = _add_tool_node(
            commands=_recorded_commands(recording, calls[index - 1].id, call.id)
            if execute_tools
            else [],
            nodes=nodes,
            tool_index=tool_index,
            arrival_offset_us=max(
                0, int((calls[index - 1].timestamp - trace_start_unix_s) * 1e6)
            ),
        )

        if tool_node_id is not None:
            # The tool now costs real time, so the recorded gap must NOT also be
            # replayed as a delay -- that would double-count it.
            edges.append(StaticEdge(source=previous_id, target=tool_node_id))
            edges.append(StaticEdge(source=tool_node_id, target=node_id))
        else:
            edges.append(
                StaticEdge(
                    source=previous_id,
                    target=node_id,
                    delay_after_predecessor_us=_edge_delay_us(
                        calls[index - 1],
                        call,
                        delay_cap_seconds=delay_cap_seconds,
                        ignore_delays=ignore_delays,
                    ),
                )
            )
        previous_id = node_id

    if execute_tools:
        # A trajectory typically ends with a submit/finalize command recorded
        # AFTER the last model call. It is real measured work, so it gets its
        # own terminal node rather than being dropped for lack of a successor.
        tail_id, tool_index = _add_tool_node(
            commands=_recorded_commands(recording, calls[-1].id, None),
            nodes=nodes,
            tool_index=tool_index,
            arrival_offset_us=max(
                0, int((calls[-1].timestamp - trace_start_unix_s) * 1e6)
            ),
        )
        if tail_id is not None:
            edges.append(StaticEdge(source=previous_id, target=tail_id))
            previous_id = tail_id

    edges.append(StaticEdge(source=previous_id, target=END))
    provenance = ProvenanceSpec(
        source="agent-trace-benchmark",
        tool="mini_swe_agent_trace adapter",
        extra={
            "format": recording.format,
            "benchmark": recording.metadata.benchmark,
            "recorded_model": recording.metadata.model_name,
        },
    )
    profiling_record = GraphRecord(
        provenance=provenance,
        # Every node's output channel must be DECLARED, even though nothing
        # consumes it: the runtime channel store rejects a write to an
        # undeclared channel (`UnknownChannelError`), so an omitted state map
        # fails at dispatch rather than at parse.
        state={node.output: ChannelSpec() for node in nodes.values()},
        nodes=nodes,
        edges=edges,
    )

    if not emit_warmup:
        return profiling_record, None

    # Build the warmup graph: one LlmNode, same tools + family sampling as the
    # first real call, but a trivial prompt and a tight output cap.  This
    # mirrors Agent Trace Replay's per-recording warmup call ("Reply with exactly: ok").
    first_tools = calls[0].provider_request.tools
    warmup_extra: dict[str, Any] | None = dict(sampling) if sampling else None
    warmup_node = LlmNode(
        prompt=[],
        output="warmup_out",
        streaming=streaming,
        model=calls[0].provider_request.model if use_recorded_model else None,
        max_tokens=WARMUP_MAX_TOKENS,
        raw_tools=first_tools,
        extra_body=warmup_extra,
        arrival_offset_us=0,
        metadata={
            "dispatch": {
                "own_output_cap": True,
                # Agent Trace Replay invokes this excluded endpoint warmup directly on
                # the live model, before its replay wrapper adds isolation.
                "disable_cache_bust": True,
            }
        },
    )
    warmup_seg_ids = _intern_prompt(pool, [{"role": "user", "content": WARMUP_PROMPT}])
    warmup_node = stamp_prompt_segment_ids(warmup_node, warmup_seg_ids)
    warmup_record = GraphRecord(
        provenance=provenance,
        state={warmup_node.output: ChannelSpec()},
        nodes={"warmup": warmup_node},
        edges=[
            StaticEdge(source=START, target="warmup"),
            StaticEdge(source="warmup", target=END),
        ],
    )
    return profiling_record, warmup_record


def from_mini_swe_agent_trace(
    path: Path | str,
    *,
    delay_cap_seconds: float | None = None,
    ignore_delays: bool = False,
    use_recorded_model: bool = False,
    use_recorded_sampling: bool = False,
    family_sampling: bool = True,
    emit_warmup: bool = False,
    streaming: bool = True,
    num_traces: int | None = None,
    execute_tools: bool = False,
) -> ParsedGraph:
    """Convert an Agent Trace Replay recording (or a directory of them) into a ParsedGraph.

    Every recording becomes its own graph, keyed by trace id, so a corpus
    lowers as a multi-graph workload. The segment pool is shared across all of
    them: identical system prompts and task preambles intern once, which is
    both a large memory win on a corpus whose histories are quadratic in call
    count, and a faithful model of the prefix sharing a real deployment sees.
    """
    root = Path(path)
    recording_paths = discover_recordings(root)
    if not recording_paths:
        raise EmptyAgentTraceRecordingError(
            f"{root}: no Agent Trace Replay recordings found (expected *.json / *.json.gz "
            f"carrying a 'mini-swe-agent-recording-*' format marker)"
        )

    pool = SegmentPool()
    graphs: dict[str, GraphRecord] = {}
    traces: list[TraceRecord] = []
    warmup_traces: list[TraceRecord] = []

    for recording_path, recording in iter_recordings(root):
        if num_traces is not None and len(traces) >= num_traces:
            break
        trace_id = _trace_id(recording_path, recording)
        if trace_id in graphs:
            raise AgentTraceRecordingError(
                f"{recording_path}: duplicate trace id {trace_id!r}; two recordings "
                "in the same corpus resolve to the same instance id"
            )
        try:
            profiling_record, warmup_record = _lower_recording(
                recording,
                pool=pool,
                delay_cap_seconds=delay_cap_seconds,
                ignore_delays=ignore_delays,
                use_recorded_model=use_recorded_model,
                use_recorded_sampling=use_recorded_sampling,
                family_sampling=family_sampling,
                emit_warmup=emit_warmup,
                streaming=streaming,
                execute_tools=execute_tools,
            )
        except AgentTraceRecordingError as exc:
            raise AgentTraceRecordingError(f"{recording_path}: {exc}") from exc
        graphs[trace_id] = profiling_record
        sandbox_image = recording.metadata.docker_image
        if sandbox_image is None and recording.metadata.benchmark == "pinchbench":
            sandbox_image = AGENT_TRACE_PINCHBENCH_IMAGE
        tool_sandbox = (
            ToolSandboxSpec(
                container=sandbox_image,
                cwd=SWE_BENCH_CWD
                if recording.metadata.benchmark == "swebench"
                else None,
                interpreter=(
                    SWE_BENCH_INTERPRETER
                    if recording.metadata.benchmark == "swebench"
                    else None
                ),
            )
            if sandbox_image or recording.metadata.benchmark == "swebench"
            else None
        )
        traces.append(
            TraceRecord(id=trace_id, graph_ref=trace_id, tool_sandbox=tool_sandbox)
        )
        if warmup_record is not None:
            warmup_id = f"warmup-{trace_id}"
            graphs[warmup_id] = warmup_record
            warmup_traces.append(
                TraceRecord(
                    id=warmup_id, graph_ref=warmup_id, tool_sandbox=tool_sandbox
                )
            )

    if not traces:
        raise EmptyAgentTraceRecordingError(f"{root}: no usable recordings")

    profiling_traces = traces
    warmup_count = len(warmup_traces)
    _logger.info(
        lambda: f"lowered {len(profiling_traces)} Agent Trace Replay recording(s) from {root}: "
        f"{sum(len(graphs[t.id].nodes) for t in profiling_traces)} model calls, "
        f"{warmup_count} warmup trace(s), "
        f"{len(pool.by_id)} interned segments"
    )
    return ParsedGraph(
        graph=graphs[traces[0].id],
        graphs=graphs,
        traces=traces,
        warmup_traces=warmup_traces,
        segment_pool=pool,
    )


def _assert_no_open_loop_pacing(ctx: GraphParseContext) -> None:
    """Refuse open-loop pacing on a run that executes tools for real.

    Keys on the run's RESOLVED ``open_loop_replay`` rather than on whether the
    operator named a flag: open-loop replay defaults to True, so a guard keyed
    on ``replay_only_knobs`` (value-differs-from-default) could never fire on
    the default configuration -- the exact one a tool-execution run hits.

    A slower-than-recorded host degrades gracefully (paced targets fall into
    the past, so dispatch proceeds on readiness), but a FASTER one is held back
    to the recorded schedule -- exactly the host the benchmark exists to
    distinguish, and exactly the case that yields a plausible wrong number
    rather than a crash.
    """
    strict = [k for k in ctx.replay_only_knobs if k.startswith("--open-loop-strict")]
    if not ctx.open_loop_replay and not strict:
        return
    named = f" ({', '.join(strict)})" if strict else ""
    raise NotImplementedError(
        f"open-loop replay{named} cannot be combined with tool execution: node "
        "arrival is paced against the recorded timeline, which already "
        "contains the recorded tool durations. A host faster than the capture "
        "host would be held back to the recorded schedule, flooring end-to-end "
        "time at the capture host's wall clock -- measuring the recording "
        "instead of the device. Use --no-open-loop-replay for tool-execution "
        "runs."
    )


class AgentTraceRecordingAdapter:
    """Agent Trace Replay performance-replay recordings (mini-swe-agent recording 1.x)."""

    @classmethod
    def can_load(cls, path: Path) -> bool:
        if path.is_file():
            return is_recording_file(path)
        if path.is_dir():
            return bool(discover_recordings(path))
        return False

    @classmethod
    def parse(
        cls,
        path: Path,
        ctx: GraphParseContext | None = None,
        *,
        execute_tools: bool | None = None,
    ) -> ParsedGraph:
        """Convert ``path`` via :func:`from_mini_swe_agent_trace`.

        Only knobs the ctx actually sets are forwarded, so a ctx-less parse is
        byte-equal to the entry defaults. `ignore_trace_delays` and
        `run_streaming` are plain bools on the ctx rather than tri-state, so
        they forward unconditionally -- both match this entry's own defaults
        when the run never named them.

        ``execute_tools`` is tri-state so an explicit keyword (either value)
        beats the ctx and ``None`` defers to it: the CLI-driven path
        (``--graph-execute-tools``) passes NO keyword, since the sole production
        parse caller forwards no adapter kwargs. The open-loop refusal then runs
        against the RESOLVED value, not the keyword -- guarding the keyword
        alone would wave through exactly the CLI-driven combination it exists
        to reject.
        """
        effective = execute_tools
        if effective is None:
            effective = ctx.execute_tools if ctx is not None else None
        effective = bool(effective)
        if effective and ctx is not None:
            _assert_no_open_loop_pacing(ctx)
        kwargs: dict[str, Any] = {
            "ignore_delays": ctx.ignore_trace_delays if ctx else False,
            "execute_tools": effective,
            "family_sampling": ctx.use_family_sampling if ctx is not None else True,
            "emit_warmup": ctx.emit_warmup if ctx is not None else False,
        }
        if ctx is not None:
            if ctx.delay_cap_seconds is not None:
                kwargs["delay_cap_seconds"] = ctx.delay_cap_seconds
            if ctx.run_streaming is not None:
                kwargs["streaming"] = ctx.run_streaming
            if ctx.num_dataset_entries is not None:
                kwargs["num_traces"] = ctx.num_dataset_entries
        return from_mini_swe_agent_trace(path, **kwargs)
