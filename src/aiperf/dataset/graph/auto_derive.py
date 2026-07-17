# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-derivation: linear-chat synthesis, channel auto-derive.

Pure transform on ParsedGraph.
"""

from __future__ import annotations

from typing import Any

import msgspec

from aiperf.dataset.graph.decode import decode_node
from aiperf.dataset.graph.models import (
    END_NODE_ID,
    START_NODE_ID,
    ChannelSpec,
    ChannelType,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    StaticEdge,
    TraceRecord,
    _collect_content_block_channels,
    _collect_prompt_array_channels,
)

_LINEAR_CHAT_NODE_ID = "_llm"
_MESSAGES_CHANNEL = "messages"


def auto_derive(parsed: ParsedGraph) -> ParsedGraph:
    """Apply all auto-derivation transforms in order. Returns a new ParsedGraph."""
    pb = parsed
    pb = _lift_trace_messages(pb)
    pb = _synthesize_linear_chat_if_needed(pb)
    pb = _auto_derive_channels(pb)
    return normalize_runtime_channels(pb)


def _lift_trace_messages(pb: ParsedGraph) -> ParsedGraph:
    """Lift the ``traces[].messages`` shorthand into ``initial_state.messages``.

    The shorthand is documented as equivalent to ``initial_state.messages``
    (see ``TraceRecord.messages``), so it applies regardless of whether the
    graph declares explicit nodes — not only on the linear-chat synthesis
    path. An already-present ``initial_state.messages`` wins.
    """
    if not any(t.messages is not None for t in pb.traces):
        return pb
    return msgspec.structs.replace(
        pb, traces=[_lift_messages_into_init(t) for t in pb.traces]
    )


def _synthesize_linear_chat_if_needed(pb: ParsedGraph) -> ParsedGraph:
    if pb.graph.nodes:
        return pb
    system_prompt = pb.graph.system
    prompt: list[Any] = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append("@" + _MESSAGES_CHANNEL)
    llm_node = decode_node(
        {
            "prompt": prompt,
            "output": _MESSAGES_CHANNEL,
            "streaming": True,
        }
    )
    new_graph = msgspec.structs.replace(
        pb.graph,
        nodes={_LINEAR_CHAT_NODE_ID: llm_node},
        state={
            **pb.graph.state,
            _MESSAGES_CHANNEL: ChannelSpec(
                type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
            ),
        },
        edges=list(pb.graph.edges)
        + [
            StaticEdge(source=START_NODE_ID, target=_LINEAR_CHAT_NODE_ID),
            StaticEdge(source=_LINEAR_CHAT_NODE_ID, target=END_NODE_ID),
        ],
    )
    return msgspec.structs.replace(pb, graph=new_graph)


def _lift_messages_into_init(t: TraceRecord) -> TraceRecord:
    if t.messages is None:
        return t
    new_init = dict(t.initial_state)
    new_init.setdefault("messages", t.messages)
    return msgspec.structs.replace(t, initial_state=new_init, messages=None)


def _auto_derive_channels(pb: ParsedGraph) -> ParsedGraph:
    declared = dict(pb.graph.state)
    for node in pb.graph.nodes.values():
        if not isinstance(node, LlmNode):
            continue
        for ch in _collect_prompt_array_channels(node):
            if ch not in declared:
                declared[ch] = ChannelSpec(
                    type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
                )
        for ch in _collect_content_block_channels(node):
            if ch not in declared:
                declared[ch] = ChannelSpec(
                    type=ChannelType.TEXT, reducer=ReducerName.OVERWRITE
                )
    if declared == pb.graph.state:
        return pb
    return msgspec.structs.replace(
        pb, graph=msgspec.structs.replace(pb.graph, state=declared)
    )


def normalize_runtime_channels(parsed: ParsedGraph) -> ParsedGraph:
    """Return ``parsed`` with safe inferred channel declarations added."""
    # Collect the trace-driven channels that must be declared on the top-level
    # graph; subgraphs only gain their own node-output channels.
    trace_channels: list[str] = []
    for trace in parsed.traces:
        trace_channels.extend(trace.initial_state)
        for outputs in trace.replay_outputs.values():
            trace_channels.extend(outputs)

    new_graph = _normalize_graph(parsed.graph, extra_channels=trace_channels)
    if new_graph is parsed.graph:
        return parsed
    return msgspec.structs.replace(parsed, graph=new_graph)


def _normalize_graph(graph: GraphRecord, *, extra_channels: Any) -> GraphRecord:
    new_state = dict(graph.state)
    for node in graph.nodes.values():
        for channel in _node_output_channels(node):
            _declare_missing(new_state, channel)
    for channel in extra_channels:
        _declare_missing(new_state, channel)
    if new_state == graph.state:
        return graph
    return msgspec.structs.replace(graph, state=new_state)


def _node_output_channels(node: Any) -> list[str]:
    write_channels = getattr(node, "write_channels", None)
    if isinstance(write_channels, list):
        return [channel for channel in write_channels if isinstance(channel, str)]
    return []


def _declare_missing(state: dict[str, ChannelSpec], channel: str) -> None:
    if channel in state:
        return
    state[channel] = _default_channel_spec(channel)


def _default_channel_spec(channel: str) -> ChannelSpec:
    if channel == "messages":
        return ChannelSpec(type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES)
    return ChannelSpec(type=ChannelType.TEXT, reducer=ReducerName.OVERWRITE)
