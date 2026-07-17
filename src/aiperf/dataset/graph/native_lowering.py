# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-graph lowering onto the unified interned segment store.

Lowers a hand-authored (native) :class:`ParsedGraph` onto the same content
plane the weka and dynamo adapters emit: every ``LlmNode`` prompt is
canonicalized to ``[{role, content: str}, ...]`` messages, interned into a
content-addressed :class:`SegmentPool`, and stamped with
``metadata["trie"]["prompt_segment_ids"]``. The build plane then drains the
pool into the ONE unified interned store exactly as it does for the trace
adapters (``dataset_manager`` routes on ``parsed.segment_pool``), and the
worker materializes prompts by handle.

Dynamic content (dynamic-content-pool spec, Phase 3): a ``@channel`` prompt
reference whose channel is written by ancestor ``LlmNode``\\ s lowers to
DYNAMIC SLOTS — an ordered assembly program stamped as
``metadata["trie"]["assembly"]`` whose tokens interleave interned static
segments with references to producer nodes. At run time the sticky worker
fills slots from its dynamic pool (the producers' actual responses).

- Array-level splices (messages channels): composition at the read point
  reconstructs the FULL user/assistant alternation = init-seeded messages, then
  for each completion-ordered ancestor writer its authored user turn (its
  ``delta`` = its prompt minus its own ``@C`` splice, static, interned) followed
  by a slot for its live reply. Every writer except the first must itself read
  the channel (the accumulate shape) so injected input gates make build-time
  order equal runtime commit order. The reading node is excluded (read observes
  pre-write state), so the linear-chat shorthand stays fully static. This is why
  the naive single-channel chain (``output: C`` on each turn, ``["@C", <turn>]``)
  yields a well-formed conversation with no re-stating of prior user turns.
- Block-level refs (text channels): at most one ancestor writer; the slot
  value is the writer's response, composed INTO the containing message
  (static text parts + slot-value parts, concatenated).
- Producers referenced by any slot are stamped ``capture: true``; readers get
  an injected ``ChannelRequirement`` per spliced channel so their credits are
  minted only after every producer resolved (completion gating regardless of
  edge anchors).

Trace-dependent content (init splices) lowers to PER-TRACE graphs
(``parsed.graphs[trace.id]`` + ``trace.graph_ref``) against one shared pool.
The lowering is a pure function of the parsed file (content-addressed ids, no
RNG, no clock), so the dataset plane and timing plane re-parses stamp
identical graphs.

Canonical wire shape: message ``content`` is always a plain string; a content
list of string blocks is concatenated in order (no separator). Typed /
multimodal / directive blocks are gated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import msgspec

from aiperf.dataset.graph.models import (
    ChannelRequirement,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
    _channel_ref,
    _collect_content_block_channels,
    _collect_prompt_array_channels,
)
from aiperf.dataset.graph.segment_ir.envelope import stamp_prompt_segment_ids
from aiperf.dataset.graph.segment_ir.pool import SegmentPool

__all__ = ["lower_native_to_unified"]


def lower_native_to_unified(parsed: ParsedGraph) -> ParsedGraph:
    """Stamp a native parse onto the unified segment-store content plane."""
    _gate_unsupported_topology(parsed)
    plan = _analyze_dynamic_refs(parsed)

    pool = SegmentPool()
    has_refs = any(
        _collect_prompt_array_channels(node) or _collect_content_block_channels(node)
        for node in parsed.graph.nodes.values()
    )
    if not has_refs:
        stamped = _stamp_graph(pool, parsed.graph, plan, trace=None)
        return msgspec.structs.replace(parsed, graph=stamped, segment_pool=pool)

    if not parsed.traces:
        raise NotImplementedError(
            "traces: '@channel' prompt references splice per-trace initial_state; "
            "a workload with no trace records has no content to lower"
        )
    seen_ids: set[str] = set()
    graphs: dict[str, GraphRecord] = {}
    new_traces: list[TraceRecord] = []
    for trace in parsed.traces:
        if trace.graph_ref is not None:
            raise NotImplementedError(
                f"traces[{trace.id}].graph_ref: pre-set graph refs are not "
                f"supported by the unified-store lowering"
            )
        if trace.id in seen_ids:
            raise NotImplementedError(
                f"traces[{trace.id}]: duplicate trace ids cannot key per-trace "
                f"lowered graphs"
            )
        seen_ids.add(trace.id)
        graphs[trace.id] = _stamp_graph(pool, parsed.graph, plan, trace=trace)
        new_traces.append(msgspec.structs.replace(trace, graph_ref=trace.id))
    first_graph = graphs[new_traces[0].id]
    return msgspec.structs.replace(
        parsed,
        graph=first_graph,
        graphs=graphs,
        traces=new_traces,
        segment_pool=pool,
    )


def _gate_unsupported_topology(parsed: ParsedGraph) -> None:
    for nid, node in parsed.graph.nodes.items():
        if not isinstance(node, LlmNode):
            raise NotImplementedError(
                f"graph.nodes.{nid}: node_type '{node.node_type}' is not supported "
                f"by the unified-store lowering (LlmNode only)"
            )
        # replay_reducers flips the node's write to replace_channel, wiping the
        # init seed the static composition relies on.
        if "replay_reducers" in (node.metadata or {}):
            raise NotImplementedError(
                f"graph.nodes.{nid}.metadata.replay_reducers: reducer overrides "
                f"are not supported by the unified-store lowering"
            )
    for edge in parsed.graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        # Rule-55 shape, enforced here because the profile pipeline lowers at
        # parse time without running validate(): a first-token anchor with no
        # dispatch fallback would otherwise be silently lowered as a
        # completion edge (see _completion_adjacency).
        if (
            edge.delay_after_predecessor_first_token_us is not None
            and edge.delay_after_predecessor_start_us is None
        ):
            raise NotImplementedError(
                f"graph.edges[{edge.source}->{edge.target}]: edge is "
                f"first-token-anchored (delay_after_predecessor_first_token_us) "
                f"but sets no delay_after_predecessor_start_us; the runtime "
                f"needs the start delay as the dispatch fallback when the "
                f"predecessor terminates without a first token (validator "
                f"rule 55)"
            )


# ---------------------------------------------------------------------------
# Dynamic-ref analysis: which prompt refs become slots, and their gates
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LoweringPlan:
    """Per-parse dynamic-slot plan (topology-derived; trace-independent)."""

    array_slots: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    """(reader, channel) -> completion-ordered writer node ids (array splice)."""
    block_slots: dict[tuple[str, str], str] = field(default_factory=dict)
    """(reader, channel) -> single writer node id (block-level text ref)."""
    capture_nodes: set[str] = field(default_factory=set)
    """Producers referenced by any slot; stamped ``capture: true``."""
    injected_inputs: dict[str, dict[str, int]] = field(default_factory=dict)
    """reader -> {channel: required write count} (completion gating)."""


def _completion_adjacency(graph: GraphRecord) -> dict[str, set[str]]:
    """src -> targets over COMPLETION edges only.

    Start-anchored and first-token-anchored edges schedule the successor while
    the predecessor is still in flight, so they contribute no completion
    ordering; excluding them makes "ancestor" mean commit-before by
    construction.
    """
    adj: dict[str, set[str]] = {}
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        if edge.delay_after_predecessor_start_us is not None:
            continue
        adj.setdefault(edge.source, set()).add(edge.target)
    return adj


def _ancestors(adj: dict[str, set[str]], target: str) -> set[str]:
    rev: dict[str, set[str]] = {}
    for src, dsts in adj.items():
        for dst in dsts:
            rev.setdefault(dst, set()).add(src)
    seen: set[str] = set()
    queue: deque[str] = deque(rev.get(target, ()))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(rev.get(node, ()))
    return seen


def _anchored_pairs(graph: GraphRecord) -> set[tuple[str, str]]:
    return {
        (e.source, e.target)
        for e in graph.edges
        if isinstance(e, StaticEdge) and e.delay_after_predecessor_start_us is not None
    }


def _analyze_dynamic_refs(parsed: ParsedGraph) -> _LoweringPlan:
    """Classify every ``@channel`` ref as static or slot; gate the unlowerable.

    Slot legality: writers must be completion-ancestors of
    the reader; array-splice writers must form a completion-ordered chain in
    which every writer except the first also reads the channel; block refs
    take at most one writer and must not mix with init seeding; anchored
    edges between a producer and its reader (or successive writers) are
    contradictory timing intent.
    """
    graph = parsed.graph
    plan = _LoweringPlan()
    writers: dict[str, list[str]] = {}
    for nid, node in graph.nodes.items():
        writers.setdefault(node.output, []).append(nid)
    adj = _completion_adjacency(graph)
    anchored = _anchored_pairs(graph)
    ancestors: dict[str, set[str]] = {nid: _ancestors(adj, nid) for nid in graph.nodes}
    array_refs: dict[str, set[str]] = {
        nid: set(_collect_prompt_array_channels(node))
        for nid, node in graph.nodes.items()
    }

    for nid, node in graph.nodes.items():
        loc = f"graph.nodes.{nid}.prompt"
        block_refs = set(_collect_content_block_channels(node))
        for ch in sorted(array_refs[nid] | block_refs):
            all_writers = sorted(w for w in writers.get(ch, []) if w != nid)
            if not all_writers:
                continue  # static (init-seeded or self-written)
            # A reader sees only the channel's writers that COMMITTED before it
            # (its completion-ancestors). DESCENDANT writers are future writes
            # this read legitimately precedes (read-before-write) — excluded,
            # not an error (this is what lets the accumulate chain's root read
            # @C past its own downstream re-writers). CONCURRENT writers
            # (neither ancestor nor descendant) are a genuine race → gate.
            others = [w for w in all_writers if w in ancestors[nid]]
            concurrent = [
                w
                for w in all_writers
                if w not in ancestors[nid] and nid not in ancestors[w]
            ]
            if concurrent:
                raise NotImplementedError(
                    f"{loc}: '@{ch}' is written by LlmNode(s) {concurrent} "
                    f"concurrent with '{nid}' (neither ancestor nor descendant); "
                    f"a dynamic slot needs a deterministic commit order"
                )
            if not others:
                continue  # only future (descendant) writers → static for this reader
            for w in others:
                if (w, nid) in anchored:
                    raise NotImplementedError(
                        f"graph.edges[{w}->{nid}]: a start/first-token-anchored "
                        f"edge from slot producer '{w}' to its reader '{nid}' "
                        f"is contradictory timing intent (the anchor dispatches "
                        f"the reader while the producer is still in flight)"
                    )
            if ch in block_refs and ch not in array_refs[nid]:
                if len(others) > 1:
                    raise NotImplementedError(
                        f"{loc}: '@{ch}' block ref has multiple writers "
                        f"{others}; overwrite channels latch a single writer "
                        f"for the trace lifetime"
                    )
                if any(ch in t.initial_state for t in parsed.traces):
                    raise NotImplementedError(
                        f"{loc}: '@{ch}' block ref channel is both init-seeded "
                        f"and written by '{others[0]}'; the init value is never "
                        f"observable past the completion gate — remove one "
                        f"source"
                    )
                plan.block_slots[(nid, ch)] = others[0]
            else:
                _gate_array_splice_shape(loc, nid, node, ch, writers)
                ordered = _ordered_writer_chain(loc, ch, others, ancestors, anchored)
                for i, w in enumerate(ordered[1:], start=1):
                    if ch not in array_refs[w]:
                        raise NotImplementedError(
                            f"{loc}: '@{ch}' writers must chain through reads — "
                            f"writer '{w}' does not itself splice '@{ch}', so "
                            f"its commit order after "
                            f"'{ordered[i - 1]}' is not gated"
                        )
                # The ROOT writer has no committed ancestor writers, so the
                # reader-side Gate A above never runs for it; a non-leading
                # (or repeated) '@{ch}' there would survive to _delta_messages,
                # which drops only a LEADING splice and re-expands the init
                # seed for anything else — silently duplicating it in every
                # reader's reconstruction.
                for w in ordered:
                    _gate_array_splice_shape(
                        f"graph.nodes.{w}.prompt", w, graph.nodes[w], ch, writers
                    )
                plan.array_slots[(nid, ch)] = ordered
            plan.capture_nodes.update(others)
            plan.injected_inputs.setdefault(nid, {})[ch] = len(others)

    _gate_init_bearing_root_reads(parsed, plan, writers, adj, array_refs)
    return plan


def _gate_array_splice_shape(
    loc: str, nid: str, node: LlmNode, ch: str, writers: dict[str, list[str]]
) -> None:
    """Gate A: a slot channel's ``@C`` array-splice must be single, and leading
    for any node that also writes ``C``.

    Runs for every reconstructing reader AND every writer in its ordered
    chain (the root writer has no committed ancestor writers, so only the
    chain-side call reaches it). A repeated ``@C`` would double-expand the
    whole reconstructed conversation. A writer whose ``@C`` is not
    the first prompt item has an ill-defined ``delta`` (items before ``@C``
    would be misplaced after the reconstructed history), so its ``@C`` must
    lead.
    """
    positions = [
        i
        for i, item in enumerate(node.prompt)
        if isinstance(item, str) and _channel_ref(item) == ch
    ]
    if len(positions) > 1:
        raise NotImplementedError(
            f"{loc}: '@{ch}' appears {len(positions)} times as an array splice; "
            f"a slot channel may be spliced at most once per node (a repeat "
            f"would duplicate the reconstructed conversation)"
        )
    is_writer = nid in writers.get(ch, [])
    if is_writer and positions and positions[0] != 0:
        raise NotImplementedError(
            f"{loc}: node writes '{ch}' and splices '@{ch}', so '@{ch}' must be "
            f"the first prompt item (its authored delta is everything after the "
            f"history it reads); move '@{ch}' to the front"
        )


def _gate_init_bearing_root_reads(
    parsed: ParsedGraph,
    plan: _LoweringPlan,
    writers: dict[str, list[str]],
    adj: dict[str, set[str]],
    array_refs: dict[str, set[str]],
) -> None:
    """Gate B: an init-bearing messages channel's ROOT writer must read ``@C``
    — but only when some reader actually reconstructs the channel.

    A reconstructing reader splices ``@C`` with committed writers (an array
    slot), expanding ``[init, …writer deltas/replies…]``. If such a reader
    exists and the root writer does not read ``@C``, the root dispatches
    without the seeded prefix while readers reconstruct ``[init, …]`` —
    inserting context the root never saw (unfaithful) and diverging its
    interned handles from the reconstruction. Requiring the root to read
    ``@C`` makes every writer see the same prefix it is later reconstructed
    with.

    When NO reader reconstructs the channel (e.g. a write-only channel whose
    init seed is consumed only as a static block ref, the "rewrite your own
    draft" workload), the divergence cannot occur and the gate must not fire.
    """
    reconstructed = {ch for (_, ch) in plan.array_slots}
    init_channels = {ch for t in parsed.traces for ch in t.initial_state}
    for ch in sorted(init_channels):
        ch_writers = writers.get(ch, [])
        if not ch_writers:
            continue  # pure static init splice; no reconstruction
        if ch not in reconstructed:
            continue  # nothing reconstructs [init, ...]; no divergence possible
        ch_ancestors = {w: _ancestors(adj, w) for w in ch_writers}
        roots = [
            w
            for w in ch_writers
            if not any(o in ch_ancestors[w] for o in ch_writers if o != w)
        ]
        for root in roots:
            if ch not in array_refs.get(root, set()):
                raise NotImplementedError(
                    f"graph.nodes.{root}.prompt: channel '{ch}' is init-seeded "
                    f"and written here, but this root writer does not splice "
                    f"'@{ch}'; it would dispatch without the seed while readers "
                    f"reconstruct it — add '@{ch}' as the first prompt item"
                )


def _ordered_writer_chain(
    loc: str,
    channel: str,
    writer_ids: list[str],
    ancestors: dict[str, set[str]],
    anchored: set[tuple[str, str]],
) -> list[str]:
    """Totally order writers by completion ancestry, or gate."""
    ordered = sorted(
        writer_ids, key=lambda w: sum(1 for o in writer_ids if o in ancestors[w])
    )
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if earlier not in ancestors[later]:
            raise NotImplementedError(
                f"{loc}: '@{channel}' writers {sorted(writer_ids)} are not "
                f"totally completion-ordered ('{earlier}' and '{later}' are "
                f"mutually unordered); parallel producers of one spliced "
                f"channel have no deterministic assembly order"
            )
        if (earlier, later) in anchored:
            raise NotImplementedError(
                f"graph.edges[{earlier}->{later}]: successive writers of "
                f"spliced channel '@{channel}' must be completion-ordered; a "
                f"start/first-token-anchored edge between them is "
                f"contradictory timing intent"
            )
    return ordered


# ---------------------------------------------------------------------------
# Stamping: canonical messages -> pool segments + assembly tokens
# ---------------------------------------------------------------------------


def _stamp_graph(
    pool: SegmentPool,
    graph: GraphRecord,
    plan: _LoweringPlan,
    trace: TraceRecord | None,
) -> GraphRecord:
    stamped = {
        nid: _stamp_node(pool, nid, node, graph.nodes, plan, trace)
        for nid, node in graph.nodes.items()
    }
    return msgspec.structs.replace(graph, nodes=stamped)


def _stamp_node(
    pool: SegmentPool,
    node_id: str,
    node: LlmNode,
    nodes: dict[str, LlmNode],
    plan: _LoweringPlan,
    trace: TraceRecord | None,
) -> LlmNode:
    assembly = _assemble_prompt(pool, node_id, node, nodes, plan, trace)
    segment_ids = [token["seg"] for token in assembly if "seg" in token]
    if not assembly:
        raise NotImplementedError(
            f"graph.nodes.{node_id}.prompt: an empty prompt cannot be lowered "
            f"to the unified store"
        )
    has_slots = any("seg" not in token for token in assembly)
    extra: dict[str, Any] = {}
    if has_slots:
        extra["assembly"] = assembly
    if node_id in plan.capture_nodes:
        extra["capture"] = True
    injected = plan.injected_inputs.get(node_id)
    if injected:
        node = msgspec.structs.replace(
            node, inputs=_upsert_inputs(node.inputs, injected)
        )
    return stamp_prompt_segment_ids(node, segment_ids, extra=extra or None)


def _upsert_inputs(
    existing: list[ChannelRequirement], injected: dict[str, int]
) -> list[ChannelRequirement]:
    """Merge injected completion gates with authored input requirements."""
    out: list[ChannelRequirement] = []
    remaining = dict(injected)
    for req in existing:
        needed = remaining.pop(req.channel, None)
        if needed is not None and isinstance(req.count, int) and req.count < needed:
            out.append(ChannelRequirement(channel=req.channel, count=needed))
        else:
            out.append(req)
    for channel, needed in sorted(remaining.items()):
        out.append(ChannelRequirement(channel=channel, count=needed))
    return out


def _assemble_prompt(
    pool: SegmentPool,
    node_id: str,
    node: LlmNode,
    nodes: dict[str, LlmNode],
    plan: _LoweringPlan,
    trace: TraceRecord | None,
) -> list[dict[str, Any]]:
    """Walk the prompt into assembly tokens, interning static messages.

    Token shapes (node-id references; the store builder resolves them to
    handles/ordinals): ``{"seg": <hex sid>}`` static message,
    ``{"s": {"src": <node_id>}}`` array-splice slot, and
    ``{"m": {"role": r, "parts": [{"t": text} | {"sv": <node_id>}]}}``
    composed message. Static segments parent-chain in prompt order.

    An array ``@C`` slot expansion reconstructs full alternation: init messages,
    then for each ordered writer its ``delta`` (authored user turn, static,
    interned inline for this trace) followed by its reply slot.
    """
    loc = f"graph.nodes.{node_id}.prompt"
    assembly: list[dict[str, Any]] = []
    parent: str | None = None

    def _intern(message: dict[str, str]) -> None:
        nonlocal parent
        parent = pool.add_text(
            role=message["role"], content=message["content"], parent_id=parent
        )
        assembly.append({"seg": parent})

    for item in node.prompt:
        if isinstance(item, str):
            channel = _channel_ref(item)
            if channel is None:
                raise NotImplementedError(
                    f"{loc}: top-level string items must be '@channel' messages splices"
                )
            writer_chain = plan.array_slots.get((node_id, channel))
            if writer_chain is None:
                for message in _init_messages(channel, trace, required=True):
                    _intern(message)
            else:
                for message in _init_messages(channel, trace, required=False):
                    _intern(message)
                for writer in writer_chain:
                    for message in _delta_messages(
                        writer, nodes[writer], channel, plan, trace
                    ):
                        _intern(message)
                    assembly.append({"s": {"src": writer}})
        elif isinstance(item, dict):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                raise NotImplementedError(
                    f"{loc}: message items require a non-empty 'role' string"
                )
            extra_keys = sorted(set(item) - {"role", "content"})
            if extra_keys:
                raise NotImplementedError(
                    f"{loc}: message keys {extra_keys} are not representable in "
                    f"the unified store (role/content only); dropping them "
                    f"silently would corrupt the authored wire bytes"
                )
            parts = _content_parts(loc, node_id, item.get("content"), plan, trace)
            if any("sv" in part for part in parts):
                assembly.append({"m": {"role": role, "parts": parts}})
            else:
                _intern(
                    {
                        "role": role,
                        "content": "".join(part["t"] for part in parts),
                    }
                )
        else:
            raise NotImplementedError(
                f"{loc}: unsupported prompt item type "
                f"'{type(item).__name__}' (expected str or message dict)"
            )
    return assembly


def _content_parts(
    loc: str,
    node_id: str,
    content: Any,
    plan: _LoweringPlan,
    trace: TraceRecord | None,
) -> list[dict[str, Any]]:
    """Resolve message content into ``{"t": text}`` / ``{"sv": node_id}`` parts."""
    if isinstance(content, str):
        return [{"t": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, str):
                raise NotImplementedError(
                    f"{loc}: non-string content blocks (directives / typed "
                    f"blocks) are not supported by the unified-store lowering"
                )
            channel = _channel_ref(block)
            if channel is None:
                parts.append({"t": block[1:] if block.startswith("@@") else block})
            elif (node_id, channel) in plan.block_slots:
                parts.append({"sv": plan.block_slots[(node_id, channel)]})
            else:
                parts.append({"t": _init_text(loc, channel, trace)})
        return parts
    raise NotImplementedError(
        f"{loc}: message content must be a string or a list of string blocks"
    )


def _delta_messages(
    writer_id: str,
    writer: LlmNode,
    channel: str,
    plan: _LoweringPlan,
    trace: TraceRecord | None,
) -> list[dict[str, str]]:
    """A writer's authored contribution to conversation ``channel``: its prompt
    messages minus the ``@channel`` splice it read (the recursive history).

    Computed inline per trace, in the WRITER's node_id context so block-slot
    classification resolves against the writer. v1 requires the delta
    to be **static** — it may not itself produce a live slot; gate loudly
    otherwise (a writer that both contributes to a conversation and splices
    another live channel is v2). Gate A (run for every writer in the slot
    chain, root included) guarantees a writer's ``@channel`` splice, if
    present, is the leading item, so dropping ``prompt[0]`` when it is that
    splice yields the whole delta.
    """
    loc = f"graph.nodes.{writer_id}.prompt (delta for @{channel})"
    items = list(writer.prompt)
    if items and isinstance(items[0], str) and _channel_ref(items[0]) == channel:
        items = items[1:]  # drop the leading @channel history read
    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            ch2 = _channel_ref(item)
            if ch2 is None:
                raise NotImplementedError(
                    f"{loc}: top-level string items must be '@channel' splices"
                )
            if (writer_id, ch2) in plan.array_slots:
                raise NotImplementedError(
                    f"{loc}: a conversation writer's delta may not splice live "
                    f"channel '@{ch2}' (v1 static-delta only)"
                )
            out.extend(_init_messages(ch2, trace, required=True))
        elif isinstance(item, dict):
            role = item.get("role")
            if not isinstance(role, str) or not role:
                raise NotImplementedError(
                    f"{loc}: message items require a non-empty 'role' string"
                )
            extra_keys = sorted(set(item) - {"role", "content"})
            if extra_keys:
                raise NotImplementedError(
                    f"{loc}: message keys {extra_keys} are not representable in "
                    f"the unified store (role/content only)"
                )
            parts = _content_parts(loc, writer_id, item.get("content"), plan, trace)
            if any("sv" in part for part in parts):
                raise NotImplementedError(
                    f"{loc}: a conversation writer's delta may not splice a live "
                    f"block channel (v1 static-delta only)"
                )
            out.append({"role": role, "content": "".join(p["t"] for p in parts)})
        else:
            raise NotImplementedError(
                f"{loc}: unsupported prompt item type '{type(item).__name__}'"
            )
    return out


# ---------------------------------------------------------------------------
# Init-value resolution (per trace)
# ---------------------------------------------------------------------------


def _init_messages(
    channel: str, trace: TraceRecord | None, *, required: bool
) -> list[dict[str, str]]:
    if trace is None or channel not in trace.initial_state:
        if not required:
            return []  # slot-backed splice: init prefix is optional
        trace_id = trace.id if trace is not None else "?"
        raise NotImplementedError(
            f"traces[{trace_id}].initial_state.{channel}: '@{channel}' splices "
            f"init-seeded content; every trace firing the reader must supply it"
        )
    value = trace.initial_state[channel]
    trace_id = trace.id
    loc = f"traces[{trace_id}].initial_state.{channel}"
    if not isinstance(value, list):
        raise NotImplementedError(
            f"{loc}: a messages splice requires a list of message dicts"
        )
    out: list[dict[str, str]] = []
    for i, message in enumerate(value):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise NotImplementedError(
                f"{loc}[{i}]: messages must be dicts with a 'role' string"
            )
        extra_keys = sorted(set(message) - {"role", "content"})
        if extra_keys:
            raise NotImplementedError(
                f"{loc}[{i}]: message keys {extra_keys} are not representable in "
                f"the unified store (role/content only); dropping them silently "
                f"would corrupt the authored wire bytes"
            )
        content = _canonical_static_content(f"{loc}[{i}]", message.get("content"))
        out.append({"role": message["role"], "content": content})
    return out


def _canonical_static_content(loc: str, content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, str):
                raise NotImplementedError(
                    f"{loc}: non-string content blocks (directives / typed "
                    f"blocks) are not supported by the unified-store lowering"
                )
            if _channel_ref(block) is not None:
                raise NotImplementedError(
                    f"{loc}: '@' refs are not supported inside init-seeded "
                    f"message content"
                )
            parts.append(block[1:] if block.startswith("@@") else block)
        return "".join(parts)
    raise NotImplementedError(
        f"{loc}: message content must be a string or a list of string blocks"
    )


def _init_text(loc: str, channel: str, trace: TraceRecord | None) -> str:
    if trace is None or channel not in trace.initial_state:
        trace_id = trace.id if trace is not None else "?"
        raise NotImplementedError(
            f"traces[{trace_id}].initial_state.{channel}: '@{channel}' splices "
            f"init-seeded content; every trace firing the reader must supply it"
        )
    value = trace.initial_state[channel]
    if not isinstance(value, str):
        raise NotImplementedError(
            f"{loc}: '@{channel}' text refs require a string initial_state value"
        )
    return value
