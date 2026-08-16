# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build-time persistence for the segment trie.

The segment trie (produced by any adapter, e.g. the dynamo build-replay
path) realizes a trace as a flat ``LlmNode`` +
``StaticEdge`` graph whose nodes each carry
``metadata["trie"]["prompt_segment_ids"]`` -- a path into the content-addressed
:class:`~aiperf.dataset.graph.segment_trie.pool.SegmentPool`.

The trie graph persists as ONE :class:`GraphSegmentUnifiedBackingStore` -- a
content pool (every pool ``Segment`` written as ``id -> {role, content}``)
plus a per-node manifest region carrying each node's interned int-handle path,
``dispatch_overrides``, and ``stream``:

* :func:`build_unified_trie_store_interned` drains a whole-graph parse into the
  unified store eagerly. It is the in-process drain for EVERY format, and the
  byte-parity oracle for the streaming split below.
* :func:`iter_trace_segment_payloads` + :func:`build_unified_trie_store_from_payloads`
  are the streaming split for a worker-pool build (corpus-scale recorded
  sources): per-row workers emit :class:`TraceSegmentPayload`\\ s and
  the parent drains them into the SAME unified store shape, so the worker opens
  one reader either way.

Ordinals are assigned densely over the trie graph's ``LlmNode``s in their
recorded ``arrival_offset_us`` order (ties broken by node id), which is the
node-creation order :func:`~aiperf.dataset.graph.segment_trie.trie_content.assemble_trie_graph` emits -- a stable, deterministic
contract both the build plane (here) and the schedule plane resolve from the
same parsed graph.

Schedule-plane agreement: :func:`graph_path_catalog._catalog_for_trace` builds
the per-trace catalog by reusing THIS module's :func:`flat_trie_ordinals` over
the trace's flat
``LlmNode``s, so the live ``AgentGraphReplayStrategy`` dispatches every trie node at
the SAME ordinal this builder wrote its envelope at. The build->persist->
worker-materialize half (byte-correct prompts on dispatch) and the schedule half
(catalog/strategy enumerate + dispatch trie ``LlmNode``s) now share one ordinal
contract, so a full ``aiperf profile`` run dispatches the trie graph end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import orjson

from aiperf.dataset.graph.models import LlmNode, ParsedGraph, TraceRecord
from aiperf.dataset.graph.segment_trie.envelope import read_prompt_segment_ids
from aiperf.dataset.graph_segment_unified_store import NodeEnvelope

if TYPE_CHECKING:
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
    )


@dataclass(slots=True)
class TraceSegmentPayload:
    """Worker-serialized trie artifacts for ONE trace, shipped across the pool.

    A trie row crosses the worker boundary as content-addressed segments +
    ``prompt_segment_ids`` envelopes. This is the parallel payload the per-row
    worker emits when ``pg.segment_pool is not None`` -- the consumer
    (:func:`build_unified_trie_store_from_payloads`) drains ``segments`` into
    the unified content pool (idempotent dedup on the content-addressed id
    bounds RAM) and writes ``envelopes`` as per-node manifests, mirroring the
    eager :func:`build_unified_trie_store_interned` output. All fields are
    plain (``str`` / ``int`` / ``bytes`` / ``tuple``) so the dataclass pickles
    cleanly through ``multiprocessing.Pool``.
    """

    trace_id: str
    """Trace identifier."""

    node_ordinals: dict[str, int]
    """Node id -> dense ordinal map for the trace (the addressing catalog)."""

    envelopes: list[NodeEnvelope]
    """Per-node ``prompt_segment_ids`` envelopes (profiling variant)."""

    segments: list[tuple[str, str, str, str | None]] = field(default_factory=list)
    """The row's pool segments as ``(id, role, content, wire_json)`` tuples to ``put``.

    ``wire_json`` is the verbatim ``orjson.dumps(message)`` for a raw-authored dag
    segment (persisted byte-for-byte) or ``None`` for a role/content segment (the
    store derives the ``{"role", "content"}`` blob). Plain (``str``/``None``) so the
    dataclass still pickles cleanly through ``multiprocessing.Pool``."""

    structural_graph: bytes = b""
    """Content-free structural ``ParsedGraph`` (msgpack) for this row: topology +
    timing with ``replay_outputs`` emptied and the segment pool emptied (but kept
    non-None so the loaded graph still reads as a segment-store-backed graph). The streaming consumer
    merges these across rows into the corpus structural graph and writes the
    ``graph_meta`` sidecar, so the TimingManager loads it instead of the
    whole-corpus eager re-parse (which balloons to tens of GB). Empty (``b""``)
    for non-first payloads of a multi-trace parse (attached once, like
    ``segments``)."""


def trie_node_ordinals(trace_graph_nodes: dict[str, LlmNode]) -> dict[str, int]:
    """Assign a dense ordinal to each trie ``LlmNode`` in deterministic order.

    The order is the node's recorded ``arrival_offset_us`` (its warped recorded
    ``t``), ties broken by node id -- identical to the recorded-order list
    :func:`~aiperf.dataset.graph.segment_trie.trie_content.assemble_trie_graph` walks, so the build-plane ordinal here matches the
    schedule-plane ordinal resolved from the same parsed graph. Ordinals are dense
    (``0..N-1``) and unique within the trace.
    """
    ordered = sorted(
        trace_graph_nodes.items(),
        key=lambda kv: (kv[1].arrival_offset_us or 0, kv[0]),
    )
    return {node_id: ordinal for ordinal, (node_id, _) in enumerate(ordered)}


def flat_trie_ordinals(parsed: ParsedGraph, trace: TraceRecord) -> dict[str, int]:
    """THE shared build-plane/schedule-plane ordinal scheme.

    Both :func:`build_unified_trie_store_interned` (build) and
    ``graph_path_catalog._catalog_for_trace`` (schedule) call this so a node's
    build-time manifest ordinal equals its dispatch-time catalog ordinal. Every
    live producer emits a flat LlmNode graph, so this is
    :func:`trie_node_ordinals` over the trace's :func:`_trie_llm_nodes` keyed by
    bare node id.
    """
    return trie_node_ordinals(_trie_llm_nodes(parsed, trace))


def _trie_llm_nodes(parsed: ParsedGraph, trace: TraceRecord) -> dict[str, LlmNode]:
    """The trie graph's ``LlmNode``s for ``trace``.

    Non-LLM kinds are filtered out rather than assumed absent: a ``ToolNode``
    has no prompt manifest, so letting one through would both consume an ordinal
    (desynchronising the build-plane and schedule-plane ordinal schemes, which
    both derive from this function) and crash ``_trie_envelope``, which reads
    ``extra_body`` / ``model`` / ``max_tokens`` / ``raw_tools``.
    """
    from aiperf.dataset.graph.models import LlmNode, resolve_trace_graph

    top = resolve_trace_graph(parsed, trace)
    return {
        node_id: node
        for node_id, node in top.nodes.items()
        if isinstance(node, LlmNode)
    }


def _prompt_segment_ids(node: LlmNode) -> list[str] | None:
    """Read a trie node's ``prompt_segment_ids`` path, or ``None`` if absent."""
    return read_prompt_segment_ids(node)


def _trie_envelope(node: LlmNode, prompt_segment_ids: list[str]) -> dict:
    """Compose the per-node trie envelope the worker materializes from.

    Carries ``prompt_segment_ids`` (the segment-pool path the worker walks),
    a ``dispatch_overrides`` wire-body dict assembled from the node's
    ``extra_body`` (vendor keys) with the typed ``LlmNode.model`` /
    ``max_tokens`` / ``raw_tools`` FOLDED IN (``model`` and ``tools`` verbatim;
    the cap as the worker-mapped ``max_output_tokens`` entry; a hand-authored
    ``extra_body`` entry wins over any fold), the ``stream`` flag (from the
    typed ``LlmNode.streaming``), and -- when the adapter stamped per-node
    HTTP headers on the typed ``LlmNode.extra_headers`` field
    (dynamo session identity) -- an ``extra_headers`` map the worker attaches
    to the request HEADERS, never the body. The trie path is self-contained:
    the worker walks the pool path directly, no ancestor accumulation.

    When the adapter stamped ``metadata["dispatch"]["endpoint_extra_applied"] =
    True`` (it already folded the run's ``--extra-inputs`` into
    ``dispatch_overrides`` at parse), the envelope carries
    ``endpoint_extra_applied: True`` so the worker skips re-merging
    ``endpoint.extra`` and the adapter-owned values win. Both keys are OMITTED
    when unset, so header-less / flag-less corpora envelopes stay byte-identical.
    """
    overrides = dict(node.extra_body or {})
    if node.model is not None and "model" not in overrides:
        overrides["model"] = node.model
    if node.max_tokens is not None and "max_output_tokens" not in overrides:
        overrides["max_output_tokens"] = node.max_tokens
    if node.raw_tools is not None and "tools" not in overrides:
        overrides["tools"] = node.raw_tools
    envelope = {
        "prompt_segment_ids": prompt_segment_ids,
        "dispatch_overrides": overrides,
        "stream": bool(node.streaming),
    }
    if node.extra_headers:
        envelope["extra_headers"] = dict(node.extra_headers)
    dispatch_meta = (node.metadata or {}).get("dispatch", {})
    if dispatch_meta.get("endpoint_extra_applied"):
        envelope["endpoint_extra_applied"] = True
    if dispatch_meta.get("own_output_cap"):
        # The adapter stamped an explicit output cap that must not be overridden
        # by the worker's warmup cap (which is correct for BOUNDARY_SNAPSHOT
        # priming but wrong for RECORDED warmup with its own authored max_tokens).
        envelope["own_output_cap"] = True
    if dispatch_meta.get("disable_cache_bust"):
        envelope["disable_cache_bust"] = True
    return envelope


def _trace_trie_envelopes(
    llm_nodes: dict[str, LlmNode], ordinals: dict[str, int]
) -> list[NodeEnvelope]:
    """Build the per-node ``prompt_segment_ids`` envelopes for one trie trace.

    Used by the streaming :func:`iter_trace_segment_payloads`; the envelope
    bytes are ``orjson.dumps(_trie_envelope(...))`` -- byte-identical to what
    the eager unified builders derive from the same nodes (the byte-equality
    contract). Nodes without a ``prompt_segment_ids`` path are skipped.

    Slot-carrying nodes (assembly ``items`` / ``capture`` in
    ``metadata["trie"]``) are REJECTED loudly: this streaming envelope carries
    neither, so silently continuing would persist a manifest missing the node's
    dynamic slots (the eager :func:`build_unified_trie_store_interned` is the
    only path that persists them today).
    """
    envelopes: list[NodeEnvelope] = []
    for node_id, ordinal in sorted(ordinals.items(), key=lambda kv: kv[1]):
        node = llm_nodes[node_id]
        trie_meta = (node.metadata or {}).get("trie") or {}
        if trie_meta.get("assembly") or trie_meta.get("capture"):
            raise NotImplementedError(
                f"node {node_id!r}: carries trie assembly items/capture, which "
                "the streaming segment-store split does not support; "
                "slot-carrying graphs must build through the eager store path"
            )
        prompt_segment_ids = _prompt_segment_ids(node)
        if prompt_segment_ids is None:
            continue
        envelopes.append(
            NodeEnvelope(
                node_ordinal=ordinal,
                envelope_bytes=orjson.dumps(_trie_envelope(node, prompt_segment_ids)),
            )
        )
    return envelopes


def graph_carries_assembly_slots(parsed: ParsedGraph) -> bool:
    """True when any ``LlmNode`` carries trie assembly items/capture metadata.

    The schedule-plane t*-gate predicate: ``workload_detect._gate_dynamic_slots_vs_tstar``
    is now its only production caller, using it to reject a graph that both
    carries dynamic slots AND engages a t* snapshot window (a slot producer
    chopped into warmup would leave its consumer's pool value undefined). It no
    longer routes the store build: every format takes the in-process
    interned drain regardless of slots (that drain is the only one that persists
    slot envelopes anyway) -- so this is the ONE
    definition of "carries dynamic slots" for the gate, not a store-route fork.

    It still scans the SAME ``trie_meta.get("assembly") or
    trie_meta.get("capture")`` condition the streaming envelope
    (:func:`_trace_trie_envelopes`) raises ``NotImplementedError`` on, so the
    two agree on what "slots" means; but because slot-carrying graphs never take
    the streaming route, that rejection is unreachable armor rather than a
    routing dependency.

    Why the ``capture`` clause is safe for the t*-gate: at graph level
    ``capture`` and ``assembly`` are equivalent. ``capture: true`` is only ever
    stamped on producers referenced by some slot's assembly program by every
    lowering, so a capture-without-assembly node cannot exist in a
    lowered graph. The ``capture`` clause is therefore redundant-but-cheap armor
    here, which is why ``_gate_dynamic_slots_vs_tstar`` can resolve this same
    union predicate without any behavior difference from an assembly-only check.
    """
    graphs = [parsed.graph, *parsed.graphs.values()]
    for graph in graphs:
        for node in graph.nodes.values():
            trie_meta = (node.metadata or {}).get("trie") or {}
            if trie_meta.get("assembly") or trie_meta.get("capture"):
                return True
    return False


def iter_trace_segment_payloads(parsed: ParsedGraph) -> Iterator[TraceSegmentPayload]:
    """Yield per-trace trie payloads (envelopes + segments) for ``parsed``.

    The worker-side counterpart to the eager unified builders: for each trace it
    builds the same ``prompt_segment_ids`` envelopes AND carries the trace's
    pool segments as ``(id, role, content, wire_json)`` tuples so the parent
    (:func:`build_unified_trie_store_from_payloads`) can drain the unified store
    without re-parsing the row. ``parsed.segment_pool`` MUST be present (this is a
    segment trie graph); the pool's content-addressed entries dedup across rows in the consumer.

    Its main use is inside pool workers, for which it is the reference shape of
    the ``TraceSegmentPayload``s those workers ship to
    :func:`build_unified_trie_store_from_payloads`; the one main-process caller
    is ``trace_parallel``'s sub-threshold fallback, which yields payloads from
    an in-parent parse. The corpus-scale eager route drains the whole parse
    through :func:`build_unified_trie_store_interned` instead, never through
    payloads. It is also the streaming-drain oracle for the byte-parity suite
    (``test_dynamo_streaming_store_parity``), which feeds a whole parse through
    it to prove the streamed store equals the interned store byte-for-byte.
    """
    from aiperf.dataset.graph.codecs import encode_parsed_graph_msgpack
    from aiperf.dataset.graph.graph_meta_sidecar import strip_replay_text

    pool = parsed.segment_pool
    segments: list[tuple[str, str, str, str | None]] = (
        [(s.id, s.role, s.content, s.wire_json) for s in pool.by_id.values()]
        if pool is not None
        else []
    )
    # Content-free structural graph for the sidecar: the canonical
    # ``strip_replay_text`` empties ``replay_outputs``, the segment pool (kept
    # non-None so the loaded graph still reads as a segment-store-backed graph), AND
    # every ``LlmNode.prompt`` (the
    # dominant inline-content field for the segment trie). Shipped once (like
    # ``segments``); the streaming consumer merges these across rows and writes the
    # ``graph_meta`` sidecar so the TimingManager skips the whole-corpus re-parse.
    # Topology, edges, arrival offsets, and node metadata (incl.
    # ``prompt_segment_ids`` refs) are preserved verbatim.
    structural_bytes = encode_parsed_graph_msgpack(strip_replay_text(parsed))
    # Segments are shipped once per trace; the consumer dedups by id. A
    # single-trace ParsedGraph (the streaming worker case) carries only its own
    # trace's pool, so this stays memory-bounded.
    for index, trace in enumerate(parsed.all_traces):
        llm_nodes = _trie_llm_nodes(parsed, trace)
        ordinals = trie_node_ordinals(llm_nodes)
        envelopes = _trace_trie_envelopes(llm_nodes, ordinals)
        yield TraceSegmentPayload(
            trace_id=trace.id,
            node_ordinals=ordinals,
            envelopes=envelopes,
            # Attach the pool only to the first trace's payload to avoid shipping
            # the same single-trace pool N times; multi-trace ParsedGraphs are
            # not produced by the streaming worker (one row == one trace).
            segments=segments if index == 0 else [],
            structural_graph=structural_bytes if index == 0 else b"",
        )


async def build_unified_trie_store_from_payloads(
    payloads: Iterable[TraceSegmentPayload],
    store: GraphSegmentUnifiedBackingStore,
    *,
    structural_sink: list[bytes] | None = None,
) -> dict[str, dict[str, int]]:
    """Drain STREAMED trie payloads into the ONE unified store; return the catalog.

    The streaming counterpart to the eager
    :func:`build_unified_trie_store_interned`, and -- after the in-process-drain
    flip -- the worker-pool-only drain (`GraphStoreBuilder._build_graph_store_streaming_trie`
    feeds it the worker-pool payload stream; the in-process route drains the whole
    parse eagerly instead). Each payload's segments are
    ``put_segment``'d into the unified content pool (idempotent dedup on the
    content-addressed id bounds RAM -- the streaming property) and each node's
    ``prompt_segment_ids`` envelope is resolved to int handles and written as
    an interned manifest. This
    is what makes the unified store the ACTUAL store on the corpus-scale
    worker-pool path, not just the eager path -- the worker opens the SAME unified store it
    opens for an eager run. Segments are put before the manifests that reference
    them (within a payload, and earlier payloads' segments are already resident),
    so every handle resolves. Returns ``{trace_id: {node_id: node_ordinal}}``.

    When ``structural_sink`` is provided, each payload's non-empty
    ``structural_graph`` (content-free per-trace topology, msgpack) is appended
    to it as the stream drains. The caller merges these into the corpus
    structural graph and writes the ``graph_meta`` sidecar so the TimingManager
    skips its whole-corpus re-parse. The bytes are content-free (empty pool +
    empty ``replay_outputs``), so accumulating all of them stays bounded.
    """
    catalog: dict[str, dict[str, int]] = {}
    for payload in payloads:
        for segment_id, role, content, wire_json in payload.segments:
            store.put_segment(segment_id, role, content, wire_json=wire_json)
        for rec in payload.envelopes:
            envelope = orjson.loads(rec.envelope_bytes)
            handles: list[int] = []
            for sid in envelope["prompt_segment_ids"]:
                handle = store.segment_handle(sid)
                if handle is None:
                    raise ValueError(
                        f"trace {payload.trace_id} node ordinal "
                        f"{rec.node_ordinal} references segment {sid!r} absent "
                        "from the unified pool"
                    )
                handles.append(handle)
            store.add_node_manifest_interned(
                payload.trace_id,
                rec.node_ordinal,
                handles,
                envelope.get("dispatch_overrides", {}),
                bool(envelope.get("stream", False)),
                extra_headers=envelope.get("extra_headers"),
                endpoint_extra_applied=bool(envelope.get("endpoint_extra_applied")),
                own_output_cap=bool(envelope.get("own_output_cap")),
            )
        catalog[payload.trace_id] = payload.node_ordinals
        if structural_sink is not None and payload.structural_graph:
            structural_sink.append(payload.structural_graph)
    store.finalize_sync()
    return catalog


def _resolve_assembly_items(
    assembly: list[dict] | None,
    catalog_key: str,
    ordinals: dict[str, int],
    handle_for,
) -> list[dict] | None:
    """Resolve a lowering assembly program to store-addressed ``items``.

    The lowering stamps tokens with hex segment ids and producer NODE IDS
    (``metadata["trie"]["assembly"]``); the persisted
    envelope carries int handles and node ORDINALS — the worker's native keys.
    A producer absent from the trace's ordinal map is build-time corruption.
    """
    if not assembly:
        return None

    def _ordinal_for(node_id: str) -> int:
        ordinal = ordinals.get(node_id)
        if ordinal is None:
            raise ValueError(
                f"trie node {catalog_key} slot references producer "
                f"{node_id!r} absent from the trace's ordinal map"
            )
        return ordinal

    items: list[dict] = []
    for token in assembly:
        if "seg" in token:
            items.append({"h": handle_for(token["seg"])})
        elif "s" in token:
            items.append({"s": {"src": _ordinal_for(token["s"]["src"])}})
        elif "m" in token:
            parts = [
                {"sv": _ordinal_for(part["sv"])} if "sv" in part else part
                for part in token["m"]["parts"]
            ]
            items.append({"m": {"role": token["m"]["role"], "parts": parts}})
        else:
            raise ValueError(
                f"trie node {catalog_key} assembly carries unknown token {token!r}"
            )
    return items


async def build_unified_trie_store_interned(
    parsed: ParsedGraph, store: GraphSegmentUnifiedBackingStore
) -> dict[str, dict[str, int]]:
    """Drain an eager trie ``ParsedGraph`` into ONE interned (A2) unified store:
    assign int handles during pool drain, then write each node's manifest as a
    handle path.

    Every pool ``Segment`` is written via :meth:`put_segment`, which assigns a
    dense insertion-index handle (the ``i``-th segment drained gets handle ``i``).
    Each trie ``LlmNode``'s hex ``prompt_segment_ids`` path is then resolved to
    those int handles via :meth:`segment_handle`, and the profiling manifest is
    written as a handle path via :meth:`add_node_manifest_interned`. The
    hex->handle map lives ONLY in ``store`` (build-time): workers read the handle
    path directly and never see hex ids. A node referencing a segment absent from
    the pool is a build-time corruption and raises ``ValueError``.

    Node set + ordinal source: iterates :func:`flat_trie_ordinals` so it
    covers the top graph, keying each
    manifest by the SINGLE monotonic ordinal that helper assigns (the SAME ordinal
    the schedule plane resolves). This
    reduces to :func:`trie_node_ordinals` over the top graph, so manifests are
    keyed by bare-id ordinals exactly as before. Returns the
    ``{trace_id: {node_id: node_ordinal}}`` catalog keyed by bare node id (the graph
    is flat).
    """

    pool_segments = parsed.segment_pool.by_id
    for segment in pool_segments.values():
        store.put_segment(
            segment.id, segment.role, segment.content, wire_json=segment.wire_json
        )

    catalog: dict[str, dict[str, int]] = {}
    for trace in parsed.all_traces:
        nodes = _trie_llm_nodes(parsed, trace)
        ordinals = flat_trie_ordinals(parsed, trace)
        for catalog_key, ordinal in ordinals.items():
            node = nodes[catalog_key]
            prompt_segment_ids = _prompt_segment_ids(node)
            if prompt_segment_ids is None:
                continue

            def _handle_for(sid: str, catalog_key: str = catalog_key) -> int:
                handle = store.segment_handle(sid)
                if handle is None:
                    raise ValueError(
                        f"trie node {catalog_key} references segment {sid!r} "
                        "absent from pool"
                    )
                return handle

            handles = [_handle_for(sid) for sid in prompt_segment_ids]
            trie_meta = (node.metadata or {}).get("trie") or {}
            items = _resolve_assembly_items(
                trie_meta.get("assembly"), catalog_key, ordinals, _handle_for
            )
            envelope = _trie_envelope(node, prompt_segment_ids)
            store.add_node_manifest_interned(
                trace.id,
                ordinal,
                handles,
                envelope.get("dispatch_overrides") or {},
                bool(envelope.get("stream", False)),
                items=items,
                capture=bool(trie_meta.get("capture")),
                extra_headers=envelope.get("extra_headers"),
                endpoint_extra_applied=bool(envelope.get("endpoint_extra_applied")),
                own_output_cap=bool(envelope.get("own_output_cap")),
            )
        catalog[trace.id] = ordinals
    await store.finalize()
    return catalog


__all__ = [
    "TraceSegmentPayload",
    "build_unified_trie_store_from_payloads",
    "build_unified_trie_store_interned",
    "flat_trie_ordinals",
    "graph_carries_assembly_slots",
    "iter_trace_segment_payloads",
    "trie_node_ordinals",
]
