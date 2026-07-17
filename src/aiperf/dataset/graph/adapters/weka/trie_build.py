# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trivial dependency-only ``ParsedGraph`` builder for a single Weka trace.

:func:`build_trie_graph` is the simplest possible IR realization of a
:class:`WekaTrace`: one :class:`LlmNode` per recorded ``n``/``s`` request and
plain :class:`StaticEdge` "waits-for" dependency edges between them. There are
NO reducers / channels / subgraph / spawn / await primitives and NO
chain-detection / ``::fa``-``::aux`` classification -- the conversation
structure and the dependency structure are derived purely from the recorded
hash-id prefix tree and the recorded spawn + timing facts.

The trie machinery itself (content-parent resolution, frozen block tags,
message assembly, covered-count ISL gate, interval-order edges) is the shared
format-agnostic core in
:mod:`~aiperf.dataset.graph.segment_ir.trie_content`
(:func:`~aiperf.dataset.graph.segment_ir.trie_content.build_trie_ir`).
This module only flattens the recorded weka requests into normalized
:class:`~aiperf.dataset.graph.segment_ir.trie_content.TrieNode`s and
assembles the resulting per-node artifacts into weka ``LlmNode``s:

1. **Conversation structure = hash-id prefix tree.** Every request (top-level
   ``n``/``s`` AND subagent-inner ``n``/``s``, recursively) is collected in
   recorded ``t`` order; a request's CONTENT-PARENT is its longest full-prefix
   (else branch-point) predecessor
   (:func:`~aiperf.dataset.graph.segment_ir.trie_content.resolve_content_parents`).

2. **Segments (message units).** Each covered block gets a frozen
   ``(role, starts_new_message)`` tag
   (:func:`~aiperf.dataset.graph.segment_ir.trie_content.assign_block_tags`);
   :func:`~aiperf.dataset.graph.segment_ir.trie_content.assemble_messages`
   groups those tags into messages and emits one content-addressed
   :class:`SegmentPool` entry per message, chained root->tip. Because the tags
   are frozen per trie position, a shared block prefix yields identical message
   ids -- a real KV-cache prefix. The node's own assistant output is appended
   as one ``"assistant"`` segment sized to the recorded ``out`` so
   the successor prompt bytes stay block-exact.

3. **Dependency edges (interval order).** R's incoming edges come from
   :func:`~aiperf.dataset.graph.segment_ir.interval_order.build_interval_edges`: a candidate ``A`` yields an edge into R iff
   ``A`` FINISHED before R STARTED (raw ``A.t + A.api_time <= R.t``) AND
   ``rank(A) < rank(R)``, after async exclusion and a frontier (transitive-
   reduction) filter. The latest-ending frontier predecessor carries the warped
   end-to-start delay; every other is an AND-fan-in wait (delay 0). With NO
   finished-before cause R is wired ``StaticEdge(START, R)`` with
   ``min_start_delay_us = R.start * 1e6`` -- UNLESS R's recorded causal parent
   (spawner / chain-prev, stamped in :func:`_walk`) was still IN FLIGHT at R's
   start, in which case
   :func:`~aiperf.dataset.graph.segment_ir.interval_order.apply_start_anchors`
   collapses R's incoming edges to one start-anchored ``StaticEdge(parent, R)``
   (``delay_after_predecessor_start_us`` = warped start-to-start gap) so recorded
   mid-flight concurrency tracks the parent's dispatch instead of freezing to the
   recorded wall clock.

4. **Concurrency is emergent.** Two requests sharing a cause with no edge
   between them stay edge-free; the builder adds an inter-sibling edge ONLY when
   the start-anchor pass wires an in-flight causal parent (point 3).
"""

from __future__ import annotations

from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.dataset.graph.adapters.shared.output_cap import wire_output_cap
from aiperf.dataset.graph.adapters.weka.trace_models import (
    EmptyWekaTraceError,
    WekaSubagentEntry,
    WekaTrace,
    WekaTraceAdapterError,
)
from aiperf.dataset.graph.models import (
    ExpectedTokens,
    LlmNode,
    ParsedGraph,
    ProvenanceSpec,
)
from aiperf.dataset.graph.segment_ir.envelope import stamp_prompt_segment_ids
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph.segment_ir.trie_content import (
    ReconCallbacks,
    TrieNode,
    TrieNodeBuild,
    TrieRequest,
    assemble_trie_graph,
    build_trie_ir,
)


def build_trie_graph(
    trace: WekaTrace,
    *,
    callbacks: ReconCallbacks | None = None,
    tokenizer_name: str = "builtin",
    prompt_corpus: str = "coding",
    root_seed: int | None = None,
    idle_gap_cap_seconds: float | None = None,
    max_osl: int | None = None,
) -> tuple[ParsedGraph, SegmentPool]:
    """Build a dependency-only :class:`ParsedGraph` + :class:`SegmentPool`.

    ``callbacks`` supplies the reconstructor's block-decode / partial-tail /
    decode-to-text functions. When ``None`` (production), a
    :class:`~aiperf.dataset.graph.adapters.shared.content.CorpusContentSynthesizer`
    is built from ``(tokenizer_name, prompt_corpus, root_seed)`` and its
    byte-faithful callbacks are used, decoding each hash block to the trace's
    own ``block_size`` tokens. Unit tests pass deterministic stubs.

    ``idle_gap_cap_seconds`` caps inter-request-start idle gaps: every recorded
    node start ``request.t`` is mapped through an
    :class:`~aiperf.dataset.graph.segment_ir.trie_content.ActiveIdleWarp`
    so node arrival offsets, edge ``delay_after_predecessor_us``, and root
    ``min_start_delay_us`` all sit on the SAME warped clock. When ``None``
    (warp disabled) the raw ``request.t`` timeline is used unchanged. Without
    the cap a recorded multi-hour idle gap survives into the warmup phase and
    parks it forever; the cap compresses each over-long gap to ``cap_seconds``.

    ``max_osl`` (``--synthesis-max-osl``) caps each TOP-LEVEL chain request's
    native ``LlmNode.max_tokens`` to ``min(recorded out, max_osl)``
    (agentx ``_cap_output`` parity); subagent-body requests are left uncapped.
    ``None`` leaves the recorded ``out`` uncapped everywhere. The cap touches
    dispatch only -- synthesized history content stays sized to the recorded
    ``out`` so successor prompt bytes (and ISL) are unchanged.

    The returned :class:`ParsedGraph` carries the trace's nodes + edges on its
    single top-level ``graph``; ``graphs`` stays empty (a single top-level
    graph, no per-trace graph variants in this trivial IR).

    Raises:
        EmptyWekaTraceError: the recorded ``requests`` flatten to zero
            normal/streaming leaves (e.g. only subagent markers with empty
            bodies) -- such a trace could never fire a node.
    """
    if callbacks is None:
        callbacks = _default_callbacks(
            tokenizer_name,
            prompt_corpus,
            root_seed,
            trace_id=trace.id,
            block_size=trace.block_size,
            hash_scope=trace.hash_id_scope,
        )

    nodes = _flatten_requests(trace.requests, root_scope=trace.id)
    if not nodes:
        raise EmptyWekaTraceError(
            f"trace {trace.id!r}: requests flatten to zero normal/streaming leaf "
            "requests (only subagent markers with empty bodies) -- the trace "
            "could never fire a node"
        )
    top_level_ids = _top_level_leaf_ids(trace.requests, trace.id)
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=trace.block_size,
        callbacks=callbacks,
        pool=pool,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        # A recorded request with zero covered blocks (in < block_size,
        # hash_ids []) still carried a real prompt: synthesize the sampled
        # sub-block user message (dynamo parity) instead of the legacy empty
        # prompt, which produced an unreplayable empty messages array.
        small_prompt_fallback=True,
    )

    def build_node(node: TrieNode) -> LlmNode:
        return _build_llm_node(
            node,
            build=result.builds[node.node_id],
            max_osl=max_osl if node.node_id in top_level_ids else None,
        )

    graph = assemble_trie_graph(
        nodes,
        result,
        build_node=build_node,
        provenance=ProvenanceSpec(source="weka_trace", tool="aiperf-weka-trie/1"),
    )
    return ParsedGraph(graph=graph), pool


# --- flattening -----------------------------------------------------------


def _guard_scope(scope: str, seen: set[str]) -> None:
    """Fail loud on scope ids the ``{scope}:{turn}`` grammar cannot carry.

    ``:`` is the ONE reserved character of the identity grammar (node ids are
    ``{scope}:{turn}``; runtime ids append ``::{nonce}``), and a duplicate
    scope (a repeated ``agent_id``, or an ``agent_id`` equal to the trace id)
    would merge two trajectories' node namespaces.
    """
    if not scope or ":" in scope:
        raise WekaTraceAdapterError(
            f"scope id {scope!r} cannot form node ids: scope ids must be "
            "non-empty and contain no ':' (the reserved identity separator)"
        )
    if scope in seen:
        raise WekaTraceAdapterError(
            f"duplicate trajectory scope {scope!r}: agent_ids must be unique "
            "within a trace and distinct from the trace id"
        )
    seen.add(scope)


def _flatten_requests(requests: list, *, root_scope: str) -> list[TrieNode]:
    """Collect every recorded leaf request in recorded ``t`` order.

    Recurses into subagent markers (and nested markers). Node ids are
    ``{scope}:{turn}`` -- the trajectory scope (the trace id for the root
    chain, the recorded ``agent_id`` for each subagent) plus the 0-based turn
    within that scope. Pure recorded data: identical logical traces produce
    identical node ids regardless of file layout, and a node id IS the legacy
    ``(conversation, turn_index)`` coordinate of its trajectory.
    """
    out: list[TrieNode] = []
    seen: set[str] = set()
    _guard_scope(root_scope, seen)
    _walk(
        requests,
        out,
        scope=root_scope,
        scopes_seen=seen,
        async_ancestors=frozenset(),
    )
    return out


def _top_level_leaf_ids(requests: list, root_scope: str) -> frozenset[str]:
    """Node ids of the leaves sitting DIRECTLY in the trace's top-level list.

    Mirrors :func:`_walk`'s id assignment for the root scope: the k-th
    top-level leaf is ``{root_scope}:{k}``; anything under a subagent marker
    lives in that agent's own scope. Consumed by the ``max_osl`` cap, which
    applies to top-level chain requests only (agentx ``_cap_output`` parity).
    """
    return frozenset(
        f"{root_scope}:{k}"
        for k, req in enumerate(
            req for req in requests if not isinstance(req, WekaSubagentEntry)
        )
    )


_ASYNC_LAUNCHED_STATUS = "async_launched"


def _walk(
    requests: list,
    out: list[TrieNode],
    *,
    scope: str,
    scopes_seen: set[str],
    async_ancestors: frozenset[str],
    inherited_causal: str | None = None,
) -> None:
    """Depth-first walk appending leaves to ``out`` in recorded order.

    Recurses into subagent markers, threading each ``async_launched`` subtree's
    id into ``async_ancestors`` so the interval-order edge builder can exclude
    fire-and-forget children (:func:`~aiperf.dataset.graph.segment_ir.interval_order.build_interval_edges`). Dependency structure is
    derived globally afterward from the recorded finished-before intervals and
    ranks, not from the walk order (see :func:`~aiperf.dataset.graph.segment_ir.interval_order.build_interval_edges`).

    Each leaf's ``causal_parent_id`` is stamped here: the previous n/s leaf in
    this same list (chain-prev), else the nearest preceding n/s leaf in an
    enclosing list (``inherited_causal`` -- the spawner), else ``None``. A
    subagent marker inherits the current ``prev_leaf_id`` (the leaf that spawned
    it) as its subtree's ``inherited_causal``, falling back to this list's own
    ``inherited_causal`` when no preceding leaf exists. This feeds
    :func:`~aiperf.dataset.graph.segment_ir.interval_order.apply_start_anchors`
    (wired inside ``build_trie_ir``): when the causal parent is still in flight
    at a node's recorded start, its incoming edges collapse to one start-anchored
    edge -- recorded mid-flight concurrency tracks the parent causally instead of
    freezing to the recorded wall clock.
    """
    prev_leaf_id: str | None = None
    turn = 0
    for req in requests:
        if isinstance(req, WekaSubagentEntry):
            child_scope = req.agent_id
            _guard_scope(child_scope, scopes_seen)
            child_async = (
                async_ancestors | {child_scope}
                if req.status == _ASYNC_LAUNCHED_STATUS
                else async_ancestors
            )
            _walk(
                req.requests,
                out,
                scope=child_scope,
                scopes_seen=scopes_seen,
                async_ancestors=child_async,
                inherited_causal=(
                    prev_leaf_id if prev_leaf_id is not None else inherited_causal
                ),
            )
        else:
            node_id = f"{scope}:{turn}"
            turn += 1
            out.append(
                TrieNode(
                    node_id=node_id,
                    request=TrieRequest(
                        hash_ids=list(req.hash_ids),
                        input_length=req.input_length,
                        output_length=req.output_length,
                        t=req.t,
                        api_time=req.api_time or 0.0,
                        model=req.model,
                        streaming=req.type == "s",
                        ttft=req.ttft if req.type == "s" else None,
                    ),
                    order=len(out),
                    async_ancestors=async_ancestors,
                    causal_parent_id=(
                        prev_leaf_id if prev_leaf_id is not None else inherited_causal
                    ),
                )
            )
            prev_leaf_id = node_id


# --- node construction ------------------------------------------------------


def _build_llm_node(
    node: TrieNode,
    *,
    build: TrieNodeBuild,
    max_osl: int | None = None,
) -> LlmNode:
    """Assemble ``node``'s prompt-segment path + assistant output into the LlmNode.

    The node carries NO inline prompt (``prompt=[]``): on the trie route the
    prompt content lives ONLY in the run's ``SegmentPool``, addressed by
    ``metadata["trie"]["prompt_segment_ids"]`` (stamped below). The store build,
    ``graph_meta`` sidecar (``strip_replay_text`` forces ``prompt=[]``), and the
    worker all materialize from the segment pool / mmap store -- none reads
    ``node.prompt`` -- matching the dynamo and dag_jsonl adapters' convention.
    For in-process debugging, recover the content with
    ``segment_pool.materialize(read_prompt_segment_ids(node))``.

    The prompt path comes from
    :func:`~aiperf.dataset.graph.segment_ir.trie_content.assemble_messages`,
    which groups the node's frozen per-block ``(role, starts_new_message)`` tags
    into messages and emits one content-addressed pool entry per message. Because
    the tags are frozen per trie position, a shared block prefix produces
    byte-identical message ids -- a real KV-cache prefix -- with NO relabeling.
    The recorded ``out`` is emitted as one trailing ``"assistant"`` segment
    parented at the prompt tip.

    ``max_osl`` (already resolved to this node: the caller passes ``None`` for
    subagent-body nodes) caps the wire generation cap (the native
    ``LlmNode.max_tokens``, endpoint-mapped by the
    worker) to ``min(recorded out, max_osl)``; a recorded ``out`` of 0
    upgrades to 1 with a warning (:func:`wire_output_cap`). The response
    SEGMENT stays sized to the recorded ``out`` so successor prompt content is
    unchanged.
    """
    req = node.request
    max_tokens = (
        req.output_length if max_osl is None else min(req.output_length, max_osl)
    )
    llm = LlmNode(
        prompt=[],
        output=f"{node.node_id}_out",
        streaming=req.streaming,
        model=req.model,
        max_tokens=wire_output_cap(max_tokens, node_id=node.node_id),
        arrival_offset_us=int(round(node.start * MICROS_PER_SECOND)),
        # Recorded token expectations (dynamo parity): the RECORDED lengths,
        # not the capped wire value. Cache fields stay None -- weka records no
        # engine cache counts, and the loader's theoretical prefix-cache
        # prediction is a model, not a recording.
        expected=ExpectedTokens(
            input_tokens=req.input_length,
            output_tokens=req.output_length,
        ),
    )
    # ONLY prompt_segment_ids is stamped (dynamo parity): the build-synthesis
    # response_id / hash_ids companions reached no build, sidecar, store, or
    # dispatch consumer, and the per-node hash-list copy was pure build-plane
    # RAM at corpus scale.
    return stamp_prompt_segment_ids(llm, build.prompt_path)


# --- default production callbacks -----------------------------------------


def _default_callbacks(
    tokenizer_name: str,
    prompt_corpus: str,
    root_seed: int | None,
    trace_id: str | None = None,
    *,
    block_size: int = 64,
    hash_scope: str = "local",
) -> ReconCallbacks:
    """Build byte-faithful reconstructor callbacks from a real synthesizer.

    Imported lazily so the heavy corpus build stays off the import path for
    callers (and tests) that inject their own stub callbacks.

    ``hash_scope`` is the trace's declared ``hash_id_scope`` and picks the
    block-decode namespace. Under ``"local"``, ``trace_id`` scopes the hash-id
    namespace per trace: block decode reseeds per ``(trace_id, hash_id)``, so
    equal hash ids across traces synthesize DIFFERENT bytes (no manufactured
    cross-trace KV-cache sharing). Under ``"global"``, block decode reseeds per
    bare ``hash_id`` -- the SAME namespace dynamo's content-global recorded
    hashes use -- so equal hash ids across traces synthesize byte-identical
    blocks, reproducing recorded cross-trace KV-cache sharing. Sharing needs no
    cross-file coordination: the reseed is a pure function of ``(root seed,
    hash_id)``, so independently-parsed traces (including parallel pool
    workers) decode identical bytes for equal ids.

    Partial-tail seeds -- which the shared trie driver keys only by node id
    (``"{node_id}:response"`` / ``"{node_id}:tiny"``) -- get the trace id
    prefixed under BOTH scopes: trajectory-local node ids recur across traces and
    are a driver artifact, not part of the recorded hash namespace, so leaving
    them unscoped would fabricate identical response bytes corpus-wide.

    ``block_size`` is the TRACE's recorded block size: each hash id decodes to
    exactly that many corpus tokens (mirrors ``dynamo_recon_callbacks``). The
    synthesizer's own default is weka-legacy 64; threading the trace value keeps
    a ``block_size != 64`` trace's covered-count ISL intact. Decodes always use
    a PRIVATE per-build cache: the synthesizer instance (and its shared cache)
    is reused across traces and block sizes, and mixing block sizes in the
    shared bare-hash-id cache would hand wrong-sized blocks across builds.
    """
    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer(
        tokenizer_name, prompt_corpus=prompt_corpus, root_seed=root_seed
    )
    trace_cache: dict[int, list[int]] = {}

    decode_trace_id = None if hash_scope == "global" else trace_id

    def _decode_scoped(hash_ids: list[int]) -> list[int]:
        return synth._decode_block_tokens(
            hash_ids, block_size=block_size, cache=trace_cache, trace_id=decode_trace_id
        )

    def _tail_scoped(n_tokens: int, seed: str) -> list[int]:
        return synth._sample_partial_tail_tokens(n_tokens, f"{trace_id}:{seed}")

    return ReconCallbacks(
        decode_block_tokens=_decode_scoped,
        sample_partial_tail_tokens=_tail_scoped,
        decode_tokens_to_text=synth._decode_tokens_to_text,
    )


__all__ = [
    "ReconCallbacks",
    "build_trie_graph",
]
