# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize dynamo trace chains into the shared segment-trie IR shape.

One ``TrieNode`` per ``request_end``; the hash source is the recorded
``input_sequence_hashes`` when a record carries replay metadata, else
per-session VIRTUAL negative ids (tagged ``virtual-hash-fallback``) -- the
same recorded-when-present rule as the weka path. Content sharing, frozen
roles, edges, and the segment pool are all owned by
:func:`segment_ir.trie_content.build_trie_ir` -- this module only maps dynamo
record fields onto :class:`TrieRequest` and assembles the dynamo-flavored
``LlmNode``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    DynamoTraceAdapterError,
)
from aiperf.dataset.graph.adapters.shared.output_cap import wire_output_cap
from aiperf.dataset.graph.models import ExpectedTokens, LlmNode
from aiperf.dataset.graph.segment_ir.envelope import stamp_prompt_segment_ids
from aiperf.dataset.graph.segment_ir.trie_content import (
    ReconCallbacks,
    TrieNode,
    TrieNodeBuild,
    TrieRequest,
)

if TYPE_CHECKING:
    from aiperf.dataset.graph.adapters.dynamo.trace import _Chain, _Turn

DEFAULT_VIRTUAL_BLOCK_SIZE = 16
_VIRTUAL_HASH_FALLBACK_TAG = "virtual-hash-fallback"


class DynamoISLMismatchError(ValueError):
    """Raised when a recorded ``input_length`` disagrees with its replay hashes.

    Dynamo records ``input_length`` (the full prompt token count) alongside
    ``input_sequence_hashes`` (full-block hashes plus one partial tail over that
    same prompt). A block-consistent record satisfies
    ``(n_hashes - 1) * block_size < input_length <= n_hashes * block_size``; a
    divergence means the trace metadata is internally inconsistent and no
    reconstruction can honor both fields -- the parse aborts fail-loud rather
    than materialize a wrong-sized prompt.
    """


def _guard_session_scopes(session_ids: Iterable[str]) -> None:
    """Fail loud on session ids the ``{scope}:{turn}`` grammar cannot carry.

    Node ids are ``{session_id}:{k}`` with the RECORDED session id verbatim
    (no hashing -- the id is the data), so ``:`` -- the one reserved identity
    separator (runtime ids append ``::{nonce}``) -- must not appear inside a
    recorded session id.
    """
    for sid in session_ids:
        if not sid or ":" in sid:
            raise ValueError(
                f"session id {sid!r} cannot form node ids: session ids must "
                "be non-empty and contain no ':' (the reserved identity "
                "separator)"
            )


def _turn_times_ms(turn: _Turn) -> tuple[int, int]:
    """(start_ms, total_ms) mirroring upstream agent_trace_to_mooncake."""
    rec = turn.record
    req = rec.request
    event = rec.event_time_unix_ms
    received = req.request_received_ms if req is not None else None
    total_f = req.total_time_ms if req is not None else None
    if total_f is not None:
        total = max(0, round(total_f))
    elif received is not None:
        total = max(0, event - received)
    else:
        total = 0
    start = received if received is not None else event - total
    return start, total


def _assert_block_aligned(
    node_id: str, hashes: list[int], input_length: int, bs: int
) -> None:
    """Fail-loud gate: ``input_length`` must be spanned by its replay hashes.

    Dynamo generates full-block hashes plus one partial tail
    (``request_trace/replay.rs``), so a block-consistent record satisfies
    ``(n-1)*bs < input_length <= n*bs``.
    """
    n = len(hashes)
    lo, hi = (n - 1) * bs, n * bs
    if not (lo < input_length <= hi):
        raise DynamoISLMismatchError(
            f"dynamo node {node_id!r}: recorded input_length={input_length} is not "
            f"block-aligned to its {n} replay hashes at block_size={bs} "
            f"(expected {lo} < input_length <= {hi})."
        )


def _resolve_block_size(chains: dict[str, _Chain]) -> int | None:
    """Single recorded ``trace_block_size`` across all replay turns; fail-loud on mix."""
    block_size: int | None = None
    for chain in chains.values():
        for turn in chain.turns:
            req = turn.record.request
            if req is None or req.replay is None:
                continue
            tbs = req.replay.trace_block_size
            if block_size is None:
                block_size = tbs
            elif tbs != block_size:
                raise DynamoTraceAdapterError(
                    f"mixed replay trace_block_size values are not "
                    f"supported: {block_size} and {tbs}"
                )
    return block_size


def dynamo_trie_nodes(
    chains: dict[str, _Chain],
    *,
    release_replay: bool = False,
    block_size: int | None = None,
) -> tuple[list[TrieNode], int, list[str]]:
    """Flatten every session's turns into shared TrieNodes.

    Returns ``(nodes, block_size, extra_trace_tags)``. Nodes are ``{session_id}:{k}``
    (recorded session id verbatim + 0-based turn), ordered by
    recorded start time (ties by ``(session_id, turn_index)``) -- the
    deterministic ``order`` the content-parent trie resolves in.
    ``block_size`` is the single recorded ``trace_block_size`` when any record
    carries replay metadata, else :data:`DEFAULT_VIRTUAL_BLOCK_SIZE`.

    ``block_size`` (keyword) pins the block size instead of resolving it from
    these chains. The per-session-tree build (:func:`from_dynamo_trace`)
    resolves ONE size across the WHOLE capture (fail-loud on a mix) and pins it
    into every tree so a tree that happens to carry no replay never silently
    drops to the virtual default while a sibling tree carries a recorded size --
    keeping the flat graph byte-identical to a single global build. ``None``
    (every other caller) resolves it from ``chains`` exactly as before.

    Recorded hash objects arrive already interned to canonical ints from
    :func:`~aiperf.dataset.graph.adapters.dynamo.trace._collect_records`, so
    the recorded branch copies them with a plain ``list()`` (not ``int(h)``),
    preserving that shared identity into every ``TrieRequest.hash_ids``.

    ``release_replay`` (opt-in) nulls each record's ``request.replay`` right
    after its ``input_sequence_hashes`` / ``input_length`` have been copied into
    the emitted :class:`TrieRequest`, freeing the pydantic replay hash lists the
    ``chains`` otherwise pin for the whole build. It is OFF by default because
    the ``recorded is None`` branch below is the LEGITIMATE virtual-hash
    fallback, not an error: re-lowering the SAME in-memory ``chains`` after a
    release would silently degrade every previously-recorded turn to negative
    virtual ids (a DIFFERENT, wrong trie), so only a caller that lowers a fresh
    ``chains`` exactly once (the production store build, via
    :meth:`DynamoTraceAdapter.parse`) may opt in. :func:`_resolve_block_size`
    has already consumed every replay before this loop runs.
    """
    _guard_session_scopes(chains)
    if block_size is None:
        block_size = _resolve_block_size(chains)
    bs = block_size or DEFAULT_VIRTUAL_BLOCK_SIZE

    # Flatten turns with timing, sorted globally by recorded start. The
    # tie-break is the NUMERIC (sid, k) pair, never the node-id string:
    # ":10" sorts before ":2" lexicographically, which would misorder
    # same-millisecond turns past index 9 (order is the trie's ground truth).
    flat: list[
        tuple[int, str, int, int, _Chain, _Turn]
    ] = []  # (start_ms, sid, k, total_ms, ...)
    for sid, chain in sorted(chains.items()):
        for k, turn in enumerate(chain.turns):
            start_ms, total_ms = _turn_times_ms(turn)
            flat.append((start_ms, sid, k, total_ms, chain, turn))
    flat.sort(key=lambda item: (item[0], item[1], item[2]))
    t0 = flat[0][0] if flat else 0

    # Per-session (start_ms, node_id) in start order, for causal-parent
    # resolution of subagent first turns. flat is globally start-sorted, so
    # per-session sublists inherit that order.
    session_turns: dict[str, list[tuple[int, str]]] = {}
    for start_ms, sid, k, _total, chain, _turn in flat:
        session_turns.setdefault(chain.session_id, []).append((start_ms, f"{sid}:{k}"))

    virtual = itertools.count(-1, -1)
    virtual_prev: dict[str, list[int]] = {}
    tags: set[str] = set()
    nodes: list[TrieNode] = []

    for order, (start_ms, sid, k, total_ms, chain, turn) in enumerate(flat):
        node_id = f"{sid}:{k}"
        req = turn.record.request
        input_tokens = int(req.input_tokens) if req and req.input_tokens else 1
        output_tokens = int(req.output_tokens) if req and req.output_tokens else 0

        recorded = req.replay if req is not None else None
        if recorded is not None:
            # ``list()`` (not ``[int(h) for h in ...]``): the hash objects arrive
            # already interned to canonical ints from ``_collect_records``, so a
            # plain copy preserves that shared identity (and drops ~H int() C-calls
            # per parse). Value-identical -- pydantic guarantees exact ints post-
            # validation -- and it still copies, so the release_replay / virtual_prev
            # aliasing below is unchanged.
            hashes = list(recorded.input_sequence_hashes)
            input_length = int(recorded.input_length)
            _assert_block_aligned(node_id, hashes, input_length, bs)
            # A non-block-aligned input records ONE trailing partial-tail
            # hash. Drop it from the lowered hash list: engines cache/share
            # FULL blocks only (a partial block is never a prefix-cache hit),
            # its tail content is seed-sampled rather than decoded, and
            # keeping it would skew block-tag segmentation and theoretical
            # block totals against the same recording in other formats.
            # ``input_length`` still carries the tail tokens for the sampled
            # sub-block remainder.
            if input_length < len(hashes) * bs:
                hashes.pop()
            virtual_prev[chain.session_id] = hashes
            if release_replay:
                # hashes/input_length are now copied into the TrieRequest below;
                # free the recorded replay list (mutable BaseModel) so the whole
                # capture's hash lists are not pinned for the rest of the build.
                req.replay = None
        else:
            tags.add(_VIRTUAL_HASH_FALLBACK_TAG)
            input_length = input_tokens
            prev = virtual_prev.get(chain.session_id, [])
            m = input_length // bs
            hashes = (
                prev[:m]
                if m <= len(prev)
                else prev + [next(virtual) for _ in range(m - len(prev))]
            )
            virtual_prev[chain.session_id] = hashes

        if k > 0:
            causal = f"{sid}:{k - 1}"
        elif chain.parent_session_id is not None:
            causal = None
            for p_start, pid in session_turns.get(chain.parent_session_id, []):
                if p_start <= start_ms:
                    causal = pid
                else:
                    break
        else:
            causal = None

        node = TrieNode(
            node_id=node_id,
            request=TrieRequest(
                hash_ids=hashes,
                input_length=input_length,
                output_length=output_tokens,
                t=(start_ms - t0) / 1000.0,
                api_time=total_ms / 1000.0,
                model=req.model if req is not None else None,
                ttft=(
                    (req.ttft_ms / 1000.0)
                    if (req is not None and req.ttft_ms is not None)
                    else None
                ),
            ),
            order=order,
            causal_parent_id=causal,
        )
        node.dynamo_meta = _node_meta(chain, turn, k=k)
        nodes.append(node)

    return nodes, bs, sorted(tags)


def _node_meta(chain: _Chain, turn: _Turn, *, k: int) -> dict[str, Any]:
    """Per-node dynamo payload: session headers, identity breadcrumbs, expected tokens."""
    rec = turn.record
    req = rec.request
    n = len(chain.turns)

    # Session identity rides in HTTP HEADERS, never the body: dynamo's NvExt
    # is deny_unknown_fields and explicitly rejects body-level agent_context
    # (extensions.rs::nvext_agent_context_is_rejected); no session_control
    # object exists in the protocol. The frontend ingests x-dynamo-session-id
    # / x-dynamo-parent-session-id on every request and x-dynamo-session-final
    # on the session's LAST request (protocols/agents.rs), which drives
    # session-affinity routing and end-of-session KV eviction. A single-turn
    # session (n == 1, k == 0) stamps BOTH the session id and session-final on
    # its only request, so the session it opens is also evicted. The RECORDED
    # ids are stamped verbatim here (build time knows nothing of replay
    # concurrency); the worker suffixes the two identity headers per replay
    # instance at dispatch (worker_materialize.uniquify_dynamo_session_headers)
    # so concurrent instances of one trace never share a server session.
    extra_headers: dict[str, str] = {"x-dynamo-session-id": chain.session_id}
    if chain.parent_session_id is not None:
        extra_headers["x-dynamo-parent-session-id"] = chain.parent_session_id
    if k == n - 1:
        extra_headers["x-dynamo-session-final"] = "true"

    # No recorded-scalar round-trip is stamped: everything metadata used to
    # duplicate lives on native fields (model / max_tokens / expected) or in
    # the capture file itself, and node metadata survives the graph_meta
    # sidecar strip (which clears only prompt and metadata["trie"]), so every
    # extra key here bloats the content-free structural plane at corpus scale.
    return {
        "extra_headers": extra_headers,
        "session_id": chain.session_id,
        "parent_session_id": chain.parent_session_id,
        "turn_index": k,
        "expected": ExpectedTokens(
            input_tokens=req.input_tokens if req else None,
            output_tokens=req.output_tokens if req else None,
            cache_read_tokens=req.cached_tokens if req else None,
            cache_creation_tokens=None,
        ),
    }


def dynamo_recon_callbacks(
    tokenizer_name: str,
    prompt_corpus: str,
    root_seed: int | None,
    *,
    block_size: int,
    trace_scope: str,
) -> ReconCallbacks:
    """Byte-faithful reconstructor callbacks at the DYNAMO trie block size.

    Reuses the shared (cached) ``CorpusContentSynthesizer`` for the corpus /
    tokenizer / partial-tail machinery, but decodes each hash to ``block_size``
    tokens instead of the synthesizer's weka-fixed 64: dynamo records blocks
    at the model's KV cache block size (default 16, backend-overridable; or
    :data:`DEFAULT_VIRTUAL_BLOCK_SIZE` for virtual fallback ids), and the
    covered-count ISL contract requires exactly ``block_size`` tokens per
    covered block. A PRIVATE per-parse cache is used because the synthesizer
    instance is shared across adapters within a process and its ``pg._cache``
    is keyed by bare hash id -- mixing block sizes in it would hand a 64-token
    weka block to a 16-token dynamo decode (or vice versa). The cache stores one
    ``int`` corpus OFFSET per hash id (not the decoded token list) via
    :meth:`~aiperf.dataset.graph.adapters.shared.content.CorpusContentSynthesizer._decode_block_tokens_offset_cached`,
    re-slicing the block on each hit -- the decode-cache tier's largest on-peak
    memory shave at corpus scale, byte-identical to the list cache by that
    method's full-reseed contract.

    Hash-id scope is deliberately GLOBAL (no ``trace_id`` threading, unlike
    the weka path's per-trace ``hash_id_scope: "local"`` namespace): dynamo's
    recorded ``input_sequence_hashes`` are chained sequence hashes over the
    actual prompt tokens -- equal hash means equal upstream content by
    construction, so a single namespace reproduces the recorded sharing
    exactly. Virtual ids (negative, per-parse counter) never recur across
    sessions within a parse.
    """
    from aiperf.dataset.graph.adapters.shared.content import (
        get_or_build_synthesizer,
    )

    synth = get_or_build_synthesizer(
        tokenizer_name, prompt_corpus=prompt_corpus, root_seed=root_seed
    )
    offsets: dict[int, int] = {}

    # Partial-tail / response seeds are trace-prefixed (weka parity): node ids
    # are trajectory-local (``{session}:{k}``) and would otherwise synthesize
    # identical tiny-prompt/response bytes for same-shaped nodes of DIFFERENT
    # trees -- manufactured cross-trace content sharing. Block decode stays
    # bare-hash-id global (equal recorded hash == equal content).
    def _tail_scoped(n_tokens: int, seed: str) -> list[int]:
        return synth._sample_partial_tail_tokens(n_tokens, f"{trace_scope}:{seed}")

    return ReconCallbacks(
        decode_block_tokens=lambda hash_ids: synth._decode_block_tokens_offset_cached(
            hash_ids, block_size=block_size, offset_cache=offsets
        ),
        sample_partial_tail_tokens=_tail_scoped,
        decode_tokens_to_text=synth._decode_tokens_to_text,
    )


def build_dynamo_llm_node(
    node: TrieNode,
    *,
    build: TrieNodeBuild,
) -> LlmNode:
    """Assemble the dynamo-flavored LlmNode from the shared build artifacts.

    The node carries NO inline prompt (``prompt=[]``): on the trie route the
    prompt content lives ONLY in the run's ``SegmentPool``, addressed by
    ``metadata["trie"]["prompt_segment_ids"]`` (stamped below). The store build,
    ``graph_meta`` sidecar (``strip_replay_text`` forces ``prompt=[]``), and the
    worker all materialize from the segment pool / mmap store -- none reads
    ``node.prompt`` -- matching the dag_jsonl adapter's ``prompt=[]`` convention.
    For in-process debugging, recover the content with
    ``segment_pool.materialize(read_prompt_segment_ids(node))``.

    Body params ride the NATIVE node fields (``model`` / ``streaming`` /
    ``max_tokens``); the envelope builder folds them into the wire body (the
    graph-credit path bypasses the endpoint's ``format_payload``).

    Must run AFTER :func:`~aiperf.dataset.graph.segment_ir.trie_content.build_trie_ir`:
    ``node.start`` (= ``warped_start``) is only stamped by the driver's idle-warp
    pass, so calling this earlier would freeze every ``arrival_offset_us`` at 0.

    Generation is ALWAYS pinned to the recording (weka parity): the native
    ``LlmNode.max_tokens`` carries the recorded ``output_tokens``
    (``Turn.max_tokens`` naming), folded into the wire body by the envelope
    builder and endpoint-mapped to the wire token field by the worker.
    A recorded 0 (zero-output/aborted turn, or a capture without
    ``output_tokens``) upgrades to 1 with a warning (:func:`wire_output_cap`)
    -- a 0 cap is a meaningless (and for some servers harmful) wire value.
    """
    meta = node.dynamo_meta
    req = node.request

    # Best-available streaming proxy: dynamo does NOT record the client's
    # stream flag; ttft_ms presence only means the request was routed through
    # a first-token-recording path (the push router records it regardless of
    # client streaming). The native ``streaming`` field reaches the envelope's
    # top-level ``stream``, which the worker's per-request stream override
    # (apply_run_level_payload_options) honors.
    streaming = req.ttft is not None

    llm = LlmNode(
        prompt=[],
        output=f"{node.node_id}_out",
        streaming=streaming,
        model=req.model,
        max_tokens=wire_output_cap(int(req.output_length), node_id=node.node_id),
        # Session identity is header-borne (see _node_meta); the native field
        # reaches the envelope and the worker attaches + uniquifies it.
        extra_headers=meta["extra_headers"],
        arrival_offset_us=int(round(node.start * MICROS_PER_SECOND)),
        expected=meta["expected"],
        metadata={
            # Session identity breadcrumbs: node ids are HASHES of session ids,
            # so this is the only way to map a lowered node back to its
            # recorded session. small_prompt marks the covered-count-0
            # reconstruction fallback (not byte-faithful) for ISL audits.
            "dynamo": {
                "session_id": meta["session_id"],
                "parent_session_id": meta["parent_session_id"],
                "turn_index": meta["turn_index"],
                "small_prompt": build.small_prompt,
            },
        },
    )
    # Only ``prompt_segment_ids`` is stamped: the build-synthesis
    # ``response_id`` / ``hash_ids`` companions (a duplicate per-node hash-list
    # copy) reach no build, sidecar, store, or dispatch consumer -- the sidecar
    # empties ``metadata["trie"]`` and the store envelope carries
    # ``prompt_segment_ids`` only -- so dynamo persists the same trie shape as
    # the dag_jsonl/native adapters. ``extra`` stays available for weka.
    return stamp_prompt_segment_ids(llm, build.prompt_path)


__all__ = [
    "DEFAULT_VIRTUAL_BLOCK_SIZE",
    "DynamoISLMismatchError",
    "build_dynamo_llm_node",
    "dynamo_recon_callbacks",
    "dynamo_trie_nodes",
]
