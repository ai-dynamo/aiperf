# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential oracle for ``build_trie_ir``'s prefix-path reuse.

``build_trie_ir`` splices the
content-parent's already-emitted sid chain for whole messages strictly inside
the inherited block prefix, instead of re-decoding + re-hashing every covered
message of every node -- avoiding a quadratic prefix re-emission at corpus
scale.

This file pins that optimization byte-for-byte against the straightforward
emission loop:
:func:`_oracle_build` is the naive ``build_trie_ir`` algorithm (the
quadratic per-node emission with fully inlined message assembly), kept here
as the reference ORACLE. Every generated and every ``@example``-pinned node set
must produce an IDENTICAL result from :func:`build_trie_ir` and the oracle:

* pool key INSERTION ORDER,
* full :class:`Segment` tuples (role, content, parent_id, wire_json),
* every node's ``prompt_path`` / ``response_id`` / ``small_prompt`` flag.

The oracle depends only on helpers whose behavior production shares
(``resolve_content_parents``, ``compute_asst_caps``, ``assign_block_tags``,
``add_message_chain``, ``assert_covered_isl``, ``emit_response_segment``), so a
reuse/splice bug shows up as a differential mismatch rather than being masked.

One deliberate non-difference: production INTERLEAVES callback calls
differently (the oracle decodes ALL of a node's messages, then adds the chain;
``_assemble_messages_from`` decodes and ``pool.add``s per message). The
``pool.add`` call sequence -- the byte-ordering authority -- is identical, and
the callbacks are pure per-build functions of their arguments (the
``ReconCallbacks`` determinism contract), so reordering decode calls between
adds is byte-invisible; this file asserts outputs, never callback interleaving.
"""

from __future__ import annotations

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aiperf.dataset.graph.adapters.dynamo.store_backed_pool import InterningSegmentPool
from aiperf.dataset.graph.segment_ir.interval_order import (
    apply_start_anchors,
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph.segment_ir.trie_content import (
    ReconCallbacks,
    TrieBuild,
    TrieISLMismatchError,
    TrieNode,
    TrieNodeBuild,
    TrieRequest,
    add_message_chain,
    apply_idle_gap_warp,
    assert_covered_isl,
    assign_block_tags,
    build_trie_ir,
    compute_asst_caps,
    emit_response_segment,
    resolve_content_parents,
)

BS = 4


# --------------------------------------------------------------------------- #
# The verbatim PRE-change emission loop, kept as the reference oracle.
# --------------------------------------------------------------------------- #
def _oracle_build(
    nodes: list[TrieNode],
    *,
    block_size: int,
    callbacks: ReconCallbacks,
    pool: SegmentPool,
    small_prompt_fallback: bool = False,
) -> TrieBuild:
    """Pre-rewrite ``build_trie_ir``: re-decode + re-emit EVERY covered message.

    Byte-for-byte the emission loop as it stood before the prefix-path-reuse
    rewrite -- the quadratic path that re-grouped every node's full covered
    prefix and re-``pool.add``ed each message (dedup made repeats no-ops). The
    non-emission pipeline (parent resolution, warp, ranks, edges, anchors, caps,
    tags) is the shared helper chain both paths run; only the per-node emission
    below differs from the rewrite.
    """
    resolve_content_parents(nodes)
    apply_idle_gap_warp(nodes, None)
    compute_ranks(nodes)
    edges_by_node = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges_by_node)
    caps = compute_asst_caps(nodes, block_size)
    tags = assign_block_tags(nodes, block_size, caps)

    builds: dict[str, TrieNodeBuild] = {}
    for node in nodes:
        node_tags = tags[node.node_id]
        covered = len(node_tags)
        small = False
        if covered == 0 and small_prompt_fallback and node.request.input_length > 0:
            toks = callbacks.sample_partial_tail_tokens(
                node.request.input_length, f"{node.node_id}:tiny"
            )
            prompt_path = [
                pool.add(
                    role="user",
                    content=callbacks.decode_tokens_to_text(toks),
                    tokens=toks,
                    parent_id=None,
                )
            ]
            small = True
        else:
            # Inlined verbatim pre-change ``assemble_messages``: group the full
            # covered tag list and emit every message fresh, root->tip.
            groups: list[tuple[str, list[int]]] = []
            for j, (role, starts) in enumerate(node_tags):
                if starts or not groups:
                    groups.append((role, [j]))
                else:
                    groups[-1][1].append(j)
            hash_ids = node.request.hash_ids
            messages: list[tuple[str, str, list[int]]] = []
            assembled_tokens = 0
            for role, idxs in groups:
                toks: list[int] = []
                for j in idxs:
                    toks.extend(callbacks.decode_block_tokens([hash_ids[j]]))
                assembled_tokens += len(toks)
                messages.append((role, callbacks.decode_tokens_to_text(toks), toks))
            prompt_path = add_message_chain(pool, messages)
            if callbacks.block_exact:
                assert_covered_isl(node, assembled_tokens, block_size)
        response_id = emit_response_segment(
            node,
            pool=pool,
            parent_id=prompt_path[-1] if prompt_path else None,
            callbacks=callbacks,
        )
        builds[node.node_id] = TrieNodeBuild(
            prompt_path=prompt_path, response_id=response_id, small_prompt=small
        )
    return TrieBuild(builds=builds, edges_by_node=edges_by_node)


# --------------------------------------------------------------------------- #
# Collision-free stub callbacks + node construction.
# --------------------------------------------------------------------------- #
def _stub_callbacks(block_size: int = BS, block_exact: bool = True) -> ReconCallbacks:
    """Deterministic, injective stubs: block tokens = [hash_id] * block_size."""
    return ReconCallbacks(
        decode_block_tokens=lambda hids: [t for h in hids for t in [h] * block_size],
        # Keyed on the seed string so response/tiny tokens stay node-unique and
        # never alias a prompt block's tokens.
        sample_partial_tail_tokens=lambda n, seed: [abs(hash(seed)) % 100003] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
        block_exact=block_exact,
    )


def _node(
    nid: str,
    order: int,
    hashes: list[int],
    in_tok: int,
    out_tok: int = 4,
    t: float | None = None,
) -> TrieNode:
    return TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=list(hashes),
            input_length=in_tok,
            output_length=out_tok,
            t=float(order) if t is None else t,
            api_time=0.5,
        ),
        order=order,
    )


def _fresh(spec: list[tuple[str, int, list[int], int, int]]) -> list[TrieNode]:
    """Rebuild nodes from a (nid, order, hashes, in_tok, out_tok) spec.

    A fresh node list per leg is mandatory: the trie pipeline mutates nodes in
    place (content_parent, warped_start, rank), so oracle and rewrite must each
    get their own copies.
    """
    return [
        _node(nid, order, hashes, in_tok, out_tok)
        for nid, order, hashes, in_tok, out_tok in spec
    ]


# --------------------------------------------------------------------------- #
# Differential harness.
# --------------------------------------------------------------------------- #
def _assert_identical(
    spec: list[tuple[str, int, list[int], int, int]],
    *,
    block_size: int = BS,
    small_prompt_fallback: bool = False,
    block_exact: bool = True,
    interning: bool = False,
) -> tuple[SegmentPool, TrieBuild]:
    """Build with the oracle and the rewrite; assert byte-identical results."""

    def _pool() -> SegmentPool:
        return InterningSegmentPool() if interning else SegmentPool()

    pool_o = _pool()
    res_o = _oracle_build(
        _fresh(spec),
        block_size=block_size,
        callbacks=_stub_callbacks(block_size, block_exact),
        pool=pool_o,
        small_prompt_fallback=small_prompt_fallback,
    )
    pool_n = _pool()
    res_n = build_trie_ir(
        _fresh(spec),
        block_size=block_size,
        callbacks=_stub_callbacks(block_size, block_exact),
        pool=pool_n,
        idle_gap_cap_seconds=None,
        small_prompt_fallback=small_prompt_fallback,
    )

    # (1) pool insertion ORDER is byte-identical.
    assert list(pool_o.by_id.keys()) == list(pool_n.by_id.keys()), (
        "pool insertion order diverged"
    )
    # (2) full Segment tuples are byte-identical.
    for sid in pool_o.by_id:
        so = pool_o.by_id[sid]
        sn = pool_n.by_id[sid]
        assert (so.role, so.content, so.parent_id, so.wire_json) == (
            sn.role,
            sn.content,
            sn.parent_id,
            sn.wire_json,
        ), f"segment {sid} tuple diverged"
    # (3) per-node prompt_path / response_id / small_prompt.
    assert res_o.builds.keys() == res_n.builds.keys()
    for nid, bo in res_o.builds.items():
        bn = res_n.builds[nid]
        assert bo.prompt_path == bn.prompt_path, f"node {nid} prompt_path diverged"
        assert bo.response_id == bn.response_id, f"node {nid} response_id diverged"
        assert bo.small_prompt == bn.small_prompt, f"node {nid} small_prompt diverged"
    return pool_n, res_n


# --------------------------------------------------------------------------- #
# Named adversarial legs (the geometries from /tmp/adv_reuse_diff.py, adopted
# as the mandated @example pins per both adversarial reviews).
# --------------------------------------------------------------------------- #
# depth >= 8 extending chain -- the deep-chain gate-outcome pin (Review 0 rev 2):
# a cumulative-count bookkeeping slip on a long reuse chain trips the ISL gate.
_EXTEND_CHAIN_DEEP = [
    (f"x{i}", i, list(range(50, 50 + i + 1)), BS * (i + 1), 4) for i in range(9)
]
# child == parent: whole prefix reused, empty fresh region.
_DUPLICATE = [
    ("a", 0, [1, 2, 3], 12, 4),
    ("b", 1, [1, 2, 3, 4, 5], 20, 4),
    ("c", 2, [1, 2, 3, 4, 5], 20, 4),
    ("c2", 3, [1, 2, 3, 4, 5], 20, 4),
]
# branch point: BOTH grandchild record-composition cases live here. 'q'
# truncates p's message into a fragment ([0,2) of p's [0,4)); 's' then splices
# straight THROUGH q's fragment-bearing record (fragment-then-reuse), while
# 'r' branches off q past the fragment (reuse-after-fragment).
_BRANCH_POINT_FRAGMENT = [
    ("p", 0, [60, 61, 62, 63], 16, 4),
    ("q", 1, [60, 61, 70, 71], 16, 4),
    ("r", 2, [60, 61, 70, 99], 16, 4),
    ("s", 3, [60, 61, 70, 71, 80, 81], 24, 4),
]
# over-share: 'e' declares in//bs = 2 < lcp = 6 -> covered clamp bounds reuse.
_OVER_SHARE = [
    ("a", 0, [1, 2, 3], 12, 4),
    ("b", 1, [1, 2, 3, 4, 5, 6], 24, 4),
    ("e", 2, [1, 2, 3, 4, 5, 6], 8, 4),
    ("e2", 3, [1, 2, 3, 4, 5, 6, 7], 28, 4),
]
# under-covering parent: 'p' covers 2 blocks < child lcp 4 -> len(parent_tags) clamp.
_UNDER_COVER_PARENT = [
    ("p", 0, [20, 21, 22, 23], 8, 4),
    ("q", 1, [20, 21, 22, 23, 24], 20, 4),
    ("r", 2, [20, 21, 22, 23, 24, 25], 24, 4),
]
# degenerate pull-back caps: r4 has new_n 0 (empty fresh region, whole-splice
# prompt_path) and caps its owner's frozen tags via compute_asst_caps.
_DEGENERATE_PULLBACK = [
    ("r1", 0, [30], 4, 4),
    ("r2", 1, [30, 31], 8, 4),
    ("r3", 2, [30, 31, 32], 12, 4),
    ("r4", 3, [30, 31, 32], 8, 4),
    ("r5", 4, [30, 31, 32, 33], 16, 4),
]
# covered==0 parent (no record published) with covered>0 children reusing NOTHING.
_ZERO_COVERED_PARENT = [
    ("h", 0, [1, 2, 3], 2, 4),
    ("i", 1, [1, 2, 3, 4], 16, 4),
    ("j", 2, [1, 2, 3, 4], 16, 4),
]
# boundary landing exactly on a message end.
_BOUNDARY_EXACT_MESSAGE_END = [
    ("a", 0, [1], 4, 8),
    ("b", 1, [1, 2, 3], 12, 4),
    ("c", 2, [1, 2, 3, 9], 16, 4),
    ("d", 3, [1, 2, 9, 9], 12, 4),
]
# tags coincide beyond the lcp but hash ids differ -- the geometry-not-tags trap.
_TAGS_COINCIDE_BEYOND_LCP = [
    ("a", 0, [1, 2, 3, 4], 16, 4),
    ("b", 1, [1, 2, 8, 9], 16, 4),
    ("c", 2, [1, 2, 8, 9, 10], 20, 4),
]
# j>=1 splice PLUS a straddling fragment in ONE node: 'c' over-shares b
# (covered 4 < lcp 6) while c's degenerate pull-back caps b's assistant run to
# 0, merging b's new region into one message [3,6) -- so c splices j=1 whole
# message [0,3), resumes at block 3 (a PARENT-COPIED tag, exercising that branch
# of the production resume-point assert), and re-emits the fragment [3,4) fresh.
# 'd' then splices straight through c's fragment-bearing record (j=2).
_SPLICE_PLUS_FRAGMENT = [
    ("a", 0, [1, 2, 3], 12, 4),
    ("b", 1, [1, 2, 3, 4, 5, 6], 24, 4),
    ("c", 2, [1, 2, 3, 4, 5, 6], 16, 4),
    ("d", 3, [1, 2, 3, 4, 5, 6, 7], 28, 4),
]
# dynamo-shaped: negative virtual hash ids, extending chain (block_size=16 leg
# supplied via the parametrize below).
_DYNAMO_NEGATIVE_IDS = [
    ("d0", 0, [-1, -2], 32, 8),
    ("d1", 1, [-1, -2, -3], 48, 8),
    ("d2", 2, [-1, -2, -3, -4], 64, 8),
    ("d3", 3, [-1, -2, -3, -4, -5], 80, 8),
]

_NAMED_CASES = {
    "extend_chain_deep": _EXTEND_CHAIN_DEEP,
    "duplicate": _DUPLICATE,
    "branch_point_fragment": _BRANCH_POINT_FRAGMENT,
    "over_share": _OVER_SHARE,
    "under_cover_parent": _UNDER_COVER_PARENT,
    "degenerate_pullback_caps": _DEGENERATE_PULLBACK,
    "zero_covered_parent": _ZERO_COVERED_PARENT,
    "boundary_exact_message_end": _BOUNDARY_EXACT_MESSAGE_END,
    "tags_coincide_beyond_lcp": _TAGS_COINCIDE_BEYOND_LCP,
    "splice_plus_fragment": _SPLICE_PLUS_FRAGMENT,
}


@pytest.mark.parametrize("name", list(_NAMED_CASES))
@pytest.mark.parametrize("fallback", [False, True], ids=["no_fallback", "fallback"])
def test_named_cases_byte_identical(name: str, fallback: bool) -> None:
    """Every adversarial geometry emits byte-identically under both fallbacks."""
    _assert_identical(_NAMED_CASES[name], small_prompt_fallback=fallback)


@pytest.mark.parametrize("name", list(_NAMED_CASES))
def test_named_cases_block_exact_false(name: str) -> None:
    """block_exact=False (scheduling-only placeholder parse) stays byte-identical."""
    _assert_identical(_NAMED_CASES[name], block_exact=False, small_prompt_fallback=True)


@pytest.mark.parametrize("name", list(_NAMED_CASES))
def test_named_cases_interning_pool(name: str) -> None:
    """Every geometry stays byte-identical on the eager InterningSegmentPool route."""
    _assert_identical(_NAMED_CASES[name], small_prompt_fallback=True, interning=True)


def test_dynamo_shaped_negative_ids_block16() -> None:
    """Dynamo-shaped leg: negative virtual hash ids, block_size=16, fallback on."""
    spec = [
        ("d0", 0, [-1, -2], 32, 8),
        ("d1", 1, [-1, -2, -3], 48, 8),
        ("d2", 2, [-1, -2, -3, -4], 64, 8),
        ("d3", 3, [-1, -2, -3, -4, -5], 80, 8),
    ]
    _assert_identical(spec, block_size=16, small_prompt_fallback=True)


def test_small_prompt_parent_with_covered_child() -> None:
    """A covered>0 CHILD of a small-prompt (covered==0) parent must reuse NOTHING.

    The small-prompt parent publishes no emission record; the child's inherited
    count is structurally 0 (its parent tags are empty), so the missing-parent-
    record guard (records.get() AND inherited > 0) must hold. Both fallback and
    non-fallback are exercised: with fallback the parent emits a single tiny user
    message (still no record), without it the parent emits an empty prompt.
    """
    # 'tiny' covers 0 blocks (in=2 < bs=4); 'child' shares its hash prefix but
    # declares enough input to cover blocks.
    spec = [
        ("tiny", 0, [1, 2, 3], 2, 4),
        ("child", 1, [1, 2, 3, 4], 16, 4),
        ("grandchild", 2, [1, 2, 3, 4, 5], 20, 4),
    ]
    _assert_identical(spec, small_prompt_fallback=True)
    _assert_identical(spec, small_prompt_fallback=False)


def test_interning_pool_splice_is_identical_to_canonical_first_born() -> None:
    """On InterningSegmentPool spliced prompt_path entries ARE the parent's
    canonical first-born str objects (claim vi -- identity, not just equality).

    Every value re-listed across nodes must resolve to ONE canonical object, so
    the reuse splice must hand back the exact object the first emitter interned,
    never a fresh-but-equal str.
    """
    spec = _EXTEND_CHAIN_DEEP
    pool = InterningSegmentPool()
    result = build_trie_ir(
        _fresh(spec),
        block_size=BS,
        callbacks=_stub_callbacks(),
        pool=pool,
        idle_gap_cap_seconds=None,
    )
    canonical: dict[str, str] = {}
    for build in result.builds.values():
        for sid in build.prompt_path:
            if sid in canonical:
                # Same value re-listed downstream must be the SAME object.
                assert sid is canonical[sid], "interning identity broken by splice"
            else:
                canonical[sid] = sid
    # The deep chain must actually re-list shared prefixes (else the assert is vacuous).
    total = sum(len(b.prompt_path) for b in result.builds.values())
    assert total > len(canonical), "expected shared prefixes to be re-listed"


def test_resume_point_lands_on_message_start() -> None:
    """The reuse resume block must open a message (the production-assert invariant).

    Re-derives, per node, the geometric ``inherited`` and the parent's message-end
    boundaries the way the rewrite does, and checks the resume tag is a message
    start -- the cheapest tripwire against grouping drift in the frozen file.
    """
    import bisect

    from aiperf.dataset.graph.segment_ir.trie_content import (
        compute_turn_block_geometry,
    )

    spec = _EXTEND_CHAIN_DEEP + [("z", 9, list(range(50, 60)), 40, 4)]
    nodes = _fresh(spec)
    resolve_content_parents(nodes)
    caps = compute_asst_caps(nodes, BS)
    tags = assign_block_tags(nodes, BS, caps)
    # message end blocks per node (exclusive), from the frozen tags.
    ends: dict[str, list[int]] = {}
    for node in nodes:
        node_tags = tags[node.node_id]
        me: list[int] = []
        for k, (_role, starts) in enumerate(node_tags):
            if starts and k > 0:
                me.append(k)
        me.append(len(node_tags))
        ends[node.node_id] = me
        parent = node.content_parent
        if parent is None:
            continue
        geo = compute_turn_block_geometry(
            parent.request.hash_ids,
            node.request.hash_ids,
            node.request.input_length,
            BS,
        )
        inherited = min(geo.lcp, len(tags[parent.node_id]), geo.m_curr_covered)
        if inherited <= 0:
            continue
        j = bisect.bisect_right(ends[parent.node_id], inherited)
        if j <= 0:
            continue
        start_block = ends[parent.node_id][j - 1]
        if start_block < len(node_tags):
            assert node_tags[start_block][1] is True, (
                f"resume block {start_block} of {node.node_id} is not a message start"
            )


def test_reused_prefix_decoded_exactly_once_kills_quadratic() -> None:
    """A deep chain decodes each shared block ONCE -- the quadratic is dead.

    Counting stub: every ``decode_block_tokens`` call is recorded. Under the
    rewrite each block is materialized exactly once (at first occurrence) and
    reused by splice thereafter, so the total decode count equals the number of
    DISTINCT covered blocks across the chain (linear), not the per-node
    re-decode sum (quadratic). This simultaneously pins that a reused prefix is
    never re-decoded -- so a callback that would drift on a repeat emission
    cannot even be called there (repeat-emission drift is excluded by the
    ReconCallbacks determinism contract, not re-checked at the child).
    """
    n = 12
    spec = [
        (f"x{i}", i, list(range(100, 100 + i + 1)), BS * (i + 1), 4) for i in range(n)
    ]
    calls: list[int] = []

    def _counting_decode(hids: list[int]) -> list[int]:
        calls.append(len(hids))
        return [t for h in hids for t in [h] * BS]

    cb = ReconCallbacks(
        decode_block_tokens=_counting_decode,
        sample_partial_tail_tokens=lambda n, seed: [abs(hash(seed)) % 100003] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
    )
    build_trie_ir(
        _fresh(spec),
        block_size=BS,
        callbacks=cb,
        pool=SegmentPool(),
        idle_gap_cap_seconds=None,
    )
    # Node i covers i+1 blocks; distinct covered blocks across the chain = the
    # LAST node's covered count = n (blocks 100..100+n-1). The quadratic path
    # decoded sum(i+1 for i) = n(n+1)/2 blocks instead.
    distinct_blocks = n
    total_decoded = sum(calls)
    assert total_decoded == distinct_blocks, (
        f"decoded {total_decoded} blocks; linear reuse must decode {distinct_blocks} "
        f"(quadratic would decode {n * (n + 1) // 2})"
    )


def test_first_occurrence_decode_drift_still_aborts() -> None:
    """First-occurrence (fresh materialization) decode drift still hard-aborts.

    The reuse narrowing does NOT weaken the gate on freshly materialized blocks:
    a decode that emits 2x block_size tokens on the FIRST occurrence of a block
    must still trip the ISL gate (at the parent, before any child could reuse).
    """
    drifted = ReconCallbacks(
        decode_block_tokens=lambda hids: [t for h in hids for t in [h] * (BS * 2)],
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=lambda toks: " ".join(str(t) for t in toks),
    )
    spec = [
        ("a", 0, [1, 2], 8, 4),
        ("b", 1, [1, 2, 3], 12, 4),
    ]
    with pytest.raises(TrieISLMismatchError):
        build_trie_ir(
            _fresh(spec),
            block_size=BS,
            callbacks=drifted,
            pool=SegmentPool(),
            idle_gap_cap_seconds=None,
        )


# --------------------------------------------------------------------------- #
# Hypothesis differential over arbitrary node sets.
# --------------------------------------------------------------------------- #
# Small alphabet -> dense shared prefixes, branch points, and exact duplicates.
_HASH = st.integers(min_value=-3, max_value=6)
_ROW = st.tuples(st.lists(_HASH, max_size=8), st.integers(min_value=0, max_value=40))
_NODE_SETS = st.lists(_ROW, min_size=1, max_size=12)


def _spec_from_rows(
    rows: list[tuple[list[int], int]],
) -> list[tuple[str, int, list[int], int, int]]:
    return [(f"r{i}", i, hashes, in_tok, 4) for i, (hashes, in_tok) in enumerate(rows)]


@settings(deadline=None, max_examples=300)
@given(rows=_NODE_SETS)
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _EXTEND_CHAIN_DEEP])
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _BRANCH_POINT_FRAGMENT])
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _OVER_SHARE])
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _UNDER_COVER_PARENT])
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _TAGS_COINCIDE_BEYOND_LCP])
@example(rows=[(h, in_tok) for _, _, h, in_tok, _ in _SPLICE_PLUS_FRAGMENT])
def test_prefix_reuse_matches_oracle_property(
    rows: list[tuple[list[int], int]],
) -> None:
    """The rewrite == the verbatim pre-change oracle on arbitrary node sets.

    Both fallback modes are exercised so the small-prompt / record-publication
    branches are differenced too (record composition is the only inductive state
    the rewrite introduces).
    """
    spec = _spec_from_rows(rows)
    _assert_identical(spec, small_prompt_fallback=False)
    _assert_identical(spec, small_prompt_fallback=True)


# --------------------------------------------------------------------------- #
# Real-fixture leg: a committed weka trace through the REAL weka lowering.
# --------------------------------------------------------------------------- #
def test_real_weka_fixture_byte_matches_oracle() -> None:
    """A real weka fixture trace, real production callbacks: oracle == rewrite.

    Stub-free: the trace is parsed through the real weka models + flattening and
    both legs run the real builtin-tokenizer/coding-corpus callbacks
    (fresh per leg, so per-trace decode caches are cold), byte-comparing pool
    key INSERTION ORDER, full Segment tuples, and every node's build artifacts.
    The fixture is an extending chain, so the reuse path genuinely fires.
    """
    from pathlib import Path

    import orjson

    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
    from aiperf.dataset.graph.adapters.weka.trie_build import (
        _default_callbacks,
        _flatten_requests,
    )

    fixture = Path(__file__).parent / "fixtures" / "weka_min.json"
    trace = WekaTrace.model_validate(orjson.loads(fixture.read_bytes()))
    bs = trace.block_size

    def _real_callbacks() -> ReconCallbacks:
        return _default_callbacks(
            "builtin",
            "coding",
            1234,
            trace_id=trace.id,
            block_size=bs,
            hash_scope=trace.hash_id_scope,
        )

    pool_o = SegmentPool()
    res_o = _oracle_build(
        _flatten_requests(trace.requests, root_scope=trace.id),
        block_size=bs,
        callbacks=_real_callbacks(),
        pool=pool_o,
    )
    pool_n = SegmentPool()
    res_n = build_trie_ir(
        _flatten_requests(trace.requests, root_scope=trace.id),
        block_size=bs,
        callbacks=_real_callbacks(),
        pool=pool_n,
        idle_gap_cap_seconds=None,
    )

    assert list(pool_o.by_id.keys()) == list(pool_n.by_id.keys()), (
        "pool insertion order diverged on the real fixture"
    )
    for sid, so in pool_o.by_id.items():
        sn = pool_n.by_id[sid]
        assert (so.role, so.content, so.parent_id, so.wire_json) == (
            sn.role,
            sn.content,
            sn.parent_id,
            sn.wire_json,
        ), f"segment {sid} tuple diverged on the real fixture"
    assert res_o.builds.keys() == res_n.builds.keys()
    for nid, bo in res_o.builds.items():
        bn = res_n.builds[nid]
        assert bo.prompt_path == bn.prompt_path
        assert bo.response_id == bn.response_id
        assert bo.small_prompt == bn.small_prompt
    # The fixture must actually exercise reuse: its requests extend one chain
    # ([1,2] -> [1,2,3] -> [1,2,3,4]), so later prompt paths share a prefix.
    paths = [res_n.builds[nid].prompt_path for nid in sorted(res_n.builds)]
    assert len(paths) >= 3
    assert paths[1][: len(paths[0])] == paths[0]
    assert paths[2][: len(paths[1])] == paths[1]
