# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-parent resolution: O(n) trie equivalence + perf gate.

:func:`~aiperf.dataset.graph.segment_ir.trie_content.resolve_content_parents`
was an O(n^2 * m) double loop (every node scanned against all earlier nodes,
each comparison O(m) over ``hash_ids`` lists 2000+ long on real corpus traces).
It is now an incremental prefix-trie pass, O(sum of hash_ids lengths).

These tests pin the trie pass byte-for-byte to the OLD double-loop semantics:
``_old_resolve_reference`` is the verbatim pre-fix algorithm kept here as an
ORACLE, and the equivalence test asserts the new pass returns the IDENTICAL
``content_parent`` mapping on hand-built multi-branch node sets (sequential
continuation, parallel fan-out, truncation branch, tie-break toward most-recent,
disjoint roots, exact duplicates). The perf test asserts a corpus-scale node set
resolves in well under the wall the old loop hit.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import orjson
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from aiperf.dataset.graph.adapters.weka.trace_models import (
    WekaNormalRequest,
    WekaTrace,
)
from aiperf.dataset.graph.adapters.weka.trie_build import _flatten_requests
from aiperf.dataset.graph.segment_ir.trie_content import (
    TrieNode,
    TrieRequest,
    resolve_content_parents,
)


def _is_full_prefix(prefix: list[int], seq: list[int]) -> bool:
    """``True`` when ``prefix`` is a full leading slice of ``seq``."""
    return len(prefix) <= len(seq) and seq[: len(prefix)] == prefix


def _lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two hash-id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _old_resolve_reference(nodes: list[TrieNode]) -> None:
    """Verbatim pre-fix O(n^2 * m) resolution -- the equivalence ORACLE.

    For node R, scan all earlier nodes and pick the one whose ``hash_ids`` is the
    longest FULL prefix of R's, tie-broken toward the most recent (``>=``). When
    no earlier node is a full prefix, fall back to the earlier node with the
    longest partial LCP, tie-broken toward the earliest (strict ``>``). With no
    overlap at all, R stays a fresh root.
    """
    for idx, r in enumerate(nodes):
        r_hashes = r.request.hash_ids
        best_full: TrieNode | None = None
        best_full_len = -1
        best_partial: TrieNode | None = None
        best_partial_lcp = 0
        for p in nodes[:idx]:
            p_hashes = p.request.hash_ids
            if p_hashes and _is_full_prefix(p_hashes, r_hashes):
                if len(p_hashes) >= best_full_len:
                    best_full_len = len(p_hashes)
                    best_full = p
                continue
            lcp = _lcp_len(p_hashes, r_hashes)
            if lcp > best_partial_lcp:
                best_partial_lcp = lcp
                best_partial = p
        if best_full is not None:
            r.content_parent = best_full
        elif best_partial is not None:
            r.content_parent = best_partial


def _mk_nodes(hash_id_lists: list[list[int]]) -> list[TrieNode]:
    """Build a recorded-order ``TrieNode`` list from hash-id sequences only.

    The resolution pass touches only ``request.hash_ids`` + ``order``; the other
    request fields are filled with valid placeholders.
    """
    nodes: list[TrieNode] = []
    for i, hashes in enumerate(hash_id_lists):
        req = WekaNormalRequest.model_validate(
            {
                "t": float(i),
                "type": "n",
                "model": "M",
                "in": 1,
                "out": 1,
                "hash_ids": hashes,
            }
        )
        nodes.append(TrieNode(node_id=f"r_{i}", request=req, order=i))
    return nodes


def _parent_orders(nodes: list[TrieNode]) -> list[int | None]:
    """Map each node's resolved content-parent to the parent's ``order`` (or None)."""
    order_of = {id(n): n.order for n in nodes}
    return [
        None if n.content_parent is None else order_of[id(n.content_parent)]
        for n in nodes
    ]


# Hand-built multi-branch corpus stressing every resolution branch. The trailing
# comment on each row names the case the row exercises.
_CASES = [
    pytest.param(
        [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
        id="sequential_continuation",
    ),
    pytest.param(
        [[1, 2], [1, 2, 3], [1, 2, 4]],
        id="parallel_fanout_branch_point",
    ),
    pytest.param(
        # node3 [1,2,3] is a SHORTER (truncated) reuse of node1 [1,2,3,4,5]:
        # node1 is not a full prefix of node3 (longer); node0 [1,2] IS -> parent 0.
        [[1, 2], [1, 2, 3, 4, 5], [1, 2, 3]],
        id="truncation_branch_prefers_shorter_full_prefix",
    ),
    pytest.param(
        # Two identical [1,2] precede R [1,2,9]; both are full prefixes of R of
        # equal length -> tie-break toward the MOST RECENT (order 1, not 0).
        [[1, 2], [1, 2], [1, 2, 9]],
        id="full_prefix_tie_break_most_recent",
    ),
    pytest.param(
        # Two nodes share R's partial LCP of 2 with no full prefix; the partial
        # tie-break favors the EARLIEST (order 0).
        [[1, 2, 7], [1, 2, 8], [1, 2, 9, 9]],
        id="partial_lcp_tie_break_earliest",
    ),
    pytest.param(
        [[1, 2], [3, 4], [5, 6]],
        id="disjoint_roots_no_overlap",
    ),
    pytest.param(
        [[], [1, 2], [], [1, 2, 3]],
        id="empty_hash_ids_never_parent_or_child",
    ),
    pytest.param(
        # A deep shared trunk with a late short reuse + a most-recent equal-length
        # duplicate competing on the full-prefix tie-break.
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7],
        ],
        id="mixed_trunk_reuse_and_duplicate",
    ),
]


@pytest.mark.parametrize("hash_id_lists", _CASES)
def test_trie_resolution_matches_old_oracle(
    hash_id_lists: list[list[int]],
) -> None:
    """New trie pass == old O(n^2) oracle on every hand-built multi-branch set."""
    new_nodes = _mk_nodes(hash_id_lists)
    resolve_content_parents(new_nodes)

    old_nodes = _mk_nodes(hash_id_lists)
    _old_resolve_reference(old_nodes)

    assert _parent_orders(new_nodes) == _parent_orders(old_nodes)


def test_trie_resolution_explicit_expected_parents() -> None:
    """Lock the resolved parents to explicit ordinals (not just oracle parity)."""
    nodes = _mk_nodes(
        [
            [1, 2],  # 0: root (no earlier)
            [1, 2, 3, 4, 5],  # 1: extends 0 (full prefix 0)
            [1, 2, 3],  # 2: 1 is longer (partial); 0 is full prefix -> 0
            [1, 2, 3, 4, 5],  # 3: equals 1 (full prefix, most recent of {1}) -> 1
            [9, 9],  # 4: disjoint root
        ]
    )
    resolve_content_parents(nodes)
    assert _parent_orders(nodes) == [None, 0, 0, 1, None]


def _mk_nodes_direct(hash_id_lists: list[list[int]]) -> list[TrieNode]:
    """Build nodes straight from :class:`TrieRequest` (no ``ge=0`` model gate).

    :func:`resolve_content_parents` operates on the FULL opaque-key domain of
    ``TrieRequest.hash_ids``: dynamo emits negative VIRTUAL ids
    (``itertools.count(-1, -1)`` in ``adapters/dynamo/trie_lowering.py``) and
    weka JSON ids need not fit 64 bits. ``WekaNormalRequest``'s ``ge=0``
    constraint (used by :func:`_mk_nodes`) would reject the negative-id
    adversarial cases, so the differential property builds nodes directly.
    """
    return [
        TrieNode(
            node_id=f"r_{i}",
            request=TrieRequest(
                hash_ids=list(hashes),
                input_length=1,
                output_length=1,
                t=float(i),
                api_time=0.0,
            ),
            order=i,
        )
        for i, hashes in enumerate(hash_id_lists)
    ]


# Small-alphabet strategy: a narrow id range makes shared prefixes, exact
# duplicates, and branch points DENSE (the resolution branches that matter),
# and the bounds keep the O(n^2 * m) oracle sub-second (<=30 nodes x <=40 ids).
_HASH_ID_LISTS = st.lists(
    st.lists(st.integers(min_value=-3, max_value=6), max_size=40),
    max_size=30,
)


@settings(deadline=None, max_examples=250)
@given(node_lists=_HASH_ID_LISTS)
# hash(-1) == hash(-2) == -2 in CPython: the collision pair that distinguishes
# nested-int-dict hashing from tuple-key ((state, h)) hashing. -1 and -2 must
# stay DISTINCT sibling transitions despite the equal hash; a trailing duplicate
# [-1, -2] also pins the most-recent full-prefix tie-break under the collision.
@example(node_lists=[[-1, -2], [-1, -2, 5], [-2, -1], [-1, -2]])
# A >2**64 id: arbitrary-width Python ints as keys (tuple-key hashing must not
# alias them -- the combined-single-int-key variant was REJECTED for this).
@example(node_lists=[[2**64 + 1, 7], [2**64 + 1, 7, 9], [2**64 + 1]])
# Duplicate hash_ids node pair: nodes 0 and 1 are identical full prefixes of
# node 2 -> the full-prefix tie-break must select the MOST RECENT (order 1).
@example(node_lists=[[1, 2, 3], [1, 2, 3], [1, 2, 3, 4]])
def test_trie_resolution_matches_oracle_property(
    node_lists: list[list[int]],
) -> None:
    """The resolution pass == the verbatim O(n^2 * m) oracle on arbitrary inputs.

    The differential harness for the flat-int-automaton rewrite: for every
    generated (and every ``@example``-pinned adversarial) node set, the
    content-parent mapping the trie pass produces must be IDENTICAL to the one
    :func:`_old_resolve_reference` produces by brute force. This is the property
    that makes the frozen-behavior claim a theorem rather than a hope.
    """
    new_nodes = _mk_nodes_direct(node_lists)
    resolve_content_parents(new_nodes)

    old_nodes = _mk_nodes_direct(node_lists)
    _old_resolve_reference(old_nodes)

    assert _parent_orders(new_nodes) == _parent_orders(old_nodes)


def _synth_corpus_node_lists(
    n_nodes: int, base_len: int, branch_stride: int
) -> list[list[int]]:
    """A corpus-scale stressor: many nodes with long shared-prefix ``hash_ids``.

    Reproduces the O(n^2 * m) wall shape -- a deep shared trunk (so every pairwise
    comparison scans ~``base_len`` ids) with periodic branches. ``branch_stride``
    forks a fresh sub-trunk so both the full-prefix and partial-LCP branches fire.
    """
    lists: list[list[int]] = []
    trunk = list(range(base_len))
    for i in range(n_nodes):
        if i % branch_stride == 0 and i > 0:
            # Fork: replace the tail so this node branches off the shared trunk.
            seq = trunk[: base_len // 2] + list(range(10_000 + i, 10_000 + i + 8))
        else:
            seq = trunk + [base_len + i]
        lists.append(seq)
    return lists


def test_trie_resolution_corpus_scale_is_fast() -> None:
    """O(n) trie pass resolves a corpus-scale node set well under the old wall.

    A 466-node trace with ~2000-id ``hash_ids`` is the real corpus shape that the
    old O(n^2 * m) loop ground through in seconds-to-minutes; this synthetic set
    is comparably sized. The trie pass must finish in a small fraction of that.
    Equivalence to the old oracle is asserted alongside the timing so a faster but
    wrong implementation cannot pass.
    """
    lists = _synth_corpus_node_lists(n_nodes=500, base_len=2000, branch_stride=37)

    new_nodes = _mk_nodes(lists)
    t0 = time.perf_counter()
    resolve_content_parents(new_nodes)
    elapsed = time.perf_counter() - t0

    old_nodes = _mk_nodes(lists)
    _old_resolve_reference(old_nodes)
    assert _parent_orders(new_nodes) == _parent_orders(old_nodes)

    assert elapsed < 2.0, f"trie resolution took {elapsed:.3f}s (expected < 2s)"


# Real corpus traces are not committed (multi-MB each). The real-corpus perf
# test reads the largest suitable trace from the directory named by the
# AIPERF_TEST_WEKA_CORPUS_DIR env var and skips when it is unset, so CI stays
# green without the corpus while a developer with a local corpus can reproduce
# the unblock claim.
_CORPUS_DIR_ENV = "AIPERF_TEST_WEKA_CORPUS_DIR"


# Cap the oracle-equivalence trace so the deliberately-slow O(n^2 * m) reference
# (the thing we replaced) stays bounded; the new pass is timed on it too.
_MAX_ORACLE_TRACE_BYTES = 10 * 1024 * 1024


def _largest_local_trace(max_bytes: int | None = None) -> Path | None:
    """Largest ``*.json`` (optionally <= ``max_bytes``) in the env-named corpus dir."""
    corpus_dir = os.environ.get(_CORPUS_DIR_ENV)
    if not corpus_dir:
        return None
    candidates = glob.glob(os.path.join(corpus_dir, "*.json"))
    if max_bytes is not None:
        candidates = [c for c in candidates if os.path.getsize(c) <= max_bytes]
    if not candidates:
        return None
    return Path(max(candidates, key=os.path.getsize))


def test_real_corpus_parent_resolution_under_10s() -> None:
    """Content-parent resolution of a REAL corpus trace completes fast.

    The pre-fix O(n^2 * m) double loop ground through real corpus traces
    (hundreds of nodes, ``hash_ids`` thousands long) for seconds-to-minutes and
    blocked corpus-scale builds. The trie pass must resolve the largest local
    trace in well under 10s. Equivalence to the old oracle is re-asserted on the
    real node set so a wrong-but-fast pass cannot slip through.

    NOTE (second bottleneck, out of scope here): the FULL ``build_trie_graph`` of
    this same trace still exceeds 120s, dominated by per-node content-synthesis
    re-replay (``_replay_lineage`` -> ``SegmentPool.add`` -> ``segment_id`` token
    hashing), NOT by parent resolution.
    """
    trace_path = _largest_local_trace(max_bytes=_MAX_ORACLE_TRACE_BYTES)
    if trace_path is None:
        pytest.skip(
            f"no local weka corpus trace available: set {_CORPUS_DIR_ENV} "
            "to a directory of raw weka trace .json files to run this test"
        )

    trace = WekaTrace.model_validate(orjson.loads(trace_path.read_bytes()))
    nodes = _flatten_requests(trace.requests, root_scope=trace.id)

    t0 = time.perf_counter()
    resolve_content_parents(nodes)
    elapsed = time.perf_counter() - t0

    oracle = _flatten_requests(trace.requests, root_scope=trace.id)
    _old_resolve_reference(oracle)
    assert _parent_orders(nodes) == _parent_orders(oracle)

    assert elapsed < 10.0, (
        f"real-corpus parent resolution took {elapsed:.3f}s "
        f"(trace {trace_path.name}, {len(nodes)} nodes, expected < 10s)"
    )
