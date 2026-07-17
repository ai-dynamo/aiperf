# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session-tree dynamo build: grouping, cross-parent edge drop, byte-identity.

Dynamo graphs are built per session-tree (a root session plus every descendant
linked by ``agent_context.parent_trajectory_id``; ``trajectory_id ==
session_id`` in real captures). Building each tree independently drops
cross-parent interval-order edges by construction while preserving within-tree
edges (parent<->subagent + intra-session). The differential oracle is the OLD
single global build, reached by calling the per-tree seam
(``_build_graph_from_chains``) with the WHOLE chain set.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import (
    _build_graph_from_chains,
    _Chain,
    _collect_chains,
    from_dynamo_trace,
    group_chains_into_trees,
)
from aiperf.dataset.graph.adapters.shared.content import resolve_effective_root_seed
from aiperf.dataset.graph.adapters.shared.idle_gap import DEFAULT_IDLE_GAP_CAP_SECONDS
from aiperf.dataset.graph.segment_ir.pool import SegmentPool

_MAX_DEPTH = Environment.DYNAMO.MAX_SUBAGENT_DEPTH

# --- fixture builders (REAL agent_context field shape) --------------------


def _ctx(*, sid: str, parent: str | None, use_trajectory: bool) -> dict:
    """Agent context in the real capture shape (``trajectory_id == session_id``).

    ``use_trajectory=True`` mirrors real captures: ``trajectory_id`` is set to
    the session id and a subagent's ``parent_trajectory_id`` names the parent's
    trajectory id (== parent session id). ``use_trajectory=False`` exercises the
    legacy ``parent_session_id`` fallback (older / hand-authored traces).
    """
    ctx: dict = {"session_id": sid}
    if use_trajectory:
        ctx["trajectory_id"] = sid
        if parent is not None:
            ctx["parent_trajectory_id"] = parent
    elif parent is not None:
        ctx["parent_session_id"] = parent
    return ctx


def _re(
    *,
    ts: int,
    sid: str,
    hashes: list[int],
    ilen: int,
    parent: str | None = None,
    otok: int = 8,
    bs: int = 16,
    use_trajectory: bool = True,
) -> dict:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": _ctx(sid=sid, parent=parent, use_trajectory=use_trajectory),
        "request": {
            "request_id": f"r-{sid}-{ts}",
            "model": "m",
            "input_tokens": ilen,
            "output_tokens": otok,
            "cached_tokens": 0,
            "ttft_ms": 10.0,
            "replay": {
                "trace_block_size": bs,
                "input_length": ilen,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _two_independent_trees() -> list[dict]:
    """Two parentless roots with DISJOINT hashes and DISJOINT intervals.

    A single global build finished-before-links p1's last turn to p2's first
    (p1 all ends by 1100, p2 starts at 5000); a per-tree build must not.
    """
    return [
        _re(ts=1000, sid="p1", hashes=[1, 2], ilen=32),
        _re(ts=1100, sid="p1", hashes=[1, 2, 3], ilen=48),
        _re(ts=5000, sid="p2", hashes=[4, 5], ilen=32),
        _re(ts=5100, sid="p2", hashes=[4, 5, 6], ilen=48),
    ]


def _oracle_global_build(chains: dict[str, _Chain], *, seed: int) -> tuple:
    """The OLD single global build: the per-tree seam over the WHOLE chain set."""
    pool = SegmentPool()
    graph, tags = _build_graph_from_chains(
        chains,
        pool=pool,
        content_root_seed=resolve_effective_root_seed(seed),
        block_size=16,
        idle_gap_cap_seconds=DEFAULT_IDLE_GAP_CAP_SECONDS,
        content_tokenizer=None,
        prompt_corpus="coding",
        release_replay=False,
    )
    return graph, pool, tags


def _agent_prefix(node_id: str) -> str:
    """The session id (drops the ``:{turn}`` suffix from ``{session_id}:{k}``)."""
    return node_id.rsplit(":", 1)[0]


def _nonterminal_edges(pb) -> list[tuple[str, str]]:
    return [
        (e.source, e.target)
        for e in pb.graph.edges
        if e.source not in ("START", "END") and e.target not in ("START", "END")
    ]


# --- 1. group_chains_into_trees (pure) ------------------------------------


def _mk_chains(links: dict[str, str | None]) -> dict[str, _Chain]:
    return {
        sid: _Chain(sid, parent_session_id=parent, turns=[])
        for sid, parent in links.items()
    }


def _plink(chains: dict[str, _Chain]) -> dict[str, str]:
    return {
        sid: c.parent_session_id
        for sid, c in chains.items()
        if c.parent_session_id is not None
    }


def test_group_parent_with_two_subagents_forms_one_tree() -> None:
    chains = _mk_chains({"p": None, "s1": "p", "s2": "p"})
    trees = group_chains_into_trees(chains, _plink(chains))
    assert len(trees) == 1
    assert set(trees[0]) == {"p", "s1", "s2"}


def test_group_nested_subagent_chain_forms_one_tree() -> None:
    # p -> s1 -> s2 (grandchild): the fixpoint walk unions all three.
    chains = _mk_chains({"p": None, "s1": "p", "s2": "s1"})
    trees = group_chains_into_trees(chains, _plink(chains))
    assert len(trees) == 1
    assert set(trees[0]) == {"p", "s1", "s2"}


def test_group_two_unrelated_roots_form_two_trees() -> None:
    chains = _mk_chains({"a": None, "b": None})
    trees = group_chains_into_trees(chains, _plink(chains))
    assert [sorted(t) for t in trees] == [["a"], ["b"]]


def test_group_external_parent_is_its_own_root() -> None:
    # 'a' points at 'x' which is NOT in the chain set -> 'a' is a forest root.
    chains = _mk_chains({"a": "x", "b": None})
    trees = group_chains_into_trees(chains, {"a": "x"})
    assert [sorted(t) for t in trees] == [["a"], ["b"]]


def test_group_cycle_guard_terminates_and_covers_all() -> None:
    # A -> B -> A: _guard_chain_forest rejects this upstream; the grouping guard
    # must still terminate and partition every session exactly once.
    chains = _mk_chains({"a": "b", "b": "a"})
    trees = group_chains_into_trees(chains, {"a": "b", "b": "a"})
    covered = sorted(sid for t in trees for sid in t)
    assert covered == ["a", "b"]


# --- 2. end-to-end linkage from the REAL field shape ----------------------


def test_parent_trajectory_id_links_subagent_into_one_tree(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32),
            _re(ts=2000, sid="S", parent="P", hashes=[7, 8], ilen=32),
            _re(ts=3000, sid="P", hashes=[1, 2, 3], ilen=48),
        ],
    )
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    # Linkage came from parent_trajectory_id (parent_session_id was never set).
    assert chains["S"].parent_session_id == "P"
    trees = group_chains_into_trees(chains, _plink(chains))
    assert len(trees) == 1
    assert set(trees[0]) == {"P", "S"}


def test_two_unrelated_trajectories_form_two_trees(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="a", hashes=[1, 2], ilen=32),
            _re(ts=2000, sid="b", hashes=[4, 5], ilen=32),
        ],
    )
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    trees = group_chains_into_trees(chains, _plink(chains))
    assert [sorted(t) for t in trees] == [["a"], ["b"]]


def test_parent_session_id_fallback_when_no_trajectory(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32, use_trajectory=False),
            _re(
                ts=2000,
                sid="S",
                parent="P",
                hashes=[7, 8],
                ilen=32,
                use_trajectory=False,
            ),
        ],
    )
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    assert chains["S"].parent_session_id == "P"
    trees = group_chains_into_trees(chains, _plink(chains))
    assert len(trees) == 1
    assert set(trees[0]) == {"P", "S"}


# --- 3. cross-parent edge drop + within-tree edge preservation ------------


def test_independent_trees_have_zero_cross_session_edges(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _two_independent_trees())
    pb = from_dynamo_trace(p, content_root_seed=7)

    for s, t in _nonterminal_edges(pb):
        assert _agent_prefix(s) == _agent_prefix(t), f"cross-session edge {s} -> {t}"

    # The specific p1-finished-before-p2 edge a global build emits is absent.
    edges = {(e.source, e.target) for e in pb.graph.edges}
    assert ("p1:1", "p2:0") not in edges


def test_global_build_carries_the_cross_edge_tree_build_drops(tmp_path: Path) -> None:
    # Differential oracle: the OLD single global build DOES emit the cross-parent
    # edge, proving the tree build actively drops it (not that it never existed).
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _two_independent_trees())
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    graph, _pool, _tags = _oracle_global_build(chains, seed=7)
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("p1:1", "p2:0") in edges


def test_parent_subagent_within_tree_cross_session_edge_present(
    tmp_path: Path,
) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32),
            _re(ts=2000, sid="S", parent="P", hashes=[7, 8], ilen=32),
            _re(ts=3000, sid="P", hashes=[1, 2, 3], ilen=48),
        ],
    )
    pb = from_dynamo_trace(p, content_root_seed=7)
    edges = {(e.source, e.target) for e in pb.graph.edges}

    cross = [
        (s, t)
        for (s, t) in _nonterminal_edges(pb)
        if _agent_prefix(s) != _agent_prefix(t)
    ]
    assert cross, "within-tree parent<->subagent edge expected"
    # P's first turn finished (1000) before S started (2000): interval edge.
    assert ("P:0", "S:0") in edges


# --- 4. pool byte-identity vs the global build + determinism --------------


def test_tree_scoped_pool_identical_to_per_tree_seam_union(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _two_independent_trees())
    seed = 12345

    pb = from_dynamo_trace(p, content_root_seed=seed)

    # Reference: build each independent tree ALONE through the same per-tree seam,
    # then union. Trace-scoped response/tail seeds make each tree's content scope
    # on its OWN root, so the reference is this per-tree-scoped union -- NOT the
    # single global build (which scopes every tree by one root, giving different
    # response bytes for every non-root tree). Hashes are disjoint, so the union
    # is a plain by_id merge with no cross-tree collision.
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    trees = group_chains_into_trees(chains, _plink(chains))
    union: dict = {}
    for tree in trees:
        tree_chains = {sid: chains[sid] for sid in tree}
        _graph, tree_pool, _tags = _oracle_global_build(tree_chains, seed=seed)
        union.update(tree_pool.by_id)

    assert pb.segment_pool is not None
    assert len(union) > 0
    assert pb.segment_pool.by_id == union


def test_tree_scoped_build_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        _two_independent_trees()
        + [_re(ts=2000, sid="S", parent="p1", hashes=[1, 2, 9], ilen=48)],
    )

    a = from_dynamo_trace(p, content_root_seed=999)
    b = from_dynamo_trace(p, content_root_seed=999)

    assert a.segment_pool is not None and b.segment_pool is not None
    assert set(a.graph.nodes) == set(b.graph.nodes)
    assert set(a.segment_pool.by_id) == set(b.segment_pool.by_id)
