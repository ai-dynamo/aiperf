# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session-tree dynamo build: tree grouping, cross-parent edge drop, and pool byte-identity against the OLD single global build as differential oracle."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import (
    _build_graph_from_chains,
    _Chain,
    _collect_chains,
    from_dynamo_trace,
    group_chains_into_trees,
)
from aiperf.dataset.graph.adapters.shared.content import resolve_effective_root_seed
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.segment_trie.pool import SegmentPool

from .conftest import write_jsonl

_MAX_DEPTH = Environment.DYNAMO.MAX_SUBAGENT_DEPTH

# --- fixture builders (REAL agent_context field shape) --------------------


def _ctx(*, sid: str, parent: str | None, use_trajectory: bool) -> dict:
    """Agent context in the real capture shape: ``use_trajectory=True`` sets ``trajectory_id``/``parent_trajectory_id``, False exercises the legacy ``parent_session_id`` fallback."""
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
    """One ``request_end`` record carrying recorded replay hashes at block size ``bs``."""
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


def _two_independent_trees() -> list[dict]:
    """Two parentless roots with DISJOINT hashes and DISJOINT intervals, so a global build finished-before-links p1's last turn to p2's first but a per-tree build must not."""
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
        idle_gap_cap_seconds=60.0,
        content_tokenizer=None,
        prompt_corpus="coding",
        release_replay=False,
        max_osl=None,
    )
    return graph, pool, tags


def _agent_prefix(node_id: str) -> str:
    """The session id (drops the ``:{turn}`` suffix from ``{session_id}:{k}``)."""
    return node_id.rsplit(":", 1)[0]


def _nonterminal_edges(pb: ParsedGraph) -> list[tuple[str, str]]:
    """Edge endpoints excluding the synthetic START/END anchors."""
    return [
        (e.source, e.target)
        for e in pb.graph.edges
        if e.source not in ("START", "END") and e.target not in ("START", "END")
    ]


# --- 1. group_chains_into_trees (pure) ------------------------------------


def _mk_chains(links: dict[str, str | None]) -> dict[str, _Chain]:
    """Turnless ``_Chain`` stubs from a ``{session_id: parent_session_id}`` map."""
    return {
        sid: _Chain(sid, parent_session_id=parent, turns=[])
        for sid, parent in links.items()
    }


def _plink(chains: dict[str, _Chain]) -> dict[str, str]:
    """The parent-link map ``group_chains_into_trees`` expects, skipping roots."""
    return {
        sid: c.parent_session_id
        for sid, c in chains.items()
        if c.parent_session_id is not None
    }


@pytest.mark.parametrize(
    "links, expected_trees",
    [
        param(
            {"p": None, "s1": "p", "s2": "p"},
            [["p", "s1", "s2"]],
            id="parent_with_two_subagents_forms_one_tree",
        ),
        # p -> s1 -> s2 (grandchild): the fixpoint walk unions all three.
        param(
            {"p": None, "s1": "p", "s2": "s1"},
            [["p", "s1", "s2"]],
            id="nested_subagent_chain_forms_one_tree",
        ),
        param(
            {"a": None, "b": None},
            [["a"], ["b"]],
            id="two_unrelated_roots_form_two_trees",
        ),
        # 'a' points at 'x' which is NOT in the chain set -> 'a' is a forest root.
        param(
            {"a": "x", "b": None},
            [["a"], ["b"]],
            id="external_parent_is_its_own_root",
        ),
    ],
)  # fmt: skip
def test_group_chains_into_trees_partitions_the_forest(
    links: dict[str, str | None],
    expected_trees: list[list[str]],
) -> None:
    """The pure grouping walk unions each root with every descendant and keeps unrelated roots apart."""
    chains = _mk_chains(links)
    trees = group_chains_into_trees(chains, _plink(chains))
    assert [sorted(t) for t in trees] == expected_trees


def test_group_cycle_guard_terminates_and_covers_all() -> None:
    """A -> B -> A terminates and partitions every session exactly once (shape is unconstrained; ``_guard_chain_forest`` rejects cycles upstream)."""
    chains = _mk_chains({"a": "b", "b": "a"})
    trees = group_chains_into_trees(chains, {"a": "b", "b": "a"})
    covered = sorted(sid for t in trees for sid in t)
    assert covered == ["a", "b"]


# --- 2. end-to-end linkage from the REAL field shape ----------------------


def _parent_subagent_records(*, use_trajectory: bool, parent_turns: int) -> list[dict]:
    """Parent P interleaved with subagent S, linked through whichever parent-link field ``use_trajectory`` selects."""
    records = [
        _re(ts=1000, sid="P", hashes=[1, 2], ilen=32, use_trajectory=use_trajectory),
        _re(
            ts=2000,
            sid="S",
            parent="P",
            hashes=[7, 8],
            ilen=32,
            use_trajectory=use_trajectory,
        ),
    ]
    if parent_turns > 1:
        records.append(
            _re(
                ts=3000,
                sid="P",
                hashes=[1, 2, 3],
                ilen=48,
                use_trajectory=use_trajectory,
            )
        )
    return records


@pytest.mark.parametrize(
    "use_trajectory, parent_turns",
    [
        param(True, 2, id="parent_trajectory_id_real_capture_shape"),
        param(False, 1, id="parent_session_id_legacy_fallback"),
    ],
)  # fmt: skip
def test_parent_link_field_joins_subagent_into_one_tree(
    tmp_path: Path, use_trajectory: bool, parent_turns: int
) -> None:
    """Either parent-link field collapses parent plus subagent into a single session tree."""
    # With use_trajectory=True the linkage comes from parent_trajectory_id alone;
    # parent_session_id is never set on the record.
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        _parent_subagent_records(
            use_trajectory=use_trajectory, parent_turns=parent_turns
        ),
    )
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    assert chains["S"].parent_session_id == "P"
    trees = group_chains_into_trees(chains, _plink(chains))
    assert len(trees) == 1
    assert set(trees[0]) == {"P", "S"}


def test_two_unrelated_trajectories_form_two_trees(tmp_path: Path) -> None:
    """Two parentless sessions in one file collect into two separate trees."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _re(ts=1000, sid="a", hashes=[1, 2], ilen=32),
            _re(ts=2000, sid="b", hashes=[4, 5], ilen=32),
        ],
    )
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    trees = group_chains_into_trees(chains, _plink(chains))
    assert [sorted(t) for t in trees] == [["a"], ["b"]]


# --- 3. cross-parent edge drop + within-tree edge preservation ------------


def test_independent_trees_have_zero_cross_session_edges(tmp_path: Path) -> None:
    """Building each tree independently drops every cross-parent interval-order edge by construction."""
    p = write_jsonl(tmp_path / "trace.jsonl", _two_independent_trees())
    pb = from_dynamo_trace(p, content_root_seed=7)

    for s, t in _nonterminal_edges(pb):
        assert _agent_prefix(s) == _agent_prefix(t), f"cross-session edge {s} -> {t}"

    # The specific p1-finished-before-p2 edge a global build emits is absent.
    edges = {(e.source, e.target) for e in pb.graph.edges}
    assert ("p1:1", "p2:0") not in edges


def test_global_build_carries_the_cross_edge_tree_build_drops(tmp_path: Path) -> None:
    """Differential oracle: the OLD single global build DOES emit the cross-parent edge, proving the tree build actively drops it rather than it never existing."""
    p = write_jsonl(tmp_path / "trace.jsonl", _two_independent_trees())
    chains = _collect_chains(p, None, max_depth=_MAX_DEPTH)
    graph, _pool, _tags = _oracle_global_build(chains, seed=7)
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("p1:1", "p2:0") in edges


def test_parent_subagent_within_tree_cross_session_edge_present(
    tmp_path: Path,
) -> None:
    """Within-tree parent<->subagent edges survive the per-tree build; only cross-parent edges are dropped."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        _parent_subagent_records(use_trajectory=True, parent_turns=2),
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
    """The tree build's segment pool is byte-identical to the union of each tree built alone through the same per-tree seam."""
    p = write_jsonl(tmp_path / "trace.jsonl", _two_independent_trees())
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
    """Two parses of the same trace at the same seed produce the same node ids and the same pool segment ids."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        _two_independent_trees()
        + [_re(ts=2000, sid="S", parent="p1", hashes=[1, 2, 9], ilen=48)],
    )

    a = from_dynamo_trace(p, content_root_seed=999)
    b = from_dynamo_trace(p, content_root_seed=999)

    assert a.segment_pool is not None and b.segment_pool is not None
    assert set(a.graph.nodes) == set(b.graph.nodes)
    assert set(a.segment_pool.by_id) == set(b.segment_pool.by_id)


class TestDroppedTailRollup:
    """The dropped-partial-tail summary is ONE corpus-level line, not one per tree."""

    def test_single_rollup_line_over_multi_tree_corpus(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two trees with non-aligned prompts log one aggregated INFO, not two."""
        # ilen 40 at bs 16 = 2 whole blocks + an 8-token tail dropped per turn.
        records = [
            _re(ts=1000, sid="p1", hashes=[1, 2, 3], ilen=40),
            _re(ts=1100, sid="p1", hashes=[1, 2, 3, 4], ilen=56),
            _re(ts=5000, sid="p2", hashes=[5, 6, 7], ilen=40),
            _re(ts=5100, sid="p2", hashes=[5, 6, 7, 8], ilen=56),
        ]
        path = write_jsonl(tmp_path / "two_trees.jsonl", records)
        with caplog.at_level("INFO"):
            from_dynamo_trace(path, content_root_seed=7)
        rollups = [
            r.getMessage()
            for r in caplog.records
            if "partial block tail" in r.getMessage()
        ]
        assert len(rollups) == 1
        # 4 turns, 8 dropped tokens each.
        assert "4" in rollups[0] and "32" in rollups[0]
