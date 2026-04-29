# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topology pathological tests for ``DagJsonlLoader._compute_depths``.

The happy-path tests in ``test_dag_jsonl_plugin.py`` cover the simple
1-root-2-fork shape from ``small.dag.jsonl``. This file pins the BFS
depth computation against the few topology shapes that catch real
regressions:

- Linear chain (depth must accumulate, not flat-stamp at 1).
- Wide fanout (BFS doesn't drop sibling children).
- Balanced tree of depth 2 (BFS frontier expansion at multiple levels).
- Multi-root forest (BFS seeds from every root, not just the first).
- Single conversation, no forks (degenerate edge — depth 0, no orphan).

Skipped as overkill: very large fanouts (≥ 50), very deep chains (≥
50), branching > 2 (covered by branching=2), unicode session ids,
unreachable-orphan backstop (requires monkey-patching the BFS to
trigger; the validators upstream make it unreachable in practice).
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "dag.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return path


def _conv(sid: str, *, forks: list[str] | None = None, n_turns: int = 1) -> dict:
    """Build a minimal valid conversation entry. ``forks`` lands on the
    LAST turn (FORK rule: forks declared on the terminal turn only).
    """
    turns: list[dict] = [
        {"messages": [{"role": "user", "content": f"{sid}-t{i}"}]}
        for i in range(n_turns)
    ]
    if forks:
        turns[-1]["forks"] = forks
    return {"session_id": sid, "turns": turns}


def test_linear_chain_accumulates_depth(tmp_path: Path) -> None:
    """Linear chain ``r -> c1 -> c2 -> c3`` must produce depths
    0/1/2/3. A bug that flat-stamps every fork target at depth 1 would
    show up here.
    """
    path = _write_jsonl(
        tmp_path,
        [
            _conv("r", forks=["c1"], n_turns=2),
            _conv("c1", forks=["c2"], n_turns=2),
            _conv("c2", forks=["c3"], n_turns=2),
            _conv("c3", n_turns=1),
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert by_id["r"].agent_depth == 0
    assert by_id["c1"].agent_depth == 1
    assert by_id["c2"].agent_depth == 2
    assert by_id["c3"].agent_depth == 3


@pytest.mark.parametrize("fanout", [1, 10])
def test_wide_fanout_all_children_depth_one(tmp_path: Path, fanout: int) -> None:
    """Single root forking to N children — every child must land at
    depth 1. Catches a BFS bug that drops siblings after the first."""
    child_ids = [f"c{i}" for i in range(fanout)]
    path = _write_jsonl(
        tmp_path,
        [_conv("r", forks=child_ids)] + [_conv(cid) for cid in child_ids],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert by_id["r"].agent_depth == 0
    for cid in child_ids:
        assert by_id[cid].agent_depth == 1


def test_balanced_tree_two_levels(tmp_path: Path) -> None:
    """Root with 2 forks, each forking to 2 grandchildren. Pins that
    BFS frontier expansion is correct across multiple levels — every
    grandchild lands at depth 2 regardless of which parent it came
    through.
    """
    lines: list[dict] = [_conv("r", forks=["c0", "c1"])]
    for ci in ["c0", "c1"]:
        grandchild_ids = [f"g_{ci}_0", f"g_{ci}_1"]
        lines.append(_conv(ci, forks=grandchild_ids, n_turns=2))
        for gj in grandchild_ids:
            lines.append(_conv(gj))
    path = _write_jsonl(tmp_path, lines)
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert by_id["r"].agent_depth == 0
    for ci in ["c0", "c1"]:
        assert by_id[ci].agent_depth == 1
        for j in (0, 1):
            assert by_id[f"g_{ci}_{j}"].agent_depth == 2


def test_multi_root_forest_seeds_from_every_root(tmp_path: Path) -> None:
    """Multiple independent root trees in one file. The BFS must seed
    from EVERY root; a bug that grabs only the first root would
    leave the second tree's children unreached and trigger the
    "missed conversations" backstop.
    """
    path = _write_jsonl(
        tmp_path,
        [
            _conv("r1", forks=["c1"]),
            _conv("c1"),
            _conv("r2", forks=["c2"]),
            _conv("c2"),
        ],
    )
    by_id = {c.session_id: c for c in DagJsonlLoader(path).load()}
    assert by_id["r1"].agent_depth == 0
    assert by_id["r2"].agent_depth == 0
    assert by_id["c1"].agent_depth == 1
    assert by_id["c2"].agent_depth == 1


def test_all_roots_no_forks_loads_cleanly(tmp_path: Path) -> None:
    """Every conversation is a root (no forks). Topology has no edges,
    BFS frontier never expands. All depths must be 0 and the orphan
    backstop must NOT fire."""
    path = _write_jsonl(tmp_path, [_conv(f"r{i}") for i in range(5)])
    convs = DagJsonlLoader(path).load()
    assert len(convs) == 5
    assert all(c.agent_depth == 0 for c in convs)
