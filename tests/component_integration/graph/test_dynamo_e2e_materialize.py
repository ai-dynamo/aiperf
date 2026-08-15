# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof the dynamo segment trie is both runnable through the worker materialize path and KV-dedup'd across turns."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.graph.graph_path_catalog import (
    build_catalog_context,
    node_ordinal_for,
)
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from aiperf.graph.worker_materialize import materialize_graph_request_unified
from tests.component_integration.graph.conftest import (
    dynamo_request_end,
    dynamo_run,
    dynamo_tool_event,
    write_dynamo_jsonl,
)

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
def dyn_subagent_fixture(tmp_path: Path) -> Path:
    """Multi-turn root session (4 nested-hash turns) plus a 2-turn subagent child."""
    # Both sessions record strictly NESTED replay hashes, so the LCP content trie
    # re-derives the same leading pool segments turn over turn -- a genuine
    # multi-element cross-turn shared prefix. The tool events between root turns
    # 3 and 4 land on turn 3's tool_breakdown metadata (no tool nodes exist).
    return write_dynamo_jsonl(
        tmp_path / "dyn_subagent.jsonl",
        [
            dynamo_request_end(ts=1000, session_id="parent", hashes=[111, 222]),
            dynamo_request_end(ts=1100, session_id="parent", hashes=[111, 222, 333]),
            dynamo_request_end(
                ts=1200, session_id="parent", hashes=[111, 222, 333, 444]
            ),
            dynamo_tool_event(
                ts=1220,
                session_id="parent",
                event_type="tool_start",
                tool_call_id=":subagent:child:invoke",
            ),
            dynamo_tool_event(
                ts=1260,
                session_id="parent",
                event_type="tool_end",
                tool_call_id=":subagent:child:invoke",
            ),
            dynamo_request_end(
                ts=1280,
                session_id="child",
                parent_session_id="parent",
                hashes=[900, 901],
            ),
            dynamo_request_end(
                ts=1300,
                session_id="child",
                parent_session_id="parent",
                hashes=[900, 901, 902],
            ),
            dynamo_request_end(
                ts=1400, session_id="parent", hashes=[111, 222, 333, 444, 555]
            ),
        ],
    )


@pytest.fixture
def dm(dyn_subagent_fixture: Path) -> DatasetManager:
    """A DatasetManager bound to the subagent fixture's resolved run."""
    return DatasetManager(run=dynamo_run(dyn_subagent_fixture), service_id="test")


def _reparse_pool_and_paths(dm: DatasetManager, fixture: Path):
    """Re-derive the parse plus segment pool through the same shared seam the build used."""
    # _configure_graph_workload builds the store but does not hand back the
    # parse; parse_graph_workload is deterministic from (run, path, env), so
    # re-running it yields the ground truth the store must round-trip.
    parsed = parse_graph_workload(dm.run, fixture)
    assert parsed.segment_pool is not None
    return parsed, parsed.segment_pool


def _prompt_path(node: LlmNode) -> list[str]:
    """The node's stamped prompt segment-id path."""
    return node.metadata["trie"]["prompt_segment_ids"]


def _shared_prefix_len(a: list[str], b: list[str]) -> int:
    """Number of leading segment ids two prompt paths have in common."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _session_turns(parsed, session_id: str) -> list[tuple[str, LlmNode]]:
    """The session's ``(node_id, node)`` turns in turn-index order."""
    turns = [
        (nid, node)
        for nid, node in parsed.graph.nodes.items()
        if isinstance(node, LlmNode) and nid.startswith(f"{session_id}:")
    ]
    return sorted(turns, key=lambda kv: int(kv[0].rsplit(":", 1)[-1]))


async def test_dynamo_build_store_worker_materialize_byte_equal(
    dm: DatasetManager,
    dyn_subagent_fixture: Path,
    mmap_base_path: Path,
) -> None:
    """Every dynamo node (root and child session) materializes byte-equal to the SegmentPool ground truth via the worker path."""
    benchmark_id = dm.run.benchmark_id

    convs = await dm._configure_graph_workload(dyn_subagent_fixture)
    assert convs.trace_ids, "dynamo build produced no graph traces"

    stamped, pool = _reparse_pool_and_paths(dm, dyn_subagent_fixture)
    ctx = build_catalog_context(stamped)
    trace = stamped.traces[0]

    checked = 0
    with GraphSegmentUnifiedClient(mmap_base_path, benchmark_id).open() as client:
        for node_id, node in stamped.graph.nodes.items():
            assert isinstance(node, LlmNode)
            ordinal = node_ordinal_for(ctx, trace.id, node_id)
            assert ordinal is not None, f"node {node_id!r} has no ordinal"
            req = materialize_graph_request_unified(
                client, trace.id, ordinal, "profiling"
            )
            assert req is not None, (
                f"node {node_id!r} (ord {ordinal}) did not materialize "
                "(GraphEnvelopeMissing)"
            )
            assert req["messages"] == pool.materialize(_prompt_path(node)), node_id
            checked += 1

    assert checked >= 6, "root (4 turns) + child (2 turns) must all materialize"


async def test_dynamo_multiturn_shares_multi_element_prefix_and_dedups(
    dm: DatasetManager,
    dyn_subagent_fixture: Path,
    mmap_base_path: Path,
) -> None:
    """Confirm the KV-cache dedup the segment trie exists for: across-turn shared prefix on the root session, internal dedup on the child, and pool-level sharing overall."""
    await dm._configure_graph_workload(dyn_subagent_fixture)

    stamped, _pool = _reparse_pool_and_paths(dm, dyn_subagent_fixture)

    root = _session_turns(stamped, "parent")
    assert len(root) == 4, "root session must contribute 4 turns"
    # Every later root turn extends the previous turn's whole path -- including
    # turn 4, which follows the tool + subagent window (the old channel-replay
    # mixed-K fan-in used to break this).
    for (_prev_id, prev), (_curr_id, curr) in zip(root, root[1:], strict=False):
        p_prev, p_curr = _prompt_path(prev), _prompt_path(curr)
        assert p_curr[: len(p_prev)] == p_prev, (
            f"turn {_curr_id} does not extend {_prev_id}: {p_prev} vs {p_curr}"
        )
    shared = _shared_prefix_len(_prompt_path(root[1][1]), _prompt_path(root[2][1]))
    assert shared >= 2, (
        "CONTENT-MODEL CONCERN: clean-chain root turns do NOT share a "
        f"multi-element segment prefix (shared={shared}); each turn would be "
        "one monolithic message with no cross-turn KV dedup."
    )

    # The child session ALSO dedups internally: turn 2 extends turn 1.
    child = _session_turns(stamped, "child")
    assert len(child) == 2, "fixture must produce a 2-turn child session"
    c1, c2 = _prompt_path(child[0][1]), _prompt_path(child[1][1])
    assert c2[: len(c1)] == c1, f"child turn 2 does not extend turn 1: {c1} vs {c2}"

    # Pool-level dedup: total unique segments < sum of every node's path length.
    all_paths = [
        _prompt_path(n) for n in stamped.graph.nodes.values() if isinstance(n, LlmNode)
    ]
    total = sum(len(p) for p in all_paths)
    unique = len({sid for p in all_paths for sid in p})
    assert unique < total, (
        f"CONTENT-MODEL CONCERN: no dedup -- {unique} unique segments == "
        f"{total} total across paths; the pool is not sharing any content."
    )

    # Emit the evidence when explicitly requested.
    if os.environ.get("AIPERF_TASK10_EVIDENCE"):
        for nid, node in root + child:
            print(f"\nEVIDENCE {nid} path: {_prompt_path(node)}")
        print(f"EVIDENCE shared(turn2,turn3) prefix len: {shared}")
        print(f"EVIDENCE unique={unique} total={total}")
