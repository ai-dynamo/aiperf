# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof the dynamo trie IR is RUNNABLE + DEDUPS.

Drives the REAL build path -- ``DatasetManager._configure_graph_workload`` ->
``parse_graph_workload`` (dynamo lowers through the shared LCP segment-trie
core at parse time) -> ``build_unified_trie_store_from_payloads`` ->
``GraphSegmentUnifiedBackingStore`` -- on a MULTI-TURN + SUBAGENT dynamo
trace, then materializes every node the SAME way the worker does
(``materialize_graph_request_unified`` reading the interned int-handle
manifests) and asserts the materialized messages equal the
:class:`SegmentPool` ground truth (``pool.materialize(path)``). No
``GraphEnvelopeMissing``, no empty prompts.

Beyond byte-parity it validates the KV-cache dedup the trie IR exists for:
the root session's strictly-nested replay hashes make every later turn's
``prompt_segment_ids`` EXTEND the earlier turn's (shared segment-id prefix),
and the child session dedups internally the same way. The old channel-replay
mixed-K fan-in break is gone -- the flat lowering shares the prefix across
ALL root turns, including the one after the tool + subagent window.
"""

from __future__ import annotations

import os
from pathlib import Path

import orjson
import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.graph.graph_path_catalog import (
    build_catalog_context,
    node_ordinal_for,
)
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from aiperf.graph.worker_materialize import materialize_graph_request_unified
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_SEED = 1234


def _request_end(
    *,
    ts: int,
    session_id: str,
    parent_session_id: str | None = None,
    hashes: list[int] | None = None,
) -> dict:
    """One ``dynamo.request.trace.v1`` ``request_end`` with recorded replay hashes.

    One 16-token block per replay hash so ``input_length == 16 * len(hashes)``
    stays block-aligned for the parse-time alignment gate at ``block_size=16``
    (``(n-1)*16 < input_length <= n*16``).
    """
    ctx: dict = {"session_id": session_id}
    if parent_session_id is not None:
        ctx["parent_session_id"] = parent_session_id
    input_length = 16 * len(hashes) if hashes else 32
    req: dict = {
        "request_id": f"{session_id}-{ts}",
        "model": "m",
        "input_tokens": input_length,
        "output_tokens": 16,
        "cached_tokens": 0,
    }
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": 16,
            "input_length": input_length,
            "input_sequence_hashes": hashes,
        }
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": req,
    }


def _tool_event(
    *, ts: int, session_id: str, event_type: str, tool_call_id: str
) -> dict:
    tool: dict = {"tool_call_id": tool_call_id, "tool_class": "search"}
    if event_type == "tool_end":
        tool["duration_ms"] = 40.0
        tool["status"] = "succeeded"
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "event_time_unix_ms": ts,
        "event_source": "harness",
        "agent_context": {"session_id": session_id},
        "tool": tool,
    }


@pytest.fixture
def dyn_subagent_fixture(tmp_path: Path) -> Path:
    """Multi-turn root session (4 nested-hash turns) + a 2-turn subagent child.

    The root ``parent`` records strictly NESTED replay hashes
    (``[111,222]`` prefix-of ``[..,333]`` prefix-of ``[..,444]`` prefix-of
    ``[..,555]``), so the LCP content trie re-derives the SAME leading pool
    segments turn over turn -- a genuine multi-element cross-turn shared
    prefix. The child session (``child``, nested hashes ``[900,901]`` ->
    ``[900,901,902]``) flattens into the same graph and dedups internally.
    Tool events between root turns 3 and 4 land on turn 3's tool_breakdown
    metadata (the trie IR has no tool nodes).
    """
    p = tmp_path / "dyn_subagent.jsonl"
    records = [
        _request_end(ts=1000, session_id="parent", hashes=[111, 222]),
        _request_end(ts=1100, session_id="parent", hashes=[111, 222, 333]),
        _request_end(ts=1200, session_id="parent", hashes=[111, 222, 333, 444]),
        _tool_event(
            ts=1220,
            session_id="parent",
            event_type="tool_start",
            tool_call_id=":subagent:child:invoke",
        ),
        _tool_event(
            ts=1260,
            session_id="parent",
            event_type="tool_end",
            tool_call_id=":subagent:child:invoke",
        ),
        _request_end(
            ts=1280, session_id="child", parent_session_id="parent", hashes=[900, 901]
        ),
        _request_end(
            ts=1300,
            session_id="child",
            parent_session_id="parent",
            hashes=[900, 901, 902],
        ),
        _request_end(ts=1400, session_id="parent", hashes=[111, 222, 333, 444, 555]),
    ]
    with p.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return p


def _build_dm(fixture: Path) -> DatasetManager:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
            input_file=str(fixture),
            random_seed=_SEED,
        )
    )
    return DatasetManager(run=run, service_id="test")


def _reparse_pool_and_paths(dm: DatasetManager, fixture: Path):
    """Re-derive the parse + pool through the SAME shared seam the build used.

    ``_configure_graph_workload`` builds the store but does not hand back the
    parse. ``parse_graph_workload`` is deterministic from ``(run, path, env)``,
    so re-running it yields the SAME pool + ``prompt_segment_ids`` per node --
    the ground truth the store must round-trip.
    """
    parsed = parse_graph_workload(dm.run, fixture)
    assert parsed.segment_pool is not None
    return parsed, parsed.segment_pool


def _prompt_path(node: LlmNode) -> list[str]:
    return node.metadata["trie"]["prompt_segment_ids"]


def _shared_prefix_len(a: list[str], b: list[str]) -> int:
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
    dyn_subagent_fixture: Path,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every dynamo node (root + child session) materializes byte-equal via the
    worker path."""
    dm = _build_dm(dyn_subagent_fixture)
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
    dyn_subagent_fixture: Path,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONFIRM the KV-cache dedup the trie IR exists for -- across-turn shared
    prefix on the root session, internal dedup on the child session, and
    pool-level sharing overall."""
    dm = _build_dm(dyn_subagent_fixture)
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
