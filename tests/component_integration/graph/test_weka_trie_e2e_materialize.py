# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof the weka segment-trie IR is RUNNABLE.

The build plane parses a weka trace into a trie ``ParsedGraph`` (every
``LlmNode`` carrying a ``prompt_segment_ids`` pool path) and drains the
:class:`SegmentPool` plus every node's manifest into the ONE interned unified
store (``aiperf_graph_segments_<id>/``) -- the sole trie store shape.

This test drives that REAL build+persist path through
:meth:`DatasetManager._configure_graph_workload`, then materializes a node the
SAME way the worker does -- via
:func:`aiperf.graph.worker_materialize.materialize_graph_request_unified`
reading the persisted unified store -- and asserts the materialized prompt is
BYTE-EQUAL to ``pool.materialize(node.prompt_segment_ids)``. That is the
deliverable's proof the trie graph runs with correct prompts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.segment_ir.store_builder import trie_node_ordinals
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from aiperf.graph.worker_materialize import materialize_graph_request_unified
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli

_FIX = Path(__file__).parents[2] / "unit" / "graph" / "fixtures" / "weka_subagent.json"


@pytest.fixture
def trie_dataset_manager(
    mmap_base_path: Path,  # noqa: ARG001  # side-effect: patches MMAP_BASE_PATH
) -> DatasetManager:
    """A DatasetManager pointed at the subagent fixture."""
    cli_config = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        url="http://localhost:8000",
        input_file=str(_FIX),
    )
    run = make_run_from_cli(cli_config)
    return DatasetManager(run=run, service_id="test")


@pytest.mark.asyncio
@pytest.mark.component_integration
async def test_trie_build_persists_and_worker_materializes_byte_equal(
    trie_dataset_manager: DatasetManager,
    mmap_base_path: Path,
) -> None:
    """Every trie node's worker-materialized prompt == ``pool.materialize(path)``."""
    dm = trie_dataset_manager
    benchmark_id = dm.run.benchmark_id

    # REAL build+persist path: writes the ONE unified store (content pool +
    # per-node interned manifests).
    convs = await dm._configure_graph_workload(_FIX)
    assert convs.trace_ids, "build must yield at least one graph trace"

    # The unified store artifacts exist under the shared base path.
    unified_dir = mmap_base_path / f"aiperf_graph_segments_{benchmark_id}"
    for name in ("content.blob", "content.idx", "nodes.blob", "nodes.idx"):
        assert (unified_dir / name).exists(), f"unified store missing {name}"

    # Re-parse to recover the in-memory trie graph + pool the build persisted
    # (deterministic from the same run), so we know each node's expected
    # prompt path and the SegmentPool ground truth.
    parsed = parse_graph_workload(dm.run, _FIX)
    assert parsed.segment_pool is not None, "trie parse must surface a SegmentPool"
    pool = parsed.segment_pool

    trace = parsed.traces[0]
    llm_nodes = {
        nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)
    }
    assert llm_nodes, "trie graph must have LlmNodes"
    ordinals = trie_node_ordinals(llm_nodes)

    client = GraphSegmentUnifiedClient(
        base_path=mmap_base_path, benchmark_id=benchmark_id
    ).open()
    try:
        checked = 0
        for node_id, node in llm_nodes.items():
            path = node.metadata["trie"]["prompt_segment_ids"]
            expected = pool.materialize(path)

            req = materialize_graph_request_unified(
                client, trace.id, ordinals[node_id], "profiling"
            )
            assert req is not None, f"node {node_id!r} has no persisted manifest"
            # Byte-equal: the worker walked the persisted unified store along
            # the interned handle path and rebuilt the exact pool prompt.
            assert req["messages"] == expected, node_id
            checked += 1
        assert checked == len(llm_nodes)
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.component_integration
async def test_trie_node_prompt_matches_pool_materialize_for_subagent_turn(
    trie_dataset_manager: DatasetManager,
    mmap_base_path: Path,
) -> None:
    """A multi-segment (deepest-path) node round-trips byte-for-byte through persist."""
    dm = trie_dataset_manager
    benchmark_id = dm.run.benchmark_id
    await dm._configure_graph_workload(_FIX)

    parsed = parse_graph_workload(dm.run, _FIX)
    pool = parsed.segment_pool
    trace = parsed.traces[0]
    llm_nodes = {
        nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)
    }
    ordinals = trie_node_ordinals(llm_nodes)

    # The node with the longest prompt path exercises the deepest shared-prefix
    # chain -- the hardest case for byte-equality across the persist boundary.
    deepest_id = max(
        llm_nodes,
        key=lambda nid: len(llm_nodes[nid].metadata["trie"]["prompt_segment_ids"]),
    )
    path = llm_nodes[deepest_id].metadata["trie"]["prompt_segment_ids"]
    assert len(path) >= 1

    client = GraphSegmentUnifiedClient(
        base_path=mmap_base_path, benchmark_id=benchmark_id
    ).open()
    try:
        req = materialize_graph_request_unified(
            client, trace.id, ordinals[deepest_id], "profiling"
        )
    finally:
        client.close()

    assert req is not None
    assert req["messages"] == pool.materialize(path)
