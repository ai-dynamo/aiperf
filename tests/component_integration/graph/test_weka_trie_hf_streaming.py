# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF-streaming trie build end-to-end through DatasetManager (Task T8-hf).

Drives the REAL :meth:`DatasetManager._configure_graph_workload` HF-streaming
build path -- the ``--public-dataset`` path that streams per row instead of
parsing the whole corpus eagerly -- with the segment-trie IR enabled. The HF
``_load_hf_rows`` is monkeypatched to return synthetic weka rows so the path
runs hermetically (no network), exactly as a real HF corpus would feed it.

Asserts the streaming build produces the ONE interned unified store
(``aiperf_graph_segments_<id>/``), and that worker
``materialize_graph_request_unified`` over the streamed store reproduces the
SAME prompt as the in-memory pool ground truth (the failure mode this task
closes: the streaming path previously dropped the pool and served empty
prompts).
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

# An HF-shaped id carrying the weka marker so ``_looks_like_hf_dataset_id``
# routes the build through the STREAMING path. It is not a real path on disk.
_HF_ID = "synthetic/cc-traces-weka-streamtest"

_ROWS = [
    {
        "id": "trace_alpha",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 180,
                "out": 25,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 0.8,
                "think_time": 0.0,
            },  # noqa: E501
            {
                "t": 1.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 240,
                "out": 30,
                "hash_ids": [1, 2, 3],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 0.9,
                "think_time": 0.1,
            },  # noqa: E501
        ],
    },
    {
        "id": "trace_beta",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 180,
                "out": 25,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "tool_use",
                "api_time": 0.8,
                "think_time": 0.0,
            },  # noqa: E501
            {
                "t": 1.0,
                "type": "subagent",
                "agent_id": "agent_001",
                "subagent_type": "Explore",
                "duration_ms": 4000,
                "total_tokens": 600,
                "tool_use_count": 1,
                "status": "completed",
                "models": ["claude-opus-4-5-20251101"],
                "tool_tokens": 0,
                "system_tokens": 0,
                "requests": [
                    {
                        "t": 1.2,
                        "type": "n",
                        "model": "claude-opus-4-5-20251101",
                        "in": 200,
                        "out": 30,
                        "hash_ids": [10, 11],
                        "input_types": ["text"],
                        "output_types": ["text"],
                        "stop": "end_turn",
                        "api_time": 0.9,
                        "think_time": 0.0,
                    },  # noqa: E501
                ],
            },
            {
                "t": 6.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 280,
                "out": 45,
                "hash_ids": [1, 2, 3, 4],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 1.3,
                "think_time": 0.5,
            },  # noqa: E501
        ],
    },
]


@pytest.fixture
def hf_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the HF row loader to return synthetic weka rows (no network)."""
    import aiperf.dataset.graph.adapters.weka.trace as weka_trace

    def _fake_load(repo_id, *, split, revision):  # noqa: ANN001, ANN202, ARG001
        for row in _ROWS:
            yield dict(row)

    monkeypatch.setattr(weka_trace, "_load_hf_rows", _fake_load)


@pytest.fixture
def trie_streaming_dm(
    mmap_base_path: Path,  # noqa: ARG001  # side-effect: patches MMAP_BASE_PATH
    hf_rows: None,  # noqa: ARG001  # side-effect: stubs the HF loader
) -> DatasetManager:
    """A DatasetManager pointed at the synthetic HF id."""
    cli_config = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        url="http://localhost:8000",
        input_file=_HF_ID,
    )
    run = make_run_from_cli(cli_config)
    return DatasetManager(run=run, service_id="test")


@pytest.mark.asyncio
@pytest.mark.component_integration
async def test_hf_streaming_trie_build_persists_and_materializes(
    trie_streaming_dm: DatasetManager,
    mmap_base_path: Path,
) -> None:
    """HF streaming build writes the unified store; worker prompts == pool truth."""
    dm = trie_streaming_dm
    benchmark_id = dm.run.benchmark_id

    convs = await dm._configure_graph_workload(Path(_HF_ID))
    assert convs.trace_ids, "streaming build must yield graph traces"

    # The unified store artifacts MUST exist (the gap this task closes).
    unified_dir = mmap_base_path / f"aiperf_graph_segments_{benchmark_id}"
    for name in ("content.blob", "content.idx", "nodes.blob", "nodes.idx"):
        assert (unified_dir / name).exists(), (
            f"streaming trie build wrote no unified store {name}"
        )

    parsed = parse_graph_workload(dm.run, Path(_HF_ID))
    assert parsed.segment_pool is not None
    pool = parsed.segment_pool

    client = GraphSegmentUnifiedClient(
        base_path=mmap_base_path, benchmark_id=benchmark_id
    ).open()
    try:
        checked = 0
        for trace in parsed.traces:
            llm_nodes = {
                nid: n
                for nid, n in parsed.graphs[trace.graph_ref].nodes.items()
                if isinstance(n, LlmNode)
            }
            ordinals = trie_node_ordinals(llm_nodes)
            for node_id, node in llm_nodes.items():
                path = node.metadata["trie"]["prompt_segment_ids"]
                req = materialize_graph_request_unified(
                    client, trace.id, ordinals[node_id], "profiling"
                )
                assert req is not None, f"node {node_id!r} has no persisted manifest"
                assert req["messages"], f"node {node_id!r} materialized EMPTY prompt"
                assert req["messages"] == pool.materialize(path), node_id
                checked += 1
        assert checked > 0
    finally:
        client.close()
