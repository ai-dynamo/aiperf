# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphStoreBuilder.build`` end-to-end contract on a real run."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph import store_build
from aiperf.dataset.graph.graph_meta_sidecar import sidecar_path_for
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from tests.unit.conftest import make_run_from_cli

GRAPH_MIN = (
    Path(__file__).parents[1]
    / "dataset"
    / "graph"
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the store root to tmp_path so build artifacts land in a known dir."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_build_graph_min_returns_facet_sidecar_and_openable_store(
    store_root: Path,
) -> None:
    """A real graph build yields the trace universe, the sidecar, and a store the worker-side unified client can open and read node envelopes from."""
    # sess_A spawns subagent session sess_B, which is replayed inside the root
    # trace -- hence one trace id but prefix-cache entries under both prefixes.
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(GRAPH_MIN),
            tokenizer_name="builtin",
        )
    )

    result = await GraphStoreBuilder(run).build(GRAPH_MIN)

    assert result.base_path == store_root
    assert result.facet.trace_ids == ["sess_A"]
    assert result.facet.prefix_cache_by_trace == {
        "sess_A": {
            "sess_A:0": [0, 2],
            "sess_A:1": [2, 2],
            "sess_A:2": [2, 2],
            "sess_B:0": [0, 2],
            "sess_B:1": [2, 2],
        }
    }
    assert result.sidecar_path == sidecar_path_for(store_root, run.benchmark_id)
    assert result.sidecar_path.exists()
    with GraphSegmentUnifiedClient(store_root, run.benchmark_id).open() as client:
        assert "sess_A" in client._node_offsets
        assert client.get_node_envelope("sess_A", 0)


@pytest.mark.asyncio
async def test_build_without_sidecar_write_raises(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain that never records the sidecar path is a hard build failure."""
    del store_root
    run = make_run_from_cli(
        CLIConfig(model_names=["test-model"], input_file=str(GRAPH_MIN))
    )
    builder = GraphStoreBuilder(run)
    monkeypatch.setattr(
        store_build, "publish_graph_loader_tokenizer_env", lambda run: None
    )
    monkeypatch.setattr(builder, "_prestart_loader_forkserver", lambda: None)

    async def _no_sidecar_drain(
        graph_path: Path, base_path: Path, fmt: str | None
    ) -> tuple:
        return {"t": {"n": 0}}, None

    monkeypatch.setattr(builder, "_build_graph_store_streaming", _no_sidecar_drain)
    monkeypatch.setattr(
        builder, "_build_graph_prefix_cache_by_trace", lambda prefix_source: {}
    )

    with pytest.raises(DatasetError, match="sidecar"):
        await builder.build(GRAPH_MIN)
