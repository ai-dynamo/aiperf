# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphStoreBuilder.build`` end-to-end contract on a real run.

The builder owns the ONE graph store build the DatasetManager used to inline:
given a run and a graph path it must build the unified segment store at the
run's store root, write the mandatory graph_meta sidecar, and hand back the
graph facet + both locations as a ``GraphStoreBuildResult``. These tests drive
a REAL ``GraphStoreBuilder(run)`` (``make_run_from_cli``, no stubs) over the
weka_min fixture and check every field of that result against the on-disk
artifacts.
"""

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

WEKA_MIN = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the store root to tmp_path so build artifacts land in a known dir."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_build_weka_min_returns_facet_sidecar_and_openable_store(
    store_root: Path,
) -> None:
    """A real weka_min build yields the trace universe, the sidecar, and a store
    the worker-side unified client can open and read node envelopes from."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(WEKA_MIN),
            tokenizer_name="builtin",
        )
    )

    result = await GraphStoreBuilder(run).build(WEKA_MIN)

    assert result.base_path == store_root
    assert result.facet.trace_ids == ["trace_03_n3"]
    assert result.facet.prefix_cache_by_trace == {
        "trace_03_n3": {
            "trace_03_n3:0": [0, 2],
            "trace_03_n3:1": [2, 3],
            "trace_03_n3:2": [3, 4],
        }
    }
    assert result.sidecar_path == sidecar_path_for(store_root, run.benchmark_id)
    assert result.sidecar_path.exists()
    with GraphSegmentUnifiedClient(store_root, run.benchmark_id).open() as client:
        assert "trace_03_n3" in client._node_offsets
        assert client.get_node_envelope("trace_03_n3", 0, "profiling")


@pytest.mark.asyncio
async def test_build_without_sidecar_write_raises(
    store_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain that never records the sidecar path is a hard build failure.

    The sidecar is mandatory for graph runs (the TimingManager only ingests
    it), so ``build()`` enforces the recorded path instead of handing back a
    result with a dead location.
    """
    del store_root
    run = make_run_from_cli(
        CLIConfig(model_names=["test-model"], input_file=str(WEKA_MIN))
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
        await builder.build(WEKA_MIN)
