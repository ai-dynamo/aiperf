# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A graph run must reclaim its per-benchmark build artifacts when it stops.

The unified segment store and the graph_meta sidecar are multi-GB on a real
trace corpus and are written under the system temp dir, which is a RAM-backed
tmpfs on many hosts. Only a FAILED build removed them (``store_build`` rmtrees
inside its ``except`` blocks); a successful run left both dirs behind forever,
so a handful of runs exhausted the device and the next run failed its dataset
configuration with ``OSError(28)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
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
async def test_stop_reclaims_graph_store_and_sidecar_dirs(store_root: Path) -> None:
    """Stopping the DatasetManager after a SUCCESSFUL graph build removes both per-benchmark artifact dirs."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(GRAPH_MIN),
            tokenizer_name="builtin",
        )
    )
    manager = DatasetManager(run=run, service_id="dm-reclaim-test")

    result = await manager._build_graph_store(GRAPH_MIN)

    store_dir = store_root / f"aiperf_graph_segments_{run.benchmark_id}"
    sidecar_dir = result.sidecar_path.parent
    assert store_dir.is_dir(), "build should have written the unified store dir"
    assert sidecar_dir.is_dir(), "build should have written the sidecar dir"

    await manager._cleanup()

    assert not store_dir.exists(), "stop must reclaim the unified segment store"
    assert not sidecar_dir.exists(), "stop must reclaim the graph_meta sidecar"


@pytest.mark.asyncio
async def test_stop_without_a_graph_build_is_a_noop(tmp_path: Path) -> None:
    """A DatasetManager that never built a graph reclaims nothing and does not raise."""
    run = make_run_from_cli(
        CLIConfig(model_names=["test-model"], tokenizer_name="builtin")
    )
    manager = DatasetManager(run=run, service_id="dm-reclaim-noop")

    await manager._cleanup()


@pytest.mark.asyncio
async def test_stop_reclaims_even_when_the_dirs_are_already_gone(
    store_root: Path,
) -> None:
    """Reclaim is idempotent: an externally removed store dir must not fail the stop path."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(GRAPH_MIN),
            tokenizer_name="builtin",
        )
    )
    manager = DatasetManager(run=run, service_id="dm-reclaim-idempotent")

    await manager._build_graph_store(GRAPH_MIN)
    await manager._cleanup()
    await manager._cleanup()
