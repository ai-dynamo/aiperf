# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral regression tests for the multi-run duration-estimate probe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.resolution.resolvers import DatasetResolver
from aiperf.plugin.enums import PhaseType


def _make_config(
    *, dataset_file: Path, phase_type: PhaseType, tmp_path: Path
) -> BenchmarkConfig:
    """Build a minimal BenchmarkConfig with a file dataset and one phase."""
    dataset: dict = {
        "name": "main",
        "type": "file",
        "path": str(dataset_file),
    }
    if phase_type == PhaseType.CONCURRENCY:
        dataset["format"] = "single_turn"
    else:
        dataset["format"] = "bailian_trace"
    phase: dict = {"name": "profiling", "type": phase_type}
    if phase_type == PhaseType.CONCURRENCY:
        phase.update({"requests": 10, "concurrency": 1})
    config = BenchmarkConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[dataset],
        phases=[phase],
    )
    config.artifacts.dir = tmp_path / "artifacts"
    return config


def test_estimate_fixed_schedule_with_timestamped_jsonl_returns(tmp_path) -> None:
    """Multi-run probe must satisfy fixed_schedule timing validation."""
    from aiperf.cli_runner._multi_run import _estimate_and_log_duration

    dataset_file = tmp_path / "trace.jsonl"
    dataset_file.write_text(
        '{"chat_id": 1, "parent_chat_id": -1, "timestamp": 0.0,'
        ' "input_length": 64, "output_length": 32,'
        ' "type": "text", "turn": 1, "hash_ids": [1]}\n'
        '{"chat_id": 2, "parent_chat_id": -1, "timestamp": 10.0,'
        ' "input_length": 64, "output_length": 32,'
        ' "type": "text", "turn": 1, "hash_ids": [2]}\n'
    )
    config = _make_config(
        dataset_file=dataset_file,
        phase_type=PhaseType.FIXED_SCHEDULE,
        tmp_path=tmp_path,
    )
    plan = BenchmarkPlan(configs=[config], trials=2)
    logger = MagicMock()

    with (
        patch.object(DatasetResolver, "_count_records_and_sessions") as mock_count,
        patch.object(DatasetResolver, "_count_dag_roots") as mock_roots,
    ):
        base_dir = _estimate_and_log_duration(plan, config, 2, logger)

    mock_count.assert_not_called()
    mock_roots.assert_not_called()
    assert base_dir.is_dir()


def test_estimate_concurrency_skips_dataset_resolver(tmp_path) -> None:
    """Concurrency-only probe must not pay for dataset file reads."""
    from aiperf.cli_runner._multi_run import _estimate_and_log_duration

    dataset_file = tmp_path / "trace.jsonl"
    dataset_file.write_text('{"text": "hello", "timestamp": 1.0}\n')
    config = _make_config(
        dataset_file=dataset_file, phase_type=PhaseType.CONCURRENCY, tmp_path=tmp_path
    )
    plan = BenchmarkPlan(configs=[config], trials=2)
    logger = MagicMock()

    with patch.object(
        DatasetResolver, "resolve", autospec=True
    ) as mock_dataset_resolve:
        base_dir = _estimate_and_log_duration(plan, config, 2, logger)

    mock_dataset_resolve.assert_not_called()
    assert base_dir.is_dir()
