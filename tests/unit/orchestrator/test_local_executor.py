# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalSubprocessExecutor."""

from unittest.mock import patch

import pytest

from aiperf.config import BenchmarkConfig
from aiperf.config.benchmark import BenchmarkRun
from aiperf.orchestrator.local_executor import LocalSubprocessExecutor

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "warmup",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
            "exclude_from_results": True,
        },
        {
            "name": "default",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        },
    ],
}


def _benchmark_config() -> BenchmarkConfig:
    """Build a minimal valid BenchmarkConfig."""
    return BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)


@pytest.mark.asyncio
async def test_local_subprocess_executor_calls_subprocess(tmp_path):
    cfg = _benchmark_config()
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="run_0001",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        result = await executor.execute(run)

    mock_run.assert_called_once()
    # _extract_summary_metrics returns {} when no profile_export file exists,
    # so the executor classifies this as a non-success run.
    assert result.label == "run_0001"
    assert result.success is False  # no metrics file written -> "No metrics found"


def test_extract_summary_metrics_honors_artifacts_prefix(tmp_path):
    """Custom ``cfg.artifacts.prefix`` must be honored when locating the metrics file.

    Reproduces the --profile-export-prefix regression introduced by main's
    PR #699: the executor used to hardcode ``profile_export_aiperf.json``
    even when callers configured a different prefix, silently losing
    metrics for swept runs that customized the prefix.
    """
    import orjson

    cfg_kwargs = {**_MINIMAL_CONFIG_KWARGS, "artifacts": {"prefix": "my_run"}}
    cfg = BenchmarkConfig(**cfg_kwargs)
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="prefixed",
    )
    # Write a metrics file under the *custom* prefix; the default-prefix
    # filename should NOT exist.
    metrics_payload = {
        "request_count": {"unit": "requests", "avg": 100.0},
    }
    (tmp_path / "profile_export_my_run.json").write_bytes(orjson.dumps(metrics_payload))

    executor = LocalSubprocessExecutor(base_dir=tmp_path)
    metrics = executor._extract_summary_metrics(run)

    assert "request_count" in metrics
    assert metrics["request_count"].avg == 100.0


def test_extract_summary_metrics_default_prefix(tmp_path):
    """Default ``prefix='aiperf'`` still resolves to ``profile_export_aiperf.json``."""
    import orjson

    cfg = _benchmark_config()
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="default-prefix",
    )
    (tmp_path / "profile_export_aiperf.json").write_bytes(
        orjson.dumps({"request_count": {"unit": "requests", "avg": 5.0}})
    )

    executor = LocalSubprocessExecutor(base_dir=tmp_path)
    metrics = executor._extract_summary_metrics(run)

    assert metrics["request_count"].avg == 5.0
