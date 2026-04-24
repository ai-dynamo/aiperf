# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MultiRunOrchestrator."""

import json
from unittest.mock import Mock, patch

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {
        "warmup": {
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
            "exclude_from_results": True,
        },
        "default": {"type": "concurrency", "requests": 100, "concurrency": 1},
    },
    "random_seed": 42,
}


def _make_config(**overrides) -> BenchmarkConfig:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _mock_subprocess_success(artifacts_path, metrics=None):
    """Write metrics file and return a mock subprocess result."""
    if metrics is None:
        metrics = {
            "time_to_first_token": {"unit": "ms", "avg": 150.0, "p99": 195.0},
            "request_count": {"unit": "requests", "avg": 10.0},
        }
    artifacts_path.mkdir(parents=True, exist_ok=True)
    with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
        json.dump(metrics, f)
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    return mock_result


class TestMultiRunOrchestrator:
    """Tests for MultiRunOrchestrator."""

    def test_execute_with_multiple_trials(self, tmp_path):
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=3)
        orchestrator = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            from pathlib import Path

            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with patch("subprocess.run", side_effect=mock_subprocess):
            results = orchestrator.execute(config, strategy)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_execute_handles_failures(self, tmp_path):
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=3)
        orchestrator = MultiRunOrchestrator(tmp_path)

        call_count = 0

        def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = Mock()
            if call_count == 2:
                mock_result.returncode = 1
                mock_result.stderr = "Connection timeout"
            else:
                config_path = args[0][4]
                from pathlib import Path

                art_dir = Path(config_path).parent
                return _mock_subprocess_success(art_dir)
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess):
            results = orchestrator.execute(config, strategy)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert "exit code 1" in results[1].error
        assert results[2].success is True

    def test_execute_with_cooldown(self, tmp_path):
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, cooldown_seconds=0.5)
        orchestrator = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            from pathlib import Path

            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with (
            patch("subprocess.run", side_effect=mock_subprocess),
            patch("time.sleep") as mock_sleep,
        ):
            results = orchestrator.execute(config, strategy)

        mock_sleep.assert_called_once_with(0.5)
        assert len(results) == 2

    def test_no_cooldown_after_last_run(self, tmp_path):
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1, cooldown_seconds=1.0)
        orchestrator = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            from pathlib import Path

            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with (
            patch("subprocess.run", side_effect=mock_subprocess),
            patch("time.sleep") as mock_sleep,
        ):
            results = orchestrator.execute(config, strategy)

        mock_sleep.assert_not_called()
        assert len(results) == 1

    def test_warmup_disabled_after_first_run(self, tmp_path):
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2)

        # Verify via strategy directly
        config1 = strategy.get_next_config(config, [])
        config2 = strategy.get_next_config(
            config, [RunResult(label="r0", success=True)]
        )

        assert "warmup" in config1.phases
        assert "warmup" not in config2.phases

    def test_get_aggregate_path(self, tmp_path):
        strategy = FixedTrialsStrategy(num_trials=1)
        assert strategy.get_aggregate_path(tmp_path) == tmp_path / "aggregate"


class TestExtractSummaryMetrics:
    """Tests for _extract_summary_metrics."""

    def test_extract_summary_metrics(self, tmp_path):
        orchestrator = MultiRunOrchestrator(tmp_path)
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "min": 100.0,
                "max": 200.0,
                "p50": 145.0,
                "p99": 195.0,
            },
            "request_throughput": {
                "unit": "requests/sec",
                "avg": 25.4,
                "min": 20.0,
                "max": 30.0,
            },
        }

        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        metrics = orchestrator._extract_summary_metrics(artifacts_path)

        assert "time_to_first_token" in metrics
        assert metrics["time_to_first_token"].avg == 150.5
        assert metrics["time_to_first_token"].p99 == 195.0
        assert metrics["time_to_first_token"].unit == "ms"
        assert "request_throughput" in metrics
        assert metrics["request_throughput"].avg == 25.4

    def test_extract_summary_metrics_missing_file(self, tmp_path):
        orchestrator = MultiRunOrchestrator(tmp_path)
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)
        assert orchestrator._extract_summary_metrics(artifacts_path) == {}

    def test_extract_summary_metrics_invalid_json(self, tmp_path):
        orchestrator = MultiRunOrchestrator(tmp_path)
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)
        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            f.write("{ invalid json }")
        assert orchestrator._extract_summary_metrics(artifacts_path) == {}

    def test_extract_summary_metrics_preserves_structure(self, tmp_path):
        orchestrator = MultiRunOrchestrator(tmp_path)
        artifacts_path = tmp_path / "run_0001"
        artifacts_path.mkdir(parents=True)

        json_content = {
            "time_to_first_token": {
                "unit": "ms",
                "avg": 150.5,
                "p50": 145.0,
                "p99": 195.0,
            },
        }
        with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
            json.dump(json_content, f)

        metrics = orchestrator._extract_summary_metrics(artifacts_path)
        assert isinstance(metrics["time_to_first_token"], JsonMetricResult)
        assert metrics["time_to_first_token"].p50 == 145.0
