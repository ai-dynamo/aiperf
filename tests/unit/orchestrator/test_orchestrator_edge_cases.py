# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for MultiRunOrchestrator and FixedTrialsStrategy.

Focuses on:
- FixedTrialsStrategy seed handling, warmup removal, config isolation
- Cooldown timing
- Subprocess execution: config file, stderr truncation, exit codes
- Metrics extraction: valid/partial/empty/missing JSON, request_count=0
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import orjson

from aiperf.config import BenchmarkConfig
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.orchestrator.strategies import FixedTrialsStrategy

_MINIMAL_CONFIG_KWARGS: dict = {
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
        "profiling": {"type": "concurrency", "requests": 100, "concurrency": 1},
    },
    "random_seed": 42,
}


def _make_config(**overrides: object) -> BenchmarkConfig:
    kwargs = {**_MINIMAL_CONFIG_KWARGS, **overrides}
    return BenchmarkConfig(**kwargs)


def _mock_subprocess_success(artifacts_path, metrics=None):
    if metrics is None:
        metrics = {"request_count": {"unit": "requests", "avg": 10.0}}
    artifacts_path.mkdir(parents=True, exist_ok=True)
    with open(artifacts_path / "profile_export_aiperf.json", "w") as f:
        json.dump(metrics, f)
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    return mock_result


# ============================================================
# Strategy: seed handling, warmup, config isolation
# ============================================================


class TestFixedTrialsStrategyEdgeCases:
    """Verify seed handling, warmup removal, config isolation."""

    def test_seed_set_when_none_and_auto_true(self) -> None:
        config = _make_config(random_seed=None)
        strategy = FixedTrialsStrategy(num_trials=2, auto_set_seed=True)
        result = strategy.get_next_config(config, [])
        assert result.random_seed == FixedTrialsStrategy.DEFAULT_SEED

    def test_seed_preserved_when_already_set(self) -> None:
        config = _make_config(random_seed=123)
        strategy = FixedTrialsStrategy(num_trials=2, auto_set_seed=True)
        result = strategy.get_next_config(config, [])
        assert result.random_seed == 123

    def test_seed_not_set_when_auto_false(self) -> None:
        config = _make_config(random_seed=None)
        strategy = FixedTrialsStrategy(num_trials=2, auto_set_seed=False)
        result = strategy.get_next_config(config, [])
        assert result.random_seed is None

    def test_warmup_removed_on_subsequent_runs(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run0 = strategy.get_next_config(config, [])
        run1 = strategy.get_next_config(config, [RunResult(label="r0", success=True)])

        assert "warmup" in run0.phases
        assert "warmup" not in run1.phases
        assert "profiling" in run1.phases

    def test_warmup_kept_on_first_run(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run = strategy.get_next_config(config, [])
        assert "warmup" in run.phases

    def test_warmup_kept_when_flag_false(self) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=False)
        run = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert "warmup" in run.phases

    def test_warmup_removal_only_removes_excluded_phases(self) -> None:
        config = _make_config(
            phases={
                "warmup": {
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": 1,
                    "exclude_from_results": True,
                },
                "main": {"type": "concurrency", "requests": 50, "concurrency": 4},
                "cooldown_phase": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                },
            }
        )
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        run = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert "warmup" not in run.phases
        assert "main" in run.phases
        assert "cooldown_phase" in run.phases

    def test_config_deep_copy_when_seed_set(self) -> None:
        """Mutating returned config must not affect the original when auto_set_seed modifies it."""
        config = _make_config(random_seed=None)
        strategy = FixedTrialsStrategy(num_trials=2, auto_set_seed=True)
        result = strategy.get_next_config(config, [])
        result.random_seed = 999
        assert config.random_seed is None

    def test_config_deep_copy_when_warmup_disabled(self) -> None:
        """Warmup removal must not affect the original config."""
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=2, disable_warmup_after_first=True)
        result = strategy.get_next_config(config, [RunResult(label="r0", success=True)])
        assert "warmup" not in result.phases
        assert "warmup" in config.phases

    def test_run_label_format(self) -> None:
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_run_label(0) == "run_0001"
        assert strategy.get_run_label(1) == "run_0002"
        assert strategy.get_run_label(9) == "run_0010"

    def test_run_path(self, tmp_path: Path) -> None:
        strategy = FixedTrialsStrategy(num_trials=3)
        assert strategy.get_run_path(tmp_path, 0) == (
            tmp_path / "profile_runs" / "run_0001"
        )


# ============================================================
# Cooldown behavior
# ============================================================


class TestCooldownBehavior:
    def test_cooldown_between_every_run(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=5.0)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with (
            patch("subprocess.run", side_effect=mock_subprocess),
            patch("time.sleep") as mock_sleep,
        ):
            orch.execute(config, strategy)

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5.0)

    def test_zero_cooldown_no_sleep(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=3, cooldown_seconds=0.0)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with (
            patch("subprocess.run", side_effect=mock_subprocess),
            patch("time.sleep") as mock_sleep,
        ):
            orch.execute(config, strategy)

        mock_sleep.assert_not_called()

    def test_single_run_no_cooldown(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1, cooldown_seconds=5.0)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with (
            patch("subprocess.run", side_effect=mock_subprocess),
            patch("time.sleep") as mock_sleep,
        ):
            orch.execute(config, strategy)

        mock_sleep.assert_not_called()


# ============================================================
# Subprocess execution
# ============================================================


class TestSubprocessExecution:
    def test_run_config_json_written(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(art_dir)

        with patch("subprocess.run", side_effect=mock_subprocess):
            orch.execute(config, strategy)

        config_file = tmp_path / "profile_runs" / "run_0001" / "run_config.json"
        assert config_file.exists()
        data = orjson.loads(config_file.read_bytes())
        assert "benchmark_id" in data
        assert "cfg" in data

    def test_subprocess_stderr_truncated(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "X" * 5000

        with patch("subprocess.run", return_value=mock_result):
            results = orch.execute(config, strategy)

        assert results[0].success is False
        assert "Stderr:" in results[0].error
        assert len(results[0].error.split("Stderr: ", 1)[1]) <= 2000

    def test_subprocess_nonzero_exit(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "segfault"

        with patch("subprocess.run", return_value=mock_result):
            results = orch.execute(config, strategy)

        assert results[0].success is False
        assert "exit code 1" in results[0].error

    def test_subprocess_exception_caught(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        with patch("subprocess.run", side_effect=OSError("No such file")):
            results = orch.execute(config, strategy)

        assert results[0].success is False
        assert "No such file" in results[0].error


# ============================================================
# Metrics extraction
# ============================================================


class TestMetricsExtraction:
    def _write_metrics(self, path: Path, data: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(data, option=orjson.OPT_INDENT_2)
        )

    def test_valid_metrics_parsed(self, tmp_path: Path) -> None:
        art = tmp_path / "run"
        self._write_metrics(
            art,
            {
                "ttft": {"unit": "ms", "avg": 150.0, "p99": 195.0},
                "throughput": {"unit": "req/s", "avg": 30.0},
            },
        )

        orch = MultiRunOrchestrator(tmp_path)
        metrics = orch._extract_summary_metrics(art)

        assert "ttft" in metrics
        assert metrics["ttft"].avg == 150.0
        assert metrics["throughput"].avg == 30.0

    def test_non_metric_fields_skipped(self, tmp_path: Path) -> None:
        art = tmp_path / "run"
        self._write_metrics(
            art,
            {
                "ttft": {"unit": "ms", "avg": 100.0},
                "metadata": {"version": "1.0"},
                "name": "benchmark_xyz",
            },
        )

        orch = MultiRunOrchestrator(tmp_path)
        metrics = orch._extract_summary_metrics(art)
        assert list(metrics.keys()) == ["ttft"]

    def test_empty_json(self, tmp_path: Path) -> None:
        art = tmp_path / "run"
        self._write_metrics(art, {})
        orch = MultiRunOrchestrator(tmp_path)
        assert orch._extract_summary_metrics(art) == {}

    def test_missing_file(self, tmp_path: Path) -> None:
        art = tmp_path / "no_such_dir"
        art.mkdir(parents=True, exist_ok=True)
        orch = MultiRunOrchestrator(tmp_path)
        assert orch._extract_summary_metrics(art) == {}

    def test_request_count_zero_is_failure(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(
                art_dir, {"request_count": {"unit": "requests", "avg": 0.0}}
            )

        with patch("subprocess.run", side_effect=mock_subprocess):
            results = orch.execute(config, strategy)

        assert results[0].success is False
        assert "No requests completed" in results[0].error

    def test_all_error_requests(self, tmp_path: Path) -> None:
        config = _make_config()
        strategy = FixedTrialsStrategy(num_trials=1)
        orch = MultiRunOrchestrator(tmp_path)

        def mock_subprocess(*args, **kwargs):
            config_path = args[0][4]
            art_dir = Path(config_path).parent
            return _mock_subprocess_success(
                art_dir,
                {
                    "request_count": {"unit": "requests", "avg": 0.0},
                    "error_request_count": {"unit": "requests", "avg": 50.0},
                },
            )

        with patch("subprocess.run", side_effect=mock_subprocess):
            results = orch.execute(config, strategy)

        assert results[0].success is False
        assert "All 50 requests failed" in results[0].error
