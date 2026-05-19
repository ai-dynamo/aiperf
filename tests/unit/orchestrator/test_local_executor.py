# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalSubprocessExecutor."""

from unittest.mock import patch

import pytest

from aiperf.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun
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
            "name": "profiling",
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

    Under ``--profile-export-prefix`` parity, ``prefix='my_run'`` rebases
    the summary file to ``my_run.json`` (not ``profile_export_my_run.json``);
    the executor must consult ``ArtifactsConfig.profile_export_json_file``
    instead of hardcoding the old filename pattern.
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
    metrics_payload = {
        "request_count": {"unit": "requests", "avg": 100.0},
    }
    (tmp_path / "my_run.json").write_bytes(orjson.dumps(metrics_payload))

    executor = LocalSubprocessExecutor(base_dir=tmp_path)
    metrics = executor._extract_summary_metrics(run)

    assert "request_count" in metrics
    assert metrics["request_count"].avg == 100.0


def test_extract_summary_metrics_default_prefix(tmp_path):
    """Unset prefix resolves to the historical ``profile_export_aiperf.json``."""
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


@pytest.mark.asyncio
async def test_run_config_never_contains_plaintext_api_key(tmp_path):
    """run_config.json must never contain the plaintext api_key.

    The api_key is forwarded to the subprocess via the
    ``AIPERF_INJECTED_API_KEY`` env var; the on-disk config carries
    only ``<redacted>`` (via the EndpointConfig field_serializer).
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    cfg.endpoint.api_key = "sk-secret-1234567890"
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="redact-success",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    on_disk = _orjson.loads((tmp_path / "run_config.json").read_bytes())
    assert on_disk["cfg"]["endpoint"]["api_key"] != "sk-secret-1234567890"


@pytest.mark.asyncio
async def test_subprocess_receives_api_key_via_env(tmp_path):
    """The api_key reaches the subprocess via AIPERF_INJECTED_API_KEY env var.

    REGRESSION-LOCK: pre-fix the parent wrote a redacted ``run_config.json``
    (because ``EndpointConfig.api_key`` has ``when_used="json"`` field
    serializer) and the subprocess loaded ``<redacted>`` as the api_key,
    so production runs against an auth-validating server would 401.
    """
    cfg = _benchmark_config()
    cfg.endpoint.api_key = "sk-real-prod-key"
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="env-forward",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    assert env.get("AIPERF_INJECTED_API_KEY") == "sk-real-prod-key"


@pytest.mark.asyncio
async def test_no_env_var_when_api_key_unset(tmp_path):
    """When no api_key is configured, AIPERF_INJECTED_API_KEY is NOT set."""
    import os as _os

    cfg = _benchmark_config()  # no api_key
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="no-key",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    # Strip the env var from the parent's environment first so we test
    # the executor's own decision (not the parent's leftover state).
    _os.environ.pop("AIPERF_INJECTED_API_KEY", None)
    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    assert "AIPERF_INJECTED_API_KEY" not in env


def test_subprocess_runner_pops_and_restores_api_key(tmp_path, monkeypatch):
    """subprocess_runner.main consumes AIPERF_INJECTED_API_KEY (so children don't inherit it)
    and restores it onto the loaded BenchmarkRun before invoking the benchmark."""
    import orjson as _orjson

    cfg = _benchmark_config()
    # Write a redacted config to disk (mirrors what _prepare_run_artifacts does).
    cfg.endpoint.api_key = "<redacted>"
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    monkeypatch.setenv("AIPERF_INJECTED_API_KEY", "sk-real-prod-key")
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    captured: dict = {}

    def fake_run_single_benchmark(run):
        captured["api_key"] = run.cfg.endpoint.api_key
        captured["env_after_pop"] = (
            "AIPERF_INJECTED_API_KEY" in __import__("os").environ
        )

    with patch(
        "aiperf.cli_runner._run_single_benchmark", side_effect=fake_run_single_benchmark
    ):
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        subprocess_main()

    assert captured["api_key"] == "sk-real-prod-key"
    assert captured["env_after_pop"] is False, (
        "AIPERF_INJECTED_API_KEY must be popped before benchmark runs so "
        "child processes can't inherit the secret"
    )
