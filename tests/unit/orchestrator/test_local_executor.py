# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalSubprocessExecutor."""

from pathlib import Path
from typing import Any
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


@pytest.mark.asyncio
async def test_run_config_never_contains_plaintext_sensitive_headers(
    tmp_path: Path,
) -> None:
    """run_config.json must redact credential-bearing header values.

    Mirrors ``test_run_config_never_contains_plaintext_api_key`` for the
    headers field. ``EndpointConfig.headers`` has a field_serializer that
    replaces sensitive values with ``<redacted>`` on every JSON dump, so
    the on-disk artifact is secret-free.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    cfg.endpoint.headers = {
        "Authorization": "Api-Key real-secret-value",
        "X-Trace-Id": "trace-001",
    }
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="headers-redact",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    on_disk = _orjson.loads((tmp_path / "run_config.json").read_bytes())
    on_disk_headers = on_disk["cfg"]["endpoint"]["headers"]
    assert on_disk_headers["Authorization"] != "Api-Key real-secret-value"
    # Non-sensitive headers must still round-trip through the JSON normally.
    assert on_disk_headers["X-Trace-Id"] == "trace-001"


@pytest.mark.asyncio
async def test_subprocess_receives_sensitive_headers_via_env(tmp_path: Path) -> None:
    """Sensitive headers reach the subprocess via AIPERF_INJECTED_HEADERS.

    REGRESSION-LOCK: pre-fix the parent wrote a redacted ``run_config.json``
    (because ``EndpointConfig.headers`` has a ``when_used="json"`` field
    serializer) and the subprocess loaded ``Authorization: <redacted>`` as
    the header value, so sweep runs against endpoints requiring custom auth
    schemes (e.g. ``Api-Key`` instead of ``Bearer``) would all 403.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    cfg.endpoint.headers = {
        "Authorization": "Api-Key real-secret-value",
        "X-Trace-Id": "trace-001",
    }
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="headers-env-forward",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    raw = env.get("AIPERF_INJECTED_HEADERS")
    assert raw is not None, "Sensitive headers must be forwarded via env var"
    forwarded = _orjson.loads(raw)
    # Only sensitive entries should ride the env channel; non-sensitive ones
    # round-trip through the JSON dump as-is.
    assert forwarded == {"Authorization": "Api-Key real-secret-value"}


@pytest.mark.asyncio
async def test_no_headers_env_var_when_no_sensitive_headers(tmp_path: Path) -> None:
    """AIPERF_INJECTED_HEADERS is absent when no sensitive headers are set."""
    import os as _os

    cfg = _benchmark_config()
    cfg.endpoint.headers = {"X-Trace-Id": "trace-001"}  # non-sensitive only
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="no-sensitive-headers",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    _os.environ.pop("AIPERF_INJECTED_HEADERS", None)
    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    assert "AIPERF_INJECTED_HEADERS" not in env


def test_subprocess_runner_pops_and_restores_sensitive_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """subprocess_runner.main consumes AIPERF_INJECTED_HEADERS and overlays
    the real values onto the loaded BenchmarkRun, leaving non-sensitive
    headers from run_config.json untouched."""
    import orjson as _orjson

    cfg = _benchmark_config()
    # Write a redacted config to disk (mirrors what _prepare_run_artifacts does).
    cfg.endpoint.headers = {
        "Authorization": "<redacted>",
        "X-Trace-Id": "trace-001",
    }
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    monkeypatch.setenv(
        "AIPERF_INJECTED_HEADERS",
        _orjson.dumps({"Authorization": "Api-Key real-secret"}).decode(),
    )
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    captured: dict[str, Any] = {}

    def fake_run_single_benchmark(run: BenchmarkRun) -> None:
        captured["headers"] = dict(run.cfg.endpoint.headers)
        captured["env_after_pop"] = (
            "AIPERF_INJECTED_HEADERS" in __import__("os").environ
        )

    with patch(
        "aiperf.cli_runner._run_single_benchmark", side_effect=fake_run_single_benchmark
    ):
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        subprocess_main()

    assert captured["headers"]["Authorization"] == "Api-Key real-secret"
    assert captured["headers"]["X-Trace-Id"] == "trace-001"
    assert captured["env_after_pop"] is False, (
        "AIPERF_INJECTED_HEADERS must be popped before benchmark runs so "
        "child processes can't inherit the secret"
    )


# ---------------------------------------------------------------------------
# Stale-env isolation (PR #982 review feedback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_parent_env_headers_not_forwarded_when_run_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION-LOCK: a stale ``AIPERF_INJECTED_HEADERS`` in the parent
    shell must not leak into a child run that has no sensitive headers.

    Reproduced on PR #982 by ajcasagrande: parent env carried
    ``AIPERF_INJECTED_HEADERS={"Authorization":"Bearer stale-parent-secret"}``;
    the child run configured only ``X-Trace-Id`` (non-sensitive). Pre-fix
    the subprocess overlay still applied the stale Authorization onto
    ``run.cfg.endpoint.headers``, causing the benchmark to send an
    unintended credential-bearing header.

    Fix: ``_run_benchmark_subprocess`` pops both internal injection vars
    from the copied env before conditionally re-setting them.
    """
    monkeypatch.setenv(
        "AIPERF_INJECTED_HEADERS",
        '{"Authorization":"Bearer stale-parent-secret"}',
    )
    monkeypatch.setenv("AIPERF_INJECTED_API_KEY", "stale-parent-api-key")

    cfg = _benchmark_config()
    cfg.endpoint.headers = {"X-Trace-Id": "trace-001"}  # non-sensitive only
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="stale-env-isolation",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    assert "AIPERF_INJECTED_HEADERS" not in env, (
        "Stale AIPERF_INJECTED_HEADERS from parent must not reach child"
    )
    assert "AIPERF_INJECTED_API_KEY" not in env, (
        "Stale AIPERF_INJECTED_API_KEY from parent must not reach child"
    )


# ---------------------------------------------------------------------------
# URL userinfo IPC (PR #982 review feedback — dynamo-ops)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_receives_userinfo_urls_via_env(tmp_path: Path) -> None:
    """REGRESSION-LOCK: ``EndpointConfig.urls`` has an unconditional
    _redact_urls serializer that strips ``user:pass@`` even on mode="python"
    dumps, so the on-disk ``run_config.json`` shows ``http://<redacted>@host``.
    The parent must forward the real URLs out-of-band when at least one
    URL carries userinfo, mirroring the api_key + headers env-var channels.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    cfg.endpoint.urls = ["http://alice:s3cret@host1.example.com/v1/chat/completions"]
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="userinfo-url-env-forward",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    raw = env.get("AIPERF_INJECTED_ENDPOINT_URLS")
    assert raw is not None, "Userinfo URLs must be forwarded via env var"
    forwarded = _orjson.loads(raw)
    assert forwarded == ["http://alice:s3cret@host1.example.com/v1/chat/completions"]


@pytest.mark.asyncio
async def test_no_urls_env_var_when_urls_are_plain(tmp_path: Path) -> None:
    """AIPERF_INJECTED_ENDPOINT_URLS is absent when no URL carries userinfo.

    Plain URLs round-trip through run_config.json unchanged; only userinfo
    URLs need the env-var bypass.
    """
    cfg = _benchmark_config()  # default urls are plain http://localhost:8000
    run = BenchmarkRun(
        benchmark_id="test-id",
        cfg=cfg,
        artifact_dir=tmp_path,
        label="plain-urls",
    )
    executor = LocalSubprocessExecutor(base_dir=tmp_path)

    with patch("aiperf.orchestrator.local_executor.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        await executor.execute(run)

    _, call_kwargs = mock_run.call_args
    env = call_kwargs.get("env") or {}
    assert "AIPERF_INJECTED_ENDPOINT_URLS" not in env


def test_subprocess_runner_pops_and_restores_userinfo_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """subprocess_runner.main consumes AIPERF_INJECTED_ENDPOINT_URLS and
    restores the original userinfo-bearing URL list onto the loaded
    BenchmarkRun. Stale env value must be popped before benchmark runs.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    cfg.endpoint.urls = ["http://<redacted>@host1.example.com/v1/chat/completions"]
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    monkeypatch.setenv(
        "AIPERF_INJECTED_ENDPOINT_URLS",
        _orjson.dumps(
            ["http://alice:s3cret@host1.example.com/v1/chat/completions"]
        ).decode(),
    )
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    captured: dict[str, Any] = {}

    def fake_run_single_benchmark(run: BenchmarkRun) -> None:
        captured["urls"] = list(run.cfg.endpoint.urls)
        captured["env_after_pop"] = (
            "AIPERF_INJECTED_ENDPOINT_URLS" in __import__("os").environ
        )

    with patch(
        "aiperf.cli_runner._run_single_benchmark", side_effect=fake_run_single_benchmark
    ):
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        subprocess_main()

    assert captured["urls"] == [
        "http://alice:s3cret@host1.example.com/v1/chat/completions"
    ]
    assert captured["env_after_pop"] is False


# ---------------------------------------------------------------------------
# Malformed-env-var hardening (PR #982 review feedback — coderabbitai)
# ---------------------------------------------------------------------------


def test_subprocess_runner_rejects_non_dict_headers_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed ``AIPERF_INJECTED_HEADERS`` payload (non-dict JSON or
    invalid JSON) must surface via the structured error envelope, not as
    an unguarded exception at module scope.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    # Valid JSON but wrong shape — must be a dict, here passed as a list.
    monkeypatch.setenv("AIPERF_INJECTED_HEADERS", '["not", "a", "dict"]')
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    with patch("aiperf.cli_runner._run_single_benchmark") as fake_run:
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        with pytest.raises(SystemExit) as exc:
            subprocess_main()
        assert exc.value.code == 1
        fake_run.assert_not_called()


def test_subprocess_runner_rejects_non_list_urls_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AIPERF_INJECTED_ENDPOINT_URLS`` must decode to a list of strings.
    A wrong-shape payload exits via the structured error envelope.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    monkeypatch.setenv("AIPERF_INJECTED_ENDPOINT_URLS", '{"not": "a list"}')
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    with patch("aiperf.cli_runner._run_single_benchmark") as fake_run:
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        with pytest.raises(SystemExit) as exc:
            subprocess_main()
        assert exc.value.code == 1
        fake_run.assert_not_called()


def test_parse_injected_dict_rejects_non_string_values() -> None:
    """REGRESSION-LOCK (dynamo-ops): ``_parse_injected_dict`` must reject a
    JSON object whose values are not strings. ``EndpointConfig.headers`` is
    ``dict[str, str]`` but ``run.cfg.endpoint.headers.update(...)`` mutates in
    place without re-validating, so an int value like ``{"Authorization": 123}``
    would otherwise reach aiohttp as an invalid header.
    """
    from aiperf.orchestrator.subprocess_runner import _parse_injected_dict

    with pytest.raises(ValueError, match="string values"):
        _parse_injected_dict("AIPERF_INJECTED_HEADERS", '{"Authorization": 123}')


def test_parse_injected_dict_malformed_json_raises_value_error_not_decode_error() -> (
    None
):
    """REGRESSION-LOCK (coderabbitai): malformed env-var JSON must surface as a
    ``ValueError`` naming the env var, NOT a raw ``orjson.JSONDecodeError`` that
    main()'s ``except orjson.JSONDecodeError`` block would misreport as a
    config-file error.
    """
    import orjson as _orjson

    from aiperf.orchestrator.subprocess_runner import _parse_injected_dict

    with pytest.raises(ValueError, match="AIPERF_INJECTED_HEADERS contains invalid"):
        _parse_injected_dict("AIPERF_INJECTED_HEADERS", "{not valid json")
    # And specifically not the bare decode error type.
    try:
        _parse_injected_dict("AIPERF_INJECTED_HEADERS", "{not valid json")
    except _orjson.JSONDecodeError:  # pragma: no cover - must not happen
        pytest.fail("malformed env var leaked a raw orjson.JSONDecodeError")
    except ValueError:
        pass


def test_parse_injected_str_list_malformed_json_raises_value_error() -> None:
    """REGRESSION-LOCK (coderabbitai): malformed ``AIPERF_INJECTED_ENDPOINT_URLS``
    JSON surfaces as a ``ValueError`` naming the env var, not a decode error.
    """
    from aiperf.orchestrator.subprocess_runner import _parse_injected_str_list

    with pytest.raises(
        ValueError, match="AIPERF_INJECTED_ENDPOINT_URLS contains invalid"
    ):
        _parse_injected_str_list("AIPERF_INJECTED_ENDPOINT_URLS", "[not valid")


def test_subprocess_runner_rejects_non_string_header_values_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION-LOCK (dynamo-ops): a non-string header value in
    ``AIPERF_INJECTED_HEADERS`` exits via the structured error envelope and
    never reaches the benchmark.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))

    monkeypatch.setenv("AIPERF_INJECTED_HEADERS", '{"Authorization": 123}')
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    with patch("aiperf.cli_runner._run_single_benchmark") as fake_run:
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        with pytest.raises(SystemExit) as exc:
            subprocess_main()
        assert exc.value.code == 1
        fake_run.assert_not_called()


def test_subprocess_runner_malformed_env_not_attributed_to_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REGRESSION-LOCK (coderabbitai): a malformed ``AIPERF_INJECTED_HEADERS``
    must not be reported as ``Invalid JSON in config file`` — the config file
    here is well-formed. The error must name the offending env var instead.
    """
    import orjson as _orjson

    cfg = _benchmark_config()
    run_data = {
        "benchmark_id": "x",
        "cfg": cfg.model_dump(mode="json", exclude_none=True),
        "label": "r1",
        "artifact_dir": str(tmp_path),
    }
    config_file = tmp_path / "run_config.json"
    config_file.write_bytes(_orjson.dumps(run_data))  # valid config on disk

    monkeypatch.setenv("AIPERF_INJECTED_HEADERS", "{not valid json")
    monkeypatch.setattr("sys.argv", ["subprocess_runner", str(config_file)])

    with patch("aiperf.cli_runner._run_single_benchmark") as fake_run:
        from aiperf.orchestrator.subprocess_runner import main as subprocess_main

        with pytest.raises(SystemExit) as exc:
            subprocess_main()
        assert exc.value.code == 1
        fake_run.assert_not_called()

    err = capsys.readouterr().err
    assert "Invalid JSON in config file" not in err, (
        "malformed env var was misattributed to the config file"
    )
    assert "AIPERF_INJECTED_HEADERS" in err
