# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.orchestrator.subprocess_runner``.

The runner is a tiny CLI shim that loads a BenchmarkRun JSON file and hands it
to ``_run_single_benchmark``. Tests exercise the argv/file/JSON guard rails;
the success path is mocked because the real callee invokes ``os._exit`` and
spins up a SystemController.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.orchestrator import subprocess_runner


@pytest.fixture
def mock_run_single(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", mock)
    return mock


@pytest.fixture
def mock_benchmark_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_run = MagicMock(name="BenchmarkRun-instance")
    cls = MagicMock()
    cls.model_validate.return_value = fake_run
    monkeypatch.setattr("aiperf.config.BenchmarkRun", cls)
    return cls


def _set_argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr("sys.argv", ["aiperf.orchestrator.subprocess_runner", *args])


class TestArgvGuards:
    def test_no_args_exits_with_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _set_argv(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()
        assert exc.value.code == 1
        assert "Usage" in capsys.readouterr().err

    def test_too_many_args_exits_with_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _set_argv(monkeypatch, "a.json", "b.json")
        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()
        assert exc.value.code == 1
        assert "Usage" in capsys.readouterr().err


class TestFileGuards:
    def test_missing_file_exits_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "nope.json"
        _set_argv(monkeypatch, str(missing))
        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Config file not found" in err
        assert str(missing) in err

    def test_invalid_json_exits_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_bytes(b"{not valid json")
        _set_argv(monkeypatch, str(bad))
        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()
        assert exc.value.code == 1
        assert "Invalid JSON" in capsys.readouterr().err


class TestSuccessPath:
    def test_valid_run_calls_run_single_benchmark(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_run_single: MagicMock,
        mock_benchmark_run: MagicMock,
    ) -> None:
        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

        subprocess_runner.main()

        mock_benchmark_run.model_validate.assert_called_once_with(
            {"placeholder": "value"}
        )
        mock_run_single.assert_called_once_with(
            mock_benchmark_run.model_validate.return_value
        )


class TestCredentialConsumption:
    """``OPENAI_API_KEY`` must not survive into the spawned service processes.

    The parent resolves ``endpoint.api_key`` (YAML ``${OPENAI_API_KEY}``
    substitution and CLI parsing both run there) and forwards it through
    ``AIPERF_INJECTED_API_KEY``, so the raw shell variable has no reader below
    this point -- leaving it set would expose it via ``/proc/<pid>/environ`` of
    every service.
    """

    def test_openai_api_key_is_popped_before_benchmark_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_benchmark_run: MagicMock,
    ) -> None:
        import os

        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-credential")
        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

        observed: dict[str, bool] = {}

        def record_env(*_: object, **__: object) -> None:
            observed["present"] = "OPENAI_API_KEY" in os.environ

        monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", record_env)

        subprocess_runner.main()

        assert observed["present"] is False, (
            "OPENAI_API_KEY must be popped before services are spawned"
        )
        assert "OPENAI_API_KEY" not in os.environ

    def test_openai_api_key_fills_a_redacted_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_run_single: MagicMock,
        mock_benchmark_run: MagicMock,
    ) -> None:
        """The failure message advertises ``OPENAI_API_KEY`` as an accepted
        source, so a hand-replayed run_config.json must actually resolve from it.
        """
        from aiperf.common.redact import REDACTED_VALUE

        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell-credential")
        run = mock_benchmark_run.model_validate.return_value
        run.cfg.endpoint.api_key = REDACTED_VALUE
        run.cfg.endpoint.headers = {}
        run.cfg.endpoint.urls = ["http://localhost:8000"]

        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

        subprocess_runner.main()

        assert run.cfg.endpoint.api_key == "sk-shell-credential"
        mock_run_single.assert_called_once_with(run)


class TestExceptionHandling:
    def test_unexpected_exception_exits_with_error_and_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_benchmark_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

        def boom(*_: object, **__: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", boom)

        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Failed to run benchmark" in err
        assert "boom" in err
        assert "Traceback" in err

    def test_key_error_during_validate_exits_with_missing_key_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

        cls = MagicMock()
        cls.model_validate.side_effect = KeyError("required_field")
        monkeypatch.setattr("aiperf.config.BenchmarkRun", cls)

        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()
        assert exc.value.code == 1
        assert "Missing required config key" in capsys.readouterr().err


class TestErrorOutputRedaction:
    """Credentials rehydrated by the runner must never reach stderr.

    ``LocalSubprocessExecutor`` splices this process's stderr verbatim into
    ``RunResult.error`` and into the orchestrator's logs, so any credential
    echoed by an exception message or traceback would be persisted there.
    """

    @staticmethod
    def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "run.json"
        cfg.write_bytes(orjson.dumps({"placeholder": "value"}))
        _set_argv(monkeypatch, str(cfg))

    def test_injected_api_key_is_not_echoed_in_error_or_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_benchmark_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from aiperf.common.redact import REDACTED_VALUE

        secret = "sk-secret12345"
        monkeypatch.setenv("AIPERF_INJECTED_API_KEY", secret)
        run = mock_benchmark_run.model_validate.return_value
        run.cfg.endpoint.api_key = REDACTED_VALUE
        run.cfg.endpoint.headers = {}
        run.cfg.endpoint.urls = ["http://localhost:8000"]
        self._write_config(tmp_path, monkeypatch)

        def boom(*_: object, **__: object) -> None:
            raise RuntimeError(f"connection refused while authenticating {secret}")

        monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", boom)

        with pytest.raises(SystemExit) as exc:
            subprocess_runner.main()

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert secret not in err
        assert "Failed to run benchmark" in err
        assert "Traceback" in err
        assert REDACTED_VALUE in err

    def test_sensitive_header_value_is_not_echoed_in_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_benchmark_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        secret = "hdr-secret-98765"
        monkeypatch.setenv(
            "AIPERF_INJECTED_HEADERS", orjson.dumps({"X-Api-Key": secret}).decode()
        )
        run = mock_benchmark_run.model_validate.return_value
        run.cfg.endpoint.api_key = None
        run.cfg.endpoint.headers = {}
        run.cfg.endpoint.urls = ["http://localhost:8000"]
        self._write_config(tmp_path, monkeypatch)

        def boom(*_: object, **__: object) -> None:
            raise RuntimeError(f"upstream rejected header value {secret}")

        monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", boom)

        with pytest.raises(SystemExit):
            subprocess_runner.main()

        assert secret not in capsys.readouterr().err

    def test_userinfo_url_is_not_echoed_in_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mock_benchmark_run: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        secret = "urlpass4321"
        url = f"http://user:{secret}@example.com:8000"
        monkeypatch.setenv(
            "AIPERF_INJECTED_ENDPOINT_URLS", orjson.dumps([url]).decode()
        )
        run = mock_benchmark_run.model_validate.return_value
        run.cfg.endpoint.api_key = None
        run.cfg.endpoint.headers = {}
        run.cfg.endpoint.urls = []
        self._write_config(tmp_path, monkeypatch)

        def boom(*_: object, **__: object) -> None:
            raise RuntimeError(f"could not connect to {url}")

        monkeypatch.setattr("aiperf.cli_runner._run_single_benchmark", boom)

        with pytest.raises(SystemExit):
            subprocess_runner.main()

        assert secret not in capsys.readouterr().err
