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
