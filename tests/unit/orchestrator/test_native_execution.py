# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the native execution bridge (binary resolution + parsing).

Replaces the deleted ``runner_installation``/``rust_executor`` discovery tests.
Precedence for :func:`resolve_native_binary`: explicit -> ``AIPERF_EXEC_BIN`` env
-> ``aiperf`` on PATH -> the binary alongside ``sys.executable``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.orchestrator import native_execution
from aiperf.orchestrator.native_execution import resolve_native_binary


def _make_executable(path: Path) -> Path:
    path.write_bytes(b"#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_explicit_binary_takes_precedence(tmp_path: Path) -> None:
    explicit = _make_executable(tmp_path / "explicit-aiperf")
    assert resolve_native_binary(explicit) == explicit.resolve()


def test_env_override_selected_when_no_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _make_executable(tmp_path / "env-aiperf")
    monkeypatch.setenv("AIPERF_EXEC_BIN", str(configured))
    monkeypatch.setattr(native_execution.shutil, "which", lambda _name: None)
    assert resolve_native_binary(None) == configured.resolve()


def test_path_lookup_when_no_explicit_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_path = _make_executable(tmp_path / "path-aiperf")
    monkeypatch.delenv("AIPERF_EXEC_BIN", raising=False)
    monkeypatch.setattr(native_execution.shutil, "which", lambda _name: str(on_path))
    assert resolve_native_binary(None) == on_path.resolve()


def test_sys_executable_neighbour_is_last_resort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    neighbour = _make_executable(bindir / "aiperf")
    monkeypatch.delenv("AIPERF_EXEC_BIN", raising=False)
    monkeypatch.setattr(native_execution.shutil, "which", lambda _name: None)
    monkeypatch.setattr(sys, "executable", str(bindir / "python"))
    assert resolve_native_binary(None) == neighbour


def test_missing_binary_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AIPERF_EXEC_BIN", raising=False)
    monkeypatch.setattr(native_execution.shutil, "which", lambda _name: None)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    with pytest.raises(FileNotFoundError):
        resolve_native_binary(None)


def test_env_override_that_is_not_executable_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_exec = tmp_path / "plain"
    not_exec.write_text("data")
    monkeypatch.setenv("AIPERF_EXEC_BIN", str(not_exec))
    with pytest.raises(FileNotFoundError):
        resolve_native_binary(None)


def _terminal_line(**overrides: object) -> bytes:
    payload = {
        "protocol_version": 2,
        "event": "run_terminal",
        "benchmark_id": "bench-1",
        "success": True,
    }
    payload.update(overrides)
    return orjson.dumps(payload) + b"\n"


def test_parse_terminal_accepts_one_valid_line() -> None:
    run = SimpleNamespace(benchmark_id="bench-1")
    terminal = native_execution._parse_terminal(
        _terminal_line(), run, protocol_version=2, returncode=0
    )
    assert terminal["success"] is True


def test_parse_terminal_rejects_multiple_lines() -> None:
    run = SimpleNamespace(benchmark_id="bench-1")
    two = _terminal_line() + _terminal_line()
    with pytest.raises(ValueError, match="exactly one terminal JSON line"):
        native_execution._parse_terminal(two, run, protocol_version=2, returncode=0)


def test_parse_terminal_rejects_wrong_benchmark_id() -> None:
    run = SimpleNamespace(benchmark_id="bench-1")
    with pytest.raises(ValueError, match="benchmark_id"):
        native_execution._parse_terminal(
            _terminal_line(benchmark_id="other"), run, protocol_version=2, returncode=0
        )


def test_classify_success_and_failure() -> None:
    from aiperf.common.models.export_models import JsonMetricResult

    run = SimpleNamespace(label="r", trial=0, artifact_dir=Path("/tmp"))
    ok = native_execution._classify(
        {"request_count": JsonMetricResult(unit="requests", avg=5.0)}, run
    )
    assert ok.success is True
    empty = native_execution._classify(
        {"request_count": JsonMetricResult(unit="requests", avg=0.0)}, run
    )
    assert empty.success is False
    assert empty.error == "No requests completed"


def test_run_cell_process_invokes_cell_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv, check):  # noqa: ANN001
        seen["argv"] = argv
        seen["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native_execution.subprocess, "run", fake_run)
    rc = native_execution.run_cell_process(Path("/opt/aiperf"))
    assert rc == 0
    assert seen["argv"] == ["/opt/aiperf", "--cell"]
    assert seen["check"] is False


def test_env_override_ignores_dangling_relative_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A relative env override is expanded/resolved; a nonexistent one still fails.
    monkeypatch.setenv("AIPERF_EXEC_BIN", "does-not-exist-aiperf")
    with pytest.raises(FileNotFoundError):
        resolve_native_binary(None)
    assert os.environ["AIPERF_EXEC_BIN"] == "does-not-exist-aiperf"
