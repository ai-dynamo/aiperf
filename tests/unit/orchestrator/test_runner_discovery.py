# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic discovery tests for the interned native runner.

Precedence: ``explicit --runner-bin -> AIPERF_RUNNER_BIN -> interned package
data -> PATH``. The runner is interned in the one ``aiperf`` wheel as package
data (``aiperf/_bin/aiperf-runner``); there is no separate companion
distribution.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aiperf.orchestrator import runner_installation


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"native-runner")
    path.chmod(0o755)
    return path.resolve()


class _FakeAnchor:
    """Stand-in for ``importlib.resources.files('aiperf')``.

    ``joinpath(subdir, command)`` returns whatever concrete path the test wants
    the interned tier to see.
    """

    def __init__(self, target: Path) -> None:
        self._target = target

    def joinpath(self, *_parts: str) -> Path:
        return self._target


def _set_interned(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    monkeypatch.setattr(
        runner_installation.resources, "files", lambda _pkg: _FakeAnchor(target)
    )


def test_explicit_runner_precedes_environment_interned_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = _executable(tmp_path, "explicit-runner")
    configured = _executable(tmp_path, "configured-runner")
    interned = _executable(tmp_path, "interned-runner")
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(configured))
    _set_interned(monkeypatch, interned)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(explicit) == explicit


def test_environment_runner_precedes_interned_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _executable(tmp_path, "configured-runner")
    interned = _executable(tmp_path, "interned-runner")
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(configured))
    _set_interned(monkeypatch, interned)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(None) == configured


def test_fresh_install_discovers_interned_without_environment_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interned = _executable(tmp_path, "aiperf-runner")
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: None)
    _set_interned(monkeypatch, interned)

    assert runner_installation._resolve_runner_binary(None) == interned


def test_path_is_only_the_development_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_runner = _executable(tmp_path, "aiperf-runner")
    missing = tmp_path / "_bin" / "aiperf-runner"  # not created -> interned absent
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    _set_interned(monkeypatch, missing)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(None) == path_runner


def test_no_tier_resolves_raises_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "_bin" / "aiperf-runner"
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    _set_interned(monkeypatch, missing)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="did not intern"):
        runner_installation._resolve_runner_binary(None)


@pytest.mark.parametrize("selection", ["explicit", "environment"])
def test_broken_selected_tier_never_substitutes_a_lower_precedence_runner(
    selection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-runner"
    interned = _executable(tmp_path, "interned-runner")
    path_runner = _executable(tmp_path, "path-runner")
    _set_interned(monkeypatch, interned)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)
    explicit = missing if selection == "explicit" else None
    if selection == "environment":
        monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(missing))

    with pytest.raises(FileNotFoundError, match="refusing to substitute"):
        runner_installation._resolve_runner_binary(explicit)


def test_interned_tier_requires_an_executable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    non_executable = tmp_path / "aiperf-runner"
    non_executable.write_bytes(b"native-runner")
    non_executable.chmod(0o644)  # readable, not executable
    _set_interned(monkeypatch, non_executable)

    assert runner_installation._interned_binary() is None
