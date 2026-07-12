# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic discovery tests for the native runner companion."""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest

from aiperf.orchestrator import runner_installation


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"native-runner")
    path.chmod(0o755)
    return path.resolve()


def test_explicit_runner_precedes_environment_companion_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = _executable(tmp_path, "explicit-runner")
    configured = _executable(tmp_path, "configured-runner")
    companion = _executable(tmp_path, "companion-runner")
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(configured))
    monkeypatch.setattr(
        runner_installation, "_installed_companion_binary", lambda: companion
    )
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(explicit) == explicit


def test_environment_runner_precedes_companion_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _executable(tmp_path, "configured-runner")
    companion = _executable(tmp_path, "companion-runner")
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(configured))
    monkeypatch.setattr(
        runner_installation, "_installed_companion_binary", lambda: companion
    )
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(None) == configured


def test_fresh_install_discovers_companion_without_environment_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = _executable(tmp_path, "aiperf-runner")
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: None)

    class InstalledCompanion:
        files = [Path("../../../bin/aiperf-runner")]

        @staticmethod
        def locate_file(_entry: Path) -> Path:
            return companion

    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda name: InstalledCompanion()
        if name == "aiperf-runner"
        else pytest.fail(f"unexpected distribution lookup {name!r}"),
    )

    assert runner_installation._resolve_runner_binary(None) == companion


def test_path_is_only_the_development_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_runner = _executable(tmp_path, "aiperf-runner")
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: (_ for _ in ()).throw(metadata.PackageNotFoundError),
    )
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    assert runner_installation._resolve_runner_binary(None) == path_runner


@pytest.mark.parametrize("selection", ["explicit", "environment"])
def test_broken_selected_tier_never_substitutes_a_lower_precedence_runner(
    selection: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-runner"
    companion = _executable(tmp_path, "companion-runner")
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.setattr(
        runner_installation, "_installed_companion_binary", lambda: companion
    )
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)
    explicit = missing if selection == "explicit" else None
    if selection == "environment":
        monkeypatch.setenv("AIPERF_RUNNER_BIN", os.fspath(missing))

    with pytest.raises(FileNotFoundError, match="refusing to substitute"):
        runner_installation._resolve_runner_binary(explicit)


def test_broken_installed_companion_never_substitutes_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-companion-runner"
    path_runner = _executable(tmp_path, "path-runner")
    monkeypatch.delenv("AIPERF_RUNNER_BIN", raising=False)
    monkeypatch.setattr(runner_installation.shutil, "which", lambda _name: path_runner)

    class BrokenCompanion:
        files = [Path("../../../bin/aiperf-runner")]

        @staticmethod
        def locate_file(_entry: Path) -> Path:
            return missing

    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: BrokenCompanion(),
    )

    with pytest.raises(FileNotFoundError, match="installed companion.*refusing"):
        runner_installation._resolve_runner_binary(None)


def test_companion_record_must_identify_exactly_one_native_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AmbiguousCompanion:
        files = [
            Path("../../../bin/aiperf-runner"),
            Path("aiperf_runner/bin/aiperf-runner"),
        ]

        @staticmethod
        def locate_file(entry: Path) -> Path:
            return entry

    monkeypatch.setattr(
        runner_installation.metadata,
        "distribution",
        lambda _name: AmbiguousCompanion(),
    )

    with pytest.raises(RuntimeError, match="exactly one.*found 2"):
        runner_installation._installed_companion_binary()
