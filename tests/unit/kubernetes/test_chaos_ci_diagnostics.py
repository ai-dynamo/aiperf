# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-time diagnostics capture for the Kubernetes CI workflows.

The chaos workflow's "Collect cluster diagnostics on failure" step ran after
the pytest session fixtures had already uninstalled the operator and deleted
the Kind cluster, so every artifact was empty or connection-refused. The
fixtures now (a) dump diagnostics from within teardown, before the operator
uninstall and cluster delete, when the session saw failures, and (b) keep the
cluster on failure only, so the workflow step reads a live cluster and the
always() teardown still protects shared runners after green runs. These tests
lock the settings resolution, the failure tracker, and the capture's
never-raise / kind-export behavior.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.kubernetes import conftest as k8s_conftest
from tests.kubernetes.conftest import _FAILED_TESTS_KEY, _resolve_settings
from tests.kubernetes.helpers import diagnostics as diag_mod
from tests.kubernetes.helpers.diagnostics import dump_cluster_diagnostics


class _FakeConfig:
    """Pytest-config-shaped stub with no explicit CLI options set."""

    def __init__(self) -> None:
        self.option = SimpleNamespace()

    def getoption(self, name: str, default: object = None) -> object:
        return getattr(self.option, name, default)


@pytest.fixture(autouse=True)
def _clear_diag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "K8S_TEST_KEEP_CLUSTER_ON_FAILURE",
        "K8S_TEST_DIAGNOSTICS_DIR",
        "K8S_TEST_SKIP_CLEANUP",
        "K8S_TEST_QUICK",
    ):
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults_are_inert() -> None:
    settings = _resolve_settings(_FakeConfig())

    assert settings.keep_cluster_on_failure is False
    assert settings.diagnostics_dir is None


def test_settings_env_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K8S_TEST_KEEP_CLUSTER_ON_FAILURE", "true")
    monkeypatch.setenv("K8S_TEST_DIAGNOSTICS_DIR", "out/diag")

    settings = _resolve_settings(_FakeConfig())

    assert settings.keep_cluster_on_failure is True
    assert settings.diagnostics_dir == "out/diag"


def test_failure_tracker_records_only_failures() -> None:
    tracker = k8s_conftest._FailureTracker()

    tracker.pytest_runtest_logreport(SimpleNamespace(failed=True, nodeid="t::a"))
    tracker.pytest_runtest_logreport(SimpleNamespace(failed=False, nodeid="t::b"))

    assert tracker.failed == ["t::a"]


def test_session_failures_reads_stash() -> None:
    config = SimpleNamespace(stash={})
    assert k8s_conftest._session_failures(config) == []

    config.stash[_FAILED_TESTS_KEY] = ["t::a"]
    assert k8s_conftest._session_failures(config) == ["t::a"]


@pytest.mark.asyncio
async def test_dump_captures_state_pod_logs_and_kind_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[tuple[list[str], Path]] = []

    async def fake_run(cmd: list[str], out_path: Path, timeout: int) -> None:  # noqa: ARG001
        captured.append((cmd, out_path))
        if out_path.name == "_pod-index.txt":
            out_path.write_text("aiperf-system aiperf-operator-abc\n")
        else:
            out_path.write_text("<fake>")

    monkeypatch.setattr(diag_mod, "_run_capture", fake_run)

    out = await dump_cluster_diagnostics(
        "kind-ci", tmp_path / "diag", cluster_name="ci"
    )

    cmds = [cmd for cmd, _ in captured]
    files = [p.name for _, p in captured]
    assert "events.txt" in files
    assert "describe-nodes.txt" in files
    assert "describe-pods.txt" in files
    assert "pods.yaml" in files
    assert "jobsets.yaml" in files
    # Per-pod logs for the pod listed by the index capture.
    assert "logs-aiperf-system-aiperf-operator-abc.txt" in files
    assert any(
        cmd[:4] == ["kubectl", "--context", "kind-ci", "logs"] for cmd in cmds
    )
    # kind export runs only because a cluster name was passed.
    assert any(cmd[:3] == ["kind", "export", "logs"] for cmd in cmds)


@pytest.mark.asyncio
async def test_dump_without_cluster_name_skips_kind_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[list[str]] = []

    async def fake_run(cmd: list[str], out_path: Path, timeout: int) -> None:  # noqa: ARG001
        captured.append(cmd)
        if out_path.name == "_pod-index.txt":
            out_path.write_text("")

    monkeypatch.setattr(diag_mod, "_run_capture", fake_run)

    await dump_cluster_diagnostics("kind-ci", tmp_path / "diag")

    assert not any(cmd[:2] == ["kind", "export"] for cmd in captured)


@pytest.mark.asyncio
async def test_dump_never_raises_on_capture_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def boom(cmd: list[str], out_path: Path, timeout: int) -> None:  # noqa: ARG001
        raise RuntimeError("cluster is gone")

    monkeypatch.setattr(diag_mod, "_run_capture", fake := boom)

    # Even with every command failing, the dump completes (the pod-index read
    # just finds nothing) so fixture teardown is never broken by diagnostics.
    await dump_cluster_diagnostics(None, tmp_path / "diag")
    assert fake is boom  # silence unused; the assertion is "no exception"
