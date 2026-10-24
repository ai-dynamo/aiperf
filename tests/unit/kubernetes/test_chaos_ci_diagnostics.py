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


def test_failure_tracker_records_only_failures_and_dedupes() -> None:
    tracker = k8s_conftest._FailureTracker()

    tracker.pytest_runtest_logreport(SimpleNamespace(failed=True, nodeid="t::a"))
    tracker.pytest_runtest_logreport(SimpleNamespace(failed=False, nodeid="t::b"))
    tracker.pytest_runtest_logreport(SimpleNamespace(failed=True, nodeid="t::a"))

    assert tracker.failed == ["t::a"]


def test_record_failure_helper_appends_to_stash() -> None:
    config = SimpleNamespace(stash={k8s_conftest._FAILED_TESTS_KEY: []})

    k8s_conftest._record_failure(config, "operator-uninstall")
    k8s_conftest._record_failure(config, "operator-uninstall")

    assert k8s_conftest._session_failures(config) == ["operator-uninstall"]


def test_finalizer_failure_is_recorded_before_later_finalizers_run(
    tmp_path: Path,
) -> None:
    """Real pytest lifecycle: an otherwise-passing final test, a failing
    earlier package finalizer (the operator-uninstall analog), and a later
    finalizer (the cluster-teardown analog) whose keep/dump decision must
    already see the failure. Finalizer failures produce no test report, so
    only boundary recording can close this blind spot."""
    testdir = tmp_path / "lifecycle"
    testdir.mkdir()
    sentinel = tmp_path / "evidence-kept"
    (testdir / "conftest.py").write_text(
        """import os

import pytest

from tests.kubernetes.conftest import (
    _FAILED_TESTS_KEY,
    _FailureTracker,
    _record_failure,
    _session_failures,
)


def pytest_configure(config):
    tracker = _FailureTracker()
    config.stash[_FAILED_TESTS_KEY] = tracker.failed
    config.pluginmanager.register(tracker, "failure-tracker")


@pytest.fixture(scope="package")
def cluster_sentinel(request):
    yield
    # Tears down SECOND (reverse setup order): the cluster-teardown analog —
    # must already see the earlier finalizer's failure.
    if _session_failures(request.config):
        with open(os.environ["SENTINEL_PATH"], "w") as f:
            f.write("kept")


@pytest.fixture(scope="package")
def failing_operator_finalizer(request, cluster_sentinel):
    yield
    # Tears down FIRST: the operator-uninstall analog.
    try:
        raise RuntimeError("uninstall exploded")
    except RuntimeError:
        _record_failure(request.config, "operator-uninstall")
        raise
"""
    )
    (testdir / "test_lifecycle.py").write_text(
        """def test_passes(failing_operator_finalizer):
    assert True
"""
    )

    import subprocess
    import sys

    repo_root = Path(k8s_conftest.__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(testdir / "test_lifecycle.py"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **__import__("os").environ,
            "SENTINEL_PATH": str(sentinel),
            "PYTHONPATH": str(repo_root),
        },
    )
    assert proc.returncode != 0, (
        f"the failing finalizer must fail the run.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert sentinel.exists(), (
        "evidence sentinel missing: the later finalizer did not see the "
        f"recorded failure.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


@pytest.mark.asyncio
async def test_helm_deployer_finalizer_records_failure_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the ACTUAL helm_deployer fixture (not an analog): a failing
    cleanup must be recorded in the session stash before the exception
    propagates to the remaining finalizer stack."""
    failures: list[str] = []
    config = SimpleNamespace(stash={k8s_conftest._FAILED_TESTS_KEY: failures})
    request = SimpleNamespace(config=config)

    helm_deployer_mod = __import__(
        "tests.kubernetes.helpers.helm", fromlist=["HelmDeployer"]
    )
    monkeypatch.setattr(
        helm_deployer_mod.HelmDeployer,
        "cleanup_all",
        AsyncMock(side_effect=RuntimeError("cleanup exploded")),
    )

    kubectl = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(returncode=0)))
    operator_deployer = SimpleNamespace(
        is_operator_healthy=AsyncMock(return_value=True)
    )
    k8s_settings = SimpleNamespace(image_pull_secret=None)

    gen = k8s_conftest.helm_deployer.__wrapped__(
        kubectl=kubectl,
        helm_client=SimpleNamespace(),
        project_root=Path("/nonexistent"),
        loaded_images=None,
        jobset_controller=None,
        helm_values=SimpleNamespace(),
        operator_deployer=operator_deployer,
        worker_namespace_suffix="w0",
        k8s_settings=k8s_settings,
        request=request,
    )
    await gen.__anext__()  # fixture setup
    with pytest.raises(RuntimeError, match="cleanup exploded"):
        await gen.__anext__()  # finalizer raises

    assert failures == ["helm-deployer-teardown"], (
        "the destructive finalizer must record its failure before propagating"
    )
