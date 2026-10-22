# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes CI test-harness contracts that only surface on a GitHub runner.

Two failure modes from the chaos workflow that no live-cluster test can catch:

* The workflow's diagnostics step ran after the session fixture had already
  deleted the Kind cluster, so every artifact was ``connection refused``.
  ``K8S_TEST_KEEP_CLUSTER`` must suppress only that final delete.
* ``azure/setup-helm@v4`` now resolves Helm 4, whose server-side apply refuses
  fields owned by ``kubectl-client-side-apply`` (the CRDs the fixtures
  pre-install), so ``helm install`` of the same chart fails with conflicts.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest import param

from tests.kubernetes.conftest import _resolve_settings
from tests.kubernetes.helpers.helm import HelmClient


class _FakeConfig:
    """Pytest-config-shaped stub with no explicit CLI options set."""

    def __init__(self) -> None:
        self.option = SimpleNamespace()

    def getoption(self, name: str, default: object = None) -> object:
        return getattr(self.option, name, default)


@pytest.fixture(autouse=True)
def _clear_k8s_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("K8S_TEST_KEEP_CLUSTER", "K8S_TEST_SKIP_CLEANUP", "K8S_TEST_QUICK"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_settings_keep_cluster_defaults_false() -> None:
    settings = _resolve_settings(_FakeConfig())

    assert settings.keep_cluster is False
    assert settings.skip_cleanup is False


def test_resolve_settings_keep_cluster_env_does_not_imply_skip_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K8S_TEST_KEEP_CLUSTER", "true")

    settings = _resolve_settings(_FakeConfig())

    assert settings.keep_cluster is True
    assert settings.skip_cleanup is False


def test_resolve_settings_quick_does_not_imply_keep_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K8S_TEST_QUICK", "1")

    settings = _resolve_settings(_FakeConfig())

    assert settings.skip_cleanup is True
    assert settings.keep_cluster is False


def _helm_with_version(version_stdout: str) -> tuple[HelmClient, AsyncMock]:
    """HelmClient whose ``_run`` answers ``helm version`` with ``version_stdout``."""
    client = HelmClient(kubecontext="kind-test")
    run = AsyncMock(
        return_value=SimpleNamespace(returncode=0, stdout=version_stdout, stderr="")
    )
    client._run = run
    return client, run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version_stdout", "expected_major"),
    [
        param("v4.3.0", 4, id="helm4"),
        param("v3.16.2", 3, id="helm3"),
        param("4.0.0-rc.1", 4, id="no_v_prefix"),
        param("", 0, id="unparseable"),
    ],
)  # fmt: skip
async def test_helm_major_version_parses_and_caches(
    version_stdout: str, expected_major: int
) -> None:
    client, run = _helm_with_version(version_stdout)

    assert await client.major_version() == expected_major
    assert await client.major_version() == expected_major
    run.assert_awaited_once_with("version", "--template", "{{.Version}}", check=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version_stdout", "expects_flag"),
    [
        param("v4.3.0", True, id="helm4_forces_conflicts"),
        param("v3.16.2", False, id="helm3_rejects_flag"),
        param("", False, id="unknown_version_stays_conservative"),
    ],
)  # fmt: skip
async def test_helm_install_and_upgrade_force_conflicts_only_on_helm4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,  # noqa: ANN001 - pytest fixture
    version_stdout: str,
    expects_flag: bool,
) -> None:
    client, _ = _helm_with_version(version_stdout)
    captured: list[list[str]] = []

    async def fake_streaming(cmd: list[str], *_args: object, **_kwargs: object) -> None:
        captured.append(cmd)

    monkeypatch.setattr("tests.kubernetes.helpers.helm._run_streaming", fake_streaming)

    await client.install("rel", tmp_path, "ns", wait=False)
    await client.upgrade("rel", tmp_path, "ns", wait=False)

    assert len(captured) == 2
    for cmd in captured:
        assert ("--force-conflicts" in cmd) is expects_flag
        assert cmd[:3] == ["helm", "--kube-context", "kind-test"]
