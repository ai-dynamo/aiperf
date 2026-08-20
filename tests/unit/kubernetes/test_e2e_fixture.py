# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kubernetes end-to-end fixture recovery."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from tests.kubernetes import conftest as e2e_conftest
from tests.kubernetes.conftest import (
    K8sTestSettings,
    _create_helm_values,
    _create_image_manager,
    _ensure_cluster_api_ready_after_image_setup,
    _ensure_cluster_api_stable,
    _ensure_jobset_controller_ready,
    _mock_server_deployment_is_healthy,
    _purge_reused_cluster_resources,
    _render_mock_server_manifest,
)


def test_create_image_manager_uses_resolved_custom_images(tmp_path: Path) -> None:
    settings = K8sTestSettings(
        aiperf_image="registry.example:5000/team/aiperf:pr-42",
        mock_server_image="registry.example/team/mock-server:test",
    )

    manager = _create_image_manager(tmp_path, settings)

    assert manager.get_image_name("aiperf") == settings.aiperf_image
    assert manager.get_image_name("mock-server") == settings.mock_server_image


def test_create_helm_values_uses_resolved_custom_aiperf_image() -> None:
    settings = K8sTestSettings(aiperf_image="registry.example:5000/team/aiperf:pr-42")

    values = _create_helm_values(settings)

    assert values.image_repository == "registry.example:5000/team/aiperf"
    assert values.image_tag == "pr-42"
    assert values.default_image == settings.aiperf_image


def test_render_mock_server_manifest_replaces_default_image() -> None:
    template = "containers:\n        - name: mock-server\n          image: aiperf-mock-server:latest\n"

    rendered = _render_mock_server_manifest(template, "example/mock-server:test")

    assert "image: example/mock-server:test" in rendered
    assert "image: aiperf-mock-server:latest" not in rendered


@pytest.mark.asyncio
async def test_mock_server_deployment_healthy_requires_configured_image() -> None:
    kubectl = AsyncMock()
    kubectl.run.return_value = SimpleNamespace(
        returncode=0,
        stdout="1|example/mock-server:old",
    )

    healthy = await _mock_server_deployment_is_healthy(
        kubectl,
        "example/mock-server:new",
    )

    assert not healthy


@pytest.mark.asyncio
async def test_purge_reused_cluster_resources_includes_sweeps_and_worker_namespaces() -> (
    None
):
    kubectl = AsyncMock()

    async def _run(*args: str, **_: object) -> SimpleNamespace:
        namespace = args[args.index("-n") + 1] if "-n" in args else ""
        if args[:2] == ("get", "aiperfsweeps") and namespace == "aiperf-jobs-gw2":
            return SimpleNamespace(returncode=0, stdout="stale-sweep")
        return SimpleNamespace(returncode=0, stdout="")

    kubectl.run.side_effect = _run

    await _purge_reused_cluster_resources(kubectl, "gw2")

    assert (
        call(
            "patch",
            "aiperfsweep",
            "stale-sweep",
            "-n",
            "aiperf-jobs-gw2",
            "--type=json",
            '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
            check=False,
        )
        in kubectl.run.await_args_list
    )
    deleted_namespaces = {
        invocation.args[4]
        for invocation in kubectl.run.await_args_list
        if invocation.args[:2] == ("delete", "aiperfjobs,aiperfsweeps,jobsets")
    }
    assert "aiperf-bench-gw2" in deleted_namespaces
    assert "aiperf-jobs-gw2" in deleted_namespaces


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded_any", [False, True])
async def test_image_setup_always_checks_cluster_api_readiness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loaded_any: bool,
) -> None:
    stable = AsyncMock()
    monkeypatch.setattr(e2e_conftest, "_ensure_cluster_api_stable", stable)
    kubectl = AsyncMock()

    await _ensure_cluster_api_ready_after_image_setup(
        kubectl,
        loaded_any=loaded_any,
    )

    if loaded_any:
        stable.assert_awaited_once_with(kubectl)
    else:
        assert stable.await_args is not None
        assert stable.await_args.kwargs == {
            "stable_probes": 1,
            "max_probes": 6,
            "interval": 2,
        }


@pytest.mark.asyncio
async def test_ensure_cluster_api_stable_requires_consecutive_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = AsyncMock()
    kubectl.run.side_effect = [
        SimpleNamespace(returncode=0),
        SimpleNamespace(returncode=0),
        SimpleNamespace(returncode=1),
        SimpleNamespace(returncode=0),
        SimpleNamespace(returncode=0),
        SimpleNamespace(returncode=0),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr("tests.kubernetes.conftest.asyncio.sleep", sleep)

    await _ensure_cluster_api_stable(
        kubectl,
        stable_probes=3,
        max_probes=6,
        interval=0,
    )

    assert kubectl.run.await_count == 6
    assert sleep.await_count == 5


@pytest.mark.asyncio
async def test_ensure_cluster_api_stable_exhausted_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = AsyncMock()
    kubectl.run.return_value.returncode = 1
    monkeypatch.setattr("tests.kubernetes.conftest.asyncio.sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="did not remain ready"):
        await _ensure_cluster_api_stable(
            kubectl,
            stable_probes=2,
            max_probes=3,
            interval=0,
        )


@pytest.mark.asyncio
async def test_ensure_jobset_controller_ready_available_does_not_restart() -> None:
    kubectl = AsyncMock()
    kubectl.wait_for_condition.return_value = True

    await _ensure_jobset_controller_ready(kubectl)

    kubectl.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_jobset_controller_ready_initial_race_restarts() -> None:
    kubectl = AsyncMock()
    kubectl.wait_for_condition.side_effect = [False, True]

    await _ensure_jobset_controller_ready(kubectl)

    kubectl.run.assert_awaited_once_with(
        "rollout",
        "restart",
        "deployment/jobset-controller-manager",
        namespace="jobset-system",
    )


@pytest.mark.asyncio
async def test_ensure_jobset_controller_ready_restart_still_unavailable_raises() -> (
    None
):
    kubectl = AsyncMock()
    kubectl.wait_for_condition.side_effect = [False, False]

    with pytest.raises(RuntimeError, match="did not become available after restart"):
        await _ensure_jobset_controller_ready(kubectl)
