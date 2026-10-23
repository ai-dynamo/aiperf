# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-overcommit scheduling floor for local Kind test clusters.

The weekly audit workflow failed 4/4 scheduled runs and the chaos workflow
has never passed because the operator's production-sized pod requests (~6
CPUs per AIPerfJob plus a 2-CPU mock server) can never be admitted by a
single-node Kind cluster on a 4-vCPU ubuntu-latest runner; pods sat Pending
with FailedScheduling until timeout. The fixture raises the node's advertised
CPU floor instead of weakening the pod requests, so production defaults keep
their meaning. Kubelet re-asserts real capacity on every node-status sync, so
the fixture re-applies the patch on an interval from a daemon thread (a
pytest-asyncio loop would freeze the re-assertion for the duration of every
test). These tests lock the floor semantics, the opt-in setting, the
never-shrink invariant, and the kind:// providerID locality guard.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from pytest import param

from tests.kubernetes.conftest import _resolve_settings
from tests.kubernetes.helpers import cluster as cluster_mod
from tests.kubernetes.helpers.cluster import (
    KIND_CPU_OVERCOMMIT_FLOOR,
    node_cpu_overcommit_patch,
    parse_cpu_millicores,
)


class _FakeConfig:
    """Pytest-config-shaped stub with no explicit CLI options set."""

    def __init__(self) -> None:
        self.option = SimpleNamespace()

    def getoption(self, name: str, default: object = None) -> object:
        return getattr(self.option, name, default)


@pytest.fixture(autouse=True)
def _clear_overcommit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K8S_TEST_NODE_CPU_OVERCOMMIT", raising=False)


@pytest.mark.parametrize(
    ("quantity", "expected_millicores"),
    [
        param("4", 4000, id="whole_cores"),
        param("4000m", 4000, id="millicores"),
        param("2500m", 2500, id="fractional_via_milli"),
        param("1.5", 1500, id="decimal_cores"),
    ],
)
def test_parse_cpu_millicores(quantity: str, expected_millicores: int) -> None:
    assert parse_cpu_millicores(quantity) == expected_millicores


def test_patch_raises_small_node_to_floor() -> None:
    patch = node_cpu_overcommit_patch("4", "3900m", 32)

    assert patch is not None
    ops = json.loads(patch)
    assert ops == [
        {"op": "replace", "path": "/status/capacity/cpu", "value": "32"},
        {"op": "replace", "path": "/status/allocatable/cpu", "value": "32"},
    ]


@pytest.mark.parametrize(
    ("capacity", "allocatable"),
    [
        param("32", "32", id="exactly_at_floor"),
        param("64", "63400m", id="above_floor"),
        param("64", "64", id="above_floor_millicores"),
    ],
)
def test_patch_never_shrinks_large_nodes(capacity: str, allocatable: str) -> None:
    """GPU/dev nodes with real capacity above the floor must be left alone."""
    assert node_cpu_overcommit_patch(capacity, allocatable, 32) is None


def test_patch_floors_fields_independently() -> None:
    """A node with capacity above the floor but allocatable below it must only
    have allocatable raised — capacity must never be shrunk to the floor."""
    patch = node_cpu_overcommit_patch("64", "4", 32)

    assert patch is not None
    ops = json.loads(patch)
    assert ops == [
        {"op": "replace", "path": "/status/allocatable/cpu", "value": "32"},
    ]


def test_settings_default_is_disabled() -> None:
    """Opt-in only: a plain local run must never fake node capacity."""
    settings = _resolve_settings(_FakeConfig())

    assert settings.node_cpu_overcommit == 0
    assert KIND_CPU_OVERCOMMIT_FLOOR >= 32, (
        "the floor must cover a full AIPerfJob (controller ~2.6 CPUs + worker "
        "pod 3.35 CPUs) beside the mock server (2 CPUs), kube-system (~1 CPU), "
        "the JobSet controller (0.5 CPUs), and the operator pod (0.35 CPUs), "
        "with headroom for the chaos suite's concurrent job sets"
    )


def test_settings_env_override_and_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "32")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 32

    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "0")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 0


@pytest.mark.asyncio
async def test_apply_refuses_non_kind_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A context repointed at a remote cluster must never get faked capacity."""
    calls: list[list[str]] = []

    async def fake_kubectl(context: str, *args: str, check: bool = True):  # noqa: ANN202, ARG001
        calls.append(list(args))
        if args[:2] == ("get", "nodes"):
            return (
                0,
                "gpu-node-0 h100-node-pool 64 63400m\n"
                "remote-1 aks-agentpool 8 7900m\n",
                "",
            )
        return 0, "", ""

    monkeypatch.setattr(cluster_mod, "_kubectl", fake_kubectl)

    patched = await cluster_mod.apply_cpu_overcommit("kind-fake", 32)

    assert patched == []
    assert all(c[:2] == ["get", "nodes"] for c in calls), (
        f"no patch may be issued against non-kind nodes, got {calls}"
    )


@pytest.mark.asyncio
async def test_apply_patches_only_under_floor_kind_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_kubectl(context: str, *args: str, check: bool = True):  # noqa: ANN202, ARG001
        calls.append(list(args))
        if args[:2] == ("get", "nodes"):
            return (
                0,
                "ci-control-plane kind://docker/ci/ci-control-plane 4 3900m\n"
                "gpu-0 kind://docker/ci/gpu-0 64 63400m\n",
                "",
            )
        return 0, "", ""

    monkeypatch.setattr(cluster_mod, "_kubectl", fake_kubectl)

    patched = await cluster_mod.apply_cpu_overcommit("kind-ci", 32)

    assert patched == ["ci-control-plane"]
    patch_calls = [c for c in calls if c[:2] == ["patch", "node"]]
    assert len(patch_calls) == 1
    assert patch_calls[0][2] == "ci-control-plane"
    assert "--subresource=status" in patch_calls[0]


def test_reconcile_loop_runs_off_the_pytest_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-assert loop must live on a daemon thread, not a pytest-asyncio
    loop, so it keeps firing while function-scoped test loops are suspended."""
    passes = threading.Event()

    async def fake_apply(context: str, floor_cpus: int):  # noqa: ANN202, ARG001
        passes.set()
        return []

    monkeypatch.setattr(cluster_mod, "apply_cpu_overcommit", fake_apply)

    thread, stop = cluster_mod.start_cpu_overcommit_thread("kind-ci", 32)
    try:
        assert thread.daemon is True
        assert passes.wait(timeout=5.0), "overcommit loop never ran a pass"
    finally:
        stop.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
