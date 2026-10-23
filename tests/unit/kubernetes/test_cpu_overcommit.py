# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-overcommit scheduling floor for local Kind test clusters.

The weekly audit workflow failed 4/4 scheduled runs because the operator's
production-sized pod requests (~6 CPUs per AIPerfJob plus a 2-CPU mock
server) can never be admitted by a single-node Kind cluster on a 4-vCPU
ubuntu-latest runner; pods sat Pending with FailedScheduling until timeout.
The fixture raises the node's advertised CPU floor instead of weakening the
pod requests, so production defaults keep their meaning. Kubelet re-asserts
real capacity on every node-status sync, so the fixture re-applies the patch
on an interval. These tests lock the floor semantics and the setting.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pytest import param

from tests.kubernetes.conftest import _resolve_settings
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
    patch = node_cpu_overcommit_patch("4", 32)

    assert patch is not None
    ops = json.loads(patch)
    assert ops == [
        {"op": "replace", "path": "/status/capacity/cpu", "value": "32"},
        {"op": "replace", "path": "/status/allocatable/cpu", "value": "32"},
    ]


@pytest.mark.parametrize(
    "allocatable",
    [
        param("32", id="exactly_at_floor"),
        param("64", id="above_floor"),
        param("63400m", id="above_floor_millicores"),
    ],
)
def test_patch_never_shrinks_large_nodes(allocatable: str) -> None:
    """GPU/dev nodes with real capacity above the floor must be left alone."""
    assert node_cpu_overcommit_patch(allocatable, 32) is None


def test_settings_default_is_the_kind_floor() -> None:
    settings = _resolve_settings(_FakeConfig())

    assert settings.node_cpu_overcommit == KIND_CPU_OVERCOMMIT_FLOOR
    assert KIND_CPU_OVERCOMMIT_FLOOR >= 32, (
        "the floor must cover a full AIPerfJob (controller ~2.6 CPUs + worker "
        "pod 3.35 CPUs) beside the mock server (2 CPUs), kube-system (~1 CPU), "
        "the JobSet controller (0.5 CPUs), and the operator pod (0.35 CPUs), "
        "with headroom for the chaos suite's concurrent job sets"
    )


def test_settings_env_override_and_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "8")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 8

    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "0")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 0
