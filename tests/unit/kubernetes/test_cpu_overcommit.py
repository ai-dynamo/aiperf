# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU-overcommit scheduling floor for local Kind test clusters.

The weekly audit workflow failed 4/4 scheduled runs because the operator's
production-sized pod requests (~6 CPUs per AIPerfJob plus a 2-CPU mock
server) can never be admitted by a single-node Kind cluster on a 4-vCPU
ubuntu-latest runner; pods sat Pending with FailedScheduling until timeout.
Suites may opt into raising the node's advertised CPU floor instead of
weakening the pod requests, so production defaults keep their meaning.

Two non-obvious invariants are guarded here:

- Kubelet re-asserts real capacity on every node-status sync (measured:
  <=30s on kind v0.27.0 / k8s v1.32.2), so reconciliation must run
  continuously AND independently of pytest-asyncio's loop scoping — the
  package fixture's loop is suspended for the whole duration of each
  function-scoped test loop, which is exactly when pods schedule. The
  reconciler therefore runs on a plain daemon thread.
- The floor only ever RAISES capacity/allocatable (per field), and only
  nodes carrying the kind:// providerID marker are eligible, so a repointed
  kubectl context can never fake capacity on a remote cluster.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from pytest import param

from tests.kubernetes.conftest import _resolve_settings
from tests.kubernetes.helpers.cluster import (
    CpuOvercommitReconciler,
    node_cpu_overcommit_patch,
    parse_cpu_millicores,
    select_overcommit_targets,
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
    assert json.loads(patch) == [
        {"op": "replace", "path": "/status/capacity/cpu", "value": "32"},
        {"op": "replace", "path": "/status/allocatable/cpu", "value": "32"},
    ]


def test_patch_raises_each_field_independently() -> None:
    """capacity 64 / allocatable 31: allocatable rises, capacity is NOT shrunk."""
    patch = node_cpu_overcommit_patch("64", "31", 32)

    assert patch is not None
    assert json.loads(patch) == [
        {"op": "replace", "path": "/status/allocatable/cpu", "value": "32"},
    ]


@pytest.mark.parametrize(
    ("capacity", "allocatable"),
    [
        param("32", "32", id="exactly_at_floor"),
        param("64", "63400m", id="above_floor"),
    ],
)
def test_patch_never_shrinks_large_nodes(capacity: str, allocatable: str) -> None:
    """GPU/dev nodes with real capacity at or above the floor are untouched."""
    assert node_cpu_overcommit_patch(capacity, allocatable, 32) is None


def test_select_targets_requires_kind_provider_id() -> None:
    """Locality gate: only kind:// nodes may have their capacity faked."""
    nodes = [
        ("kind-cp", "kind://docker/aiperf/aiperf-control-plane", "4", "4"),
        ("remote-1", "aws:///us-west-2/i-0abc123", "4", "4"),
        ("gke-node", "gce://proj/zone/vm", "96", "96"),
    ]

    targets, skipped = select_overcommit_targets(nodes, 32)

    assert [name for name, _ in targets] == ["kind-cp"]
    assert skipped == ["remote-1", "gke-node"]


def test_select_targets_skips_nodes_already_at_floor() -> None:
    nodes = [("big-kind", "kind://docker/c/c-control-plane", "64", "64")]

    targets, skipped = select_overcommit_targets(nodes, 32)

    assert targets == []
    assert skipped == []


def test_settings_default_is_opt_in_off() -> None:
    """The floor must be opt-in: shared suites (chaos, main gate) opt in
    explicitly from their own workflow/fixture configuration."""
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 0


def test_settings_env_override_and_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "32")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 32

    monkeypatch.setenv("K8S_TEST_NODE_CPU_OVERCOMMIT", "0")
    assert _resolve_settings(_FakeConfig()).node_cpu_overcommit == 0


def test_reconciler_start_applies_eagerly_and_propagates_failure() -> None:
    calls: list[int] = []

    def _failing_apply() -> list[str]:
        calls.append(1)
        raise RuntimeError("kubectl exploded")

    reconciler = CpuOvercommitReconciler("kind-x", 32, apply_fn=_failing_apply)
    with pytest.raises(RuntimeError, match="kubectl exploded"):
        reconciler.start()
    assert calls == [1]
    reconciler.stop()  # never started; must be a no-op


def test_reconciler_thread_runs_without_any_event_loop() -> None:
    """Liveness proof: reconciliation keeps firing while the calling thread
    runs NO asyncio loop — the exact situation while a function-scoped
    pytest-asyncio test body occupies its own loop and the package fixture's
    loop (where an asyncio task would live) is suspended."""
    calls: list[float] = []

    def _apply() -> list[str]:
        calls.append(time.monotonic())
        return []

    reconciler = CpuOvercommitReconciler(
        "kind-x", 32, interval_seconds=0.05, apply_fn=_apply
    )
    reconciler.start()
    assert reconciler.apply_count == 1  # eager apply in start()

    # Block this thread with no event loop running; the daemon thread must
    # keep reconciling through the blockage.
    time.sleep(0.3)
    reconciler.stop()

    assert reconciler.apply_count >= 3, (
        f"reconciler fired {reconciler.apply_count}x in ~0.3s at 0.05s "
        "interval; it must not depend on any asyncio event loop"
    )


def test_reconciler_stop_join_verifies_thread_exit() -> None:
    reconciler = CpuOvercommitReconciler(
        "kind-x", 32, interval_seconds=60.0, apply_fn=lambda: []
    )
    reconciler.start()
    thread = reconciler._thread

    reconciler.stop(timeout=5.0)

    assert thread is not None and not thread.is_alive()
    # stop() is idempotent after a successful join
    reconciler.stop()
