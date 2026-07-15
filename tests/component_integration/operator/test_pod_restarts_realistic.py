# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Realistic component-integration tests for handle_pod_restart.

Drives the post-fix-A/B implementation against an in-memory fake apiserver
with a focus on:
  - Above-threshold detection across multiple pods
  - (pod, restart_count) dedup semantics under varied traffic
  - The bug-A fix: no apiserver lookup when nothing is above threshold
  - The bug-B fix: sweep-owned JobSets do not leak _warned_pod_restarts entries
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch as mock_patch

import pytest

from aiperf.kubernetes.cr_refs import AIPERF_PLURAL
from aiperf.operator import client_cache
from aiperf.operator.client_cache import _warned_pod_restarts
from aiperf.operator.handlers.pod_restarts import handle_pod_restart
from tests.component_integration.operator._fake_apiserver import FakeApiserver

NS = "bench"


def _ajob_body(name: str) -> dict[str, Any]:
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {"name": name, "namespace": NS, "annotations": {}},
        "spec": {},
        "status": {"jobId": name, "jobSetName": f"aiperf-{name}", "phase": "Running"},
    }


def _pod_meta(jobset: str, pod_name: str) -> dict[str, Any]:
    return {
        "name": pod_name,
        "namespace": NS,
        "labels": {"jobset.sigs.k8s.io/jobset-name": jobset},
    }


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    client_cache._reset_for_testing()
    yield
    client_cache._reset_for_testing()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_three_pods_mixed_restart_counts() -> None:
    """Three pods: A=5, B=1, C=7 (threshold=3). Expect events for A and C only."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "j", _ajob_body("j"))

    events_seen: list[tuple[str, int]] = []

    def _record(_body, pod, count, _reason):
        events_seen.append((pod, count))

    with (
        fake.context(),
        mock_patch(
            "aiperf.operator.handlers.pod_restarts.events.pod_restarts",
            side_effect=_record,
        ),
    ):
        for pod, count in [("pod-A", 5), ("pod-B", 1), ("pod-C", 7)]:
            await handle_pod_restart(
                old=[],
                new=[{"name": "ctr", "restartCount": count}],
                body={"metadata": _pod_meta("aiperf-j", pod)},
                meta=_pod_meta("aiperf-j", pod),
                namespace=NS,
                name=pod,
                threshold=3,
            )

    assert events_seen == [("pod-A", 5), ("pod-C", 7)]


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_same_pod_same_count_dedups() -> None:
    """Two fires with the same (pod, count) → one event."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "j", _ajob_body("j"))

    with (
        fake.context(),
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
    ):
        for _ in range(2):
            await handle_pod_restart(
                old=[],
                new=[{"name": "ctr", "restartCount": 5}],
                body={"metadata": _pod_meta("aiperf-j", "pod-A")},
                meta=_pod_meta("aiperf-j", "pod-A"),
                namespace=NS,
                name="pod-A",
                threshold=3,
            )
    assert evt.call_count == 1


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_monotonically_increasing_count_each_emits() -> None:
    """count 3 → 5 → 8 produces three events: each is a fresh (pod, count) key."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "j", _ajob_body("j"))

    with (
        fake.context(),
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
    ):
        for c in [3, 5, 8]:
            await handle_pod_restart(
                old=[],
                new=[{"name": "ctr", "restartCount": c}],
                body={"metadata": _pod_meta("aiperf-j", "pod-A")},
                meta=_pod_meta("aiperf-j", "pod-A"),
                namespace=NS,
                name="pod-A",
                threshold=3,
            )
    assert evt.call_count == 3


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_multi_container_dedup_per_restart_count() -> None:
    """Two containers with DIFFERENT counts → 2 events; same counts → 1 event.

    This pins the dedup-key semantic: ``(pod_name, restart_count)``, not
    per-container.
    """
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "j", _ajob_body("j"))

    with (
        fake.context(),
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
    ):
        await handle_pod_restart(
            old=[],
            new=[
                {"name": "c1", "restartCount": 5},
                {"name": "c2", "restartCount": 7},
            ],
            body={"metadata": _pod_meta("aiperf-j", "pod-A")},
            meta=_pod_meta("aiperf-j", "pod-A"),
            namespace=NS,
            name="pod-A",
            threshold=3,
        )
        assert evt.call_count == 2

        # Same pod, both containers same count → just one new event.
        await handle_pod_restart(
            old=[],
            new=[
                {"name": "c1", "restartCount": 9},
                {"name": "c2", "restartCount": 9},
            ],
            body={"metadata": _pod_meta("aiperf-j", "pod-A")},
            meta=_pod_meta("aiperf-j", "pod-A"),
            namespace=NS,
            name="pod-A",
            threshold=3,
        )
        assert evt.call_count == 3, "exactly one new event for the (pod-A, 9) key"


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_sweep_owned_pod_does_not_leak_warned_state() -> None:
    """Bug-B regression: 1000 sweep-owned pod events must not populate
    ``_warned_pod_restarts`` because the AIPerfJob lookup returns 404.

    Pre-fix, dedup state was pre-claimed BEFORE the lookup, leaving entries
    keyed by jobset-name that no eviction path ever cleaned up.
    """
    fake = FakeApiserver()
    # No CR added — apiserver returns 404 for every lookup.

    with (
        fake.context(),
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
    ):
        for i in range(1000):
            pod = f"sweep-pod-{i}"
            await handle_pod_restart(
                old=[],
                new=[{"name": "ctr", "restartCount": 5}],
                body={"metadata": _pod_meta("aiperf-someweep", pod)},
                meta=_pod_meta("aiperf-someweep", pod),
                namespace=NS,
                name=pod,
                threshold=3,
            )

    evt.assert_not_called()
    assert _warned_pod_restarts == {}, (
        "_warned_pod_restarts must remain empty for sweep-owned pods; "
        "see bug-B fix in handlers/pod_restarts.py:handle_pod_restart"
    )


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_below_threshold_traffic_skips_apiserver_lookup() -> None:
    """Bug-A regression: 1000 healthy pods at restartCount=0 must not pay
    the apiserver round-trip for the AIPerfJob lookup."""
    fake = FakeApiserver()
    fake.add_cr(NS, AIPERF_PLURAL, "j", _ajob_body("j"))

    # Spy on the lookup helper without intercepting (the early-out lives
    # before this call site, so a real spy on the real function is what we
    # want — but we also need apiserver behavior, so we route through fake).
    lookup_calls = 0
    real_lookup = __import__(
        "aiperf.operator.handlers.pod_restarts", fromlist=["_lookup_aiperfjob_body"]
    )._lookup_aiperfjob_body

    async def counting_lookup(namespace: str, jobset_name: str) -> Any:
        nonlocal lookup_calls
        lookup_calls += 1
        return await real_lookup(namespace, jobset_name)

    with (
        fake.context(),
        mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            side_effect=counting_lookup,
        ),
        mock_patch("aiperf.operator.handlers.pod_restarts.events.pod_restarts") as evt,
    ):
        for i in range(1000):
            pod = f"healthy-pod-{i}"
            await handle_pod_restart(
                old=[],
                new=[{"name": "ctr", "restartCount": 0}],
                body={"metadata": _pod_meta("aiperf-j", pod)},
                meta=_pod_meta("aiperf-j", pod),
                namespace=NS,
                name=pod,
                threshold=3,
            )

    evt.assert_not_called()
    assert lookup_calls == 0, (
        "below-threshold pod events must NOT trigger an apiserver lookup; "
        "see _has_above_threshold early-out in handlers/pod_restarts.py"
    )
    assert _warned_pod_restarts == {}, (
        "below-threshold traffic must not populate dedup state"
    )
