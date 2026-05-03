# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch-driven pod-restart handler."""

from __future__ import annotations

from unittest.mock import patch as mock_patch

import pytest

from aiperf.operator.client_cache import _warned_pod_restarts
from aiperf.operator.handlers.pod_restarts import handle_pod_restart


@pytest.fixture(autouse=True)
def _clear_warned_restarts() -> None:
    """Reset the module-level dedup state so tests don't leak state."""
    _warned_pod_restarts.clear()
    yield
    _warned_pod_restarts.clear()


@pytest.mark.asyncio
async def test_emits_event_when_threshold_exceeded() -> None:
    """When a containerStatuses entry has restartCount above the threshold, emit one event."""
    pod_body = {
        "metadata": {
            "name": "controller-0",
            "namespace": "bench",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
            "ownerReferences": [
                {"kind": "Job", "name": "aiperf-bench-controller-0"},
            ],
        },
    }
    new_statuses = [
        {
            "name": "controller",
            "restartCount": 5,
            "lastState": {"terminated": {"reason": "OOMKilled"}},
            "state": {},
        }
    ]
    with mock_patch(
        "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
    ) as evt:
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            return_value=pod_body,
        ):
            await handle_pod_restart(
                old=[],
                new=new_statuses,
                body=pod_body,
                meta=pod_body["metadata"],
                namespace="bench",
                name="controller-0",
                threshold=3,
            )
    evt.assert_called_once()


@pytest.mark.asyncio
async def test_does_not_emit_below_threshold() -> None:
    pod_body = {"metadata": {"name": "controller-0", "namespace": "bench"}}
    with mock_patch(
        "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
    ) as evt:
        await handle_pod_restart(
            old=[],
            new=[{"name": "controller", "restartCount": 1}],
            body=pod_body,
            meta=pod_body["metadata"],
            namespace="bench",
            name="controller-0",
            threshold=3,
        )
    evt.assert_not_called()


@pytest.mark.asyncio
async def test_dedup_same_count_only_emits_once() -> None:
    pod_body = {
        "metadata": {
            "name": "controller-0",
            "namespace": "bench",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "aiperf-bench"},
        },
    }
    new_statuses = [{"name": "controller", "restartCount": 5}]
    with mock_patch(
        "aiperf.operator.handlers.pod_restarts.events.pod_restarts"
    ) as evt:
        with mock_patch(
            "aiperf.operator.handlers.pod_restarts._lookup_aiperfjob_body",
            return_value=pod_body,
        ):
            await handle_pod_restart(
                old=[],
                new=new_statuses,
                body=pod_body,
                meta=pod_body["metadata"],
                namespace="bench",
                name="controller-0",
                threshold=3,
            )
            await handle_pod_restart(
                old=new_statuses,
                new=new_statuses,
                body=pod_body,
                meta=pod_body["metadata"],
                namespace="bench",
                name="controller-0",
                threshold=3,
            )
    assert evt.call_count == 1
