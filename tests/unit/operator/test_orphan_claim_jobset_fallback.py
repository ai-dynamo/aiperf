# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``_benchmark_appears_complete`` no-controller-pod fallback.

When the operator pod crashes after ``_maybe_delete_jobset_after_success``
deletes the JobSet (success-only path), the controller pod is gone.
Without a fallback, ``_benchmark_appears_complete`` returns False and
orphan-claim recovery never fires — the AIPerfJob CR sticks at phase
``Initializing`` forever.

The fallback queries the JobSet itself: a 404 (deleted) OR a
``Completed``/``Failed`` condition is unambiguous evidence that the
benchmark is done and recovery is safe.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.operator.handlers.monitor import (
    _benchmark_appears_complete,
    _jobset_has_terminal_condition,
)


@pytest.mark.asyncio
async def test_jobset_404_returns_terminal():
    """A deleted JobSet (404) signals successful completion-then-cleanup."""
    api = MagicMock()
    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=404)
    )
    with patch(
        "aiperf.operator.handlers.monitor.CustomObjectsApi",
        return_value=fake_custom,
    ):
        assert await _jobset_has_terminal_condition(api, "ns", "js") is True


@pytest.mark.asyncio
async def test_jobset_completed_condition_returns_terminal():
    api = MagicMock()
    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "status": {
                "conditions": [
                    {"type": "Completed", "status": "True"},
                    {"type": "Suspended", "status": "False"},
                ]
            }
        }
    )
    with patch(
        "aiperf.operator.handlers.monitor.CustomObjectsApi",
        return_value=fake_custom,
    ):
        assert await _jobset_has_terminal_condition(api, "ns", "js") is True


@pytest.mark.asyncio
async def test_jobset_failed_condition_returns_terminal():
    api = MagicMock()
    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"conditions": [{"type": "Failed", "status": "True"}]}}
    )
    with patch(
        "aiperf.operator.handlers.monitor.CustomObjectsApi",
        return_value=fake_custom,
    ):
        assert await _jobset_has_terminal_condition(api, "ns", "js") is True


@pytest.mark.asyncio
async def test_jobset_running_returns_non_terminal():
    api = MagicMock()
    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"conditions": []}}
    )
    with patch(
        "aiperf.operator.handlers.monitor.CustomObjectsApi",
        return_value=fake_custom,
    ):
        assert await _jobset_has_terminal_condition(api, "ns", "js") is False


@pytest.mark.asyncio
async def test_jobset_500_returns_non_terminal():
    """Apiserver hiccup falls through to no-evidence; caller retries next tick."""
    api = MagicMock()
    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=500)
    )
    with patch(
        "aiperf.operator.handlers.monitor.CustomObjectsApi",
        return_value=fake_custom,
    ):
        assert await _jobset_has_terminal_condition(api, "ns", "js") is False


@pytest.mark.asyncio
async def test_appears_complete_falls_back_to_jobset_when_pod_gone():
    """If the controller pod is gone but the JobSet is gone or terminal, return True.

    This is the regression test for the operator-restart-after-jobset-delete
    case that left AIPerfJob CRs stuck at phase=Initializing forever.
    """
    api = MagicMock()
    progress_client = MagicMock()
    progress_client.get_progress = AsyncMock(
        return_value=MagicMock(connection_error=True, is_complete=False)
    )
    with (
        patch(
            "aiperf.operator.handlers.monitor.get_or_create_progress_client",
            AsyncMock(return_value=progress_client),
        ),
        patch(
            "aiperf.operator.handlers.monitor._get_controller_pod",
            AsyncMock(return_value=None),
        ),
        patch(
            "aiperf.operator.handlers.monitor._jobset_has_terminal_condition",
            AsyncMock(return_value=True),
        ),
    ):
        result = await _benchmark_appears_complete(
            api=api, namespace="ns", jobset_name="js", key="ns/js"
        )
    assert result is True


@pytest.mark.asyncio
async def test_appears_complete_pod_gone_jobset_running_is_no_evidence():
    """Pod gone + JobSet still running = no orphan recovery yet (expected behavior)."""
    api = MagicMock()
    progress_client = MagicMock()
    progress_client.get_progress = AsyncMock(
        return_value=MagicMock(connection_error=True, is_complete=False)
    )
    with (
        patch(
            "aiperf.operator.handlers.monitor.get_or_create_progress_client",
            AsyncMock(return_value=progress_client),
        ),
        patch(
            "aiperf.operator.handlers.monitor._get_controller_pod",
            AsyncMock(return_value=None),
        ),
        patch(
            "aiperf.operator.handlers.monitor._jobset_has_terminal_condition",
            AsyncMock(return_value=False),
        ),
    ):
        result = await _benchmark_appears_complete(
            api=api, namespace="ns", jobset_name="js", key="ns/js"
        )
    assert result is False
