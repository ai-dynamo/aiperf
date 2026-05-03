# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch-driven JobSet terminal-condition handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.handlers.jobset_terminal import handle_jobset_conditions


@pytest.mark.asyncio
async def test_completed_condition_triggers_annotation_patch() -> None:
    """When the JobSet flips to type=Completed/status=True, set BENCHMARK_COMPLETE on the parent."""
    new_conditions = [
        {"type": "Completed", "status": "True", "reason": "AllJobsCompleted"},
    ]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(
            return_value={"metadata": {"name": "ajob", "annotations": {}}, "status": {}}
        ),
    ), patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new_conditions, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_awaited_once_with("ns", "ajob")


@pytest.mark.asyncio
async def test_non_terminal_condition_change_does_nothing() -> None:
    """A non-terminal condition (Suspended) is a no-op for this handler."""
    new = [{"type": "Suspended", "status": "True"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter, patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(),
    ) as lookup:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_false_status_does_nothing() -> None:
    """A Completed condition with status=False is not terminal-success."""
    new = [{"type": "Completed", "status": "False"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_condition_is_no_op() -> None:
    """type=Failed/status=True stays on the existing monitor-timer recovery path."""
    new = [{"type": "Failed", "status": "True", "reason": "ControllerCrashed"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter, patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(),
    ) as lookup:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_completed_in_old_conditions_skips() -> None:
    """Re-firing on the same Completed condition list is a no-op (saves a CR get)."""
    completed = [{"type": "Completed", "status": "True"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(),
    ) as lookup, patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=completed,
            new=completed,
            namespace="ns",
            jobset_name="aiperf-ajob",
        )
    lookup.assert_not_awaited()
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_annotation_skips_redundant_patch() -> None:
    """If the controller pod already set BENCHMARK_COMPLETE, skip the redundant patch."""
    new = [{"type": "Completed", "status": "True"}]
    body = {
        "metadata": {
            "name": "ajob",
            "annotations": {Annotations.BENCHMARK_COMPLETE: "true"},
        },
        "status": {},
    }
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(return_value=body),
    ), patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_owned_jobset_skips_silently() -> None:
    """Sweep-owned JobSets resolve to a non-existent AIPerfJob CR and skip."""
    new = [{"type": "Completed", "status": "True"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(return_value=None),
    ), patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="aiperf-someweep"
        )
    setter.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobset_name_without_aiperf_prefix_skips() -> None:
    """A JobSet whose name doesn't start with 'aiperf-' is not ours."""
    new = [{"type": "Completed", "status": "True"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
        new=AsyncMock(return_value=None),
    ), patch(
        "aiperf.operator.handlers.jobset_terminal._set_benchmark_complete_annotation",
        new=AsyncMock(),
    ) as setter:
        await handle_jobset_conditions(
            old=[], new=new, namespace="ns", jobset_name="some-other-jobset"
        )
    setter.assert_not_awaited()
