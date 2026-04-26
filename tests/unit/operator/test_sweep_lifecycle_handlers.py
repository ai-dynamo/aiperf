# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import kopf
import pytest

from aiperf.operator.handlers.sweep import child_rollup, lifecycle


@pytest.mark.asyncio
async def test_cancel_handler_writes_cancelling_condition():
    patch = kopf.Patch()
    await lifecycle.cancel(
        body={"metadata": {"name": "s"}, "spec": {"cancel": True}},
        spec={"cancel": True},
        name="s",
        namespace="ns",
        patch=patch,
    )
    conditions = patch.status.get("conditions") or []
    assert any(
        c.get("type") == "Cancelling" and c.get("status") == "True" for c in conditions
    )


@pytest.mark.asyncio
async def test_cancel_noop_when_cancel_false():
    """If spec.cancel is false, the handler does nothing."""
    patch = kopf.Patch()
    await lifecycle.cancel(
        body={"metadata": {"name": "s"}, "spec": {"cancel": False}},
        spec={"cancel": False},
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert "conditions" not in patch.status or not patch.status["conditions"]


@pytest.mark.asyncio
async def test_child_rollup_skips_unowned_aiperfjob(monkeypatch):
    """A standalone AIPerfJob (no AIPerfSweep ownerRef) is a no-op."""
    patch_parent = AsyncMock()
    monkeypatch.setattr(child_rollup, "_patch_parent_status", patch_parent)
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 0,
                "failed": 0,
                "in_flight": 0,
                "total_terminal_phase": None,
            }
        ),
    )

    body = {"metadata": {"name": "child", "namespace": "ns", "ownerReferences": []}}
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    patch_parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_child_rollup_increments_counts_and_transitions_phase(monkeypatch):
    """When all children terminal, parent transitions to Aggregating."""
    parent_patches: list[dict] = []

    async def fake_patch(*, group, version, plural, name, namespace, body):
        parent_patches.append(body)

    monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
    monkeypatch.setattr(
        child_rollup,
        "_count_owned_children",
        AsyncMock(
            return_value={
                "completed": 5,
                "failed": 1,
                "in_flight": 0,
                "total_terminal_phase": "Aggregating",
            }
        ),
    )

    body = {
        "metadata": {
            "name": "child",
            "namespace": "ns",
            "ownerReferences": [{"kind": "AIPerfSweep", "name": "s", "uid": "u"}],
        },
    }
    await child_rollup.on_child_phase_transition(
        body=body,
        status={"phase": "Succeeded"},
        name="child",
        namespace="ns",
    )
    assert len(parent_patches) == 1
    body_patch = parent_patches[0]
    assert body_patch["status"]["completedRuns"] == 5
    assert body_patch["status"]["failedRuns"] == 1
    assert body_patch["status"]["phase"] == "Aggregating"
