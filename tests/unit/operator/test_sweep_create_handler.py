# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import kopf
import pytest

from aiperf.operator.handlers.sweep import create as sweep_create


def _valid_body() -> dict:
    return {
        "metadata": {
            "name": "s",
            "namespace": "ns",
            "uid": "u",
            "creationTimestamp": "2024-04-25T18:22:03Z",
        },
        "spec": {
            "multiRun": {"trials": 3},
            "template": {
                "spec": {
                    "image": "x:latest",
                    "benchmark": {
                        "models": ["m"],
                        "endpoint": {"urls": ["http://x"], "type": "chat"},
                        "datasets": {"main": {"type": "synthetic"}},
                        "phases": [
                            {
                                "name": "profiling",
                                "type": "concurrency",
                                "duration": 1,
                                "concurrency": 1,
                            }
                        ],
                    },
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_handle_validates_spec_and_creates_jobset(monkeypatch):
    body = _valid_body()
    patch = kopf.Patch()
    provision_rbac = AsyncMock()
    create_jobset = AsyncMock()
    monkeypatch.setattr(sweep_create, "_provision_rbac", provision_rbac)
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", create_jobset)

    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )

    provision_rbac.assert_awaited_once()
    create_jobset.assert_awaited_once()
    assert patch.status["phase"] == "Pending"
    assert patch.status["totalVariations"] == 1
    assert patch.status["maxTotalRuns"] == 3
    assert "runtimeRef" in patch.status


@pytest.mark.asyncio
async def test_handle_rejects_invalid_spec(monkeypatch):
    body = {
        "metadata": {"name": "s", "namespace": "ns", "uid": "u"},
        "spec": {"template": {"spec": {"benchmark": {}}}},  # no axes
    }
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    with pytest.raises(kopf.PermanentError, match="at least one of"):
        await sweep_create.handle(
            body=body,
            spec=body["spec"],
            name="s",
            namespace="ns",
            patch=patch,
        )


@pytest.mark.asyncio
async def test_epoch_from_creation_timestamp():
    """`metadata.creationTimestamp` parses to a decimal epoch in status.runEpoch."""
    from datetime import datetime, timezone

    expected = int(datetime(2024, 4, 25, 18, 22, 3, tzinfo=timezone.utc).timestamp())
    assert sweep_create._epoch_from_creation_ts("2024-04-25T18:22:03Z") == str(expected)


@pytest.mark.asyncio
async def test_handle_computes_max_total_runs_grid_x_trials(monkeypatch):
    body = _valid_body()
    body["spec"]["sweep"] = {
        "type": "grid",
        "variables": {"random_seed": [1, 2, 3, 4]},
    }
    body["spec"]["multiRun"]["trials"] = 5
    patch = kopf.Patch()
    monkeypatch.setattr(sweep_create, "_provision_rbac", AsyncMock())
    monkeypatch.setattr(sweep_create, "_create_sweep_controller_jobset", AsyncMock())
    await sweep_create.handle(
        body=body,
        spec=body["spec"],
        name="s",
        namespace="ns",
        patch=patch,
    )
    assert patch.status["totalVariations"] == 4
    assert patch.status["maxTotalRuns"] == 20
