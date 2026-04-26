# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.sweep_controller.status_writer import (
    SWEEP_CONTROLLER_FIELD_MANAGER,
    SweepStatusWriter,
)


@pytest.mark.asyncio
async def test_aggregation_running_patches_status(monkeypatch):
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_running()

    custom.patch_namespaced_custom_object_status.assert_awaited_once()
    call = custom.patch_namespaced_custom_object_status.call_args
    body = call.kwargs.get("body") or call.args[-1]
    assert body["status"]["aggregation"]["phase"] == "Running"


@pytest.mark.asyncio
async def test_aggregation_complete_sets_aggregate_ref(monkeypatch):
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_complete(
        aggregate_path="/api/v1/results/ns/s/aggregate",
        controller_host="host",
        port=19090,
    )
    body = custom.patch_namespaced_custom_object_status.call_args.kwargs.get("body")
    if body is None:
        body = custom.patch_namespaced_custom_object_status.call_args.args[-1]
    assert body["status"]["aggregation"]["phase"] == "Complete"
    assert body["status"]["aggregateRef"]["resultsServerHost"] == "host"
    assert body["status"]["aggregateRef"]["port"] == 19090


@pytest.mark.asyncio
async def test_current_cell_writes_index_label_trial(monkeypatch):
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )
    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.current_cell(variation_index=7, label="c=64", trial=3, converged=False)
    body = custom.patch_namespaced_custom_object_status.call_args.kwargs.get("body")
    if body is None:
        body = custom.patch_namespaced_custom_object_status.call_args.args[-1]
    assert body["status"]["currentCell"]["variationIndex"] == 7
    assert body["status"]["currentCell"]["label"] == "c=64"
    assert body["status"]["currentCell"]["trial"] == 3


def test_field_manager_constant():
    assert SWEEP_CONTROLLER_FIELD_MANAGER == "aiperf-sweep-controller"
