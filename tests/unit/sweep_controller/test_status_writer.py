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


# ---------------------------------------------------------------------------
# Adversarial regression-locks: SSA contract on every patch.
#
# The status writer uses Server-Side Apply (`application/apply-patch+yaml`)
# with `field_manager=aiperf-sweep-controller` and `force=True`. The body
# carries `apiVersion`, `kind`, `metadata.name`, and the `status` fields the
# controller owns. The earlier merge-patch+json contract was upgraded so
# the apiserver tracks per-field ownership in managedFields and concurrent
# writes from the operator (which uses a distinct field manager) cannot
# clobber the controller's fields.
# ---------------------------------------------------------------------------


def _patch_call_kwargs(custom_mock):
    return custom_mock.patch_namespaced_custom_object_status.call_args.kwargs


async def _invoke(method_name: str, writer: SweepStatusWriter) -> None:
    if method_name == "current_cell":
        await writer.current_cell(variation_index=0, label="v0", trial=0)
    elif method_name == "aggregation_running":
        await writer.aggregation_running()
    elif method_name == "aggregation_complete":
        await writer.aggregation_complete(
            aggregate_path="/api/v1/results/ns/s/aggregate",
            controller_host="ctrl-host",
            port=19090,
        )
    elif method_name == "aggregation_failed":
        await writer.aggregation_failed(error="boom")
    else:
        raise ValueError(method_name)


@pytest.mark.parametrize(
    "method_name",
    [
        "current_cell",
        "aggregation_running",
        "aggregation_complete",
        "aggregation_failed",
    ],
)
@pytest.mark.asyncio
async def test_patch_uses_apply_patch_content_type_for_every_writer(
    method_name: str, monkeypatch
):
    """Every status writer call MUST set _content_type=application/apply-patch+yaml.

    Regression-lock: missing this kwarg falls back to a non-SSA path and the
    apiserver does not track per-field ownership, allowing the operator's
    rollup writes to clobber the controller's status.
    """
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await _invoke(method_name, writer)

    kwargs = _patch_call_kwargs(custom)
    assert kwargs.get("_content_type") == "application/apply-patch+yaml", (
        f"{method_name}: expected SSA content-type, got {kwargs.get('_content_type')!r}"
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "current_cell",
        "aggregation_running",
        "aggregation_complete",
        "aggregation_failed",
    ],
)
@pytest.mark.asyncio
async def test_patch_uses_force_true_for_every_writer(method_name: str, monkeypatch):
    """SSA force=True is required so a controller restart can re-claim its fields."""
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await _invoke(method_name, writer)

    kwargs = _patch_call_kwargs(custom)
    assert kwargs.get("force") is True, (
        f"{method_name}: SSA must pass force=True, got {kwargs.get('force')!r}"
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "current_cell",
        "aggregation_running",
        "aggregation_complete",
        "aggregation_failed",
    ],
)
@pytest.mark.asyncio
async def test_patch_body_carries_apiversion_kind_metadata(
    method_name: str, monkeypatch
):
    """SSA bodies MUST include apiVersion, kind, and metadata.name."""
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await _invoke(method_name, writer)

    body = _patch_call_kwargs(custom).get("body")
    assert body is not None, f"{method_name}: body kwarg missing"
    assert body.get("apiVersion") == "aiperf.nvidia.com/v1alpha1", (
        f"{method_name}: SSA body apiVersion is {body.get('apiVersion')!r}"
    )
    assert body.get("kind") == "AIPerfSweep"
    assert (body.get("metadata") or {}).get("name") == "s"
    # Status content lives at top level of body["status"], not nested under
    # an outer "status" key (which was the merge-patch shape).
    assert "status" in body, f"{method_name}: body missing status key"


@pytest.mark.parametrize(
    "method_name",
    [
        "current_cell",
        "aggregation_running",
        "aggregation_complete",
        "aggregation_failed",
    ],
)
@pytest.mark.asyncio
async def test_patch_uses_sweep_controller_field_manager_for_every_writer(
    method_name: str, monkeypatch
):
    """Every status writer call MUST set field_manager=aiperf-sweep-controller.

    Required for SSA co-ownership with the operator (which writes phase /
    completedRuns / etc.). Without it, conflict resolution fails on shared paths.
    """
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await _invoke(method_name, writer)

    kwargs = _patch_call_kwargs(custom)
    assert kwargs.get("field_manager") == SWEEP_CONTROLLER_FIELD_MANAGER


@pytest.mark.asyncio
async def test_aggregation_complete_writes_full_aggregate_ref(monkeypatch):
    """Locked-in payload shape: aggregateRef carries apiPath, port, resultsServerHost."""
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_complete(
        aggregate_path="/api/v1/results/ns/s/aggregate",
        controller_host="ctrl-host",
        port=19090,
    )
    body = _patch_call_kwargs(custom)["body"]

    aggregation = body["status"]["aggregation"]
    assert aggregation["phase"] == "Complete"
    assert "completedAt" in aggregation and aggregation["completedAt"]

    aggregate_ref = body["status"]["aggregateRef"]
    assert aggregate_ref["apiPath"] == "/api/v1/results/ns/s/aggregate"
    assert aggregate_ref["port"] == 19090
    assert aggregate_ref["resultsServerHost"] == "ctrl-host"


@pytest.mark.asyncio
async def test_aggregation_failed_writes_error_and_completed_at(monkeypatch):
    """Locked-in payload shape: aggregation_failed carries error and completedAt."""
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_failed(error="export blew up at row 17")
    body = _patch_call_kwargs(custom)["body"]

    aggregation = body["status"]["aggregation"]
    assert aggregation["phase"] == "Failed"
    assert aggregation["error"] == "export blew up at row 17"
    assert aggregation["completedAt"]  # non-empty
