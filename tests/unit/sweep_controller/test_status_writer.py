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
# Adversarial regression-locks: content-type and field-manager on every patch.
#
# The status writer uses merge-patch+json with `field_manager=
# aiperf-sweep-controller` as observability metadata. Without
# `_content_type="application/merge-patch+json"`, kubernetes_asyncio defaults
# to `application/json-patch+json`, which expects a JSON-Patch list of ops
# (not a dict body) — the apiserver returns 422 and the patch is silently
# swallowed.
#
# Server-Side Apply was tried and reverted: SSA's relinquishment semantics
# drop a single manager's previously-set fields between calls when the new
# apply body doesn't include them, which broke the imperative event-style
# write pattern (e.g. aggregation_running would erase currentCell). The
# disjoint-top-level-field invariant between operator and controller writers
# means merge-patch is the right primitive here.
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
async def test_patch_uses_merge_patch_content_type_for_every_writer(
    method_name: str, monkeypatch
):
    """Every status writer call MUST set _content_type=application/merge-patch+json.

    Regression-lock: missing this kwarg silently 422s on the apiserver and
    the sweep CR never reflects the controller's progress. Also locks the
    revert from the SSA experiment — apply-patch+yaml drops the writer's
    own previously-set fields between calls.
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
    assert kwargs.get("_content_type") == "application/merge-patch+json", (
        f"{method_name}: expected merge-patch content-type, got {kwargs.get('_content_type')!r}"
    )
    # SSA-only kwargs must NOT be present — they would change semantics.
    assert "force" not in kwargs or kwargs["force"] is None, (
        f"{method_name}: force={kwargs.get('force')!r} leaked from the SSA "
        "experiment; merge-patch must not pass force=True"
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


# ---------------------------------------------------------------------------
# Regression: top-level `status.completionTime` (CRD-declared name) MUST be
# written by both terminal writers. Without it, the operator's TTL reaper
# (`operator/handlers/sweep/lifecycle.maybe_reap_finished`) falls back to
# `metadata.creationTimestamp` and reaps long-running sweeps mid-flight.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregation_complete_writes_top_level_completion_time(monkeypatch):
    """`aggregation_complete` writes top-level `status.completionTime` (CRD name).

    The TTL reaper reads exactly this field — `status.completedAt` is the
    nested aggregation timestamp (present too, but in `aggregation`), not
    the top-level field the CRD declares.
    """
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
    assert body["status"]["completionTime"]
    # Top-level completionTime equals nested aggregation.completedAt: same
    # event, two field-paths (CRD vs. internal aggregation phase tracking).
    assert (
        body["status"]["completionTime"] == body["status"]["aggregation"]["completedAt"]
    )


@pytest.mark.asyncio
async def test_aggregation_failed_writes_top_level_completion_time(monkeypatch):
    """`aggregation_failed` writes top-level `status.completionTime` (CRD name)."""
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_failed(error="boom")
    body = _patch_call_kwargs(custom)["body"]
    assert body["status"]["completionTime"]
    assert (
        body["status"]["completionTime"] == body["status"]["aggregation"]["completedAt"]
    )


@pytest.mark.asyncio
async def test_aggregation_failed_promotes_top_level_phase_to_failed(monkeypatch):
    """`aggregation_failed` MUST also write top-level `status.phase=Failed`.

    Otherwise, after an aggregation exception, only `status.aggregation.phase`
    becomes `Failed` while top-level `status.phase` is stuck at `Aggregating`
    forever (the rollup advanced it out of `Running` and refuses to clobber
    its own non-terminal write back to a terminal value).

    The rollup's `_conditional_phase_set` skips writes when `parent_phase`
    is already in `PARENT_TERMINAL_PHASES`, so this merge-patch wins races
    with concurrent rollup ticks.
    """
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.aggregation_failed(error="boom")
    body = _patch_call_kwargs(custom)["body"]
    assert body["status"]["phase"] == "Failed"


# ---------------------------------------------------------------------------
# parent_running — Pending -> Running phase promotion (new in this slice).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_running_uses_json_patch_test_op(monkeypatch):
    """`parent_running` issues a json-patch `test`+`replace` (atomic vs. peers).

    Without the test op, a concurrent terminal write from the rollup or
    aggregation-complete path could be clobbered back to ``Running``.
    """
    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.parent_running()

    kwargs = custom.patch_namespaced_custom_object_status.call_args.kwargs
    body = kwargs["body"]
    assert isinstance(body, list), "must use json-patch (list of ops), not merge-patch"
    assert body[0] == {"op": "test", "path": "/status/phase", "value": "Pending"}
    assert body[1] == {"op": "replace", "path": "/status/phase", "value": "Running"}
    assert kwargs["_content_type"] == "application/json-patch+json"


@pytest.mark.asyncio
async def test_parent_running_swallows_422_test_failed(monkeypatch):
    """422 from the json-patch `test` op (peer already advanced phase) is silent."""
    from kubernetes_asyncio.client import ApiException

    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=422, reason="Unprocessable Entity")
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    # Must not raise.
    await writer.parent_running()


@pytest.mark.asyncio
async def test_parent_running_swallows_404_cr_gone(monkeypatch):
    """404 (CR deleted while patching) is also a silent no-op."""
    from kubernetes_asyncio.client import ApiException

    api = MagicMock()
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=404, reason="Not Found")
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.CustomObjectsApi", lambda _api: custom
    )

    writer = SweepStatusWriter(api, name="s", namespace="ns")
    await writer.parent_running()
