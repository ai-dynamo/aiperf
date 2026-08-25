# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch-driven JobSet terminal-condition handler."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_API_VERSION,
    AIPERF_SWEEP_API_VERSION,
)
from aiperf.operator.handlers.jobset_terminal import (
    _patch_sweep_controller_failure,
    handle_jobset_conditions,
)


def _aiperfjob_body(*, annotations: dict[str, str] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": "ajob",
        "uid": "uid-ajob",
        "resourceVersion": "42",
    }
    if annotations is not None:
        metadata["annotations"] = annotations
    return {"metadata": metadata, "status": {}}


def _trusted_jobset_body() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "aiperf-ajob",
            "labels": {
                AIPerfLabels.APP_KEY: AIPerfLabels.APP_VALUE,
                AIPerfLabels.JOB_ID: "ajob",
            },
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_JOB_API_VERSION,
                    "kind": "AIPerfJob",
                    "name": "ajob",
                    "uid": "uid-ajob",
                    "controller": True,
                }
            ],
        }
    }


def _aiperfsweep_body(*, phase: str = "Running") -> dict[str, Any]:
    return {
        "metadata": {
            "name": "sweep",
            "uid": "uid-sweep",
            "resourceVersion": "42",
        },
        "status": {"phase": phase, "completedRuns": 1},
    }


def _trusted_sweep_jobset_body() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "aiperf-sweep",
            "ownerReferences": [
                {
                    "apiVersion": AIPERF_SWEEP_API_VERSION,
                    "kind": "AIPerfSweep",
                    "name": "sweep",
                    "uid": "uid-sweep",
                    "controller": True,
                }
            ],
        }
    }


class _ApiContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_failed_condition_dispatches_aiperfjob_recovery() -> None:
    """An AIPerfJob failure enters the existing classifier from the watch."""
    new = [{"type": "Failed", "status": "True", "reason": "ControllerCrashed"}]
    parent = _aiperfjob_body()
    parent["metadata"]["namespace"] = "ns"
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(return_value=parent),
        ) as lookup,
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfsweep_body",
            new=AsyncMock(),
        ) as sweep_lookup,
        patch(
            "aiperf.operator.handlers.monitor.handle_jobset_failure_event",
            new=AsyncMock(),
        ) as recovery,
    ):
        await handle_jobset_conditions(
            old=[],
            new=new,
            namespace="ns",
            jobset_name="aiperf-ajob",
            jobset_body=_trusted_jobset_body(),
        )
    lookup.assert_awaited_once()
    recovery.assert_awaited_once_with(
        body=parent,
        jobset_body={
            **_trusted_jobset_body(),
            "status": {"conditions": new},
        },
        namespace="ns",
        name="ajob",
    )
    sweep_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_sweep_controller_jobset_terminalizes_exact_parent() -> None:
    """A trusted sweep-controller failure reaches the parent failure writer."""
    conditions = [
        {
            "type": "Failed",
            "status": "True",
            "reason": "ReplicatedJobFailed",
            "message": "controller exceeded its backoff limit",
        }
    ]
    parent = _aiperfsweep_body()
    with (
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfsweep_body",
            new=AsyncMock(return_value=parent),
        ) as lookup,
        patch(
            "aiperf.operator.handlers.jobset_terminal._patch_sweep_controller_failure",
            new=AsyncMock(),
        ) as failure_patch,
        patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as job_lookup,
    ):
        await handle_jobset_conditions(
            old=[],
            new=conditions,
            namespace="ns",
            jobset_name="aiperf-sweep",
            jobset_body=_trusted_sweep_jobset_body(),
        )

    lookup.assert_awaited_once_with("ns", "sweep")
    job_lookup.assert_not_awaited()
    failure_patch.assert_awaited_once()
    kwargs = failure_patch.await_args.kwargs
    assert kwargs["sweep_name"] == "sweep"
    assert kwargs["sweep_uid"] == "uid-sweep"
    assert "ReplicatedJobFailed" in kwargs["error"]
    assert "backoff limit" in kwargs["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner_field", "owner_value"),
    [
        param("uid", None, id="missing_uid"),
        param("name", "other-sweep", id="wrong_name"),
        param("controller", False, id="not_controller"),
        param("apiVersion", "example.com/v1", id="wrong_api_version"),
    ],
)  # fmt: skip
async def test_failed_sweep_jobset_requires_exact_controller_owner(
    owner_field: str, owner_value: object
) -> None:
    """Forged or incomplete owner references cannot reach the parent lookup."""
    jobset = _trusted_sweep_jobset_body()
    jobset["metadata"]["ownerReferences"][0][owner_field] = owner_value
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfsweep_body",
        new=AsyncMock(),
    ) as lookup:
        await handle_jobset_conditions(
            old=[],
            new=[{"type": "Failed", "status": "True"}],
            namespace="ns",
            jobset_name="aiperf-sweep",
            jobset_body=jobset,
        )
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_failure_patch_is_uid_and_resource_version_fenced() -> None:
    """The emergency writer mirrors aggregation_failed through JSON Patch."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    with (
        patch("aiperf.kubernetes.client.k8s_client", return_value=_ApiContext()),
        patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
    ):
        await _patch_sweep_controller_failure(
            namespace="ns",
            sweep_name="sweep",
            sweep_uid="uid-sweep",
            parent_body=_aiperfsweep_body(),
            error="Sweep controller JobSet aiperf-sweep failed",
        )

    kwargs = custom.patch_namespaced_custom_object_status.await_args.kwargs
    assert kwargs["_content_type"] == "application/json-patch+json"
    assert kwargs["field_manager"] == "aiperf-operator-jobset-terminal"
    operations = kwargs["body"]
    assert operations[:3] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-sweep"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "42"},
        {"op": "test", "path": "/status/phase", "value": "Running"},
    ]
    values = {operation["path"]: operation["value"] for operation in operations[3:]}
    assert values["/status/phase"] == "Failed"
    assert values["/status/resultsAvailable"] is False
    assert values["/status/completionTime"]
    assert values["/status/completedAt"] == values["/status/completionTime"]
    aggregation = values["/status/aggregation"]
    assert aggregation["phase"] == "Failed"
    assert aggregation["error"] == "Sweep controller JobSet aiperf-sweep failed"
    assert aggregation["completedAt"] == values["/status/completionTime"]


@pytest.mark.asyncio
async def test_sweep_failure_patch_handles_parent_before_initial_status_lands() -> None:
    """A very early controller crash can race the create handler's status patch."""
    parent = _aiperfsweep_body()
    parent.pop("status")
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock()
    with (
        patch("aiperf.kubernetes.client.k8s_client", return_value=_ApiContext()),
        patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
    ):
        await _patch_sweep_controller_failure(
            namespace="ns",
            sweep_name="sweep",
            sweep_uid="uid-sweep",
            parent_body=parent,
            error="controller failed before startup",
        )

    operations = custom.patch_namespaced_custom_object_status.await_args.kwargs["body"]
    assert operations[:2] == [
        {"op": "test", "path": "/metadata/uid", "value": "uid-sweep"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "42"},
    ]
    assert operations[2]["op"] == "add"
    assert operations[2]["path"] == "/status"
    assert operations[2]["value"]["phase"] == "Failed"
    assert operations[2]["value"]["aggregation"]["phase"] == "Failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        param("Succeeded", id="succeeded"),
        param("Failed", id="failed"),
        param("Cancelled", id="cancelled"),
        param("PartiallyFailed", id="partially_failed"),
    ],
)  # fmt: skip
async def test_sweep_failure_does_not_overwrite_terminal_parent(phase: str) -> None:
    """A delayed JobSet event cannot demote or rewrite any terminal parent."""
    with patch("aiperf.kubernetes.client.k8s_client") as client_factory:
        await _patch_sweep_controller_failure(
            namespace="ns",
            sweep_name="sweep",
            sweep_uid="uid-sweep",
            parent_body=_aiperfsweep_body(phase=phase),
            error="late failure",
        )
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_failure_rejects_recreated_same_name_parent() -> None:
    """The JobSet owner UID cannot patch a replacement sweep with the same name."""
    parent = _aiperfsweep_body()
    parent["metadata"]["uid"] = "replacement-uid"
    with patch("aiperf.kubernetes.client.k8s_client") as client_factory:
        await _patch_sweep_controller_failure(
            namespace="ns",
            sweep_name="sweep",
            sweep_uid="uid-sweep",
            parent_body=parent,
            error="stale controller failed",
        )
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_sweep_failure_patch_conflict_retries_instead_of_losing_event() -> None:
    """A concurrent status write triggers a retry so the parent cannot stay stuck."""
    custom = MagicMock()
    custom.patch_namespaced_custom_object_status = AsyncMock(
        side_effect=ApiException(status=422, reason="test operation failed")
    )
    with (
        patch("aiperf.kubernetes.client.k8s_client", return_value=_ApiContext()),
        patch("kubernetes_asyncio.client.CustomObjectsApi", return_value=custom),
        pytest.raises(kopf.TemporaryError, match="terminalize AIPerfSweep"),
    ):
        await _patch_sweep_controller_failure(
            namespace="ns",
            sweep_name="sweep",
            sweep_uid="uid-sweep",
            parent_body=_aiperfsweep_body(),
            error="controller failed",
        )


@pytest.mark.asyncio
async def test_already_failed_in_old_conditions_skips_parent_lookup() -> None:
    """A later JobSet status edit does not repeat the terminal failure write."""
    failed = [{"type": "Failed", "status": "True"}]
    with patch(
        "aiperf.operator.handlers.jobset_terminal._lookup_aiperfsweep_body",
        new=AsyncMock(),
    ) as lookup:
        await handle_jobset_conditions(
            old=failed,
            new=failed,
            namespace="ns",
            jobset_name="aiperf-sweep",
            jobset_body=_trusted_sweep_jobset_body(),
        )
    lookup.assert_not_awaited()


# =============================================================================
# Adversarial tests — production-hostile inputs
# =============================================================================


class TestJobsetTerminalAdversarial:
    """Adversarial coverage for ``handle_jobset_conditions``.

    These probe the production-hostile shapes kopf can deliver as the
    JobSet conditions list mutates: None / non-dict entries, missing or
    lowercased status fields, race-with-controller-pod annotations, and
    boundary cases on the ``aiperf-`` prefix-strip.
    """

    @pytest.mark.asyncio
    async def test_only_non_dict_entries_treated_as_no_terminal(self) -> None:
        """A list of all junk entries → no Completed → no-op."""
        new = [None, "junk", 7]
        with patch(
            "aiperf.operator.handlers.jobset_terminal._lookup_aiperfjob_body",
            new=AsyncMock(),
        ) as lookup:
            await handle_jobset_conditions(
                old=[], new=new, namespace="ns", jobset_name="aiperf-ajob"
            )
        lookup.assert_not_awaited()
