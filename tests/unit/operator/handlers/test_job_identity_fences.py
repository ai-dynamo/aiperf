# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial identity-fence tests for AIPerfJob lifecycle publication."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.phase import Phase
from aiperf.operator.client_cache import (
    _build_claim_patch_ops,
    _reset_for_testing,
    is_cancellation_requested,
    job_key,
)
from aiperf.operator.handlers import _completion_fetch as completion_fetch
from aiperf.operator.handlers import completion, lifecycle
from aiperf.operator.handlers._job_identity import (
    StaleAIPerfJobCallback,
    aiperfjob_jobset_uid,
    delete_owned_aiperfjob_jobset,
)
from aiperf.operator.status import StatusBuilder


def _body(uid: str = "job-uid-old") -> dict[str, object]:
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": "bench",
            "namespace": "ns",
            "uid": uid,
            "resourceVersion": "11",
            "generation": 3,
            "creationTimestamp": "2026-01-01T00:00:00Z",
        },
    }


def _status() -> dict[str, object]:
    return {
        "phase": Phase.RUNNING,
        "jobId": "bench",
        "jobSetName": "aiperf-bench",
    }


def _jobset(*, owner_uid: str = "job-uid-old") -> dict[str, object]:
    return {
        "metadata": {
            "name": "aiperf-bench",
            "uid": "jobset-uid-old",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfJob",
                    "name": "bench",
                    "uid": owner_uid,
                    "controller": True,
                }
            ],
        }
    }


@pytest.fixture(autouse=True)
def _reset_cache() -> AsyncIterator[None]:
    _reset_for_testing()
    yield
    _reset_for_testing()


def test_job_key_uid_scopes_recreated_job_state() -> None:
    """Same-name CR incarnations must never share process-local state."""
    assert job_key("ns", "bench", "old") == "ns/bench@old"
    assert job_key("ns", "bench", "new") == "ns/bench@new"
    assert job_key("ns", "bench") == "ns/bench"


def test_completion_claim_json_patch_tests_parent_uid() -> None:
    """The durable claim cannot land on a same-name replacement CR."""
    operations = _build_claim_patch_ops(_body(), "2026-01-01T00:00:01Z")
    assert operations[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "job-uid-old",
    }


@pytest.mark.asyncio
async def test_completion_fetch_uses_uid_scoped_progress_client_key() -> None:
    """Result retries cannot attach to a replacement job's cached client."""
    result = ControllerFetchResult(metrics=None, downloaded=[])
    with (
        mock_patch.object(
            completion_fetch,
            "get_or_create_progress_client",
            new=AsyncMock(return_value=MagicMock()),
        ) as get_client,
        mock_patch.object(
            completion_fetch, "get_cancellation_event", return_value=MagicMock()
        ) as get_event,
        mock_patch.object(
            completion_fetch,
            "_run_fetch_loop_safely",
            new=AsyncMock(return_value=result),
        ),
    ):
        fetched = await completion_fetch.fetch_results_with_retry(
            "controller.ns.svc",
            "ns",
            "bench",
            dest_dir=Path("/tmp/aiperf-test-results"),
            body=_body(),
        )

    assert fetched is result
    get_client.assert_awaited_once_with("ns/bench@job-uid-old")
    get_event.assert_called_once_with("ns/bench@job-uid-old")


def test_aiperfjob_jobset_uid_rejects_same_name_foreign_owner() -> None:
    """A deterministic JobSet name is not ownership proof."""
    with pytest.raises(StaleAIPerfJobCallback, match="not controlled"):
        aiperfjob_jobset_uid(
            _jobset(owner_uid="job-uid-new"),
            jobset_name="aiperf-bench",
            parent_name="bench",
            parent_uid="job-uid-old",
        )


@pytest.mark.asyncio
async def test_delete_owned_jobset_uses_uid_precondition() -> None:
    """The delete request is fenced to the exact validated JobSet object."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_jobset())
    custom.delete_namespaced_custom_object = AsyncMock(return_value={})

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[MagicMock]:
        yield MagicMock()

    with (
        mock_patch(
            "aiperf.operator.handlers._job_identity.k8s_client",
            side_effect=fake_client,
        ),
        mock_patch(
            "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
            return_value=custom,
        ),
    ):
        deleted = await delete_owned_aiperfjob_jobset(
            "ns",
            "aiperf-bench",
            parent_name="bench",
            parent_uid="job-uid-old",
            context="test",
        )

    assert deleted is True
    options = custom.delete_namespaced_custom_object.call_args.kwargs["body"]
    assert options.preconditions.uid == "jobset-uid-old"


@pytest.mark.asyncio
async def test_delete_owned_jobset_uid_conflict_rejects_publication() -> None:
    """A replacement between JobSet read and delete is a stale no-op."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=_jobset())
    custom.delete_namespaced_custom_object = AsyncMock(
        side_effect=ApiException(status=409, reason="UID precondition failed")
    )

    @asynccontextmanager
    async def fake_client() -> AsyncIterator[MagicMock]:
        yield MagicMock()

    with (
        mock_patch(
            "aiperf.operator.handlers._job_identity.k8s_client",
            side_effect=fake_client,
        ),
        mock_patch(
            "aiperf.operator.handlers._job_identity.client.CustomObjectsApi",
            return_value=custom,
        ),
    ):
        deleted = await delete_owned_aiperfjob_jobset(
            "ns",
            "aiperf-bench",
            parent_name="bench",
            parent_uid="job-uid-old",
            context="test",
        )

    assert deleted is False


@pytest.mark.asyncio
async def test_on_cancel_stale_parent_has_no_side_effects() -> None:
    """An old cancel callback cannot poison or patch the replacement job."""
    patch = kopf.Patch()
    with (
        mock_patch(
            "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
            new=AsyncMock(side_effect=StaleAIPerfJobCallback("same-name replacement")),
        ),
        mock_patch(
            "aiperf.operator.handlers.lifecycle.delete_owned_aiperfjob_jobset",
            new_callable=AsyncMock,
        ) as delete_jobset,
        mock_patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            new_callable=AsyncMock,
        ) as close_client,
    ):
        await lifecycle.on_cancel(
            body=_body(),
            spec={"cancel": True},
            status=_status(),
            name="bench",
            namespace="ns",
            patch=patch,
            expected_parent_uid="job-uid-old",
        )

    assert dict(patch) == {}
    assert is_cancellation_requested("ns/bench@job-uid-old") is False
    assert is_cancellation_requested("ns/bench@job-uid-new") is False
    delete_jobset.assert_not_awaited()
    close_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_cancel_fences_cache_delete_and_status_to_exact_identity() -> None:
    """A valid cancel uses UID-scoped cache state and a resourceVersion patch fence."""
    patch = kopf.Patch()
    with (
        mock_patch(
            "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
            new=AsyncMock(side_effect=["11", "12"]),
        ),
        mock_patch(
            "aiperf.operator.handlers.lifecycle.delete_owned_aiperfjob_jobset",
            new=AsyncMock(return_value=True),
        ) as delete_jobset,
        mock_patch(
            "aiperf.operator.handlers.lifecycle.close_progress_client",
            new_callable=AsyncMock,
        ) as close_client,
        mock_patch("aiperf.operator.handlers.lifecycle.events.cancelled"),
    ):
        await lifecycle.on_cancel(
            body=_body(),
            spec={"cancel": True},
            status=_status(),
            name="bench",
            namespace="ns",
            patch=patch,
            expected_parent_uid="job-uid-old",
        )

    assert patch["metadata"]["resourceVersion"] == "12"
    assert patch.status["phase"] == Phase.CANCELLED
    assert is_cancellation_requested("ns/bench@job-uid-old") is True
    assert is_cancellation_requested("ns/bench@job-uid-new") is False
    delete_jobset.assert_awaited_once_with(
        "ns",
        "aiperf-bench",
        parent_name="bench",
        parent_uid="job-uid-old",
        context="cancel",
    )
    close_client.assert_awaited_once_with("ns/bench@job-uid-old")


@pytest.mark.asyncio
async def test_completion_fences_delete_and_status_to_exact_parent() -> None:
    """Completion carries the parent UID into deletion and status publication."""
    patch = kopf.Patch()
    sb = StatusBuilder(patch, _status())
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
    )

    async def apply_results(
        *, sb: StatusBuilder, flags: completion._ResultFlags, **_: object
    ) -> tuple[completion._ResultFlags, tuple[completion._KeyArtifactFingerprint, ...]]:
        sb.set_phase(Phase.COMPLETED)
        return flags, ()

    with (
        mock_patch.object(
            completion,
            "current_aiperfjob_resource_version",
            new=AsyncMock(return_value="11"),
        ),
        mock_patch.object(
            completion, "_refresh_final_phase_progress", new_callable=AsyncMock
        ),
        mock_patch.object(completion, "_key_files_materialized", return_value=True),
        mock_patch.object(
            completion,
            "_maybe_delete_jobset_after_success",
            new=AsyncMock(return_value=True),
        ) as delete_jobset,
        mock_patch.object(
            completion,
            "_apply_completion_results",
            new=AsyncMock(side_effect=apply_results),
        ),
        mock_patch.object(
            completion,
            "_verify_final_artifact_publication",
            new=AsyncMock(side_effect=lambda *, flags, **_: (flags, True)),
        ),
        mock_patch.object(completion.events, "completed"),
    ):
        await completion.handle_completion(
            _body(),
            "ns",
            "aiperf-bench",
            "bench",
            status=_status(),
            sb=sb,
            result=result,
            expected_parent_uid="job-uid-old",
        )

    assert "metadata" not in patch  # resourceVersion fence removed from completion path
    assert patch.status["phase"] == Phase.COMPLETED
    assert delete_jobset.call_args.kwargs["parent_name"] == "bench"
    assert delete_jobset.call_args.kwargs["parent_uid"] == "job-uid-old"


@pytest.mark.asyncio
async def test_completion_jobset_uid_conflict_stops_publication() -> None:
    """A delete UID conflict stops status and index publication."""
    patch = kopf.Patch()
    sb = StatusBuilder(patch, _status())
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
    )
    with (
        mock_patch.object(
            completion,
            "current_aiperfjob_resource_version",
            new=AsyncMock(return_value="11"),
        ),
        mock_patch.object(
            completion, "_refresh_final_phase_progress", new_callable=AsyncMock
        ),
        mock_patch.object(completion, "_key_files_materialized", return_value=True),
        mock_patch.object(
            completion,
            "_maybe_delete_jobset_after_success",
            new=AsyncMock(return_value=False),
        ),
        mock_patch.object(
            completion, "_apply_completion_results", new_callable=AsyncMock
        ) as apply_results,
    ):
        await completion.handle_completion(
            _body(),
            "ns",
            "aiperf-bench",
            "bench",
            status=_status(),
            sb=sb,
            result=result,
            expected_parent_uid="job-uid-old",
        )

    assert dict(patch) == {}
    apply_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_completion_replacement_after_apply_discards_status_and_index() -> None:
    """A replacement observed at the final fence compensates index and drops status."""
    patch = kopf.Patch()
    sb = StatusBuilder(patch, _status())
    result = ControllerFetchResult(
        metrics={"metrics": {"request_count": {"avg": 1}}},
        downloaded=["profile_export_aiperf.json"],
    )

    async def apply_results(
        *, sb: StatusBuilder, flags: completion._ResultFlags, **_: object
    ) -> tuple[completion._ResultFlags, tuple[completion._KeyArtifactFingerprint, ...]]:
        sb.set_phase(Phase.COMPLETED)
        return flags, ()

    with (
        mock_patch.object(
            completion,
            "current_aiperfjob_resource_version",
            new=AsyncMock(
                side_effect=[
                    "11",
                    "12",
                    StaleAIPerfJobCallback("same-name replacement"),
                ]
            ),
        ),
        mock_patch.object(
            completion, "_refresh_final_phase_progress", new_callable=AsyncMock
        ),
        mock_patch.object(completion, "_key_files_materialized", return_value=True),
        mock_patch.object(
            completion,
            "_maybe_delete_jobset_after_success",
            new=AsyncMock(return_value=True),
        ),
        mock_patch.object(
            completion,
            "_apply_completion_results",
            new=AsyncMock(side_effect=apply_results),
        ),
        mock_patch.object(
            completion,
            "_verify_final_artifact_publication",
            new=AsyncMock(side_effect=lambda *, flags, **_: (flags, True)),
        ),
        mock_patch.object(
            completion, "_drop_index_row", new_callable=AsyncMock
        ) as drop_index,
        mock_patch.object(completion.events, "completed") as completed_event,
        mock_patch.object(completion.events, "results_stored") as stored_event,
        mock_patch.object(completion.events, "failed") as failed_event,
        mock_patch.object(completion.events, "results_failed") as results_failed_event,
        mock_patch.object(completion.events, "index_update_failed") as index_event,
    ):
        await completion.handle_completion(
            _body(),
            "ns",
            "aiperf-bench",
            "bench",
            status=_status(),
            sb=sb,
            result=result,
            expected_parent_uid="job-uid-old",
        )

    assert dict(patch) == {}
    drop_index.assert_awaited_once()
    completed_event.assert_not_called()
    stored_event.assert_not_called()
    failed_event.assert_not_called()
    results_failed_event.assert_not_called()
    index_event.assert_not_called()


@pytest.mark.asyncio
async def test_update_index_does_not_override_rejected_latest_epoch() -> None:
    """SQLite latest follows latest.txt when a delayed older run is rejected."""
    patch = kopf.Patch()
    sb = StatusBuilder(patch)
    with (
        mock_patch.object(
            completion.runs_index,
            "upsert_run_completed",
            new_callable=AsyncMock,
        ) as upsert,
        mock_patch.object(
            completion.runs_index, "set_latest", new_callable=AsyncMock
        ) as set_latest,
        mock_patch.object(completion, "_key_files_materialized", return_value=True),
        mock_patch.object(completion, "resolve_latest", return_value="2000000000"),
    ):
        await completion._update_job_index_safe(
            namespace="ns",
            job_id="bench",
            epoch="1000000000",
            body=_body(),
            sb=sb,
            phase="Succeeded",
            summary_blob=b"summary",
            metrics={},
            downloaded_files=["profile_export_aiperf.json"],
            error=None,
            mtime_epoch=0,
            end_time=None,
            total_size_bytes=1,
        )

    upsert.assert_awaited_once()
    set_latest.assert_not_awaited()


@pytest.mark.asyncio
async def test_drop_stale_index_row_restores_accepted_latest() -> None:
    """Compensation restores the row accepted by latest.txt after deleting stale latest."""
    accepted_row = MagicMock()
    with (
        mock_patch.object(
            completion.runs_index, "delete_run", new_callable=AsyncMock
        ) as delete_run,
        mock_patch.object(
            completion.runs_index,
            "get_run",
            new=AsyncMock(return_value=accepted_row),
        ) as get_run,
        mock_patch.object(
            completion.runs_index, "set_latest", new_callable=AsyncMock
        ) as set_latest,
        mock_patch.object(completion, "resolve_latest", return_value="2000000000"),
    ):
        await completion._drop_index_row("ns", "bench", "1000000000")

    delete_run.assert_awaited_once_with("ns", "bench", "1000000000")
    get_run.assert_awaited_once_with("ns", "bench", "2000000000")
    set_latest.assert_awaited_once_with("ns", "bench", "2000000000")
