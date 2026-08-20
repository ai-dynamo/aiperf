# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for completion-to-runs-index integration.

Focuses on:
- non-finite controller metrics scrubbed before they become index compare columns
- missing and stale summary blobs degrading the index without clobbering CR status
- index upsert failures surfacing as conditions while results remain available
- sweep child metadata already present in the run row surviving completion upserts
- exactly-once completion claim boundaries preventing duplicate index writes

Out of scope: JSON-patch claim atomicity itself, covered by
``tests/unit/operator/test_completion_claim_adversarial.py``; standalone
runs_index bootstrap behavior, covered by
``tests/unit/operator/test_runs_index_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import orjson
import pytest
import zstandard
from aiperf.operator.results_layout import epoch_key_from_body, run_dir

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.crd_models import ControllerFetchResult
from aiperf.kubernetes.phase import Phase
from aiperf.operator import runs_index
from aiperf.operator.client_cache import _reset_for_testing
from aiperf.operator.handlers import completion, lifecycle
from aiperf.operator.status import ConditionType, StatusBuilder

# =============================================================================
# Helpers
# =============================================================================

_FIXTURE_NAMESPACE = "aiperf-prod"
_FIXTURE_JOB_ID = "llama-throughput-7f2a"
_FIXTURE_JOBSET = "llama-throughput-7f2a-js"
_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_UID = "3d9a77c1-5b42-4e0a-8f6d-0c1e2a3b4c5d"
# Whole-second creationTimestamps get a deterministic uid-derived suffix so two
# same-name resubmits inside one apiserver second cannot share a run directory.
_FIXTURE_EPOCH = epoch_key_from_body(
    {"metadata": {"creationTimestamp": _FIXTURE_CREATION_TS, "uid": _FIXTURE_UID}}
)


@pytest.fixture(autouse=True)
def _reset_completion_state() -> Iterator[None]:
    """Clear process-local completion claim/cancellation state around each test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


@pytest.fixture
async def opened_index(tmp_path: Path) -> AsyncGenerator[Path, None]:
    """Open a fresh writable runs_index database for completion integration tests."""
    index_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(index_path)
    try:
        yield index_path
    finally:
        await runs_index.close()


def _body_with_claim() -> dict[str, Any]:
    """Build a realistic AIPerfJob body whose creation timestamp fixes the epoch."""
    return {
        "metadata": {
            "name": _FIXTURE_JOB_ID,
            "namespace": _FIXTURE_NAMESPACE,
            "uid": _FIXTURE_UID,
            "creationTimestamp": _FIXTURE_CREATION_TS,
            "generation": 7,
            "annotations": {Annotations.COMPLETION_CLAIMED: "2024-04-25T17:02:04Z"},
        },
        "spec": {
            "benchmark": {
                "models": {"items": [{"name": "meta-llama/Llama-3-8B"}]},
                "endpoint": {"urls": ["http://vllm.prod.svc:8000"]},
            }
        },
    }


def _patch_obj() -> MagicMock:
    """Build a kopf-like patch object with a mutable status mapping."""
    patch = MagicMock()
    patch.status = {}
    return patch


def _status_builder(patch: MagicMock) -> StatusBuilder:
    """Build a StatusBuilder with existing worker state for completion paths."""
    return StatusBuilder(
        patch,
        existing_status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
    )


def _metrics_payload(
    *,
    throughput_avg: float = 4772.5,
    latency_p99: float = 900.2,
) -> dict[str, Any]:
    """Return the profile export shape consumed by status and index writers."""
    return {
        "metrics": {
            "request_throughput": {
                "avg": throughput_avg,
                "p50": 4500.0,
                "p99": 5100.0,
                "unit": "req/s",
            },
            "request_latency": {
                "avg": 96.5,
                "p50": 71.2,
                "p99": latency_p99,
                "unit": "ms",
            },
            "time_to_first_token": {"avg": 71.1, "p99": 240.0, "unit": "ms"},
            "request_count": {"avg": 8192.0, "unit": "requests"},
            "error_request_count": {"avg": 0.0, "unit": "requests"},
        },
        "input_config": {
            "models": {"items": [{"name": "meta-llama/Llama-3-8B"}]},
            "endpoint": {"urls": ["http://vllm.prod.svc:8000"]},
        },
        "start_time": "2026-05-17T00:00:00Z",
        "end_time": "2026-05-17T00:03:21Z",
    }


def _write_result_file(
    base_dir: Path,
    relative_name: str,
    payload: dict[str, Any] | bytes,
    *,
    compress: bool = False,
) -> Path:
    """Write one artifact under the epoch-keyed operator results directory."""
    dest_dir = run_dir(base_dir, _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH)
    path = dest_dir / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = orjson.dumps(payload) if isinstance(payload, dict) else payload
    path.write_bytes(zstandard.ZstdCompressor().compress(raw) if compress else raw)
    return path


@contextmanager
def _patched_completion_environment(tmp_path: Path) -> Iterator[SimpleNamespace]:
    """Patch cluster side effects while leaving the real runs_index module wired."""
    captured = SimpleNamespace(
        completed=MagicMock(),
        index_update_failed=MagicMock(),
        results_failed=MagicMock(),
        results_stored=MagicMock(),
        delete_jobset=AsyncMock(),
        fetch_results=AsyncMock(),
        # Completion is fenced on live CR identity: the handler re-reads the
        # AIPerfJob and abandons the callback if the UID no longer resolves.
        # These tests exercise index integration, not apiserver liveness.
        resource_version=AsyncMock(return_value="4242"),
    )
    with (
        mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
            RETAIN_RUNS=5,
            RETAIN_DAYS=0,
            TRANSIENT_FETCH_RETRY_BUDGET_SEC=0.0,
            TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.events.completed",
            new=captured.completed,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.events.index_update_failed",
            new=captured.index_update_failed,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.events.results_failed",
            new=captured.results_failed,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.events.results_stored",
            new=captured.results_stored,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion._delete_backing_jobset",
            new=captured.delete_jobset,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.fetch_results_with_retry",
            new=captured.fetch_results,
        ),
        mock_patch(
            "aiperf.operator.handlers.completion.current_aiperfjob_resource_version",
            new=captured.resource_version,
        ),
    ):
        yield captured


def _conditions_by_type(patch: MagicMock) -> dict[str, dict[str, Any]]:
    """Return finalized status conditions keyed by Kubernetes condition type."""
    return {
        condition["type"]: condition for condition in patch.status.get("conditions", [])
    }


# =============================================================================
# Metrics scrubbed before index upsert
# =============================================================================


class TestCompletionIndexMetricScrubbing:
    """Non-finite controller metrics must not leak into SQLite compare columns."""

    @pytest.mark.asyncio
    async def test_handle_completion_non_finite_controller_metrics_indexes_null_compare_columns(
        self,
        tmp_path: Path,
        opened_index: Path,
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={
                "metrics": {
                    "request_throughput": {
                        "avg": float("nan"),
                        "p50": 4500.0,
                        "p99": float("inf"),
                        "unit": "req/s",
                    },
                    "request_latency": {
                        "avg": 96.5,
                        "p99": float("-inf"),
                        "unit": "ms",
                    },
                }
            },
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        narrow = await runs_index.get_run_narrow_metrics(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )
        assert narrow is not None
        assert narrow["request_throughput_avg"] is None
        assert narrow["request_throughput_p50"] == 4500.0
        assert narrow["request_throughput_p99"] is None
        assert narrow["request_latency_avg"] == 96.5
        assert narrow["request_latency_p99"] is None
        assert patch.status["phase"] == Phase.COMPLETED


# =============================================================================
# Missing and stale summary blobs
# =============================================================================


class TestCompletionIndexSummaryBlobBoundaries:
    """Index rows should degrade when key exports exist but summary blobs are unusable."""

    @pytest.mark.asyncio
    async def test_handle_completion_csv_only_missing_summary_marks_status_complete_but_index_unusable(
        self,
        tmp_path: Path,
        opened_index: Path,
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        row = await runs_index.get_run(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )
        assert row is not None
        # A csv-authoritative run that succeeded must be recorded as Succeeded
        # with no error, mirroring the CR's Succeeded/ResultsAvailable verdict
        # and the disk-fallback path (results_db._index_from_disk). The summary
        # blob stays unusable (no readable JSON), but that is not a failure.
        assert row.phase == "Succeeded"
        assert row.error is None
        assert (
            await runs_index.get_summary_blob(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
            )
            is None
        )
        assert patch.status["phase"] == Phase.COMPLETED
        assert patch.status["runEpoch"] == int(_FIXTURE_EPOCH)
        conditions = _conditions_by_type(patch)
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "True"

    @pytest.mark.asyncio
    async def test_handle_completion_stale_ready_marker_without_summary_does_not_create_metrics_blob(
        self,
        tmp_path: Path,
        opened_index: Path,
    ) -> None:
        _write_result_file(tmp_path, runs_index.READY_MARKER, b'{"ready": true}')
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        row = await runs_index.get_run(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )
        assert row is not None
        # Success verdict, no readable summary blob: completed row, no error.
        assert row.phase == "Succeeded"
        assert row.error is None
        assert (
            await runs_index.get_summary_blob(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
            )
            is None
        )


# =============================================================================
# Index failure isolation
# =============================================================================


class TestCompletionIndexFailureIsolation:
    """Index write failures must not corrupt already-staged completion status."""

    @pytest.mark.asyncio
    async def test_handle_completion_index_upsert_failure_keeps_results_available_condition(
        self,
        tmp_path: Path,
    ) -> None:
        runs_index.mark_catalog_complete(tmp_path)
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": _metrics_payload()["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )

        with (
            _patched_completion_environment(tmp_path) as captured,
            mock_patch(
                "aiperf.operator.runs_index.upsert_run_completed",
                new=AsyncMock(
                    side_effect=RuntimeError("sqlite disk full for aiperf index")
                ),
            ),
        ):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        conditions = _conditions_by_type(patch)
        assert patch.status["phase"] == Phase.COMPLETED
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "True"
        assert conditions[ConditionType.COMPLETE.value]["status"] == "True"
        assert conditions[ConditionType.INDEX_UPDATED.value]["status"] == "False"
        assert (
            "sqlite disk full"
            in conditions[ConditionType.INDEX_UPDATED.value]["message"]
        )
        assert runs_index.catalog_is_complete(tmp_path) is False
        captured.index_update_failed.assert_called_once()
        captured.completed.assert_called_once()
        captured.delete_jobset.assert_awaited_once_with(
            _FIXTURE_NAMESPACE,
            _FIXTURE_JOBSET,
            parent_name=_FIXTURE_JOB_ID,
            parent_uid=_FIXTURE_UID,
        )

    @pytest.mark.asyncio
    async def test_handle_completion_artifacts_removed_during_index_update_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A vanished final export cannot leave a completed status or index row."""
        run = tmp_path / _FIXTURE_NAMESPACE / _FIXTURE_JOB_ID / _FIXTURE_EPOCH
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": _metrics_payload()["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )

        async def index_then_remove_key_export(*args: Any, **kwargs: Any) -> bool:
            (run / "profile_export_aiperf.json").unlink()
            (run / "profile_export_aiperf.csv").unlink()
            return True

        update_index = AsyncMock(side_effect=index_then_remove_key_export)
        delete_run = AsyncMock()
        monkeypatch.setattr(completion, "_update_job_index_safe", update_index)
        monkeypatch.setattr(completion.runs_index, "delete_run", delete_run)
        assert completion._update_job_index_safe is update_index

        with (
            _patched_completion_environment(tmp_path),
            mock_patch(
                "aiperf.operator.handlers.completion._run_retention_pass",
                new=AsyncMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._load_phase_manifest_payload",
                return_value=None,
            ),
        ):
            flags, _ = await completion._apply_completion_results(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
                flags=completion._compute_result_flags(result, _FIXTURE_JOB_ID),
                parent_name=_FIXTURE_JOB_ID,
                parent_uid=_FIXTURE_UID,
            )

        sb.finalize()

        conditions = _conditions_by_type(patch)
        update_index.assert_awaited_once()
        assert flags.success is False
        assert patch.status["phase"] == Phase.FAILED
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "False"
        assert "resultsPath" not in patch.status
        assert "runEpoch" not in patch.status
        delete_run.assert_awaited_once_with(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )

    @pytest.mark.asyncio
    async def test_handle_completion_missing_initial_artifacts_never_indexes_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A post-delete disappearance cannot publish a zero-file success row."""
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        run = tmp_path / _FIXTURE_NAMESPACE / _FIXTURE_JOB_ID / _FIXTURE_EPOCH
        (run / "profile_export_aiperf.json").unlink()
        (run / "profile_export_aiperf.csv").unlink()
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": _metrics_payload()["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )
        update_index = AsyncMock(return_value=True)
        monkeypatch.setattr(completion, "_update_job_index_safe", update_index)

        with (
            _patched_completion_environment(tmp_path),
            mock_patch(
                "aiperf.operator.handlers.completion._load_phase_manifest_payload",
                return_value=None,
            ),
        ):
            flags, _ = await completion._apply_completion_results(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={},
                sb=sb,
                result=result,
                flags=completion._compute_result_flags(result, _FIXTURE_JOB_ID),
                parent_name=_FIXTURE_JOB_ID,
                parent_uid=_FIXTURE_UID,
            )

        sb.finalize()
        assert flags.success is False
        assert update_index.await_args.kwargs["phase"] == "Failed"
        assert patch.status["phase"] == Phase.FAILED
        assert "resultsPath" not in patch.status

    @pytest.mark.asyncio
    async def test_final_publication_fence_revalidates_artifacts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Artifacts lost during the final parent fence cannot publish success."""
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        run = tmp_path / _FIXTURE_NAMESPACE / _FIXTURE_JOB_ID / _FIXTURE_EPOCH
        result = ControllerFetchResult(
            metrics={"metrics": _metrics_payload()["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )
        staged_patch = _patch_obj()
        target_patch = _patch_obj()
        staged_sb = _status_builder(staged_patch)
        target_sb = _status_builder(target_patch)
        fence_calls = 0

        async def fence_then_remove(*_args: Any, **_kwargs: Any) -> str:
            nonlocal fence_calls
            fence_calls += 1
            if fence_calls == 4:
                (run / "profile_export_aiperf.json").unlink()
                (run / "profile_export_aiperf.csv").unlink()
            return "4242"

        monkeypatch.setattr(
            completion,
            "_update_job_index_safe",
            AsyncMock(return_value=True),
        )
        monkeypatch.setattr(completion.runs_index, "delete_run", AsyncMock())

        with (
            _patched_completion_environment(tmp_path),
            mock_patch(
                "aiperf.operator.handlers.completion.current_aiperfjob_resource_version",
                side_effect=fence_then_remove,
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._run_retention_pass",
                new=AsyncMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._load_phase_manifest_payload",
                return_value=None,
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.events.completed"
            ) as completed,
            mock_patch(
                "aiperf.operator.handlers.completion.events.results_stored"
            ) as stored,
        ):
            await completion._publish_completion_after_jobset_delete(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                result=result,
                staged_sb=staged_sb,
                target_sb=target_sb,
                status={},
                flags=completion._compute_result_flags(result, _FIXTURE_JOB_ID),
                key_names=completion.DEFAULT_KEY_EXPORT_NAMES,
                parent_name=_FIXTURE_JOB_ID,
                parent_uid=_FIXTURE_UID,
                duration_sec=None,
            )

        conditions = _conditions_by_type(target_patch)
        assert target_patch.status["phase"] == Phase.FAILED
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "False"
        assert "resultsPath" not in target_patch.status
        completed.assert_not_called()
        stored.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_publication_delete_recreate_fingerprint_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transient empty index scan cannot be healed by recreating exports."""
        json_payload = _metrics_payload()
        csv_payload = b"metric,value\n"
        _write_result_file(tmp_path, "profile_export_aiperf.json", json_payload)
        _write_result_file(tmp_path, "profile_export_aiperf.csv", csv_payload)
        run = tmp_path / _FIXTURE_NAMESPACE / _FIXTURE_JOB_ID / _FIXTURE_EPOCH
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": json_payload["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )
        original_gather = completion._gather_index_inputs

        def gather_during_transient_gap(
            *args: Any, **kwargs: Any
        ) -> tuple[bytes | None, int, str | None, int]:
            (run / "profile_export_aiperf.json").unlink()
            (run / "profile_export_aiperf.csv").unlink()
            gathered = original_gather(*args, **kwargs)
            _write_result_file(tmp_path, "profile_export_aiperf.json", json_payload)
            _write_result_file(tmp_path, "profile_export_aiperf.csv", csv_payload)
            return gathered

        update_index = AsyncMock(return_value=True)
        delete_run = AsyncMock()
        monkeypatch.setattr(
            completion, "_gather_index_inputs", gather_during_transient_gap
        )
        monkeypatch.setattr(completion, "_update_job_index_safe", update_index)
        monkeypatch.setattr(completion.runs_index, "delete_run", delete_run)

        with (
            _patched_completion_environment(tmp_path),
            mock_patch(
                "aiperf.operator.handlers.completion._run_retention_pass",
                new=AsyncMock(),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion._load_phase_manifest_payload",
                return_value=None,
            ),
        ):
            flags, _ = await completion._apply_completion_results(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={},
                sb=sb,
                result=result,
                flags=completion._compute_result_flags(result, _FIXTURE_JOB_ID),
                parent_name=_FIXTURE_JOB_ID,
                parent_uid=_FIXTURE_UID,
            )

        assert update_index.await_args.kwargs["summary_blob"] is None
        assert flags.success is False
        delete_run.assert_awaited_once_with(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )


# =============================================================================
# Sweep child metadata preservation
# =============================================================================


class TestCompletionIndexSweepMetadata:
    """Completion upserts must not erase sweep linkage already stored on the run row."""

    @pytest.mark.asyncio
    async def test_handle_completion_sweep_child_existing_linkage_survives_completed_upsert(
        self,
        tmp_path: Path,
        opened_index: Path,
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        await runs_index.upsert_run_created(
            _FIXTURE_NAMESPACE,
            _FIXTURE_JOB_ID,
            _FIXTURE_EPOCH,
            spec=_body_with_claim()["spec"],
        )
        await runs_index._conn().execute(
            "UPDATE runs SET sweep_namespace = ?, sweep_name = ?, sweep_epoch = ?, "
            "sweep_variation_idx = ? WHERE namespace = ? AND job_id = ? AND epoch = ?",
            (
                _FIXTURE_NAMESPACE,
                "token-sweep-8a4e",
                "1714064400",
                3,
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB_ID,
                _FIXTURE_EPOCH,
            ),
        )
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": _metrics_payload()["metrics"]},
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 8}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        row = await runs_index.get_run(
            _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
        )
        assert row is not None
        assert row.phase == "Succeeded"
        assert row.sweep_namespace == _FIXTURE_NAMESPACE
        assert row.sweep_name == "token-sweep-8a4e"
        assert row.sweep_epoch == "1714064400"
        assert row.sweep_variation_idx == 3


# =============================================================================
# Exactly-once claim boundaries
# =============================================================================


class TestCompletionClaimIndexBoundary:
    """The lifecycle handler must not update index state unless it wins the claim."""

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_lost_claim_skips_completion_index_write(
        self,
    ) -> None:
        patch = _patch_obj()
        status = {
            "phase": Phase.RUNNING,
            "jobId": _FIXTURE_JOB_ID,
            "jobSetName": _FIXTURE_JOBSET,
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                new=AsyncMock(return_value=False),
            ) as claim,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                new=AsyncMock(),
            ) as handle,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.runs_index.upsert_run_completed",
                new=AsyncMock(),
            ) as upsert_completed,
            mock_patch(
                "aiperf.operator.handlers.lifecycle.current_aiperfjob_resource_version",
                new=AsyncMock(return_value="4242"),
            ),
        ):
            await lifecycle.on_benchmark_complete(
                body=_body_with_claim(),
                status=status,
                name=_FIXTURE_JOB_ID,
                namespace=_FIXTURE_NAMESPACE,
                patch=patch,
            )

        claim.assert_awaited_once()
        handle.assert_not_awaited()
        upsert_completed.assert_not_awaited()
        assert patch.status == {}
