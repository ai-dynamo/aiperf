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

from aiperf.kubernetes.constants import Annotations
from aiperf.operator import runs_index
from aiperf.operator.client_cache import _reset_for_testing
from aiperf.operator.handlers import completion, lifecycle
from aiperf.operator.models import ControllerFetchResult
from aiperf.operator.results_layout import run_dir
from aiperf.operator.status import ConditionType, Phase, StatusBuilder

# =============================================================================
# Helpers
# =============================================================================

_FIXTURE_NAMESPACE = "aiperf-prod"
_FIXTURE_JOB_ID = "llama-throughput-7f2a"
_FIXTURE_JOBSET = "llama-throughput-7f2a-js"
_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_EPOCH = "1714064523"


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
        captured.index_update_failed.assert_called_once()
        captured.completed.assert_called_once()
        captured.delete_jobset.assert_awaited_once_with(
            _FIXTURE_NAMESPACE, _FIXTURE_JOBSET
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
