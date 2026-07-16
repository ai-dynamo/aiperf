# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes completion parsing and harvesting.

Focuses on:
- malformed metric-file candidates skipped without zeroing a valid summary
- missing and partial artifact trees surfaced as partial, not successful, harvests
- ResultsAvailable gating for kubectl-compatible Complete/Failed conditions
- completion cancellation/claim-adjacent short-circuits before side effects
- non-finite metric values at the CR status serialization boundary

Out of scope: JSON-patch claim atomicity, covered by
``tests/unit/operator/test_completion_claim_adversarial.py``; HTTP retry
backoff internals, covered by ``tests/unit/operator/test_completion_retry.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import kopf
import orjson
import pytest
import zstandard
from pytest import param

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.client_cache import (
    _reset_for_testing,
    job_key,
    request_cancellation,
)
from aiperf.operator.handlers import completion
from aiperf.operator.models import ControllerFetchResult
from aiperf.operator.results_layout import run_dir
from aiperf.operator.status import ConditionType, Phase, StatusBuilder

# =============================================================================
# Helpers
# =============================================================================

_FIXTURE_NAMESPACE = "aiperf-prod"
_FIXTURE_JOB_ID = "aiperf-bench-7f2a"
_FIXTURE_JOBSET = "aiperf-bench-7f2a-js"
_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_EPOCH = "1714064523"


def _body_with_claim() -> dict[str, Any]:
    """Build a realistic AIPerfJob body carrying a fresh completion claim."""
    return {
        "metadata": {
            "name": _FIXTURE_JOB_ID,
            "namespace": _FIXTURE_NAMESPACE,
            "creationTimestamp": _FIXTURE_CREATION_TS,
            "annotations": {Annotations.COMPLETION_CLAIMED: "2024-04-25T17:02:04Z"},
        }
    }


def _patch_obj() -> MagicMock:
    """Build a kopf-like patch object with mutable status."""
    patch = MagicMock()
    patch.status = {}
    return patch


def _status_builder(patch: MagicMock) -> StatusBuilder:
    """Build a StatusBuilder with enough existing status for completion paths."""
    return StatusBuilder(
        patch,
        existing_status={"workers": {"total": 4}, "startTime": "2026-05-17T00:00:00Z"},
    )


def _metrics_payload(*, throughput: float = 4772.5) -> dict[str, Any]:
    """Return the file-export metrics shape consumed by status.summary."""
    return {
        "metrics": {
            "request_throughput": {"avg": throughput, "unit": "req/s"},
            "request_latency": {"avg": 96.5, "p99": 900.2, "unit": "ms"},
            "time_to_first_token": {"avg": 71.1, "p99": 240.0, "unit": "ms"},
            "request_count": {"avg": 8192.0, "unit": "requests"},
            "error_request_count": {"avg": 0.0, "unit": "requests"},
        },
        "end_time": "2026-05-17T00:03:21Z",
    }


def _write_result_file(
    base_dir: Path,
    relative_name: str,
    payload: dict[str, Any] | bytes,
    *,
    compress: bool = False,
) -> Path:
    """Write a completion artifact under the epoch-keyed results directory."""
    dest_dir = run_dir(
        base_dir,
        _FIXTURE_NAMESPACE,
        _FIXTURE_JOB_ID,
        _FIXTURE_EPOCH,
    )
    path = dest_dir / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = orjson.dumps(payload) if isinstance(payload, dict) else payload
    path.write_bytes(zstandard.ZstdCompressor().compress(raw) if compress else raw)
    return path


@contextmanager
def _patched_completion_environment(
    tmp_path: Path,
) -> Iterator[SimpleNamespace]:
    """Patch external completion side effects and expose captured mocks."""
    captured = SimpleNamespace(
        completed=MagicMock(),
        results_failed=MagicMock(),
        results_stored=MagicMock(),
        delete_jobset=AsyncMock(),
        fetch_results=AsyncMock(),
        upsert_completed=AsyncMock(),
        upsert_failed=AsyncMock(),
        set_latest=AsyncMock(),
    )
    fake_index = MagicMock(
        upsert_run_completed=captured.upsert_completed,
        upsert_run_failed=captured.upsert_failed,
        set_latest=captured.set_latest,
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
        mock_patch("aiperf.operator.handlers.completion.runs_index", new=fake_index),
        mock_patch(
            "aiperf.operator.handlers.completion.events.completed",
            new=captured.completed,
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
    """Return finalized status conditions keyed by condition type."""
    return {
        condition["type"]: condition for condition in patch.status.get("conditions", [])
    }


def _non_finite_paths(value: Any, prefix: str = "status") -> list[str]:
    """Return paths to NaN/Inf floats that would cross the CR status boundary."""
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        out: list[str] = []
        for key, nested in value.items():
            out.extend(_non_finite_paths(nested, f"{prefix}.{key}"))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for index, nested in enumerate(value):
            out.extend(_non_finite_paths(nested, f"{prefix}[{index}]"))
        return out
    return []


@pytest.fixture(autouse=True)
def _reset_completion_state() -> Iterator[None]:
    """Clear process-local completion cancellation state around each test."""
    _reset_for_testing()
    yield
    _reset_for_testing()


# =============================================================================
# Malformed metric files and summary harvesting
# =============================================================================


class TestCompletionMetricFileCandidateResilience:
    """Malformed sibling artifacts must not hide a valid AIPerf JSON export."""

    @pytest.mark.parametrize(
        "bad_name,bad_payload",
        [
            param("profile_export.jsonl.zst", b'{"row": 1}\n{"row": 2}\n', id="jsonl-zst"),
            param("server_metrics_export.parquet.zst", b"PAR1\x00\x01", id="parquet-zst"),
            param("profile_export_aiperf.json.zst", b"not-zstd", id="corrupt-key-zst"),
        ],
    )  # fmt: skip
    def test_parse_metrics_from_files_malformed_candidate_keeps_valid_summary(
        self,
        tmp_path: Path,
        bad_name: str,
        bad_payload: bytes,
    ) -> None:
        bad_is_compressed = bad_name.endswith(".zst") and bad_payload != b"not-zstd"
        _write_result_file(
            tmp_path,
            bad_name,
            bad_payload,
            compress=bad_is_compressed,
        )
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())

        with mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ):
            result = completion._parse_metrics_from_files(
                [bad_name, "profile_export_aiperf.json"],
                _FIXTURE_NAMESPACE,
                _FIXTURE_JOB_ID,
                epoch=_FIXTURE_EPOCH,
            )

        assert result is not None
        assert result["metrics"]["request_throughput"]["avg"] == 4772.5

    def test_record_results_on_status_malformed_candidate_does_not_zero_summary(
        self, tmp_path: Path
    ) -> None:
        _write_result_file(
            tmp_path,
            "profile_export.jsonl.zst",
            b'{"trace": 1}\n{"trace": 2}\n',
            compress=True,
        )
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export.jsonl.zst", "profile_export_aiperf.json"],
        )

        with _patched_completion_environment(tmp_path):
            completion._record_results_on_status(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                job_id=_FIXTURE_JOB_ID,
                result=result,
                sb=sb,
                has_metrics=False,
                has_files=True,
            )

        assert patch.status["results"]["metrics"]["request_latency"]["p99"] == 900.2
        assert patch.status["summary"]["request_throughput"]["avg"] == 4772.5
        assert patch.status["summary"]["total_requests"] == 8192


# =============================================================================
# Key-artifact materialization: existence is not enough
# =============================================================================


class TestKeyFilesMaterializedValidation:
    """A truncated/unparsable key artifact must NOT count as materialized.

    Regression: a disk-full mid-write leaves a truncated
    ``profile_export_aiperf.json`` on disk. The old check passed on file
    EXISTENCE only, so the operator advanced ``latest.txt``/``runEpoch`` and
    served corrupt bytes as a Complete benchmark. Materialization now requires
    the JSON export to decode and parse to a non-empty dict.
    """

    def _materialized(self, tmp_path: Path) -> bool:
        with mock_patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            DIR=tmp_path,
        ):
            return completion._key_files_materialized(
                _FIXTURE_NAMESPACE, _FIXTURE_JOB_ID, _FIXTURE_EPOCH
            )

    def test_missing_run_dir_not_materialized(self, tmp_path: Path) -> None:
        assert self._materialized(tmp_path) is False

    def test_truncated_json_not_materialized(self, tmp_path: Path) -> None:
        _write_result_file(
            tmp_path, "profile_export_aiperf.json", b'{"metrics": {"request_throughp'
        )
        assert self._materialized(tmp_path) is False

    def test_empty_json_not_materialized(self, tmp_path: Path) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json", b"")
        assert self._materialized(tmp_path) is False

    def test_corrupt_json_zst_not_materialized(self, tmp_path: Path) -> None:
        # Non-empty bytes that are not a valid zstd frame → truncated .zst.
        _write_result_file(tmp_path, "profile_export_aiperf.json.zst", b"not-zstd")
        assert self._materialized(tmp_path) is False

    def test_valid_json_materialized(self, tmp_path: Path) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        assert self._materialized(tmp_path) is True

    def test_valid_json_zst_materialized(self, tmp_path: Path) -> None:
        _write_result_file(
            tmp_path,
            "profile_export_aiperf.json.zst",
            _metrics_payload(),
            compress=True,
        )
        assert self._materialized(tmp_path) is True

    def test_csv_only_still_materialized(self, tmp_path: Path) -> None:
        # A csv-authoritative run has no readable JSON summary but is still a
        # valid completion; a non-empty CSV counts as materialized.
        _write_result_file(
            tmp_path, "profile_export_aiperf.csv", b"metric,value\nrequest_count,8192\n"
        )
        assert self._materialized(tmp_path) is True

    def test_truncated_json_beside_valid_csv_is_materialized(
        self, tmp_path: Path
    ) -> None:
        # The truncated JSON does not count, but the valid CSV does → the run
        # is materialized on the csv-authoritative path.
        _write_result_file(
            tmp_path, "profile_export_aiperf.json", b'{"metrics": {"request_throughp'
        )
        _write_result_file(
            tmp_path, "profile_export_aiperf.csv", b"metric,value\nrequest_count,8192\n"
        )
        assert self._materialized(tmp_path) is True


# =============================================================================
# Missing files and partial artifact trees
# =============================================================================


class TestCompletionPartialArtifactTrees:
    """Partial trees should surface as partial harvests, not full success."""

    def test_record_results_on_status_csv_without_json_keeps_summary_absent(
        self, tmp_path: Path
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path) as captured:
            completion._record_results_on_status(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                job_id=_FIXTURE_JOB_ID,
                result=result,
                sb=sb,
                has_metrics=False,
                has_files=True,
            )

        assert "results" not in patch.status
        assert "summary" not in patch.status
        assert patch.status["runEpoch"] == int(_FIXTURE_EPOCH)
        assert patch.status["resultsPath"].endswith(
            f"/{_FIXTURE_NAMESPACE}/{_FIXTURE_JOB_ID}/{_FIXTURE_EPOCH}"
        )
        captured.results_stored.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_completion_partial_tree_without_key_exports_marks_failed(
        self, tmp_path: Path
    ) -> None:
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": {"request_throughput": {"avg": 101.0}}},
            downloaded=["inputs.json", "checkpoints/aggregator-0.parquet"],
            checkpoints=["checkpoints/aggregator-0.parquet"],
            error="controller results incomplete: missing profile_export_aiperf.json",
        )

        with _patched_completion_environment(tmp_path) as captured:
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 4}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        conditions = _conditions_by_type(patch)
        assert patch.status["phase"] == Phase.FAILED
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "False"
        assert (
            "missing profile_export_aiperf.json"
            in conditions[ConditionType.RESULTS_AVAILABLE.value]["message"]
        )
        captured.completed.assert_not_called()
        captured.delete_jobset.assert_not_awaited()

    def test_gather_index_inputs_missing_summary_file_keeps_artifact_size(
        self, tmp_path: Path
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")

        with _patched_completion_environment(tmp_path):
            summary_blob, mtime_epoch, end_time, total_size_bytes = (
                completion._gather_index_inputs(
                    _FIXTURE_NAMESPACE,
                    _FIXTURE_JOB_ID,
                    _FIXTURE_EPOCH,
                )
            )

        assert summary_blob is None
        assert end_time is None
        assert mtime_epoch > 0
        assert total_size_bytes == len(b"metric,value\n")

    def test_gather_index_inputs_malformed_summary_logs_path_and_counts_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json.zst", b"not-zstd")
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")

        with _patched_completion_environment(tmp_path), caplog.at_level("WARNING"):
            summary_blob, mtime_epoch, end_time, total_size_bytes = (
                completion._gather_index_inputs(
                    _FIXTURE_NAMESPACE,
                    _FIXTURE_JOB_ID,
                    _FIXTURE_EPOCH,
                )
            )

        assert summary_blob is None
        assert end_time is None
        assert mtime_epoch > 0
        assert total_size_bytes >= len(b"not-zstd") + len(b"metric,value\n")
        assert "cannot read summary" in caplog.text
        assert f"{_FIXTURE_NAMESPACE}/{_FIXTURE_JOB_ID}/{_FIXTURE_EPOCH}" in caplog.text


# =============================================================================
# ResultsAvailable and completion-claim interactions
# =============================================================================


class TestCompletionResultsAvailableGating:
    """Complete/Failed conditions derive from result availability, not phase alone."""

    @pytest.mark.asyncio
    async def test_handle_completion_files_only_success_latches_complete_condition(
        self, tmp_path: Path
    ) -> None:
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics=None,
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )

        with _patched_completion_environment(tmp_path) as captured:
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 4}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        conditions = _conditions_by_type(patch)
        assert patch.status["phase"] == Phase.COMPLETED
        assert conditions[ConditionType.RESULTS_AVAILABLE.value]["status"] == "True"
        assert conditions[ConditionType.COMPLETE.value]["status"] == "True"
        assert conditions[ConditionType.FAILED.value]["status"] == "False"
        assert patch.status["summary"]["request_latency"]["p99"] == 900.2
        captured.completed.assert_called_once()
        captured.delete_jobset.assert_awaited_once_with(
            _FIXTURE_NAMESPACE, _FIXTURE_JOBSET
        )

    @pytest.mark.asyncio
    async def test_handle_completion_fresh_transient_error_names_job_and_retries(
        self, tmp_path: Path
    ) -> None:
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={"metrics": {"request_throughput": {"avg": 1.0}}},
            downloaded=[],
            error="ConnectionResetError: controller closed before profile_export_aiperf.json",
        )

        with (
            _patched_completion_environment(tmp_path),
            mock_patch(
                "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
                DIR=tmp_path,
                RETAIN_RUNS=5,
                RETAIN_DAYS=0,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
            mock_patch(
                "aiperf.operator.handlers._completion_retry.datetime"
            ) as datetime_mock,
        ):
            from datetime import datetime

            datetime_mock.now.return_value = datetime(2024, 4, 25, 17, 2, 5, tzinfo=UTC)
            datetime_mock.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )
            with pytest.raises(
                kopf.TemporaryError,
                match=rf"{_FIXTURE_NAMESPACE}/{_FIXTURE_JOB_ID}.*ConnectionResetError",
            ):
                await completion.handle_completion(
                    body=_body_with_claim(),
                    namespace=_FIXTURE_NAMESPACE,
                    jobset_name=_FIXTURE_JOBSET,
                    job_id=_FIXTURE_JOB_ID,
                    status={
                        "workers": {"total": 4},
                        "startTime": "2026-05-17T00:00:00Z",
                    },
                    sb=sb,
                    result=result,
                )

        assert "phase" not in patch.status
        assert "conditions" not in patch.status

    @pytest.mark.asyncio
    async def test_handle_completion_cancellation_after_claim_skips_fetch_and_status(
        self, tmp_path: Path
    ) -> None:
        patch = _patch_obj()
        sb = _status_builder(patch)
        request_cancellation(job_key(_FIXTURE_NAMESPACE, _FIXTURE_JOB_ID))

        with _patched_completion_environment(tmp_path) as captured:
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 4}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=None,
            )

        assert patch.status == {}
        captured.fetch_results.assert_not_awaited()
        captured.completed.assert_not_called()
        captured.delete_jobset.assert_not_awaited()


# =============================================================================
# Non-finite metrics at the CR status boundary
# =============================================================================


class TestCompletionNonFiniteMetricBoundary:
    """NaN/Inf values should not be written into Kubernetes CR status."""

    @pytest.mark.asyncio
    async def test_handle_completion_non_finite_metrics_are_scrubbed_before_status(
        self, tmp_path: Path
    ) -> None:
        patch = _patch_obj()
        sb = _status_builder(patch)
        result = ControllerFetchResult(
            metrics={
                "metrics": {
                    "request_throughput": {"avg": float("nan"), "unit": "req/s"},
                    "request_latency": {
                        "avg": float("inf"),
                        "p99": 900.2,
                        "unit": "ms",
                    },
                }
            },
            downloaded=["profile_export_aiperf.json", "profile_export_aiperf.csv"],
        )
        _write_result_file(tmp_path, "profile_export_aiperf.json", _metrics_payload())
        _write_result_file(tmp_path, "profile_export_aiperf.csv", b"metric,value\n")

        with _patched_completion_environment(tmp_path):
            await completion.handle_completion(
                body=_body_with_claim(),
                namespace=_FIXTURE_NAMESPACE,
                jobset_name=_FIXTURE_JOBSET,
                job_id=_FIXTURE_JOB_ID,
                status={"workers": {"total": 4}, "startTime": "2026-05-17T00:00:00Z"},
                sb=sb,
                result=result,
            )

        assert _non_finite_paths(patch.status) == []
