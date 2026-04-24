# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kubernetes.models module.

Focuses on:
- JobSetInfo construction via from_raw and direct instantiation
- Status parsing from Kubernetes conditions
- Derived properties (job_id, created, progress)
- PodSummary construction and ready_str formatting
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.kubernetes.constants import AIPerfLabels, Annotations, ProgressAnnotations
from aiperf.kubernetes.enums import JobSetStatus
from aiperf.kubernetes.models import AIPerfJobCR, JobSetInfo, PodSummary

# =============================================================================
# Helpers
# =============================================================================


def _make_raw_jobset(
    *,
    name: str = "aiperf-bench-001",
    namespace: str = "default",
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    conditions: list[dict[str, str]] | None = None,
    creation_timestamp: str = "2026-01-15T10:30:00Z",
) -> dict[str, Any]:
    """Build a minimal raw JobSet dict for testing."""
    raw: dict[str, Any] = {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": creation_timestamp,
        },
        "status": {
            "conditions": conditions or [],
        },
    }
    if labels:
        raw["metadata"]["labels"] = labels
    if annotations:
        raw["metadata"]["annotations"] = annotations
    return raw


# =============================================================================
# JobSetInfo.from_raw — construction
# =============================================================================


class TestJobSetInfoFromRaw:
    """Verify from_raw extracts fields correctly from raw Kubernetes dicts."""

    def test_from_raw_extracts_name_and_namespace(self) -> None:
        raw = _make_raw_jobset(name="my-bench", namespace="perf")
        info = JobSetInfo.from_raw(raw)
        assert info.name == "my-bench"
        assert info.namespace == "perf"

    def test_from_raw_stores_full_jobset_dict(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.jobset is raw

    def test_from_raw_extracts_custom_name_from_label(self) -> None:
        raw = _make_raw_jobset(labels={AIPerfLabels.NAME: "nightly-gpt4"})
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name == "nightly-gpt4"

    def test_from_raw_custom_name_none_when_label_absent(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name is None

    def test_from_raw_extracts_model_from_annotation(self) -> None:
        raw = _make_raw_jobset(annotations={Annotations.MODEL: "llama-70b"})
        info = JobSetInfo.from_raw(raw)
        assert info.model == "llama-70b"

    def test_from_raw_extracts_endpoint_from_annotation(self) -> None:
        raw = _make_raw_jobset(annotations={Annotations.ENDPOINT: "http://llm:8000/v1"})
        info = JobSetInfo.from_raw(raw)
        assert info.endpoint == "http://llm:8000/v1"

    def test_from_raw_model_and_endpoint_none_when_absent(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.model is None
        assert info.endpoint is None

    def test_from_raw_handles_missing_labels_and_annotations_keys(self) -> None:
        """metadata with no labels/annotations keys at all should not raise."""
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "status": {"conditions": []},
        }
        info = JobSetInfo.from_raw(raw)
        assert info.custom_name is None
        assert info.model is None
        assert info.endpoint is None


# =============================================================================
# JobSetInfo._parse_status
# =============================================================================


class TestJobSetInfoParseStatus:
    """Verify status parsing from Kubernetes conditions."""

    def test_no_conditions_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_completed_true_returns_completed(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Completed", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.COMPLETED

    def test_failed_true_without_controller_failure_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Failed", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_failed_true_with_controller_failure_returns_failed(self) -> None:
        raw = _make_raw_jobset(
            conditions=[{"type": "Failed", "status": "True"}],
        )
        raw["status"]["replicatedJobsStatus"] = [
            {"name": "workers", "failed": 3, "ready": 247},
            {"name": "controller", "failed": 1, "ready": 0},
        ]
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.FAILED

    def test_failed_true_with_worker_only_failure_returns_running(self) -> None:
        raw = _make_raw_jobset(
            conditions=[{"type": "Failed", "status": "True"}],
        )
        raw["status"]["replicatedJobsStatus"] = [
            {"name": "workers", "failed": 3, "ready": 247},
            {"name": "controller", "failed": 0, "ready": 1},
        ]
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_completed_false_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Completed", "status": "False"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_failed_false_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Failed", "status": "False"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING

    def test_both_completed_and_failed_completed_wins(self) -> None:
        """Completed is checked first, so it takes priority."""
        raw = _make_raw_jobset(
            conditions=[
                {"type": "Completed", "status": "True"},
                {"type": "Failed", "status": "True"},
            ]
        )
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.COMPLETED

    def test_missing_status_key_returns_running(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
        }
        assert JobSetInfo._parse_status(raw) == JobSetStatus.RUNNING

    def test_unrecognized_condition_type_returns_running(self) -> None:
        raw = _make_raw_jobset(conditions=[{"type": "Suspended", "status": "True"}])
        info = JobSetInfo.from_raw(raw)
        assert info.status == JobSetStatus.RUNNING


# =============================================================================
# JobSetInfo derived properties
# =============================================================================


class TestJobSetInfoJobId:
    """Verify job_id property falls back to name."""

    def test_job_id_from_label(self) -> None:
        raw = _make_raw_jobset(labels={AIPerfLabels.JOB_ID: "uuid-123"})
        info = JobSetInfo.from_raw(raw)
        assert info.job_id == "uuid-123"

    def test_job_id_falls_back_to_name(self) -> None:
        raw = _make_raw_jobset(name="bench-xyz")
        info = JobSetInfo.from_raw(raw)
        assert info.job_id == "bench-xyz"


class TestJobSetInfoCreated:
    """Verify created timestamp property."""

    def test_created_returns_timestamp(self) -> None:
        raw = _make_raw_jobset(creation_timestamp="2026-03-11T08:00:00Z")
        info = JobSetInfo.from_raw(raw)
        assert info.created == "2026-03-11T08:00:00Z"

    def test_created_returns_empty_when_missing(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "status": {"conditions": []},
        }
        info = JobSetInfo.from_raw(raw)
        assert info.created == ""


class TestJobSetInfoProgress:
    """Verify progress property builds human-readable strings."""

    def test_progress_none_when_no_status_annotation(self) -> None:
        raw = _make_raw_jobset()
        info = JobSetInfo.from_raw(raw)
        assert info.progress is None

    def test_progress_with_phase_only(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Warmup",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Warmup"

    def test_progress_with_phase_and_requests(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Benchmark",
                ProgressAnnotations.REQUESTS: "500/1000",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Benchmark 500/1000"

    def test_progress_with_all_fields(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PHASE: "Benchmark",
                ProgressAnnotations.REQUESTS: "500/1000",
                ProgressAnnotations.PERCENT: "50",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "Benchmark 500/1000 (50%)"

    def test_progress_with_percent_only(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "active",
                ProgressAnnotations.PERCENT: "75",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "(75%)"

    def test_progress_falls_back_to_status_when_no_parts(self) -> None:
        """When STATUS is set but phase/requests/percent are all absent."""
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "initializing",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress == "initializing"

    def test_progress_none_when_status_empty_string(self) -> None:
        raw = _make_raw_jobset(
            annotations={
                ProgressAnnotations.STATUS: "",
            }
        )
        info = JobSetInfo.from_raw(raw)
        assert info.progress is None


# =============================================================================
# JobSetInfo — integration with conftest fixtures
# =============================================================================


class TestJobSetInfoWithFixtures:
    """Verify from_raw works with the shared sample_jobset fixtures."""

    def test_running_jobset(self, sample_running_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_running_jobset)
        assert info.status == JobSetStatus.RUNNING
        assert info.name == "aiperf-test-job"

    def test_completed_jobset(self, sample_completed_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_completed_jobset)
        assert info.status == JobSetStatus.COMPLETED

    def test_failed_jobset(self, sample_failed_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_failed_jobset)
        assert info.status == JobSetStatus.FAILED

    def test_job_id_from_fixture(self, sample_jobset: dict[str, Any]) -> None:
        info = JobSetInfo.from_raw(sample_jobset)
        assert info.job_id == "test-job-123"


# =============================================================================
# PodSummary
# =============================================================================


class TestPodSummary:
    """Verify PodSummary construction and ready_str formatting."""

    @pytest.mark.parametrize(
        "ready,total,expected",
        [
            (0, 0, "0/0"),
            (0, 3, "0/3"),
            (2, 3, "2/3"),
            (5, 5, "5/5"),
        ],
    )  # fmt: skip
    def test_ready_str_format(self, ready: int, total: int, expected: str) -> None:
        summary = PodSummary(ready=ready, total=total, restarts=0)
        assert summary.ready_str == expected

    def test_stores_restart_count(self) -> None:
        summary = PodSummary(ready=1, total=2, restarts=7)
        assert summary.restarts == 7

    def test_all_fields_accessible(self) -> None:
        summary = PodSummary(ready=3, total=4, restarts=1)
        assert summary.ready == 3
        assert summary.total == 4
        assert summary.restarts == 1


# =============================================================================
# AIPerfJobInfo helpers
# =============================================================================


def _make_raw_aiperfjob(
    *,
    name: str = "aiperf-bench-001",
    namespace: str = "default",
    creation_timestamp: str = "2026-01-15T10:30:00Z",
    spec: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal raw AIPerfJob CR dict for testing."""
    metadata: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "creationTimestamp": creation_timestamp,
    }
    if labels:
        metadata["labels"] = labels
    if annotations:
        metadata["annotations"] = annotations
    raw: dict[str, Any] = {
        "metadata": metadata,
        "spec": spec or {},
        "status": status or {},
    }
    return raw


# =============================================================================
# AIPerfJobInfo.from_raw — basic construction
# =============================================================================


class TestAIPerfJobInfoFromRaw:
    """Verify from_raw extracts metadata fields from raw K8s dicts."""

    def test_from_raw_extracts_name_and_namespace(self) -> None:
        raw = _make_raw_aiperfjob(name="my-job", namespace="perf")
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.name == "my-job"
        assert info.namespace == "perf"

    def test_from_raw_extracts_creation_timestamp(self) -> None:
        raw = _make_raw_aiperfjob(creation_timestamp="2026-03-11T08:00:00Z")
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.created == "2026-03-11T08:00:00Z"

    def test_from_raw_missing_creation_timestamp_defaults_empty(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "spec": {},
            "status": {},
        }
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.created == ""

    def test_from_raw_extracts_job_id_from_status(self) -> None:
        raw = _make_raw_aiperfjob(status={"jobId": "uuid-abc-123"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.job_id == "uuid-abc-123"

    def test_from_raw_job_id_falls_back_to_name(self) -> None:
        raw = _make_raw_aiperfjob(name="fallback-name", status={})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.job_id == "fallback-name"

    def test_from_raw_extracts_jobset_name(self) -> None:
        raw = _make_raw_aiperfjob(status={"jobSetName": "managed-js-001"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.jobset_name == "managed-js-001"

    def test_from_raw_jobset_name_none_when_absent(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.jobset_name is None

    def test_from_raw_extracts_current_phase(self) -> None:
        raw = _make_raw_aiperfjob(status={"currentPhase": "warmup"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.current_phase == "warmup"

    def test_from_raw_current_phase_none_when_absent(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.current_phase is None

    def test_from_raw_extracts_error(self) -> None:
        raw = _make_raw_aiperfjob(status={"error": "OOMKilled"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.error == "OOMKilled"

    def test_from_raw_error_none_when_absent(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.error is None

    def test_from_raw_extracts_start_time(self) -> None:
        raw = _make_raw_aiperfjob(status={"startTime": "2026-01-15T10:31:00Z"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.start_time == "2026-01-15T10:31:00Z"

    def test_from_raw_extracts_completion_time(self) -> None:
        raw = _make_raw_aiperfjob(status={"completionTime": "2026-01-15T11:00:00Z"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.completion_time == "2026-01-15T11:00:00Z"

    def test_from_raw_times_none_when_absent(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.start_time is None
        assert info.completion_time is None


# =============================================================================
# AIPerfJobInfo — phase defaults
# =============================================================================


class TestAIPerfJobInfoPhase:
    """Verify phase extraction and defaults."""

    def test_phase_from_status(self) -> None:
        raw = _make_raw_aiperfjob(status={"phase": "Running"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.phase == "Running"

    @pytest.mark.parametrize(
        "status_dict",
        [
            param({}, id="empty-status"),
            param({"phase": None}, id="explicit-none-phase"),
        ],
    )  # fmt: skip
    def test_phase_defaults_to_pending(self, status_dict: dict[str, Any]) -> None:
        raw = _make_raw_aiperfjob(status=status_dict)
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.phase == "Pending"

    def test_phase_completed(self) -> None:
        raw = _make_raw_aiperfjob(status={"phase": "Completed"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.phase == "Completed"

    def test_phase_failed(self) -> None:
        raw = _make_raw_aiperfjob(status={"phase": "Failed"})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.phase == "Failed"


# =============================================================================
# AIPerfJobInfo — model extraction
# =============================================================================


class TestAIPerfJobInfoModel:
    """Verify spec.benchmark.models parsing covers all shapes."""

    def test_model_from_string(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"models": "llama-70b"}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model == "llama-70b"

    def test_model_from_single_element_list(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"models": ["gpt-4"]}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model == "gpt-4"

    def test_model_from_multi_element_list_takes_first(self) -> None:
        raw = _make_raw_aiperfjob(
            spec={"benchmark": {"models": ["model-a", "model-b", "model-c"]}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model == "model-a"

    def test_model_none_when_empty_list(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"models": []}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model is None

    def test_model_none_when_absent(self) -> None:
        raw = _make_raw_aiperfjob(spec={})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model is None

    def test_model_none_when_list_contains_non_string(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"models": [{"name": "llama"}]}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model is None

    def test_model_rejects_invalid_type(self) -> None:
        from pydantic import ValidationError

        raw = _make_raw_aiperfjob(spec={"benchmark": {"models": 42}})
        with pytest.raises(ValidationError):
            AIPerfJobCR.model_validate(raw)


# =============================================================================
# AIPerfJobInfo — endpoint extraction
# =============================================================================


class TestAIPerfJobInfoEndpoint:
    """Verify spec.benchmark.endpoint URL extraction."""

    def test_endpoint_from_url(self) -> None:
        raw = _make_raw_aiperfjob(
            spec={"benchmark": {"endpoint": {"url": "http://llm:8000/v1"}}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint == "http://llm:8000/v1"

    def test_endpoint_from_urls_fallback(self) -> None:
        raw = _make_raw_aiperfjob(
            spec={
                "benchmark": {"endpoint": {"urls": ["http://a:8000", "http://b:8000"]}}
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint == "http://a:8000"

    def test_endpoint_url_takes_priority_over_urls(self) -> None:
        raw = _make_raw_aiperfjob(
            spec={
                "benchmark": {
                    "endpoint": {
                        "url": "http://primary:8000",
                        "urls": ["http://fallback:8000"],
                    }
                }
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint == "http://primary:8000"

    def test_endpoint_none_when_endpoint_absent(self) -> None:
        raw = _make_raw_aiperfjob(spec={})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint is None

    def test_endpoint_none_when_endpoint_empty(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"endpoint": {}}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint is None

    def test_endpoint_none_when_urls_empty_list(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"endpoint": {"urls": []}}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint is None

    def test_endpoint_none_when_url_empty_string(self) -> None:
        raw = _make_raw_aiperfjob(spec={"benchmark": {"endpoint": {"url": ""}}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.endpoint is None


# =============================================================================
# AIPerfJobInfo — workers extraction
# =============================================================================


class TestAIPerfJobInfoWorkers:
    """Verify status.workers parsing."""

    def test_workers_from_status(self) -> None:
        raw = _make_raw_aiperfjob(status={"workers": {"ready": 3, "total": 5}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_ready == 3
        assert info.workers_total == 5

    def test_workers_default_zero_when_absent(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_ready == 0
        assert info.workers_total == 0

    def test_workers_default_zero_when_empty(self) -> None:
        raw = _make_raw_aiperfjob(status={"workers": {}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_ready == 0
        assert info.workers_total == 0

    def test_workers_partial_ready_only(self) -> None:
        raw = _make_raw_aiperfjob(status={"workers": {"ready": 2}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_ready == 2
        assert info.workers_total == 0

    def test_workers_partial_total_only(self) -> None:
        raw = _make_raw_aiperfjob(status={"workers": {"total": 4}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_ready == 0
        assert info.workers_total == 4


# =============================================================================
# AIPerfJobInfo — workers_str property
# =============================================================================


class TestAIPerfJobInfoWorkersStr:
    """Verify workers_str formatting property."""

    @pytest.mark.parametrize(
        "ready,total,expected",
        [
            param(0, 0, "0/0", id="zero-zero"),
            param(0, 3, "0/3", id="none-ready"),
            param(2, 5, "2/5", id="partial"),
            param(5, 5, "5/5", id="all-ready"),
        ],
    )  # fmt: skip
    def test_workers_str_format(self, ready: int, total: int, expected: str) -> None:
        raw = _make_raw_aiperfjob(status={"workers": {"ready": ready, "total": total}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.workers_str == expected


# =============================================================================
# AIPerfJobInfo — progress extraction
# =============================================================================


class TestAIPerfJobInfoProgress:
    """Verify _extract_progress from status.phases."""

    def test_progress_from_single_phase(self) -> None:
        raw = _make_raw_aiperfjob(
            status={"phases": {"warmup": {"requestsProgressPercent": 50.0}}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.progress_percent == 50.0

    def test_progress_from_multiple_phases_takes_last(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "phases": {
                    "warmup": {"requestsProgressPercent": 100.0},
                    "benchmark": {"requestsProgressPercent": 35.5},
                }
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        # _extract_progress iterates all phases, keeping the last one seen
        assert info.progress_percent == 35.5

    def test_progress_none_when_no_phases(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.progress_percent is None

    def test_progress_none_when_phases_empty(self) -> None:
        raw = _make_raw_aiperfjob(status={"phases": {}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.progress_percent is None

    def test_progress_none_when_phase_has_no_percent(self) -> None:
        raw = _make_raw_aiperfjob(
            status={"phases": {"warmup": {"name": "warmup", "state": "active"}}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.progress_percent is None

    def test_progress_integer_converted_to_float(self) -> None:
        raw = _make_raw_aiperfjob(
            status={"phases": {"benchmark": {"requestsProgressPercent": 75}}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.progress_percent == 75.0
        assert isinstance(info.progress_percent, float)


# =============================================================================
# AIPerfJobInfo — metrics extraction (throughput / latency)
# =============================================================================


class TestAIPerfJobInfoMetrics:
    """Verify liveSummary vs summary fallback for throughput and latency."""

    def test_metrics_from_live_summary(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "liveSummary": {"throughput_rps": 42.5, "latency_p99_ms": 120.3},
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 42.5
        assert info.latency_p99_ms == 120.3

    def test_metrics_from_summary_fallback(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "summary": {"throughput_rps": 38.0, "latency_p99_ms": 95.1},
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 38.0
        assert info.latency_p99_ms == 95.1

    def test_metrics_live_summary_takes_priority(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "liveSummary": {"throughput_rps": 50.0, "latency_p99_ms": 100.0},
                "summary": {"throughput_rps": 30.0, "latency_p99_ms": 200.0},
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 50.0
        assert info.latency_p99_ms == 100.0

    def test_metrics_snake_case_keys(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "summary": {"throughput_rps": 25.0, "latency_p99_ms": 150.0},
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 25.0
        assert info.latency_p99_ms == 150.0

    def test_metrics_none_when_no_summary(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps is None
        assert info.latency_p99_ms is None

    def test_metrics_none_when_summaries_empty(self) -> None:
        raw = _make_raw_aiperfjob(status={"liveSummary": {}, "summary": {}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps is None
        assert info.latency_p99_ms is None

    def test_metrics_integer_converted_to_float(self) -> None:
        raw = _make_raw_aiperfjob(
            status={"liveSummary": {"throughput_rps": 100, "latency_p99_ms": 50}}
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 100.0
        assert isinstance(info.throughput_rps, float)
        assert info.latency_p99_ms == 50.0
        assert isinstance(info.latency_p99_ms, float)

    def test_metrics_partial_throughput_only(self) -> None:
        raw = _make_raw_aiperfjob(status={"liveSummary": {"throughput_rps": 60.0}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps == 60.0
        assert info.latency_p99_ms is None

    def test_metrics_partial_latency_only(self) -> None:
        raw = _make_raw_aiperfjob(status={"liveSummary": {"latency_p99_ms": 80.0}})
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.throughput_rps is None
        assert info.latency_p99_ms == 80.0


# =============================================================================
# AIPerfJobInfo — missing / empty fields (graceful degradation)
# =============================================================================


class TestAIPerfJobInfoGracefulDefaults:
    """Verify graceful handling of minimal or empty raw dicts."""

    def test_minimal_raw_dict(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "x", "namespace": "ns"},
            "spec": {},
            "status": {},
        }
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.name == "x"
        assert info.namespace == "ns"
        assert info.phase == "Pending"
        assert info.job_id == "x"
        assert info.jobset_name is None
        assert info.workers_ready == 0
        assert info.workers_total == 0
        assert info.current_phase is None
        assert info.error is None
        assert info.start_time is None
        assert info.completion_time is None
        assert info.created == ""
        assert info.progress_percent is None
        assert info.throughput_rps is None
        assert info.latency_p99_ms is None
        assert info.model is None
        assert info.endpoint is None

    def test_completely_empty_metadata(self) -> None:
        raw: dict[str, Any] = {"metadata": {}, "spec": {}, "status": {}}
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.name == ""
        assert info.namespace == ""

    def test_missing_spec_key(self) -> None:
        raw: dict[str, Any] = {
            "metadata": {"name": "a", "namespace": "b"},
            "status": {},
        }
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.model is None
        assert info.endpoint is None

    def test_missing_status_key(self) -> None:
        raw: dict[str, Any] = {"metadata": {"name": "a", "namespace": "b"}, "spec": {}}
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.phase == "Pending"
        assert info.workers_ready == 0
        assert info.workers_total == 0

    def test_completely_empty_raw(self) -> None:
        raw: dict[str, Any] = {}
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.name == ""
        assert info.namespace == ""
        assert info.phase == "Pending"


# =============================================================================
# AIPerfJobInfo — full lifecycle snapshot
# =============================================================================


class TestAIPerfJobInfoFullSnapshot:
    """Verify from_raw with a fully populated raw dict."""

    def test_full_aiperfjob_cr(self) -> None:
        raw = _make_raw_aiperfjob(
            name="nightly-bench-001",
            namespace="perf-testing",
            creation_timestamp="2026-03-13T00:00:00Z",
            spec={
                "benchmark": {
                    "models": ["llama-3-70b"],
                    "endpoint": {"url": "http://llm-svc:8000/v1"},
                },
            },
            status={
                "phase": "Running",
                "jobId": "job-uuid-xyz",
                "jobSetName": "nightly-bench-001-js",
                "currentPhase": "benchmark",
                "startTime": "2026-03-13T00:01:00Z",
                "workers": {"ready": 4, "total": 4},
                "phases": {
                    "warmup": {"requestsProgressPercent": 100.0},
                    "benchmark": {"requestsProgressPercent": 62.5},
                },
                "liveSummary": {
                    "throughput_rps": 85.3,
                    "latency_p99_ms": 45.7,
                },
            },
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        assert info.name == "nightly-bench-001"
        assert info.namespace == "perf-testing"
        assert info.phase == "Running"
        assert info.job_id == "job-uuid-xyz"
        assert info.jobset_name == "nightly-bench-001-js"
        assert info.current_phase == "benchmark"
        assert info.start_time == "2026-03-13T00:01:00Z"
        assert info.completion_time is None
        assert info.created == "2026-03-13T00:00:00Z"
        assert info.workers_ready == 4
        assert info.workers_total == 4
        assert info.workers_str == "4/4"
        assert info.progress_percent == 62.5
        assert info.throughput_rps == 85.3
        assert info.latency_p99_ms == 45.7
        assert info.model == "llama-3-70b"
        assert info.endpoint == "http://llm-svc:8000/v1"


# =============================================================================
# AIPerfJobInfo — to_k8s_dict serialization (inherited from K8sCamelModel)
# =============================================================================


class TestAIPerfJobInfoSerialization:
    """Verify camelCase serialization inherited from K8sCamelModel."""

    def test_to_k8s_dict_uses_camel_case(self) -> None:
        raw = _make_raw_aiperfjob(
            status={
                "phase": "Running",
                "workers": {"ready": 2, "total": 3},
                "liveSummary": {"throughput_rps": 10.0},
            }
        )
        info = AIPerfJobCR.model_validate(raw).to_info()
        d = info.to_k8s_dict()
        assert "workersReady" in d
        assert "workersTotal" in d
        assert "throughputRps" in d
        assert "latencyP99Ms" not in d  # None excluded

    def test_to_k8s_dict_excludes_none(self) -> None:
        raw = _make_raw_aiperfjob()
        info = AIPerfJobCR.model_validate(raw).to_info()
        d = info.to_k8s_dict()
        assert "jobsetName" not in d
        assert "error" not in d
        assert "startTime" not in d
        assert "completionTime" not in d
        assert "currentPhase" not in d
        assert "progressPercent" not in d
        assert "throughputRps" not in d
        assert "latencyP99Ms" not in d
        assert "model" not in d
        assert "endpoint" not in d
