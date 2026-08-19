# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator events module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.operator import events
from aiperf.operator.events import (
    EventReason,
    EventType,
    post_event,
)

# =============================================================================
# Test EventType Enum
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_normal_value(self) -> None:
        """Verify NORMAL has expected value."""
        assert EventType.NORMAL == "Normal"

    def test_warning_value(self) -> None:
        """Verify WARNING has expected value."""
        assert EventType.WARNING == "Warning"

    def test_case_insensitive_lookup(self) -> None:
        """Verify case-insensitive enum lookup works."""
        assert EventType("normal") == EventType.NORMAL
        assert EventType("WARNING") == EventType.WARNING


# =============================================================================
# Test EventReason Enum
# =============================================================================


class TestEventReason:
    """Tests for EventReason enum."""

    @pytest.mark.parametrize(
        "reason,expected",
        [
            param(EventReason.CREATED, "Created", id="created"),
            param(EventReason.STARTED, "Started", id="started"),
            param(EventReason.RUNNING, "Running", id="running"),
            param(EventReason.COMPLETED, "Completed", id="completed"),
            param(EventReason.FAILED, "Failed", id="failed"),
            param(EventReason.CANCELLED, "Cancelled", id="cancelled"),
            param(EventReason.SPEC_VALID, "SpecValid", id="spec_valid"),
            param(EventReason.SPEC_INVALID, "SpecInvalid", id="spec_invalid"),
            param(EventReason.ENDPOINT_REACHABLE, "EndpointReachable", id="endpoint_reachable"),
            param(EventReason.ENDPOINT_UNREACHABLE, "EndpointUnreachable", id="endpoint_unreachable"),
            param(EventReason.RESOURCES_CREATED, "ResourcesCreated", id="resources_created"),
            param(EventReason.RESOURCES_FAILED, "ResourcesFailed", id="resources_failed"),
            param(EventReason.WORKERS_READY, "WorkersReady", id="workers_ready"),
            param(EventReason.RESULTS_STORED, "ResultsStored", id="results_stored"),
            param(EventReason.RESULTS_FAILED, "ResultsFailed", id="results_failed"),
            param(EventReason.RESULTS_CLEANED, "ResultsCleaned", id="results_cleaned"),
            param(EventReason.JOB_TIMEOUT, "JobTimeout", id="job_timeout"),
            param(EventReason.POD_RESTARTS, "PodRestarts", id="pod_restarts"),
            param(EventReason.PREFLIGHT_PASSED, "PreflightPassed", id="preflight_passed"),
            param(EventReason.PREFLIGHT_FAILED, "PreflightFailed", id="preflight_failed"),
            param(EventReason.PREFLIGHT_WARNING, "PreflightWarning", id="preflight_warning"),
        ],
    )  # fmt: skip
    def test_reason_values(self, reason: EventReason, expected: str) -> None:
        """Verify all EventReason values match expected CamelCase format."""
        assert reason == expected


# =============================================================================
# Test post_event
# =============================================================================


class TestPostEvent:
    """Tests for post_event function."""

    @pytest.fixture
    def mock_kopf_info(self):
        """Mock kopf.info for testing."""
        with patch("aiperf.operator.events.kopf.info") as mock:
            yield mock

    @pytest.fixture
    def mock_kopf_warn(self):
        """Mock kopf.warn for testing."""
        with patch("aiperf.operator.events.kopf.warn") as mock:
            yield mock

    def test_posts_normal_event_with_kopf_info(
        self, sample_body: dict[str, Any], mock_kopf_info: MagicMock
    ) -> None:
        """Verify normal events use kopf.info."""
        post_event(sample_body, EventReason.CREATED, "Test message")

        mock_kopf_info.assert_called_once_with(
            sample_body, reason="Created", message="Test message"
        )

    def test_posts_warning_event_with_kopf_warn(
        self, sample_body: dict[str, Any], mock_kopf_warn: MagicMock
    ) -> None:
        """Verify warning events use kopf.warn."""
        post_event(sample_body, EventReason.FAILED, "Error message", EventType.WARNING)

        mock_kopf_warn.assert_called_once_with(
            sample_body, reason="Failed", message="Error message"
        )

    def test_defaults_to_normal_event_type(
        self, sample_body: dict[str, Any], mock_kopf_info: MagicMock
    ) -> None:
        """Verify default event_type is NORMAL."""
        post_event(sample_body, EventReason.STARTED, "Started")

        mock_kopf_info.assert_called_once()


# =============================================================================
# Test Lifecycle Event Functions
# =============================================================================


class TestLifecycleEvents:
    """Tests for lifecycle event functions."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_created(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_created posts correct event."""
        events.created(sample_body, "job-123", 5)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.CREATED,
            "Created benchmark job job-123 with 5 workers",
        )

    def test_event_started(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_started posts correct event."""
        events.started(sample_body, "job-456")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.STARTED,
            "Benchmark job-456 started",
        )

    def test_event_completed_without_duration(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_completed posts correct event without duration."""
        events.completed(sample_body, "job-789")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.COMPLETED,
            "Benchmark job-789 completed successfully",
        )

    def test_event_completed_with_duration(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_completed includes duration when provided."""
        events.completed(sample_body, "job-789", 45.5)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.COMPLETED,
            "Benchmark job-789 completed successfully in 45.5s",
        )

    def test_event_completed_with_zero_duration(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_completed includes zero duration (not omitted as falsy)."""
        events.completed(sample_body, "job-789", 0.0)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.COMPLETED,
            "Benchmark job-789 completed successfully in 0.0s",
        )

    def test_event_failed(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_failed posts warning event."""
        events.failed(sample_body, "job-err", "Connection refused")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.FAILED,
            "Benchmark job-err failed: Connection refused",
            EventType.WARNING,
        )

    def test_event_cancelled(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_cancelled posts warning event."""
        events.cancelled(sample_body, "job-cancel")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.CANCELLED,
            "Benchmark job-cancel was cancelled",
            EventType.WARNING,
        )


# =============================================================================
# Test Validation Event Functions
# =============================================================================


class TestValidationEvents:
    """Tests for validation event functions."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_spec_valid(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_spec_valid posts correct event."""
        events.spec_valid(sample_body)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.SPEC_VALID,
            "Spec validation passed",
        )

    def test_event_spec_invalid(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_spec_invalid posts warning event."""
        events.spec_invalid(sample_body, "Missing required field")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.SPEC_INVALID,
            "Spec validation failed: Missing required field",
            EventType.WARNING,
        )

    def test_event_endpoint_reachable(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_endpoint_reachable posts correct event."""
        events.endpoint_reachable(sample_body, "http://api.example.com")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.ENDPOINT_REACHABLE,
            "Endpoint http://api.example.com is reachable",
        )

    def test_event_endpoint_unreachable(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_endpoint_unreachable posts warning event."""
        events.endpoint_unreachable(
            sample_body, "http://api.example.com", "Connection timeout"
        )

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.ENDPOINT_UNREACHABLE,
            "Endpoint http://api.example.com unreachable: Connection timeout",
            EventType.WARNING,
        )


# =============================================================================
# Test Resource Event Functions
# =============================================================================


class TestResourceEvents:
    """Tests for resource event functions."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_resources_created(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_resources_created posts correct event."""
        events.resources_created(sample_body, "config-job-123", "jobset-job-123")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.RESOURCES_CREATED,
            "Created ConfigMap/config-job-123 and JobSet/jobset-job-123",
        )

    def test_event_workers_ready(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_workers_ready posts correct event."""
        events.workers_ready(sample_body, 5, 5)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.WORKERS_READY,
            "All workers ready (5/5)",
        )


# =============================================================================
# Test Results Event Functions
# =============================================================================


class TestResultsEvents:
    """Tests for results event functions."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_results_stored(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_results_stored posts correct event."""
        events.results_stored(sample_body, "/results/job-123", 3)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.RESULTS_STORED,
            "Stored 3 result files to /results/job-123",
        )

    def test_event_results_failed(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_results_failed posts warning event."""
        events.results_failed(sample_body, "Disk full")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.RESULTS_FAILED,
            "Failed to store results: Disk full",
            EventType.WARNING,
        )

    def test_event_results_cleaned(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_results_cleaned posts correct event."""
        events.results_cleaned(sample_body, "job-old", 30)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.RESULTS_CLEANED,
            "Cleaned up results for job-old (age: 30 days)",
        )


# =============================================================================
# Test Reliability Event Functions
# =============================================================================


class TestReliabilityEvents:
    """Tests for reliability event functions (timeout, restarts)."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_job_timeout(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_job_timeout posts warning event with elapsed time."""
        events.job_timeout(sample_body, "job-slow", 3600.0)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.JOB_TIMEOUT,
            "Benchmark job-slow timed out after 3600s",
            EventType.WARNING,
        )

    def test_event_job_timeout_fractional_seconds(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_job_timeout formats fractional seconds correctly."""
        events.job_timeout(sample_body, "job-slow", 123.7)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.JOB_TIMEOUT,
            "Benchmark job-slow timed out after 124s",
            EventType.WARNING,
        )

    def test_event_pod_restarts(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_pod_restarts posts warning event with pod details."""
        events.pod_restarts(sample_body, "worker-0-0", 5, "OOMKilled")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.POD_RESTARTS,
            "Pod worker-0-0 has restarted 5 times: OOMKilled",
            EventType.WARNING,
        )

    def test_event_pod_restarts_crashloop(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_pod_restarts works with CrashLoopBackOff reason."""
        events.pod_restarts(sample_body, "controller-0-0", 10, "CrashLoopBackOff")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.POD_RESTARTS,
            "Pod controller-0-0 has restarted 10 times: CrashLoopBackOff",
            EventType.WARNING,
        )


# =============================================================================
# Test Preflight Event Functions
# =============================================================================


class TestPreflightEvents:
    """Tests for preflight event functions (passed, failed, warning)."""

    @pytest.fixture
    def mock_post_event(self):
        """Mock post_event for testing individual event functions."""
        with patch("aiperf.operator.events.post_event") as mock:
            yield mock

    def test_event_preflight_passed(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_preflight_passed posts normal event with check count."""
        events.preflight_passed(sample_body, 13)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.PREFLIGHT_PASSED,
            "All 13 pre-flight checks passed",
        )

    def test_event_preflight_passed_zero_checks(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_preflight_passed handles zero checks."""
        events.preflight_passed(sample_body, 0)

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.PREFLIGHT_PASSED,
            "All 0 pre-flight checks passed",
        )

    def test_event_preflight_failed(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_preflight_failed posts warning event with error message."""
        events.preflight_failed(sample_body, "RBAC permissions missing")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.PREFLIGHT_FAILED,
            "Pre-flight checks failed: RBAC permissions missing",
            EventType.WARNING,
        )

    def test_event_preflight_failed_multiline_error(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_preflight_failed preserves full error string."""
        events.preflight_failed(sample_body, "2 checks failed: RBAC, namespace")

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.PREFLIGHT_FAILED,
            "Pre-flight checks failed: 2 checks failed: RBAC, namespace",
            EventType.WARNING,
        )

    def test_event_preflight_warning(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify event_preflight_warning posts warning with check name and message."""
        events.preflight_warning(
            sample_body, "resource_quotas", "Quota limit near capacity"
        )

        mock_post_event.assert_called_once_with(
            sample_body,
            EventReason.PREFLIGHT_WARNING,
            "Pre-flight warning [resource_quotas]: Quota limit near capacity",
            EventType.WARNING,
        )

    def test_event_preflight_warning_includes_check_name(
        self, sample_body: dict[str, Any], mock_post_event: MagicMock
    ) -> None:
        """Verify the check_name is embedded in the warning message."""
        events.preflight_warning(sample_body, "network_policies", "Policies detected")

        call_args = mock_post_event.call_args
        message = call_args[0][2]
        assert "[network_policies]" in message
        assert "Policies detected" in message
