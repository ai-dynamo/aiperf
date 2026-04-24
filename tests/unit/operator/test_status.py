# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.status module."""

import re
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

from aiperf.operator.status import (
    ConditionManager,
    ConditionType,
    Phase,
    StatusBuilder,
    format_timestamp,
    parse_timestamp,
)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_returns_iso_format(self) -> None:
        """Test timestamp is in ISO 8601 format."""
        timestamp = format_timestamp()
        # Should match ISO 8601 format with Z suffix
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z"
        assert re.match(pattern, timestamp), f"Invalid timestamp format: {timestamp}"

    def test_ends_with_z(self) -> None:
        """Test timestamp ends with Z (UTC indicator)."""
        timestamp = format_timestamp()
        assert timestamp.endswith("Z")

    def test_no_plus_offset(self) -> None:
        """Test timestamp does not contain +00:00."""
        timestamp = format_timestamp()
        assert "+00:00" not in timestamp


class TestParseTimestamp:
    """Tests for parse_timestamp function."""

    @pytest.mark.parametrize(
        "timestamp,expected_year,expected_month,expected_day",
        [
            param("2026-01-15T10:30:00Z", 2026, 1, 15, id="with_z_suffix"),
            param("2026-06-20T14:45:30+00:00", 2026, 6, 20, id="with_plus_offset"),
            param("2025-12-31T23:59:59Z", 2025, 12, 31, id="end_of_year"),
        ],
    )  # fmt: skip
    def test_parse_timestamp_formats(
        self,
        timestamp: str,
        expected_year: int,
        expected_month: int,
        expected_day: int,
    ) -> None:
        """Test parsing various timestamp formats."""
        result = parse_timestamp(timestamp)
        assert result.year == expected_year
        assert result.month == expected_month
        assert result.day == expected_day
        assert result.tzinfo == timezone.utc

    def test_parse_timestamp_with_z_suffix_returns_utc(self) -> None:
        """Test that Z suffix timestamps are parsed as UTC."""
        result = parse_timestamp("2026-01-15T10:30:00Z")
        assert result.tzinfo == timezone.utc
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0

    def test_parse_timestamp_with_plus_offset_returns_utc(self) -> None:
        """Test that +00:00 timestamps are parsed as UTC."""
        result = parse_timestamp("2026-01-15T10:30:00+00:00")
        assert result.tzinfo == timezone.utc

    def test_parse_timestamp_with_microseconds(self) -> None:
        """Test parsing timestamp with microseconds."""
        result = parse_timestamp("2026-01-15T10:30:00.123456Z")
        assert result.microsecond == 123456

    def test_parse_timestamp_roundtrip(self) -> None:
        """Test that format_timestamp output can be parsed back."""
        original = format_timestamp()
        parsed = parse_timestamp(original)
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo == timezone.utc


class TestPhase:
    """Tests for Phase enum."""

    @pytest.mark.parametrize(
        "phase,expected",
        [
            param(Phase.PENDING, "Pending", id="pending"),
            param(Phase.QUEUED, "Queued", id="queued"),
            param(Phase.INITIALIZING, "Initializing", id="initializing"),
            param(Phase.RUNNING, "Running", id="running"),
            param(Phase.COMPLETED, "Completed", id="completed"),
            param(Phase.FAILED, "Failed", id="failed"),
            param(Phase.CANCELLED, "Cancelled", id="cancelled"),
        ],
    )  # fmt: skip
    def test_phase_values(self, phase: Phase, expected: str) -> None:
        """Test Phase enum values match expected strings."""
        assert phase.value == expected

    def test_phase_from_string(self) -> None:
        """Test Phase can be created from string (case-insensitive)."""
        assert Phase("Pending") == Phase.PENDING
        assert Phase("RUNNING") == Phase.RUNNING
        assert Phase("completed") == Phase.COMPLETED

    def test_all_phases_defined(self) -> None:
        """Test all expected phases are defined."""
        expected_phases = [
            "PENDING",
            "QUEUED",
            "INITIALIZING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        ]
        for phase_name in expected_phases:
            assert hasattr(Phase, phase_name)


class TestConditionType:
    """Tests for ConditionType enum."""

    @pytest.mark.parametrize(
        "condition_type,expected",
        [
            param(ConditionType.CONFIG_VALID, "ConfigValid", id="config_valid"),
            param(ConditionType.ENDPOINT_REACHABLE, "EndpointReachable", id="endpoint_reachable"),
            param(ConditionType.RESOURCES_CREATED, "ResourcesCreated", id="resources_created"),
            param(ConditionType.WORKERS_READY, "WorkersReady", id="workers_ready"),
            param(ConditionType.BENCHMARK_RUNNING, "BenchmarkRunning", id="benchmark_running"),
            param(ConditionType.RESULTS_AVAILABLE, "ResultsAvailable", id="results_available"),
        ],
    )  # fmt: skip
    def test_condition_type_values(
        self, condition_type: ConditionType, expected: str
    ) -> None:
        """Test ConditionType enum values match expected strings."""
        assert condition_type.value == expected

    def test_condition_type_from_string(self) -> None:
        """Test ConditionType can be created from string (case-insensitive by value)."""
        assert ConditionType("ConfigValid") == ConditionType.CONFIG_VALID
        # Case-insensitive match on the VALUE, not the name
        assert ConditionType("workersready") == ConditionType.WORKERS_READY
        assert ConditionType("WorkersReady") == ConditionType.WORKERS_READY


class TestConditionManager:
    """Tests for ConditionManager class."""

    def test_init_empty(self) -> None:
        """Test ConditionManager initializes with no conditions."""
        manager = ConditionManager()
        assert manager.to_list() == []

    def test_set_condition_true(self) -> None:
        """Test setting a condition to True."""
        manager = ConditionManager()
        manager.set_condition(
            ConditionType.CONFIG_VALID,
            True,
            reason="ConfigParsed",
            message="Configuration validated successfully",
        )
        conditions = manager.to_list()
        assert len(conditions) == 1
        assert conditions[0]["type"] == "ConfigValid"
        assert conditions[0]["status"] == "True"
        assert conditions[0]["reason"] == "ConfigParsed"
        assert conditions[0]["message"] == "Configuration validated successfully"
        assert "lastTransitionTime" in conditions[0]

    def test_set_condition_false(self) -> None:
        """Test setting a condition to False."""
        manager = ConditionManager()
        manager.set_condition(
            ConditionType.WORKERS_READY,
            False,
            reason="WorkersStarting",
            message="2/5 workers ready",
        )
        condition = manager.get_condition(ConditionType.WORKERS_READY)
        assert condition is not None
        assert condition["status"] == "False"

    def test_set_multiple_conditions(self) -> None:
        """Test setting multiple different conditions."""
        manager = ConditionManager()
        manager.set_condition(ConditionType.CONFIG_VALID, True, "Valid", "OK")
        manager.set_condition(ConditionType.RESOURCES_CREATED, True, "Created", "Done")
        manager.set_condition(ConditionType.WORKERS_READY, False, "Starting", "1/5")

        conditions = manager.to_list()
        assert len(conditions) == 3
        types = [c["type"] for c in conditions]
        assert "ConfigValid" in types
        assert "ResourcesCreated" in types
        assert "WorkersReady" in types

    def test_update_existing_condition(self) -> None:
        """Test updating an existing condition."""
        manager = ConditionManager()
        manager.set_condition(
            ConditionType.WORKERS_READY, False, "Starting", "0/5 workers"
        )

        # Update to True - timestamp should change
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T10:00:05Z",
        ):
            manager.set_condition(
                ConditionType.WORKERS_READY, True, "WorkersReady", "5/5 workers"
            )

        condition = manager.get_condition(ConditionType.WORKERS_READY)
        assert condition["status"] == "True"
        assert condition["reason"] == "WorkersReady"
        # Timestamp should have changed because status changed
        assert condition["lastTransitionTime"] == "2026-01-15T10:00:05Z"

    def test_update_same_status_preserves_timestamp(self) -> None:
        """Test updating with same status preserves original timestamp."""
        manager = ConditionManager()
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T10:00:00Z",
        ):
            manager.set_condition(
                ConditionType.WORKERS_READY, False, "Starting", "1/5 workers"
            )

        original_timestamp = manager.get_condition(ConditionType.WORKERS_READY)[
            "lastTransitionTime"
        ]

        # Update with same status (False) - timestamp should be preserved
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T10:00:10Z",
        ):
            manager.set_condition(
                ConditionType.WORKERS_READY, False, "Starting", "2/5 workers"
            )

        condition = manager.get_condition(ConditionType.WORKERS_READY)
        assert condition["message"] == "2/5 workers"  # Message updated
        assert (
            condition["lastTransitionTime"] == original_timestamp
        )  # Timestamp preserved

    def test_get_condition_exists(self) -> None:
        """Test getting an existing condition."""
        manager = ConditionManager()
        manager.set_condition(ConditionType.CONFIG_VALID, True, "Valid", "OK")
        condition = manager.get_condition(ConditionType.CONFIG_VALID)
        assert condition is not None
        assert condition["type"] == "ConfigValid"

    def test_get_condition_not_exists(self) -> None:
        """Test getting a non-existent condition returns None."""
        manager = ConditionManager()
        condition = manager.get_condition(ConditionType.WORKERS_READY)
        assert condition is None

    def test_is_condition_true_when_true(self) -> None:
        """Test is_condition_true returns True for True conditions."""
        manager = ConditionManager()
        manager.set_condition(ConditionType.CONFIG_VALID, True, "Valid", "OK")
        assert manager.is_condition_true(ConditionType.CONFIG_VALID) is True

    def test_is_condition_true_when_false(self) -> None:
        """Test is_condition_true returns False for False conditions."""
        manager = ConditionManager()
        manager.set_condition(ConditionType.WORKERS_READY, False, "Starting", "0/5")
        assert manager.is_condition_true(ConditionType.WORKERS_READY) is False

    def test_is_condition_true_when_not_set(self) -> None:
        """Test is_condition_true returns False for unset conditions."""
        manager = ConditionManager()
        assert manager.is_condition_true(ConditionType.BENCHMARK_RUNNING) is False

    def test_to_list_returns_list(self) -> None:
        """Test to_list returns a proper list."""
        manager = ConditionManager()
        manager.set_condition(ConditionType.CONFIG_VALID, True, "Valid", "OK")
        result = manager.to_list()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_from_status_empty(self) -> None:
        """Test from_status with empty/None input."""
        manager = ConditionManager.from_status(None)
        assert manager.to_list() == []

        manager = ConditionManager.from_status({})
        assert manager.to_list() == []

    def test_from_status_with_conditions(
        self, sample_conditions_list: list[dict[str, Any]]
    ) -> None:
        """Test from_status reconstructs conditions from status dict."""
        status = {"conditions": sample_conditions_list}
        manager = ConditionManager.from_status(status)

        assert manager.is_condition_true(ConditionType.CONFIG_VALID) is True
        assert manager.is_condition_true(ConditionType.RESOURCES_CREATED) is True
        assert manager.is_condition_true(ConditionType.WORKERS_READY) is False

        # Check preserved values
        config_condition = manager.get_condition(ConditionType.CONFIG_VALID)
        assert config_condition["reason"] == "ConfigParsed"
        assert config_condition["lastTransitionTime"] == "2026-01-15T10:00:00Z"

    def test_from_status_preserves_timestamps(
        self, sample_conditions_list: list[dict[str, Any]]
    ) -> None:
        """Test from_status preserves original timestamps."""
        status = {"conditions": sample_conditions_list}
        manager = ConditionManager.from_status(status)

        # Update condition without changing status
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T11:00:00Z",
        ):
            manager.set_condition(
                ConditionType.WORKERS_READY, False, "StillStarting", "3/5 workers"
            )

        condition = manager.get_condition(ConditionType.WORKERS_READY)
        # Original timestamp should be preserved since status didn't change
        assert condition["lastTransitionTime"] == "2026-01-15T10:00:10Z"


class TestConditionManagerWorkflow:
    """Integration tests for typical ConditionManager workflows."""

    def test_job_creation_workflow(self) -> None:
        """Test typical workflow when creating a job."""
        manager = ConditionManager()

        # Step 1: Config validated
        manager.set_condition(
            ConditionType.CONFIG_VALID,
            True,
            reason="ConfigParsed",
            message="AIPerfJob spec validated successfully",
        )
        assert manager.is_condition_true(ConditionType.CONFIG_VALID)

        # Step 2: Resources created
        manager.set_condition(
            ConditionType.RESOURCES_CREATED,
            True,
            reason="ResourcesCreated",
            message="ConfigMap and JobSet created",
        )
        assert manager.is_condition_true(ConditionType.RESOURCES_CREATED)

        # Final state
        conditions = manager.to_list()
        assert len(conditions) == 2

    def test_job_monitoring_workflow(self) -> None:
        """Test typical workflow when monitoring a job."""
        # Start from existing conditions
        initial_conditions = [
            {
                "type": "ConfigValid",
                "status": "True",
                "reason": "Valid",
                "message": "OK",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
            {
                "type": "WorkersReady",
                "status": "False",
                "reason": "Starting",
                "message": "0/5",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
        ]
        manager = ConditionManager.from_status({"conditions": initial_conditions})

        # Update workers progress
        manager.set_condition(
            ConditionType.WORKERS_READY, False, "Starting", "3/5 workers"
        )

        # Workers all ready
        manager.set_condition(
            ConditionType.WORKERS_READY, True, "WorkersReady", "5/5 workers"
        )
        assert manager.is_condition_true(ConditionType.WORKERS_READY)

        # Benchmark starts running
        manager.set_condition(
            ConditionType.BENCHMARK_RUNNING, True, "Running", "Benchmark in progress"
        )

        conditions = manager.to_list()
        assert len(conditions) == 3

    def test_job_failure_workflow(self) -> None:
        """Test workflow when a job fails."""
        manager = ConditionManager()

        # Config valid but then fails
        manager.set_condition(
            ConditionType.CONFIG_VALID,
            False,
            reason="InvalidConfig",
            message="Missing required field: endpoint.model_names",
        )

        assert not manager.is_condition_true(ConditionType.CONFIG_VALID)
        condition = manager.get_condition(ConditionType.CONFIG_VALID)
        assert condition["reason"] == "InvalidConfig"


class TestConditionManagerConvenienceMethods:
    """Tests for set_true and set_false convenience methods."""

    def test_set_true_sets_status_to_true(self) -> None:
        """Test set_true sets condition status to True."""
        manager = ConditionManager()
        manager.set_true(ConditionType.CONFIG_VALID, "Valid", "Config is valid")

        condition = manager.get_condition(ConditionType.CONFIG_VALID)
        assert condition is not None
        assert condition["status"] == "True"
        assert condition["reason"] == "Valid"
        assert condition["message"] == "Config is valid"

    def test_set_true_with_empty_message(self) -> None:
        """Test set_true works with empty message."""
        manager = ConditionManager()
        manager.set_true(ConditionType.RESOURCES_CREATED, "Created")

        condition = manager.get_condition(ConditionType.RESOURCES_CREATED)
        assert condition is not None
        assert condition["status"] == "True"
        assert condition["message"] == ""

    def test_set_false_sets_status_to_false(self) -> None:
        """Test set_false sets condition status to False."""
        manager = ConditionManager()
        manager.set_false(ConditionType.WORKERS_READY, "Starting", "0/5 workers ready")

        condition = manager.get_condition(ConditionType.WORKERS_READY)
        assert condition is not None
        assert condition["status"] == "False"
        assert condition["reason"] == "Starting"
        assert condition["message"] == "0/5 workers ready"

    def test_set_false_with_empty_message(self) -> None:
        """Test set_false works with empty message."""
        manager = ConditionManager()
        manager.set_false(ConditionType.ENDPOINT_REACHABLE, "Unreachable")

        condition = manager.get_condition(ConditionType.ENDPOINT_REACHABLE)
        assert condition is not None
        assert condition["status"] == "False"
        assert condition["message"] == ""


class TestConditionManagerApplyToPatch:
    """Tests for apply_to_patch method."""

    def test_apply_to_patch_adds_conditions_to_status(self) -> None:
        """Test apply_to_patch adds conditions to patch status."""
        manager = ConditionManager()
        manager.set_true(ConditionType.CONFIG_VALID, "Valid", "OK")
        manager.set_false(ConditionType.WORKERS_READY, "Starting", "0/5")

        mock_patch = MagicMock()
        mock_patch.status = {}

        manager.apply_to_patch(mock_patch)

        assert "conditions" in mock_patch.status
        assert len(mock_patch.status["conditions"]) == 2

    def test_apply_to_patch_empty_conditions(self) -> None:
        """Test apply_to_patch with no conditions."""
        manager = ConditionManager()

        mock_patch = MagicMock()
        mock_patch.status = {}

        manager.apply_to_patch(mock_patch)

        assert mock_patch.status["conditions"] == []


class TestConditionManagerFromStatus:
    """Tests for from_status class method edge cases."""

    def test_from_status_with_full_status_dict(self) -> None:
        """Test from_status with full status dict containing conditions."""
        full_status = {
            "phase": "Running",
            "workers": {"ready": 3, "total": 5},
            "conditions": [
                {
                    "type": "ConfigValid",
                    "status": "True",
                    "reason": "Valid",
                    "message": "OK",
                    "lastTransitionTime": "2026-01-15T10:00:00Z",
                },
            ],
        }

        manager = ConditionManager.from_status(full_status)

        assert manager.is_condition_true(ConditionType.CONFIG_VALID)
        condition = manager.get_condition(ConditionType.CONFIG_VALID)
        assert condition is not None
        assert condition["reason"] == "Valid"

    def test_from_status_skips_invalid_condition_type(self) -> None:
        """Test from_status skips conditions with invalid type."""
        conditions_with_invalid = [
            {
                "type": "ConfigValid",
                "status": "True",
                "reason": "Valid",
                "message": "OK",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
            {
                "type": "InvalidConditionType",
                "status": "True",
                "reason": "Unknown",
                "message": "This should be skipped",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
        ]

        manager = ConditionManager.from_status({"conditions": conditions_with_invalid})

        # Only valid condition should be loaded
        conditions = manager.to_list()
        assert len(conditions) == 1
        assert conditions[0]["type"] == "ConfigValid"

    def test_from_status_skips_condition_missing_type_key(self) -> None:
        """Test from_status skips conditions missing type key."""
        conditions_missing_key = [
            {
                "type": "ConfigValid",
                "status": "True",
                "reason": "Valid",
                "message": "OK",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
            {
                # Missing "type" key
                "status": "True",
                "reason": "NoType",
                "message": "No type field",
                "lastTransitionTime": "2026-01-15T10:00:00Z",
            },
        ]

        manager = ConditionManager.from_status({"conditions": conditions_missing_key})

        conditions = manager.to_list()
        assert len(conditions) == 1
        assert conditions[0]["type"] == "ConfigValid"

    def test_from_status_with_empty_conditions_key(self) -> None:
        """Test from_status with status dict that has empty conditions list."""
        status_empty_conditions = {
            "phase": "Pending",
            "conditions": [],
        }

        manager = ConditionManager.from_status(status_empty_conditions)
        assert manager.to_list() == []

    def test_from_status_with_none_conditions(self) -> None:
        """Test from_status with status dict that has None conditions."""
        status_none_conditions: dict[str, Any] = {
            "phase": "Pending",
            "conditions": None,
        }

        manager = ConditionManager.from_status(status_none_conditions)
        assert manager.to_list() == []


class TestStatusBuilder:
    """Tests for StatusBuilder class."""

    def test_init_with_no_existing_status(self) -> None:
        """Test StatusBuilder initialization without existing status."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)

        assert builder.conditions.to_list() == []

    def test_init_with_existing_status(self) -> None:
        """Test StatusBuilder initialization with existing status."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        existing_status = {
            "phase": "Running",
            "conditions": [
                {
                    "type": "ConfigValid",
                    "status": "True",
                    "reason": "Valid",
                    "message": "OK",
                    "lastTransitionTime": "2026-01-15T10:00:00Z",
                },
            ],
        }

        builder = StatusBuilder(mock_patch, existing_status)

        assert builder.conditions.is_condition_true(ConditionType.CONFIG_VALID)

    def test_conditions_property(self) -> None:
        """Test conditions property returns ConditionManager."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)

        assert isinstance(builder.conditions, ConditionManager)

    def test_set_phase(self) -> None:
        """Test set_phase updates patch status."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        result = builder.set_phase(Phase.RUNNING)

        assert mock_patch.status["phase"] == "Running"
        assert result is builder  # Returns self for chaining

    def test_set_error(self) -> None:
        """Test set_error updates patch status."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        result = builder.set_error("Connection failed")

        assert mock_patch.status["error"] == "Connection failed"
        assert result is builder

    def test_set_completion_time(self) -> None:
        """Test set_completion_time sets timestamp."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T12:00:00Z",
        ):
            result = builder.set_completion_time()

        assert mock_patch.status["completionTime"] == "2026-01-15T12:00:00Z"
        assert result is builder

    def test_set_workers(self) -> None:
        """Test set_workers updates worker counts."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        result = builder.set_workers(ready=3, total=5)

        assert mock_patch.status["workers"] == {"ready": 3, "total": 5}
        assert result is builder

    def test_set_worker_aggregate_status_writes_camel_case_worker_keys(self) -> None:
        """Test set_worker_aggregate_status writes CR-facing camelCase worker keys."""
        mock_patch = MagicMock()
        mock_patch.status = {}
        builder = StatusBuilder(mock_patch)

        result = builder.set_worker_aggregate_status(
            {
                "ready": 4,
                "total": 8,
                "dispatchable": 3,
                "router_connected": 6,
                "ready_record_processors": 2,
                "declared_record_processors": 4,
                "ready_pods": 2,
                "total_pods": 4,
                "degraded_pods": 1,
            }
        )

        assert mock_patch.status["workers"] == {
            "ready": 4,
            "total": 8,
            "dispatchable": 3,
            "routerConnected": 6,
            "readyRecordProcessors": 2,
            "declaredRecordProcessors": 4,
            "readyPods": 2,
            "totalPods": 4,
            "degradedPods": 1,
        }
        assert result is builder

    def test_set_results(self) -> None:
        """Test set_results updates results dict."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        results = {
            "throughput": 100.5,
            "latency_p50": 25.0,
            "latency_p99": 150.0,
        }

        builder = StatusBuilder(mock_patch)
        result = builder.set_results(results)

        assert mock_patch.status["results"] == results
        assert result is builder

    def test_set_results_path(self) -> None:
        """Test set_results_path updates results path."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        result = builder.set_results_path("/data/results/job-123")

        assert mock_patch.status["resultsPath"] == "/data/results/job-123"
        assert result is builder

    def test_set_summary(self) -> None:
        """Test set_summary updates summary dict."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        summary = {
            "total_requests": 1000,
            "successful_requests": 995,
            "failed_requests": 5,
        }

        builder = StatusBuilder(mock_patch)
        result = builder.set_summary(summary)

        assert mock_patch.status["summary"] == summary
        assert result is builder

    def test_finalize_adds_conditions(self) -> None:
        """Test finalize adds conditions to patch."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        builder.conditions.set_true(ConditionType.CONFIG_VALID, "Valid", "OK")
        builder.conditions.set_true(
            ConditionType.RESOURCES_CREATED, "Created", "Resources ready"
        )

        builder.finalize()

        assert "conditions" in mock_patch.status
        assert len(mock_patch.status["conditions"]) == 2

    def test_finalize_no_conditions(self) -> None:
        """Test finalize with no conditions does not add conditions key."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        builder.finalize()

        # Conditions list is empty, so should not add conditions key
        assert "conditions" not in mock_patch.status

    def test_fluent_interface_chaining(self) -> None:
        """Test fluent interface allows method chaining."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)

        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T12:00:00Z",
        ):
            (
                builder.set_phase(Phase.COMPLETED)
                .set_workers(5, 5)
                .set_results({"throughput": 100.0})
                .set_results_path("/results/job-1")
                .set_summary({"total": 1000})
                .set_completion_time()
            )

        assert mock_patch.status["phase"] == "Completed"
        assert mock_patch.status["workers"] == {"ready": 5, "total": 5}
        assert mock_patch.status["results"] == {"throughput": 100.0}
        assert mock_patch.status["resultsPath"] == "/results/job-1"
        assert mock_patch.status["summary"] == {"total": 1000}
        assert mock_patch.status["completionTime"] == "2026-01-15T12:00:00Z"

    def test_full_workflow_with_builder(self) -> None:
        """Test complete workflow using StatusBuilder."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        # Simulate existing status from a running job
        existing_status = {
            "phase": "Running",
            "conditions": [
                {
                    "type": "ConfigValid",
                    "status": "True",
                    "reason": "Valid",
                    "message": "Config OK",
                    "lastTransitionTime": "2026-01-15T10:00:00Z",
                },
                {
                    "type": "WorkersReady",
                    "status": "True",
                    "reason": "WorkersReady",
                    "message": "5/5 workers",
                    "lastTransitionTime": "2026-01-15T10:05:00Z",
                },
            ],
        }

        builder = StatusBuilder(mock_patch, existing_status)

        # Complete the job
        with patch(
            "aiperf.operator.status.format_timestamp",
            return_value="2026-01-15T11:00:00Z",
        ):
            builder.set_phase(Phase.COMPLETED)
            builder.set_results({"throughput": 150.0, "latency_p99": 100.0})
            builder.set_results_path("/data/results/job-123")
            builder.conditions.set_true(
                ConditionType.RESULTS_AVAILABLE, "ResultsReady", "Results exported"
            )
            builder.conditions.set_false(
                ConditionType.BENCHMARK_RUNNING, "Completed", "Benchmark finished"
            )
            builder.set_completion_time()
            builder.finalize()

        # Verify final state
        assert mock_patch.status["phase"] == "Completed"
        assert mock_patch.status["results"] == {
            "throughput": 150.0,
            "latency_p99": 100.0,
        }
        assert mock_patch.status["resultsPath"] == "/data/results/job-123"
        assert mock_patch.status["completionTime"] == "2026-01-15T11:00:00Z"
        assert len(mock_patch.status["conditions"]) == 4


class TestStatusBuilderErrorWorkflow:
    """Tests for StatusBuilder error handling workflow."""

    def test_set_error_with_phase_failed(self) -> None:
        """Test setting error message with failed phase."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        builder.set_phase(Phase.FAILED)
        builder.set_error("Connection refused to endpoint")
        builder.conditions.set_false(
            ConditionType.ENDPOINT_REACHABLE, "Unreachable", "Cannot connect"
        )
        builder.finalize()

        assert mock_patch.status["phase"] == "Failed"
        assert mock_patch.status["error"] == "Connection refused to endpoint"
        assert len(mock_patch.status["conditions"]) == 1

    def test_cancelled_phase(self) -> None:
        """Test setting cancelled phase."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        builder.set_phase(Phase.CANCELLED)
        builder.finalize()

        assert mock_patch.status["phase"] == "Cancelled"


# =============================================================================
# Tests for PREFLIGHT_PASSED condition type
# =============================================================================


class TestPreflightPassedConditionType:
    """Tests for the PREFLIGHT_PASSED member of ConditionType."""

    def test_preflight_passed_enum_value(self) -> None:
        """Verify ConditionType.PREFLIGHT_PASSED has expected string value."""
        assert ConditionType.PREFLIGHT_PASSED == "PreflightPassed"
        assert ConditionType.PREFLIGHT_PASSED.value == "PreflightPassed"

    def test_set_true_for_preflight_passed(self) -> None:
        """Verify set_true produces correct condition dict for PREFLIGHT_PASSED."""
        manager = ConditionManager()
        manager.set_true(
            ConditionType.PREFLIGHT_PASSED, "PreflightPassed", "All checks passed"
        )

        condition = manager.get_condition(ConditionType.PREFLIGHT_PASSED)
        assert condition is not None
        assert condition["type"] == "PreflightPassed"
        assert condition["status"] == "True"
        assert condition["reason"] == "PreflightPassed"
        assert condition["message"] == "All checks passed"
        assert "lastTransitionTime" in condition

    def test_set_false_for_preflight_passed(self) -> None:
        """Verify set_false produces correct condition dict for PREFLIGHT_PASSED."""
        manager = ConditionManager()
        manager.set_false(
            ConditionType.PREFLIGHT_PASSED,
            "PreflightFailed",
            "Endpoint health check failed",
        )

        condition = manager.get_condition(ConditionType.PREFLIGHT_PASSED)
        assert condition is not None
        assert condition["type"] == "PreflightPassed"
        assert condition["status"] == "False"
        assert condition["reason"] == "PreflightFailed"
        assert condition["message"] == "Endpoint health check failed"

    def test_is_condition_true_for_preflight_passed(self) -> None:
        """Verify is_condition_true returns True after set_true for PREFLIGHT_PASSED."""
        manager = ConditionManager()
        manager.set_true(
            ConditionType.PREFLIGHT_PASSED, "PreflightPassed", "All checks passed"
        )
        assert manager.is_condition_true(ConditionType.PREFLIGHT_PASSED) is True

    def test_preflight_passed_ordering_in_enum(self) -> None:
        """Verify PREFLIGHT_PASSED sits between ENDPOINT_REACHABLE and RESOURCES_CREATED."""
        members = list(ConditionType)
        ep_idx = members.index(ConditionType.ENDPOINT_REACHABLE)
        pf_idx = members.index(ConditionType.PREFLIGHT_PASSED)
        rc_idx = members.index(ConditionType.RESOURCES_CREATED)
        assert ep_idx < pf_idx < rc_idx

    def test_from_status_restores_preflight_passed(self) -> None:
        """Verify from_status correctly parses a PreflightPassed condition."""
        status: dict[str, Any] = {
            "conditions": [
                {
                    "type": "PreflightPassed",
                    "status": "True",
                    "reason": "PreflightPassed",
                    "message": "All checks passed",
                    "lastTransitionTime": "2026-03-15T08:00:00Z",
                },
            ],
        }
        manager = ConditionManager.from_status(status)
        assert manager.is_condition_true(ConditionType.PREFLIGHT_PASSED) is True

        condition = manager.get_condition(ConditionType.PREFLIGHT_PASSED)
        assert condition is not None
        assert condition["reason"] == "PreflightPassed"
        assert condition["lastTransitionTime"] == "2026-03-15T08:00:00Z"

    def test_status_builder_with_preflight_passed(self) -> None:
        """Verify StatusBuilder includes PREFLIGHT_PASSED condition in finalized patch."""
        mock_patch = MagicMock()
        mock_patch.status = {}

        builder = StatusBuilder(mock_patch)
        builder.conditions.set_true(
            ConditionType.PREFLIGHT_PASSED,
            "PreflightPassed",
            "All pre-flight checks passed",
        )
        builder.finalize()

        assert "conditions" in mock_patch.status
        conditions = mock_patch.status["conditions"]
        assert len(conditions) == 1
        assert conditions[0]["type"] == "PreflightPassed"
        assert conditions[0]["status"] == "True"
        assert conditions[0]["reason"] == "PreflightPassed"
        assert conditions[0]["message"] == "All pre-flight checks passed"
