# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for orchestrator data models."""

from pathlib import Path

from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.orchestrator.models import RunConfig, RunResult


class TestRunConfig:
    """Tests for RunConfig data model."""

    def test_create_run_config_valid_inputs_sets_fields(self):
        """Test creating a RunConfig with valid inputs sets all fields correctly."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        run_config = RunConfig(
            config=config, label="run_0001", metadata={"run_index": 0, "seed": 42}
        )

        assert run_config.config == config
        assert run_config.label == "run_0001"
        assert run_config.metadata == {"run_index": 0, "seed": 42}

    def test_create_run_config_empty_metadata_returns_empty_dict(self):
        """Test creating a RunConfig with empty metadata returns empty dict."""
        config = UserConfig(endpoint=EndpointConfig(model_names=["test-model"]))
        run_config = RunConfig(config=config, label="run_0001", metadata={})

        assert run_config.metadata == {}


class TestRunResult:
    """Tests for RunResult data model."""

    def test_create_run_result_successful_sets_fields_and_no_error(self):
        """Test creating a successful RunResult sets fields correctly and has no error."""
        result = RunResult(
            label="run_0001",
            success=True,
            summary_metrics={
                "ttft": JsonMetricResult(unit="ms", avg=100.0),
                "tpot": JsonMetricResult(unit="ms", avg=10.0),
            },
            artifacts_path=Path("/tmp/run_0001"),
        )

        assert result.label == "run_0001"
        assert result.success is True
        assert result.summary_metrics["ttft"].avg == 100.0
        assert result.summary_metrics["ttft"].unit == "ms"
        assert result.summary_metrics["tpot"].avg == 10.0
        assert result.summary_metrics["tpot"].unit == "ms"
        assert result.artifacts_path == Path("/tmp/run_0001")
        assert result.error is None

    def test_create_run_result_failed_sets_error_and_empty_summary_metrics(self):
        """Test creating a failed RunResult sets error and defaults to empty summary metrics."""
        result = RunResult(
            label="run_0002",
            success=False,
            error="Connection timeout",
            artifacts_path=Path("/tmp/run_0002"),
        )

        assert result.label == "run_0002"
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.summary_metrics == {}  # Defaults to empty dict
        assert result.artifacts_path == Path("/tmp/run_0002")

    def test_run_result_with_none_metrics(self):
        """Test RunResult with None summary_metrics."""
        result = RunResult(
            label="run_0003",
            success=True,
            summary_metrics={},  # Use empty dict instead of None
            artifacts_path=Path("/tmp/run_0003"),
        )

        assert result.summary_metrics == {}
