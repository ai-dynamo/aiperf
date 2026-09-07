# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial finite-numeric tests for Kubernetes/operator models.

Focuses on:
- CRD spec fields whose numeric values cross the Kubernetes API boundary.
- Sweep numeric knobs and parameter values before variation expansion.
- Status/progress floats before they are serialized into AIPerfJob status.
- Summary metric projection that mirrors controller metrics into CR status.

Out of scope:
- Global finite-field discovery: tests/unit/property/test_finite_invariants.py.
- CRD schema/CEL rendering: tests/unit/operator/test_aiperfsweep_crd_generation.py.
- Runtime handler state machines: tests/unit/operator/test_monitor_state_machine_edges.py.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.crd_models import (
    AIPerfJobSpec,
    AIPerfSweepSpec,
    MetricsSummary,
    PhaseProgress,
)

# =============================================================================
# Helpers
# =============================================================================

_VALID_BENCHMARK: dict[str, object] = {
    "models": ["meta-llama/Llama-3-8B"],
    "endpoint": {"url": "http://vllm-router.aiperf-system:8000/v1/chat/completions"},
    "datasets": [
        {
            "name": "synthetic-main",
            "type": "synthetic",
            "entries": 4,
            "prompts": {"isl": 128, "osl": 32},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 4,
            "concurrency": 2,
        }
    ],
}

_VALID_GRID_SWEEP: dict[str, object] = {
    "type": "grid",
    "parameters": {"phases.profiling.concurrency": [1, 2]},
}

_VALID_ADAPTIVE_SWEEP: dict[str, object] = {
    "type": "adaptive_search",
    "searchSpace": [
        {
            "path": "phases.profiling.concurrency",
            "kind": "int",
            "lo": 1,
            "hi": 4,
        }
    ],
    "objectives": [
        {
            "metric": "output_token_throughput",
            "direction": "maximize",
        }
    ],
    "maxIterations": 4,
    "nInitialPoints": 1,
}


def _benchmark_with(**overrides: object) -> dict[str, object]:
    """Return a real benchmark baseline with adversarial overrides."""
    baseline = deepcopy(_VALID_BENCHMARK)
    baseline.update(overrides)
    return baseline


def _job_spec(**overrides: object) -> dict[str, object]:
    """Build a minimal AIPerfJobSpec dict with optional top-level overrides."""
    baseline: dict[str, object] = {
        "image": "nvcr.io/nvidia/aiperf:finite-adversarial",
        "benchmark": _benchmark_with(),
    }
    baseline.update(overrides)
    return baseline


def _sweep_spec(sweep: dict[str, object], **overrides: object) -> dict[str, object]:
    """Build a minimal AIPerfSweepSpec dict with an adversarial sweep block."""
    baseline: dict[str, object] = {
        "image": "nvcr.io/nvidia/aiperf:finite-adversarial",
        "benchmark": _benchmark_with(),
        "sweep": deepcopy(sweep),
    }
    baseline.update(overrides)
    return baseline


def _progress_with(**overrides: object) -> dict[str, object]:
    """Build a status.phase progress payload as the operator receives it."""
    baseline: dict[str, object] = {
        "requests_completed": 1,
        "requests_sent": 1,
        "requests_total": 4,
        "requests_cancelled": 0,
        "requests_errors": 0,
        "requests_in_flight": 0,
        "requests_per_second": 12.5,
        "requests_progress_percent": 25.0,
        "sessions_sent": 1,
        "sessions_completed": 1,
        "sessions_cancelled": 0,
        "sessions_in_flight": 0,
        "records_success": 1,
        "records_error": 0,
        "records_per_second": 12.0,
        "records_progress_percent": 25.0,
        "sending_complete": False,
        "is_requests_complete": False,
        "is_records_complete": False,
        "timeout_triggered": False,
        "was_cancelled": False,
        "requests_eta_seconds": 3,
        "records_eta_seconds": 3,
        "expected_duration_seconds": 2.0,
        "elapsed_time_seconds": 0.5,
    }
    baseline.update(overrides)
    return baseline


# =============================================================================
# CRD spec numeric boundary fields
# =============================================================================


class TestWorkloadSpecFiniteBoundaries:
    """Deployment fields reject non-finite numbers before CRD/status emission."""

    @pytest.mark.parametrize(
        "field_name,bad_value,error_match",
        [
            param("timeoutSeconds", float("nan"), r"(?i)timeout|finite|nan", id="timeout-nan"),
            param("timeoutSeconds", float("inf"), r"(?i)timeout|finite|inf", id="timeout-positive-inf"),
            param("timeoutSeconds", float("-inf"), r"(?i)timeout|finite|inf", id="timeout-negative-inf"),
            param("ttlSecondsAfterFinished", float("inf"), r"(?i)ttl|finite|inf", id="ttl-positive-inf"),
        ],
    )  # fmt: skip
    def test_aiperfjob_spec_non_finite_deployment_number_rejected(
        self, field_name: str, bad_value: float, error_match: str
    ) -> None:
        with pytest.raises(ValidationError, match=error_match):
            AIPerfJobSpec.model_validate(_job_spec(**{field_name: bad_value}))


# =============================================================================
# Sweep numeric values and knobs
# =============================================================================


class TestSweepFiniteBoundaries:
    """Sweep parameters reject non-finite values before expansion or execution."""

    @pytest.mark.parametrize(
        "sweep_type,bad_value",
        [
            ("grid", float("nan")),
            param("grid", float("inf"), id="grid-positive-inf"),
            param("zip", float("inf"), id="zip-positive-inf"),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_grid_style_variable_non_finite_rejected_with_path(
        self, sweep_type: str, bad_value: float
    ) -> None:
        sweep = {
            "type": sweep_type,
            "parameters": {"phases.profiling.concurrency": [1, bad_value]},
        }

        with pytest.raises(
            ValidationError,
            match=r"(?i)benchmark\.phases\.profiling\.concurrency.*finite|nan|inf",
        ):
            AIPerfSweepSpec.model_validate(_sweep_spec(sweep))

    @pytest.mark.parametrize(
        "sweep,error_match",
        [
            param(
                {**_VALID_GRID_SWEEP, "cooldownSeconds": float("inf")},
                r"(?i)cooldown|finite|inf",
                id="grid-cooldown-positive-inf",
            ),
            param(
                {**_VALID_ADAPTIVE_SWEEP, "cooldownSeconds": float("inf")},
                r"(?i)cooldown|finite|inf",
                id="adaptive-cooldown-positive-inf",
            ),
            param(
                {**_VALID_ADAPTIVE_SWEEP, "plateauThreshold": float("inf")},
                r"(?i)plateau|finite|inf",
                id="adaptive-plateau-threshold-positive-inf",
            ),
            param(
                {**_VALID_ADAPTIVE_SWEEP, "slaWarmupSeconds": float("inf")},
                r"(?i)slaWarmupSeconds|warmup|finite|inf",
                id="adaptive-sla-warmup-positive-inf",
            ),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_non_finite_sweep_knob_rejected(
        self, sweep: dict[str, object], error_match: str
    ) -> None:
        with pytest.raises(ValidationError, match=error_match):
            AIPerfSweepSpec.model_validate(_sweep_spec(sweep))

    @pytest.mark.parametrize(
        "sweep,error_match",
        [
            param(
                {
                    **_VALID_ADAPTIVE_SWEEP,
                    "objectives": [
                        {
                            "metric": "output_token_throughput",
                            "direction": "maximize",
                            "threshold": float("inf"),
                        }
                    ],
                },
                r"(?i)threshold|finite|inf",
                id="adaptive-objective-threshold-positive-inf",
            ),
            param(
                {
                    **_VALID_ADAPTIVE_SWEEP,
                    "outcomeConstraints": [
                        {
                            "metric": "time_to_first_token",
                            "op": "<=",
                            "bound": float("inf"),
                        }
                    ],
                },
                r"(?i)bound|finite|inf",
                id="adaptive-outcome-bound-positive-inf",
            ),
            param(
                {
                    **_VALID_ADAPTIVE_SWEEP,
                    "searchSpace": [
                        {
                            "path": "phases.profiling.concurrency",
                            "kind": "int",
                            "lo": 1,
                            "hi": float("inf"),
                        }
                    ],
                },
                r"(?i)lo/hi|finite|inf",
                id="adaptive-search-space-hi-positive-inf",
            ),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_non_finite_adaptive_numeric_value_rejected(
        self, sweep: dict[str, object], error_match: str
    ) -> None:
        with pytest.raises(ValidationError, match=error_match):
            AIPerfSweepSpec.model_validate(_sweep_spec(sweep))


# =============================================================================
# Status/progress numeric boundaries
# =============================================================================


class TestStatusFiniteBoundaries:
    """Status models do not serialize non-finite progress or summary metrics."""

    @pytest.mark.parametrize(
        "field_name,bad_value,error_match",
        [
            param("requests_per_second", float("inf"), r"(?i)requests_per_second|finite|inf", id="requests-per-second-inf"),
            param("records_per_second", float("nan"), r"(?i)records_per_second|finite|nan", id="records-per-second-nan"),
            param("requests_progress_percent", float("inf"), r"(?i)requests_progress_percent|finite|inf", id="requests-progress-percent-inf"),
            param("expected_duration_seconds", float("inf"), r"(?i)expected_duration_seconds|finite|inf", id="expected-duration-inf"),
            param("elapsed_time_seconds", float("nan"), r"(?i)elapsed_time_seconds|finite|nan", id="elapsed-time-nan"),
        ],
    )  # fmt: skip
    def test_phase_progress_non_finite_status_float_rejected(
        self, field_name: str, bad_value: float, error_match: str
    ) -> None:
        with pytest.raises(ValidationError, match=error_match):
            PhaseProgress.model_validate(_progress_with(**{field_name: bad_value}))

    def test_metrics_summary_from_metrics_non_finite_metric_not_written_to_status(
        self,
    ) -> None:
        metrics = {
            "metrics": {
                "output_token_throughput": {
                    "avg": float("inf"),
                    "unit": "tokens/sec",
                },
                "request_count": {"avg": 100},
                "error_request_count": {"avg": float("nan")},
            }
        }

        result = MetricsSummary.from_metrics(metrics).to_status_dict()

        assert "output_token_throughput" not in result
        assert "error_rate" not in result
        assert result["total_requests"] == 100
