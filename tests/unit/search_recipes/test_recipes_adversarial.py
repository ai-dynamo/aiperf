# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial regression tests for search-recipe post-process handlers.

Each test reproduces a confirmed silent-wrong-output / UX bug in the
post-process pipeline that prior unit suites missed by never feeding
non-finite metric values, negative baselines, or non-positive concurrency
bounds. See the bug report at /tmp/adversarial-recipes.md for the original
black-box probes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.search_recipes.builtins import _logspace_int_steps
from aiperf.search_recipes.post_process import (
    DegradationKneeDetect,
    ParetoSweepExport,
    SLABreachKnee,
    TTFTCurveFit,
)


def _pareto_params() -> dict[str, Any]:
    return {
        "x_metric": "request_latency",
        "x_stat": "p95",
        "y_metric": "output_token_throughput",
        "y_stat": "avg",
        "isl_key": "isl",
        "osl_key": "osl",
        "concurrency_key": "concurrency",
    }


def _pareto_row(
    isl: int, osl: int, conc: int, *, latency_p95: float, throughput: float
) -> dict[str, Any]:
    return {
        "parameters": {"isl": isl, "osl": osl, "concurrency": conc},
        "metrics": {
            "request_latency_p95": {"mean": latency_p95},
            "output_token_throughput_avg": {"mean": throughput},
        },
    }


def test_pareto_excludes_nan_cells() -> None:
    """A NaN-axis cell must NEVER be flagged Pareto-optimal.

    Pre-fix: NaN comparisons return False, so a NaN-x cell was never dominated
    AND never dominated -- it floated to the frontier and demoted real winners.
    Post-fix: non-finite cells are quarantined with pareto_optimal=False, a
    warning is emitted, and the dominance pass runs only over finite cells.
    """
    agg = {
        "per_combination_metrics": [
            # Failing cell: NaN latency (zero successful requests).
            _pareto_row(128, 128, 1, latency_p95=float("nan"), throughput=50.0),
            # Real winner the NaN cell would have demoted.
            _pareto_row(128, 128, 4, latency_p95=200.0, throughput=30.0),
            # An inf-y cell should also be excluded.
            _pareto_row(256, 256, 1, latency_p95=10.0, throughput=float("inf")),
        ]
    }
    with pytest.warns(UserWarning, match="non-finite"):
        out = ParetoSweepExport().process(agg, _pareto_params())

    by_id = {(c["isl"], c["osl"], c["concurrency"]): c for c in out["cells"]}
    # The NaN-x and inf-y cells must NOT be flagged optimal.
    assert by_id[(128, 128, 1)]["pareto_optimal"] is False
    assert by_id[(256, 256, 1)]["pareto_optimal"] is False
    # The lone finite cell is trivially optimal (no other finite cell dominates it).
    assert by_id[(128, 128, 4)]["pareto_optimal"] is True


def test_ttft_curve_fit_below_floor_on_nan() -> None:
    """NaN/inf trial rows must not produce NaN-coefficient "healthy" fits.

    Pre-fix: np.polyfit propagated NaN into coefficients and r^2; the
    `r_squared < r2_floor` check is False for NaN, so below_floor stayed
    False and downstream consumers saw a fake healthy curve.
    Post-fix: non-finite rows are dropped before polyfit; if too few finite
    points remain, below_floor=True with an error_reason.
    """
    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"datasets.main.prompts.isl": 256},
                "metrics": {"time_to_first_token_avg": {"mean": float("nan")}},
            },
            {
                "parameters": {"datasets.main.prompts.isl": 512},
                "metrics": {"time_to_first_token_avg": {"mean": 24.0}},
            },
            {
                "parameters": {"datasets.main.prompts.isl": 1024},
                "metrics": {"time_to_first_token_avg": {"mean": float("inf")}},
            },
        ]
    }
    out = TTFTCurveFit().process(
        agg,
        {
            "metric_tag": "time_to_first_token",
            "stat": "avg",
            "swept_param": "datasets.main.prompts.isl",
        },
    )
    # Only one finite point remains -> can't fit a line.
    assert out["below_floor"] is True
    assert out["coefficients"] == []
    assert out["r_squared"] == 0.0
    assert "error_reason" in out
    assert "non-finite" in out["error_reason"]
    # No NaN/inf must leak into the artifact.
    for c in out["coefficients"]:
        assert math.isfinite(c)
    assert math.isfinite(out["r_squared"])


def test_sla_breach_knee_serializes_nan_safely(tmp_path: Path) -> None:
    """NaN observed values must serialize as null AND match the documented contract.

    Pre-fix: orjson silently maps NaN -> null, indistinguishable from the
    "metric was missing" sentinel. Post-fix: the post-process glue scrubs
    non-finite floats to None before serialization, so the wire format keeps
    null reserved for "absent" only.
    """
    from aiperf.common.finite import scrub_non_finite

    agg = {
        "per_combination_metrics": [
            {
                "parameters": {"phases.profiling.concurrency": 256},
                "metrics": {"time_to_first_token_p95": {"mean": float("nan")}},
            },
            {
                "parameters": {"phases.profiling.concurrency": 384},
                "metrics": {"time_to_first_token_p95": {"mean": 250.0}},
            },
        ]
    }
    sla_filter = SLAFilter(
        metric_tag="time_to_first_token",
        stat="p95",
        op="lt",
        threshold=200.0,
    )
    out = SLABreachKnee().process(
        agg,
        {
            "sla_filters": [sla_filter],
            "swept_param": "phases.profiling.concurrency",
        },
    )
    # Pre-scrub: at least one breach has a NaN observed.
    nan_observed_seen = False
    for point in out["all_points"]:
        for breach in point["breaches"]:
            if isinstance(breach["observed"], float) and math.isnan(breach["observed"]):
                nan_observed_seen = True
    assert nan_observed_seen, "Test setup expected a NaN observed value"

    scrubbed = scrub_non_finite(out)
    encoded = orjson.dumps(scrubbed)
    decoded = orjson.loads(encoded)

    # All NaN-observed values must now be exactly None (matches "missing" contract).
    for point in decoded["all_points"]:
        for breach in point["breaches"]:
            obs = breach["observed"]
            assert obs is None or (isinstance(obs, (int, float)) and math.isfinite(obs))


def test_logspace_int_steps_rejects_nonpositive_concurrency() -> None:
    """A programmatic caller passing concurrency_min <= 0 must get a clear ValueError.

    Pre-fix: math.log(0) raised "math domain error" with no recipe context.
    Post-fix: an explicit guard fires first with a message naming the bound.
    """
    with pytest.raises(ValueError, match="lo must be > 0"):
        _logspace_int_steps(0, 100, 4)
    with pytest.raises(ValueError, match="lo must be > 0"):
        _logspace_int_steps(-1, 100, 4)


def test_degradation_knee_rejects_negative_baseline() -> None:
    """Negative baseline flips cutoff sign -> handler must reject it loudly.

    Also verifies that integer-valued concurrency axes round-trip as int
    rather than float in the artifact.
    """
    # First: negative baseline rejection.
    bad_agg = {
        "per_combination_metrics": [
            {
                "parameters": {"phases.profiling.concurrency": 1},
                "metrics": {"request_latency_p99": {"mean": -5.0}},
            },
            {
                "parameters": {"phases.profiling.concurrency": 100},
                "metrics": {"request_latency_p99": {"mean": -1.0}},
            },
        ]
    }
    with pytest.raises(ValueError, match="negative"):
        DegradationKneeDetect().process(
            bad_agg,
            {
                "threshold_pct": 0.20,
                "metric_tag": "request_latency",
                "stat": "p99",
                "swept_param": "phases.profiling.concurrency",
            },
        )

    # Second: integer concurrency axis -> int output (real config path uses
    # SLAFilter pydantic model so the validator is engaged elsewhere; here we
    # exercise the int-vs-float coercion contract directly).
    good_agg = {
        "per_combination_metrics": [
            {
                "parameters": {"phases.profiling.concurrency": 1},
                "metrics": {"request_latency_p99": {"mean": 10.0}},
            },
            {
                "parameters": {"phases.profiling.concurrency": 100},
                "metrics": {"request_latency_p99": {"mean": 25.0}},
            },
        ]
    }
    out = DegradationKneeDetect().process(
        good_agg,
        {
            "threshold_pct": 0.20,
            "metric_tag": "request_latency",
            "stat": "p99",
            "swept_param": "phases.profiling.concurrency",
        },
    )
    assert out["baseline_concurrency"] == 1
    assert isinstance(out["baseline_concurrency"], int)
    assert out["knee_concurrency"] == 100
    assert isinstance(out["knee_concurrency"], int)
    # Build a real Pydantic SLAFilter to verify the test pipeline isn't
    # silently no-op'ing on MagicMock-shaped inputs anywhere downstream.
    real_filter = SLAFilter(
        metric_tag="time_to_first_token",
        stat="p95",
        op="lt",
        threshold=200.0,
    )
    assert real_filter.threshold == 200.0
