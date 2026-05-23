# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SLA-helper unmeasurable-path diagnostics.

The SLA-aware planners (``smooth_isotonic``, ``monotonic_sla``) latch
``infeasible_min`` whenever ``first_failing_filter`` returns ``observed: null``
and terminate with ``no_pass_in_range``. Pre-fix, that null collapse was
silent: a production sweep against ``nemotron-super-vllm-agg-frontend:8000``
on DGX 2026-05-06 hit it with no log line naming the failing metric tag,
because the SLA filter happily returned ``observed: null`` whether the
metric was missing, the stat was None, or all runs failed.

The fix logs which of those three branches fired with the offending
metric_tag/stat, so a single grep against the sweep-controller pod tells
oncall whether the file is genuinely missing TTFT, the percentile is
absent, or every run died before measurement. The regression test below
fences each branch.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from aiperf import __version__ as aiperf_version
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner._sla_helpers import (
    first_failing_filter,
    iteration_feasibility,
)

_TTFT_SLA = SLAFilter(
    metric_tag="time_to_first_token",
    stat="p95",
    op="lt",
    threshold=30000.0,
)


def _run(
    *,
    success: bool = True,
    summary: dict[str, JsonMetricResult] | None = None,
) -> RunResult:
    return RunResult(
        label="test/run_0001",
        success=success,
        summary_metrics=summary or {},
    )


# ---------------------------------------------------------------------------
# Happy path: populated p95 → observed echoes the value.
# ---------------------------------------------------------------------------


def test_first_failing_filter_returns_observed_when_p95_populated() -> None:
    """A run with TTFT.p95 above threshold → observed is the actual value."""
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(
                unit="ms",
                avg=37313.0,
                p50=35000.0,
                p95=40000.0,
                p99=43165.0,
            ),
        }
    )

    breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is not None
    assert breach["metric_tag"] == "time_to_first_token"
    assert breach["observed"] == 40000.0
    assert breach["threshold"] == 30000.0
    assert breach["op"] == "lt"


# ---------------------------------------------------------------------------
# Three silent unmeasurable paths — each must log a clear cause AND
# preserve the ``observed: null`` contract for the planner.
# ---------------------------------------------------------------------------


def test_first_failing_filter_logs_missing_metric_tag_and_returns_null(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful run, but its summary lacks the SLA filter's metric tag.

    This is the production scenario from DGX 2026-05-06: the sweep-controller
    recovered 19 metrics via the operator API but ``time_to_first_token`` was
    not among them. Pre-fix, the planner latched infeasible_min with no log.
    """
    run = _run(
        summary={
            "request_latency": JsonMetricResult(unit="ms", avg=1500.0, p95=2000.0),
            "output_token_throughput": JsonMetricResult(
                unit="tokens/sec", avg=100.0, p95=95.0
            ),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] is None  # contract preserved
    # The log must name the offending tag AND list available tags so a
    # production grep finds the mismatch immediately.
    assert any(
        "time_to_first_token" in r.message
        and "missing" in r.message
        and "request_latency" in r.message
        for r in caplog.records
    ), (
        "expected WARNING naming missing metric_tag and listing available tags; "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_first_failing_filter_logs_missing_stat_and_returns_null(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Metric present but ``p95`` is None — common when an export omits a percentile.

    The legacy ``aiperf-llama3-c128`` fixture has TTFT with p50/p90/p99 only,
    no p95; pre-fix this collapsed to ``observed: null`` without any
    indication that the percentile was the problem.
    """
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(
                unit="ms",
                avg=150.0,
                p50=142.5,
                p90=165.0,
                p99=195.0,
                # p95 deliberately omitted
            ),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] is None
    assert any(
        "time_to_first_token" in r.message
        and "p95" in r.message
        and "None" in r.message
        for r in caplog.records
    ), (
        "expected WARNING naming the missing stat=p95; "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_first_failing_filter_logs_no_successful_runs_and_returns_null(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All runs failed — observed is null and the cause is named."""
    failed_run = _run(success=False)

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter([failed_run], [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] is None
    assert any("no successful trials" in r.message for r in caplog.records), (
        "expected WARNING naming the no-successful-trials cause; "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_first_failing_filter_no_successful_runs_lists_failed_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When every run failed, the diagnosis names the failure reasons.

    Pre-fix the planner only said "no successful trials in this iteration",
    forcing oncall into a second log query to find why the children died. Now
    the diagnostic enumerates per-child error strings (truncated past 3) so
    the failure cause is one log line away.
    """
    fails = [
        RunResult(
            label="probe-c01-t0",
            success=False,
            error="child terminal phase=Failed",
        ),
        RunResult(
            label="probe-c01-t1",
            success=False,
            error="endpoint refused: connect ECONNREFUSED 10.0.0.5:8000",
        ),
    ]

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter(fails, [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] is None
    msg_text = "\n".join(r.message for r in caplog.records)
    assert "2 failed" in msg_text
    assert "probe-c01-t0" in msg_text
    assert "child terminal phase=Failed" in msg_text
    assert "ECONNREFUSED" in msg_text


def test_first_failing_filter_no_successful_runs_truncates_past_three(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With more than 3 failed runs, the diagnostic truncates to keep logs readable."""
    fails = [
        RunResult(label=f"probe-{i}", success=False, error=f"err-{i}") for i in range(5)
    ]

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        first_failing_filter(fails, [_TTFT_SLA])

    msg_text = "\n".join(r.message for r in caplog.records)
    assert "5 failed" in msg_text
    assert "(+2 more)" in msg_text


def test_first_failing_filter_missing_stat_lists_available_stats(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When p95 is None, the diagnosis enumerates which stats DO have values.

    Distinguishes "exporter dropped p95" (avg/p50/p99 present) from "metric is
    a degenerate scalar with no percentiles" (only avg present), so oncall
    knows whether to widen the SLA stat or check the exporter.
    """
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(
                unit="ms",
                avg=150.0,
                p50=142.5,
                p99=195.0,
                # p90, p95, min, max deliberately omitted
            ),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        first_failing_filter([run], [_TTFT_SLA])

    msg_text = "\n".join(r.message for r in caplog.records)
    assert "stats with values:" in msg_text
    assert "avg" in msg_text
    assert "p50" in msg_text
    assert "p99" in msg_text


def test_first_failing_filter_missing_stat_no_values_at_all(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A degenerate JsonMetricResult with only ``unit`` populated → diagnosis names that.

    This is the boundary case where the metric tag exists but the exporter
    didn't actually populate any stat value. The diagnostic must NOT crash
    and must surface "(no stats populated)" so oncall sees a unit-only entry
    rather than mistaking it for a missing-tag.
    """
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(unit="ms"),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        first_failing_filter([run], [_TTFT_SLA])

    msg_text = "\n".join(r.message for r in caplog.records)
    assert "no stats populated" in msg_text


# ---------------------------------------------------------------------------
# No spurious warning when the SLA is genuinely satisfied (or when the
# feasible run path holds — silence on the happy path is contract too).
# ---------------------------------------------------------------------------


def test_no_warning_when_filter_passes(caplog: pytest.LogCaptureFixture) -> None:
    """A run satisfying the SLA → no breach, no warning."""
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=100.0),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is None
    # No diagnostic log emitted on the happy path.
    assert all("time_to_first_token" not in (r.message or "") for r in caplog.records)


def test_no_warning_when_observed_populated_and_breaching(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A breaching run with a populated stat → breach reported, no warning."""
    run = _run(
        summary={
            "time_to_first_token": JsonMetricResult(unit="ms", p95=40000.0),
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="aiperf.orchestrator.search_planner._sla_helpers"
    ):
        breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] == 40000.0
    # The unmeasurable-path warning must NOT fire when the breach has data.
    assert all(
        "missing" not in (r.message or "")
        and "no successful trials" not in (r.message or "")
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# iteration_feasibility contract preserved across the three unmeasurable paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param({}, id="empty-summary"),
        pytest.param(
            {"time_to_first_token": JsonMetricResult(unit="ms")},
            id="metric-without-p95",
        ),
        pytest.param(
            {"request_latency": JsonMetricResult(unit="ms", p95=200.0)},
            id="other-metric-only",
        ),
    ],
)
def test_iteration_feasibility_false_when_unmeasurable(
    summary: dict[str, JsonMetricResult],
) -> None:
    """All three unmeasurable shapes must return feasible=False, matching the
    pre-fix planner-bracket contract (silent treat-as-pass would invert)."""
    run = _run(summary=summary)
    assert iteration_feasibility([run], [_TTFT_SLA]) is False


def test_iteration_feasibility_true_when_p95_under_threshold() -> None:
    """Sanity: the happy path stays feasible."""
    run = _run(
        summary={"time_to_first_token": JsonMetricResult(unit="ms", p95=100.0)},
    )
    assert iteration_feasibility([run], [_TTFT_SLA]) is True


# ---------------------------------------------------------------------------
# End-to-end shape check: project_summary_dict → first_failing_filter
# mirrors the production sweep-child fallback path. Asserts that a real
# profile_export-shaped payload with a populated TTFT entry surfaces the
# observed value (i.e. the path that should have worked but produced ``null``
# in the DGX 2026-05-06 incident IS actually wired correctly when the data
# is present).
# ---------------------------------------------------------------------------


def test_project_summary_dict_to_sla_filter_end_to_end() -> None:
    """Mirrors ``K8sChildJobExecutor._fetch_summary_from_operator`` plumbing.

    The sweep-controller projects the operator-API JSON via
    :func:`JsonMetricResult.project_summary_dict` and stuffs the result onto
    ``RunResult.summary_metrics``; the planner then reads it. With a realistic
    ``profile_export_aiperf.json`` shape, the SLA filter must surface the
    measured ``time_to_first_token.p95`` value, not ``null``.
    """
    # Mirrors the real on-disk profile_export shape: top-level metric tags
    # with full percentile dicts.
    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "aiperf_version": aiperf_version,
        "request_throughput": {"unit": "req/sec", "avg": 1.5, "p95": 1.6},
        "request_latency": {"unit": "ms", "avg": 1500.0, "p95": 2000.0},
        "time_to_first_token": {
            "unit": "ms",
            "avg": 37313.0,
            "p50": 35000.0,
            "p95": 40000.0,
            "p99": 43165.0,
        },
        "output_token_throughput": {"unit": "tokens/sec", "avg": 100.0, "p95": 95.0},
    }
    projected = JsonMetricResult.project_summary_dict(payload)
    assert "time_to_first_token" in projected
    assert projected["time_to_first_token"].p95 == 40000.0

    run = RunResult(label="x", success=True, summary_metrics=projected)
    breach = first_failing_filter([run], [_TTFT_SLA])

    assert breach is not None
    assert breach["observed"] == 40000.0  # not None
    assert breach["metric_tag"] == "time_to_first_token"
