# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for issue #1145 (secondary bug): accuracy summary tags must
survive the metric display/filter path without raising ``MetricTypeError``.

The ``accuracy.overall`` / ``accuracy.task.<name>`` / ``accuracy.unparsed`` /
``accuracy.unparsed.task.<name>`` tags are produced by
``AccuracySummary.to_metric_results()`` as plain ``MetricResult`` objects. They
are intentionally NOT registered in ``MetricRegistry`` (they are not
``BaseMetric`` subclasses): their values come from grading on a dedicated
result-producer channel, and the per-task tags are dynamic (one per benchmark
subtask, unknown at class-definition time), so static registration can't cover
them.

On aiperf v0.11.0 the realtime/dashboard path resolved every incoming tag with
the *raising* ``MetricRegistry.get_class`` and crashed once per graded result
with ``Metric class with tag 'accuracy.overall' not found``. That was fixed by
making the consumers tolerant of unregistered tags (``get_class_or_none`` in the
dashboard table, ``try/except MetricTypeError`` in ``filter_display_metrics``).
These tests pin that tolerance so a future refactor can't silently reintroduce a
raising lookup on the accuracy path.
"""

from __future__ import annotations

from aiperf.accuracy.models import (
    ACCURACY_OVERALL_TAG,
    ACCURACY_UNPARSED_TAG,
    AccuracySummary,
    TaskAccuracyStats,
    accuracy_task_tag,
    accuracy_unparsed_task_tag,
)
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.records.records_manager_processing import filter_display_metrics


def _summary() -> AccuracySummary:
    """A summary with an overall rollup plus two per-task rollups, so
    to_metric_results() emits every accuracy tag shape (overall, per-task,
    unparsed overall, unparsed per-task)."""
    return AccuracySummary(
        total_evaluated=10,
        total_passed=7,
        accuracy_rate=0.7,
        overall_unparsed=1,
        per_task={
            "boolean_expressions": TaskAccuracyStats(
                total=5, passed=4, unparsed=0, accuracy_rate=0.8, unparsed_rate=0.0
            ),
            "navigate": TaskAccuracyStats(
                total=5, passed=3, unparsed=1, accuracy_rate=0.6, unparsed_rate=0.2
            ),
        },
    )


class TestAccuracyTagsAreUnregistered:
    """Documents the invariant that makes the display-path guard necessary."""

    def test_to_metric_results_accuracy_tags_absent_from_registry(self) -> None:
        results = _summary().to_metric_results()
        assert results, "expected the summary to emit accuracy MetricResults"

        expected_tags = {
            ACCURACY_OVERALL_TAG,
            ACCURACY_UNPARSED_TAG,
            accuracy_task_tag("boolean_expressions"),
            accuracy_task_tag("navigate"),
            accuracy_unparsed_task_tag("boolean_expressions"),
            accuracy_unparsed_task_tag("navigate"),
        }
        assert {r.tag for r in results} == expected_tags

        # None are registered — so any raising get_class on this path would crash.
        for tag in expected_tags:
            assert MetricRegistry.get_class_or_none(tag) is None


class TestAccuracyTagsSurviveDisplayFilter:
    """filter_display_metrics feeds the dashboard's realtime view. It must not
    raise on unregistered accuracy tags — the exact crash from issue #1145."""

    def test_filter_display_metrics_unregistered_accuracy_tags_passes_through(
        self,
    ) -> None:
        results = _summary().to_metric_results()
        # Must not raise MetricTypeError; unregistered tags pass through as-is.
        filtered = filter_display_metrics(results)
        assert {r.tag for r in filtered} == {r.tag for r in results}


class TestAccuracyTagsSurviveDashboardTable:
    """The realtime dashboard table resolves each tag to its metric class for
    display metadata. On v0.11.0 it used the raising get_class and crashed in
    ``on_realtime_metrics``; it must use the tolerant lookup for accuracy tags."""

    def test__should_skip_unregistered_accuracy_tag_returns_bool(self) -> None:
        from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsTable

        # The tag->class resolution helpers use only MetricRegistry + Environment,
        # not widget state, so a bare instance exercises them without a running app.
        table = RealtimeMetricsTable.__new__(RealtimeMetricsTable)
        for metric in _summary().to_metric_results():
            assert table._should_skip(metric) in (True, False)

    def test__metric_display_order_unregistered_accuracy_tag_returns_int(self) -> None:
        from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsTable

        table = RealtimeMetricsTable.__new__(RealtimeMetricsTable)
        for metric in _summary().to_metric_results():
            assert isinstance(table._metric_display_order(metric), int)

    def test__metric_short_header_unregistered_accuracy_tag_returns_str(self) -> None:
        from aiperf.ui.dashboard.realtime_metrics_dashboard import RealtimeMetricsTable

        table = RealtimeMetricsTable.__new__(RealtimeMetricsTable)
        for metric in _summary().to_metric_results():
            assert isinstance(table._metric_short_header(metric), str)
