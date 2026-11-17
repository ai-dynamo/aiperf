# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.accuracy.models import AccuracySummary, TaskAccuracyStats
from aiperf.common.enums import MetricConsoleGroup


def _make_summary() -> AccuracySummary:
    return AccuracySummary(
        total_evaluated=10,
        total_passed=8,
        accuracy_rate=0.8,
        overall_unparsed=1,
        grader_name="mmlu",
        per_task={
            "history": TaskAccuracyStats(
                total=5, passed=5, unparsed=0, accuracy_rate=1.0, unparsed_rate=0.0
            ),
            "algebra": TaskAccuracyStats(
                total=5, passed=3, unparsed=1, accuracy_rate=0.6, unparsed_rate=0.2
            ),
        },
    )


class TestToMetricResults:
    def test_order_and_tags(self) -> None:
        results = _make_summary().to_metric_results()
        assert [r.tag for r in results] == [
            "accuracy.overall",
            "accuracy.task.algebra",
            "accuracy.task.history",
            "accuracy.unparsed",
            "accuracy.unparsed.task.algebra",
            "accuracy.unparsed.task.history",
        ]

    def test_overall_fields(self) -> None:
        overall = _make_summary().to_metric_results()[0]
        assert overall.header == "Accuracy (Overall)"
        assert overall.unit == "ratio"
        assert overall.count == 10
        assert overall.current == 0.8
        assert overall.sum == 8
        assert overall.console_group == MetricConsoleGroup.NONE

    def test_task_fields(self) -> None:
        by_tag = {r.tag: r for r in _make_summary().to_metric_results()}
        algebra = by_tag["accuracy.task.algebra"]
        assert algebra.header == "Accuracy (algebra)"
        assert algebra.unit == "ratio"
        assert algebra.count == 5
        assert algebra.current == 0.6
        assert algebra.sum == 3
        assert algebra.console_group == MetricConsoleGroup.NONE

    def test_unparsed_overall_fields(self) -> None:
        by_tag = {r.tag: r for r in _make_summary().to_metric_results()}
        unparsed = by_tag["accuracy.unparsed"]
        assert unparsed.header == "Accuracy Unparsed (Overall)"
        assert unparsed.unit == "ratio"
        assert unparsed.count == 10
        assert unparsed.current == 0.1
        assert unparsed.sum == 1
        assert unparsed.console_group == MetricConsoleGroup.NONE

    def test_unparsed_task_fields(self) -> None:
        by_tag = {r.tag: r for r in _make_summary().to_metric_results()}
        algebra = by_tag["accuracy.unparsed.task.algebra"]
        assert algebra.header == "Accuracy Unparsed (algebra)"
        assert algebra.count == 5
        assert algebra.current == 0.2
        assert algebra.sum == 1
        assert algebra.console_group == MetricConsoleGroup.NONE

    def test_json_shape_is_unit_count_sum(self) -> None:
        """The perf JSON exporter projects each MetricResult through
        JsonMetricResult (unit/avg/p*/count/sum only) and drops None fields,
        yielding exactly ``{"unit":"ratio","count":N,"sum":M}`` for accuracy.*."""
        overall = _make_summary().to_metric_results()[0]
        projected = overall.to_json_result().model_dump(exclude_none=True)
        assert projected == {"unit": "ratio", "count": 10, "sum": 8}

    def test_empty_summary_emits_nothing(self) -> None:
        empty = AccuracySummary(
            total_evaluated=0,
            total_passed=0,
            accuracy_rate=0.0,
            overall_unparsed=0,
        )
        assert empty.to_metric_results() == []


class TestToMetricResultsGraderErrors:
    def test_no_grader_error_tags_when_zero(self) -> None:
        tags = [r.tag for r in _make_summary().to_metric_results()]
        assert not any(t.startswith("accuracy.grader_error") for t in tags)

    def test_grader_error_tags_appended_after_unparsed_in_sorted_order(self) -> None:
        summary = _make_summary().model_copy(
            update={
                "overall_grader_errors": 2,
                "per_task": {
                    "history": TaskAccuracyStats(
                        total=5,
                        passed=5,
                        unparsed=0,
                        accuracy_rate=1.0,
                        unparsed_rate=0.0,
                    ),
                    "algebra": TaskAccuracyStats(
                        total=5,
                        passed=3,
                        unparsed=1,
                        accuracy_rate=0.6,
                        unparsed_rate=0.2,
                        grader_errors=2,
                        grader_error_rate=0.4,
                    ),
                },
            }
        )

        results = summary.to_metric_results()

        assert [r.tag for r in results][-3:] == [
            "accuracy.grader_error",
            "accuracy.grader_error.task.algebra",
            "accuracy.grader_error.task.history",
        ]
        overall, algebra, history = results[-3:]
        assert (overall.sum, overall.count, overall.current) == (2, 10, 0.2)
        assert (algebra.sum, algebra.count, algebra.current) == (2, 5, 0.4)
        assert (history.sum, history.current) == (0, 0.0)
        assert overall.console_group == MetricConsoleGroup.NONE
