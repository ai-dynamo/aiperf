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
