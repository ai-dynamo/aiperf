# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived-metric instance state is scoped per processor (per run), not
process-wide via the MetricRegistry singleton: the degraded warn-once latch
must fire once per run, and derive funcs must be bound to per-processor
instances."""

import logging
import uuid

import pytest

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.metrics.metric_dicts import MetricArray
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.replay_sched_lag_metrics import (
    ReplaySchedDegradedMetric,
    ReplaySendScheduleOffsetMetric,
)
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor


@pytest.fixture
def fixed_schedule_run():
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "type": EndpointType.COMPLETIONS,
                "urls": ["http://localhost:8000/v1"],
                "streaming": False,
            },
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "type": "fixed_schedule"}],
        }
    )
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


def _feed_degraded_offsets(processor: MetricResultsProcessor) -> None:
    arr = MetricArray()
    arr.extend([0] * 5 + [int(600 * NANOS_PER_MILLIS)] * 5)
    processor._results[ReplaySendScheduleOffsetMetric.tag] = arr


@pytest.mark.asyncio
async def test_degraded_warning_fires_once_per_run_not_once_per_process(
    fixed_schedule_run, caplog: pytest.LogCaptureFixture
):
    tag = ReplaySchedDegradedMetric.tag

    with caplog.at_level(logging.WARNING):
        # Run 1: two summarize ticks -> exactly one warning.
        p1 = MetricResultsProcessor(fixed_schedule_run)
        assert tag in p1.derive_funcs
        _feed_degraded_offsets(p1)
        await p1.update_derived_metrics()
        assert p1._results[tag] == 1
        await p1.update_derived_metrics()

        run1_warnings = [
            r for r in caplog.records if "Replay schedule degraded" in r.message
        ]
        assert len(run1_warnings) == 1

        # Run 2 (same process): fresh processor -> the warning fires again.
        p2 = MetricResultsProcessor(fixed_schedule_run)
        _feed_degraded_offsets(p2)
        await p2.update_derived_metrics()
        assert p2._results[tag] == 1

    all_warnings = [
        r for r in caplog.records if "Replay schedule degraded" in r.message
    ]
    assert len(all_warnings) == 2

    # The latch lives on per-processor instances, not the registry singleton.
    f1, f2 = p1.derive_funcs[tag], p2.derive_funcs[tag]
    assert f1.__self__ is not f2.__self__
    singleton = MetricRegistry.get_instance(tag)
    assert f1.__self__ is not singleton
    assert f2.__self__ is not singleton
