# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived-metric instance state is scoped per accumulator (per run), not
process-wide via the MetricRegistry singleton: derive funcs must be bound to
fresh per-run instances so any instance state (e.g. the degraded warn-once
latch) fires once per run rather than once per process."""

import uuid

import pytest

from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.metrics.types.replay_sched_lag_metrics import ReplaySchedDegradedMetric
from aiperf.plugin.enums import EndpointType


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


def test_derive_funcs_bound_to_fresh_per_run_instances(fixed_schedule_run) -> None:
    """Each accumulator binds derive funcs to its own metric instances, distinct
    from other runs and from the MetricRegistry singleton, so per-instance state
    (e.g. warn-once latches) is scoped to the run rather than the process."""
    tag = ReplaySchedDegradedMetric.tag

    acc1 = MetricsAccumulator(fixed_schedule_run)
    acc2 = MetricsAccumulator(fixed_schedule_run)

    assert tag in acc1._derive_funcs
    assert tag in acc2._derive_funcs

    f1 = acc1._derive_funcs[tag]
    f2 = acc2._derive_funcs[tag]
    # Bound to per-run instances, not shared and not the registry singleton.
    assert f1.__self__ is not f2.__self__
    singleton = MetricRegistry.get_instance(tag)
    assert f1.__self__ is not singleton
    assert f2.__self__ is not singleton
