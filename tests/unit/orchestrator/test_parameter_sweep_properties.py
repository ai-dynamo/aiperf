# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property tests for parameter-sweep variation expansion.

Small slice of main's ``test_parameter_sweep_properties.py`` (843 LOC),
adapted to k8s's ``AIPerfConfig`` + ``build_benchmark_plan`` + ``expand_sweep``
pipeline. Asserts hypothesis-driven invariants over the variation count,
per-cell concurrency value, and the order-stability of grouping.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf._cli_runner_sweep_helpers import _group_results_by_variation
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import AIPerfConfig
from aiperf.config.loader import build_benchmark_plan
from aiperf.orchestrator.models import RunResult


def _make_config(concurrency: list[int]) -> AIPerfConfig:
    """Build a minimal AIPerfConfig with a sweep over ``phases.profiling.concurrency``."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets=[
            {
                "name": "default",
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            }
        ],
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 5,
                "concurrency": 1,
            }
        ],
        sweep={
            "type": "grid",
            "variables": {"phases.profiling.concurrency": concurrency},
        },
    )


def _profiling_concurrency(cfg: Any) -> int:
    """Pull the concurrency from the profiling phase of a BenchmarkConfig."""
    for phase in cfg.phases:
        if phase.name == "profiling":
            return phase.concurrency
    raise AssertionError("no profiling phase on resolved config")


# Distinct, modest-magnitude positives keep AIPerfConfig validators happy
# (concurrency must be >= 1 and a sane sweep size). min_size=2 because
# BenchmarkPlan.is_sweep is False on single-variation plans — the property
# tests target the multi-variation expansion path.
_concurrency_lists = st.lists(
    st.integers(min_value=1, max_value=512),
    min_size=2,
    max_size=6,
    unique=True,
)


@given(concurrency=_concurrency_lists)
@settings(deadline=None, max_examples=25)
def test_expand_sweep_variation_count_matches_concurrency_list_length(
    concurrency: list[int],
) -> None:
    """``len(plan.configs) == len(concurrency)`` for any distinct positive list.

    Key guarantee from ``expand_sweep`` + ``build_benchmark_plan``:
    one BenchmarkConfig per swept value, in deterministic order.
    """
    plan = build_benchmark_plan(_make_config(concurrency))

    assert plan.is_sweep
    assert len(plan.configs) == len(concurrency)
    assert len(plan.variations) == len(concurrency)


@given(concurrency=_concurrency_lists)
@settings(deadline=None, max_examples=25)
def test_each_variation_carries_its_swept_concurrency_value(
    concurrency: list[int],
) -> None:
    """The ith plan.configs entry has phases.profiling.concurrency == input[i].

    Order must be stable: the i-th variation must carry the i-th input
    value in both ``cfg.phases[profiling].concurrency`` and
    ``variation.values``.
    """
    plan = build_benchmark_plan(_make_config(concurrency))

    actual_per_config = [_profiling_concurrency(c) for c in plan.configs]
    assert actual_per_config == concurrency

    actual_per_variation = [
        v.values["phases.profiling.concurrency"] for v in plan.variations
    ]
    assert actual_per_variation == concurrency


@given(concurrency=_concurrency_lists)
@settings(deadline=None, max_examples=25)
def test_group_results_by_variation_preserves_first_seen_order(
    concurrency: list[int],
) -> None:
    """``_group_results_by_variation`` keys mirror first-seen order.

    Sweep-aggregate CSV row order is downstream of this dict's iteration
    order; if grouping shuffled keys, reruns would diff against each
    other for cosmetic reasons. One trial per cell so groups are 1:1.
    """
    results = [
        RunResult(
            label=f"run-{c}",
            success=True,
            summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
            variation_label=f"phases.profiling.concurrency={c}",
            variation_values={"phases.profiling.concurrency": c},
            trial_index=0,
        )
        for c in concurrency
    ]
    groups = _group_results_by_variation(results)

    keyed_concurrencies = [dict(key)["phases.profiling.concurrency"] for key in groups]
    assert keyed_concurrencies == concurrency
    assert all(len(group) == 1 for group in groups.values())
