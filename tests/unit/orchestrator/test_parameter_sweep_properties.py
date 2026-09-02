# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property tests for parameter-sweep correctness.

Hypothesis-driven invariants ported and adapted from main's
``test_parameter_sweep_properties.py`` to k8s' ``AIPerfConfig`` +
``build_benchmark_plan`` + ``expand_sweep`` + ``MultiRunOrchestrator``
pipeline. Each ``TestPropertyN*`` class mirrors the corresponding
property from the parameter-sweeping design doc.

Architectural mapping (main -> HEAD):
- ``ParameterSweepStrategy`` and ``SweepConfidenceStrategy`` were
  collapsed into ``MultiRunOrchestrator`` + ``expand_sweep``; the
  per-cell strategy is ``FixedTrialsStrategy`` only.
- ``CLIConfig.loadgen.concurrency`` -> ``AIPerfConfig.phases[i].concurrency``
  (with magic-list -> sweep-block promotion in the v1->v2 converter).
- Variation labels: ``concurrency_10`` -> ``phases.profiling.concurrency=10``.
- Seed derivation: ``base_seed + variation.index`` is now applied by
  ``_apply_sweep_seed_derivation`` in ``config/loader/plan.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pytest import param

from aiperf.cli_runner._sweep_aggregate import _group_results_by_variation
from aiperf.common.enums import SweepMode
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import AIPerfConfig
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import expand_sweep
from aiperf.orchestrator.aggregation.sweep import (
    DEFAULT_PARETO_OBJECTIVES,
    ParameterCombination,
    identify_pareto_optimal,
)
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

# =============================================================================
# Test Helpers
# =============================================================================


def _make_config(
    concurrency: list[int] | int = 1,
    *,
    random_seed: int | None = None,
    parameter_sweep_same_seed: bool = False,
    parameter_sweep_cooldown_seconds: float = 0.0,
    mode: SweepMode | str = SweepMode.REPEATED,
    num_runs: int = 1,
) -> AIPerfConfig:
    """Build a minimal AIPerfConfig with optional sweep over profiling concurrency.

    If ``concurrency`` is a list, expands to a grid sweep at
    ``phases.profiling.concurrency``. Otherwise sets a scalar.
    """
    payload: dict[str, Any] = {
        "benchmark": {
            "models": ["test-model"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32, "osl": 16},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 5,
                    "concurrency": concurrency if isinstance(concurrency, int) else 1,
                }
            ],
        },
        "multi_run": {
            "num_runs": num_runs,
        },
    }
    if random_seed is not None:
        payload["random_seed"] = random_seed
    if isinstance(concurrency, list):
        payload["sweep"] = {
            "type": "grid",
            "parameters": {"phases.profiling.concurrency": concurrency},
            "same_seed": parameter_sweep_same_seed,
            "cooldown_seconds": parameter_sweep_cooldown_seconds,
            "iteration_order": mode,
        }
    return AIPerfConfig(**payload)


def _profiling_concurrency(cfg: Any) -> int:
    """Extract concurrency from the profiling phase of a BenchmarkConfig."""
    for phase in cfg.phases:
        if phase.name == "profiling":
            return phase.concurrency
    raise AssertionError("no profiling phase on resolved config")


def _per_variation_seeds(plan: BenchmarkPlan) -> list[int | None]:
    """Project plan.variation_seeds (parallel to plan.configs)."""
    return list(plan.variation_seeds)


# Distinct, modest-magnitude positives keep AIPerfConfig validators happy
# (concurrency must be >= 1). min_size=2 because BenchmarkPlan.is_sweep is
# False on single-variation plans -- the property tests target the
# multi-variation expansion path.
_concurrency_lists_unique = st.lists(
    st.integers(min_value=1, max_value=512),
    min_size=2,
    max_size=6,
    unique=True,
)

# For Property 3 (duplicates): allow duplicates so we exercise the
# variation-per-occurrence semantic.
_concurrency_lists_with_duplicates = st.lists(
    st.integers(min_value=1, max_value=64),
    min_size=2,
    max_size=6,
    unique=False,
)


# =============================================================================
# Property 1: Concurrency List Parsing
# =============================================================================


class TestProperty1PBT:
    """Property 1: Concurrency List Parsing.

    For any list of valid integers, sweep expansion preserves order and
    one variation per list element.
    """

    @given(concurrency=_concurrency_lists_unique)
    @settings(
        deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_concurrency_list_preserves_order_and_values(
        self, concurrency: list[int]
    ) -> None:
        """For any list of distinct positive ints, plan.configs preserves order."""
        plan = build_benchmark_plan(_make_config(concurrency))

        assert plan.is_sweep
        assert len(plan.configs) == len(concurrency)
        actual = [_profiling_concurrency(c) for c in plan.configs]
        assert actual == concurrency

    @given(concurrency=_concurrency_lists_unique)
    @settings(
        deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_each_variation_carries_its_swept_value(
        self, concurrency: list[int]
    ) -> None:
        """The i-th variation's values dict carries the i-th concurrency."""
        plan = build_benchmark_plan(_make_config(concurrency))

        actual = [v.values["phases.profiling.concurrency"] for v in plan.variations]
        assert actual == concurrency


# =============================================================================
# Property 2: Invalid Input Rejection
# =============================================================================


class TestProperty2PBT:
    """Property 2: Invalid Input Rejection.

    For any concurrency list containing values < 1, AIPerfConfig
    validation should reject the input.
    """

    @given(
        valid_values=st.lists(
            st.integers(min_value=1, max_value=100), min_size=1, max_size=4
        ),
        invalid_value=st.integers(max_value=0),
        data=st.data(),
    )
    @settings(
        deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_rejects_invalid_concurrency_in_sweep(
        self, valid_values: list[int], invalid_value: int, data: Any
    ) -> None:
        """Inserting any value < 1 into a sweep list should make the plan fail.

        The phase ``concurrency`` field has ``ge=1``, so when ``expand_sweep``
        materializes the variation that carries the invalid value,
        ``BenchmarkConfig.model_validate`` rejects it.
        """
        position = data.draw(st.integers(min_value=0, max_value=len(valid_values)))
        test_values = (
            valid_values[:position] + [invalid_value] + valid_values[position:]
        )

        # Building the config object is allowed (sweep.parameters is a
        # dict[str, list[Any]]); the rejection happens at plan-build time
        # when the per-variation BenchmarkConfig is validated.
        cfg = _make_config(test_values)
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError surfaces here
            build_benchmark_plan(cfg)

    def test_rejects_zero_scalar_concurrency(self) -> None:
        """Scalar concurrency=0 fails AIPerfConfig validation directly."""
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            _make_config(0)


# =============================================================================
# Property 3: Duplicate Values Allowed
# =============================================================================


class TestProperty3DuplicateValuesPBT:
    """Property 3: Duplicate Values Allowed.

    For any concurrency list with duplicates, sweep expansion produces
    one variation per occurrence (preserving order).
    """

    @given(concurrency=_concurrency_lists_with_duplicates)
    @settings(
        deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_duplicates_create_one_variation_per_occurrence(
        self, concurrency: list[int]
    ) -> None:
        """``expand_sweep`` honors duplicate values by emitting one config each."""
        # Use expand_sweep directly: build_benchmark_plan goes through Pydantic
        # validation that may dedup on the dict path. The pure expansion is
        # what the orchestrator iterates over.
        data = {
            "benchmark": {
                "phases": [{"name": "profiling", "concurrency": 1}],
            },
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": concurrency},
            },
        }
        expanded = expand_sweep(data)

        assert len(expanded) == len(concurrency)
        actual = [
            variation_dict["benchmark"]["phases"][0]["concurrency"]
            for variation_dict, _ in expanded
        ]
        assert actual == concurrency


# =============================================================================
# Property 11: Seed Derivation Consistency
# =============================================================================


class TestProperty11SeedDerivationPBT:
    """Property 11: Seed Derivation Consistency.

    Two builds of the same AIPerfConfig must produce identical per-variation
    seeds. The derivation formula is ``base_seed + variation.index``.
    """

    @given(
        concurrency=_concurrency_lists_unique,
        base_seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(
        deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_seed_derivation_is_reproducible(
        self, concurrency: list[int], base_seed: int
    ) -> None:
        """Same base seed, two builds -> identical per-variation seeds."""
        seeds1 = _per_variation_seeds(
            build_benchmark_plan(_make_config(concurrency, random_seed=base_seed))
        )
        seeds2 = _per_variation_seeds(
            build_benchmark_plan(_make_config(concurrency, random_seed=base_seed))
        )

        assert seeds1 == seeds2
        # Variation 0 keeps base; subsequent ones are base + index.
        expected = [base_seed] + [base_seed + i for i in range(1, len(concurrency))]
        assert seeds1 == expected

    @given(
        concurrency=_concurrency_lists_unique,
        base_seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(
        deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_same_seed_mode_uses_identical_seed(
        self, concurrency: list[int], base_seed: int
    ) -> None:
        """``parameter_sweep_same_seed=True`` -> every variation reuses base_seed."""
        plan = build_benchmark_plan(
            _make_config(
                concurrency,
                random_seed=base_seed,
                parameter_sweep_same_seed=True,
            )
        )

        seeds = _per_variation_seeds(plan)
        assert all(seed == base_seed for seed in seeds)


# =============================================================================
# Property 15: Cooldown Application
# =============================================================================


class TestProperty15CooldownApplication:
    """Property 15: Cooldown Application.

    Plan-level cooldowns propagate cleanly to ``BenchmarkPlan``.
    Trial-level cooldowns live on ``FixedTrialsStrategy`` (per-cell);
    sweep-level cooldown lives on the plan and is honored between
    variations by ``MultiRunOrchestrator._execute_independent``.
    """

    @pytest.mark.parametrize(
        "cooldown",
        [
            param(0.0, id="zero"),
            param(0.5, id="fractional"),
            param(5.0, id="five"),
            param(60.0, id="sixty"),
        ],
    )  # fmt: skip
    def test_sweep_cooldown_propagates_to_plan(self, cooldown: float) -> None:
        plan = build_benchmark_plan(
            _make_config([10, 20], parameter_sweep_cooldown_seconds=cooldown)
        )
        assert plan.sweep is not None
        assert plan.sweep.cooldown_seconds == cooldown

    def test_default_cooldown_is_zero(self) -> None:
        plan = build_benchmark_plan(_make_config([10, 20]))
        assert plan.sweep is not None
        assert plan.sweep.cooldown_seconds == 0.0


# =============================================================================
# Property 13: Pareto Optimal Identification
# =============================================================================


class TestProperty13ParetoPBT:
    """Property 13: Pareto Optimal Identification.

    For any set of (throughput, latency) points, ``identify_pareto_optimal``
    must (a) return only undominated points, and (b) every non-Pareto point
    must be dominated by at least one Pareto point. This is the canonical
    correctness invariant of the multi-objective frontier.
    """

    @given(
        metrics=st.lists(
            st.tuples(
                st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.floats(
                    min_value=1.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(
        deadline=None, max_examples=100, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_pareto_optimal_correctness(
        self, metrics: list[tuple[float, float]]
    ) -> None:
        per_combination_stats: dict[ParameterCombination, dict] = {}
        for i, (throughput, latency) in enumerate(metrics):
            combo = ParameterCombination({"config_id": i})
            per_combination_stats[combo] = {
                "request_throughput_avg": {"mean": throughput},
                "time_to_first_token_p99": {"mean": latency},
            }

        pareto = identify_pareto_optimal(per_combination_stats)

        # (a) Every Pareto point is undominated.
        objectives = DEFAULT_PARETO_OBJECTIVES
        for p in pareto:
            p_vals = [
                per_combination_stats[p][obj.metric_key]["mean"] for obj in objectives
            ]
            for other, stats in per_combination_stats.items():
                if other == p:
                    continue
                o_vals = [stats[obj.metric_key]["mean"] for obj in objectives]
                # Throughput maximized, latency minimized.
                better_or_equal_tput = o_vals[0] >= p_vals[0]
                better_or_equal_lat = o_vals[1] <= p_vals[1]
                strictly_better = (o_vals[0] > p_vals[0]) or (o_vals[1] < p_vals[1])
                dominates = (
                    better_or_equal_tput and better_or_equal_lat and strictly_better
                )
                assert not dominates, (
                    f"Pareto point {p} is dominated by {other}: "
                    f"p={p_vals}, other={o_vals}"
                )

        # (b) Every non-Pareto point is dominated by at least one Pareto point.
        non_pareto = [c for c in per_combination_stats if c not in pareto]
        for np_combo in non_pareto:
            np_vals = [
                per_combination_stats[np_combo][obj.metric_key]["mean"]
                for obj in objectives
            ]
            is_dominated = False
            for other, stats in per_combination_stats.items():
                if other == np_combo:
                    continue
                o_vals = [stats[obj.metric_key]["mean"] for obj in objectives]
                better_or_equal_tput = o_vals[0] >= np_vals[0]
                better_or_equal_lat = o_vals[1] <= np_vals[1]
                strictly_better = (o_vals[0] > np_vals[0]) or (o_vals[1] < np_vals[1])
                if better_or_equal_tput and better_or_equal_lat and strictly_better:
                    is_dominated = True
                    break
            assert is_dominated, (
                f"Non-Pareto point {np_combo} ({np_vals}) is dominated by no point"
            )


# =============================================================================
# Property 8 / 9: Execution Order (repeated vs independent)
# =============================================================================


class _RecordingExecutor(RunExecutor):
    """Stand-in RunExecutor that records every (var_idx, trial) pair it sees."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def derive_id(self, plan: BenchmarkPlan, var_idx: int, trial: int) -> str:
        return f"v{var_idx}-t{trial}"

    async def execute(self, run: Any) -> RunResult:
        var_idx = run.variation.index if run.variation else -1
        self.calls.append((var_idx, run.trial))
        return RunResult(
            label=run.label,
            success=True,
            artifacts_path=run.artifact_dir,
        )


def _build_plan_via_orchestrator(
    n_variations: int, n_trials: int, mode: SweepMode
) -> BenchmarkPlan:
    """Build a deterministic N-variation x M-trial plan via the real loader."""
    plan = build_benchmark_plan(
        _make_config(
            list(range(1, n_variations + 1)),
            num_runs=n_trials,
            mode=mode,
        )
    )
    assert plan.is_sweep
    return plan


class TestPropertyExecutionOrderPBT:
    """Properties 8 & 9: Execution Order Patterns.

    For any (N variations, M trials) pair:
    - REPEATED mode produces M cycles of [v0, v1, ..., vN-1]
      (trial-outer, variation-inner).
    - INDEPENDENT mode produces N cycles of [t0, t1, ..., tM-1]
      (variation-outer, trial-inner).

    Adapted from main's ``Property8PBT`` / ``Property9PBT`` (which composed
    ``ParameterSweepStrategy`` + ``FixedTrialsStrategy`` directly) to k8s'
    ``MultiRunOrchestrator.execute`` dispatch path.
    """

    @given(
        n_variations=st.integers(min_value=2, max_value=4),
        n_trials=st.integers(min_value=1, max_value=3),
    )
    @settings(
        deadline=None,
        max_examples=15,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_pbt_repeated_mode_is_trial_outer_variation_inner(
        self, n_variations: int, n_trials: int, tmp_path: Path
    ) -> None:
        plan = _build_plan_via_orchestrator(
            n_variations, n_trials, mode=SweepMode.REPEATED
        )
        assert plan.sweep is not None
        assert plan.sweep.iteration_order == SweepMode.REPEATED

        executor = _RecordingExecutor()
        asyncio.run(MultiRunOrchestrator(base_dir=tmp_path).execute(plan, executor))

        # Repeated: trial outer, variation inner.
        expected = [
            (var_idx, trial)
            for trial in range(n_trials)
            for var_idx in range(n_variations)
        ]
        assert executor.calls == expected

    @given(
        n_variations=st.integers(min_value=2, max_value=4),
        n_trials=st.integers(min_value=1, max_value=3),
    )
    @settings(
        deadline=None,
        max_examples=15,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_pbt_independent_mode_is_variation_outer_trial_inner(
        self, n_variations: int, n_trials: int, tmp_path: Path
    ) -> None:
        plan = _build_plan_via_orchestrator(
            n_variations, n_trials, mode=SweepMode.INDEPENDENT
        )
        assert plan.sweep is not None
        assert plan.sweep.iteration_order == SweepMode.INDEPENDENT

        executor = _RecordingExecutor()
        asyncio.run(MultiRunOrchestrator(base_dir=tmp_path).execute(plan, executor))

        # Independent: variation outer, trial inner.
        expected = [
            (var_idx, trial)
            for var_idx in range(n_variations)
            for trial in range(n_trials)
        ]
        assert executor.calls == expected


# =============================================================================
# Variation Grouping (HEAD-specific: variation-key dict iteration order)
# =============================================================================


class TestGroupResultsByVariationPBT:
    """``_group_results_by_variation`` preserves first-seen order.

    Sweep-aggregate CSV row order is downstream of this dict's iteration
    order; if grouping shuffled keys, reruns would diff against each other
    for cosmetic reasons.
    """

    @given(concurrency=_concurrency_lists_unique)
    @settings(
        deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_group_results_preserves_first_seen_order_one_trial(
        self, concurrency: list[int]
    ) -> None:
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

        # Keys are (variation_label, sorted_values_tuple); the values-tuple is
        # the second element. Each value tuple is e.g. (("phases.profiling.concurrency", 4),).
        keyed = [
            dict(values)["phases.profiling.concurrency"] for _label, values in groups
        ]
        assert keyed == concurrency
        assert all(len(group) == 1 for group in groups.values())

    @given(
        concurrency=_concurrency_lists_unique,
        n_trials=st.integers(min_value=1, max_value=4),
    )
    @settings(
        deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pbt_group_results_collects_all_trials_per_variation(
        self, concurrency: list[int], n_trials: int
    ) -> None:
        """N variations x M trials -> N groups, each of size M."""
        results = [
            RunResult(
                label=f"run-{c}-t{t}",
                success=True,
                summary_metrics={"ttft": JsonMetricResult(unit="ms", avg=100.0)},
                variation_label=f"phases.profiling.concurrency={c}",
                variation_values={"phases.profiling.concurrency": c},
                trial_index=t,
            )
            for c in concurrency
            for t in range(n_trials)
        ]
        groups = _group_results_by_variation(results)

        assert len(groups) == len(concurrency)
        assert all(len(group) == n_trials for group in groups.values())
