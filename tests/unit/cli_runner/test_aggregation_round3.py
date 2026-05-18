# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-3 multi-run aggregation regression tests.

Covers fixes for the round-2 adversarial findings on multi-run/sweep
aggregation correctness:

  * R2-H4: QMC-collision cells no longer pool into one aggregate dir.
  * R2-M5: NaN samples don't poison sibling metric aggregations.
  * R2-M7: ``BenchmarkPlan.trials`` and ``MultiRunConfig.num_runs`` caps
    stay aligned, surfaced via cross-validator.
  * R2-M8: ``AIPERF_RAISE_ON_CALLBACK_ERROR`` is read through the
    Pydantic Settings registry (Field on ``_CLIRunnerSettings``) rather
    than via raw ``os.environ.get``, picking up Pydantic's bool coercion.
  * R2-L7: ``zip(plan.configs, plan.variations)`` raises on length
    mismatch instead of silently truncating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.cli_runner._sweep_aggregate import (
    _aggregate_group_to_stats,
    _group_results_by_variation,
)
from aiperf.common.environment import _CLIRunnerSettings
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.sweep.multi_run import MultiRunConfig
from aiperf.orchestrator.models import RunResult


def _result(
    label: str,
    *,
    variation_label: str,
    variation_values: dict[str, Any],
    metrics: dict[str, JsonMetricResult] | None = None,
    success: bool = True,
) -> RunResult:
    return RunResult(
        label=label,
        success=success,
        summary_metrics=metrics or {},
        variation_label=variation_label,
        variation_values=variation_values,
    )


# =============================================================================
# R2-H4: QMC-collision cells get distinct aggregate keys
# =============================================================================


def test_qmc_collision_cells_get_distinct_aggregate_dirs() -> None:
    """Sobol over int dims that collides on ``values`` keeps distinct cells.

    Repro from the round-2 adversarial report: Sobol over ``lo=1, hi=4,
    kind=int, samples=8`` produced 4 unique values from 8 distinct cells.
    Pre-fix: one aggregate dir for all 3 cells that mapped to concurrency=3,
    inflated ``num_profile_runs=3``. Post-fix: each cell keys distinctly.
    """
    # Reproduce the collision: 3 distinct sobol cells map to concurrency=3.
    results = [
        _result(
            "sobol_0001-trial0",
            variation_label="sobol_0001",
            variation_values={"concurrency": 3},
        ),
        _result(
            "sobol_0002-trial0",
            variation_label="sobol_0002",
            variation_values={"concurrency": 3},
        ),
        _result(
            "sobol_0006-trial0",
            variation_label="sobol_0006",
            variation_values={"concurrency": 3},
        ),
    ]

    groups = _group_results_by_variation(results)

    # Three distinct cells -> three distinct groups, each with exactly one run.
    assert len(groups) == 3, (
        f"QMC-collision cells were pooled by values; expected 3 groups, got "
        f"{len(groups)}: {[k for k in groups]}"
    )
    for group in groups.values():
        assert len(group) == 1


def test_grid_sweep_distinct_values_still_group_per_label() -> None:
    """Grid sweep with unique values still groups trials of the same cell.

    Sanity check: the new keying must NOT split trials of the same
    variation cell — pooling within a cell is the multi-run design.
    """
    results = [
        _result(
            f"sobol_0001-t{i}",
            variation_label="sobol_0001",
            variation_values={"concurrency": 4},
        )
        for i in range(3)
    ]
    groups = _group_results_by_variation(results)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 3


# =============================================================================
# R2-M5: NaN-safe stats
# =============================================================================


def test_aggregator_nan_in_one_trial_does_not_poison_metric() -> None:
    """One NaN sample should not poison the aggregated mean across trials.

    Three trials at the same cell; one trial reports NaN for metric X.
    Pre-fix: ``np.mean([10, NaN, 20]) -> NaN`` and the JSON exporter
    silently coerces NaN to ``null``. Post-fix: NaN is filtered before
    aggregation, mean is computed from the remaining 2 finite samples,
    and other metrics in the same trial are unaffected.
    """
    nan = float("nan")
    results = [
        _result(
            "trial-0",
            variation_label="cell_0",
            variation_values={"concurrency": 2},
            metrics={
                "throughput": JsonMetricResult(unit="rps", avg=10.0),
                "latency": JsonMetricResult(unit="ms", avg=50.0),
            },
        ),
        _result(
            "trial-1",
            variation_label="cell_0",
            variation_values={"concurrency": 2},
            metrics={
                "throughput": JsonMetricResult(unit="rps", avg=nan),
                "latency": JsonMetricResult(unit="ms", avg=55.0),
            },
        ),
        _result(
            "trial-2",
            variation_label="cell_0",
            variation_values={"concurrency": 2},
            metrics={
                "throughput": JsonMetricResult(unit="rps", avg=20.0),
                "latency": JsonMetricResult(unit="ms", avg=60.0),
            },
        ),
    ]

    stats = _aggregate_group_to_stats(results, confidence_level=0.95)
    assert stats is not None

    # Throughput: 2 finite samples (10, 20). Mean MUST be 15, not NaN.
    throughput = stats["throughput_avg"]
    assert throughput["mean"] == pytest.approx(15.0), (
        f"NaN poisoned throughput mean: {throughput['mean']!r}"
    )

    # Latency: all 3 trials finite — should aggregate normally.
    latency = stats["latency_avg"]
    assert latency["mean"] == pytest.approx(55.0)


# =============================================================================
# R2-M7: trials cap == num_runs cap (cross-validator)
# =============================================================================


def test_benchmark_plan_trials_cap_matches_num_runs_cap() -> None:
    """The two caps are aligned — both at le=10."""
    from aiperf.config.resolution.plan import BenchmarkPlan

    trials_field = BenchmarkPlan.model_fields["trials"]
    num_runs_field = MultiRunConfig.model_fields["num_runs"]

    def _le(field: Any) -> int | None:
        for entry in field.metadata:
            value = getattr(entry, "le", None)
            if value is not None:
                return int(value)
        return None

    trials_cap = _le(trials_field)
    num_runs_cap = _le(num_runs_field)

    assert trials_cap is not None and num_runs_cap is not None
    assert trials_cap == num_runs_cap, (
        f"Cap drift: trials.le={trials_cap}, num_runs.le={num_runs_cap}. "
        f"Either the validator silently widened or the schema diverged."
    )


def test_multi_run_num_runs_accepts_up_to_cap() -> None:
    """``num_runs=10`` validates; ``num_runs=11`` is rejected."""
    cfg = MultiRunConfig(num_runs=10)
    assert cfg.num_runs == 10

    with pytest.raises(ValidationError):
        MultiRunConfig(num_runs=11)


def test_multi_run_cooldown_seconds_capped_at_24h() -> None:
    """A typo like ``cooldown_seconds=1e18`` no longer sneaks past validation."""
    MultiRunConfig(cooldown_seconds=86400)
    with pytest.raises(ValidationError):
        MultiRunConfig(cooldown_seconds=86401)


# =============================================================================
# R2-M8: RAISE_ON_CALLBACK_ERROR via Pydantic Settings
# =============================================================================


@pytest.mark.parametrize(
    "value,expected",
    [
        param("1", True, id="one"),
        param("true", True, id="true_lower"),
        param("True", True, id="true_title"),
        param("yes", True, id="yes"),
        param("on", True, id="on"),
        param("0", False, id="zero"),
        param("false", False, id="false"),
        param("", False, id="empty"),
    ],
)  # fmt: skip
def test_raise_on_callback_error_via_pydantic_settings(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Pydantic Settings reads the env var with full bool coercion.

    Pre-fix: ``cli_runner.py`` read ``os.environ.get`` directly with a
    hand-rolled list ``("1", "true", "yes")`` — diverged from Pydantic
    (which also accepts ``on``, ``True``, etc.). Post-fix: the Field
    fires through ``_CLIRunnerSettings`` so all coerced bool spellings
    are honored consistently.
    """
    if value:
        monkeypatch.setenv("AIPERF_RAISE_ON_CALLBACK_ERROR", value)
    else:
        monkeypatch.delenv("AIPERF_RAISE_ON_CALLBACK_ERROR", raising=False)

    settings = _CLIRunnerSettings()
    assert settings.RAISE_ON_CALLBACK_ERROR is expected


def test_raise_on_callback_error_invalid_value_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid bool spellings raise rather than silently default-False.

    Pre-fix: ``=invalid`` silently fell out of the ``("1","true","yes")``
    membership check and defaulted to False. Post-fix: Pydantic raises
    ``ValidationError`` so the typo is surfaced rather than swallowed.
    """
    monkeypatch.setenv("AIPERF_RAISE_ON_CALLBACK_ERROR", "invalid")
    with pytest.raises(ValidationError):
        _CLIRunnerSettings()


def test_invoke_callbacks_reads_setting_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_invoke_callbacks`` re-reads the setting on each call.

    Regression guard: if the setting were instantiated at import-time,
    ``monkeypatch.setenv`` in tests would not be visible. Verify by
    flipping the env var between calls.
    """
    from aiperf.cli_runner import CompletedRun
    from aiperf.cli_runner._callbacks import _invoke_callbacks

    logger = MagicMock()

    def boom(_completed: CompletedRun) -> None:
        raise RuntimeError("callback raised")

    completed = CompletedRun(artifact_dir=Path("/tmp"))

    monkeypatch.delenv("AIPERF_RAISE_ON_CALLBACK_ERROR", raising=False)
    code = _invoke_callbacks([boom], completed, exit_code=0, logger=logger)
    assert code == 1  # default-False path: exit code elevated, no re-raise

    monkeypatch.setenv("AIPERF_RAISE_ON_CALLBACK_ERROR", "true")
    with pytest.raises(RuntimeError, match="callback raised"):
        _invoke_callbacks([boom], completed, exit_code=0, logger=logger)


# =============================================================================
# R2-L7: strict=True zip surfaces orchestrator config/variation drift
# =============================================================================


def test_strict_zip_in_aggregator_raises_on_length_mismatch() -> None:
    """Orchestrator's per-variation iteration zips with strict=True.

    Direct call to the iteration is hard to fixture; instead verify the
    flag is set in the source so a future refactor that flips it back to
    ``strict=False`` breaks this test.
    """
    import inspect

    from aiperf.orchestrator import orchestrator as orch_mod

    source = inspect.getsource(orch_mod)
    # Both sweep iteration sites (variations-outer and trials-inner) should
    # be strict=True so a config/variation length mismatch surfaces as
    # ValueError instead of being silently truncated.
    assert source.count("zip(plan.configs, plan.variations, strict=True)") >= 2, (
        "Expected both per-variation zip sites to use strict=True; "
        "a strict=False slipped back in."
    )
    assert "zip(plan.configs, plan.variations, strict=False)" not in source, (
        "strict=False zip(plan.configs, plan.variations) reintroduced; "
        "see round-2 R2-L7."
    )
