# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-3 adversarial regressions for the sweep + config system.

Continues the H-series after ``test_sweep_round2_adversarial.py``:

- H15: ``_set_nested_value`` raises a clear path-aware ValueError when a
  dotted path crosses a scalar/None mid-traversal (was: opaque
  ``TypeError: 'X' object does not support item assignment``).
- H16: Grid sweep parameter keys go through the shared dotted-path
  validator -- empty/leading-dot/``..``/non-sweepable-prefix paths are
  rejected with the same messages QMC and BO use.
- H17: AdaptiveSearchSweep rejects duplicate dim paths in
  ``search_space`` (mirrors the QMC sampling-sweep dedup; would otherwise
  silently waste BO iterations on a phantom degree of freedom).
- H18: Sobol/LHS reject negative ``seed`` at validation time instead of
  surfacing scipy's opaque "expected non-negative integer" error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    LatinHypercubeSweep,
    Objective,
    SamplingDimension,
    SobolSweep,
    expand_sweep,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.config.sweep.expand import _set_nested_value
from aiperf.plugin.enums import SearchPlannerType

# -- H15: _set_nested_value clear-error fix --------------------------------


class TestH15SetNestedValueClearErrors:
    """``_set_nested_value`` used to bubble a low-level TypeError when a
    path crossed a scalar or None mid-traversal -- e.g. ``'str' object does
    not support item assignment``. The user got no path context.
    """

    def test_descend_into_str_raises_clear_error(self) -> None:
        # 4-segment path where mid-segment is a scalar -> "descend" branch.
        data = {"a": {"b": "scalar"}}
        with pytest.raises(ValueError, match=r"cannot descend into str"):
            _set_nested_value(data, "a.b.c.d", 42)

    def test_descend_into_int_raises_clear_error(self) -> None:
        data = {"a": {"b": 5}}
        with pytest.raises(ValueError, match=r"cannot descend into int"):
            _set_nested_value(data, "a.b.c.d", 42)

    def test_assign_into_scalar_at_leaf_raises_clear_error(self) -> None:
        # 3-segment path where penultimate is a scalar -> "assign" branch.
        data = {"a": {"b": "scalar"}}
        with pytest.raises(ValueError, match=r"cannot assign into str"):
            _set_nested_value(data, "a.b.c", 42)

    def test_descend_into_none_raises_clear_error(self) -> None:
        data = {"a": None}
        with pytest.raises(ValueError, match=r"cannot assign into NoneType"):
            _set_nested_value(data, "a.b", 42)

    def test_assign_into_scalar_raises_clear_error(self) -> None:
        data = {"a": "leaf"}
        with pytest.raises(ValueError, match=r"cannot assign into str"):
            _set_nested_value(data, "a.b", 42)

    def test_error_includes_failing_path(self) -> None:
        data = {"phases": [{"name": "profiling", "concurrency": 8}]}
        with pytest.raises(ValueError, match=r"phases\.profiling\.concurrency\.x"):
            _set_nested_value(data, "phases.profiling.concurrency.x", 42)


# -- H16: grid sweep parameter path validation -----------------------------


class TestH16GridParameterPathValidation:
    """Grid sweep parameter keys now go through the shared dotted-path
    validator. Before, ``parameters: {a..b: [1, 2]}`` produced a phantom
    empty-string key in the variant dict; ``random_seed`` paths were
    rejected with a different message than QMC; etc.
    """

    @pytest.mark.parametrize(
        "bad_path",
        [
            "",
            ".rate",
            "rate.",
            "phases..rate",
            "sweep.samples",
            "multi_run.num_runs",
            "random_seed",
            "benchmark.phases.profiling.rate",
        ],
    )
    def test_grid_rejects_bad_paths(self, bad_path: str) -> None:
        data = {
            "benchmark": {},
            "sweep": {"type": "grid", "parameters": {bad_path: [1, 2]}},
        }
        with pytest.raises(ValueError, match=r"grid sweep parameter"):
            expand_sweep(data)

    def test_grid_allows_variables_envelope_path(self) -> None:
        # variables.* is the documented envelope-level escape; still allowed.
        data = {
            "benchmark": {},
            "variables": {"foo": 1},
            "sweep": {"type": "grid", "parameters": {"variables.foo": [1, 2]}},
        }
        out = expand_sweep(data)
        assert len(out) == 2
        assert out[0][0]["variables"]["foo"] == 1
        assert out[1][0]["variables"]["foo"] == 2


# -- H17: AdaptiveSearchSweep duplicate dim paths --------------------------


class TestH17AdaptiveDuplicateDimPaths:
    """Two dimensions writing to the same dotted path used to validate
    fine; the BO planner would explore both axes against the same backing
    field and waste iterations on a phantom DoF.
    """

    def _obj(self) -> Objective:
        return Objective(
            metric="output_token_throughput",
            direction=OptimizationDirection.MAXIMIZE,
        )

    def _dim(self, lo: float = 1, hi: float = 64) -> SearchSpaceDimension:
        return SearchSpaceDimension(
            path="phases.profiling.concurrency",
            lo=lo,
            hi=hi,
            kind="int",
        )

    def test_duplicate_paths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique paths"):
            AdaptiveSearchSweep(
                planner=SearchPlannerType.BAYESIAN,
                search_space=[self._dim(), self._dim(lo=2, hi=128)],
                objectives=[self._obj()],
                max_iterations=5,
                n_initial_points=2,
            )

    def test_distinct_paths_allowed(self) -> None:
        sw = AdaptiveSearchSweep(
            planner=SearchPlannerType.BAYESIAN,
            search_space=[
                SearchSpaceDimension(
                    path="phases.profiling.concurrency", lo=1, hi=64, kind="int"
                ),
                SearchSpaceDimension(path="phases.profiling.rate", lo=1, hi=100),
            ],
            objectives=[self._obj()],
            max_iterations=5,
            n_initial_points=2,
        )
        assert len(sw.search_space) == 2


# -- H18: Sobol/LHS negative seed rejected ---------------------------------


class TestH18SamplingNegativeSeedRejected:
    """``seed=-1`` used to slip past validation and surface scipy's opaque
    ``ValueError: expected non-negative integer`` at sample-time.
    """

    def _dim(self) -> dict:
        return {
            "path": "phases.profiling.concurrency",
            "lo": 1,
            "hi": 64,
            "kind": "int",
        }

    def test_sobol_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            SobolSweep(
                samples=4, seed=-1, dimensions=[SamplingDimension(**self._dim())]
            )

    def test_lhs_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            LatinHypercubeSweep(
                samples=4, seed=-1, dimensions=[SamplingDimension(**self._dim())]
            )

    def test_seed_zero_allowed(self) -> None:
        sw = SobolSweep(
            samples=4, seed=0, dimensions=[SamplingDimension(**self._dim())]
        )
        assert sw.seed == 0

    def test_seed_none_allowed(self) -> None:
        sw = SobolSweep(samples=4, dimensions=[SamplingDimension(**self._dim())])
        assert sw.seed is None


# -- H19: AIPerfConfig.random_seed non-negative ----------------------------


class TestH19EnvelopeRandomSeedNonNegative:
    """``AIPerfConfig.random_seed=-1`` used to validate fine; the seed
    flowed into per-variation derivation (base + N) and on into numpy /
    scipy primitives that reject negative seeds.
    """

    def _envelope(self) -> dict:
        return {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [
                    {
                        "name": "d",
                        "type": "synthetic",
                        "entries": 1,
                        "prompts": {"isl": 1, "osl": 1},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
        }

    def test_negative_random_seed_rejected(self) -> None:
        from aiperf.config.config import AIPerfConfig

        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            AIPerfConfig.model_validate({**self._envelope(), "random_seed": -1})

    def test_zero_random_seed_allowed(self) -> None:
        from aiperf.config.config import AIPerfConfig

        cfg = AIPerfConfig.model_validate({**self._envelope(), "random_seed": 0})
        assert cfg.random_seed == 0

    def test_none_random_seed_allowed(self) -> None:
        from aiperf.config.config import AIPerfConfig

        cfg = AIPerfConfig.model_validate(self._envelope())
        assert cfg.random_seed is None
