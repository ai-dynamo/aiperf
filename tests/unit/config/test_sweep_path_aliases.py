# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bare-name sweep path sugar (`_SWEEP_PATH_ALIASES`).

Covers the four sweep surfaces that share `_validate_dotted_path`: grid,
zip, QMC (`SamplingDimension`), and adaptive (`SearchSpaceDimension`).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.loader.dotted_path import (
    _SWEEP_PATH_ALIASES,
    _resolve_path_alias,
    _validate_dotted_path,
)
from aiperf.config.sweep import expand_sweep
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.config.sweep.sampling import SamplingDimension


class TestResolvePathAlias:
    """Pure unit tests on `_resolve_path_alias`."""

    @pytest.mark.parametrize(
        "alias,expanded",
        sorted(_SWEEP_PATH_ALIASES.items()),
    )
    def test_every_alias_resolves_to_phases_profiling(self, alias, expanded):
        assert _resolve_path_alias(alias) == expanded
        assert expanded.startswith("phases.profiling.")

    @pytest.mark.parametrize(
        "path",
        [
            param("phases.profiling.concurrency", id="already-canonical"),
            param("phases.warmup.requests", id="warmup-path-untouched"),
            param("concurrency.value", id="compound-not-rewritten"),
            param("rate.foo", id="compound-rate-not-rewritten"),
            param("endpoint.streaming", id="non-alias-unchanged"),
            param("variables.my_var", id="envelope-path-unchanged"),
            param("", id="empty-string-passes-through"),
        ],
    )
    def test_non_alias_input_returns_input(self, path):
        assert _resolve_path_alias(path) == path

    def test_validate_dotted_path_resolves_alias(self):
        assert _validate_dotted_path("concurrency") == "phases.profiling.concurrency"
        assert _validate_dotted_path("rate") == "phases.profiling.rate"

    def test_validate_dotted_path_passes_non_alias_unchanged(self):
        assert (
            _validate_dotted_path("phases.warmup.requests") == "phases.warmup.requests"
        )

    def test_validate_dotted_path_still_rejects_benchmark_prefix(self):
        # Alias resolution must not bypass existing rejection rules.
        with pytest.raises(ValueError, match="redundant 'benchmark.' prefix"):
            _validate_dotted_path("benchmark.phases.profiling.concurrency")


class TestGridSweepSugar:
    """End-to-end: sugar key in `GridSweep.parameters` writes the canonical path."""

    def _base(self):
        return {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 64, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
        }

    def test_bare_concurrency_alias_expands_in_grid(self):
        data = self._base()
        data["sweep"] = {"type": "grid", "parameters": {"concurrency": [4, 8, 16]}}
        result = expand_sweep(data)
        assert len(result) == 3
        concurrencies = [v[0]["benchmark"]["phases"][0]["concurrency"] for v in result]
        assert sorted(concurrencies) == [4, 8, 16]
        # Values dict and label use canonical resolved path, not sugar.
        for _, variation in result:
            assert "phases.profiling.concurrency" in variation.values
            assert "concurrency" not in {k for k in variation.values if "." not in k}
            assert variation.label.startswith("phases.profiling.concurrency=")

    def test_bare_alias_against_default_phase_shorthand(self):
        # YAML `phases: {type: concurrency}` shorthand emits a phase named
        # `default`; the existing recipe-fallback in
        # `_find_phase_or_recipe_alias` resolves `profiling` to the unique
        # non-warmup phase. Combined with sugar, `concurrency: [..]` should
        # work against shorthand-defined phases too.
        data = self._base()
        data["benchmark"]["phases"] = [
            {"name": "default", "type": "concurrency", "requests": 1, "concurrency": 1}
        ]
        data["sweep"] = {"type": "grid", "parameters": {"concurrency": [4, 8]}}
        result = expand_sweep(data)
        assert len(result) == 2
        concurrencies = [v[0]["benchmark"]["phases"][0]["concurrency"] for v in result]
        assert sorted(concurrencies) == [4, 8]

    def test_mixed_sugar_and_full_path_in_grid(self):
        data = self._base()
        # Warmup phase added so phases.warmup.requests has a target.
        data["benchmark"]["phases"].insert(
            0,
            {"name": "warmup", "type": "concurrency", "requests": 1, "concurrency": 1},
        )
        data["sweep"] = {
            "type": "grid",
            "parameters": {
                "concurrency": [2, 4],
                "phases.warmup.requests": [50, 100],
            },
        }
        result = expand_sweep(data)
        assert len(result) == 4  # 2 x 2 grid
        for variant, variation in result:
            assert "phases.profiling.concurrency" in variation.values
            assert "phases.warmup.requests" in variation.values
            assert variation.values["phases.profiling.concurrency"] in (2, 4)
            assert variation.values["phases.warmup.requests"] in (50, 100)
            assert (
                variant["benchmark"]["phases"][1]["concurrency"]
                == variation.values["phases.profiling.concurrency"]
            )

    def test_duplicate_sugar_and_full_path_rejected(self):
        data = self._base()
        data["sweep"] = {
            "type": "grid",
            "parameters": {
                "concurrency": [4, 8],
                "phases.profiling.concurrency": [16, 32],
            },
        }
        with pytest.raises(ValueError, match="already a parameter|Pick one spelling"):
            expand_sweep(data)


class TestZipSweepSugar:
    """Sugar works on zip-sweep parameters identically to grid."""

    def _base(self):
        return {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 64, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 1,
                        "concurrency": 1,
                    }
                ],
            },
        }

    def test_bare_aliases_zip_lockstep(self):
        data = self._base()
        data["sweep"] = {
            "type": "zip",
            "parameters": {
                "concurrency": [4, 8, 16],
                "requests": [100, 200, 400],
            },
        }
        result = expand_sweep(data)
        assert len(result) == 3
        pairs = [
            (
                v.values["phases.profiling.concurrency"],
                v.values["phases.profiling.requests"],
            )
            for _, v in result
        ]
        assert sorted(pairs) == [(4, 100), (8, 200), (16, 400)]


class TestDimensionPathSugar:
    """`SamplingDimension` (QMC) and `SearchSpaceDimension` (adaptive) get sugar
    for free via the shared `_validate_dotted_path` call.
    """

    def test_sampling_dimension_resolves_alias(self):
        d = SamplingDimension(path="concurrency", lo=1, hi=64, kind="int")
        assert d.path == "phases.profiling.concurrency"

    def test_sampling_dimension_passes_through_full_path(self):
        d = SamplingDimension(path="phases.warmup.requests", lo=1, hi=100, kind="int")
        assert d.path == "phases.warmup.requests"

    def test_search_space_dimension_resolves_alias(self):
        d = SearchSpaceDimension(path="rate", lo=1.0, hi=100.0, kind="real")
        assert d.path == "phases.profiling.rate"
