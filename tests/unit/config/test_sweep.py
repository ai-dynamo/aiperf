# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep configuration models and expansion."""

import pytest
from pydantic import ValidationError

from aiperf.config.sweep import (
    GridSweep,
    ScenarioSweep,
    SweepVariation,
    _deep_merge,
    _set_nested_value,
    detect_sweep_fields,
    expand_sweep,
)


class TestSweepModels:
    """Tests for sweep Pydantic models."""

    def test_grid_sweep_basic(self):
        sweep = GridSweep(variables={"phases.concurrency": [8, 16, 32]})
        assert sweep.type == "grid"
        assert sweep.variables == {"phases.concurrency": [8, 16, 32]}

    def test_grid_sweep_multiple_variables(self):
        sweep = GridSweep(
            variables={
                "phases.concurrency": [8, 16],
                "phases.rate": [10.0, 20.0],
            }
        )
        assert len(sweep.variables) == 2

    def test_grid_sweep_requires_variables(self):
        with pytest.raises(ValidationError):
            GridSweep(variables={})

    def test_scenario_sweep_basic(self):
        sweep = ScenarioSweep(runs=[{"phases": {"concurrency": 8}}])
        assert sweep.type == "scenarios"
        assert len(sweep.runs) == 1

    def test_scenario_sweep_requires_runs(self):
        with pytest.raises(ValidationError):
            ScenarioSweep(runs=[])

    def test_sweep_variation_model(self):
        v = SweepVariation(
            index=0, label="concurrency=8", values={"phases.concurrency": 8}
        )
        assert v.index == 0
        assert v.label == "concurrency=8"
        assert v.values == {"phases.concurrency": 8}

    def test_grid_sweep_forbids_extra(self):
        with pytest.raises(ValidationError):
            GridSweep(variables={"x": [1]}, unknown="bad")

    def test_scenario_sweep_forbids_extra(self):
        with pytest.raises(ValidationError):
            ScenarioSweep(runs=[{"x": 1}], unknown="bad")


class TestExpandSweep:
    """Tests for sweep expansion functions."""

    def _base_config(self, **overrides):
        base = {
            "models": ["test-model"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": {
                "default": {
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
            "phases": {
                "default": {
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            },
        }
        base.update(overrides)
        return base

    def test_no_sweep_returns_single(self):
        data = self._base_config()
        result = expand_sweep(data)
        assert len(result) == 1
        config_dict, variation = result[0]
        assert variation.index == 0
        assert variation.label == "base"
        assert "sweep" not in config_dict

    def test_grid_sweep_cartesian_product(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "variables": {
                    "phases.default.concurrency": [8, 16],
                    "phases.default.requests": [100, 200, 300],
                },
            }
        )
        result = expand_sweep(data)
        assert len(result) == 6  # 2 x 3

        values_seen = set()
        for config_dict, _variation in result:
            conc = config_dict["phases"]["default"]["concurrency"]
            reqs = config_dict["phases"]["default"]["requests"]
            values_seen.add((conc, reqs))
            assert "sweep" not in config_dict

        assert values_seen == {
            (8, 100),
            (8, 200),
            (8, 300),
            (16, 100),
            (16, 200),
            (16, 300),
        }

    def test_grid_sweep_single_variable(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [1, 2, 4, 8]},
            }
        )
        result = expand_sweep(data)
        assert len(result) == 4

        concurrencies = [r[0]["phases"]["default"]["concurrency"] for r in result]
        assert concurrencies == [1, 2, 4, 8]

    def test_scenario_sweep_deep_merge(self):
        data = self._base_config(
            sweep={
                "type": "scenarios",
                "runs": [
                    {"name": "low", "phases": {"default": {"concurrency": 2}}},
                    {"name": "high", "phases": {"default": {"concurrency": 64}}},
                ],
            }
        )
        result = expand_sweep(data)
        assert len(result) == 2

        assert result[0][0]["phases"]["default"]["concurrency"] == 2
        assert result[0][1].label == "low"

        assert result[1][0]["phases"]["default"]["concurrency"] == 64
        assert result[1][1].label == "high"

        # Other fields preserved
        assert result[0][0]["phases"]["default"]["requests"] == 10
        assert result[1][0]["phases"]["default"]["requests"] == 10

    def test_magic_list_detection(self):
        data = self._base_config()
        data["phases"]["default"]["concurrency"] = [8, 16, 32]

        result = expand_sweep(data)
        assert len(result) == 3

        concurrencies = [r[0]["phases"]["default"]["concurrency"] for r in result]
        assert concurrencies == [8, 16, 32]

    def test_magic_list_multiple_fields(self):
        data = self._base_config()
        data["phases"]["default"]["concurrency"] = [8, 16]
        data["phases"]["default"]["requests"] = [100, 200]

        result = expand_sweep(data)
        assert len(result) == 4  # Cartesian product

    def test_explicit_sweep_takes_precedence_over_magic(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "variables": {"phases.default.concurrency": [1, 2]},
            }
        )
        # Also add magic list (should be ignored since explicit sweep exists)
        data["phases"]["default"]["requests"] = [100, 200]

        result = expand_sweep(data)
        assert len(result) == 2  # Only explicit sweep

    def test_sweep_section_removed_from_output(self):
        data = self._base_config(
            sweep={"type": "grid", "variables": {"phases.default.concurrency": [1]}}
        )
        result = expand_sweep(data)
        for config_dict, _ in result:
            assert "sweep" not in config_dict

    def test_variation_metadata_correct(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "variables": {
                    "phases.default.concurrency": [8, 16],
                },
            }
        )
        result = expand_sweep(data)

        assert result[0][1].index == 0
        assert result[0][1].values == {"phases.default.concurrency": 8}

        assert result[1][1].index == 1
        assert result[1][1].values == {"phases.default.concurrency": 16}

    def test_sweep_none_returns_single(self):
        data = self._base_config(sweep=None)
        result = expand_sweep(data)
        assert len(result) == 1


class TestHelpers:
    """Tests for helper functions."""

    def test_set_nested_value_simple(self):
        data = {"a": {"b": 1}}
        _set_nested_value(data, "a.b", 2)
        assert data["a"]["b"] == 2

    def test_set_nested_value_creates_intermediates(self):
        data = {}
        _set_nested_value(data, "a.b.c", 42)
        assert data["a"]["b"]["c"] == 42

    def test_set_nested_value_top_level(self):
        data = {"x": 1}
        _set_nested_value(data, "x", 2)
        assert data["x"] == 2

    def test_deep_merge_basic(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": {"c": 2, "d": 3}}

    def test_deep_merge_overwrites_non_dict(self):
        base = {"a": 1}
        override = {"a": 2}
        _deep_merge(base, override)
        assert base["a"] == 2

    def test_detect_sweep_fields_finds_numeric_lists(self):
        data = {
            "phases": {
                "default": {
                    "concurrency": [8, 16, 32],
                    "name": "test",
                }
            }
        }
        fields = detect_sweep_fields(data)
        assert "phases.default.concurrency" in fields
        assert fields["phases.default.concurrency"] == [8, 16, 32]

    def test_detect_sweep_fields_ignores_string_lists(self):
        data = {
            "phases": {
                "default": {
                    "concurrency": ["a", "b"],
                }
            }
        }
        fields = detect_sweep_fields(data)
        assert len(fields) == 0

    def test_detect_sweep_fields_ignores_non_sweep_keys(self):
        data = {
            "models": [1, 2, 3],
            "endpoint": {"urls": [1, 2]},
        }
        fields = detect_sweep_fields(data)
        assert len(fields) == 0
