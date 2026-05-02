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
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
        base.update(overrides)
        return base

    def _phase(self, cfg: dict, name: str) -> dict:
        return next(p for p in cfg["phases"] if p["name"] == name)

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
            phase = self._phase(config_dict, "default")
            values_seen.add((phase["concurrency"], phase["requests"]))
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

        concurrencies = [self._phase(r[0], "default")["concurrency"] for r in result]
        assert concurrencies == [1, 2, 4, 8]

    def test_scenario_sweep_deep_merge(self):
        data = self._base_config(
            sweep={
                "type": "scenarios",
                "runs": [
                    {"name": "low", "phases": [{"name": "default", "concurrency": 2}]},
                    {
                        "name": "high",
                        "phases": [{"name": "default", "concurrency": 64}],
                    },
                ],
            }
        )
        result = expand_sweep(data)
        assert len(result) == 2

        assert self._phase(result[0][0], "default")["concurrency"] == 2
        assert result[0][1].label == "low"

        assert self._phase(result[1][0], "default")["concurrency"] == 64
        assert result[1][1].label == "high"

        # Other fields preserved (deep-merge by name keeps base requests=10)
        assert self._phase(result[0][0], "default")["requests"] == 10
        assert self._phase(result[1][0], "default")["requests"] == 10

    def test_magic_list_detection(self):
        data = self._base_config()
        # Replace the default phase with one whose concurrency is a magic list.
        data["phases"] = [
            {"name": "default", "type": "concurrency", "concurrency": [8, 16, 32]}
        ]

        result = expand_sweep(data)
        assert len(result) == 3

        concurrencies = [self._phase(r[0], "default")["concurrency"] for r in result]
        assert concurrencies == [8, 16, 32]

    def test_magic_list_multiple_fields(self):
        data = self._base_config()
        data["phases"] = [
            {
                "name": "default",
                "type": "concurrency",
                "concurrency": [8, 16],
                "requests": [100, 200],
            }
        ]

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
        data["phases"][0]["requests"] = [100, 200]

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

    def test_grid_sweep_field_order_is_alphabetical_not_insertion(self):
        """Grid sweep variation order must be deterministic across CR storage.

        K8s apiserver alphabetizes object-typed map keys at storage (CRD
        `additionalProperties` schemas), so a Python dict's insertion order
        on submit does not survive a re-read. We sort field names so child
        names line up between submit and resume — letting the operator
        idempotently reconcile after a restart. This test pins that
        contract: insertion order `(z, a)` must produce variations whose
        `values` dicts iterate `(a=…, z=…)`.
        """
        data = self._base_config(
            sweep={
                "type": "grid",
                "variables": {
                    "phases.default.concurrency": [4, 8],
                    "phases.default.requests": [10, 20],
                },
            }
        )
        # insertion-order keys deliberately reversed; expansion must still
        # produce alphabetical-key combinations.
        result_a = expand_sweep(data)

        data_reversed = self._base_config(
            sweep={
                "type": "grid",
                "variables": {
                    "phases.default.requests": [10, 20],
                    "phases.default.concurrency": [4, 8],
                },
            }
        )
        result_b = expand_sweep(data_reversed)

        # Same expansions regardless of submit-time dict order.
        assert [v.values for _, v in result_a] == [v.values for _, v in result_b]
        # And the first variation's keys iterate alphabetically.
        first_keys = list(result_a[0][1].values.keys())
        assert first_keys == sorted(first_keys)
        # Specifically: concurrency before requests.
        assert first_keys[0] == "phases.default.concurrency"


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
            "phases": [
                {
                    "name": "default",
                    "concurrency": [8, 16, 32],
                }
            ]
        }
        fields = detect_sweep_fields(data)
        assert "phases.default.concurrency" in fields
        assert fields["phases.default.concurrency"] == [8, 16, 32]

    def test_detect_sweep_fields_ignores_string_lists(self):
        data = {
            "phases": [
                {
                    "name": "default",
                    "concurrency": ["a", "b"],
                }
            ]
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


# ===========================================================================
# Adversarial regression-locks for second-pass fix (commit 793260d7b):
# `_set_nested_value` now raises ValueError on unknown named-list segments
# (typo trap) instead of silently auto-creating phantom entries. Scenario-
# sweep `_deep_merge` retains the auto-create semantics intentionally.
# ===========================================================================


class TestSetNestedValueStrictNamedList:
    """Lock in the strict-mode behaviour for grid/magic sweep paths."""

    def test_set_nested_value_unknown_named_segment_raises_value_error(self):
        """A typo in a phase name (`profilling` vs `profiling`) must error
        loudly rather than silently appending a phantom phase entry."""
        data = {
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ]
        }
        with pytest.raises(ValueError, match=r"no entry named 'profilling'"):
            _set_nested_value(data, "phases.profilling.rate", 1)
        # The phantom phase MUST NOT have been added.
        names = [p["name"] for p in data["phases"]]
        assert names == ["profiling"], (
            "strict-mode must not auto-append on unknown name"
        )

    def test_set_nested_value_known_named_segment_succeeds(self):
        """Existing named entry: assignment proceeds normally."""
        data = {
            "phases": [
                {"name": "profiling", "concurrency": 1},
                {"name": "warmup", "concurrency": 2},
            ]
        }
        _set_nested_value(data, "phases.profiling.concurrency", 64)
        # Find profiling entry and verify update.
        prof = next(p for p in data["phases"] if p["name"] == "profiling")
        assert prof["concurrency"] == 64
        # Other entry untouched.
        warm = next(p for p in data["phases"] if p["name"] == "warmup")
        assert warm["concurrency"] == 2

    def test_expand_sweep_grid_typo_named_path_errors_at_expand_time(self):
        """A grid-sweep typo'd named-list path errors at `expand_sweep` time
        (not silently in a downstream stage)."""
        data = {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "phases": [{"name": "profiling", "type": "concurrency", "concurrency": 1}],
            "sweep": {
                "type": "grid",
                "variables": {"phases.profilling.concurrency": [1, 2]},
            },
        }
        with pytest.raises(ValueError, match=r"no entry named 'profilling'"):
            expand_sweep(data)

    def test_deep_merge_appends_new_named_phase_entry(self):
        """Scenario-sweep deep-merge auto-appends new named entries
        (regression-lock for the intentional behaviour); only grid/magic
        paths got strict-mode."""
        base = {
            "phases": [
                {"name": "profiling", "concurrency": 1},
            ]
        }
        override = {
            "phases": [
                {"name": "warmup", "concurrency": 99},
            ]
        }
        _deep_merge(base, override)
        names = [p["name"] for p in base["phases"]]
        assert "warmup" in names, "deep_merge must auto-append the new name"
        assert "profiling" in names, "deep_merge must keep the existing name"


class TestScenarioSingularDatasetShorthand:
    """Tests for the singular `dataset:` shorthand inside scenario sweep runs.

    Spec: docs/superpowers/specs/2026-05-02-scenario-sweep-singular-dataset-design.md
    Each test runs the full load_config_from_string -> build_benchmark_plan
    path so regressions anywhere in load -> expand -> render -> validate
    surface here.
    """

    BASE_HEADER = (
        "models:\n"
        "  - test/model\n"
        "endpoint:\n"
        "  type: chat\n"
        '  urls: ["http://localhost:8000/v1/chat/completions"]\n'
    )
    PHASES_TAIL = (
        "phases:\n"
        "  - name: profiling\n"
        "    type: concurrency\n"
        "    requests: 10\n"
        "    concurrency: 1\n"
    )

    def _isl_osl(self, cfg, ds_idx: int = 0):
        ds = cfg.datasets[ds_idx]
        isl = getattr(ds.prompts.isl, "value", ds.prompts.isl)
        osl = getattr(ds.prompts.osl, "value", ds.prompts.osl)
        return isl, osl

    def test_scenario_singular_dataset_against_plural_single_entry_base(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {dataset: {isl: 128, osl: 128}}\n"
                "    - {dataset: {isl: 256, osl: 256}}\n"
                "    - {dataset: {isl: 512, osl: 1024}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected = [(128, 128), (256, 256), (512, 1024)]
        for variation_cfg, (want_isl, want_osl) in zip(
            plan.configs, expected, strict=True
        ):
            assert variation_cfg.datasets[0].name == "main"
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == want_isl
            assert osl == want_osl

    def test_scenario_singular_dataset_against_singular_base_form(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "dataset:\n"
                "  name: main\n"
                "  type: synthetic\n"
                "  entries: 200\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {dataset: {isl: 128, osl: 128}}\n"
                "    - {dataset: {isl: 256, osl: 256}}\n"
                "    - {dataset: {isl: 512, osl: 1024}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        assert plan.is_sweep
        assert len(plan.configs) == 3
        expected = [(128, 128), (256, 256), (512, 1024)]
        for variation_cfg, (want_isl, want_osl) in zip(
            plan.configs, expected, strict=True
        ):
            assert variation_cfg.datasets[0].name == "main"
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == want_isl
            assert osl == want_osl

    def test_scenario_singular_against_multi_dataset_base_requires_name(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "  - {name: secondary, type: synthetic, entries: 100}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {dataset: {isl: 128, osl: 128}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        with pytest.raises((ValueError, ValidationError)) as exc_info:
            build_benchmark_plan(cfg)
        msg = str(exc_info.value)
        assert "[0]" in msg
        assert "main" in msg
        assert "secondary" in msg

    def test_scenario_singular_with_explicit_name_against_multi_base(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "  - {name: secondary, type: synthetic, entries: 100}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {dataset: {name: secondary, isl: 128, osl: 128}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        variation_cfg = plan.configs[0]
        names = [d.name for d in variation_cfg.datasets]
        assert "main" in names
        assert "secondary" in names

        secondary = next(d for d in variation_cfg.datasets if d.name == "secondary")
        sec_isl = getattr(secondary.prompts.isl, "value", secondary.prompts.isl)
        sec_osl = getattr(secondary.prompts.osl, "value", secondary.prompts.osl)
        assert sec_isl == 128
        assert sec_osl == 128

        main = next(d for d in variation_cfg.datasets if d.name == "main")
        main_isl_attr = getattr(main.prompts, "isl", None)
        main_isl = (
            getattr(main_isl_attr, "value", main_isl_attr)
            if main_isl_attr is not None
            else None
        )
        assert main_isl != 128, (
            "explicit-name scenario must not bleed isl into the unrelated 'main' entry"
        )

    def test_scenario_singular_and_plural_in_same_run_errors(self):
        from aiperf.config._benchmark_normalizers import DATASET_VS_DATASETS_MSG
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - dataset: {isl: 128, osl: 128}\n"
                "      datasets:\n"
                "        - {name: main, isl: 256, osl: 256}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        with pytest.raises((ValueError, ValidationError)) as exc_info:
            build_benchmark_plan(cfg)
        msg = str(exc_info.value)
        assert "[0]" in msg
        assert DATASET_VS_DATASETS_MSG in msg

    def test_scenario_no_dataset_keys_unchanged(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts: {isl: 64, osl: 32}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {phases: [{name: profiling, concurrency: 4}]}\n"
                "    - {phases: [{name: profiling, concurrency: 8}]}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        assert len(plan.configs) == 2
        for variation_cfg in plan.configs:
            assert len(variation_cfg.datasets) == 1
            assert variation_cfg.datasets[0].name == "main"
            isl, osl = self._isl_osl(variation_cfg)
            assert isl == 64
            assert osl == 32

    def test_scenario_singular_dataset_preserves_run_label(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {name: pair_0, dataset: {isl: 128, osl: 128}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        assert plan.variations[0].label == "pair_0"
        assert plan.configs[0].datasets[0].name == "main"
        isl, osl = self._isl_osl(plan.configs[0])
        assert isl == 128
        assert osl == 128

    def test_scenario_dataset_with_name_overrides_base_resolution(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "datasets:\n"
                "  - name: main\n"
                "    type: synthetic\n"
                "    entries: 200\n"
                "    prompts: {isl: 64, osl: 32}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - {dataset: {name: explicit, type: synthetic, entries: 50, isl: 128, osl: 64}}\n"
            )
            + self.PHASES_TAIL
        )

        cfg = load_config_from_string(yaml_str)
        plan = build_benchmark_plan(cfg)

        names = [d.name for d in plan.configs[0].datasets]
        assert "main" in names, (
            "explicit name should not erase base 'main' entry; deep-merge "
            "appends a new named entry"
        )
        assert "explicit" in names, (
            "explicit name on scenario dataset should create a new 'explicit' entry"
        )
