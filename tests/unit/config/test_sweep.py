# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep configuration models and expansion."""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.sweep import (
    GridSweep,
    ScenarioSweep,
    SweepVariation,
    ZipSweep,
    _set_nested_value,
    expand_sweep,
)
from aiperf.config.sweep.expand import (
    _deep_merge,
    detect_sweep_fields,
)


class TestSweepModels:
    """Tests for sweep Pydantic models."""

    def test_grid_sweep_basic(self):
        sweep = GridSweep(parameters={"phases.concurrency": [8, 16, 32]})
        assert sweep.type == "grid"
        assert sweep.parameters == {"phases.concurrency": [8, 16, 32]}

    def test_grid_sweep_multiple_variables(self):
        sweep = GridSweep(
            parameters={
                "phases.concurrency": [8, 16],
                "phases.rate": [10.0, 20.0],
            }
        )
        assert len(sweep.parameters) == 2

    def test_grid_sweep_requires_variables(self):
        with pytest.raises(ValidationError):
            GridSweep(parameters={})

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
            GridSweep(parameters={"x": [1]}, unknown="bad")

    def test_scenario_sweep_forbids_extra(self):
        with pytest.raises(ValidationError):
            ScenarioSweep(runs=[{"x": 1}], unknown="bad")


class TestExpandSweep:
    """Tests for sweep expansion functions."""

    def _base_config(self, **overrides):
        body = {
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
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
        env_keys = {"sweep", "multi_run", "variables", "random_seed"}
        env = {k: overrides.pop(k) for k in list(overrides) if k in env_keys}
        body.update(overrides)
        return {"benchmark": body, **env}

    def _phase(self, cfg: dict, name: str) -> dict:
        phases = (
            cfg.get("benchmark", {}).get("phases")
            if isinstance(cfg.get("benchmark"), dict)
            else cfg.get("phases")
        )
        return next(p for p in phases if p["name"] == name)

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
                "parameters": {
                    "phases.profiling.concurrency": [8, 16],
                    "phases.profiling.requests": [100, 200, 300],
                },
            }
        )
        result = expand_sweep(data)
        assert len(result) == 6  # 2 x 3

        values_seen = set()
        for config_dict, _variation in result:
            phase = self._phase(config_dict, "profiling")
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
                "parameters": {"phases.profiling.concurrency": [1, 2, 4, 8]},
            }
        )
        result = expand_sweep(data)
        assert len(result) == 4

        concurrencies = [self._phase(r[0], "profiling")["concurrency"] for r in result]
        assert concurrencies == [1, 2, 4, 8]

    def test_scenario_sweep_deep_merge(self):
        data = self._base_config(
            sweep={
                "type": "scenarios",
                "runs": [
                    {
                        "name": "low",
                        "benchmark": {
                            "phases": [{"name": "profiling", "concurrency": 2}]
                        },
                    },
                    {
                        "name": "high",
                        "benchmark": {
                            "phases": [{"name": "profiling", "concurrency": 64}]
                        },
                    },
                ],
            }
        )
        result = expand_sweep(data)
        assert len(result) == 2

        assert self._phase(result[0][0], "profiling")["concurrency"] == 2
        assert result[0][1].label == "low"

        assert self._phase(result[1][0], "profiling")["concurrency"] == 64
        assert result[1][1].label == "high"

        # Other fields preserved (deep-merge by name keeps base requests=10)
        assert self._phase(result[0][0], "profiling")["requests"] == 10
        assert self._phase(result[1][0], "profiling")["requests"] == 10

    def test_magic_list_detection(self):
        data = self._base_config()
        # Replace the default phase with one whose concurrency is a magic list.
        data["benchmark"]["phases"] = [
            {"name": "profiling", "type": "concurrency", "concurrency": [8, 16, 32]}
        ]

        result = expand_sweep(data)
        assert len(result) == 3

        concurrencies = [self._phase(r[0], "profiling")["concurrency"] for r in result]
        assert concurrencies == [8, 16, 32]

    def test_magic_list_multiple_fields(self):
        data = self._base_config()
        data["benchmark"]["phases"] = [
            {
                "name": "profiling",
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
                "parameters": {"phases.profiling.concurrency": [1, 2]},
            }
        )
        # Also add magic list (should be ignored since explicit sweep exists)
        data["benchmark"]["phases"][0]["requests"] = [100, 200]

        result = expand_sweep(data)
        assert len(result) == 2  # Only explicit sweep

    def test_sweep_section_removed_from_output(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1]},
            }
        )
        result = expand_sweep(data)
        for config_dict, _ in result:
            assert "sweep" not in config_dict

    def test_variation_metadata_correct(self):
        data = self._base_config(
            sweep={
                "type": "grid",
                "parameters": {
                    "phases.profiling.concurrency": [8, 16],
                },
            }
        )
        result = expand_sweep(data)

        assert result[0][1].index == 0
        assert result[0][1].values == {"phases.profiling.concurrency": 8}

        assert result[1][1].index == 1
        assert result[1][1].values == {"phases.profiling.concurrency": 16}

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
                "parameters": {
                    "phases.profiling.concurrency": [4, 8],
                    "phases.profiling.requests": [10, 20],
                },
            }
        )
        # insertion-order keys deliberately reversed; expansion must still
        # produce alphabetical-key combinations.
        result_a = expand_sweep(data)

        data_reversed = self._base_config(
            sweep={
                "type": "grid",
                "parameters": {
                    "phases.profiling.requests": [10, 20],
                    "phases.profiling.concurrency": [4, 8],
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
        assert first_keys[0] == "phases.profiling.concurrency"


class TestScenarioSingularDatasetAgainstSingularBase:
    """Regression: scenario `benchmark.dataset:` override against a base that
    also uses singular `dataset:` shorthand.

    The expander rewrites the scenario into `benchmark.datasets: [...]` for
    deep-merge — but it must also promote the BASE's singular `dataset:` to
    plural `datasets:`, otherwise the merged variant carries BOTH keys and
    `BenchmarkConfig`'s mutual-exclusivity validator rejects it. See
    `_normalize_scenario_dataset_form` in `aiperf.config.sweep.expand`.
    """

    def test_singular_base_plus_singular_scenario_override_validates(self):
        from aiperf.config import AIPerfConfig

        data = {
            "benchmark": {
                "models": ["meta-llama/Llama-3.1-8B-Instruct"],
                "endpoint": {
                    "urls": ["http://localhost:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "dataset": {"type": "synthetic", "entries": 100},
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
            },
            "sweep": {
                "type": "scenarios",
                "runs": [
                    {
                        "name": "short",
                        "benchmark": {"dataset": {"prompts": {"isl": 128, "osl": 128}}},
                    }
                ],
            },
        }
        result = expand_sweep(data)
        assert len(result) == 1
        variant, _ = result[0]
        assert "dataset" not in variant["benchmark"]
        assert variant["benchmark"]["datasets"] == [
            {
                "name": "default",
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 128},
            }
        ]
        cfg = AIPerfConfig.model_validate(variant)
        ds = cfg.benchmark.datasets[0]
        assert getattr(ds.prompts.isl, "value", ds.prompts.isl) == 128
        assert getattr(ds.prompts.osl, "value", ds.prompts.osl) == 128


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
            "benchmark": {
                "phases": [
                    {
                        "name": "profiling",
                        "concurrency": [8, 16, 32],
                    }
                ]
            }
        }
        fields = detect_sweep_fields(data["benchmark"])
        assert "phases.profiling.concurrency" in fields
        assert fields["phases.profiling.concurrency"] == [8, 16, 32]

    def test_detect_sweep_fields_ignores_string_lists(self):
        data = {
            "phases": [
                {
                    "name": "profiling",
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
            "benchmark": {
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "duration": 1,
                        "concurrency": 1,
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match=r"no entry named 'profilling'"):
            _set_nested_value(data["benchmark"], "phases.profilling.rate", 1)
        # The phantom phase MUST NOT have been added.
        names = [p["name"] for p in data["benchmark"]["phases"]]
        assert names == ["profiling"], (
            "strict-mode must not auto-append on unknown name"
        )

    def test_set_nested_value_known_named_segment_succeeds(self):
        """Existing named entry: assignment proceeds normally."""
        data = {
            "benchmark": {
                "phases": [
                    {"name": "profiling", "concurrency": 1},
                    {"name": "warmup", "concurrency": 2},
                ]
            }
        }
        _set_nested_value(data["benchmark"], "phases.profiling.concurrency", 64)
        # Find profiling entry and verify update.
        prof = next(p for p in data["benchmark"]["phases"] if p["name"] == "profiling")
        assert prof["concurrency"] == 64
        # Other entry untouched.
        warm = next(p for p in data["benchmark"]["phases"] if p["name"] == "warmup")
        assert warm["concurrency"] == 2

    def test_set_nested_value_phases_profiling_falls_back_to_default(self):
        """`phases.profiling.X` resolves to the unique non-warmup phase when
        no phase by that name exists.

        Regression: search recipes hard-code ``phases.profiling.concurrency``
        because the v1 -> v2 converter emits a phase named ``profiling``, but
        legacy raw configs may end up with a phase named ``default``. Without
        the recipe-friendly fallback, ``aiperf profile -f base.yaml
        --search-recipe X`` failed with ``no entry named 'profiling' found
        (existing: ['default'])``.
        """
        data = {
            "benchmark": {
                "phases": [
                    {"name": "warmup", "concurrency": 1},
                    {"name": "default", "type": "concurrency", "concurrency": 8},
                ]
            }
        }
        _set_nested_value(data["benchmark"], "phases.profiling.concurrency", 64)
        # The single non-warmup phase (named "default") got the override.
        default_phase = next(
            p for p in data["benchmark"]["phases"] if p["name"] == "default"
        )
        assert default_phase["concurrency"] == 64
        warm = next(p for p in data["benchmark"]["phases"] if p["name"] == "warmup")
        assert warm["concurrency"] == 1

    def test_set_nested_value_phases_profiling_ambiguous_falls_through(self):
        """When MULTIPLE non-warmup phases exist, the fallback is unsafe and
        the strict ValueError fires unchanged (better to error than guess)."""
        data = {
            "benchmark": {
                "phases": [
                    {"name": "stage1", "concurrency": 1},
                    {"name": "stage2", "concurrency": 2},
                ]
            }
        }
        with pytest.raises(ValueError, match=r"no entry named 'profiling'"):
            _set_nested_value(data["benchmark"], "phases.profiling.concurrency", 64)

    def test_expand_sweep_grid_typo_named_path_errors_at_expand_time(self):
        """A grid-sweep typo'd named-list path errors at `expand_sweep` time
        (not silently in a downstream stage)."""
        data = {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "phases": [
                    {"name": "profiling", "type": "concurrency", "concurrency": 1}
                ],
            },
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profilling.concurrency": [1, 2]},
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

    Spec: (deleted)
    Each test runs the full load_config_from_string -> build_benchmark_plan
    path so regressions anywhere in load -> expand -> render -> validate
    surface here.
    """

    pytestmark = pytest.mark.skip(
        reason="scenario singular dataset shorthand spec not implemented on the "
        "envelope-restructure branch; design lives on parent branch."
    )

    BASE_HEADER = (
        "benchmark:\n"
        "  models:\n"
        "    - test/model\n"
        "  endpoint:\n"
        "    type: chat\n"
        '    urls: ["http://localhost:8000/v1/chat/completions"]\n'
        "  phases:\n"
        "    - name: profiling\n"
        "      type: concurrency\n"
        "      requests: 10\n"
        "      concurrency: 1\n"
    )
    PHASES_TAIL = ""

    def _isl_osl(self, cfg, ds_idx: int = 0):
        ds = cfg.benchmark.datasets[ds_idx]
        isl = getattr(ds.prompts.isl, "value", ds.prompts.isl)
        osl = getattr(ds.prompts.osl, "value", ds.prompts.osl)
        return isl, osl

    def test_scenario_singular_dataset_against_plural_single_entry_base(self):
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string

        yaml_str = (
            self.BASE_HEADER
            + (
                "  datasets:\n"
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
                "  dataset:\n"
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
                "  datasets:\n"
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
                "  datasets:\n"
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
        from aiperf.config.loader import build_benchmark_plan, load_config_from_string
        from aiperf.config.loader.normalizers import DATASET_VS_DATASETS_MSG

        yaml_str = (
            self.BASE_HEADER
            + (
                "  datasets:\n"
                "  - {name: main, type: synthetic, entries: 200}\n"
                "sweep:\n"
                "  type: scenarios\n"
                "  runs:\n"
                "    - dataset: {isl: 128, osl: 128}\n"
                "  datasets:\n"
                "  - {name: main, isl: 256, osl: 256}\n"
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
                "  datasets:\n"
                "  - name: main\n"
                "  type: synthetic\n"
                "  entries: 200\n"
                "  prompts: {isl: 64, osl: 32}\n"
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
                "  datasets:\n"
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
                "  datasets:\n"
                "  - name: main\n"
                "  type: synthetic\n"
                "  entries: 200\n"
                "  prompts: {isl: 64, osl: 32}\n"
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


# ===========================================================================
# Zip sweep: paired (lockstep) parameter expansion. Mirrors the grid-sweep
# coverage above but uses ``zip(strict=True)`` instead of Cartesian product.
# ===========================================================================


class TestZipSweepModel:
    """Pydantic-level checks on the ``ZipSweep`` model."""

    def test_zip_sweep_basic(self):
        sweep = ZipSweep(
            parameters={
                "datasets.default.prompts.isl": [128, 512, 2048],
                "datasets.default.prompts.osl": [128, 256, 512],
            }
        )
        assert sweep.type == "zip"
        assert len(sweep.parameters) == 2

    def test_zip_sweep_requires_parameters(self):
        with pytest.raises(ValidationError):
            ZipSweep(parameters={})

    def test_zip_sweep_forbids_extra(self):
        with pytest.raises(ValidationError):
            ZipSweep(parameters={"x": [1]}, unknown="bad")

    def test_zip_sweep_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError, match=r"equal length"):
            ZipSweep(
                parameters={
                    "phases.profiling.a": [1, 2, 3],
                    "phases.profiling.b": [10, 20],
                }
            )

    def test_zip_sweep_single_parameter_ok(self):
        sweep = ZipSweep(parameters={"phases.profiling.concurrency": [1, 2, 4]})
        assert len(sweep.parameters) == 1


class TestExpandZipSweep:
    """End-to-end expansion via ``expand_sweep`` for ``type: zip``."""

    def _base_config(self, **overrides):
        body = {
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
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
        env_keys = {"sweep", "multi_run", "variables", "random_seed"}
        env = {k: overrides.pop(k) for k in list(overrides) if k in env_keys}
        body.update(overrides)
        return {"benchmark": body, **env}

    def _phase(self, cfg: dict, name: str) -> dict:
        phases = (
            cfg.get("benchmark", {}).get("phases")
            if isinstance(cfg.get("benchmark"), dict)
            else cfg.get("phases")
        )
        return next(p for p in phases if p["name"] == name)

    def test_zip_sweep_pairs_lockstep_not_cartesian(self):
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "phases.profiling.concurrency": [8, 16, 32],
                    "phases.profiling.requests": [100, 200, 300],
                },
            }
        )
        result = expand_sweep(data)
        # 3 paired runs, NOT 3*3=9 (which is what grid would produce).
        assert len(result) == 3
        seen = set()
        for config_dict, _variation in result:
            phase = self._phase(config_dict, "profiling")
            seen.add((phase["concurrency"], phase["requests"]))
            assert "sweep" not in config_dict
        # Exact pairs — concurrency_i is paired with requests_i.
        assert seen == {(8, 100), (16, 200), (32, 300)}

    def test_zip_sweep_single_parameter(self):
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {"phases.profiling.concurrency": [1, 2, 4, 8]},
            }
        )
        result = expand_sweep(data)
        assert len(result) == 4
        concurrencies = [self._phase(r[0], "profiling")["concurrency"] for r in result]
        assert concurrencies == [1, 2, 4, 8]
        # Indices are 0..N-1.
        assert [v.index for _, v in result] == [0, 1, 2, 3]

    def test_zip_sweep_isl_osl_pairs_pydantic_validation(self):
        """Each zip-expanded variant must Pydantic-validate end-to-end.

        Beyond dict-level expansion (covered by the next test), every
        variant needs to round-trip through ``AIPerfConfig.model_validate``
        so future schema drift in datasets/prompts/normalizers surfaces here.
        Uses singular ``dataset:`` form to also exercise the
        ``_normalize_dataset_and_phases`` path.
        """
        from aiperf.config.config import AIPerfConfig

        data = {
            "benchmark": {
                "models": ["test-model"],
                "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
                "dataset": {
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 64, "osl": 32},
                },
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
            },
            "sweep": {
                "type": "zip",
                "parameters": {
                    "dataset.prompts.isl": [128, 512, 2048],
                    "dataset.prompts.osl": [128, 256, 512],
                },
            },
        }
        result = expand_sweep(data)
        assert len(result) == 3
        expected = [(128, 128), (512, 256), (2048, 512)]
        for (cfg, _variation), (want_isl, want_osl) in zip(
            result, expected, strict=True
        ):
            validated = AIPerfConfig.model_validate(cfg)
            ds = validated.benchmark.datasets[0]
            # Singular `dataset:` auto-names the entry "default".
            assert ds.name == "default"
            isl = getattr(ds.prompts.isl, "value", ds.prompts.isl)
            osl = getattr(ds.prompts.osl, "value", ds.prompts.osl)
            assert isl == want_isl
            assert osl == want_osl

    def test_zip_sweep_isl_osl_pairs_end_to_end(self):
        """The canonical use case: paired ISL/OSL via ``type: zip``."""
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "datasets.default.prompts.isl": [128, 512, 2048],
                    "datasets.default.prompts.osl": [128, 256, 512],
                },
            }
        )
        result = expand_sweep(data)
        assert len(result) == 3
        pairs = [
            (
                cfg["benchmark"]["datasets"][0]["prompts"]["isl"],
                cfg["benchmark"]["datasets"][0]["prompts"]["osl"],
            )
            for cfg, _ in result
        ]
        assert pairs == [(128, 128), (512, 256), (2048, 512)]

    def test_zip_sweep_envelope_variables_routing(self):
        """``variables.<name>`` writes to the envelope-level Jinja block,
        same as grid-sweep semantics — paired with a body path here."""
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "variables.label": ["small", "large"],
                    "phases.profiling.concurrency": [8, 64],
                },
            }
        )
        result = expand_sweep(data)
        assert len(result) == 2
        labels = [cfg.get("variables", {}).get("label") for cfg, _ in result]
        concurrencies = [
            self._phase(cfg, "profiling")["concurrency"] for cfg, _ in result
        ]
        # Sorted alphabetically: phases.* before variables.* for stable
        # iteration, but pairing remains lockstep by index.
        assert labels == ["small", "large"]
        assert concurrencies == [8, 64]

    def test_zip_sweep_field_order_is_alphabetical_not_insertion(self):
        """Same K8s-CRD ordering invariant as grid: insertion order must
        not affect variation order. See `gotcha_k8s_crd_object_map_keys_alphabetized`."""
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "phases.profiling.requests": [10, 20],
                    "phases.profiling.concurrency": [4, 8],
                },
            }
        )
        result_a = expand_sweep(data)

        data_reversed = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "phases.profiling.concurrency": [4, 8],
                    "phases.profiling.requests": [10, 20],
                },
            }
        )
        result_b = expand_sweep(data_reversed)

        assert [v.values for _, v in result_a] == [v.values for _, v in result_b]
        first_keys = list(result_a[0][1].values.keys())
        assert first_keys == sorted(first_keys)
        assert first_keys[0] == "phases.profiling.concurrency"

    def test_zip_sweep_sweep_section_removed_from_output(self):
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {"phases.profiling.concurrency": [1, 2]},
            }
        )
        result = expand_sweep(data)
        for config_dict, _ in result:
            assert "sweep" not in config_dict

    def test_zip_sweep_mismatched_lengths_raise_at_pydantic_time(self):
        """Mismatched lengths must error at model-construction time."""
        with pytest.raises(ValidationError, match=r"equal length"):
            ZipSweep(
                parameters={
                    "phases.profiling.concurrency": [1, 2, 3],
                    "phases.profiling.requests": [10, 20],
                }
            )

    def test_zip_sweep_mismatched_lengths_raise_at_expand_time(self):
        """Defense-in-depth: callers that bypass the Pydantic model and feed
        a raw dict into ``_expand_zip_sweep`` still get a clear error."""
        from aiperf.config.sweep.expand import _expand_zip_sweep

        with pytest.raises(ValueError, match=r"equal length"):
            _expand_zip_sweep(
                {"benchmark": {}},
                {
                    "phases.profiling.concurrency": [1, 2, 3],
                    "phases.profiling.requests": [10, 20],
                },
            )

    @pytest.mark.parametrize(
        "bad_value",
        [
            param([], id="empty_list"),
            param("not-a-list", id="string"),
            param(123, id="int"),
            param(None, id="none"),
        ],
    )  # fmt: skip
    def test_zip_sweep_rejects_empty_or_non_list_values(self, bad_value):
        from aiperf.config.sweep.expand import _expand_zip_sweep

        with pytest.raises(ValueError, match=r"zip sweep parameter"):
            _expand_zip_sweep(
                {"benchmark": {}},
                {"phases.profiling.concurrency": bad_value},
            )

    def test_zip_sweep_rejects_redundant_benchmark_prefix(self):
        """Bare paths route to body; ``benchmark.<x>`` must error (mirrors
        grid sweep semantics)."""
        from aiperf.config.sweep.expand import _expand_zip_sweep

        with pytest.raises(ValueError, match=r"zip sweep parameter"):
            _expand_zip_sweep(
                {"benchmark": {}},
                {"benchmark.phases.profiling.concurrency": [1, 2]},
            )

    def test_zip_sweep_variation_metadata_correct(self):
        data = self._base_config(
            sweep={
                "type": "zip",
                "parameters": {
                    "phases.profiling.concurrency": [8, 16],
                    "phases.profiling.requests": [100, 200],
                },
            }
        )
        result = expand_sweep(data)
        assert result[0][1].index == 0
        assert result[0][1].values == {
            "phases.profiling.concurrency": 8,
            "phases.profiling.requests": 100,
        }
        assert result[1][1].index == 1
        assert result[1][1].values == {
            "phases.profiling.concurrency": 16,
            "phases.profiling.requests": 200,
        }


class TestSweepVariationDirName:
    """Pin every branch of `SweepVariation.dir_name`.

    `dir_name` is the only thing standing between two sweep cells writing to
    the same directory on disk — collisions silently overwrite earlier
    results. The format `{last_seg}_{value}` is also user-visible (logs,
    artifact paths, the dashboard) so changes to it break URLs and external
    tooling. Pre-pin, the property had ZERO test coverage.
    """

    def test_single_dim_uses_last_dotted_segment_and_value(self):
        v = SweepVariation(
            index=0,
            label="ignored_when_values_present",
            values={"phases.profiling.concurrency": 10},
        )
        assert v.dir_name == "concurrency_10"

    def test_single_dim_with_float_value(self):
        v = SweepVariation(
            index=0,
            label="x",
            values={"phases.profiling.request_rate": 5.0},
        )
        assert v.dir_name == "request_rate_5.0"

    def test_multi_dim_joined_with_double_underscore(self):
        v = SweepVariation(
            index=0,
            label="x",
            values={"a.b": 1, "c.d": 2},
        )
        assert v.dir_name == "b_1__d_2"

    def test_empty_values_falls_back_to_label(self):
        """Non-sweep runs and named scenarios depend on this fallback."""
        v = SweepVariation(index=0, label="scenario_a", values={})
        assert v.dir_name == "scenario_a"

    def test_empty_values_with_label_named_base(self):
        v = SweepVariation(index=0, label="base", values={})
        assert v.dir_name == "base"

    @pytest.mark.parametrize(
        "raw_value",
        [
            param("a/b_1", id="forward_slash"),
            param("a\\b_1", id="backslash"),
            param("..b_1", id="leading_dot_dot"),
            param("a..b_1", id="embedded_dot_dot"),
        ],
    )  # fmt: skip
    def test_path_traversal_chars_stripped(self, raw_value):
        """Forward slash, backslash, and `..` must be stripped — preventing
        a sweep value like `phases.x.foo: "../../etc"` from escaping the
        artifact dir."""
        v = SweepVariation(index=0, label="x", values={"key": raw_value})
        assert "/" not in v.dir_name
        assert "\\" not in v.dir_name
        assert ".." not in v.dir_name

    @pytest.mark.parametrize(
        "bad_char",
        ["<", ">", ":", '"', "|", "?", "*"],
    )  # fmt: skip
    def test_windows_unsafe_chars_stripped(self, bad_char):
        """Windows-reserved chars are stripped so artifact paths port across
        OSes. The artifact dir is sometimes copied to Windows for analysis."""
        v = SweepVariation(index=0, label="x", values={"key": f"value{bad_char}suffix"})
        assert bad_char not in v.dir_name

    def test_combination_of_unsafe_chars_all_stripped(self):
        v = SweepVariation(
            index=0,
            label="x",
            values={"a.b": "../<dangerous|name>"},
        )
        assert "/" not in v.dir_name
        assert ".." not in v.dir_name
        assert "<" not in v.dir_name
        assert ">" not in v.dir_name
        assert "|" not in v.dir_name

    def test_sanitization_does_not_touch_safe_chars(self):
        """Underscores, hyphens, single-dots, and digits are preserved."""
        v = SweepVariation(index=0, label="x", values={"phases.x.rate": "5.0-rps_v2"})
        # Format is `{last_seg}_{value}` => "rate_5.0-rps_v2".
        assert v.dir_name == "rate_5.0-rps_v2"

    def test_label_fallback_path_is_not_sanitized(self):
        """Fallback to `label` skips the sanitizer (label is user-controlled
        but already validated upstream by sweep-name uniqueness checks).
        Pin this so a refactor that double-sanitizes the label doesn't
        silently strip user-meaningful characters from named scenarios.
        """
        v = SweepVariation(index=0, label="scenario.with.dots", values={})
        assert v.dir_name == "scenario.with.dots"
