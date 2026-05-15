# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-pipeline expand+validate coverage across sweep types.

Each test runs the production CLI flow end-to-end:
    YAML string -> load_config_from_string -> build_benchmark_plan
        -> for each variation: BenchmarkConfig is valid AND round-trippable
                              AND swept paths/template bodies materialize.

This file complements the unit-level ``expand_sweep`` tests in
``test_sweep.py`` by exercising the post-load re-render path that the
``_raw_envelope`` fix relies on. A regression in any sweep type that
either fails Pydantic validation, fails round-trip dump+reload, or fails
to propagate the swept value into the body will trip here.
"""

from __future__ import annotations

import pytest

from aiperf.config.config import BenchmarkConfig
from aiperf.config.loader import build_benchmark_plan, load_config_from_string

_BASE_BODY = """\
benchmark:
  models: [llama]
  endpoint:
    type: chat
    urls: ["http://x:8000/v1/chat/completions"]
  datasets:
    - name: default
      type: synthetic
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
"""


def _expand(yaml_str: str):
    """Load YAML and return ``(plan, variations, configs)`` for assertions."""
    config = load_config_from_string(yaml_str)
    plan = build_benchmark_plan(config)
    return plan, plan.variations, plan.configs


def _assert_each_round_trips(configs: list[BenchmarkConfig]) -> None:
    """Every variation's BenchmarkConfig must dump+reload to an equal model.

    Catches expansion paths that produce a dict Pydantic accepts at first
    sight but mutates during validation (default-fill, alias-coerce,
    discriminator narrowing) such that re-loading the dump fails or drifts.
    """
    for i, cfg in enumerate(configs):
        dumped = cfg.model_dump(mode="json", exclude_none=True)
        reloaded = BenchmarkConfig.model_validate(dumped)
        assert reloaded.model_dump(mode="json", exclude_none=True) == dumped, (
            f"variation {i} did not round-trip cleanly through dump+validate"
        )


# ---------------------------------------------------------------------------
# Grid sweep
# ---------------------------------------------------------------------------


class TestGridSweepExpandsAndValidates:
    def test_single_axis_bare_path_materializes_in_body(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: grid\n"
            "  parameters:\n"
            "    phases.profiling.concurrency: [1, 4, 16]\n" + _BASE_BODY
        )
        _, variations, configs = _expand(yaml_str)

        assert len(configs) == 3
        assert [c.phases[0].concurrency for c in configs] == [1, 4, 16]
        # Variation metadata reflects the swept path verbatim.
        assert [v.values["phases.profiling.concurrency"] for v in variations] == [
            1,
            4,
            16,
        ]
        _assert_each_round_trips(configs)

    def test_multi_axis_produces_cartesian_product(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: grid\n"
            "  parameters:\n"
            "    phases.profiling.concurrency: [1, 2]\n"
            "    phases.profiling.requests: [10, 20]\n" + _BASE_BODY
        )
        _, _, configs = _expand(yaml_str)

        # 2 x 2 = 4 variations; pairs come back in field-name-sorted order.
        assert len(configs) == 4
        materialized = sorted(
            (c.phases[0].concurrency, c.phases[0].requests) for c in configs
        )
        assert materialized == [(1, 10), (1, 20), (2, 10), (2, 20)]
        _assert_each_round_trips(configs)

    def test_variables_axis_propagates_into_jinja_body(self) -> None:
        # Pinned alongside test_variables_persist.py::test_swept_variable_*
        # so a regression that affects only the grid path (vs. zip / sobol)
        # is still caught even if the other regression test is moved.
        yaml_str = (
            "variables: {load: 100}\n"
            "sweep:\n"
            "  type: grid\n"
            "  parameters:\n"
            "    variables.load: [10, 50, 100]\n"
            "benchmark:\n"
            "  models: [llama]\n"
            "  endpoint:\n"
            "    type: chat\n"
            "    urls: ['http://x:8000/v1/chat/completions']\n"
            "  datasets:\n"
            "    - name: default\n"
            "      type: synthetic\n"
            "  phases:\n"
            "    - name: profiling\n"
            "      type: concurrency\n"
            "      requests: '{{ load * 5 }}'\n"
            "      concurrency: '{{ load }}'\n"
        )
        _, _, configs = _expand(yaml_str)

        assert [c.phases[0].concurrency for c in configs] == [10, 50, 100]
        assert [c.phases[0].requests for c in configs] == [50, 250, 500]
        _assert_each_round_trips(configs)


# ---------------------------------------------------------------------------
# Zip sweep
# ---------------------------------------------------------------------------


class TestZipSweepExpandsAndValidates:
    def test_paired_isl_osl_lockstep(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: zip\n"
            "  parameters:\n"
            "    datasets.default.prompts.isl: [128, 512, 2048]\n"
            "    datasets.default.prompts.osl: [128, 256, 512]\n" + _BASE_BODY
        )
        _, _, configs = _expand(yaml_str)

        assert len(configs) == 3
        pairs = [
            (c.datasets[0].prompts.isl.value, c.datasets[0].prompts.osl.value)
            for c in configs
        ]
        assert pairs == [(128, 128), (512, 256), (2048, 512)]
        _assert_each_round_trips(configs)

    def test_paired_variables_with_jinja_body(self) -> None:
        # Two Jinja vars swept in lockstep, both flowing into different body
        # fields. Verifies the per-variation re-render picks up BOTH overrides.
        yaml_str = (
            "variables: {conc: 1, reqs: 10}\n"
            "sweep:\n"
            "  type: zip\n"
            "  parameters:\n"
            "    variables.conc: [4, 16]\n"
            "    variables.reqs: [40, 160]\n"
            "benchmark:\n"
            "  models: [llama]\n"
            "  endpoint:\n"
            "    type: chat\n"
            "    urls: ['http://x:8000/v1/chat/completions']\n"
            "  datasets:\n"
            "    - name: default\n"
            "      type: synthetic\n"
            "  phases:\n"
            "    - name: profiling\n"
            "      type: concurrency\n"
            "      requests: '{{ reqs }}'\n"
            "      concurrency: '{{ conc }}'\n"
        )
        _, _, configs = _expand(yaml_str)

        assert [c.phases[0].concurrency for c in configs] == [4, 16]
        assert [c.phases[0].requests for c in configs] == [40, 160]
        _assert_each_round_trips(configs)

    def test_zip_mismatched_lengths_rejected_at_load(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: zip\n"
            "  parameters:\n"
            "    phases.profiling.concurrency: [1, 2]\n"
            "    phases.profiling.requests: [10, 20, 30]\n" + _BASE_BODY
        )
        with pytest.raises(Exception, match="equal length"):
            load_config_from_string(yaml_str)


# ---------------------------------------------------------------------------
# Scenarios sweep
# ---------------------------------------------------------------------------


class TestScenariosSweepExpandsAndValidates:
    def test_named_runs_deep_merge_and_round_trip(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: scenarios\n"
            "  runs:\n"
            "    - name: small\n"
            "      benchmark:\n"
            "        phases:\n"
            "          - name: profiling\n"
            "            concurrency: 4\n"
            "            requests: 40\n"
            "    - name: medium\n"
            "      benchmark:\n"
            "        phases:\n"
            "          - name: profiling\n"
            "            concurrency: 16\n"
            "            requests: 160\n"
            "    - name: large\n"
            "      benchmark:\n"
            "        phases:\n"
            "          - name: profiling\n"
            "            concurrency: 64\n"
            "            requests: 640\n" + _BASE_BODY
        )
        _, variations, configs = _expand(yaml_str)

        assert len(configs) == 3
        assert [v.label for v in variations] == ["small", "medium", "large"]
        assert [c.phases[0].concurrency for c in configs] == [4, 16, 64]
        assert [c.phases[0].requests for c in configs] == [40, 160, 640]
        # Base fields survive the deep-merge unchanged.
        for c in configs:
            assert c.models.items[0].name == "llama"
            assert c.datasets[0].prompts.isl.value == 128
        _assert_each_round_trips(configs)


# ---------------------------------------------------------------------------
# Sobol / Latin Hypercube
# ---------------------------------------------------------------------------


class TestQMCSweepsExpandAndValidate:
    def test_sobol_body_path_within_bounds(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: sobol\n"
            "  samples: 8\n"
            "  seed: 42\n"
            "  dimensions:\n"
            "    - {path: phases.profiling.concurrency, lo: 1, hi: 32, kind: int}\n"
            + _BASE_BODY
        )
        _, variations, configs = _expand(yaml_str)

        assert len(configs) == 8
        # Every variation's swept value must lie inside the declared range
        # AND match the body field after validation.
        for v, c in zip(variations, configs, strict=True):
            sampled = v.values["phases.profiling.concurrency"]
            assert 1 <= sampled <= 32
            assert c.phases[0].concurrency == sampled
        _assert_each_round_trips(configs)

    def test_sobol_variables_axis_drives_jinja_body(self) -> None:
        yaml_str = (
            "variables: {conc: 1}\n"
            "sweep:\n"
            "  type: sobol\n"
            "  samples: 4\n"
            "  seed: 7\n"
            "  dimensions:\n"
            "    - {path: variables.conc, lo: 1, hi: 100, kind: int}\n"
            "benchmark:\n"
            "  models: [llama]\n"
            "  endpoint:\n"
            "    type: chat\n"
            "    urls: ['http://x:8000/v1/chat/completions']\n"
            "  datasets:\n"
            "    - name: default\n"
            "      type: synthetic\n"
            "  phases:\n"
            "    - name: profiling\n"
            "      type: concurrency\n"
            "      requests: 10\n"
            "      concurrency: '{{ conc }}'\n"
        )
        _, variations, configs = _expand(yaml_str)

        assert len(configs) == 4
        for v, c in zip(variations, configs, strict=True):
            sampled = v.values["variables.conc"]
            assert 1 <= sampled <= 100
            assert c.phases[0].concurrency == sampled
        _assert_each_round_trips(configs)

    def test_latin_hypercube_body_path_within_bounds(self) -> None:
        yaml_str = (
            "sweep:\n"
            "  type: latin_hypercube\n"
            "  samples: 6\n"
            "  seed: 13\n"
            "  dimensions:\n"
            "    - {path: phases.profiling.concurrency, lo: 1, hi: 64, kind: int}\n"
            + _BASE_BODY
        )
        _, variations, configs = _expand(yaml_str)

        assert len(configs) == 6
        for v, c in zip(variations, configs, strict=True):
            sampled = v.values["phases.profiling.concurrency"]
            assert 1 <= sampled <= 64
            assert c.phases[0].concurrency == sampled
        _assert_each_round_trips(configs)


# ---------------------------------------------------------------------------
# No-sweep base case
# ---------------------------------------------------------------------------


def test_no_sweep_block_yields_single_base_variation() -> None:
    """Sanity: the absence of a `sweep:` block produces one `base` variation.

    Pinned so a future change to the raw-envelope plumbing that accidentally
    requires a sweep block to set `_raw_envelope` is caught immediately.
    """
    _, variations, configs = _expand(_BASE_BODY)

    assert len(configs) == 1
    assert variations[0].label == "base"
    assert variations[0].values == {}
    assert configs[0].phases[0].concurrency == 1
    _assert_each_round_trips(configs)
