# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan A behavior tests for the envelope shape restructure.

Spec: docs/superpowers/specs/2026-05-03-aiperf-config-envelope-restructure-design.md
"""

from __future__ import annotations

import textwrap

import pytest

from aiperf.config.loader.core import load_config_from_string
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.loader.plan import build_benchmark_plan


def _load_plan_from_string(yaml_str: str, *, substitute_env: bool = False):
    """Test helper: parse YAML envelope -> AIPerfConfig -> BenchmarkPlan."""
    config = load_config_from_string(yaml_str, substitute_env=substitute_env)
    return build_benchmark_plan(config)


class TestFlatShapeRejection:
    """The loader rejects pre-restructure flat-shape YAML with a clear migration error."""

    def test_flat_models_at_top_raises_with_migration_hint(self):
        flat = textwrap.dedent("""
            models: [test/model]
            endpoint:
              type: chat
              urls: ["http://localhost:8000/v1/chat/completions"]
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
        """).strip()

        with pytest.raises(ConfigurationError) as excinfo:
            load_config_from_string(flat, substitute_env=False)
        msg = str(excinfo.value)
        assert "flat shape" in msg
        assert "benchmark:" in msg
        assert "migrate_config_yaml.py" in msg
        assert "models" in msg

    def test_envelope_shape_loads_cleanly(self):
        envelope = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            random_seed: 42
        """).strip()

        cfg = load_config_from_string(envelope, substitute_env=False)
        assert cfg.benchmark.models.items[0].name == "test/model"
        assert cfg.random_seed == 42

    def test_envelope_only_no_benchmark_raises_clearly(self):
        envelope_only = "random_seed: 42\nvariables:\n  isl: 128\n"

        with pytest.raises(Exception) as excinfo:
            load_config_from_string(envelope_only, substitute_env=False)
        msg = str(excinfo.value).lower()
        assert "benchmark" in msg


class TestScenarioRunValidation:
    """Sweep scenario `runs[i]` allow only {name, variables, benchmark}."""

    def test_run_with_top_level_phases_rejects(self):
        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: scenarios
              runs:
                - phases:
                    - {name: profiling, type: concurrency, concurrency: 5}
        """).strip()

        with pytest.raises((ValueError, ConfigurationError)) as excinfo:
            _load_plan_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "unknown field" in msg or "phases" in msg
        assert "name" in msg or "variables" in msg or "benchmark" in msg

    def test_run_with_benchmark_wrapper_accepted(self):
        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: scenarios
              runs:
                - benchmark:
                    phases:
                      - {name: profiling, type: concurrency, concurrency: 5}
                - benchmark:
                    phases:
                      - {name: profiling, type: concurrency, concurrency: 10}
        """).strip()

        plan = _load_plan_from_string(yaml_str)
        assert plan.is_sweep
        assert len(plan.configs) == 2


class TestGridSweepPathValidation:
    """Grid sweep variable paths must start with `benchmark.` or `variables.`."""

    def test_unprefixed_path_rejects(self):
        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        with pytest.raises((ValueError, ConfigurationError)) as excinfo:
            _load_plan_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "non-sweepable" in msg or "benchmark" in msg

    def test_prefixed_path_accepts(self):
        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "benchmark.phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        plan = _load_plan_from_string(yaml_str)
        assert plan.is_sweep
        assert len(plan.configs) == 3

    def test_runtime_path_rejected(self):
        yaml_str = textwrap.dedent("""
            benchmark:
              models: [test/model]
              endpoint:
                type: chat
                urls: ["http://localhost:8000/v1/chat/completions"]
              datasets:
                - {name: main, type: synthetic, entries: 100}
              phases:
                - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            sweep:
              type: grid
              variables:
                "runtime.workers": [1, 2]
        """).strip()

        with pytest.raises((ValueError, ConfigurationError)) as excinfo:
            _load_plan_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "runtime.workers" in msg or "non-sweepable" in msg
