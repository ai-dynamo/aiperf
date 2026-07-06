# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the envelope shape restructure.

Spec: (deleted)
"""

from __future__ import annotations

import logging
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
    """The loader auto-migrates pre-restructure flat-shape YAML to envelope shape."""

    def test_flat_models_at_top_auto_migrates_to_envelope(self, caplog):
        flat = textwrap.dedent("""
            models: [test/model]
            endpoint:
              type: chat
              urls: ["http://localhost:8000/v1/chat/completions"]
            datasets:
              - {name: main, type: synthetic, entries: 100}
            phases:
              - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
        """).strip()

        logging.getLogger(__name__).warning("unrelated flat shape setup noise")

        cfg = load_config_from_string(flat, substitute_env=False)
        assert cfg.benchmark.models.items[0].name == "test/model"
        # Deprecation warning should fire pointing at the migrate tool.
        warnings = [
            r
            for r in caplog.records
            if "flat shape" in r.getMessage()
            and "migrate_config_yaml.py" in r.getMessage()
        ]
        assert warnings, "expected deprecation warning for flat-shape YAML"

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
    """Grid sweep parameter paths are body-rooted under ``benchmark``."""

    def test_body_rooted_path_accepts(self):
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
              parameters:
                "phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        plan = _load_plan_from_string(yaml_str)
        assert plan.is_sweep
        assert len(plan.configs) == 3

    def test_redundant_benchmark_prefix_rejects(self):
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
              parameters:
                "benchmark.phases.profiling.concurrency": [1, 2, 4]
        """).strip()

        with pytest.raises((ValueError, ConfigurationError)) as excinfo:
            _load_plan_from_string(yaml_str)
        assert "benchmark." in str(excinfo.value)

    def test_non_benchmark_top_level_rejected(self):
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
              parameters:
                "multi_run.num_runs": [1, 2]
        """).strip()

        with pytest.raises((ValueError, ConfigurationError)) as excinfo:
            _load_plan_from_string(yaml_str)
        assert "non-sweepable" in str(excinfo.value)
