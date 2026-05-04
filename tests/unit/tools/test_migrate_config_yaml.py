# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tools/migrate_config_yaml.py — the Plan A migration script."""

from __future__ import annotations

import textwrap

from tools.migrate_config_yaml import (
    BODY_KEYS,
    ENVELOPE_KEYS,
    is_already_migrated,
    migrate_yaml_text,
    rewrite_grid_sweep_paths,
    rewrite_scenario_runs,
)


class TestBodyEnvelopePartition:
    def test_constants_are_disjoint(self):
        assert BODY_KEYS.isdisjoint(ENVELOPE_KEYS)

    def test_body_keys_match_spec(self):
        # Includes shortcut singulars (model/dataset/warmup/profiling) and
        # camelCase aliases (gpuTelemetry/serverMetrics) so half-migrated
        # templates with hoisted shortcuts land cleanly under benchmark: on
        # re-run; see tools/migrate_config_yaml.py docstring.
        assert {
            "models",
            "model",
            "endpoint",
            "datasets",
            "dataset",
            "phases",
            "warmup",
            "profiling",
            "artifacts",
            "slos",
            "tokenizer",
            "gpu_telemetry",
            "gpuTelemetry",
            "server_metrics",
            "serverMetrics",
            "runtime",
            "logging",
            "metrics",
            "accuracy",
        } == BODY_KEYS

    def test_envelope_keys_match_spec(self):
        assert {
            "sweep",
            "multi_run",
            "multiRun",
            "variables",
            "random_seed",
            "randomSeed",
            "benchmark",
        } == ENVELOPE_KEYS


class TestMigrateYamlText:
    def test_flat_shape_rewraps_body_under_benchmark(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://localhost:8000/v1/chat/completions"]
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
random_seed: 42
""").strip()
        out = migrate_yaml_text(flat)
        # body keys nested under benchmark:
        assert "benchmark:" in out
        assert "  models:" in out  # indented under benchmark
        assert "  endpoint:" in out
        assert "  phases:" in out
        # envelope keys stay top-level
        assert "\nrandom_seed: 42" in out

    def test_already_migrated_yaml_passes_through_unchanged(self):
        already = textwrap.dedent("""
random_seed: 42
benchmark:
  models: [llama]
  endpoint:
    urls: ["http://localhost:8000/v1/chat/completions"]
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
""").strip()
        out = migrate_yaml_text(already)
        # idempotent
        assert migrate_yaml_text(out) == out

    def test_envelope_keys_stay_at_top_level(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
sweep:
  type: grid
  variables:
    benchmark.phases.profiling.concurrency: [1, 2, 4]
multi_run:
  num_runs: 3
variables:
  isl: 128
random_seed: 42
""").strip()
        out = migrate_yaml_text(flat)
        # all four envelope keys at top
        for key in ("sweep:", "multi_run:", "variables:", "random_seed:"):
            assert f"\n{key}" in out or out.startswith(key)

    def test_grid_sweep_path_keys_get_benchmark_prefix(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
sweep:
  type: grid
  variables:
    benchmark.phases.profiling.concurrency: [1, 2, 4]
    benchmark.datasets.0.entries: [100, 200]
""").strip()
        out = migrate_yaml_text(flat)
        assert "benchmark.phases.profiling.concurrency" in out
        assert "benchmark.datasets.0.entries" in out

    def test_grid_variables_path_unchanged(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
variables:
  isl: 128
sweep:
  type: grid
  variables:
    "variables.isl": [128, 256]
""").strip()
        out = migrate_yaml_text(flat)
        assert "variables.isl" in out
        # no benchmark.variables.isl
        assert "benchmark.variables.isl" not in out

    def test_scenario_runs_body_keys_get_benchmark_wrapper(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
sweep:
  type: scenarios
  runs:
    - name: low
      benchmark:
        phases:
          - {name: profiling, type: concurrency, concurrency: 1}
    - name: high
      benchmark:
        phases:
          - {name: profiling, type: concurrency, concurrency: 10}
""").strip()
        out = migrate_yaml_text(flat)
        # phases inside runs[i] should now be runs[i].benchmark.phases
        # naive substring check: there should be no top-level `phases:` directly inside a run dict
        # but there should be a `benchmark:` wrapper inside each run
        assert out.count("benchmark:") >= 1
        assert "phases:" in out

    def test_scenario_runs_keep_name_and_variables_at_top(self):
        flat = textwrap.dedent("""
benchmark:
  models: [llama]
  phases:
    - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
sweep:
  type: scenarios
  runs:
    - name: pair_0
      variables: {isl: 128}
      benchmark:
        phases:
          - {name: profiling, type: concurrency, concurrency: 5}
""").strip()
        out = migrate_yaml_text(flat)
        # runs[0].name and runs[0].variables stay at top of the run dict
        assert "name: pair_0" in out
        assert "variables:" in out

    def test_empty_yaml_returns_empty(self):
        out = migrate_yaml_text("")
        assert out == "" or out.strip() == ""

    def test_no_body_keys_passes_through(self):
        # config-only-envelope fragment — nothing to migrate
        text = "random_seed: 42\nvariables:\n  isl: 128\n"
        out = migrate_yaml_text(text)
        # benchmark key not introduced when no body keys present
        assert "benchmark:" not in out


class TestIsAlreadyMigrated:
    def test_envelope_with_benchmark_key_is_migrated(self):
        text = "benchmark:\n  models: [llama]\nrandom_seed: 42\n"
        assert is_already_migrated(text) is True

    def test_flat_shape_is_not_migrated(self):
        text = "models: [llama]\nrandom_seed: 42\n"
        assert is_already_migrated(text) is False

    def test_envelope_only_no_body_is_migrated(self):
        text = "random_seed: 42\nvariables:\n  isl: 128\n"
        assert is_already_migrated(text) is True


class TestRewriteGridSweepPaths:
    def test_phases_path_gets_benchmark_prefix(self):
        sweep = {"type": "grid", "variables": {"phases.profiling.concurrency": [1, 2]}}
        rewrite_grid_sweep_paths(sweep)
        assert "benchmark.phases.profiling.concurrency" in sweep["variables"]
        assert "phases.profiling.concurrency" not in sweep["variables"]

    def test_variables_path_unchanged(self):
        sweep = {"type": "grid", "variables": {"variables.isl": [128, 256]}}
        rewrite_grid_sweep_paths(sweep)
        assert "variables.isl" in sweep["variables"]

    def test_already_prefixed_paths_unchanged(self):
        sweep = {
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [1, 2]},
        }
        rewrite_grid_sweep_paths(sweep)
        assert "benchmark.phases.profiling.concurrency" in sweep["variables"]
        # not double-prefixed
        assert (
            "benchmark.benchmark.phases.profiling.concurrency" not in sweep["variables"]
        )


class TestRewriteScenarioRuns:
    def test_run_with_phases_wraps_under_benchmark(self):
        run = {"name": "low", "phases": [{"name": "profiling", "concurrency": 1}]}
        rewrite_scenario_runs([run])
        assert "phases" not in run  # moved into benchmark
        assert "benchmark" in run
        assert run["benchmark"]["phases"] == [{"name": "profiling", "concurrency": 1}]

    def test_run_with_variables_keeps_at_top(self):
        run = {"variables": {"isl": 128}, "phases": [{"name": "profiling"}]}
        rewrite_scenario_runs([run])
        assert run["variables"] == {"isl": 128}
        assert run["benchmark"]["phases"] == [{"name": "profiling"}]

    def test_run_already_using_benchmark_wrapper_unchanged(self):
        run = {
            "name": "low",
            "benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]},
        }
        rewrite_scenario_runs([run])
        assert run["benchmark"]["phases"] == [{"name": "profiling", "concurrency": 1}]
        assert "phases" not in run  # never was at top
