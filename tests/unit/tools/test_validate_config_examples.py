# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tools/validate_config_examples.py.

Historically the validator only called ``load_config`` and never expanded
sweeps, despite its docstring claiming it checks that "sweep configurations
expand properly." A broken sweep template (singular ``dataset:`` combined with a
``datasets.*`` sweep path) therefore shipped and crashed on use. These tests
lock in the extended behavior: sweep configs are expanded via
``build_benchmark_plan`` and a broken sweep is reported as a failure, while
non-sweep configs keep their original single-load behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.validate_config_examples as vce

_VALID_SWEEP = """\
schemaVersion: "2.0"
sweep:
  type: grid
  variables:
    datasets.default.prompts.isl: [128, 512, 2048]
    rate: [10.0, 30.0, 50.0]
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
  datasets:
    - name: default
      type: synthetic
      entries: 500
      prompts:
        isl: 512
        osl: {mean: 128, stddev: 25}
  phases:
    - name: profiling
      type: poisson
      rate: 20.0
      duration: 120
"""

# Same as _VALID_SWEEP but with the singular `dataset:` shorthand — loads fine,
# fails to expand (the defect that once shipped in sweep_distributions.yaml).
_BROKEN_SWEEP = """\
schemaVersion: "2.0"
sweep:
  type: grid
  variables:
    datasets.default.prompts.isl: [128, 512, 2048]
    rate: [10.0, 30.0, 50.0]
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
  dataset:
    type: synthetic
    entries: 500
    prompts:
      isl: 512
      osl: {mean: 128, stddev: 25}
  phases:
    - name: profiling
      type: poisson
      rate: 20.0
      duration: 120
"""

_PLAIN_CONFIG = """\
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://localhost:8000/v1/chat/completions
    type: chat
  dataset:
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


def _run_on(tmp_path: Path, name: str, body: str) -> tuple[int, int, int]:
    """Point the validator at a temp dir holding one file; return counts."""
    (tmp_path / name).write_text(body)
    vce.EXAMPLES_DIR = tmp_path
    failed, total, configs, _elapsed = vce.validate_examples(
        verbose=True, skip_schema=True
    )
    return failed, total, configs


@pytest.fixture(autouse=True)
def _restore_examples_dir():
    """Restore the module-level EXAMPLES_DIR after each test mutates it."""
    original = vce.EXAMPLES_DIR
    yield
    vce.EXAMPLES_DIR = original


class TestSweepExpansionGate:
    def test_valid_sweep_passes_and_counts_all_variations(self, tmp_path: Path) -> None:
        failed, total, configs = _run_on(tmp_path, "valid_sweep.yaml", _VALID_SWEEP)
        assert failed == 0
        assert total == 1
        # 3 ISL x 3 rate == 9 expanded variations, not 1.
        assert configs == 9

    def test_broken_sweep_is_reported_as_failure(self, tmp_path: Path) -> None:
        # Sanity-check the fixture actually uses the singular shorthand.
        assert "  dataset:\n" in _BROKEN_SWEEP
        assert "datasets.default.prompts.isl" in _BROKEN_SWEEP

        failed, total, _configs = _run_on(tmp_path, "broken_sweep.yaml", _BROKEN_SWEEP)
        assert failed == 1
        assert total == 1


class TestNonSweepBehaviorUnchanged:
    def test_plain_config_counts_one(self, tmp_path: Path) -> None:
        failed, total, configs = _run_on(tmp_path, "plain.yaml", _PLAIN_CONFIG)
        assert failed == 0
        assert total == 1
        assert configs == 1
