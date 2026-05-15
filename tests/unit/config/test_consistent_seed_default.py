# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-default `random_seed=42` when consistent-seed is on for multi-run/sweep.

Without the auto-fill, a default-flagged multi-run or sweep produces
non-deterministic workloads and confidence statistics conflate runtime
variance with input-noise.
"""

from __future__ import annotations

import textwrap

from aiperf.config.loader.core import load_config_from_string
from aiperf.config.loader.plan import build_benchmark_plan

_BASE = textwrap.dedent("""
    benchmark:
      models: [test/model]
      endpoint:
        type: chat
        urls: ["http://localhost:8000/v1/chat/completions"]
      datasets:
        - {name: main, type: synthetic, entries: 100}
      phases:
        - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
""").strip()


def _load(extra: str = ""):
    return load_config_from_string(
        _BASE + ("\n" + textwrap.dedent(extra).strip() if extra else ""),
        substitute_env=False,
    )


def test_single_run_no_auto_fill():
    """Single run with default flags leaves random_seed=None — no consistency need."""
    cfg = _load()
    assert cfg.random_seed is None


def test_multi_run_auto_fills_42():
    """num_runs > 1 with set_consistent_seed=True (default) fills random_seed=42."""
    cfg = _load(
        """
        multi_run:
          num_runs: 3
        """
    )
    assert cfg.random_seed == 42


def test_sweep_auto_fills_42():
    """Sweep auto-fills random_seed=42 even for num_runs=1 (per-variation parity)."""
    cfg = _load(
        """
        sweep:
          type: grid
          parameters:
            phases.profiling.concurrency: [1, 2, 4]
        """
    )
    assert cfg.random_seed == 42


def test_explicit_seed_preserved_under_multi_run():
    """User-set random_seed always wins over auto-fill."""
    cfg = _load(
        """
        random_seed: 99
        multi_run:
          num_runs: 3
        """
    )
    assert cfg.random_seed == 99


def test_set_consistent_seed_false_disables_auto_fill():
    """When the user opts out of consistency, multi-run keeps random_seed=None."""
    cfg = _load(
        """
        multi_run:
          num_runs: 3
          set_consistent_seed: false
        """
    )
    assert cfg.random_seed is None


def test_plan_propagates_auto_filled_seed_into_variations():
    """Auto-filled envelope seed reaches plan.variation_seeds via base+idx derivation."""
    cfg = _load(
        """
        sweep:
          type: grid
          parameters:
            phases.profiling.concurrency: [1, 2, 4]
        """
    )
    plan = build_benchmark_plan(cfg)
    assert plan.variation_seeds == [42, 43, 44]


def test_explicit_seed_zero_preserved_under_multi_run():
    """random_seed=0 is a valid user-set seed and must NOT be auto-overwritten.

    Zero is falsy but not None. The auto-fill validator uses `is not None`
    correctly today; refactoring to truthiness (`if not cfg.random_seed`)
    would silently overwrite a user's deliberate seed=0 with 42, producing
    different reproducible workloads with no test failure.
    """
    cfg = _load(
        """
        random_seed: 0
        multi_run:
          num_runs: 3
        """
    )
    assert cfg.random_seed == 0


def test_zip_sweep_same_seed_true_reuses_envelope_seed_for_all_variations():
    """ZipSweep inherits same_seed from _GridSweepBase — every variation uses base seed.

    `same_seed` lives on `_GridSweepBase`, so all three subclasses (Grid,
    Zip, Scenario) inherit the flag. If someone moves it to `GridSweep`
    only, or splits the base class, zip's variation seeds silently flip
    from `[base, base, base]` to `[base, base+1, base+2]`. This test
    catches that as a hard failure.
    """
    cfg = _load(
        """
        sweep:
          type: zip
          same_seed: true
          parameters:
            phases.profiling.concurrency: [1, 2, 4]
            datasets.main.entries: [10, 20, 40]
        """
    )
    plan = build_benchmark_plan(cfg)
    # All three lockstep variations must share the envelope seed.
    assert plan.variation_seeds == [42, 42, 42]


def test_zip_sweep_same_seed_false_uses_indexed_derivation():
    """Default `same_seed=False` keeps the additive `base + idx` derivation."""
    cfg = _load(
        """
        sweep:
          type: zip
          parameters:
            phases.profiling.concurrency: [1, 2, 4]
            datasets.main.entries: [10, 20, 40]
        """
    )
    plan = build_benchmark_plan(cfg)
    assert plan.variation_seeds == [42, 43, 44]


def test_scenario_sweep_same_seed_true_reuses_envelope_seed():
    """ScenarioSweep also inherits same_seed from _GridSweepBase."""
    cfg = _load(
        """
        sweep:
          type: scenarios
          same_seed: true
          runs:
            - name: low
              benchmark:
                phases:
                  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            - name: high
              benchmark:
                phases:
                  - {name: profiling, type: concurrency, requests: 10, concurrency: 4}
        """
    )
    plan = build_benchmark_plan(cfg)
    assert plan.variation_seeds == [42, 42]


def test_scenario_sweep_same_seed_false_uses_indexed_derivation():
    cfg = _load(
        """
        sweep:
          type: scenarios
          runs:
            - name: low
              benchmark:
                phases:
                  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
            - name: high
              benchmark:
                phases:
                  - {name: profiling, type: concurrency, requests: 10, concurrency: 4}
        """
    )
    plan = build_benchmark_plan(cfg)
    assert plan.variation_seeds == [42, 43]


def test_grid_sweep_same_seed_true_reuses_envelope_seed():
    """Grid completes the parity story: every _GridSweepBase subclass behaves identically."""
    cfg = _load(
        """
        sweep:
          type: grid
          same_seed: true
          parameters:
            phases.profiling.concurrency: [1, 2, 4]
        """
    )
    plan = build_benchmark_plan(cfg)
    assert plan.variation_seeds == [42, 42, 42]
