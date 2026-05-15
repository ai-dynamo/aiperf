# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-(variation, trial) seed resolution at run-construction time.

Grid/zip/scenario sweeps pre-compute the full ``plan.variation_seeds`` list at
plan-build (``base + variation.index``). Adaptive sweeps
discover variations on the fly past plan-build, so ``variation.index`` exceeds
the plan-time list length — the resolver falls back to SHA-256 derivation over
``(envelope_seed, variation.label)`` so iter > 0 doesn't silently drop the seed.

When ``multi_run.vary_seed_per_trial`` is True, every trial of every variation
gets a distinct SHA-derived seed instead, so confidence statistics capture
input variance in addition to runtime variance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aiperf.common.random_generator import derive_variation_seed
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.orchestrator import resolve_run_seed


def _stub_plan(
    *,
    variation_seeds: list[int | None],
    random_seed: int | None,
    vary_seed_per_trial: bool = False,
):
    plan = MagicMock()
    plan.variation_seeds = variation_seeds
    plan.random_seed = random_seed
    plan.multi_run = MagicMock()
    plan.multi_run.vary_seed_per_trial = vary_seed_per_trial
    return plan


# -- Same-seed-across-trials (default) --------------------------------------


def test_grid_uses_precomputed_indexed_lookup():
    plan = _stub_plan(variation_seeds=[42, 43, 44], random_seed=42)
    for idx in range(3):
        v = SweepVariation(index=idx, label=f"variation_{idx:04d}", values={})
        assert resolve_run_seed(plan, v) == 42 + idx


def test_grid_trials_share_variation_seed_by_default():
    """Default: trials within a variation reuse the same seed (main parity)."""
    plan = _stub_plan(variation_seeds=[42, 43], random_seed=42)
    v = SweepVariation(index=1, label="variation_0001", values={})
    seeds = {resolve_run_seed(plan, v, trial=t) for t in range(5)}
    assert seeds == {43}


def test_adaptive_overflow_falls_back_to_sha_derivation():
    plan = _stub_plan(variation_seeds=[42], random_seed=42)
    v0 = SweepVariation(index=0, label="search_iter_0000", values={})
    v1 = SweepVariation(index=1, label="search_iter_0001", values={})
    v42 = SweepVariation(index=42, label="search_iter_0042", values={})

    assert resolve_run_seed(plan, v0) == 42
    assert resolve_run_seed(plan, v1) == derive_variation_seed(42, v1.label)
    assert resolve_run_seed(plan, v42) == derive_variation_seed(42, v42.label)
    assert resolve_run_seed(plan, v1) is not None
    assert resolve_run_seed(plan, v42) is not None


def test_adaptive_overflow_with_no_envelope_seed_returns_none():
    plan = _stub_plan(variation_seeds=[None], random_seed=None)
    v5 = SweepVariation(index=5, label="search_iter_0005", values={})
    assert resolve_run_seed(plan, v5) is None


def test_adaptive_same_label_always_yields_same_seed():
    """Re-proposing the same label across replans gives the same seed."""
    plan_a = _stub_plan(variation_seeds=[42], random_seed=42)
    plan_b = _stub_plan(variation_seeds=[42], random_seed=42)
    v = SweepVariation(index=7, label="search_iter_0007", values={})
    assert resolve_run_seed(plan_a, v) == resolve_run_seed(plan_b, v)


def test_adaptive_distinct_labels_yield_distinct_seeds():
    plan = _stub_plan(variation_seeds=[42], random_seed=42)
    seeds = {
        resolve_run_seed(
            plan, SweepVariation(index=i, label=f"search_iter_{i:04d}", values={})
        )
        for i in range(1, 50)
    }
    assert len(seeds) == 49  # SHA-truncated-to-64-bit collisions vanishingly unlikely


# -- Vary-seed-per-trial mode ------------------------------------------------


def test_per_trial_seeds_distinct_within_one_variation():
    """All trials of a variation get distinct seeds when vary_seed_per_trial=True."""
    plan = _stub_plan(variation_seeds=[42], random_seed=42, vary_seed_per_trial=True)
    v = SweepVariation(index=0, label="concurrency_10", values={})
    seeds = [resolve_run_seed(plan, v, trial=t) for t in range(10)]
    assert len(set(seeds)) == 10
    # Stable: same (variation, trial) reproduces the same seed.
    assert seeds[3] == resolve_run_seed(plan, v, trial=3)


def test_per_trial_seeds_distinct_across_variations():
    """Same trial number across variations yields distinct seeds."""
    plan = _stub_plan(
        variation_seeds=[42, 43, 44], random_seed=42, vary_seed_per_trial=True
    )
    v0 = SweepVariation(index=0, label="concurrency_10", values={})
    v1 = SweepVariation(index=1, label="concurrency_20", values={})
    assert resolve_run_seed(plan, v0, trial=0) != resolve_run_seed(plan, v1, trial=0)


def test_per_trial_with_no_envelope_seed_returns_none():
    """vary_seed_per_trial honors the opt-out — None envelope -> None per-trial."""
    plan = _stub_plan(
        variation_seeds=[None], random_seed=None, vary_seed_per_trial=True
    )
    v = SweepVariation(index=0, label="concurrency_10", values={})
    assert resolve_run_seed(plan, v, trial=0) is None
    assert resolve_run_seed(plan, v, trial=5) is None


def test_per_trial_grid_does_not_use_precomputed_list():
    """When vary_seed_per_trial=True, plan.variation_seeds is bypassed entirely."""
    plan = _stub_plan(
        variation_seeds=[42, 43], random_seed=42, vary_seed_per_trial=True
    )
    v0 = SweepVariation(index=0, label="concurrency_10", values={})
    v1 = SweepVariation(index=1, label="concurrency_20", values={})
    # Trial 0 of variation 0 must NOT be the bare base_seed (it's SHA-derived).
    assert resolve_run_seed(plan, v0, trial=0) != 42
    assert resolve_run_seed(plan, v1, trial=0) != 43
    # All four (var, trial) combinations are distinct.
    seeds = {
        resolve_run_seed(plan, v0, trial=0),
        resolve_run_seed(plan, v0, trial=1),
        resolve_run_seed(plan, v1, trial=0),
        resolve_run_seed(plan, v1, trial=1),
    }
    assert len(seeds) == 4


# -- Composite-label format pin ---------------------------------------------


def test_per_trial_composite_label_format_pinned():
    """The composite label is `f"{variation.label}:trial:{trial}"`.

    Pins the exact composition at `orchestrator.py:131-134`. Refactoring the
    separator (e.g. to `/`, `_`, or `-`) would silently shift every
    `vary_seed_per_trial` seed for every existing config — every reproducible
    benchmark would generate a different workload with no test failure unless
    this exact equality is asserted.
    """
    plan = _stub_plan(variation_seeds=[42], random_seed=42, vary_seed_per_trial=True)
    v = SweepVariation(index=0, label="concurrency_10", values={})

    # Every (variation, trial) seed must equal the SHA derivation over the
    # composite label — not just "any distinct value".
    for trial in (0, 1, 5, 99):
        expected = derive_variation_seed(42, f"{v.label}:trial:{trial}")
        assert resolve_run_seed(plan, v, trial=trial) == expected


def test_adaptive_overflow_uses_bare_label_not_composite():
    """Adaptive overflow path passes ONLY `variation.label` (no `:trial:` suffix).

    Pins the dispatch in `orchestrator.py:135-137`: when vary_seed_per_trial
    is False AND variation.index >= len(variation_seeds), the seed depends on
    the variation but is CONSTANT across trials. If a future refactor ever
    accidentally appends `:trial:0` here, every adaptive overflow seed would
    shift silently.
    """
    plan = _stub_plan(variation_seeds=[42], random_seed=42, vary_seed_per_trial=False)
    v = SweepVariation(index=5, label="search_iter_0005", values={})

    expected = derive_variation_seed(42, "search_iter_0005")
    # Same value for every trial number on the overflow path.
    assert resolve_run_seed(plan, v, trial=0) == expected
    assert resolve_run_seed(plan, v, trial=3) == expected
    # And critically, NOT equal to the composite-label form.
    assert resolve_run_seed(plan, v, trial=0) != derive_variation_seed(
        42, "search_iter_0005:trial:0"
    )
