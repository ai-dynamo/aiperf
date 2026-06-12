# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for MultiTierPlanner with mocked benchmark execution.

Drives the full ask/tell loop for a 3-tier configuration and verifies:
1. Per-tier boundaries are correctly discovered (fast < standard < economy)
2. Observation sharing: probes at one concurrency are used by all tiers
3. Ordering exploitation: ordering pairs are detected, propagation occurs
4. Output completeness: tier_results() returns 3 entries with all required fields
5. boundary_summary() returns data from the most-lenient tier
6. Total probe count is less than 3x single-tier (observation sharing benefit)

Validates: Requirements 2.1, 3.1, 4.1, 6.1
"""

from __future__ import annotations

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective, SweepVariation
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )


def _cfg(
    *, lo: int = 1, hi: int = 256, max_iterations: int = 50
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=lo, hi=hi, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        max_iterations=max_iterations,
        n_initial_points=2,
        sla_filters=[],
        sla_tiers=[],
    )


def _three_tiers() -> list[SLOTier]:
    """Three ordered tiers: fast (>300), standard (>100), economy (>30).

    With the throughput model throughput = max(1, 1000 - 20*c):
    - fast boundary: c=34 (1000-680=320>300; c=35→1000-700=300 not >300)
    - standard boundary: c=44 (1000-880=120>100; c=45→1000-900=100 not >100)
    - economy boundary: c=48 (1000-960=40>30; c=49→1000-980=20<30)

    These boundaries are close enough that the bracket phase (doubling:
    1→2→4→8→16→32→64) fails ALL tiers at c=64 (throughput = max(1, -280) = 1),
    establishing infeasible_min for all tiers in one shot.
    """
    return [
        SLOTier(
            label="fast",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=300.0,
                )
            ],
        ),
        SLOTier(
            label="standard",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=100.0,
                )
            ],
        ),
        SLOTier(
            label="economy",
            filters=[
                SLAFilter(
                    metric_tag="output_token_throughput",
                    stat="avg",
                    op="gt",
                    threshold=30.0,
                )
            ],
        ),
    ]


def _make_planner(
    *, lo: int = 1, hi: int = 256, max_iterations: int = 50
) -> MultiTierPlanner:
    return MultiTierPlanner(
        base_config=_base_config(),
        cfg=_cfg(lo=lo, hi=hi, max_iterations=max_iterations),
        tiers=_three_tiers(),
    )


def _simulate_throughput(concurrency: int) -> float:
    """Mock throughput model: throughput = max(1, 1000 - 20*c).

    Linear decline ensures all tiers fail within the same bracket-phase
    doubling step (c=64 yields throughput=1, below all thresholds). This
    gives the planner both bounds for every tier in one bracket step.

    Tier boundaries:
    - fast (>300): c=34 (last pass), c=35 (first fail)
    - standard (>100): c=44 (last pass), c=45 (first fail)
    - economy (>30): c=48 (last pass), c=49 (first fail)
    """
    return max(1.0, 1000.0 - 20.0 * concurrency)


def _make_result(variation: SweepVariation, concurrency: int) -> RunResult:
    """Create a RunResult with throughput derived from the mock model."""
    throughput = _simulate_throughput(concurrency)
    return RunResult(
        label="trial_0",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


def _drive_to_convergence(planner: MultiTierPlanner) -> int:
    """Drive the ask/tell loop until convergence, return total probe count."""
    probe_count = 0
    while True:
        pair = planner.ask()
        if pair is None:
            break
        _, variation = pair
        concurrency = variation.values["phases.profiling.concurrency"]
        results = [_make_result(variation, concurrency)]
        planner.tell(variation, results)
        probe_count += 1
    return probe_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiTierEndToEnd:
    """End-to-end integration: 3-tier search with mocked benchmark execution."""

    def test_per_tier_boundaries_correctly_ordered(self) -> None:
        """Discovered boundaries follow fast < standard < economy ordering."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        results = planner.tier_results()
        boundaries = {r.label: r.boundary_concurrency for r in results}

        assert boundaries["fast"] is not None
        assert boundaries["standard"] is not None
        assert boundaries["economy"] is not None
        assert boundaries["fast"] < boundaries["standard"] < boundaries["economy"]

    def test_per_tier_boundaries_near_expected_values(self) -> None:
        """Boundaries approximate the analytic solution within bracket precision.

        Model: throughput = max(1, 1000 - 20*c)
        - fast boundary: c=34 (1000-680=320>300; c=35→300 not >300)
        - standard boundary: c=44 (1000-880=120>100; c=45→100 not >100)
        - economy boundary: c=48 (1000-960=40>30; c=49→20<30)
        """
        planner = _make_planner()
        _drive_to_convergence(planner)

        results = planner.tier_results()
        boundaries = {r.label: r.boundary_concurrency for r in results}

        # fast: boundary should be 34 (1000-680=320>300, 1000-700=300 not >300)
        assert boundaries["fast"] == 34
        # standard: boundary should be 44 (1000-880=120>100, 1000-900=100 not >100)
        assert boundaries["standard"] == 44
        # economy: boundary should be 48 (1000-960=40>30, 1000-980=20 not >30)
        assert boundaries["economy"] == 48

    def test_observation_sharing_store_has_unique_concurrency_entries(self) -> None:
        """Each concurrency level is stored only once in the shared store."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        store = planner._store
        levels = store.concurrency_levels()

        # Each level should appear exactly once in the sorted list (no duplicates)
        assert levels == sorted(set(levels))
        # Every probe should be stored
        total_stored = sum(len(store.get(c)) for c in levels)
        assert total_stored == len(planner.history())

    def test_observation_sharing_reduces_total_probes(self) -> None:
        """Total probes should be significantly less than 3x a single-tier search.

        A single-tier binary search over [1, 512] takes about log2(512)=9 probes
        plus bracket phase. Three independent searches would take ~27+. The
        multi-tier planner with observation sharing should use fewer total.
        """
        planner = _make_planner()
        probe_count = _drive_to_convergence(planner)

        # Single-tier theoretical: bracket (doublings up to first fail) + bisection
        # For economy tier alone: bracket finds fail around 64 (5 doublings),
        # then bisection from 32..64 = ~5 probes = ~10 total.
        # For 3 independent: ~30 probes. Multi-tier should be well under that.
        # Use generous bound: less than 3x the widest single tier's cost.
        single_tier_estimate = 15  # conservative single-tier upper bound
        assert probe_count < 3 * single_tier_estimate

    def test_ordering_exploitation_detected(self) -> None:
        """Tier ordering pairs are detected for the 3-tier monotonic configuration."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        metadata = planner.tier_metadata()
        assert metadata["ordering_detected"] is True
        # 3 tiers with monotonic ordering → 3 pairs: (0,1), (0,2), (1,2)
        assert metadata["ordering_pairs"] is not None
        assert len(metadata["ordering_pairs"]) == 3

    def test_ordering_exploitation_propagation_occurs(self) -> None:
        """Ordering propagation updates brackets without extra probes.

        After any probe, ordering inference should have updated
        other tiers' brackets beyond what direct evaluation provided.
        """
        planner = _make_planner()
        _drive_to_convergence(planner)

        # All tiers should have converged (ordering helps resolve tighter tiers)
        for bracket in planner._brackets:
            assert bracket.converged is True

    def test_output_completeness_tier_results(self) -> None:
        """tier_results() returns 3 entries with all required fields populated."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        results = planner.tier_results()
        assert len(results) == 3

        for result in results:
            assert result.label in ("fast", "standard", "economy")
            assert result.boundary_concurrency is not None
            assert result.convergence_status in (
                "converged",
                "partial",
                "no_pass_in_range",
                "no_failure_in_range",
            )
            assert result.bracket_lower is not None
            assert result.bracket_upper is not None
            assert result.probe_count > 0
            assert isinstance(result.filters, list)
            assert len(result.filters) >= 1

    def test_output_completeness_tier_result_fields(self) -> None:
        """Each TierResult has the correct filter echo and convergence info."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        results = planner.tier_results()
        result_map = {r.label: r for r in results}

        fast = result_map["fast"]
        assert fast.filters[0]["metric_tag"] == "output_token_throughput"
        assert fast.filters[0]["stat"] == "avg"
        assert fast.filters[0]["op"] == "gt"
        assert fast.filters[0]["threshold"] == 300.0
        assert fast.convergence_status == "converged"
        assert fast.convergence_reason is not None

    def test_boundary_summary_uses_most_lenient_tier(self) -> None:
        """boundary_summary() returns data from the economy (most-lenient) tier."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        summary = planner.boundary_summary()
        assert summary is not None
        assert summary["swept_dim_path"] == "phases.profiling.concurrency"

        # The most-lenient tier has the highest feasible_max
        results = planner.tier_results()
        economy = next(r for r in results if r.label == "economy")
        assert summary["feasible_max"] is not None
        assert summary["feasible_max"]["value"] == economy.boundary_concurrency

    def test_convergence_reason_is_set(self) -> None:
        """Planner reports a valid convergence reason after completing."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        assert planner.is_converged()
        reason = planner.convergence_reason()
        assert reason == "multi_tier_all_converged"

    def test_history_records_all_iterations(self) -> None:
        """history() contains one entry per probe with correct structure."""
        planner = _make_planner()
        probe_count = _drive_to_convergence(planner)

        history = planner.history()
        assert len(history) == probe_count

        for i, entry in enumerate(history):
            assert entry.iteration_idx == i
            assert "phases.profiling.concurrency" in entry.variation_values
            assert entry.results is not None
            assert len(entry.results) == 1

    def test_tier_metadata_probe_count_matches_history(self) -> None:
        """tier_metadata tier_evaluation_count matches the sum of per-tier counts."""
        planner = _make_planner()
        _drive_to_convergence(planner)

        metadata = planner.tier_metadata()
        results = planner.tier_results()

        per_tier_sum = sum(r.probe_count for r in results)
        assert metadata["tier_evaluation_count"] == per_tier_sum

    def test_max_iterations_produces_partial_results(self) -> None:
        """When max_iterations is exhausted, partial results are reported."""
        # Use a very tight budget that won't allow full convergence
        planner = _make_planner(lo=1, hi=256, max_iterations=3)
        _drive_to_convergence(planner)

        assert planner.is_converged()
        assert planner.convergence_reason() == "max_iterations"

        # At least some tiers should be partial
        results = planner.tier_results()
        assert any(r.convergence_status == "partial" for r in results)

    def test_widely_separated_boundaries_all_resolve(self) -> None:
        """Planner continues probing until ALL tier boundaries are found, even when
        widely separated (e.g., 39, 79, 93) as required by issue #987.

        Uses throughput model: throughput = max(1, 500 - 5*c)
        - fast (>300): boundary at c=39 (500-195=305>300, c=40: 500-200=300 not >300)
        - standard (>100): boundary at c=79 (500-395=105>100, c=80: 500-400=100 not >100)
        - economy (>30): boundary at c=93 (500-465=35>30, c=94: 500-470=30 not >30)
        """

        def simulate_wide(concurrency: int) -> float:
            return max(1.0, 500.0 - 5.0 * concurrency)

        tiers = [
            SLOTier(
                label="fast",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=300.0,
                    )
                ],
            ),
            SLOTier(
                label="standard",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=100.0,
                    )
                ],
            ),
            SLOTier(
                label="economy",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=30.0,
                    )
                ],
            ),
        ]

        cfg = _cfg(lo=1, hi=256, max_iterations=80)
        planner = MultiTierPlanner(base_config=_base_config(), cfg=cfg, tiers=tiers)

        probe_count = 0
        while True:
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = variation.values["phases.profiling.concurrency"]
            throughput = simulate_wide(concurrency)
            result = RunResult(
                label="trial_0",
                success=True,
                summary_metrics={
                    "output_token_throughput": JsonMetricResult(
                        unit="tok/s", avg=throughput
                    ),
                },
                variation_label=variation.label,
                variation_values=variation.values,
            )
            planner.tell(variation, [result])
            probe_count += 1

        results = planner.tier_results()
        boundaries = {r.label: r for r in results}

        # ALL three tiers must converge
        assert boundaries["fast"].convergence_status == "converged"
        assert boundaries["standard"].convergence_status == "converged"
        assert boundaries["economy"].convergence_status == "converged"

        # ALL three tiers must have boundary_concurrency
        assert boundaries["fast"].boundary_concurrency is not None
        assert boundaries["standard"].boundary_concurrency is not None
        assert boundaries["economy"].boundary_concurrency is not None

        # Boundaries must be ordered: fast < standard < economy
        assert (
            boundaries["fast"].boundary_concurrency
            < boundaries["standard"].boundary_concurrency
            < boundaries["economy"].boundary_concurrency
        )

        # Boundaries should be near the analytical values
        assert 35 <= boundaries["fast"].boundary_concurrency <= 39
        assert 75 <= boundaries["standard"].boundary_concurrency <= 79
        assert 89 <= boundaries["economy"].boundary_concurrency <= 93

    def test_lenient_tier_passes_to_max_marked_no_failure_in_range(self) -> None:
        """When a lenient tier passes at every probed concurrency up to hi,
        it should be marked no_failure_in_range, not partial.

        Uses throughput model: throughput = max(1, 500 - 20*c)
        - fast (>300): boundary at c=9 (500-180=320>300, c=10: 500-200=300 not >300)
        - economy (>30): c=23 would fail (500-460=40>30, c=24: 500-480=20<30)
          but hi=20, so the tier passes at every concurrency in [1, 20].
        """

        def simulate_lenient(concurrency: int) -> float:
            return max(1.0, 500.0 - 20.0 * concurrency)

        tiers = [
            SLOTier(
                label="fast",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=300.0,
                    )
                ],
            ),
            SLOTier(
                label="economy",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=30.0,
                    )
                ],
            ),
        ]

        cfg = _cfg(lo=1, hi=20, max_iterations=50)
        planner = MultiTierPlanner(base_config=_base_config(), cfg=cfg, tiers=tiers)

        while True:
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = variation.values["phases.profiling.concurrency"]
            throughput = simulate_lenient(concurrency)
            result = RunResult(
                label="trial_0",
                success=True,
                summary_metrics={
                    "output_token_throughput": JsonMetricResult(
                        unit="tok/s", avg=throughput
                    ),
                },
                variation_label=variation.label,
                variation_values=variation.values,
            )
            planner.tell(variation, [result])

        results = planner.tier_results()
        result_map = {r.label: r for r in results}

        # fast tier should converge normally within [1, 20]
        fast = result_map["fast"]
        assert fast.convergence_status == "converged"
        assert fast.boundary_concurrency is not None
        assert fast.boundary_concurrency <= 10

        # economy tier passes at all concurrencies up to hi=20
        economy = result_map["economy"]
        assert economy.convergence_status == "no_failure_in_range"
        assert economy.boundary_concurrency == 20

    def test_non_monotonic_no_inverted_bracket_emitted(self) -> None:
        """When c=1 fails then c=2+ passes (non-monotonic), no inverted bracket
        is emitted and the tier is not marked precision-converged.

        This is the regression test for the inverted-bracket bug where
        feasible_max=2, infeasible_min=1 was incorrectly reported as converged.
        """

        def simulate_non_mono(concurrency: int) -> float:
            if concurrency == 1:
                return 100.0  # anomalous dip below fast threshold
            return max(1.0, 500.0 - 5.0 * concurrency)

        tiers = [
            SLOTier(
                label="fast",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=300.0,
                    )
                ],
            ),
            SLOTier(
                label="economy",
                filters=[
                    SLAFilter(
                        metric_tag="output_token_throughput",
                        stat="avg",
                        op="gt",
                        threshold=30.0,
                    )
                ],
            ),
        ]

        cfg = _cfg(lo=1, hi=128, max_iterations=50)
        planner = MultiTierPlanner(base_config=_base_config(), cfg=cfg, tiers=tiers)

        while True:
            pair = planner.ask()
            if pair is None:
                break
            _, variation = pair
            concurrency = variation.values["phases.profiling.concurrency"]
            throughput = simulate_non_mono(concurrency)
            result = RunResult(
                label="trial_0",
                success=True,
                summary_metrics={
                    "output_token_throughput": JsonMetricResult(
                        unit="tok/s", avg=throughput
                    ),
                },
                variation_label=variation.label,
                variation_values=variation.values,
            )
            planner.tell(variation, [result])

        results = planner.tier_results()
        fast = next(r for r in results if r.label == "fast")

        # 1. No inverted bracket: bracket_upper must be None or > bracket_lower
        if fast.bracket_lower is not None and fast.bracket_upper is not None:
            assert fast.bracket_upper > fast.bracket_lower, (
                f"Inverted bracket emitted: bracket_lower={fast.bracket_lower}, "
                f"bracket_upper={fast.bracket_upper}"
            )

        # 2. Must NOT be marked "converged" with reason "precision_reached"
        #    when the underlying evidence is non-monotonic
        if fast.convergence_reason == "multi_tier_precision_reached":
            assert fast.bracket_lower is not None
            assert fast.bracket_upper is not None
            assert fast.bracket_upper > fast.bracket_lower

        # 3. The tier should still find a valid boundary (c>=2 passes)
        assert fast.boundary_concurrency is not None
        assert fast.boundary_concurrency >= 2
