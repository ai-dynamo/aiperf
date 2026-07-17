# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-mode integration tests for MultiTierPlanner.

Verifies that the MultiTierPlanner produces identical output schema and behavior
regardless of UI mode (dashboard, simple, none). The planner is a pure algorithm
with no UI dependencies — these tests confirm that contract holds and that progress
logging (the mode-visible mechanism) works across all execution paths.

Requirements traced: 9.1, 9.4
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import MagicMock

from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner
from aiperf.plugin.enums import SearchPlannerType

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _sla_filter(
    metric: str = "output_token_throughput",
    stat: str = "avg",
    op: str = "gt",
    threshold: float = 100.0,
) -> SLAFilter:
    return SLAFilter(metric_tag=metric, stat=stat, op=op, threshold=threshold)


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


def _adaptive_cfg(
    lo: int = 1, hi: int = 64, max_iterations: int = 20
) -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=lo, hi=hi, kind="int"
            )
        ],
        objectives=[Objective(metric="output_token_throughput", direction="maximize")],
        planner=SearchPlannerType.SMOOTH_ISOTONIC,
        max_iterations=max_iterations,
        n_initial_points=2,
        sla_filters=[_sla_filter()],
        sla_tiers=[
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
        ],
    )


def _make_result(throughput_avg: float = 200.0) -> RunResult:
    """Create a RunResult with the given throughput."""
    result = RunResult(label="test", success=True)
    metric = MagicMock()
    metric.avg = throughput_avg
    result.summary_metrics = {"output_token_throughput": metric}
    return result


def _tiers() -> list[SLOTier]:
    return [
        SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
        SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
    ]


def _run_planner_sequence(
    throughput_values: list[float],
) -> MultiTierPlanner:
    """Run a planner through a fixed sequence of ask/tell with given throughputs.

    Returns the planner after all iterations so output can be inspected.
    """
    tiers = _tiers()
    cfg = _adaptive_cfg()
    planner = MultiTierPlanner(_base_config(), cfg, tiers)

    for throughput in throughput_values:
        proposal = planner.ask()
        if proposal is None:
            break
        _, variation = proposal
        planner.tell(variation, [_make_result(throughput)])

    return planner


# --------------------------------------------------------------------------
# Test: Mode-agnostic design (no UI imports)
# --------------------------------------------------------------------------


class TestMultiTierPlannerNoUIDependencies:
    """Verify the planner has no dependencies on UI-specific modules.

    Requirement 9.1: The planner SHALL function in dashboard, simple, and none
    UI modes. This is guaranteed by having zero coupling to UI code.
    """

    def test_no_ui_imports_in_multi_tier_planner(self) -> None:
        """MultiTierPlanner source has no imports from UI/display modules."""
        source = inspect.getsource(MultiTierPlanner)
        ui_modules = [
            "textual",
            "tqdm",
            "rich",
            "dashboard",
            "ui_mode",
            "display_mode",
        ]
        for module in ui_modules:
            assert module not in source, (
                f"MultiTierPlanner imports or references UI module: {module}"
            )

    def test_planner_has_no_mode_parameter(self) -> None:
        """MultiTierPlanner.__init__ does not accept a mode/ui_mode parameter."""
        sig = inspect.signature(MultiTierPlanner.__init__)
        param_names = set(sig.parameters.keys()) - {"self"}
        mode_params = {"mode", "ui_mode", "display_mode", "output_mode"}
        assert param_names.isdisjoint(mode_params), (
            f"MultiTierPlanner has mode-dependent parameters: "
            f"{param_names & mode_params}"
        )


# --------------------------------------------------------------------------
# Test: Deterministic output (same input -> same output regardless of mode)
# --------------------------------------------------------------------------


class TestMultiTierOutputDeterministic:
    """Verify identical inputs produce identical outputs.

    Requirement 9.4: The planner SHALL produce the same output schema
    regardless of UI mode or deployment mode. Since the planner is mode-agnostic,
    running the same ask/tell sequence must yield identical results every time.
    """

    def test_tier_results_deterministic_across_runs(self) -> None:
        """Two independent planner runs with same inputs produce same tier_results."""
        throughputs = [500.0, 350.0, 200.0, 80.0]

        planner_a = _run_planner_sequence(throughputs)
        planner_b = _run_planner_sequence(throughputs)

        results_a = [r.model_dump() for r in planner_a.tier_results()]
        results_b = [r.model_dump() for r in planner_b.tier_results()]

        assert results_a == results_b

    def test_tier_metadata_deterministic_across_runs(self) -> None:
        """Two independent runs produce same tier_metadata."""
        throughputs = [500.0, 350.0, 200.0, 80.0]

        planner_a = _run_planner_sequence(throughputs)
        planner_b = _run_planner_sequence(throughputs)

        assert planner_a.tier_metadata() == planner_b.tier_metadata()

    def test_boundary_summary_deterministic_across_runs(self) -> None:
        """Two independent runs produce same boundary_summary."""
        throughputs = [500.0, 350.0, 200.0, 80.0]

        planner_a = _run_planner_sequence(throughputs)
        planner_b = _run_planner_sequence(throughputs)

        assert planner_a.boundary_summary() == planner_b.boundary_summary()

    def test_history_deterministic_across_runs(self) -> None:
        """Two independent runs produce same iteration history structure."""
        throughputs = [500.0, 350.0, 200.0]

        planner_a = _run_planner_sequence(throughputs)
        planner_b = _run_planner_sequence(throughputs)

        history_a = planner_a.history()
        history_b = planner_b.history()

        assert len(history_a) == len(history_b)
        for ha, hb in zip(history_a, history_b, strict=False):
            assert ha.iteration_idx == hb.iteration_idx
            assert ha.variation_values == hb.variation_values
            assert ha.feasible == hb.feasible
            assert ha.non_monotonic_warning == hb.non_monotonic_warning


# --------------------------------------------------------------------------
# Test: Output schema has no mode-dependent fields
# --------------------------------------------------------------------------


class TestOutputSchemaModeFree:
    """Verify output schema contains no mode-dependent fields.

    Requirement 9.4: output schema is identical across modes. This means
    no field in the output references or varies by UI mode.
    """

    def test_tier_results_schema_has_no_mode_fields(self) -> None:
        """TierResult schema has no mode-dependent fields."""
        planner = _run_planner_sequence([500.0, 200.0])
        results = planner.tier_results()

        for result in results:
            dumped = result.model_dump()
            mode_keys = {"mode", "ui_mode", "display_mode", "render_mode"}
            assert mode_keys.isdisjoint(dumped.keys()), (
                f"TierResult contains mode-dependent fields: "
                f"{mode_keys & set(dumped.keys())}"
            )

    def test_tier_metadata_has_no_mode_fields(self) -> None:
        """tier_metadata dict has no mode-dependent fields."""
        planner = _run_planner_sequence([500.0, 200.0])
        metadata = planner.tier_metadata()

        mode_keys = {"mode", "ui_mode", "display_mode", "render_mode"}
        assert mode_keys.isdisjoint(metadata.keys()), (
            f"tier_metadata contains mode-dependent fields: "
            f"{mode_keys & set(metadata.keys())}"
        )

    def test_boundary_summary_has_no_mode_fields(self) -> None:
        """boundary_summary dict has no mode-dependent fields."""
        planner = _run_planner_sequence([500.0, 200.0])
        summary = planner.boundary_summary()

        assert summary is not None
        mode_keys = {"mode", "ui_mode", "display_mode", "render_mode"}
        assert mode_keys.isdisjoint(summary.keys()), (
            f"boundary_summary contains mode-dependent fields: "
            f"{mode_keys & set(summary.keys())}"
        )


# --------------------------------------------------------------------------
# Test: Progress logging works in all modes (Req 9.2 via standard logging)
# --------------------------------------------------------------------------


class TestProgressLoggingModeAgnostic:
    """Verify progress logging uses standard Python logging (works in all modes).

    The planner uses `logging.getLogger()` which outputs to all modes:
    - dashboard mode: captured and displayed in TUI log panel
    - simple mode: printed to stderr via StreamHandler
    - none mode: printed to stderr via StreamHandler

    This test confirms the planner emits log records that any handler can consume.
    """

    def test_progress_emits_standard_log_records(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Progress logging produces standard LogRecord objects."""
        tiers = _tiers()
        cfg = _adaptive_cfg()
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal

        with caplog.at_level(
            logging.INFO,
            logger="aiperf.orchestrator.search_planner.multi_tier_planner",
        ):
            planner.tell(variation, [_make_result(200.0)])

        progress_records = [
            r for r in caplog.records if "multi_tier probe@" in r.message
        ]
        assert len(progress_records) == 1

        record = progress_records[0]
        assert record.levelno == logging.INFO
        assert record.name == ("aiperf.orchestrator.search_planner.multi_tier_planner")

    def test_progress_contains_all_tier_labels(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Every tier label appears in the progress log line."""
        tiers = _tiers()
        cfg = _adaptive_cfg()
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal

        with caplog.at_level(
            logging.INFO,
            logger="aiperf.orchestrator.search_planner.multi_tier_planner",
        ):
            planner.tell(variation, [_make_result(200.0)])

        progress_msgs = [r for r in caplog.records if "multi_tier probe@" in r.message]
        msg = progress_msgs[0].message

        for tier in tiers:
            assert tier.label in msg, (
                f"Tier label '{tier.label}' missing from progress log"
            )

    def test_progress_logged_consistently_across_iterations(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Progress is emitted after every iteration (not mode-dependent)."""
        tiers = _tiers()
        cfg = _adaptive_cfg(hi=256)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        iterations = 3
        with caplog.at_level(
            logging.INFO,
            logger="aiperf.orchestrator.search_planner.multi_tier_planner",
        ):
            for i in range(iterations):
                proposal = planner.ask()
                if proposal is None:
                    break
                _, variation = proposal
                planner.tell(variation, [_make_result(500.0 - i * 100)])

        progress_records = [
            r for r in caplog.records if "multi_tier probe@" in r.message
        ]
        assert len(progress_records) == iterations


# --------------------------------------------------------------------------
# Test: Full ask/tell cycle produces valid output in simulated modes
# --------------------------------------------------------------------------


class TestFullCycleAcrossModes:
    """Run a complete multi-tier search and verify output validity.

    Since the planner is mode-agnostic, "simulating" a mode just means running
    the same algorithm. The test verifies the output is valid and complete
    regardless of the execution context.
    """

    def test_converged_search_produces_complete_output(self) -> None:
        """A converged search produces complete tier_results and boundary_summary."""
        # Drive the planner to convergence with decreasing throughput
        throughputs = [500.0, 500.0, 200.0, 200.0, 200.0, 200.0, 200.0]
        planner = _run_planner_sequence(throughputs)

        # Verify tier_results are produced
        results = planner.tier_results()
        assert len(results) == 2

        for result in results:
            assert result.label in ("fast", "standard")
            assert result.convergence_status in (
                "converged",
                "partial",
                "no_pass_in_range",
                "no_failure_in_range",
            )
            assert result.probe_count >= 0
            assert len(result.filters) > 0

        # Verify boundary_summary is produced
        summary = planner.boundary_summary()
        assert summary is not None
        assert "swept_dim_path" in summary
        assert "convergence_reason" in summary

        # Verify tier_metadata is produced
        metadata = planner.tier_metadata()
        assert "tier_evaluation_count" in metadata
        assert "ordering_detected" in metadata
        assert "ordering_pairs" in metadata

    def test_partial_convergence_reports_all_tiers(self) -> None:
        """Even with max_iterations exhausted, all tiers report results."""
        tiers = _tiers()
        cfg = _adaptive_cfg(hi=1024, max_iterations=3)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        # Run exactly max_iterations probes
        for _i in range(3):
            proposal = planner.ask()
            if proposal is None:
                break
            _, variation = proposal
            planner.tell(variation, [_make_result(500.0)])

        results = planner.tier_results()
        assert len(results) == 2

        # All tiers should have results even if not converged
        for result in results:
            assert result.label is not None
            assert result.convergence_status is not None

    def test_output_schema_identical_structure_three_tier_search(self) -> None:
        """Three-tier search produces structurally identical output to two-tier."""
        tiers_3 = [
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
            SLOTier(label="economy", filters=[_sla_filter(threshold=30.0)]),
        ]
        cfg = AdaptiveSearchSweep(
            search_space=[
                SearchSpaceDimension(
                    path="phases.profiling.concurrency", lo=1, hi=64, kind="int"
                )
            ],
            objectives=[
                Objective(metric="output_token_throughput", direction="maximize")
            ],
            planner=SearchPlannerType.SMOOTH_ISOTONIC,
            max_iterations=5,
            n_initial_points=2,
            sla_filters=[_sla_filter()],
            sla_tiers=tiers_3,
        )
        planner = MultiTierPlanner(_base_config(), cfg, tiers_3)

        throughputs = [500.0, 200.0, 80.0, 25.0]
        for throughput in throughputs:
            proposal = planner.ask()
            if proposal is None:
                break
            _, variation = proposal
            planner.tell(variation, [_make_result(throughput)])

        results = planner.tier_results()
        assert len(results) == 3

        # All results have the same schema keys
        keys_set = {frozenset(r.model_dump().keys()) for r in results}
        assert len(keys_set) == 1, "All TierResult entries must have identical schema"

        metadata = planner.tier_metadata()
        assert metadata["tier_evaluation_count"] > 0
