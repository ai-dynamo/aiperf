# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test per-tier convergence progress logging in MultiTierPlanner.

Verifies that the planner emits per-tier bracket status after each probe
(Requirement 9.2) so the user can track convergence in simple/none UI modes.
"""

from __future__ import annotations

import logging

from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.search_planner.multi_tier_planner import MultiTierPlanner
from aiperf.plugin.enums import SearchPlannerType


def _sla_filter(
    metric: str = "output_token_throughput", threshold: float = 100.0
) -> SLAFilter:
    return SLAFilter(metric_tag=metric, stat="avg", op="gt", threshold=threshold)


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
    lo: int = 1, hi: int = 128, max_iterations: int = 20
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
    from unittest.mock import MagicMock

    result = RunResult(label="test", success=True)
    metric = MagicMock()
    metric.avg = throughput_avg
    result.summary_metrics = {"output_token_throughput": metric}
    return result


class TestMultiTierProgressLogging:
    """Verify per-tier convergence progress is logged after each probe."""

    def test_progress_logged_after_tell(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """After tell(), the planner logs per-tier bracket status at INFO level."""
        tiers = [
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
        ]
        cfg = _adaptive_cfg()
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal

        with caplog.at_level(
            logging.INFO, logger="aiperf.orchestrator.search_planner.multi_tier_planner"
        ):
            planner.tell(variation, [_make_result(throughput_avg=200.0)])

        # Should log per-tier progress
        progress_msgs = [r for r in caplog.records if "multi_tier probe@" in r.message]
        assert len(progress_msgs) == 1

        msg = progress_msgs[0].message
        assert "fast:" in msg
        assert "standard:" in msg
        assert "tiers converged" in msg

    def test_progress_shows_pass_fail_per_tier(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Progress line shows PASS/FAIL status per tier based on SLA evaluation."""
        tiers = [
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
        ]
        cfg = _adaptive_cfg()
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal

        # throughput=200 passes "standard" (gt:100) but fails "fast" (gt:300)
        with caplog.at_level(
            logging.INFO, logger="aiperf.orchestrator.search_planner.multi_tier_planner"
        ):
            planner.tell(variation, [_make_result(throughput_avg=200.0)])

        progress_msgs = [r for r in caplog.records if "multi_tier probe@" in r.message]
        assert len(progress_msgs) == 1
        msg = progress_msgs[0].message
        assert "fast:FAIL" in msg
        assert "standard:PASS" in msg

    def test_progress_shows_converged_count(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Progress line shows how many tiers have converged."""
        tiers = [
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
        ]
        cfg = _adaptive_cfg()
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal

        with caplog.at_level(
            logging.INFO, logger="aiperf.orchestrator.search_planner.multi_tier_planner"
        ):
            planner.tell(variation, [_make_result(throughput_avg=200.0)])

        progress_msgs = [r for r in caplog.records if "multi_tier probe@" in r.message]
        msg = progress_msgs[0].message
        # Initially no tiers are converged
        assert "0/2 tiers converged" in msg

    def test_progress_logging_works_across_iterations(
        self, caplog: logging.LogCaptureFixture
    ) -> None:
        """Progress is emitted after every tell() call, not just the first."""
        tiers = [
            SLOTier(label="fast", filters=[_sla_filter(threshold=300.0)]),
            SLOTier(label="standard", filters=[_sla_filter(threshold=100.0)]),
        ]
        cfg = _adaptive_cfg(hi=512)
        planner = MultiTierPlanner(_base_config(), cfg, tiers)

        with caplog.at_level(
            logging.INFO, logger="aiperf.orchestrator.search_planner.multi_tier_planner"
        ):
            # First iteration: both pass (high throughput so bracket keeps ramping)
            proposal = planner.ask()
            assert proposal is not None
            _, variation = proposal
            planner.tell(variation, [_make_result(throughput_avg=500.0)])

            # Second iteration: both still pass at doubled value
            proposal = planner.ask()
            assert proposal is not None
            _, variation = proposal
            planner.tell(variation, [_make_result(throughput_avg=500.0)])

        progress_msgs = [r for r in caplog.records if "multi_tier probe@" in r.message]
        assert len(progress_msgs) == 2
