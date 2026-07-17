# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property test: Warmup Sharing.

Feature: multi-tier-slo-search, Property 10: Warmup Sharing

Validates: Requirements 5.1, 5.2, 5.3

For any concurrency level X probed in a multi-tier search, the warmup phase
SHALL execute exactly once on the first probe at X (using
FIRST_PROBE_WARMUP_FLOOR) and use the reduced replicate floor on all
subsequent probes at X, regardless of which tier requested the probe.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from aiperf.common.environment import Environment
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection
from aiperf.orchestrator.search_planner._shared_warmup import apply_sla_warmup

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _concurrency_sequences() -> st.SearchStrategy[list[int]]:
    """Generate random sequences of concurrency values with repeats.

    Uses a small range to encourage duplicates, simulating multiple
    probes at the same concurrency from different tiers.
    """
    return st.lists(
        st.integers(min_value=1, max_value=20),
        min_size=2,
        max_size=30,
    )


def _make_cfg(sla_warmup_seconds: float | None = None) -> AdaptiveSearchSweep:
    """Create a minimal AdaptiveSearchSweep for warmup testing."""
    return AdaptiveSearchSweep(
        planner="smooth_isotonic",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency",
                lo=1,
                hi=1000,
                kind="int",
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=30,
        n_initial_points=1,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ],
        sla_warmup_seconds=sla_warmup_seconds,
    )


def _make_cfg_dict(concurrency: int) -> dict[str, Any]:
    """Create a minimal cfg_dict with a profiling phase."""
    return {
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": concurrency,
                "duration": 60.0,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Property 10: Warmup Sharing
# ---------------------------------------------------------------------------


class TestProperty10WarmupSharing:
    """Property 10: Warmup Sharing.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """

    @given(sequence=_concurrency_sequences())
    @settings(max_examples=100, deadline=None)
    def test_first_probe_at_tracks_all_distinct_concurrency_values(
        self,
        sequence: list[int],
    ) -> None:
        """first_probe_at set contains exactly the distinct concurrency values.

        After processing a sequence of probes, the shared warmup tracker
        records every distinct concurrency level exactly once regardless
        of how many probes hit that level.

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        first_probe_at: set[int] = set()
        cfg = _make_cfg(sla_warmup_seconds=None)

        for value in sequence:
            cfg_dict = _make_cfg_dict(value)
            apply_sla_warmup(cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at)

        expected_distinct = set(sequence)
        assert first_probe_at == expected_distinct

    @given(sequence=_concurrency_sequences())
    @settings(max_examples=100, deadline=None)
    def test_first_probe_gets_first_probe_warmup_floor(
        self,
        sequence: list[int],
    ) -> None:
        """First probe at each concurrency uses FIRST_PROBE_WARMUP_FLOOR.

        The first time a concurrency level is encountered, the warmup
        duration is at least FIRST_PROBE_WARMUP_FLOOR (60s default).

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        first_probe_at: set[int] = set()
        cfg = _make_cfg(sla_warmup_seconds=None)
        seen: set[int] = set()

        for value in sequence:
            if value not in seen:
                cfg_dict = _make_cfg_dict(value)
                apply_sla_warmup(
                    cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at
                )
                warmup_phase = cfg_dict["phases"][0]
                assert warmup_phase["name"] == "warmup"
                assert (
                    warmup_phase["duration"]
                    >= Environment.SEARCH_PLANNER.FIRST_PROBE_WARMUP_FLOOR
                )
                seen.add(value)

    @given(sequence=_concurrency_sequences())
    @settings(max_examples=100, deadline=None)
    def test_subsequent_probes_get_replicate_warmup_floor(
        self,
        sequence: list[int],
    ) -> None:
        """Subsequent probes at same concurrency use REPLICATE_WARMUP_FLOOR.

        After the first probe at a concurrency level, all further probes
        at that level receive the reduced replicate warmup duration.

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        first_probe_at: set[int] = set()
        cfg = _make_cfg(sla_warmup_seconds=None)
        seen: set[int] = set()

        for value in sequence:
            cfg_dict = _make_cfg_dict(value)
            apply_sla_warmup(cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at)

            warmup_phase = cfg_dict["phases"][0]
            if value in seen:
                # Replicate probe: should use replicate floor
                assert warmup_phase["name"] == "warmup"
                expected_duration = max(
                    Environment.SEARCH_PLANNER.REPLICATE_WARMUP_FLOOR,
                    Environment.SEARCH_PLANNER.DEFAULT_WARMUP_SECONDS,
                )
                assert warmup_phase["duration"] == expected_duration
            else:
                seen.add(value)

    @given(
        sequence=_concurrency_sequences(),
        warmup_seconds=st.floats(min_value=0.1, max_value=200.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_warmup_duration_differs_between_first_and_replicate(
        self,
        sequence: list[int],
        warmup_seconds: float,
    ) -> None:
        """First probe warmup >= replicate warmup for the same concurrency.

        For any user-specified sla_warmup_seconds, the first probe at a
        concurrency gets max(FIRST_PROBE_WARMUP_FLOOR, sla_warmup_seconds)
        while replicates get max(REPLICATE_WARMUP_FLOOR, sla_warmup_seconds).

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        first_probe_at: set[int] = set()
        cfg = _make_cfg(sla_warmup_seconds=warmup_seconds)
        first_durations: dict[int, float] = {}
        replicate_durations: dict[int, list[float]] = {}

        for value in sequence:
            cfg_dict = _make_cfg_dict(value)
            apply_sla_warmup(cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at)

            warmup_phase = cfg_dict["phases"][0]
            if warmup_phase["name"] != "warmup":
                continue

            if value not in first_durations:
                first_durations[value] = warmup_phase["duration"]
            else:
                replicate_durations.setdefault(value, []).append(
                    warmup_phase["duration"]
                )

        # First probe duration >= replicate duration for each concurrency
        for value, replicate_list in replicate_durations.items():
            first_dur = first_durations[value]
            for rep_dur in replicate_list:
                assert first_dur >= rep_dur

    @given(sequence=_concurrency_sequences())
    @settings(max_examples=100, deadline=None)
    def test_warmup_zero_opt_out_still_tracks_first_probe(
        self,
        sequence: list[int],
    ) -> None:
        """Explicit sla_warmup_seconds=0 skips warmup but still tracks probes.

        Even with warmup opt-out, the first_probe_at set is populated
        so the planner knows which concurrency levels have been visited.

        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        first_probe_at: set[int] = set()
        cfg = _make_cfg(sla_warmup_seconds=0)

        for value in sequence:
            cfg_dict = _make_cfg_dict(value)
            apply_sla_warmup(cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at)

        expected_distinct = set(sequence)
        assert first_probe_at == expected_distinct
