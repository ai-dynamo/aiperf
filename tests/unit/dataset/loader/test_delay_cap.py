# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.dataset.loader._delay_cap import (
    DelayCapTracker,
    clamp_inter_turn_delay_ms,
)


@pytest.mark.parametrize(
    "delay_ms, cap_seconds, expected",
    [
        (500.0, 1.0, 500.0),
        (1500.0, 1.0, 1000.0),
        (1500.0, None, 1500.0),
        (-50.0, 1.0, -50.0),
        (None, 1.0, None),
        (None, None, None),
        (float("nan"), 1.0, None),
        (float("nan"), None, None),
        (float("inf"), 1.0, None),
        (float("-inf"), 60.0, None),
    ],
)
def test_clamp_inter_turn_delay_ms_table(delay_ms, cap_seconds, expected):
    assert clamp_inter_turn_delay_ms(delay_ms, cap_seconds) == expected


def test_tracker_no_cap_passthrough():
    tracker = DelayCapTracker(cap_seconds=None)
    assert tracker.clamp(5_000.0) == 5_000.0
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 5_000.0


def test_tracker_under_cap_passthrough():
    tracker = DelayCapTracker(cap_seconds=60.0)
    assert tracker.clamp(30_000.0) == 30_000.0
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 30_000.0


def test_tracker_over_cap_clamps_and_counts():
    tracker = DelayCapTracker(cap_seconds=60.0)
    assert tracker.clamp(120_000.0) == 60_000.0
    assert tracker.clamp(180_000.0) == 60_000.0
    assert tracker.capped_count == 2
    assert tracker.max_observed_ms == 180_000.0


def test_tracker_none_input_passthrough():
    tracker = DelayCapTracker(cap_seconds=60.0)
    assert tracker.clamp(None) is None
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 0.0


def test_tracker_non_finite_maps_to_none_and_counts():
    tracker = DelayCapTracker(cap_seconds=60.0)
    assert tracker.clamp(float("nan")) is None
    assert tracker.clamp(float("inf")) is None
    assert tracker.clamp(float("-inf")) is None
    assert tracker.non_finite_count == 3
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 0.0


def test_parent_floor_after_clamp_skips_none_for_non_finite():
    """Parent reconstruct floors after clamp; must not ``max(None, 0.0)``."""
    tracker = DelayCapTracker(cap_seconds=60.0)
    delay_ms: float | None = float("nan")
    delay_ms = tracker.clamp(delay_ms)
    if delay_ms is not None:
        delay_ms = max(delay_ms, 0.0)
    assert delay_ms is None

    delay_ms = tracker.clamp(-50.0)
    assert delay_ms == -50.0
    if delay_ms is not None:
        delay_ms = max(delay_ms, 0.0)
    assert delay_ms == 0.0


def test_tracker_log_summary_warns_on_non_finite(caplog):
    tracker = DelayCapTracker(cap_seconds=60.0)
    tracker.clamp(float("nan"))
    with caplog.at_level(logging.WARNING, logger="aiperf"):
        tracker.log_summary(logger_name="aiperf.test")
    assert any("non-finite inter-turn" in r.message for r in caplog.records)


def test_tracker_log_summary_emits_when_capped(caplog):
    tracker = DelayCapTracker(cap_seconds=60.0)
    tracker.clamp(120_000.0)
    tracker.clamp(90_000.0)
    with caplog.at_level(logging.INFO, logger="aiperf"):
        tracker.log_summary(logger_name="aiperf.test")
    assert any("Capped 2 inter-turn" in r.message for r in caplog.records)
    assert any("max observed" in r.message for r in caplog.records)


def test_tracker_log_summary_silent_when_no_caps(caplog):
    tracker = DelayCapTracker(cap_seconds=60.0)
    tracker.clamp(30_000.0)
    with caplog.at_level(logging.INFO, logger="aiperf"):
        tracker.log_summary(logger_name="aiperf.test")
    assert not any("Capped" in r.message for r in caplog.records)


def test_tracker_log_summary_silent_when_cap_none(caplog):
    tracker = DelayCapTracker(cap_seconds=None)
    with caplog.at_level(logging.INFO, logger="aiperf"):
        tracker.log_summary(logger_name="aiperf.test")
    assert not caplog.records


def test_tracker_reset_clears_counters():
    tracker = DelayCapTracker(cap_seconds=60.0)
    tracker.clamp(120_000.0)
    tracker.clamp(float("nan"))
    tracker.reset()
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 0.0
    assert tracker.non_finite_count == 0
