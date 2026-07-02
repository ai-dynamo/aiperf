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
    ],
)
def test_clamp_inter_turn_delay_ms_table(delay_ms, cap_seconds, expected):
    assert clamp_inter_turn_delay_ms(delay_ms, cap_seconds) == expected


def test_tracker_no_cap_passthrough():
    tracker = DelayCapTracker(cap_seconds=None)
    assert tracker.clamp(5_000.0) == 5_000.0
    assert tracker.capped_count == 0
    # (b) LEGITIMATE v2 BEHAVIOR CHANGE: v1 src/aiperf/dataset/loader/_delay_cap.py:47-48
    # returned delay_ms immediately when cap_seconds is None, never touching
    # max_observed_ms (so it stayed 0.0). v2 src/aiperf/dataset/loader/_delay_cap.py:78-83
    # records max_observed_ms BEFORE the no-cap check so log_summary can warn on
    # absurd uncapped delays (UNCAPPED_DELAY_WARN_MS feature, added in v2). The
    # value passes through unclamped but max_observed_ms now tracks it. Intent
    # (no-cap = no clamping) preserved; the observed-max assertion follows v2.
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
    tracker.reset()
    assert tracker.capped_count == 0
    assert tracker.max_observed_ms == 0.0
