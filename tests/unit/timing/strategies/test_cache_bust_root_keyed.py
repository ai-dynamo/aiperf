# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the root-keyed (per trajectory tree) cache-bust marker resolver."""

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.timing.strategies.cache_bust import base_trace_id, resolve_tree_marker
from aiperf.timing.trajectory_source import CacheBustLedger

T = CacheBustTarget.FIRST_TURN_PREFIX


def _led() -> CacheBustLedger:
    return CacheBustLedger()


@pytest.mark.parametrize(
    "conv,expected",
    [
        ("abc123", "abc123"),
        ("abc123::sa:subagent_001_dead", "abc123"),
        ("abc123::fa:000", "abc123"),
        ("abc123::sa:subagent_001_dead:s1", "abc123"),
    ],
)
def test_base_trace_id_strips_descendant_suffix(conv, expected):
    assert base_trace_id(conv) == expected


def test_tree_members_share_one_marker():
    led = _led()
    root = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    child = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc::fa:000",
        target=T,
    )
    assert root is not None
    assert root == child


def test_idempotent_does_not_bump_recycle_pass():
    led = _led()
    first = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    pass_after_first = led.recycle_pass["abc"]
    again = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    assert again == first
    assert led.recycle_pass["abc"] == pass_after_first


def test_order_independent_child_resolves_first():
    led = _led()
    child = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc::sa:subagent_001_dead",
        target=T,
    )
    root = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    assert root is not None
    assert root == child


def test_distinct_across_lane_and_recycle():
    led = _led()
    lane0 = resolve_tree_marker(
        led,
        "R0",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    lane1 = resolve_tree_marker(
        led,
        "R1",
        benchmark_id="b",
        trajectory_index=1,
        conversation_id="abc",
        target=T,
    )
    recyc = resolve_tree_marker(
        led,
        "R2",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=T,
    )
    assert lane0 != lane1
    assert lane0 != recyc
    assert lane1 != recyc


def test_none_target_records_none():
    led = _led()
    marker = resolve_tree_marker(
        led,
        "ROOT",
        benchmark_id="b",
        trajectory_index=0,
        conversation_id="abc",
        target=CacheBustTarget.NONE,
    )
    assert marker is None
    assert "ROOT" in led.session_marker
    assert led.session_marker["ROOT"] is None


def test_deterministic_across_ledgers():
    a = resolve_tree_marker(
        _led(),
        "ROOT",
        benchmark_id="b",
        trajectory_index=3,
        conversation_id="abc::fa:002",
        target=T,
    )
    b = resolve_tree_marker(
        _led(),
        "OTHER",
        benchmark_id="b",
        trajectory_index=3,
        conversation_id="abc",
        target=T,
    )
    assert a == b
