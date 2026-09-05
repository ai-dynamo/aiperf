# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import CacheBustTarget


def test_cache_bust_target_values():
    assert CacheBustTarget.NONE == "none"
    assert CacheBustTarget.SYSTEM_PREFIX == "system_prefix"
    assert CacheBustTarget.SYSTEM_SUFFIX == "system_suffix"
    assert CacheBustTarget.FIRST_TURN_PREFIX == "first_turn_prefix"
    assert CacheBustTarget.FIRST_TURN_SUFFIX == "first_turn_suffix"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("NONE", CacheBustTarget.NONE),
        ("System_Prefix", CacheBustTarget.SYSTEM_PREFIX),
        ("FIRST_TURN_PREFIX", CacheBustTarget.FIRST_TURN_PREFIX),
    ],
)
def test_cache_bust_target_case_insensitive(raw, expected):
    assert CacheBustTarget(raw) == expected


def test_cache_bust_target_default_is_none():
    assert CacheBustTarget.NONE.value == "none"


def test_warmup_isolation_system_in_cache_bust_target():
    assert (
        CacheBustTarget("warmup_isolation_system")
        == CacheBustTarget.WARMUP_ISOLATION_SYSTEM
    )


def test_warmup_isolation_first_turn_in_cache_bust_target():
    assert (
        CacheBustTarget("warmup_isolation_first_turn")
        == CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN
    )
