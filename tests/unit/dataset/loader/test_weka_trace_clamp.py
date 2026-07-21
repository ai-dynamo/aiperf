# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.dataset.loader.weka_trace import _clamp_delay_ms


def test_clamp_under_cap_passes_through():
    assert _clamp_delay_ms(50_000.0, cap_seconds=60.0) == 50_000.0


def test_clamp_at_cap_inclusive_unchanged():
    assert _clamp_delay_ms(60_000.0, cap_seconds=60.0) == 60_000.0


def test_clamp_above_cap_clamps():
    assert _clamp_delay_ms(60_000.001, cap_seconds=60.0) == 60_000.0


def test_clamp_none_cap_passes_through():
    assert _clamp_delay_ms(86_400_000.0, cap_seconds=None) == 86_400_000.0


def test_clamp_negative_passes_through():
    # Clamp only enforces upper bound; corrupt-trace negatives pass through.
    assert _clamp_delay_ms(-100.0, cap_seconds=60.0) == -100.0


def test_clamp_zero_cap_clamps_everything():
    assert _clamp_delay_ms(1.0, cap_seconds=0.0) == 0.0
    assert _clamp_delay_ms(0.0, cap_seconds=0.0) == 0.0


def test_clamp_inf_clamps():
    assert _clamp_delay_ms(float("inf"), cap_seconds=60.0) == 60_000.0
