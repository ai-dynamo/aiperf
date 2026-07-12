# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for the official online parity gate."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

_GATE = runpy.run_path(
    str(Path(__file__).parents[2] / "tools" / "dynamo_online_parity_gate.py")
)
_ulp_distance = _GATE["_ulp_distance"]


def test_ulp_distance_accepts_one_binary64_step_only() -> None:
    for value, direction in [(1.0, math.inf), (-1.0, -math.inf)]:
        adjacent = math.nextafter(value, direction)
        second = math.nextafter(adjacent, direction)

        assert _ulp_distance(value, value) == 0
        assert _ulp_distance(value, adjacent) == 1
        assert _ulp_distance(value, second) == 2


def test_ulp_distance_handles_signed_zero_and_non_finite_values() -> None:
    assert _ulp_distance(-0.0, 0.0) == 0
    assert math.isinf(_ulp_distance(1.0, math.inf))
