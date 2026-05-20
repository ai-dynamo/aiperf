# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared metric helpers for Dynamo chaos scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def metric_delta(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
    key: str,
    *,
    floor_at_zero: bool = False,
) -> float:
    """Return a metric value delta, treating missing samples as zero."""
    delta = float(after.get(key, 0.0) - before.get(key, 0.0))
    if floor_at_zero:
        return max(delta, 0.0)
    return delta
