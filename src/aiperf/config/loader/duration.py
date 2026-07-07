# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duration parsing helpers shared by phase configuration types."""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

_DURATION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(s|sec|m|min|h|hr|hour)?$", re.IGNORECASE
)


def _parse_duration(v: Any) -> float | None:
    """Parse duration from various formats to seconds.

    Supports:
        - Numbers: 30, 5.5 (interpreted as seconds)
        - Strings: "30s", "5m", "2h", "30 sec", "5 min", "1hr", "1hour"
          (accepted unit suffixes: ``s|sec|m|min|h|hr|hour``,
          case-insensitive)
        - "inf" / "infinity" (case-insensitive): wait indefinitely
          (used by ``--benchmark-grace-period``)

    Returns:
        Duration in seconds, or None if input is None.

    Raises:
        ValueError: If string format is invalid.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v.strip().lower() in ("inf", "infinity"):
            return float("inf")
        match = _DURATION_PATTERN.match(v.strip())
        if not match:
            raise ValueError(
                f"Invalid duration format: {v!r}. Use number, '30s', '5m', '2h', or 'inf'."
            )
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit in ("s", "sec"):
            return value
        elif unit in ("m", "min"):
            return value * 60
        elif unit in ("h", "hr", "hour"):
            return value * 3600
    return v


def _normalize_duration(v: Any) -> Any:
    """Normalize duration fields to float seconds."""
    return _parse_duration(v)


# Type alias for duration that supports shorthand strings
DurationSpec = Annotated[float | None, BeforeValidator(_normalize_duration)]

# Bounded variants. The numeric constraint is applied to the ``float`` member
# of the union, NOT the whole ``float | None`` — otherwise an explicit ``None``
# (e.g. ``duration: null`` in YAML or a CRD spec) runs the gt/ge validator on
# None and raises a bare ``TypeError`` that escapes callers catching only
# ``ValueError`` (the operator's ``_validate_spec``). With the constraint on
# the float branch, ``None`` matches the None branch cleanly and a real value
# still validates.
PositiveDurationSpec = Annotated[
    Annotated[float, Field(gt=0)] | None, BeforeValidator(_normalize_duration)
]
NonNegativeDurationSpec = Annotated[
    Annotated[float, Field(ge=0)] | None, BeforeValidator(_normalize_duration)
]
