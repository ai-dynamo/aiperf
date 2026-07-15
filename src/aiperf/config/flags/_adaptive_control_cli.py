# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive-scale compact CLI parsing helpers."""

from __future__ import annotations

from collections.abc import Set
from typing import Any

EXPANDED_ADAPTIVE_CONTROL_FIELDS = frozenset(
    {
        "adaptive_control_variable",
        "adaptive_control_min",
        "adaptive_control_max",
    }
)


def reject_mixed_adaptive_control_cli(fields_set: Set[str]) -> None:
    if "adaptive_scale_control" not in fields_set:
        return
    if EXPANDED_ADAPTIVE_CONTROL_FIELDS & fields_set:
        raise ValueError(
            "Use either --adaptive-scale-control or "
            "--adaptive-control-variable/--adaptive-control-min/"
            "--adaptive-control-max, not both."
        )


def parse_adaptive_scale_control(value: str) -> dict[str, Any]:
    """Parse ``variable:min,max:type`` into canonical adaptive control fields."""
    try:
        variable, bounds, value_type = value.split(":", 2)
        minimum_text, maximum_text = bounds.split(",", 1)
    except ValueError as exc:
        raise ValueError(
            "--adaptive-scale-control must use variable:min,max:type, "
            "for example concurrency:1,1000:int"
        ) from exc

    variable = variable.strip()
    value_type = value_type.strip()
    if not variable:
        raise ValueError("--adaptive-scale-control requires a control variable")

    if value_type == "int":
        minimum = _parse_int_bound(minimum_text, "min")
        maximum = _parse_int_bound(maximum_text, "max")
    elif value_type == "float":
        minimum = _parse_float_bound(minimum_text, "min")
        maximum = _parse_float_bound(maximum_text, "max")
    else:
        raise ValueError(
            "--adaptive-scale-control type must be 'int' or 'float', "
            f"got {value_type!r}"
        )

    return {
        "adaptive_control_variable": variable,
        "adaptive_control_min": minimum,
        "adaptive_control_max": maximum,
    }


def _parse_int_bound(value: str, name: str) -> int:
    value = value.strip()
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"--adaptive-scale-control {name} bound must be an integer"
        ) from exc


def _parse_float_bound(value: str, name: str) -> float:
    value = value.strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"--adaptive-scale-control {name} bound must be a number"
        ) from exc
