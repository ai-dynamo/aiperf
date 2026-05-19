# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Pareto-dominance helpers used by the export post-process and the live tracker."""

from __future__ import annotations

import math
import warnings
from typing import Any


def quarantine_non_finite(
    cells: list[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
) -> list[dict[str, Any]]:
    """Split cells into the finite-axis subset that participates in dominance.

    Cells with non-finite x or y break the dominance pass: NaN comparisons
    always return False, so a NaN-axis cell is never dominated AND never
    dominates, falsely floating to the Pareto frontier. Mark them
    ``pareto_optimal=False`` in place and return only the finite-axis cells.
    Emits a ``UserWarning`` when any cell was quarantined.
    """
    finite_cells: list[dict[str, Any]] = []
    excluded = 0
    for c in cells:
        if math.isfinite(c["x"]) and math.isfinite(c["y"]):
            finite_cells.append(c)
        else:
            c["pareto_optimal"] = False
            excluded += 1
    if excluded:
        warnings.warn(
            f"pareto_dominance: excluded {excluded} cell(s) with non-finite x "
            f"({x_label}) or y ({y_label}) from the dominance pass; these "
            "cells are emitted with pareto_optimal=False. A failing sweep "
            "cell (zero successful requests) is the typical cause.",
            UserWarning,
            stacklevel=2,
        )
    return finite_cells


def mark_pareto_optimal(
    finite_cells: list[dict[str, Any]],
    *,
    x_minimize: bool = True,
    y_maximize: bool = True,
) -> None:
    """Stamp ``pareto_optimal=True`` on each non-dominated cell in-place.

    Direction flags pick the dominance relation:
        x_minimize=True, y_maximize=True (default):
            cell A dominates B iff A.x <= B.x AND A.y >= B.y, strict in one.
        x_minimize=False: invert the x relation.
        y_maximize=False: invert the y relation.

    O(n^2); fine for sweep sizes (typically < a few hundred cells).
    """

    def _x_better_or_eq(a: float, b: float) -> bool:
        return a <= b if x_minimize else a >= b

    def _y_better_or_eq(a: float, b: float) -> bool:
        return a >= b if y_maximize else a <= b

    def _x_strictly_better(a: float, b: float) -> bool:
        return a < b if x_minimize else a > b

    def _y_strictly_better(a: float, b: float) -> bool:
        return a > b if y_maximize else a < b

    for i, c in enumerate(finite_cells):
        dominated = False
        for j, other in enumerate(finite_cells):
            if i == j:
                continue
            if (
                _x_better_or_eq(other["x"], c["x"])
                and _y_better_or_eq(other["y"], c["y"])
                and (
                    _x_strictly_better(other["x"], c["x"])
                    or _y_strictly_better(other["y"], c["y"])
                )
            ):
                dominated = True
                break
        c["pareto_optimal"] = not dominated
