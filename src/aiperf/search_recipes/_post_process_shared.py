# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for post-process handlers.

Hosts the ``StatLiteral`` alias and ``_stat_or_raise`` validator so that
sibling handler modules (``_itl_surface_fit``, ``_sla_breach_knee``, ...)
can use them without importing from ``post_process.py`` and triggering a
circular import on the bottom-of-file re-export.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

# Statistic name accepted by sweep-aggregate readers. Mirrors the Literal on
# ``SLAFilter.stat`` (``aiperf.config.sweep.adaptive``); a typo here at type-
# check time beats a silent infeasible-cell at runtime.
StatLiteral: TypeAlias = Literal["avg", "p50", "p90", "p95", "p99"]
_STAT_VALUES: tuple[str, ...] = ("avg", "p50", "p90", "p95", "p99")


def _stat_or_raise(value: Any, *, handler: str) -> StatLiteral:
    """Validate a runtime ``params['stat']`` against the allowed values.

    Handler ``params`` arrive as ``dict[str, Any]`` (recipes inject heterogeneous
    payloads); this gate narrows to the ``StatLiteral`` so ``_extract_points``
    and friends keep an honest type. Raises ``ValueError`` on unknown values
    naming the handler so a typo (``p98``) surfaces with full context.
    """
    if value in _STAT_VALUES:
        return value  # type: ignore[return-value]
    raise ValueError(
        f"{handler}: params['stat']={value!r} is not a recognized statistic; "
        f"expected one of {_STAT_VALUES}."
    )
