# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-variation failure summary logging for sweep runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.orchestrator.models import RunResult


def _log_failed_sweep_variations(
    failed_runs: list[RunResult], logger: AIPerfLogger
) -> None:
    """Log per-variation failures for a sweep, grouped by (label, sorted values).

    Keying by label too is required so QMC cells with collision-prone integer
    values (Sobol/LHS) don't get pooled into one row of the summary; mirrors
    ``cli_runner._sweep_helpers``.
    """
    by_variation: dict[tuple, list[RunResult]] = {}
    for r in failed_runs:
        key = (
            r.variation_label or "",
            tuple(sorted((r.variation_values or {}).items())),
        )
        by_variation.setdefault(key, []).append(r)

    def _format_key(label: str, params: tuple) -> str:
        kvs = ", ".join(f"{k}={v}" for k, v in params)
        return f"{label}: {kvs}" if label else kvs

    failed_values_str = [_format_key(label, params) for label, params in by_variation]
    logger.warning(f"Some sweep variations failed: {failed_values_str}")
    for (label, params), group in by_variation.items():
        params_str = _format_key(label, params)
        for r in group:
            error_msg = r.error or "(no error message)"
            logger.warning(f"  {params_str}: {error_msg}")
