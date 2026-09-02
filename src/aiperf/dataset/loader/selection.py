# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-only filter-then-cap trace selection.

* FILTER: drop a candidate whose peak context exceeds ``max_context_length``.
* CAP: keep the first ``num_dataset_entries`` *eligible* candidates.

Filtering happens before the cap counts a slot: an oversized row is skipped
without consuming an entry, so the loaded pool always reaches the requested
entry count when enough eligible candidates exist.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

from aiperf.common.aiperf_logger import AIPerfLogger

_T = TypeVar("_T")

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class SelectionStats:
    """Tally of one filter-then-cap selection pass."""

    scanned: int = 0
    rejected_by_maxctx: int = 0
    largest_observed: int = 0
    smallest_observed: int = 0
    """The minimum peak context among scanned candidates."""
    eligible: int = 0
    loaded: int = 0


def filter_then_cap(
    candidates: Iterable[tuple[_T, int]],
    *,
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> tuple[list[_T], SelectionStats]:
    """Filter by peak context, then cap to the first N eligible items.

    ``candidates`` yields ``(item, peak_context)`` in deterministic scan order
    and should be lazy: scanning stops once the entry cap is filled.
    """
    stats = SelectionStats()
    kept: list[_T] = []
    for item, peak in candidates:
        stats.scanned += 1
        if peak > stats.largest_observed:
            stats.largest_observed = peak
        if stats.scanned == 1 or peak < stats.smallest_observed:
            stats.smallest_observed = peak
        if max_context_length is not None and peak > max_context_length:
            stats.rejected_by_maxctx += 1
            continue
        stats.eligible += 1
        kept.append(item)
        if num_dataset_entries is not None and len(kept) >= num_dataset_entries:
            break
    stats.loaded = len(kept)
    return kept, stats


def log_selection_summary(
    stats: SelectionStats,
    *,
    source: str,
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> None:
    """Emit the once-per-load filter-then-cap summary."""
    _logger.info(
        lambda: (
            f"weka trace selection [{source}]: scanned {stats.scanned:,}, "
            f"rejected_by_maxctx {stats.rejected_by_maxctx:,} "
            f"(--max-context-length={max_context_length}), "
            f"eligible {stats.eligible:,}, loaded {stats.loaded:,} "
            f"(--num-dataset-entries={num_dataset_entries}), "
            f"largest_observed_peak_context {stats.largest_observed:,}, "
            f"smallest_observed_peak_context {stats.smallest_observed:,}"
        )
    )


def raise_on_empty_selection(
    stats: SelectionStats,
    *,
    source: str,
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> None:
    """Raise DatasetLoaderError when no candidates are selected, with a size hint."""
    from aiperf.common.exceptions import DatasetLoaderError

    msg = (
        f"No eligible traces in {source} after "
        f"filter-then-cap (scanned {stats.scanned}, "
        f"--max-context-length={max_context_length}, "
        f"--num-dataset-entries={num_dataset_entries})."
    )
    if stats.smallest_observed > 0:
        msg += (
            f" Smallest trace requires {stats.smallest_observed} tokens; "
            f"raise --max-context-length to at least that (e.g. --max-context-length {stats.smallest_observed}) to admit any trace."
        )
    raise DatasetLoaderError(msg)


__all__ = [
    "SelectionStats",
    "filter_then_cap",
    "log_selection_summary",
    "raise_on_empty_selection",
]
