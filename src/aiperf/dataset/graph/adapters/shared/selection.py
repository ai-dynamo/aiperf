# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-only filter-then-cap trace selection shared by the recorded adapters.

The graph plane historically ignored ``--num-dataset-entries`` and
``--max-context-length`` (ai-dynamo/aiperf#1106): every recorded trace was
built and dispatch silently cloned traces to fill lanes. This module is the ONE
selection primitive both the weka and dynamo loaders call to honor those knobs
BEFORE the expensive build:

* FILTER: drop a candidate whose peak context (input + output tokens on its
  largest single request, computed schema-only by the
  :mod:`~aiperf.dataset.graph.adapters.shared.peak_context` helpers) exceeds
  ``max_context_length``.
* CAP: keep the FIRST ``num_dataset_entries`` eligible candidates (in the
  adapter's deterministic scan order).

"Filter THEN cap" -- the cap is applied to the ELIGIBLE set, never to the raw
prefix (which would drop below N whenever a rejected trace fell in the first N).
Scanning stops as soon as the cap is reached, so on a capped load only the first
N-eligible-plus-rejected-so-far candidates are ever examined; the returned
:class:`SelectionStats` reflect only what was scanned.
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
    """Tally of one filter-then-cap selection pass, for the load-summary report.

    ``scanned`` counts candidates examined (fewer than the corpus size when the
    cap short-circuits the scan); ``rejected_by_maxctx`` counts those dropped
    for exceeding ``max_context_length``; ``largest_observed`` is the peak
    context seen across every scanned candidate (eligible or not); ``eligible``
    counts candidates that passed the filter and were kept; ``loaded`` is the
    number ultimately handed to the build (``eligible`` under early-stop, since
    the scan halts exactly when the cap is filled).
    """

    scanned: int = 0
    rejected_by_maxctx: int = 0
    largest_observed: int = 0
    eligible: int = 0
    loaded: int = 0


def filter_then_cap(
    candidates: Iterable[tuple[_T, int]],
    *,
    num_dataset_entries: int | None,
    max_context_length: int | None,
) -> tuple[list[_T], SelectionStats]:
    """Filter ``candidates`` by peak context, then cap to the first N eligible.

    ``candidates`` yields ``(item, peak_context)`` pairs in the adapter's
    DETERMINISTIC scan order (dir files sorted by name, HF rows in stream order,
    dynamo trees sorted by root session id). It should be LAZY: this function
    stops pulling from it the moment ``num_dataset_entries`` eligible items are
    kept, so a capped load never computes peaks for the whole corpus.

    ``max_context_length is None`` disables the filter (every candidate is
    eligible); ``num_dataset_entries is None`` disables the cap (every eligible
    candidate is kept). Returns the kept items in scan order plus the
    :class:`SelectionStats` for the scanned prefix.
    """
    stats = SelectionStats()
    kept: list[_T] = []
    for item, peak in candidates:
        stats.scanned += 1
        if peak > stats.largest_observed:
            stats.largest_observed = peak
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
    """Emit the once-per-build filter-then-cap summary (mirrors the trace loader).

    The graph counterpart of
    :meth:`~aiperf.dataset.loader.base_trace_loader.BaseTraceLoader._log_filtering_summary`:
    called at the PARENT-side finalize point where the single
    :class:`SelectionStats` is produced (once per build, before any worker
    fan-out), so it logs exactly once no matter how the build parallelizes.
    Every adapter finalize point routes through this ONE helper so the summary
    text is uniform across the weka and dynamo loaders. When no selection knob
    is set the selection scan never runs, so this is never called (byte-identical
    quiet path).
    """
    _logger.info(
        lambda: (
            f"graph trace selection [{source}]: scanned {stats.scanned:,}, "
            f"rejected_by_maxctx {stats.rejected_by_maxctx:,} "
            f"(--max-context-length={max_context_length}), "
            f"eligible {stats.eligible:,}, loaded {stats.loaded:,} "
            f"(--num-dataset-entries={num_dataset_entries}), "
            f"largest_observed_peak_context {stats.largest_observed:,}"
        )
    )


__all__ = [
    "SelectionStats",
    "filter_then_cap",
    "log_selection_summary",
]
