# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pooled-sample percentile helper for the BO and Optuna planners.

When ``--num-profile-runs`` is N>=2, the default per-trial-mean percentile
extraction (mean of N percentiles, each computed from one trial's request
distribution) is biased relative to the percentile of the pooled
N*requests-per-trial sample. For SLO claims the pooled value is the
correct statistic; mean-of-percentiles is a heuristic.

This helper walks each ``RunResult.artifacts_path / "profile_export.jsonl"``
file -- written by :mod:`aiperf.post_processors.record_export_results_processor`
when ``--export-level records`` (or ``raw``) is set -- accumulates raw
metric samples into a single bag, and returns ``np.percentile(bag, pct)``.

Falls back to ``None`` when:
- No successful results.
- The JSONL file is missing on disk (caller should warn and degrade to
  mean-of-percentiles -- the user did not opt into ``--export-level
  records`` for the search iterations).
- The file exists but contains no samples for the requested metric.

Caveats called out in ``docs/sweeping/bayesian-optimization.md``:
- Pooled vs mean differs only for skewed distributions; for monotone-rank
  problems the BO argmax is unchanged.
- Disk I/O is bounded: with N profile runs at 10k requests each, ~2 MB/iter.
- Per-request samples within a single trial are not strictly i.i.d. (shared
  queue/GC state); a sectioning-based CI (Nakayama 2014) is the principled
  follow-up but the point estimate here is what BO consumes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import orjson

if TYPE_CHECKING:
    from aiperf.orchestrator.models import RunResult

logger = logging.getLogger(__name__)

__all__ = ["percentile_from_stat", "pooled_percentile_from_results"]


# AdaptiveObjective.stat / SLAFilter.stat use the AIPerf naming convention
# ("p50", "p90", "p95", "p99"); numpy.percentile takes a numeric quantile in
# [0, 100]. "avg" is not a percentile and never reaches the pooled path.
_STAT_TO_PERCENTILE: dict[str, float] = {
    "p50": 50.0,
    "p90": 90.0,
    "p95": 95.0,
    "p99": 99.0,
}


def percentile_from_stat(stat: str) -> float | None:
    """Map an AIPerf stat name to its numpy percentile, or None if not a percentile."""
    return _STAT_TO_PERCENTILE.get(stat)


def _samples_from_jsonl(raw: bytes, metric_tag: str) -> list[float]:
    """Extract numeric per-request samples for ``metric_tag`` from JSONL bytes."""
    out: list[float] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        metric = row.get("metrics", {}).get(metric_tag)
        if not isinstance(metric, dict):
            continue
        value = metric.get("value")
        if isinstance(value, int | float):
            out.append(float(value))
    return out


def pooled_percentile_from_results(
    results: list[RunResult],
    metric_tag: str,
    percentile: float,
) -> float | None:
    """Return ``np.percentile`` over per-request samples pooled across trials.

    The JSONL row shape (per ``MetricRecordInfo``) is::

        {"metadata": {...},
         "metrics": {"<metric_tag>": {"value": <num>, "unit": "<str>"}, ...},
         "trace_data": ..., "error": ...}

    ``percentile`` is a float in [0, 100] matching numpy's convention.
    Returns ``None`` if any successful trial is missing its JSONL (caller
    should warn and degrade to mean-of-percentiles), or if no samples were
    found.
    """
    samples: list[float] = []
    for r in results:
        if not r.success or r.artifacts_path is None:
            continue
        jsonl = r.artifacts_path / "profile_export.jsonl"
        if not jsonl.exists():
            logger.warning(
                "Pooled percentile requested for metric %r but %s is missing. "
                "Falling back to mean-of-per-trial-percentiles. Re-run with "
                "--export-level records to enable pooled aggregation.",
                metric_tag,
                jsonl,
            )
            return None
        try:
            raw = jsonl.read_bytes()
        except OSError as e:
            logger.warning(
                "Failed to read %s for pooled percentile (%r); skipping this trial.",
                jsonl,
                e,
            )
            continue
        samples.extend(_samples_from_jsonl(raw, metric_tag))
    if not samples:
        return None
    return float(np.percentile(samples, percentile))
