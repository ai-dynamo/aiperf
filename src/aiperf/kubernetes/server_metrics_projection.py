# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curated projection of live server metrics onto the AIPerfJob status subresource.

``status.serverMetrics`` is the dashboard's non-WebSocket fallback: when the
per-job socket is blocked (a common port-forward failure mode), the job-detail
page renders the server-metrics panel from the CR instead. The full server
metrics payload is unbounded and has no business inside a status subresource
with a 1.5 MB apiserver object ceiling, so this module emits an explicit
allow-list of exactly what the panel reads.

The allow-list mirrors ``backendMetric`` in
``src/aiperf/operator/ui/components/server-metrics/helpers.js``. When a metric
is added there, add it here too; the reverse is harmless but dead weight.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from aiperf.common.environment import Environment
from aiperf.common.finite import scrub_non_finite

logger = logging.getLogger(__name__)

CURATED_METRIC_NAMES: frozenset[str] = frozenset(
    {
        # kvCachePct
        "dynamo_component_kvstats_gpu_cache_usage_percent",
        "vllm:kv_cache_usage_perc",
        "sglang:token_usage",
        # requestsWaiting
        "dynamo_frontend_queued_requests",
        "vllm:num_requests_waiting",
        "sglang:num_queue_reqs",
        # reqRate
        "dynamo_frontend_requests",
        "vllm:request_success",
        "trtllm:request_success",
        # genTokRate
        "dynamo_frontend_output_tokens",
        "vllm:generation_tokens",
        "sglang:gen_throughput",
        # e2eLatency
        "dynamo_frontend_request_duration_seconds",
        "vllm:e2e_request_latency_seconds",
        "trtllm:e2e_request_latency_seconds",
        "sglang:e2e_request_latency_seconds",
        # ttft
        "dynamo_frontend_time_to_first_token_seconds",
        "vllm:time_to_first_token_seconds",
        "trtllm:time_to_first_token_seconds",
        "sglang:time_to_first_token_seconds",
    }
)

CURATED_STAT_FIELDS: frozenset[str] = frozenset(
    {"avg", "max", "rate", "p99_estimate", "count"}
)


def _iso_from_ns(value: Any) -> str | None:
    """Render a nanosecond wall-clock stamp as the ISO string the dashboard parses."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=UTC).isoformat()


def _project_series(
    series: Any, endpoint_url: str, max_labels: int
) -> dict[str, Any] | None:
    """Reduce one series to identity plus the five stat fields the panel reads.

    Returns ``None`` when the series is over-labeled, which drops its whole
    metric upstream: labels are the series identity, so trimming them would
    collapse two distinct series into one and silently corrupt every aggregate.
    """
    if not isinstance(series, dict):
        return None
    labels = series.get("labels")
    if isinstance(labels, dict) and len(labels) > max_labels:
        return None

    projected: dict[str, Any] = {
        "endpoint_url": series.get("endpoint_url") or endpoint_url
    }
    if isinstance(labels, dict):
        projected["labels"] = {str(k): str(v) for k, v in labels.items()}

    stats = series.get("stats")
    if isinstance(stats, dict):
        projected["stats"] = {
            key: value for key, value in stats.items() if key in CURATED_STAT_FIELDS
        }
    return projected


def _update_window(
    info: Any, first_ns: int | None, last_ns: int | None
) -> tuple[int | None, int | None]:
    """Widen the scrape window with one endpoint's first/last update stamps."""
    if not isinstance(info, dict):
        return first_ns, last_ns
    start = info.get("first_update_ns")
    end = info.get("last_update_ns")
    if isinstance(start, int) and start > 0:
        first_ns = start if first_ns is None else min(first_ns, start)
    if isinstance(end, int) and end > 0:
        last_ns = end if last_ns is None else max(last_ns, end)
    return first_ns, last_ns


def _merge_endpoint_metrics(
    endpoint_metrics: dict[str, Any],
    endpoint_url: str,
    *,
    metrics: dict[str, dict[str, Any]],
    dropped: set[str],
    max_series: int,
    max_labels: int,
) -> None:
    """Fold one endpoint's allow-listed metrics into the cross-endpoint accumulator."""
    for name, metric in endpoint_metrics.items():
        if name not in CURATED_METRIC_NAMES or name in dropped:
            continue
        if not isinstance(metric, dict):
            continue
        entry = metrics.setdefault(name, {"type": metric.get("type"), "series": []})
        for series in metric.get("series") or []:
            projected = _project_series(series, endpoint_url, max_labels)
            if projected is None:
                dropped.add(name)
                break
            entry["series"].append(projected)
        if name in dropped or len(entry["series"]) > max_series:
            dropped.add(name)
            metrics.pop(name, None)


def _build_summary(
    *,
    endpoints_configured: list[str],
    endpoints_successful: list[str],
    first_ns: int | None,
    last_ns: int | None,
) -> dict[str, Any]:
    """Shape the summary strip the dashboard's ``buildSummary`` reads."""
    summary: dict[str, Any] = {
        "endpoints_configured": endpoints_configured,
        "endpoints_successful": endpoints_successful,
    }
    if (start_time := _iso_from_ns(first_ns)) is not None:
        summary["start_time"] = start_time
    if (end_time := _iso_from_ns(last_ns)) is not None:
        summary["end_time"] = end_time
    return summary


def project_server_metrics_for_cr(
    server_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the curated ``status.serverMetrics`` value, or ``None`` if there is nothing to write.

    Accepts the ``RealtimeServerMetricsMessage`` dump shape (keyed by
    ``endpoint_summaries``) and emits the already-normalized ``{summary,
    metrics}`` shape, so the dashboard's ``normalizeServerMetrics`` passes it
    through untouched.

    Over-cap metrics are dropped whole with a debug log; nothing is truncated
    mid-object. The result is ``scrub_non_finite``-cleaned so a NaN gauge
    cannot reject the entire status patch.
    """
    if not isinstance(server_metrics, dict):
        return None
    endpoint_summaries = server_metrics.get("endpoint_summaries")
    if not isinstance(endpoint_summaries, dict) or not endpoint_summaries:
        return None

    max_series = Environment.SERVER_METRICS.CR_PROJECTION_MAX_SERIES
    max_labels = Environment.SERVER_METRICS.CR_PROJECTION_MAX_LABELS

    metrics: dict[str, dict[str, Any]] = {}
    dropped: set[str] = set()
    endpoints_configured: list[str] = []
    endpoints_successful: list[str] = []
    first_ns: int | None = None
    last_ns: int | None = None

    for endpoint_key, summary in endpoint_summaries.items():
        if not isinstance(summary, dict):
            continue
        endpoint_url = str(summary.get("endpoint_url") or endpoint_key)
        endpoints_configured.append(endpoint_url)
        first_ns, last_ns = _update_window(summary.get("info"), first_ns, last_ns)

        endpoint_metrics = summary.get("metrics")
        if not isinstance(endpoint_metrics, dict) or not endpoint_metrics:
            continue
        endpoints_successful.append(endpoint_url)
        _merge_endpoint_metrics(
            endpoint_metrics,
            endpoint_url,
            metrics=metrics,
            dropped=dropped,
            max_series=max_series,
            max_labels=max_labels,
        )

    if dropped:
        logger.debug(
            "Dropped %d server metric(s) from the AIPerfJob status projection "
            "(over the %d-series / %d-label cap): %s",
            len(dropped),
            max_series,
            max_labels,
            ", ".join(sorted(dropped)),
        )

    if not metrics:
        return None

    return scrub_non_finite(
        {
            "summary": _build_summary(
                endpoints_configured=endpoints_configured,
                endpoints_successful=endpoints_successful,
                first_ns=first_ns,
                last_ns=last_ns,
            ),
            "metrics": metrics,
        }
    )
