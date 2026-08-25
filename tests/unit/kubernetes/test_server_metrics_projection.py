# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the curated ``status.serverMetrics`` projection."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.kubernetes.server_metrics_projection import (
    CURATED_METRIC_NAMES,
    CURATED_STAT_FIELDS,
    project_server_metrics_for_cr,
)

_HELPERS_JS = Path("src/aiperf/operator/ui/components/server-metrics/helpers.js")


def _summary(metrics: dict[str, Any], **info: Any) -> dict[str, Any]:
    """One endpoint summary in ``RealtimeServerMetricsMessage`` dump shape."""
    return {
        "endpoint_summaries": {
            "http://e1:8000/metrics": {
                "endpoint_url": "http://e1:8000/metrics",
                "info": info,
                "metrics": metrics,
            }
        }
    }


def _gauge(*series: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "gauge",
        "description": "some very long prometheus HELP text",
        "unit": "percent",
        "series": list(series),
    }


def test_curated_metric_names_match_the_dashboard_backend_metric_list() -> None:
    """The allow-list must equal what ``backendMetric`` in helpers.js can render.

    A drift in either direction is a silent bug: a name the JS reads but the
    projection omits renders as a missing tile in the REST fallback, and a name
    the projection carries but the JS never reads is pure CR weight.
    """
    source = _HELPERS_JS.read_text()
    start = source.index("function backendMetric")
    end = source.index("function extractStatPerSeries")
    assert end > start, (
        "helpers.js reordered: backendMetric no longer precedes "
        "extractStatPerSeries, so this guard would slice an empty body"
    )
    js_names = set(re.findall(r"has\('([^']+)'\)", source[start:end]))
    assert len(js_names) == 20, (
        f"expected 20 metric names parsed out of backendMetric, got {len(js_names)}; "
        "the extraction broke and this assertion would pass vacuously"
    )

    assert js_names == set(CURATED_METRIC_NAMES)


def test_project_server_metrics_for_cr_keeps_allow_listed_and_drops_everything_else() -> (
    None
):
    """Only allow-listed metric names survive; unknown ones never reach the CR."""
    result = project_server_metrics_for_cr(
        _summary(
            {
                "dynamo_frontend_requests": _gauge({"stats": {"rate": 12.5}}),
                "vllm:num_requests_waiting": _gauge({"stats": {"avg": 3.0}}),
                "some_unrelated_exporter_metric": _gauge({"stats": {"avg": 1.0}}),
                "process_resident_memory_bytes": _gauge({"stats": {"avg": 1.0}}),
            }
        )
    )

    assert result is not None
    assert set(result["metrics"]) == {
        "dynamo_frontend_requests",
        "vllm:num_requests_waiting",
    }


def test_project_server_metrics_for_cr_keeps_only_curated_stat_fields() -> None:
    """Series carry identity plus the five read stats -- never buckets or timeslices."""
    result = project_server_metrics_for_cr(
        _summary(
            {
                "dynamo_frontend_time_to_first_token_seconds": {
                    "type": "histogram",
                    "series": [
                        {
                            "endpoint_url": "http://e1:8000/metrics",
                            "labels": {"model": "llama"},
                            "stats": {
                                "avg": 1.0,
                                "max": 2.0,
                                "rate": 3.0,
                                "p99_estimate": 4.0,
                                "count": 5,
                                "p50_estimate": 6.0,
                                "sum": 7.0,
                                "sum_rate": 8.0,
                            },
                            "buckets": {"0.1": 2000, "+Inf": 5000},
                            "timeslices": [{"avg": 1.0}] * 50,
                        }
                    ],
                }
            }
        )
    )

    assert result is not None
    series = result["metrics"]["dynamo_frontend_time_to_first_token_seconds"]["series"][
        0
    ]
    assert set(series["stats"]) == set(CURATED_STAT_FIELDS)
    assert series["labels"] == {"model": "llama"}
    assert series["endpoint_url"] == "http://e1:8000/metrics"
    assert "buckets" not in series
    assert "timeslices" not in series


def test_project_server_metrics_for_cr_drops_metric_description_and_unit() -> None:
    """Only ``series`` survives: HELP text is unbounded, and ``type`` is never read.

    ``aggregateForHit`` in helpers.js uses the hardcoded ``hit.type`` from
    ``backendMetric``, never ``metric.type``.
    """
    result = project_server_metrics_for_cr(
        _summary({"sglang:token_usage": _gauge({"stats": {"max": 0.9}})})
    )

    assert result is not None
    metric = result["metrics"]["sglang:token_usage"]
    assert set(metric) == {"series"}


def test_project_server_metrics_for_cr_merges_series_across_endpoints() -> None:
    """The projection emits the already-normalized shape, fanning endpoints in itself."""
    payload = {
        "endpoint_summaries": {
            "a": {
                "endpoint_url": "http://a/metrics",
                "metrics": {"vllm:generation_tokens": _gauge({"stats": {"rate": 1.0}})},
            },
            "b": {
                "endpoint_url": "http://b/metrics",
                "metrics": {"vllm:generation_tokens": _gauge({"stats": {"rate": 2.0}})},
            },
        }
    }

    result = project_server_metrics_for_cr(payload)

    assert result is not None
    series = result["metrics"]["vllm:generation_tokens"]["series"]
    assert [s["endpoint_url"] for s in series] == [
        "http://a/metrics",
        "http://b/metrics",
    ]
    assert result["summary"]["endpoints_configured"] == [
        "http://a/metrics",
        "http://b/metrics",
    ]
    assert result["summary"]["endpoints_successful"] == [
        "http://a/metrics",
        "http://b/metrics",
    ]


def test_project_server_metrics_for_cr_drops_over_series_cap_metric_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap metric is dropped entirely, never truncated to a wrong aggregate."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.SERVER_METRICS, "CR_PROJECTION_MAX_SERIES", 3)

    result = project_server_metrics_for_cr(
        _summary(
            {
                "dynamo_frontend_requests": _gauge(
                    *({"stats": {"rate": float(i)}} for i in range(4))
                ),
                "sglang:num_queue_reqs": _gauge({"stats": {"avg": 1.0}}),
            }
        )
    )

    assert result is not None
    assert "dynamo_frontend_requests" not in result["metrics"]
    assert "sglang:num_queue_reqs" in result["metrics"]


def test_project_server_metrics_for_cr_drops_over_label_cap_metric_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Labels are series identity, so an over-labeled series drops its metric, not its labels."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.SERVER_METRICS, "CR_PROJECTION_MAX_LABELS", 2)

    result = project_server_metrics_for_cr(
        _summary(
            {
                "trtllm:request_success": _gauge(
                    {"labels": {"a": "1", "b": "2", "c": "3"}, "stats": {"rate": 1.0}}
                ),
                "vllm:kv_cache_usage_perc": _gauge(
                    {"labels": {"a": "1"}, "stats": {"max": 0.5}}
                ),
            }
        )
    )

    assert result is not None
    assert "trtllm:request_success" not in result["metrics"]
    assert result["metrics"]["vllm:kv_cache_usage_perc"]["series"][0]["labels"] == {
        "a": "1"
    }


def test_project_server_metrics_for_cr_scrubs_non_finite_stats() -> None:
    """A NaN gauge would reject the whole status patch; it must arrive as null."""
    result = project_server_metrics_for_cr(
        _summary(
            {
                "sglang:gen_throughput": _gauge(
                    {"stats": {"avg": math.nan, "max": math.inf, "rate": 2.0}}
                )
            }
        )
    )

    assert result is not None
    stats = result["metrics"]["sglang:gen_throughput"]["series"][0]["stats"]
    assert stats["avg"] is None
    assert stats["max"] is None
    assert stats["rate"] == 2.0


def test_project_server_metrics_for_cr_emits_scrape_window_from_update_stamps() -> None:
    """start_time/end_time feed the panel's scrape-window strip via Date.parse."""
    result = project_server_metrics_for_cr(
        _summary(
            {"dynamo_frontend_queued_requests": _gauge({"stats": {"avg": 1.0}})},
            first_update_ns=1_700_000_000_000_000_000,
            last_update_ns=1_700_000_060_000_000_000,
        )
    )

    assert result is not None
    assert result["summary"]["start_time"].startswith("2023-11-14T")
    assert result["summary"]["end_time"].startswith("2023-11-14T")


@pytest.mark.parametrize(
    "payload",
    [
        param(None, id="none"),
        param({}, id="empty"),
        param({"endpoint_summaries": {}}, id="no-endpoints"),
        param(
            {"endpoint_summaries": {"a": {"metrics": {"junk_metric": _gauge()}}}},
            id="nothing-allow-listed",
        ),
        param({"snapshot": {"x": 1.0}}, id="snapshot-only"),
    ],
)  # fmt: skip
def test_project_server_metrics_for_cr_returns_none_when_there_is_nothing_to_write(
    payload: Any,
) -> None:
    """No allow-listed data means no status key at all, not an empty object."""
    assert project_server_metrics_for_cr(payload) is None


def test_project_server_metrics_for_cr_carries_no_credential_fields() -> None:
    """The push logs rejected bodies verbatim; the projection must be secret-free."""
    result = project_server_metrics_for_cr(
        _summary(
            {
                "vllm:request_success": {
                    "type": "counter",
                    "api_key": "sk-should-never-appear",
                    "series": [
                        {
                            "stats": {"rate": 1.0},
                            "headers": {"Authorization": "Bearer nope"},
                        }
                    ],
                }
            }
        )
    )

    assert result is not None
    rendered = repr(result)
    assert "sk-should-never-appear" not in rendered
    assert "Authorization" not in rendered


def test_project_server_metrics_for_cr_drops_everything_over_the_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Label strings are unbounded, so cardinality caps alone cannot bound bytes.

    Blowing the apiserver's 1.5 MB object ceiling rejects the WHOLE status
    patch, which stops phases, liveMetrics, resultsExported and
    controllerFailure along with it -- the exact silent loss this key exists to
    repair. Under budget pressure the projection yields rather than take the
    rest of status down with it.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.SERVER_METRICS, "CR_PROJECTION_MAX_BYTES", 2048)

    fat_label = {"model": "x" * 4096}
    result = project_server_metrics_for_cr(
        _summary(
            {
                "dynamo_frontend_requests": _gauge(
                    {"labels": fat_label, "stats": {"rate": 1.0}}
                )
            }
        )
    )

    assert result is not None
    assert result["metrics"] == {}
    assert result["projection_dropped"] is True
    assert "projection_message" in result


def test_project_server_metrics_for_cr_stays_within_the_byte_budget_by_default() -> (
    None
):
    """A realistically wide deployment must fit, or the tile silently disappears."""
    from aiperf.common.environment import Environment

    wide = {
        name: _gauge(
            *(
                {
                    "labels": {"model": "meta-llama/Llama-3.1-70B", "tp_rank": str(i)},
                    "stats": {"avg": 1.0, "max": 2.0, "rate": 3.0, "count": 4},
                }
                for i in range(64)
            )
        )
        for name in CURATED_METRIC_NAMES
    }

    result = project_server_metrics_for_cr(_summary(wide))

    assert result is not None
    assert set(result["metrics"]) == set(CURATED_METRIC_NAMES)
    assert (
        len(orjson.dumps(result)) <= Environment.SERVER_METRICS.CR_PROJECTION_MAX_BYTES
    )


def test_default_series_cap_clears_a_large_per_worker_deployment() -> None:
    """MAX_SERIES is a per-metric total across endpoints, so it must clear worker counts.

    Per-worker metrics such as dynamo_component_kvstats_gpu_cache_usage_percent
    carry one series per worker or GPU; a cap below the worker count drops the
    metric whole and the KV-pressure tile vanishes with no user-visible signal.
    """
    from aiperf.common.environment import Environment

    assert Environment.SERVER_METRICS.CR_PROJECTION_MAX_SERIES >= 256
