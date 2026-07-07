# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-compat coverage for the server-metrics JSON loader.

The nested server-metrics export dataclasses use ``extra="forbid"``. When a
newer AIPerf writes an additive field into ``server_metrics_export.json``, the
older reader must tolerate it — otherwise ``model_validate`` raises, the
``DataLoadError`` is swallowed by ``DataLoader._load_server_metrics``, and the
ENTIRE server-metrics section silently vanishes from the plot. These tests lock
in the projection that keeps additive schema evolution backward-compatible,
while still degrading cleanly on genuinely-malformed input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.common.models.server_metrics_models import ServerMetricsExportData
from aiperf.plot.core.data_loader import DataLoader
from aiperf.plot.exceptions import DataLoadError


def _valid_export() -> dict[str, Any]:
    """A minimal, schema-valid server_metrics_export.json payload."""
    return {
        "schema_version": "1.0",
        "aiperf_version": "1.2.3",
        "benchmark_id": "aiperf-bench-7f2a",
        "summary": {
            "endpoints_configured": ["http://localhost:8000/metrics"],
            "endpoints_successful": ["http://localhost:8000/metrics"],
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:01:00",
            "endpoint_info": {
                "http://localhost:8000/metrics": {
                    "total_fetches": 10,
                    "first_fetch_ns": 0,
                    "last_fetch_ns": 1_000_000_000,
                    "avg_fetch_latency_ms": 1.0,
                    "unique_updates": 5,
                    "first_update_ns": 0,
                    "last_update_ns": 1_000_000_000,
                    "duration_seconds": 60.0,
                    "avg_update_interval_ms": 100.0,
                }
            },
        },
        "metrics": {
            "vllm:kv_cache_usage_perc": {
                "type": "gauge",
                "description": "KV cache usage",
                "unit": "percent",
                "series": [
                    {
                        "endpoint_url": "http://localhost:8000/metrics",
                        "labels": {"model": "llama"},
                        "stats": {"avg": 0.5, "min": 0.1, "max": 0.9},
                        "timeslices": [
                            {
                                "start_ns": 0,
                                "end_ns": 1_000_000_000,
                                "avg": 0.5,
                                "min": 0.1,
                                "max": 0.9,
                            }
                        ],
                    }
                ],
            }
        },
        "input_config": {},
    }


def _write_export(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "server_metrics_export.json"
    path.write_bytes(orjson.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# project_export_dict (the projection helper)
# ---------------------------------------------------------------------------


class TestProjectExportDict:
    def test_drops_unknown_nested_fields_keeps_known(self) -> None:
        payload = _valid_export()
        payload["summary"]["future_summary_field"] = "x"
        payload["summary"]["endpoint_info"]["http://localhost:8000/metrics"][
            "future_info_field"
        ] = 1
        metric = payload["metrics"]["vllm:kv_cache_usage_perc"]
        metric["future_metric_field"] = 1
        series = metric["series"][0]
        series["future_series_field"] = True
        series["stats"]["future_stat"] = 42
        series["timeslices"][0]["future_ts"] = "z"

        projected = ServerMetricsExportData.project_export_dict(payload)

        assert "future_summary_field" not in projected["summary"]
        assert (
            "future_info_field"
            not in projected["summary"]["endpoint_info"][
                "http://localhost:8000/metrics"
            ]
        )
        p_metric = projected["metrics"]["vllm:kv_cache_usage_perc"]
        assert "future_metric_field" not in p_metric
        p_series = p_metric["series"][0]
        assert "future_series_field" not in p_series
        assert "future_stat" not in p_series["stats"]
        assert "future_ts" not in p_series["timeslices"][0]
        # Known fields survive untouched.
        assert p_series["stats"] == {"avg": 0.5, "min": 0.1, "max": 0.9}

    def test_non_dict_input_returned_unchanged(self) -> None:
        assert ServerMetricsExportData.project_export_dict("nope") == "nope"  # type: ignore[arg-type]

    def test_valid_payload_still_validates_after_projection(self) -> None:
        payload = _valid_export()
        model = ServerMetricsExportData.model_validate(
            ServerMetricsExportData.project_export_dict(payload)
        )
        assert set(model.metrics) == {"vllm:kv_cache_usage_perc"}


# ---------------------------------------------------------------------------
# _load_server_metrics_json (end-to-end loader path)
# ---------------------------------------------------------------------------


class TestLoadServerMetricsJsonForwardCompat:
    def test_unknown_nested_field_does_not_drop_section(self, tmp_path: Path) -> None:
        payload = _valid_export()
        # Additive field on a nested (extra="forbid") object.
        payload["metrics"]["vllm:kv_cache_usage_perc"]["series"][0]["stats"][
            "future_stat"
        ] = 42
        json_path = _write_export(tmp_path, payload)

        _df, aggregated = DataLoader()._load_server_metrics_json(json_path)

        # The whole section survives — the metric is present, not dropped.
        assert "vllm:kv_cache_usage_perc" in aggregated

    def test_known_values_correct_with_unknown_field(self, tmp_path: Path) -> None:
        payload = _valid_export()
        payload["metrics"]["vllm:kv_cache_usage_perc"]["series"][0]["stats"][
            "future_stat"
        ] = 42
        json_path = _write_export(tmp_path, payload)

        _df, aggregated = DataLoader()._load_server_metrics_json(json_path)

        endpoint = aggregated["vllm:kv_cache_usage_perc"][
            "http://localhost:8000/metrics"
        ]
        stats = next(iter(endpoint.values()))["stats"]
        assert stats.avg == 0.5
        assert stats.min == 0.1
        assert stats.max == 0.9

    def test_top_level_additive_field_tolerated(self, tmp_path: Path) -> None:
        payload = _valid_export()
        payload["future_top_level"] = "tolerated"
        json_path = _write_export(tmp_path, payload)

        _df, aggregated = DataLoader()._load_server_metrics_json(json_path)
        assert "vllm:kv_cache_usage_perc" in aggregated

    def test_malformed_file_still_degrades_cleanly(self, tmp_path: Path) -> None:
        json_path = tmp_path / "server_metrics_export.json"
        # summary is the wrong type and metrics is a scalar — genuinely broken,
        # not merely forward-incompatible; must still raise DataLoadError so the
        # caller can warn-and-skip rather than crash.
        json_path.write_bytes(b'{"summary": "not-an-object", "metrics": 12345}')

        with pytest.raises(DataLoadError):
            DataLoader()._load_server_metrics_json(json_path)
