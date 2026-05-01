# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``build_gpu_telemetry`` and ``build_server_metrics`` v1 converters."""

from pathlib import Path

from aiperf.common.enums import GPUTelemetryMode
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_telemetry import (
    build_gpu_telemetry,
    build_server_metrics,
)
from aiperf.plugin.enums import GPUTelemetryCollectorType


def test_gpu_telemetry_no_flag_returns_disabled():
    user = UserConfig.model_validate({"no_gpu_telemetry": True})
    out = build_gpu_telemetry(user)
    assert out == {"enabled": False}


def test_gpu_telemetry_default_returns_enabled_with_no_urls():
    user = UserConfig()
    out = build_gpu_telemetry(user)
    assert out == {"enabled": True}


def test_gpu_telemetry_with_url_token():
    user = UserConfig.model_validate({"gpu_telemetry": ["http://gpu1:9400"]})
    out = build_gpu_telemetry(user)
    assert out["enabled"] is True
    assert "http://gpu1:9400" in out["urls"]


def test_gpu_telemetry_with_csv_token_sets_metrics_file():
    user = UserConfig.model_validate({"gpu_telemetry": ["/tmp/gpu.csv"]})
    out = build_gpu_telemetry(user)
    assert out["enabled"] is True
    assert out["metrics_file"] == Path("/tmp/gpu.csv")


def test_no_server_metrics_disables_collection():
    user = UserConfig.model_validate({"no_server_metrics": True})
    out = build_server_metrics(user)
    assert out == {"enabled": False}


def test_server_metrics_formats_passed_through():
    user = UserConfig.model_validate({"server_metrics_formats": ["json", "csv"]})
    out = build_server_metrics(user)
    assert out["enabled"] is True
    assert [str(f) for f in out["formats"]] == ["json", "csv"]


def test_server_metrics_urls_from_list():
    user = UserConfig.model_validate({"server_metrics": ["http://triton:8002/metrics"]})
    out = build_server_metrics(user)
    assert out["enabled"] is True
    assert len(out["urls"]) == 1


def test_gpu_telemetry_pynvml_token_sets_collector_private_attr():
    user = UserConfig.model_validate({"gpu_telemetry": ["pynvml"]})

    out = build_gpu_telemetry(user)

    assert out["enabled"] is True
    assert user._gpu_telemetry_collector_type == GPUTelemetryCollectorType.PYNVML
    assert out["collector"] == GPUTelemetryCollectorType.PYNVML
    assert out["urls"] == []


def test_gpu_telemetry_dashboard_token_sets_mode_private_attr():
    user = UserConfig.model_validate({"gpu_telemetry": ["dashboard", "node1:9400"]})

    out = build_gpu_telemetry(user)

    assert out["enabled"] is True
    assert user._gpu_telemetry_mode == GPUTelemetryMode.REALTIME_DASHBOARD
    assert out["mode"] == GPUTelemetryMode.REALTIME_DASHBOARD
    assert out["urls"] == ["http://node1:9400"]
