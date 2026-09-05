# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.enums import GPUTelemetryMode
from aiperf.config.flags._resolver_gpu_telemetry import build_gpu_telemetry_override
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.plugin.enums import GPUTelemetryCollectorType


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


def test_no_explicit_gpu_telemetry_fields_returns_none():
    assert build_gpu_telemetry_override(_make_cli()) is None


def test_no_gpu_telemetry_disables_yaml_gpu_telemetry():
    assert build_gpu_telemetry_override(_make_cli(no_gpu_telemetry=True)) == {
        "enabled": False
    }


def test_gpu_telemetry_urls_override_yaml_gpu_telemetry():
    assert build_gpu_telemetry_override(
        _make_cli(gpu_telemetry=["localhost:9400"])
    ) == {
        "enabled": True,
        "urls": ["http://localhost:9400"],
        "collector": GPUTelemetryCollectorType.DCGM,
        "mode": GPUTelemetryMode.SUMMARY,
    }


def test_resolver_no_gpu_telemetry_overrides_yaml_enabled(tmp_path):
    cfg_file = tmp_path / "gpu.yaml"
    cfg_file.write_text(
        """benchmark:
  models:
    items:
      - name: test-model
  endpoint:
    urls:
      - http://localhost:8000/v1/completions
  datasets:
    - name: inline
      type: synthetic
      entries: 100
      prompts:
        isl: 128
        osl: 64
  phases:
    - name: load
      kind: profiling
      type: concurrency
      concurrency: 1
      requests: 1
  gpu_telemetry:
    enabled: true
    urls:
      - http://localhost:9400/metrics
"""
    )
    user = CLIConfig(no_gpu_telemetry=True)

    config = resolve_config(user, cfg_file)

    assert config.benchmark.gpu_telemetry.enabled is False
