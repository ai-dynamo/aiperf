# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.enums import ServerMetricsFormat
from aiperf.config.flags._resolver_server_metrics import build_server_metrics_override
from aiperf.config.flags.cli_config import CLIConfig


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


def test_no_explicit_server_metrics_fields_returns_none():
    assert build_server_metrics_override(_make_cli()) is None


def test_formats_only_enables_server_metrics_without_overriding_urls():
    assert build_server_metrics_override(
        _make_cli(server_metrics_formats=["json", "csv", "jsonl"])
    ) == {
        "enabled": True,
        "formats": [
            ServerMetricsFormat.JSON,
            ServerMetricsFormat.CSV,
            ServerMetricsFormat.JSONL,
        ],
    }


def test_server_metrics_urls_only_does_not_override_yaml_formats():
    assert build_server_metrics_override(
        _make_cli(server_metrics=["localhost:9400"])
    ) == {
        "enabled": True,
        "urls": ["http://localhost:9400/metrics"],
    }


def test_server_metrics_urls_and_formats_override_both_fields():
    assert build_server_metrics_override(
        _make_cli(
            server_metrics=["localhost:9400"],
            server_metrics_formats=["jsonl"],
        )
    ) == {
        "enabled": True,
        "urls": ["http://localhost:9400/metrics"],
        "formats": [ServerMetricsFormat.JSONL],
    }


def test_no_server_metrics_wins_over_formats():
    assert build_server_metrics_override(
        _make_cli(
            no_server_metrics=True,
            server_metrics_formats=["json", "csv", "jsonl"],
        )
    ) == {"enabled": False}
