# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import ServerMetricsFormat
from aiperf.config import load_config_from_string
from aiperf.config.flags._resolver_server_metrics import (
    build_server_metrics_override,
    normalize_server_metrics_base_for_override,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config


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


def test_server_metrics_headers_route_without_overriding_yaml_urls():
    assert build_server_metrics_override(
        _make_cli(
            server_metrics_headers=[
                "Authorization:Bearer metrics-secret",
                "X-Tenant:tenant-a",
            ]
        )
    ) == {
        "enabled": True,
        "headers": {
            "Authorization": "Bearer metrics-secret",
            "X-Tenant": "tenant-a",
        },
    }


def test_server_metrics_headers_yaml_load_and_json_dump_redacts_secret():
    config = load_config_from_string(
        """
benchmark:
  models: [test-model]
  endpoint:
    urls: [http://localhost:8000]
  server_metrics:
    headers:
      Authorization: Bearer metrics-secret
      X-Tenant: tenant-a
  datasets:
    - name: main
      type: synthetic
      entries: 1
      prompts:
        isl: 8
        osl: 4
  phases:
    - name: profiling
      type: concurrency
      requests: 1
      concurrency: 1
"""
    )

    assert config.benchmark.server_metrics.headers == {
        "Authorization": "Bearer metrics-secret",
        "X-Tenant": "tenant-a",
    }
    dumped = config.model_dump(mode="json", by_alias=False)
    assert dumped["benchmark"]["server_metrics"]["headers"] == {
        "Authorization": "<redacted>",
        "X-Tenant": "tenant-a",
    }


def test_server_metrics_headers_cli_override_preserves_yaml_urls(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
benchmark:
  models: [test-model]
  endpoint:
    urls: [http://localhost:8000]
  server_metrics:
    urls: [http://metrics.example:9400/metrics]
  datasets:
    - name: main
      type: synthetic
      entries: 1
      prompts:
        isl: 8
        osl: 4
  phases:
    - name: profiling
      type: concurrency
      requests: 1
      concurrency: 1
"""
    )

    config = resolve_config(
        _make_cli(
            config_file=config_file,
            server_metrics_headers=[("X-Tenant", "tenant-a")],
        ),
        config_file,
    )

    assert config.benchmark.server_metrics.urls == [
        "http://metrics.example:9400/metrics"
    ]
    assert config.benchmark.server_metrics.headers == {"X-Tenant": "tenant-a"}


def test_no_server_metrics_wins_over_formats():
    assert build_server_metrics_override(
        _make_cli(
            no_server_metrics=True,
            server_metrics_formats=["json", "csv", "jsonl"],
        )
    ) == {"enabled": False}


@pytest.mark.parametrize(
    "server_metrics_key",
    [
        param("serverMetrics", id="camel-case"),
        param("server_metrics", id="snake-case"),
    ],
)  # fmt: skip
def test_normalize_server_metrics_base_expands_url_shorthand_either_spelling(
    server_metrics_key: str,
):
    """The ``url`` -> ``urls`` shorthand must be expanded before ``deep_merge``
    regardless of whether the YAML used ``server_metrics`` or its documented
    ``serverMetrics`` camelCase alias -- otherwise a CLI ``--server-metrics``
    override adds its own ``urls`` key alongside the un-expanded YAML ``url``
    key and both survive to the ``extra="forbid"`` model.
    """
    base = {
        "benchmark": {
            server_metrics_key: {"url": "http://localhost:9090/metrics"},
        }
    }
    overrides = {"benchmark": {"server_metrics": {"enabled": True, "urls": []}}}

    normalized = normalize_server_metrics_base_for_override(base, overrides)

    normalized_section = normalized["benchmark"][server_metrics_key]
    assert "url" not in normalized_section
    assert normalized_section["urls"] == ["http://localhost:9090/metrics"]
