# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for OTel CLI conversion gating.

Ports v1 ``_parse_otel_config`` and ``_normalize_otel_metrics_url``:

1. ``--otel-url`` is normalized (auto ``http://`` prefix, ``/v1/metrics``
   path append, scheme/host validation, bare-port rejection).
2. ``--stream`` / ``--gen-ai-provider`` require ``--otel-url`` to be set;
   silently dropping them was the v2 regression.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import ServerMetricsFormat
from aiperf.config.flags._converter_telemetry import (
    _normalize_otel_metrics_url,
    build_otel,
    build_server_metrics,
)
from aiperf.config.flags.cli_config import CLIConfig


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


class TestServerMetricsCliParity:
    def test_default_formats_match_origin_main_json_csv(self):
        cli = _make_cli()
        assert cli.server_metrics_formats == [
            ServerMetricsFormat.JSON,
            ServerMetricsFormat.CSV,
        ]

    def test_no_server_metrics_with_server_metrics_raises(self):
        cli = _make_cli(no_server_metrics=True, server_metrics=["localhost:9400"])
        with pytest.raises(
            ValueError, match="Cannot use both --no-server-metrics and --server-metrics"
        ):
            build_server_metrics(cli)


class TestOtelUrlNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("localhost:4318", "http://localhost:4318/v1/metrics"),
            ("collector.example", "http://collector.example/v1/metrics"),
            (
                "http://collector:4318",
                "http://collector:4318/v1/metrics",
            ),
            (
                "http://collector:4318/v1/metrics",
                "http://collector:4318/v1/metrics",
            ),
            (
                "https://collector:4318/custom",
                "https://collector:4318/custom/v1/metrics",
            ),
            (
                "https://collector:4318/custom/",
                "https://collector:4318/custom/v1/metrics",
            ),
        ],
    )
    def test_normalizes_valid_url_forms(self, raw, expected):
        assert _normalize_otel_metrics_url(raw) == expected

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="--otel-url cannot be empty"):
            _normalize_otel_metrics_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="--otel-url cannot be empty"):
            _normalize_otel_metrics_url("   ")

    def test_bare_port_rejected(self):
        with pytest.raises(ValueError, match="Expected host"):
            _normalize_otel_metrics_url(":4318")

    def test_grpc_scheme_rejected(self):
        with pytest.raises(ValueError, match="OTLP/gRPC is not supported"):
            _normalize_otel_metrics_url("grpc://collector:4317")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="Only http and https schemes"):
            _normalize_otel_metrics_url("ftp://collector:21")


class TestStreamAndGenAiProviderRequireOtelUrl:
    def test_stream_without_otel_url_raises(self):
        cli = _make_cli(stream="metrics")
        with pytest.raises(ValueError, match="--stream.*--otel-url"):
            build_otel(cli)

    def test_gen_ai_provider_without_otel_url_raises(self):
        cli = _make_cli(gen_ai_provider="openai")
        with pytest.raises(ValueError, match="--gen-ai-provider.*--otel-url"):
            build_otel(cli)

    def test_both_without_otel_url_raises_with_both_named(self):
        cli = _make_cli(stream="metrics", gen_ai_provider="openai")
        with pytest.raises(ValueError) as exc:
            build_otel(cli)
        msg = str(exc.value)
        assert "--stream" in msg
        assert "--gen-ai-provider" in msg

    def test_otel_url_unblocks_stream(self):
        cli = _make_cli(otel_url="collector:4318", stream="metrics")
        otel = build_otel(cli)
        assert otel["metrics_url"] == "http://collector:4318/v1/metrics"
        assert otel["stream_metrics_enabled"] is True
        assert otel["stream_timing_enabled"] is False

    def test_otel_url_unblocks_repeated_stream_domains(self):
        cli = _make_cli(otel_url="collector:4318", stream=["metrics", "timing"])
        otel = build_otel(cli)
        assert otel["stream_metrics_enabled"] is True
        assert otel["stream_timing_enabled"] is True

    def test_otel_url_unblocks_gen_ai_provider(self):
        cli = _make_cli(otel_url="collector:4318", gen_ai_provider="openai")
        otel = build_otel(cli)
        assert otel["gen_ai_provider"] == "openai"

    def test_resource_attributes_without_otel_url_raises(self):
        cli = _make_cli(otel_resource_attributes=["team=inference"])
        with pytest.raises(ValueError, match="--otel-resource-attributes.*--otel-url"):
            build_otel(cli)

    def test_otel_url_unblocks_resource_attributes(self):
        cli = _make_cli(
            otel_url="collector:4318",
            otel_resource_attributes=["team=inference", "env=prod,region=us-west-2"],
        )
        otel = build_otel(cli)
        assert otel["custom_resource_attributes"] == {
            "team": "inference",
            "env": "prod",
            "region": "us-west-2",
        }

    @pytest.mark.parametrize(
        "resource_attrs",
        [
            ["missing_equals"],
            ["=missing-key"],
            ["missing-value="],
        ],
    )
    def test_resource_attributes_reject_malformed_entries(self, resource_attrs):
        cli = _make_cli(
            otel_url="collector:4318",
            otel_resource_attributes=resource_attrs,
        )
        with pytest.raises(ValueError, match="--otel-resource-attributes"):
            build_otel(cli)

    def test_no_otel_flags_returns_empty_dict(self):
        cli = _make_cli()
        assert build_otel(cli) == {}

    def test_default_stream_default_value_is_not_in_fields_set(self):
        """``stream`` defaults to ``"default"``; absent from CLI it must not
        trigger the require-otel-url error."""
        cli = _make_cli()
        # Confirm the default is the literal value, not in model_fields_set.
        assert "stream" not in cli.model_fields_set
        assert build_otel(cli) == {}
