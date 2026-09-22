# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GPU telemetry CLI conversion validation.

Ports v1 ``_parse_gpu_telemetry_config`` behaviors that initially
weren't ported into ``build_gpu_telemetry``:

1. ``--no-gpu-telemetry`` + ``--gpu-telemetry`` mutex (error, not silently
   honoring the suppression).
2. ``.csv`` metrics-file existence check at convert time.
3. Warning when a local collector (e.g. nvml) is paired with non-localhost
   server URLs — the local agent only sees the local machine's GPUs.
"""

from __future__ import annotations

import logging

import pytest

from aiperf.config.flags._converter_telemetry import (
    _is_localhost_url,
    build_gpu_telemetry,
)
from aiperf.config.flags.cli_config import CLIConfig


def _make_cli(**overrides) -> CLIConfig:
    base = {
        "url": "http://localhost:8000/test",
        "model_names": ["test-model"],
    }
    base.update(overrides)
    return CLIConfig(**base)


class TestNoGpuTelemetryMutex:
    def test_both_flags_together_raises(self):
        cli = _make_cli(no_gpu_telemetry=True, gpu_telemetry=["dashboard"])
        with pytest.raises(
            ValueError, match="Cannot use both --no-gpu-telemetry and --gpu-telemetry"
        ):
            build_gpu_telemetry(cli)

    def test_no_gpu_telemetry_alone_disables(self):
        cli = _make_cli(no_gpu_telemetry=True)
        assert build_gpu_telemetry(cli) == {"enabled": False}

    def test_gpu_telemetry_alone_enables(self):
        cli = _make_cli(gpu_telemetry=["dashboard"])
        out = build_gpu_telemetry(cli)
        assert out["enabled"] is True


class TestCsvMetricsFileExistence:
    def test_missing_csv_raises_at_convert_time(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        cli = _make_cli(gpu_telemetry=[str(missing)])
        with pytest.raises(ValueError, match="GPU metrics file not found"):
            build_gpu_telemetry(cli)

    def test_existing_csv_is_accepted(self, tmp_path):
        csv = tmp_path / "metrics.csv"
        csv.write_text("dcgm_field,metric_name\n9001,gpu_utilization\n")
        cli = _make_cli(gpu_telemetry=[str(csv)])
        out = build_gpu_telemetry(cli)
        assert out["metrics_file"] == csv


class TestIsLocalhostUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://localhost",
            "localhost:8000",
            "::1:8000",
            "[::1]:8000",
            "http://[::1]:8000",
        ],
    )
    def test_recognizes_localhost(self, url):
        assert _is_localhost_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com:8000",
            "http://10.0.0.5:8000",
            "https://server.internal:9000",
        ],
    )
    def test_rejects_non_localhost(self, url):
        assert _is_localhost_url(url) is False


class TestLocalCollectorWithRemoteUrlsWarning:
    def test_warns_when_local_collector_used_with_remote_url(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(
            logging.WARNING, logger="aiperf.config.flags._converter_telemetry"
        )
        cli = _make_cli(urls=["http://remote-server:8000"], gpu_telemetry=["pynvml"])
        build_gpu_telemetry(cli)
        assert "non-localhost" in caplog.text.lower()
        assert "pynvml" in caplog.text.lower()

    def test_does_not_warn_when_local_collector_used_with_localhost(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(
            logging.WARNING, logger="aiperf.config.flags._converter_telemetry"
        )
        cli = _make_cli(urls=["http://localhost:8000"], gpu_telemetry=["pynvml"])
        build_gpu_telemetry(cli)
        assert "non-localhost" not in caplog.text.lower()

    def test_does_not_warn_for_dcgm_collector_with_remote_url(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(
            logging.WARNING, logger="aiperf.config.flags._converter_telemetry"
        )
        cli = _make_cli(
            urls=["http://remote-server:8000"],
            gpu_telemetry=["http://remote-server:9400/metrics"],
        )
        build_gpu_telemetry(cli)
        assert "non-localhost" not in caplog.text.lower()


class TestAmdAutoDetectRespectsAnExplicitPrefix:
    """AMD auto-detection must never overrule a collector the user named.

    The probe used to run over every URL whenever ``collector_type`` still
    equalled the hardcoded DCGM default. That test cannot distinguish "the user
    wrote ``dcgm:<url>``" from "nothing was specified", because both leave the
    default in place, so an explicit prefix could be silently flipped to
    ``amd_dme`` by the heuristic. Gating on whether the item was a bare URL is
    what separates the two.

    ``_detect_amd_exporter`` is forced True here: the point is that an explicit
    prefix wins even when the endpoint really does look like an AMD exporter.
    """

    @pytest.fixture(autouse=True)
    def _always_detects_amd(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "aiperf.config.flags._converter_telemetry._detect_amd_exporter",
            lambda url: True,
        )

    def test_explicit_dcgm_prefix_is_not_overridden(self):
        cli = _make_cli(gpu_telemetry=["dcgm:http://node:9400/metrics"])
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"

    def test_bare_url_still_auto_detects(self):
        cli = _make_cli(gpu_telemetry=["http://node:5000/metrics"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_explicit_amd_dme_prefix_is_honored(self):
        cli = _make_cli(gpu_telemetry=["amd_dme:http://node:5000/metrics"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_a_local_keyword_suppresses_detection_for_a_bare_url(self):
        """`--gpu-telemetry amdsmi http://...` names a collector by keyword, so
        the URL is a plain endpoint and must not re-decide the collector."""
        cli = _make_cli(gpu_telemetry=["amdsmi", "http://node:5000/metrics"])
        assert build_gpu_telemetry(cli)["collector"] == "amdsmi"
