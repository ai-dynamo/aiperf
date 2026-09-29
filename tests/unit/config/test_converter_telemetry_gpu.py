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

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aiperf.config.flags._converter_telemetry import (
    _detect_amd_exporter as _real_detect,
)
from aiperf.config.flags._converter_telemetry import (
    _is_localhost_url,
    build_gpu_telemetry,
)
from aiperf.config.flags.cli_config import CLIConfig


@pytest.fixture(autouse=True)
def _no_network_probe(monkeypatch: pytest.MonkeyPatch):
    """Keep the AMD exporter probe off the network for every test in this module.

    ``build_gpu_telemetry`` probes every bare URL over HTTP. Left alone, a test
    passing a bare URL makes a genuine outbound request that returns fast only
    because the hostname does not resolve, and would wait out the timeout
    behind a resolving wildcard DNS or a proxy. Tests that exercise the real
    probe point it at a loopback server instead.
    """
    monkeypatch.setattr(
        "aiperf.config.flags._converter_telemetry._detect_amd_exporter",
        lambda urls: False,
    )


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
            lambda urls: True,
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

    @pytest.mark.parametrize(
        "item",
        [
            "node:5000/metrics",
            "10.0.0.1:5000/metrics",
            "node:5000",
        ],
        ids=["host-port-path", "ip-port-path", "host-port"],
    )
    def test_a_scheme_less_url_still_auto_detects(self, item: str):
        """A scheme-less endpoint is a bare URL, even though it contains a colon.

        These reach the prefixed-item branch because of the colon, but the part
        before it is not a collector name, so the user chose nothing and the URL
        is still a detection candidate. Reported from a real MI300X run where
        `--gpu-telemetry <ip>:5000/metrics` silently collected no AMD metrics.
        """
        cli = _make_cli(gpu_telemetry=[item])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_an_explicit_prefix_on_a_scheme_less_url_is_still_honoured(self):
        cli = _make_cli(gpu_telemetry=["dcgm:node:9400/metrics"])
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"

    @pytest.mark.parametrize(
        "items",
        [
            ["dcgm:http://a:9400/metrics", "http://b:5000/metrics"],
            ["http://b:5000/metrics", "dcgm:http://a:9400/metrics"],
            ["dcgm:http://a:9400/metrics", "b:5000/metrics"],
        ],
        ids=["prefix-first", "bare-first", "scheme-less-sibling"],
    )
    def test_a_bare_sibling_url_does_not_overrule_an_explicit_prefix(
        self, items: list[str]
    ):
        """One collector serves every endpoint, so probing a bare sibling URL
        would re-decide the collector for the endpoint the user named."""
        cli = _make_cli(gpu_telemetry=items)
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"


_DME_PAGE = b"""# TYPE gpu_package_power gauge
gpu_package_power{gpu_id="0"} 212
gpu_gfx_activity{gpu_id="0"} 97
"""
_DCGM_PAGE = b"""# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 212
"""


class _ExporterHandler(BaseHTTPRequestHandler):
    pages = {"/amd": _DME_PAGE, "/dcgm": _DCGM_PAGE, "/amd-cold": _DME_PAGE}

    def do_GET(self):
        if self.path == "/amd-cold":
            time.sleep(1.5)  # a cold exporter answering its first scrape
        body = self.pages.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")

    def log_message(self, *args):
        pass


@pytest.fixture
def exporter() -> Iterator[str]:
    """Base URL of a loopback server serving an AMD and a DCGM metrics page."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ExporterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def real_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aiperf.config.flags._converter_telemetry._detect_amd_exporter",
        _real_detect,
    )


@pytest.mark.usefixtures("real_probe")
class TestAmdProbe:
    def test_an_amd_exporter_is_detected(self, exporter: str):
        cli = _make_cli(gpu_telemetry=[f"{exporter}/amd"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_a_dcgm_exporter_stays_on_dcgm(self, exporter: str):
        cli = _make_cli(gpu_telemetry=[f"{exporter}/dcgm"])
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"

    def test_any_amd_endpoint_among_several_is_enough(self, exporter: str):
        cli = _make_cli(gpu_telemetry=[f"{exporter}/missing", f"{exporter}/amd"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    async def test_the_probe_runs_inside_an_event_loop(self, exporter: str):
        """`aiperf kube profile` converts the CLI from inside its own loop."""
        asyncio.get_running_loop()
        cli = _make_cli(gpu_telemetry=[f"{exporter}/amd"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_an_unreachable_endpoint_warns_without_leaking_credentials(
        self, closed_port: int, caplog: pytest.LogCaptureFixture
    ):
        """A failed probe leaves the endpoint on DCGM, which collects nothing from
        an AMD exporter, so the downgrade has to be visible in the log."""
        caplog.set_level(
            logging.WARNING, logger="aiperf.config.flags._converter_telemetry"
        )
        url = f"http://ops:s3cr3t@127.0.0.1:{closed_port}/metrics"

        cli = _make_cli(gpu_telemetry=[url])
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"

        assert "could not probe" in caplog.text.lower()
        assert (
            f"amd_dme:http://<redacted>@127.0.0.1:{closed_port}/metrics" in caplog.text
        )
        assert "s3cr3t" not in caplog.text

    def test_an_exporter_slow_on_its_first_scrape_is_still_detected(
        self, exporter: str
    ):
        """Seen on an MI300X node: the first scrape after the exporter sat idle
        took over a second, and a one-second probe left it on DCGM."""
        cli = _make_cli(gpu_telemetry=[f"{exporter}/amd-cold"])
        assert build_gpu_telemetry(cli)["collector"] == "amd_dme"

    def test_a_timeout_says_so_rather_than_an_empty_error(
        self,
        exporter: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        monkeypatch.setattr(
            "aiperf.config.flags._converter_telemetry._DETECT_TIMEOUT_SEC", 0.2
        )
        caplog.set_level(
            logging.WARNING, logger="aiperf.config.flags._converter_telemetry"
        )

        cli = _make_cli(gpu_telemetry=[f"{exporter}/amd-cold"])
        assert build_gpu_telemetry(cli)["collector"] == "dcgm"

        assert "no response within 0.2 s" in caplog.text


class TestMistypedCollectorPrefix:
    def test_an_unknown_prefix_before_a_scheme_is_rejected(self):
        """`adm_dme:http://...` can only be a typo for a collector; folded into a
        URL it fails the probe with a suggestion that repeats the typo."""
        cli = _make_cli(gpu_telemetry=["adm_dme:http://node:5000/metrics"])
        with pytest.raises(
            ValueError, match="Unknown GPU telemetry collector 'adm_dme'"
        ) as err:
            build_gpu_telemetry(cli)
        assert "'amd_dme'" in str(err.value) and "'dcgm'" in str(err.value)

    def test_a_scheme_less_host_port_is_still_a_url(self):
        cli = _make_cli(gpu_telemetry=["node:5000/metrics"])
        assert build_gpu_telemetry(cli)["urls"] == ["http://node:5000/metrics"]
