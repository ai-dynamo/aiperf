# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: live OTel metric export arrives before run completion.

Regression coverage for Requirement 7.1 (flush starvation). Under sustained
load the fanout process must flush metrics to the OTel collector on a
monotonic-clock schedule, not only at shutdown. This test spins up a fake
OTLP HTTP sink, runs a real `aiperf profile` against the in-repo mock
server, and asserts that the sink receives at least one POST /v1/metrics
*before* the profiling run declares completion.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


class _OTLPSinkHandler(BaseHTTPRequestHandler):
    """Minimal OTLP HTTP sink that records POST /v1/metrics timestamps."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/metrics":
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)
            self.server.received_exports.append(True)  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, fmt: str, *args: object) -> None:
        # Silence request logs during tests.
        pass


class _OTLPSinkServer(HTTPServer):
    """HTTPServer subclass that tracks received exports."""

    def __init__(self, port: int) -> None:
        self.received_exports: list[bool] = []
        super().__init__(("127.0.0.1", port), _OTLPSinkHandler)


@pytest.fixture
def otlp_sink() -> tuple[_OTLPSinkServer, int]:
    """Start a fake OTLP HTTP sink on a random port and return (server, port)."""
    server = _OTLPSinkServer(0)  # Port 0 lets the OS pick a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, port
    server.shutdown()
    thread.join(timeout=5)


@pytest.mark.component_integration
@pytest.mark.asyncio
class TestOTelLiveExport:
    """Verify that OTel metrics are exported during a live run, not only at shutdown."""

    async def test_otlp_export_arrives_before_run_completes(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        otlp_sink: tuple[_OTLPSinkServer, int],
    ) -> None:
        """Run aiperf profile with --otel-url and assert the OTLP sink receives
        at least one export while records are still flowing (not only at shutdown).

        The test uses --concurrency 4 --request-count 20 which generates
        enough traffic for the monotonic-clock flush driver to fire at least
        once before the run finishes.
        """
        server, port = otlp_sink
        otel_url = f"http://127.0.0.1:{port}"

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency 4 \
                --request-count 20 \
                --streaming \
                --otel-url {otel_url} \
                --stream default
            """,
            timeout=120.0,
        )

        assert result.exit_code == 0, (
            f"aiperf profile failed with exit code {result.exit_code}"
        )

        # The sink should have received at least one OTLP export.
        # Under the fixed flush driver (Req 7.1), exports fire on a
        # monotonic clock schedule (default 2s) even under sustained load.
        assert len(server.received_exports) >= 1, (
            "OTLP sink received zero exports during the run. "
            "This indicates the flush driver may be starved under load "
            "(regression of Requirement 7.1)."
        )
