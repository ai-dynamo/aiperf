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

import asyncio
import threading
import time
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
            self.server.export_timestamps_ns.append(time.monotonic_ns())  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, fmt: str, *args: object) -> None:
        pass


class _OTLPSinkServer(HTTPServer):
    """HTTPServer subclass that tracks received export timestamps."""

    def __init__(self, port: int) -> None:
        self.export_timestamps_ns: list[int] = []
        super().__init__(("127.0.0.1", port), _OTLPSinkHandler)


@pytest.fixture
def otlp_sink() -> tuple[_OTLPSinkServer, int]:
    """Start a fake OTLP HTTP sink on a random port and return (server, port)."""
    server = _OTLPSinkServer(0)
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert OTLP sink receives exports mid-run, not only at shutdown.

        Strategy: run the CLI in a background task and poll the sink for
        exports while the run is still in progress. The first export must
        arrive before the CLI process exits.
        """
        server, port = otlp_sink
        otel_url = f"http://127.0.0.1:{port}"

        # Force a sub-second flush interval so the mid-run export assertion is
        # robust on fast hosts that complete a 40-request mock-server run in
        # under the 2s default interval. The env var propagates into the
        # aiperf subprocess (create_subprocess_exec inherits os.environ).
        monkeypatch.setenv("AIPERF_OTEL_FLUSH_INTERVAL_SECONDS", "0.5")

        # Use enough requests and a small flush interval to guarantee
        # at least one mid-run flush fires before the run completes.
        cli_cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --concurrency 4 \
                --request-count 40 \
                --streaming \
                --otel-url {otel_url} \
                --stream default
        """

        # Launch the CLI in a background task so we can poll mid-run.
        cli_task = asyncio.create_task(cli.run(cli_cmd, timeout=120.0))

        # Poll for at least one export arriving while the CLI is still running.
        mid_run_export_received = False
        for _ in range(600):  # Up to 60s (100ms intervals)
            await asyncio.sleep(0.1)
            if server.export_timestamps_ns:
                # CLI task still running means this is genuinely mid-run.
                if not cli_task.done():
                    mid_run_export_received = True
                break
            if cli_task.done():
                break

        result = await cli_task

        assert result.exit_code == 0, (
            f"aiperf profile failed with exit code {result.exit_code}"
        )

        run_end_ns = time.monotonic_ns()

        # Primary assertion: at least one export arrived mid-run.
        assert mid_run_export_received, (
            "No OTLP export arrived while the CLI was still running. "
            f"Exports received: {len(server.export_timestamps_ns)}. "
            "This indicates the flush driver may be starved under load "
            "(regression of Requirement 7.1 — mid-run flush)."
        )

        # Secondary: the first export timestamp is well before run completion.
        first_export_ns = server.export_timestamps_ns[0]
        assert first_export_ns < run_end_ns, (
            "First export timestamp is not before run end. "
            f"first_export={first_export_ns}, run_end={run_end_ns}"
        )

        # Sanity: multiple exports should have arrived during a 40-request run.
        assert len(server.export_timestamps_ns) >= 1, (
            "OTLP sink received zero exports during the run."
        )
