# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The local results API must serve the finished run before its listener closes.

Regression for NVBugs 7073234: a successful ``aiperf profile --api-port`` run
exited 0 and wrote ``.aiperf_results_ready.json``, but the API listener was
already gone, so a client polling through the whole run only ever saw
``running`` and an empty ``/api/results/list``. These tests poll the live API
for the life of the CLI and only assert on what the client actually observed
over HTTP.
"""

import asyncio
import contextlib
import socket
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import orjson
import pytest
import zstandard
from pytest import param
from yarl import URL

from aiperf.common.constants import IS_WINDOWS
from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import AIPerfSignalCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults

POLL_INTERVAL_SEC = 0.01
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=2.0)
JSON_ARTIFACT = "profile_export_aiperf.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class _Download:
    """One artifact download captured while the listener was still open."""

    status: int
    """HTTP status code."""
    headers: dict[str, str]
    """Response headers, lower-cased names."""
    body: bytes
    """Raw (still encoded) response body."""


@dataclass
class ResultsApiPoller:
    """Polls the results endpoints every 10 ms, as a QA client would."""

    base_url: str
    statuses: list[str] = field(default_factory=list)
    """Every distinct ``/api/results`` status, in the order first observed."""
    listed_files: list[str] = field(default_factory=list)
    """First nonempty ``/api/results/list`` file names seen with status terminal."""
    list_ready: bool = False
    """``ready`` flag of that listing."""
    identity_download: _Download | None = None
    """``JSON_ARTIFACT`` downloaded without compression."""
    zstd_download: _Download | None = None
    """``JSON_ARTIFACT`` downloaded with ``Accept-Encoding: zstd``."""
    traversal_status: int | None = None
    """Status of a path-traversal download attempt against the live listener."""
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        # Downloads use a non-decompressing session so the test checks the
        # encoding the server applied; the JSON endpoints may be compressed by
        # the API's middleware and are read decompressed.
        async with (
            aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session,
            aiohttp.ClientSession(
                timeout=REQUEST_TIMEOUT, auto_decompress=False
            ) as raw_session,
        ):
            while not self._stop.is_set():
                with contextlib.suppress(
                    aiohttp.ClientError, TimeoutError, ConnectionError
                ):
                    await self._poll_once(session, raw_session)
                await asyncio.sleep(POLL_INTERVAL_SEC)

    async def _poll_once(
        self, session: aiohttp.ClientSession, raw_session: aiohttp.ClientSession
    ) -> None:
        async with session.get(f"{self.base_url}/api/results") as resp:
            status = orjson.loads(await resp.read())["status"]
        if not self.statuses or self.statuses[-1] != status:
            self.statuses.append(status)
        if status == "running" or self.zstd_download is not None:
            return

        async with session.get(f"{self.base_url}/api/results/list") as resp:
            listing = orjson.loads(await resp.read())
        names = [entry["name"] for entry in listing["files"]]
        if not names:
            return
        if not self.listed_files:
            self.listed_files = names
            self.list_ready = listing["ready"]

        url = f"{self.base_url}/api/results/files/{JSON_ARTIFACT}"
        async with raw_session.get(
            url, headers={"Accept-Encoding": "identity"}
        ) as resp:
            self.identity_download = _Download(
                resp.status,
                {k.lower(): v for k, v in resp.headers.items()},
                await resp.read(),
            )
        async with raw_session.get(url, headers={"Accept-Encoding": "zstd"}) as resp:
            self.zstd_download = _Download(
                resp.status,
                {k.lower(): v for k, v in resp.headers.items()},
                await resp.read(),
            )
        # Pre-encoded so the client does not normalize the dot segments away.
        traversal = URL(
            f"{self.base_url}/api/results/files/..%2F..%2Fetc%2Fpasswd", encoded=True
        )
        async with raw_session.get(traversal) as resp:
            self.traversal_status = resp.status


def _profile_command(server_url: str, api_port: int, extra: str = "") -> str:
    return f"""
        aiperf profile \
            --model {defaults.model} \
            --url {server_url} \
            --endpoint-type chat \
            --streaming \
            --tokenizer builtin \
            --isl 16 \
            --osl 16 \
            --concurrency 2 \
            --workers-max {defaults.workers_max} \
            --no-gpu-telemetry \
            --no-server-metrics \
            --api-port {api_port} \
            --ui none \
            {extra}
    """


def _assert_served_final_artifacts(
    poller: ResultsApiPoller, artifact_dir: Path, terminal_status: str
) -> None:
    assert terminal_status in poller.statuses, (
        f"never observed '{terminal_status}' over HTTP; saw {poller.statuses}"
    )
    assert poller.listed_files, "never observed a nonempty /api/results/list"
    assert poller.list_ready is True
    assert JSON_ARTIFACT in poller.listed_files

    exports = list(artifact_dir.rglob(JSON_ARTIFACT))
    assert len(exports) == 1, exports
    on_disk = exports[0].read_bytes()

    identity = poller.identity_download
    assert identity is not None
    assert identity.status == 200
    assert identity.headers["content-type"].startswith("application/json")
    assert "content-encoding" not in identity.headers
    assert identity.headers["x-filename"] == JSON_ARTIFACT
    assert identity.headers["content-disposition"] == (
        f'attachment; filename="{JSON_ARTIFACT}"'
    )
    assert identity.body == on_disk
    orjson.loads(identity.body)

    compressed = poller.zstd_download
    assert compressed is not None
    assert compressed.status == 200
    assert compressed.headers["content-encoding"] == "zstd"
    assert compressed.headers["content-type"].startswith("application/json")
    assert (
        zstandard.ZstdDecompressor().decompressobj().decompress(compressed.body)
        == on_disk
    )

    assert poller.traversal_status == 400


def _assert_api_confirmed_terminal_state(log: str | None) -> None:
    assert log, "aiperf.log was not written"
    assert "confirmed the run's terminal state" in log
    assert "did not confirm the terminal state" not in log
    assert "not the final results" not in log


@pytest.mark.integration
@pytest.mark.asyncio
class TestResultsApiLifetime:
    """The results API outlives export/readiness publication (NVBugs 7073234)."""

    @pytest.mark.parametrize(
        "grace",
        [
            param("1.0", id="grace-1s"),
            param("5.0", id="grace-default-5s"),
        ],
    )  # fmt: skip
    async def test_completed_run_is_observable_and_downloadable_before_listener_closes(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        monkeypatch: pytest.MonkeyPatch,
        grace: str,
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", grace)
        api_port = _free_port()
        poller = ResultsApiPoller(f"http://127.0.0.1:{api_port}")

        async with mock_server_factory(ttft=5, itl=1) as server:
            poll_task = asyncio.create_task(poller.run())
            try:
                result = await cli.run(
                    _profile_command(server.url, api_port, "--request-count 10")
                )
            finally:
                poller.stop()
                await poll_task

        assert result.exit_code == 0
        assert result.request_count == 10
        assert poller.statuses[0] == "running"
        _assert_served_final_artifacts(poller, result.artifacts_dir, "complete")
        _assert_api_confirmed_terminal_state(result.log)

    async def test_zero_grace_still_confirms_terminal_state_before_retiring_the_api(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no grace window a poller may miss `complete`; the handshake may not.

        The controller must still get the API's confirmation over the control
        channel before it sends SHUTDOWN, so the state is set before the
        listener starts closing.
        """
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", "0")
        api_port = _free_port()

        async with mock_server_factory(ttft=5, itl=1) as server:
            result = await cli.run(
                _profile_command(server.url, api_port, "--request-count 10")
            )

        assert result.exit_code == 0
        assert result.request_count == 10
        _assert_api_confirmed_terminal_state(result.log)

    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="Graceful SIGINT cancellation is POSIX-only (see test_ctrl_c_cancellation).",
    )
    async def test_cancelled_run_is_observable_and_downloadable_before_listener_closes(
        self,
        signal_cli: AIPerfSignalCLI,
        mock_server_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", "1.0")
        api_port = _free_port()
        poller = ResultsApiPoller(f"http://127.0.0.1:{api_port}")

        async with mock_server_factory(ttft=5, itl=1) as server:
            poll_task = asyncio.create_task(poller.run())
            try:
                result = await signal_cli.run_with_sigint(
                    _profile_command(server.url, api_port, "--benchmark-duration 30"),
                    sigint_delay=1.0,
                    wait_for_profiling=True,
                )
            finally:
                poller.stop()
                await poll_task

        assert result.json is not None
        assert result.json.was_cancelled is True
        _assert_served_final_artifacts(poller, result.artifacts_dir, "cancelled")
        _assert_api_confirmed_terminal_state(result.log)
