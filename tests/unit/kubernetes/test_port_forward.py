# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.port_forward.

Focuses on:
- Port-forward subprocess lifecycle (start, ready detection, cleanup)
- API readiness verification with retry logic
- WebSocket progress streaming with reconnection
- Context manager wrappers
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from aiperf.kubernetes.port_forward import (
    _API_MAX_RETRIES,
    _wait_for_port_forward_ready,
    cleanup_port_forward,
    port_forward_to_controller,
    port_forward_with_status,
    start_port_forward,
    stream_progress_from_api,
)

# ============================================================
# Helpers
# ============================================================


async def _async_iter(items: list) -> AsyncIterator:
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


def _make_mock_process(
    *,
    returncode: int | None = None,
    stdout_lines: list[bytes] | None = None,
    stderr_data: bytes = b"",
) -> MagicMock:
    """Build a mock asyncio.subprocess.Process.

    Args:
        returncode: Process return code (None = still running).
        stdout_lines: Lines to yield from stdout.readline().
        stderr_data: Data returned by stderr.read().
    """
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = returncode
    proc.pid = 12345

    if stdout_lines is not None:
        stdout = AsyncMock()
        lines = list(stdout_lines)

        async def _readline():
            if lines:
                return lines.pop(0)
            # After all lines consumed, simulate EOF
            proc.returncode = returncode if returncode is not None else 0
            return b""

        stdout.readline = _readline
        proc.stdout = stdout
    else:
        proc.stdout = None

    stderr = AsyncMock()
    stderr.read = AsyncMock(return_value=stderr_data)
    proc.stderr = stderr

    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    return proc


# ============================================================
# _wait_for_port_forward_ready
# ============================================================


class TestWaitForPortForwardReady:
    """Verify detection of kubectl's 'Forwarding from' ready message."""

    async def test_parses_port_from_forwarding_message(self) -> None:
        """Standard kubectl output is parsed to extract the port number."""
        proc = _make_mock_process(
            stdout_lines=[b"Forwarding from 127.0.0.1:54321 -> 9090\n"],
        )
        result = await _wait_for_port_forward_ready(proc)
        assert result == 54321

    async def test_returns_none_when_stdout_is_none(self) -> None:
        """If stdout was not captured, returns None immediately."""
        proc = _make_mock_process(stdout_lines=None)
        proc.stdout = None
        result = await _wait_for_port_forward_ready(proc)
        assert result is None

    async def test_returns_none_when_process_already_exited(self) -> None:
        """If the process has already exited before reading, returns None."""
        proc = _make_mock_process(returncode=1, stdout_lines=[])
        result = await _wait_for_port_forward_ready(proc)
        assert result is None

    async def test_skips_non_matching_lines(self) -> None:
        """Lines that don't match the forwarding pattern are skipped."""
        proc = _make_mock_process(
            stdout_lines=[
                b"some diagnostic line\n",
                b"Forwarding from 127.0.0.1:9999 -> 9090\n",
            ],
        )
        result = await _wait_for_port_forward_ready(proc)
        assert result == 9999

    @pytest.mark.parametrize(
        "line,expected_port",
        [
            (b"Forwarding from 127.0.0.1:0 -> 9090\n", 0),
            (b"Forwarding from 127.0.0.1:65535 -> 9090\n", 65535),
            (b"Forwarding from 127.0.0.1:8080 -> 80\n", 8080),
        ],
    )  # fmt: skip
    async def test_parses_various_port_numbers(
        self, line: bytes, expected_port: int
    ) -> None:
        """Various valid port numbers in the forwarding message are parsed correctly."""
        proc = _make_mock_process(stdout_lines=[line])
        result = await _wait_for_port_forward_ready(proc)
        assert result == expected_port


# ============================================================
# start_port_forward (via _start_port_forward_process)
# ============================================================


class TestStartPortForward:
    """Verify start_port_forward subprocess management and API verification."""

    async def test_returns_process_and_port_without_api_verify(self) -> None:
        """When verify_api=False, returns as soon as port-forward is ready."""
        mock_proc = _make_mock_process(
            stdout_lines=[b"Forwarding from 127.0.0.1:12345 -> 9090\n"],
        )
        with patch(
            "aiperf.kubernetes.port_forward.asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ):
            proc, port = await start_port_forward(
                "ns", "pod-0", verify_api=False, timeout=5.0
            )
        assert port == 12345
        assert proc is mock_proc

    async def test_builds_kubectl_command_with_kubeconfig_and_context(self) -> None:
        """kubeconfig and kube_context args are passed through to kubectl."""
        mock_proc = _make_mock_process(
            stdout_lines=[b"Forwarding from 127.0.0.1:5555 -> 9090\n"],
        )
        with patch(
            "aiperf.kubernetes.port_forward.asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ) as mock_exec:
            await start_port_forward(
                "my-ns",
                "my-pod",
                local_port=5555,
                remote_port=8080,
                verify_api=False,
                kubeconfig="/home/user/.kube/config",
                kube_context="staging",
            )
        args = mock_exec.call_args[0]
        assert "--kubeconfig" in args
        assert "/home/user/.kube/config" in args
        assert "--context" in args
        assert "staging" in args
        assert "pod/my-pod" in args
        assert "-n" in args
        assert "my-ns" in args

    async def test_raises_on_timeout(self) -> None:
        """RuntimeError raised when port-forward doesn't become ready in time."""

        async def mock_start_pf_process(
            ns, pod, lp, rp, *, timeout, kubeconfig=None, kube_context=None
        ):
            raise RuntimeError(f"Port-forward did not become ready within {timeout}s.")

        with (
            patch(
                "aiperf.kubernetes.port_forward._start_port_forward_process",
                side_effect=mock_start_pf_process,
            ),
            pytest.raises(RuntimeError, match="did not become ready"),
        ):
            await start_port_forward("ns", "pod-0", verify_api=False, timeout=0.01)

    async def test_raises_when_process_exits_unexpectedly(self) -> None:
        """RuntimeError when process exits before producing ready message."""
        proc = _make_mock_process(
            returncode=1, stdout_lines=[], stderr_data=b"error: pod not found"
        )

        with (
            patch(
                "aiperf.kubernetes.port_forward.asyncio.create_subprocess_exec",
                AsyncMock(return_value=proc),
            ),
            pytest.raises(RuntimeError, match="exited unexpectedly"),
        ):
            await start_port_forward("ns", "pod-0", verify_api=False, timeout=5.0)

    async def test_verify_api_calls_health_endpoint(self) -> None:
        """When verify_api=True, the /health endpoint is checked."""
        mock_proc = _make_mock_process(
            stdout_lines=[b"Forwarding from 127.0.0.1:7777 -> 9090\n"],
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aiperf.kubernetes.port_forward.asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
        ):
            proc, port = await start_port_forward(
                "ns", "pod-0", verify_api=True, timeout=30.0
            )
        assert port == 7777
        mock_session.get.assert_called_with("http://127.0.0.1:7777/health")

    async def test_verify_api_retries_on_process_death(self) -> None:
        """Port-forward is restarted when process dies during API check."""
        call_count = 0

        async def mock_start_pf_process(
            ns, pod, lp, rp, *, timeout, kubeconfig=None, kube_context=None
        ):
            nonlocal call_count
            call_count += 1
            proc = _make_mock_process(
                stdout_lines=[b"Forwarding from 127.0.0.1:7777 -> 9090\n"],
            )
            if call_count == 1:
                # First attempt: simulate process death during API check
                return proc, 7777
            return proc, 8888

        api_call_count = 0

        async def mock_wait_api(port, proc):
            nonlocal api_call_count
            api_call_count += 1
            if api_call_count == 1:
                raise RuntimeError("Port-forward process exited")
            # Second call succeeds

        with (
            patch(
                "aiperf.kubernetes.port_forward._start_port_forward_process",
                side_effect=mock_start_pf_process,
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                side_effect=mock_wait_api,
            ),
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
            patch("aiperf.kubernetes.port_forward.print_info"),
        ):
            proc, port = await start_port_forward(
                "ns", "pod-0", verify_api=True, timeout=120.0
            )
        assert port == 8888

    async def test_verify_api_exceeds_max_retries_raises(self) -> None:
        """RuntimeError after exceeding max API verification retries."""

        async def mock_start_pf_process(
            ns, pod, lp, rp, *, timeout, kubeconfig=None, kube_context=None
        ):
            proc = _make_mock_process()
            return proc, 7777

        async def mock_wait_api(port, proc):
            raise RuntimeError("Port-forward process exited")

        with (
            patch(
                "aiperf.kubernetes.port_forward._start_port_forward_process",
                side_effect=mock_start_pf_process,
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                side_effect=mock_wait_api,
            ),
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
            patch("aiperf.kubernetes.port_forward.print_info"),
            pytest.raises(
                RuntimeError, match=f"failed after {_API_MAX_RETRIES} retries"
            ),
        ):
            await start_port_forward("ns", "pod-0", verify_api=True, timeout=600.0)


# ============================================================
# cleanup_port_forward
# ============================================================


class TestCleanupPortForward:
    """Verify graceful process termination."""

    async def test_delegates_to_terminate_process(self) -> None:
        """cleanup_port_forward calls terminate_process with correct args."""
        mock_proc = _make_mock_process()
        with patch(
            "aiperf.kubernetes.subproc.terminate_process", AsyncMock()
        ) as mock_terminate:
            await cleanup_port_forward(mock_proc, timeout=10.0)
        mock_terminate.assert_awaited_once_with(mock_proc, 10.0)

    async def test_uses_default_timeout(self) -> None:
        """Default timeout is passed through when not specified."""
        mock_proc = _make_mock_process()
        with patch(
            "aiperf.kubernetes.subproc.terminate_process", AsyncMock()
        ) as mock_terminate:
            await cleanup_port_forward(mock_proc)
        assert mock_terminate.call_args[0][1] == 5.0


# ============================================================
# port_forward_to_controller (context manager)
# ============================================================


class TestPortForwardToController:
    """Verify the async context manager lifecycle."""

    async def test_yields_port_and_cleans_up(self) -> None:
        """Context manager yields port and calls cleanup on exit."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 4444)),
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                AsyncMock(),
            ) as mock_cleanup,
        ):
            async with port_forward_to_controller("ns", "pod-0") as port:
                assert port == 4444
            mock_cleanup.assert_awaited_once_with(mock_proc)

    async def test_cleanup_called_on_exception(self) -> None:
        """Cleanup runs even when the body raises an exception."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 4444)),
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                AsyncMock(),
            ) as mock_cleanup,
        ):
            with pytest.raises(ValueError, match="boom"):
                async with port_forward_to_controller("ns", "pod-0") as _port:
                    raise ValueError("boom")
            mock_cleanup.assert_awaited_once_with(mock_proc)

    async def test_passes_all_arguments_through(self) -> None:
        """All arguments are forwarded to start_port_forward."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 5555)),
            ) as mock_start,
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
        ):
            async with port_forward_to_controller(
                "custom-ns",
                "my-pod",
                local_port=5555,
                remote_port=8080,
                verify_api=False,
                kubeconfig="/path/to/config",
                kube_context="prod",
            ) as _port:
                pass

        mock_start.assert_awaited_once_with(
            "custom-ns",
            "my-pod",
            5555,
            8080,
            verify_api=False,
            kubeconfig="/path/to/config",
            kube_context="prod",
        )


# ============================================================
# stream_progress_from_api
# ============================================================


class TestStreamProgressFromApi:
    """Verify WebSocket streaming with subscription and reconnection."""

    async def test_subscribes_and_streams_messages(self) -> None:
        """Subscribes to requested types and forwards messages to callback."""
        received: list[dict] = []

        async def on_message(data: dict) -> bool:
            received.append(data)
            return len(received) >= 2

        # Build mock WS messages
        msg1 = MagicMock()
        msg1.type = aiohttp.WSMsgType.TEXT
        msg1.data = b'{"type":"progress","value":1}'

        msg2 = MagicMock()
        msg2.type = aiohttp.WSMsgType.TEXT
        msg2.data = b'{"type":"progress","value":2}'

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(return_value={"type": "subscribed"})
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=None)
        mock_ws.__aiter__ = lambda self: _async_iter([msg1, msg2])

        mock_session = AsyncMock()
        mock_session.ws_connect = MagicMock(return_value=mock_ws)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
            )

        mock_ws.send_json.assert_awaited_once_with(
            {"type": "subscribe", "message_types": ["progress"]}
        )
        assert len(received) == 2
        assert received[0]["value"] == 1
        assert received[1]["value"] == 2

    async def test_stops_on_closed_message(self) -> None:
        """Streaming stops gracefully when server sends CLOSED."""
        received: list[dict] = []

        async def on_message(data: dict) -> bool:
            received.append(data)
            return False

        msg_closed = MagicMock()
        msg_closed.type = aiohttp.WSMsgType.CLOSED

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(return_value={"type": "subscribed"})
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=None)
        mock_ws.__aiter__ = lambda self: _async_iter([msg_closed])

        mock_session = AsyncMock()
        mock_session.ws_connect = MagicMock(return_value=mock_ws)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
            )
        assert received == []

    async def test_reconnects_on_error_message(self) -> None:
        """ERROR message type triggers reconnection attempt."""
        received: list[dict] = []

        async def on_message(data: dict) -> bool:
            received.append(data)
            return True

        msg_error = MagicMock()
        msg_error.type = aiohttp.WSMsgType.ERROR

        msg_ok = MagicMock()
        msg_ok.type = aiohttp.WSMsgType.TEXT
        msg_ok.data = b'{"ok":true}'

        def make_ws(first_call: bool):
            ws = AsyncMock()
            ws.send_json = AsyncMock()
            ws.receive_json = AsyncMock(return_value={"type": "subscribed"})
            ws.__aenter__ = AsyncMock(return_value=ws)
            ws.__aexit__ = AsyncMock(return_value=None)
            if first_call:
                ws.__aiter__ = lambda self: _async_iter([msg_error])
            else:
                ws.__aiter__ = lambda self: _async_iter([msg_ok])
            return ws

        ws_instances = [make_ws(True), make_ws(False)]

        mock_session = AsyncMock()
        mock_session.ws_connect = MagicMock(side_effect=ws_instances)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        sessions = [mock_session, mock_session]

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                side_effect=sessions,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
                max_retries=3,
            )
        assert len(received) == 1
        assert received[0]["ok"] is True

    async def test_raises_connection_error_after_max_retries(self) -> None:
        """ConnectionError raised when all retry attempts are exhausted."""
        mock_session = AsyncMock()
        mock_session.ws_connect = MagicMock(side_effect=aiohttp.ClientError("refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
            patch("aiperf.kubernetes.port_forward.print_info"),
            pytest.raises(
                ConnectionError, match="Failed to connect to API after 2 attempts"
            ),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                AsyncMock(),
                ["progress"],
                max_retries=2,
            )

    async def test_resets_retry_count_on_successful_subscribe(self) -> None:
        """Retry counter resets after a successful subscription confirmation."""
        received: list[dict] = []

        async def on_message(data: dict) -> bool:
            received.append(data)
            return True

        # First attempt: ClientError (retry_count -> 1)
        # Second attempt: succeeds with subscribed, then sends message
        msg_ok = MagicMock()
        msg_ok.type = aiohttp.WSMsgType.TEXT
        msg_ok.data = b'{"done":true}'

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(return_value={"type": "subscribed"})
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=None)
        mock_ws.__aiter__ = lambda self: _async_iter([msg_ok])

        call_count = 0

        def make_session(**kwargs):
            nonlocal call_count
            call_count += 1
            session = AsyncMock()
            if call_count == 1:
                session.ws_connect = MagicMock(
                    side_effect=aiohttp.ClientError("refused")
                )
            else:
                session.ws_connect = MagicMock(return_value=mock_ws)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            return session

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                side_effect=make_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
            patch("aiperf.kubernetes.port_forward.print_info"),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
                max_retries=3,
            )
        assert len(received) == 1

    async def test_callback_returning_true_stops_streaming(self) -> None:
        """When on_message returns True, streaming stops immediately."""
        call_count = 0

        async def on_message(data: dict) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        msg1 = MagicMock()
        msg1.type = aiohttp.WSMsgType.TEXT
        msg1.data = b'{"n":1}'

        msg2 = MagicMock()
        msg2.type = aiohttp.WSMsgType.TEXT
        msg2.data = b'{"n":2}'

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(return_value={"type": "subscribed"})
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=None)
        mock_ws.__aiter__ = lambda self: _async_iter([msg1, msg2])

        mock_session = AsyncMock()
        mock_session.ws_connect = MagicMock(return_value=mock_ws)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "aiperf.kubernetes.port_forward.aiohttp.ClientSession",
                return_value=mock_session,
            ),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(),
            ),
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["progress"],
            )
        # Only first message processed before stop
        assert call_count == 1


# ============================================================
# port_forward_with_status
# ============================================================


class TestPortForwardWithStatus:
    """Verify the status-logging context manager wrapper."""

    async def test_yields_port_with_status_logging(self) -> None:
        """port_forward_with_status yields the actual port and logs status."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 6666)),
            ),
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
            patch("aiperf.kubernetes.console.print_success") as mock_success,
            patch("aiperf.kubernetes.console.status_log") as mock_status,
        ):
            # status_log is a context manager
            mock_status.return_value.__enter__ = MagicMock()
            mock_status.return_value.__exit__ = MagicMock(return_value=False)

            async with port_forward_with_status("ns", "pod-0") as port:
                assert port == 6666
            mock_success.assert_called_once()
            assert "6666" in mock_success.call_args[0][0]

    async def test_uses_default_remote_port_from_environment(self) -> None:
        """When remote_port is None, uses K8sEnvironment.PORTS.API_SERVICE."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 6666)),
            ) as mock_start,
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
            patch("aiperf.kubernetes.console.print_success"),
            patch("aiperf.kubernetes.console.status_log") as mock_status,
            patch(
                "aiperf.kubernetes.environment.K8sEnvironment.PORTS",
                MagicMock(API_SERVICE=9090),
            ),
        ):
            mock_status.return_value.__enter__ = MagicMock()
            mock_status.return_value.__exit__ = MagicMock(return_value=False)

            async with port_forward_with_status("ns", "pod-0") as _port:
                pass

        # remote_port should be the 4th positional arg to start_port_forward
        call_args = mock_start.call_args
        assert call_args[0][3] == 9090

    async def test_custom_remote_port_overrides_default(self) -> None:
        """Explicit remote_port overrides the environment default."""
        mock_proc = _make_mock_process()

        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                AsyncMock(return_value=(mock_proc, 6666)),
            ) as mock_start,
            patch("aiperf.kubernetes.port_forward.cleanup_port_forward", AsyncMock()),
            patch("aiperf.kubernetes.console.print_success"),
            patch("aiperf.kubernetes.console.status_log") as mock_status,
        ):
            mock_status.return_value.__enter__ = MagicMock()
            mock_status.return_value.__exit__ = MagicMock(return_value=False)

            async with port_forward_with_status(
                "ns", "pod-0", remote_port=7070
            ) as _port:
                pass

        call_args = mock_start.call_args
        assert call_args[0][3] == 7070
