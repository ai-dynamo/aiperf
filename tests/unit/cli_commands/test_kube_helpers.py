# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.port_forward module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_asyncio_process() -> MagicMock:
    """Create a mock asyncio subprocess."""
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = None
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


# =============================================================================
# _wait_for_port_forward_ready Tests
# =============================================================================


class TestWaitForPortForwardReady:
    """Tests for _wait_for_port_forward_ready function."""

    @pytest.mark.asyncio
    async def test_returns_port_on_forwarding_message(self) -> None:
        """Test returns parsed port when kubectl outputs forwarding message."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.returncode = None
        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )

        result = await _wait_for_port_forward_ready(proc)

        assert result == 9090

    @pytest.mark.asyncio
    async def test_returns_ephemeral_port(self) -> None:
        """Test parses ephemeral port from kubectl output."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.returncode = None
        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:54321 -> 9090\n"
        )

        result = await _wait_for_port_forward_ready(proc)

        assert result == 54321

    @pytest.mark.asyncio
    async def test_returns_none_when_stdout_none(self) -> None:
        """Test returns None when stdout is None."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.stdout = None

        result = await _wait_for_port_forward_ready(proc)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_process_exit(self) -> None:
        """Test returns None when process exits before ready."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.returncode = 1  # Process already exited
        proc.stdout = MagicMock()

        result = await _wait_for_port_forward_ready(proc)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_eof_when_process_exits(self) -> None:
        """Test handles EOF (empty bytes) and returns None when process exits."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.stdout = MagicMock()

        # Track calls and simulate process exit after EOF
        call_count = [0]
        returncode_values = [None, None, 1]  # Running, running, then exited

        async def mock_readline():
            call_count[0] += 1
            return b""  # Always return EOF

        proc.stdout.readline = mock_readline

        # Use a list to track returncode progression
        def get_returncode():
            idx = min(call_count[0], len(returncode_values) - 1)
            return returncode_values[idx]

        type(proc).returncode = property(lambda self: get_returncode())

        result = await _wait_for_port_forward_ready(proc)

        assert result is None

    @pytest.mark.asyncio
    async def test_parses_first_forwarding_line(self) -> None:
        """Test parses port from the first forwarding line seen."""
        from aiperf.kubernetes.port_forward import _wait_for_port_forward_ready

        proc = MagicMock()
        proc.returncode = None
        proc.stdout = MagicMock()

        call_count = [0]

        async def mock_readline():
            call_count[0] += 1
            if call_count[0] == 1:
                return b"some other output\n"
            return b"Forwarding from 127.0.0.1:8080 -> 9090\n"

        proc.stdout.readline = mock_readline

        result = await _wait_for_port_forward_ready(proc)

        assert result == 8080
        assert call_count[0] == 2


# =============================================================================
# _wait_for_api_ready Tests
# =============================================================================


class TestWaitForApiReady:
    """Tests for _wait_for_api_ready function."""

    @pytest.mark.asyncio
    async def test_raises_when_process_exits_immediately(self) -> None:
        """Test raises RuntimeError when port-forward process has already exited."""
        from aiperf.kubernetes.port_forward import _wait_for_api_ready

        proc = MagicMock()
        proc.returncode = 1  # Process already exited
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            await _wait_for_api_ready(9090, proc, check_interval=0.01)

        assert "Port-forward process exited" in str(exc_info.value)
        assert "code 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_process_exits_with_no_stderr(self) -> None:
        """Test raises RuntimeError with 'no output' when stderr is None."""
        from aiperf.kubernetes.port_forward import _wait_for_api_ready

        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = None

        with pytest.raises(RuntimeError) as exc_info:
            await _wait_for_api_ready(9090, proc, check_interval=0.01)

        assert "no output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_process_exits_with_empty_stderr(self) -> None:
        """Test raises RuntimeError with 'no output' when stderr is empty string."""
        from aiperf.kubernetes.port_forward import _wait_for_api_ready

        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"   ")  # Just whitespace

        with pytest.raises(RuntimeError) as exc_info:
            await _wait_for_api_ready(9090, proc, check_interval=0.01)

        assert "no output" in str(exc_info.value)


# =============================================================================
# start_port_forward Tests
# =============================================================================


class TestStartPortForward:
    """Tests for start_port_forward function."""

    @pytest.mark.asyncio
    async def test_successful_port_forward_without_api_verify(self) -> None:
        """Test successful port forwarding returns (process, port) without API verification."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )
        mock_proc.stderr = MagicMock()

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ):
            proc, port = await start_port_forward(
                "default", "my-pod", 9090, timeout=10, verify_api=False
            )

            assert proc is mock_proc
            assert port == 9090

    @pytest.mark.asyncio
    async def test_kubectl_command_construction(self) -> None:
        """Test correct kubectl command is constructed."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:8080 -> 9090\n"
        )

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ) as mock_exec:
            await start_port_forward(
                "my-namespace",
                "my-pod-name",
                local_port=8080,
                remote_port=9090,
                timeout=10,
                verify_api=False,
            )

            # Verify the command arguments
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert call_args[0] == "kubectl"
            assert call_args[1] == "port-forward"
            assert "-n" in call_args
            assert "my-namespace" in call_args
            assert "pod/my-pod-name" in call_args
            assert "8080:9090" in call_args

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_port_forward_timeout(self) -> None:
        """Test raises RuntimeError on timeout with diagnostic detail."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"some kubectl error")

        # Use an Event that never gets set - immune to asyncio.sleep mocking
        never_complete = asyncio.Event()

        async def never_ready(proc):
            await never_complete.wait()  # Never returns (event never set)
            return 9090

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_port_forward_ready",
                never_ready,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await start_port_forward("default", "my-pod", 9090, timeout=1)

        error_msg = str(exc_info.value)
        assert "did not become ready" in error_msg
        assert "some kubectl error" in error_msg
        assert "Check that the pod is running" in error_msg
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_port_forward_process_exits_early(self) -> None:
        """Test raises RuntimeError when process exits before becoming ready."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Error: pod not found")

        async def exit_immediately(proc):
            return None  # Process exited

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_port_forward_ready",
                exit_immediately,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await start_port_forward("default", "my-pod", 9090, timeout=10)

        assert "Port-forward exited unexpectedly" in str(exc_info.value)
        assert "Error: pod not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_port_forward_process_exits_early_no_stderr(self) -> None:
        """Test RuntimeError message when process exits with no stderr output."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")

        async def exit_immediately(proc):
            return None  # Process exited

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_port_forward_ready",
                exit_immediately,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await start_port_forward("default", "my-pod", 9090, timeout=10)

        assert "no error output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_port_forward_process_exits_early_stderr_none(self) -> None:
        """Test RuntimeError message when process exits with None stderr."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = None

        async def exit_immediately(proc):
            return None  # Process exited

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_port_forward_ready",
                exit_immediately,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await start_port_forward("default", "my-pod", 9090, timeout=10)

        assert "no error output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_api_false_skips_api_check(self) -> None:
        """Test verify_api=False skips API readiness check."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                AsyncMock(),
            ) as mock_api_ready,
        ):
            await start_port_forward(
                "default", "my-pod", 9090, timeout=10, verify_api=False
            )

            mock_api_ready.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_api_true_calls_api_check(self) -> None:
        """Test verify_api=True calls API readiness check."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                AsyncMock(),
            ) as mock_api_ready,
        ):
            await start_port_forward(
                "default", "my-pod", 9090, timeout=60, verify_api=True
            )

            mock_api_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_check_timeout_retries_then_raises(self, capsys) -> None:
        """API check that never succeeds exhausts retries and raises RuntimeError."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )

        async def api_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                api_timeout,
            ),
            pytest.raises(RuntimeError, match="Port-forward failed after"),
        ):
            await start_port_forward(
                "default", "my-pod", 9090, timeout=10, verify_api=True
            )

        captured = capsys.readouterr()
        assert "API not ready, restarting port-forward" in captured.out

    @pytest.mark.asyncio
    async def test_api_check_runtime_error_retries_then_succeeds(self, capsys) -> None:
        """Test API check RuntimeError triggers port-forward restart and retry."""
        from aiperf.kubernetes.port_forward import start_port_forward

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(
            return_value=b"Forwarding from 127.0.0.1:9090 -> 9090\n"
        )

        call_count = 0

        async def api_error_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Process died")

        with (
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=mock_proc),
            ),
            patch(
                "aiperf.kubernetes.port_forward._wait_for_api_ready",
                api_error_then_ok,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                AsyncMock(),
            ),
        ):
            proc, port = await start_port_forward(
                "default", "my-pod", 9090, timeout=60, verify_api=True
            )

            assert proc is mock_proc
            assert port == 9090

        captured = capsys.readouterr()
        assert "API not ready, restarting port-forward" in captured.out


# =============================================================================
# stream_progress_from_api Tests
# =============================================================================


class TestStreamProgressFromApi:
    """Tests for stream_progress_from_api function."""

    @pytest.mark.asyncio
    async def test_connection_error_raises_after_max_retries(self, capsys) -> None:
        """Test raises ConnectionError after max retries exhausted."""
        from aiperf.kubernetes.port_forward import stream_progress_from_api

        async def on_message(data: dict) -> bool:
            return False

        # Create a mock context manager that raises on entry
        mock_ws_cm = MagicMock()
        mock_ws_cm.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection failed")
        )
        mock_ws_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=mock_ws_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(ConnectionError) as exc_info,
        ):
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["PROGRESS"],
                max_retries=2,
            )

        assert "Failed to connect to API after 2 attempts" in str(exc_info.value)
        captured = capsys.readouterr()
        # Should see retry messages
        assert "Connection lost" in captured.out

    @pytest.mark.asyncio
    async def test_timeout_error_triggers_retry(self, capsys) -> None:
        """Test asyncio.TimeoutError triggers retry logic."""
        from aiperf.kubernetes.port_forward import stream_progress_from_api

        async def on_message(data: dict) -> bool:
            return False

        # Create a mock context manager that raises TimeoutError on entry
        mock_ws_cm = MagicMock()
        mock_ws_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_ws_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.ws_connect = MagicMock(return_value=mock_ws_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(ConnectionError),
        ):
            # Use max_retries=2 so we get 1 retry message before failing
            await stream_progress_from_api(
                "ws://localhost:9090/ws",
                on_message,
                ["PROGRESS"],
                max_retries=2,
            )

        captured = capsys.readouterr()
        assert "Connection lost" in captured.out


# =============================================================================
# cleanup_port_forward Tests
# =============================================================================


class TestCleanupPortForward:
    """Tests for cleanup_port_forward function."""

    @pytest.mark.asyncio
    async def test_already_exited_process(
        self, mock_asyncio_process: MagicMock
    ) -> None:
        """Test does nothing when process already exited."""
        from aiperf.kubernetes.port_forward import cleanup_port_forward

        mock_asyncio_process.returncode = 0  # Already exited

        await cleanup_port_forward(mock_asyncio_process)

        mock_asyncio_process.terminate.assert_not_called()
        mock_asyncio_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_termination(self, mock_asyncio_process: MagicMock) -> None:
        """Test graceful termination when process responds to SIGTERM."""
        from aiperf.kubernetes.port_forward import cleanup_port_forward

        mock_asyncio_process.returncode = None

        await cleanup_port_forward(mock_asyncio_process, timeout=5.0)

        mock_asyncio_process.terminate.assert_called_once()
        mock_asyncio_process.kill.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_force_kill_on_timeout(self, mock_asyncio_process: MagicMock) -> None:
        """Test force kills process when graceful termination times out."""
        from aiperf.kubernetes.port_forward import cleanup_port_forward

        mock_asyncio_process.returncode = None

        # First wait() call (inside wait_for) never completes, causing timeout
        # Second wait() call (after kill) succeeds
        never_complete = asyncio.Event()
        call_count = 0

        async def slow_then_fast_wait():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: never returns, causing wait_for to timeout
                await never_complete.wait()
            # Second call: returns immediately (process killed)
            return None

        mock_asyncio_process.wait = slow_then_fast_wait

        await cleanup_port_forward(mock_asyncio_process, timeout=1)

        mock_asyncio_process.terminate.assert_called_once()
        mock_asyncio_process.kill.assert_called_once()
