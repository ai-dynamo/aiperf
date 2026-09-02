# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SocketDefaults.apply_to_socket`` — specifically the
platform-branching socket buffer behavior.

Background: aiperf historically called ``setsockopt(SO_SNDBUF, 10MB)`` and
``setsockopt(SO_RCVBUF, 10MB)`` on every aiohttp connection as a streaming
throughput optimization for Linux. On Windows that disables TCP Auto-Tuning
and at large values causes head-of-line blocking under concurrency — we
reproduced 9-minute aiohttp request stalls at ``--concurrency 6`` on Windows
Python 3.13 with these set. The fix skips both setsockopts on Windows so
the OS auto-sizes the buffer.

These tests pin the platform check in place so future refactors don't
silently re-enable the Windows-breaking path.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from aiperf.transports.http_defaults import SocketDefaults


def _socket_options_set(mock_sock: MagicMock) -> set[tuple[int, int]]:
    """Return the set of (level, optname) pairs setsockopt was called with."""
    return {call.args[:2] for call in mock_sock.setsockopt.call_args_list}


class TestSocketDefaultsBufferSizes:
    """Verify SO_SNDBUF / SO_RCVBUF are skipped on Windows only."""

    def test_windows_skips_sndbuf_and_rcvbuf(self) -> None:
        """On Windows the two large buffer setsockopts must NOT be called —
        leaving them in disables TCP Auto-Tuning and stalls aiohttp under
        concurrency. Regression for AIP-XXX (Py 3.13 / VDI sweep stall)."""
        mock_sock = MagicMock(spec=socket.socket)
        with patch("aiperf.transports.http_defaults.IS_WINDOWS", True):
            SocketDefaults.apply_to_socket(mock_sock)

        opts = _socket_options_set(mock_sock)
        assert (socket.SOL_SOCKET, socket.SO_SNDBUF) not in opts, (
            "SO_SNDBUF must not be set on Windows"
        )
        assert (socket.SOL_SOCKET, socket.SO_RCVBUF) not in opts, (
            "SO_RCVBUF must not be set on Windows"
        )

    def test_linux_sets_sndbuf_and_rcvbuf(self) -> None:
        """On Linux/macOS the explicit buffer sizes are still applied —
        they're the original streaming-throughput optimization."""
        mock_sock = MagicMock(spec=socket.socket)
        with patch("aiperf.transports.http_defaults.IS_WINDOWS", False):
            SocketDefaults.apply_to_socket(mock_sock)

        opts = _socket_options_set(mock_sock)
        assert (socket.SOL_SOCKET, socket.SO_SNDBUF) in opts, (
            "SO_SNDBUF must still be set on non-Windows for streaming throughput"
        )
        assert (socket.SOL_SOCKET, socket.SO_RCVBUF) in opts, (
            "SO_RCVBUF must still be set on non-Windows for streaming throughput"
        )

    def test_platform_independent_options_set_on_both(self) -> None:
        """TCP_NODELAY and SO_KEEPALIVE are NOT platform-gated and must be
        applied regardless. Sanity check that we didn't over-scope the
        Windows skip to include unrelated options."""
        for is_windows in (True, False):
            mock_sock = MagicMock(spec=socket.socket)
            with patch("aiperf.transports.http_defaults.IS_WINDOWS", is_windows):
                SocketDefaults.apply_to_socket(mock_sock)

            label = "windows" if is_windows else "non-windows"
            opts = _socket_options_set(mock_sock)
            assert (socket.SOL_TCP, socket.TCP_NODELAY) in opts, (
                f"TCP_NODELAY must be set on {label}"
            )
            assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE) in opts, (
                f"SO_KEEPALIVE must be set on {label}"
            )
