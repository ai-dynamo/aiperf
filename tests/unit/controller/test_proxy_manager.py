# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for proxy_manager.py - ProxyManager lifecycle.
"""

from unittest.mock import patch

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.controller.proxy_manager import ProxyManager


class TestProxyManagerLifecycle:
    """Test ProxyManager lifecycle to ensure proper ZMQ context handling."""

    @pytest.mark.asyncio
    async def test_context_term_not_called_during_stop(self, mock_zmq_globally):
        """
        Test that context.term() is NOT called during proxy stop.

        This is critical because:
        1. The context is a singleton shared by all ZMQ clients in the process
        2. zmq_ctx_term() blocks in C code waiting for all sockets to close
        3. This causes indefinite hangs
        4. The OS kernel reliably cleans up resources on process exit
        """
        service_config = ServiceConfig()

        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(service_config=service_config)

            # Initialize, start, and stop the proxy manager
            await proxy_manager.initialize()
            await proxy_manager.start()
            await proxy_manager.stop()

            # Verify that context.term() was NEVER called
            mock_zmq_globally.term.assert_not_called()
