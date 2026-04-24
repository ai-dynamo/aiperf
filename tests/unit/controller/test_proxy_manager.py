# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for proxy_manager.py - ProxyManager lifecycle and proxy selection.
"""

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.config import BenchmarkRun
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin import plugins

plugins_get_class_original = plugins.get_class


class TestProxyManagerSelection:
    """Test that enable flags control which proxies are created."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_count"),
        [
            param(
                {"enable_event_bus": True, "enable_dataset_manager": True, "enable_raw_inference": True},
                3,
                id="all_enabled",
            ),
            param(
                {"enable_event_bus": True, "enable_dataset_manager": True},
                2,
                id="controller_k8s_mode",
            ),
            param(
                {"enable_raw_inference": True},
                1,
                id="worker_pod_mode",
            ),
            param(
                {"enable_event_bus": True},
                1,
                id="event_bus_only",
            ),
            param(
                {"enable_dataset_manager": True},
                1,
                id="dataset_manager_only",
            ),
            param(
                {},
                0,
                id="none_enabled_default",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_proxy_count(
        self, mock_zmq, run: BenchmarkRun, kwargs: dict, expected_count: int
    ) -> None:
        """Verify correct number of proxies are created for given flags."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(run=run, **kwargs)
            await proxy_manager.initialize()

            assert len(proxy_manager.proxies) == expected_count

    @pytest.mark.asyncio
    async def test_flags_request_correct_proxy_types(
        self, mock_zmq, run: BenchmarkRun
    ) -> None:
        """Each enable flag requests the correct ZMQProxyType from the plugin registry."""
        from aiperf.plugin.enums import PluginType, ZMQProxyType

        with (
            patch("zmq.proxy_steerable"),
            patch(
                "aiperf.controller.proxy_manager.plugins.get_class",
                wraps=plugins_get_class_original,
            ) as spy,
        ):
            pm = ProxyManager(
                run=run,
                enable_event_bus=True,
                enable_dataset_manager=True,
                enable_raw_inference=True,
            )
            await pm.initialize()

            spy.assert_any_call(PluginType.ZMQ_PROXY, ZMQProxyType.XPUB_XSUB)
            spy.assert_any_call(PluginType.ZMQ_PROXY, ZMQProxyType.DEALER_ROUTER)
            spy.assert_any_call(PluginType.ZMQ_PROXY, ZMQProxyType.PUSH_PULL)
            assert spy.call_count == 3

    @pytest.mark.asyncio
    async def test_disabled_flags_skip_plugin_lookup(
        self, mock_zmq, run: BenchmarkRun
    ) -> None:
        """Disabled proxy flags do not call plugins.get_class for that type."""
        from aiperf.plugin.enums import PluginType, ZMQProxyType

        with (
            patch("zmq.proxy_steerable"),
            patch(
                "aiperf.controller.proxy_manager.plugins.get_class",
                wraps=plugins_get_class_original,
            ) as spy,
        ):
            pm = ProxyManager(
                run=run,
                enable_raw_inference=True,
            )
            await pm.initialize()

            spy.assert_called_once_with(PluginType.ZMQ_PROXY, ZMQProxyType.PUSH_PULL)


class TestProxyManagerLifecycle:
    """Test ProxyManager lifecycle hooks propagate to proxies."""

    @pytest.mark.asyncio
    async def test_context_term_not_called_during_stop(
        self, mock_zmq, run: BenchmarkRun
    ) -> None:
        """context.term() must NOT be called during proxy stop to avoid hangs."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(
                run=run,
                enable_event_bus=True,
                enable_dataset_manager=True,
                enable_raw_inference=True,
            )

            await proxy_manager.initialize()
            await proxy_manager.start()
            await proxy_manager.stop()

            mock_zmq.context.term.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_proxies_initialized(self, mock_zmq, run: BenchmarkRun) -> None:
        """Each proxy's initialize() is called during ProxyManager init."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(
                run=run,
                enable_event_bus=True,
                enable_raw_inference=True,
            )
            await proxy_manager.initialize()

            for proxy in proxy_manager.proxies:
                assert proxy._state.name == "INITIALIZED"

    @pytest.mark.asyncio
    async def test_all_proxies_started(self, mock_zmq, run: BenchmarkRun) -> None:
        """Each proxy transitions to RUNNING after start."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(
                run=run,
                enable_event_bus=True,
                enable_raw_inference=True,
            )
            await proxy_manager.initialize()
            await proxy_manager.start()

            for proxy in proxy_manager.proxies:
                assert proxy._state.name == "RUNNING"

    @pytest.mark.asyncio
    async def test_all_proxies_stopped(self, mock_zmq, run: BenchmarkRun) -> None:
        """Each proxy transitions to STOPPED after stop."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(
                run=run,
                enable_event_bus=True,
                enable_dataset_manager=True,
            )
            await proxy_manager.initialize()
            await proxy_manager.start()
            await proxy_manager.stop()

            for proxy in proxy_manager.proxies:
                assert proxy._state.name == "STOPPED"

    @pytest.mark.asyncio
    async def test_empty_lifecycle_succeeds(self, mock_zmq, run: BenchmarkRun) -> None:
        """Full lifecycle with no proxies enabled completes without error."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(run=run)
            await proxy_manager.initialize()
            await proxy_manager.start()
            await proxy_manager.stop()

            assert len(proxy_manager.proxies) == 0

    @pytest.mark.asyncio
    async def test_initialize_and_start_convenience(
        self, mock_zmq, run: BenchmarkRun
    ) -> None:
        """initialize_and_start() initializes and starts all proxies in one call."""
        with patch("zmq.proxy_steerable"):
            proxy_manager = ProxyManager(
                run=run,
                enable_raw_inference=True,
            )
            await proxy_manager.initialize_and_start()

            assert len(proxy_manager.proxies) == 1
            assert proxy_manager.proxies[0]._state.name == "RUNNING"
