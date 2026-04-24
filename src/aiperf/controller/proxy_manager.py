# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ZMQProxyType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class ProxyManager(AIPerfLifecycleMixin):
    def __init__(
        self,
        run: BenchmarkRun,
        *,
        enable_event_bus: bool = False,
        enable_dataset_manager: bool = False,
        enable_raw_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.run = run
        self._enable_event_bus = enable_event_bus
        self._enable_dataset_manager = enable_dataset_manager
        self._enable_raw_inference = enable_raw_inference

    @on_init
    async def _initialize_proxies(self) -> None:
        comm_config = self.run.resolved.comm_config or self.run.cfg.comm_config
        self.proxies = []

        if self._enable_event_bus:
            XPubXSubClass = plugins.get_class(
                PluginType.ZMQ_PROXY, ZMQProxyType.XPUB_XSUB
            )
            self.proxies.append(
                XPubXSubClass(zmq_proxy_config=comm_config.event_bus_proxy_config)
            )

        if self._enable_dataset_manager:
            DealerRouterClass = plugins.get_class(
                PluginType.ZMQ_PROXY, ZMQProxyType.DEALER_ROUTER
            )
            self.proxies.append(
                DealerRouterClass(
                    zmq_proxy_config=comm_config.dataset_manager_proxy_config
                )
            )

        if self._enable_raw_inference:
            PushPullClass = plugins.get_class(
                PluginType.ZMQ_PROXY, ZMQProxyType.PUSH_PULL
            )
            self.proxies.append(
                PushPullClass(zmq_proxy_config=comm_config.raw_inference_proxy_config)
            )

        for proxy in self.proxies:
            await proxy.initialize()
        self.debug("All proxies initialized successfully")

    @on_start
    async def _start_proxies(self) -> None:
        self.debug("Starting all proxies")
        for proxy in self.proxies:
            await proxy.start()
        self.debug("All proxies started successfully")

    @on_stop
    async def _stop_proxies(self) -> None:
        self.debug("Stopping all proxies")
        for proxy in self.proxies:
            await proxy.stop()
        self.debug("All proxies stopped successfully")

        # Note: We intentionally do NOT call context.term() here because:
        #
        # 1. The context is a singleton shared by all ZMQ clients in this process
        # 2. zmq_ctx_term() blocks in C code waiting for all sockets to close
        # 3. Even if called in a thread, Python may wait for that thread on shutdown
        # 4. asyncio timeouts CANNOT interrupt blocking C code in threads
        # 5. This causes indefinite hangs
        #
        # Instead, we let the process handle cleanup:
        # - Normal completion: os._exit() forcefully cleans up (no ResourceWarnings)
        # - Exception path: May get ResourceWarning, but better than infinite hang
        # - The OS kernel reliably cleans up all resources on process exit
        #
        # This is the recommended approach per PyZMQ documentation for processes
        # that exit after completing work.
        self.debug("Proxy manager stopped (context cleanup delegated to process exit)")
