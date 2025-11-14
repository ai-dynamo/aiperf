# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.config import ServiceConfig
from aiperf.common.enums import ZMQProxyType
from aiperf.common.factories import ZMQProxyFactory
from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin


class ProxyManager(AIPerfLifecycleMixin):
    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(service_config=service_config, **kwargs)
        self.service_config = service_config

    @on_init
    async def _initialize_proxies(self) -> None:
        comm_config = self.service_config.comm_config
        self.proxies = [
            ZMQProxyFactory.create_instance(
                ZMQProxyType.XPUB_XSUB,
                zmq_proxy_config=comm_config.event_bus_proxy_config,
            ),
            ZMQProxyFactory.create_instance(
                ZMQProxyType.DEALER_ROUTER,
                zmq_proxy_config=comm_config.dataset_manager_proxy_config,
            ),
            ZMQProxyFactory.create_instance(
                ZMQProxyType.PUSH_PULL,
                zmq_proxy_config=comm_config.raw_inference_proxy_config,
            ),
        ]
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
