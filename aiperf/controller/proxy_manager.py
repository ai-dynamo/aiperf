# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import zmq.asyncio

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import ZMQProxyType
from aiperf.common.environment import Environment
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

        try:
            self.debug("Terminating ZMQ context")
            await asyncio.wait_for(
                asyncio.to_thread(zmq.asyncio.Context.instance().term),
                timeout=Environment.ZMQ_CONTEXT_TERM_TIMEOUT,
            )
            self.debug("ZMQ context terminated successfully")
        except BaseException as e:
            self.warning(f"Error terminating ZMQ context: {e}")
