# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import NodeConfig, ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommandType,
    ServiceType,
)
from aiperf.common.enums.system_enums import SystemState
from aiperf.common.factories import (
    ServiceFactory,
    ServiceManagerFactory,
)
from aiperf.common.hooks import on_command, on_init, on_start, on_stop
from aiperf.common.messages import (
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
)
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.system_mixins import SignalHandlerMixin


@ServiceFactory.register(ServiceType.NODE_CONTROLLER)
class NodeController(SignalHandlerMixin, BaseComponentService):
    """Node Controller service.

    This service is responsible for managing the lifecycle of all other services
    on a single node. It will start and stop all other services on the node.
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig,
        node_config: NodeConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )
        self.debug("Creating Node Controller")
        self._system_state = SystemState.INITIALIZING
        self.node_config = node_config

        required_services: dict[ServiceTypeT, int] = {
            ServiceType.WORKER: self.service_config.workers.max or 10,
        }
        if self.service_config.record_processor_service_count is not None:
            required_services[ServiceType.RECORD_PROCESSOR] = (
                self.service_config.record_processor_service_count
            )

        self.service_manager: ServiceManagerProtocol = (
            ServiceManagerFactory.create_instance(
                self.service_config.service_run_type.value,
                required_services=required_services,
                user_config=self.user_config,
                service_config=self.service_config,
                log_queue=None,
            )
        )

        self._stop_tasks: set[asyncio.Task] = set()
        self.info(f"Node Controller {self.service_id} created")

    @on_init
    async def _initialize_node_controller(self) -> None:
        """Initialize the node controller."""
        # Start all required services
        self.info(f"Node Controller {self.service_id} is bootstrapping services")
        await self.service_manager.initialize()
        await self.service_manager.start()

    @on_start
    async def _start_services(self) -> None:
        """Bootstrap the node services."""
        self.system_state = SystemState.CONFIGURING

    @property
    def system_state(self) -> SystemState:
        """Get the current state of the system."""
        return self._system_state

    @system_state.setter
    def system_state(self, state: SystemState) -> None:
        """Set the current state of the system."""
        if state == self._system_state:
            return
        self.info(f"AIPerf Node is {state.name}")
        self._system_state = state

    @on_command(CommandType.SPAWN_WORKERS)
    async def _handle_spawn_workers_command(self, message: SpawnWorkersCommand) -> None:
        """Handle a spawn workers command."""
        self.debug(lambda: f"Received spawn workers command: {message}")
        # Spawn the workers
        await self.service_manager.run_service(ServiceType.WORKER, message.num_workers)

    @on_command(CommandType.SHUTDOWN_WORKERS)
    async def _handle_shutdown_workers_command(
        self, message: ShutdownWorkersCommand
    ) -> None:
        """Handle a shutdown workers command."""
        self.debug(lambda: f"Received shutdown workers command: {message}")
        # TODO: Handle individual worker shutdowns via worker id
        await self.service_manager.stop_service(ServiceType.WORKER)

    @on_stop
    async def _stop_node_controller(self) -> None:
        """Stop the node controller and all running services."""
        self.system_state = SystemState.STOPPING

        await self.service_manager.shutdown_all_services()
        await self.comms.stop()

        # Exit the process in a more explicit way, to ensure that it stops
        os._exit(0)

    async def _kill(self):
        """Kill the node controller."""
        try:
            await self.service_manager.kill_all_services()
        except Exception as e:
            raise self._service_error("Failed to stop all services") from e

        await super()._kill()


def main() -> None:
    """Main entry point for the node controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(NodeController)


if __name__ == "__main__":
    main()
    os._exit(0)
