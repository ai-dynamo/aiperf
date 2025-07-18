# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from aiperf.clients import InferenceClientFactory
from aiperf.clients.client_interfaces import RequestConverterFactory
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.comms.base import (
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommunicationClientAddressType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    aiperf_task,
    on_configure,
    on_init,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    CreditDropMessage,
    CreditReturnMessage,
    WorkerHealthMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.models import WorkerPhaseTaskStats
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.workers.credit_processor_mixin import CreditProcessorMixin


@ServiceFactory.register(ServiceType.WORKER)
class Worker(BaseComponentService, ProcessHealthMixin, CreditProcessorMixin):
    """Worker is primarily responsible for making API calls to the inference server.
    It also manages the conversation between turns and returns the results to the Inference Results Parsers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None = None,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

        self.debug(lambda: f"Initializing worker process: {self.process.pid}")

        self.health_check_interval = (
            self.service_config.workers.health_check_interval_seconds
        )

        self.task_stats: dict[CreditPhase, WorkerPhaseTaskStats] = {}

        self.credit_drop_pull_client: PullClientProtocol = (
            self.comms.create_pull_client(
                CommunicationClientAddressType.CREDIT_DROP,
            )
        )
        self.credit_return_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommunicationClientAddressType.CREDIT_RETURN,
            )
        )
        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommunicationClientAddressType.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommunicationClientAddressType.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        self.debug(
            lambda: f"Creating inference client for {self.model_endpoint.endpoint.type}, "
            f"class: {InferenceClientFactory.get_class_from_type(self.model_endpoint.endpoint.type).__name__}",
        )
        self.request_converter = RequestConverterFactory.create_instance(
            self.model_endpoint.endpoint.type,
        )
        self.inference_client = InferenceClientFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )

    @property
    def service_type(self) -> ServiceType:
        return ServiceType.WORKER

    @on_init
    async def _initialize_worker(self) -> None:
        self.debug("Initializing worker")

        await self.credit_drop_pull_client.register_pull_callback(
            MessageType.CREDIT_DROP, self._credit_drop_callback
        )

        self.debug("Worker initialized")

    @on_configure
    async def _configure_worker(self, message: CommandMessage) -> None:
        # NOTE: Right now we are configuring the worker in the __init__ method,
        #       but that may change based on how we implement sweeps.
        pass

    async def _credit_drop_callback(self, message: CreditDropMessage) -> None:
        """Handle an incoming credit drop message. Every credit must be returned after processing."""

        # Create a default credit return message in case of an exception
        credit_return_message = CreditReturnMessage(
            service_id=self.service_id,
            phase=message.phase,
        )

        try:
            # NOTE: This must be awaited to ensure that the max concurrency is respected
            credit_return_message = await self._process_credit_drop_internal(message)
        except Exception as e:
            self.exception(f"Error processing credit drop: {e}")
        finally:
            # It is fine to execute the push asynchronously here because the worker is technically
            # ready to process the next credit drop.
            self.execute_async(
                self.credit_return_push_client.push(credit_return_message)
            )

    @on_stop
    async def _shutdown_worker(self) -> None:
        self.debug("Shutting down worker")
        if self.inference_client:
            await self.inference_client.close()

    @aiperf_task
    async def _health_check_task(self) -> None:
        """Task to report the health of the worker to the worker manager."""
        while True:
            try:
                health_message = self.create_health_message()
                await self.pub_client.publish(health_message)
            except Exception as e:
                self.exception(f"Error reporting health: {e}")
            except asyncio.CancelledError:
                self.debug("Health check task cancelled")
                break

            await asyncio.sleep(self.health_check_interval)

    def create_health_message(self) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            process=self.get_process_health(),
            task_stats=self.task_stats,
        )


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(Worker)


if __name__ == "__main__":
    main()
