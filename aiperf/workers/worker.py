# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import time

from aiperf.clients import InferenceClientFactory
from aiperf.clients.client_interfaces import RequestConverterFactory
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.comms.base import (
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    NANOS_PER_SECOND,
)
from aiperf.common.dataset_models import Turn
from aiperf.common.enums import (
    CommunicationClientAddressType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    aiperf_task,
    on_configure,
    on_init,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    ConversationRequestMessage,
    ConversationResponseMessage,
    CreditDropMessage,
    CreditReturnMessage,
    ErrorMessage,
    InferenceResultsMessage,
    WorkerHealthMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.record_models import ErrorDetails, RequestRecord
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.worker_models import WorkerPhaseTaskStats


@ServiceFactory.register(ServiceType.WORKER)
class Worker(BaseComponentService, ProcessHealthMixin):
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

        self.health_check_interval = self.service_config.worker_health_check_interval

        self.task_stats: dict[CreditPhase, WorkerPhaseTaskStats] = {}

        self.credit_drop_client: PullClientProtocol = self.comms.create_pull_client(
            CommunicationClientAddressType.CREDIT_DROP,
        )  # type: ignore
        self.credit_return_client: PushClientProtocol = self.comms.create_push_client(
            CommunicationClientAddressType.CREDIT_RETURN,
        )  # type: ignore
        self.inference_results_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommunicationClientAddressType.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )  # type: ignore
        self.conversation_data_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommunicationClientAddressType.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )  # type: ignore

        self.model_endpoint = ModelEndpointInfo.from_user_config(self.user_config)

        self.debug(
            lambda: f"Creating inference client for {self.model_endpoint.endpoint.type}, class: {InferenceClientFactory.get_class_from_type(self.model_endpoint.endpoint.type).__name__}",
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
    async def _do_initialize(self) -> None:
        self.debug("Initializing worker")

        await self.credit_drop_client.register_pull_callback(
            MessageType.CREDIT_DROP, self._process_credit_drop
        )

        self.debug("Worker initialized")

    @on_configure
    async def _configure(self, message: CommandMessage) -> None:
        pass

    async def _process_credit_drop(self, message: CreditDropMessage) -> None:
        """Process a credit drop message.

        - Every credit must be returned after processing
        - All results or errors should be converted to a RequestRecord and pushed to the inference results client.

        NOTE: This function MUST NOT return until the credit drop is fully processed.
        This is to ensure that the max concurrency is respected via the semaphore of the pull client.
        """
        # TODO: Add tests to ensure that the above note is never violated in the future

        self.trace(lambda: f"Processing credit drop: {message}")
        drop_perf_ns = time.perf_counter_ns()  # The time the credit was received

        if message.phase not in self.task_stats:
            self.task_stats[message.phase] = WorkerPhaseTaskStats()
        self.task_stats[message.phase].total += 1

        record: RequestRecord = RequestRecord()
        try:
            record = await self._execute_single_credit(message, time.time_ns())

        except Exception as e:
            self.exception(f"Error processing credit drop: {e}")
            record.error = ErrorDetails.from_exception(e)
            record.end_perf_ns = time.perf_counter_ns()

        finally:
            record.credit_phase = message.phase
            msg = InferenceResultsMessage(
                service_id=self.service_id,
                record=record,
            )

            if not record.valid:
                self.task_stats[message.phase].failed += 1
            else:
                self.task_stats[message.phase].completed += 1

            try:
                await self.inference_results_client.push(message=msg)
            except Exception as e:
                # If we fail to push the record, log the error and continue
                self.exception(f"Error pushing request record: {e}")
            finally:
                # Calculate the latency of the credit drop
                pre_inference_ns = record.start_perf_ns - drop_perf_ns
                # Always return the credits
                return_message = CreditReturnMessage(
                    service_id=self.service_id,
                    conversation_id=message.conversation_id,
                    credit_drop_ns=message.credit_drop_ns,
                    delayed_ns=None,
                    pre_inference_ns=pre_inference_ns,
                    phase=message.phase,
                )
                self.trace(lambda: f"Returning credit {return_message}")
                await self.credit_return_client.push(
                    message=return_message,
                )

    async def _execute_single_credit(
        self, message: CreditDropMessage, timestamp_ns: int
    ) -> RequestRecord:
        """Run a credit task for a single credit."""

        if not self.inference_client:
            raise NotInitializedError("Inference server client not initialized.")

        # retrieve the prompt from the dataset
        response: ConversationResponseMessage = (
            await self.conversation_data_client.request(
                ConversationRequestMessage(
                    service_id=self.service_id,
                    conversation_id=message.conversation_id,
                    credit_phase=message.phase,
                )
            )
        )
        self.trace(lambda: f"Received response message: {response}")

        if isinstance(response, ErrorMessage):
            return RequestRecord(
                timestamp_ns=timestamp_ns,
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=response.error,
            )

        return await self._call_inference_api(
            message, response.conversation.turns[0], timestamp_ns
        )

    async def _call_inference_api(
        self, message: CreditDropMessage, turn: Turn, timestamp_ns: int
    ) -> RequestRecord:
        """Make a single call to the inference API. Will return an error record if the call fails."""
        self.trace("Calling inference API")
        formatted_payload = None
        try:
            # Format payload for the API request
            formatted_payload = await self.request_converter.format_payload(
                model_endpoint=self.model_endpoint,
                turn=turn,
            )

            # NOTE: Current implementation of the TimingManager bypasses this, it is for future use.
            # Wait for the credit drop time if it is in the future.
            # Note that we check this after we have retrieved the data from the dataset, to ensure
            # that we are fully ready to go.
            delayed_ns = None
            drop_ns = message.credit_drop_ns
            now_ns = time.time_ns()
            if drop_ns and drop_ns > now_ns:
                await asyncio.sleep((drop_ns - now_ns) / NANOS_PER_SECOND)
            elif drop_ns and drop_ns < now_ns:
                delayed_ns = now_ns - drop_ns

            # Send the request to the Inference Server API and wait for the response
            result = await self.inference_client.send_request(
                model_endpoint=self.model_endpoint,
                payload=formatted_payload,
            )

            result.delayed_ns = delayed_ns
            return result

        except Exception as e:
            self.exception(
                f"Error calling inference server API at {self.model_endpoint.url}: {e}"
            )
            return RequestRecord(
                # Use the formatted payload if it is available, otherwise use the turn.
                request=formatted_payload or turn,
                timestamp_ns=timestamp_ns,
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )

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

    @on_stop
    async def _do_shutdown(self) -> None:
        self.debug("Shutting down worker")
        if self.inference_client:
            await self.inference_client.close()


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(Worker)


if __name__ == "__main__":
    sys.exit(main())
