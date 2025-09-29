# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import time
from collections.abc import Awaitable

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    AIPERF_HTTP_CONNECTION_LIMIT,
    DEFAULT_WORKER_HEALTH_CHECK_INTERVAL,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import CommAddress, CommandType, MessageType, ServiceType
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.factories import (
    InferenceClientFactory,
    RequestConverterFactory,
    ResponseExtractorFactory,
    ServiceFactory,
)
from aiperf.common.hooks import background_task, on_command, on_pull_message, on_stop
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    ConversationRequestMessage,
    ConversationResponseMessage,
    CreditDropMessage,
    CreditReturnMessage,
    ErrorMessage,
    InferenceResultsMessage,
    ProfileCancelCommand,
    WorkerHealthMessage,
)
from aiperf.common.mixins import ProcessHealthMixin, PullClientMixin
from aiperf.common.models import (
    ErrorDetails,
    RequestRecord,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    ResponseExtractorProtocol,
)


@ServiceFactory.register(ServiceType.WORKER)
class Worker(PullClientMixin, BaseComponentService, ProcessHealthMixin):
    """Worker is primarily responsible for making API calls to the inference server.
    It also manages the conversation between turns and returns the results to the Inference Results Parsers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            pull_client_address=CommAddress.CREDIT_DROP,
            pull_client_bind=False,
            # NOTE: We set the max concurrency to the same as the HTTP connection limit to ensure
            # that the worker will not receive any more credits while the connection limit is reached.
            pull_client_max_concurrency=AIPERF_HTTP_CONNECTION_LIMIT,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.health_check_interval = DEFAULT_WORKER_HEALTH_CHECK_INTERVAL

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_return_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.CREDIT_RETURN,
            )
        )
        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
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
        self.extractor: ResponseExtractorProtocol = (
            ResponseExtractorFactory.create_instance(
                self.model_endpoint.endpoint.type,
                model_endpoint=self.model_endpoint,
            )
        )

    @on_pull_message(MessageType.CREDIT_DROP)
    async def _credit_drop_callback(self, message: CreditDropMessage) -> None:
        """Handle an incoming credit drop message from the timing manager. Every credit must be returned after processing."""

        try:
            # NOTE: This must be awaited to ensure that the max concurrency is respected
            await self._process_credit_drop_internal(message)
        except Exception as e:
            self.error(f"Error processing credit drop: {e!r}")
            await self.credit_return_push_client.push(
                CreditReturnMessage(
                    service_id=self.service_id,
                    phase=message.phase,
                )
            )

    @on_stop
    async def _shutdown_worker(self) -> None:
        self.debug("Shutting down worker")
        if self.inference_client:
            await self.inference_client.close()

    @background_task(
        immediate=False,
        interval=lambda self: self.health_check_interval,
    )
    async def _health_check_task(self) -> None:
        """Task to report the health of the worker to the worker manager."""
        await self.publish(self.create_health_message())

    def create_health_message(self) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=self.get_process_health(),
            task_stats=self.task_stats,
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        self.debug(lambda: f"Received profile cancel command: {message}")
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )
        await self.stop()

    async def _process_credit_drop_internal(self, message: CreditDropMessage) -> None:
        """Process a credit drop message. Make sure to return the credit as soon as possible.

        - Every credit must be returned after processing
        - All results or errors should be converted to a RequestRecord and pushed to the inference results client.

        NOTE: This function MUST NOT return until the credit drop is fully processed.
        This is to ensure that the max concurrency is respected via the semaphore of the pull client.
        The way this is enforced is by requiring that this method returns a CreditReturnMessage.
        """
        if self.is_trace_enabled:
            self.trace(f"Processing credit drop: {message}")

        self.task_stats.total += 1

        try:
            await self._execute_single_credit_internal(message)
        finally:
            # Need to return the credit here to ensure it is always returned
            return_message = CreditReturnMessage(
                service_id=self.service_id,
                phase=message.phase,
                delayed_ns=None,  # TODO: set this properly (from record if available?)
            )
            if self.is_trace_enabled:
                self.trace(f"Returning credit {return_message}")
            # NOTE: Do not do this execute_async, as we want to give the credit back as soon as possible.
            await self.credit_return_push_client.push(return_message)

    async def _execute_single_credit_internal(self, message: CreditDropMessage) -> None:
        """Run a credit task for a single credit."""
        drop_perf_ns = time.perf_counter_ns()  # The time the credit was received

        if not self.inference_client:
            raise NotInitializedError("Inference server client not initialized.")

        # retrieve the prompt from the dataset
        conversation_response: ConversationResponseMessage = (
            await self.conversation_request_client.request(
                ConversationRequestMessage(
                    service_id=self.service_id,
                    conversation_id=message.conversation_id,
                    credit_phase=message.phase,
                )
            )
        )
        if self.is_trace_enabled:
            self.trace(f"Received response message: {conversation_response}")

        # Check for error in conversation response
        if isinstance(conversation_response, ErrorMessage):
            await self._send_inference_result_message(
                RequestRecord(
                    model_name=self.model_endpoint.primary_model_name,
                    conversation_id=message.conversation_id,
                    turn_index=0,
                    turn=None,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=conversation_response.error,
                )
            )
            return
        await self._process_turns(conversation_response, message, drop_perf_ns)

    async def _process_turns(
        self,
        conversation_response: ConversationResponseMessage,
        message: CreditDropMessage,
        drop_perf_ns: int,
    ) -> None:
        """Process each turn in the conversation sequentially."""
        turn_list = []
        for turn_index in range(len(conversation_response.conversation.turns)):
            turn = conversation_response.conversation.turns[turn_index]
            turn_list.append(turn)
            # TODO: how do we handle errors in the middle of a conversation?
            record = await self._call_inference_api_internal(message, turn)
            record.model_name = turn.model or self.model_endpoint.primary_model_name
            record.conversation_id = conversation_response.conversation.session_id
            record.turn_index = turn_index
            record.credit_phase = message.phase
            record.cancel_after_ns = message.cancel_after_ns
            # If this is the first turn, calculate the credit drop latency
            if turn_index == 0:
                record.credit_drop_latency = record.start_perf_ns - drop_perf_ns
            await self._send_inference_result_message(record)
            resp = await self.extractor.extract_response_data(record)
            # TODO how do we handle reasoning responses in multi turn?
            resp_text = "".join([r.data.get_text() for r in resp])
            if resp_text:
                resp_turn = Turn(
                    role="assistant",
                    texts=[Text(contents=[resp_text])],
                )
                turn_list.append(resp_turn)

    async def _call_inference_api_internal(
        self,
        message: CreditDropMessage,
        turn: Turn,
    ) -> RequestRecord:
        """Make a single call to the inference API. Will return an error record if the call fails."""
        if self.is_trace_enabled:
            self.trace(f"Calling inference API for turn: {turn}")
        formatted_payload = None
        pre_send_perf_ns = None
        timestamp_ns = None
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
                if self.is_trace_enabled:
                    self.trace(
                        f"Waiting for credit drop expected time: {(drop_ns - now_ns) / NANOS_PER_SECOND:.2f} s"
                    )
                await asyncio.sleep((drop_ns - now_ns) / NANOS_PER_SECOND)
            elif drop_ns and drop_ns < now_ns:
                delayed_ns = now_ns - drop_ns

            # Save the current perf_ns before sending the request so it can be used to calculate
            # the start_perf_ns of the request in case of an exception.
            pre_send_perf_ns = time.perf_counter_ns()
            timestamp_ns = time.time_ns()

            send_coroutine = self.inference_client.send_request(
                model_endpoint=self.model_endpoint,
                payload=formatted_payload,
            )

            maybe_result: RequestRecord | None = await self._send_with_optional_cancel(
                send_coroutine=send_coroutine,
                should_cancel=message.should_cancel,
                cancel_after_ns=message.cancel_after_ns,
            )

            if maybe_result is not None:
                result = maybe_result
                if self.is_debug_enabled:
                    self.debug(
                        f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
                    )
                result.delayed_ns = delayed_ns
                result.turn = turn
                return result
            else:
                cancellation_perf_ns = time.perf_counter_ns()
                if self.is_debug_enabled:
                    delay_s = message.cancel_after_ns / NANOS_PER_SECOND
                    self.debug(f"Request cancelled after {delay_s:.3f}s")

                return RequestRecord(
                    turn=turn,
                    timestamp_ns=timestamp_ns,
                    start_perf_ns=pre_send_perf_ns,
                    end_perf_ns=cancellation_perf_ns,
                    was_cancelled=True,
                    cancellation_perf_ns=cancellation_perf_ns,
                    delayed_ns=delayed_ns,
                    error=ErrorDetails(
                        type="RequestCancellationError",
                        message=(
                            f"Request was cancelled after "
                            f"{message.cancel_after_ns / NANOS_PER_SECOND:.3f} seconds"
                        ),
                        code=499,  # Client Closed Request
                    ),
                )
        except asyncio.CancelledError:
            # If a task is cancelled (e.g. during shutdown), propagate the cancellation.
            raise
        except Exception as e:
            self.error(
                f"Error calling inference server API at {self.model_endpoint.url}: {e!r}"
            )
            return RequestRecord(
                turn=turn,
                timestamp_ns=timestamp_ns or time.time_ns(),
                # Try and use the pre_send_perf_ns if it is available, otherwise use the current time.
                start_perf_ns=pre_send_perf_ns or time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )

    async def _send_with_optional_cancel(
        self,
        *,
        send_coroutine: Awaitable[RequestRecord],
        should_cancel: bool,
        cancel_after_ns: int,
    ) -> RequestRecord | None:
        """Send a coroutine with optional cancellation after a delay.
        Args:
            send_coroutine: The coroutine object to send.
            should_cancel: Whether to enable cancellation.
            cancel_after_ns: The delay in nanoseconds after which to cancel the request.
        Returns:
            The result of the send_coroutine, or None if it was cancelled.
        """
        if not should_cancel:
            return await send_coroutine

        timeout_s = cancel_after_ns / NANOS_PER_SECOND
        try:
            return await asyncio.wait_for(send_coroutine, timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send the inference result message to the inference results push client."""
        # All records will flow through here to be sent to the inference results push client.
        self.task_stats.task_finished(record.valid)

        msg = InferenceResultsMessage(
            service_id=self.service_id,
            record=record,
        )
        self.execute_async(self.inference_results_push_client.push(msg))


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(Worker)


if __name__ == "__main__":
    main()
