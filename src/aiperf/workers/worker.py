# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import time
import uuid
from collections.abc import Awaitable

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import MILLIS_PER_SECOND, NANOS_PER_SECOND
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CreditPhase,
    MessageType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import background_task, on_command, on_pull_message
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
    Conversation,
    ErrorDetails,
    ReasoningResponseData,
    RequestRecord,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import PushClientProtocol, RequestClientProtocol
from aiperf.workers.inference_client import InferenceClient


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
            pull_client_max_concurrency=Environment.HTTP.CONNECTION_LIMIT,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.health_check_interval = Environment.WORKER.HEALTH_CHECK_INTERVAL

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

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint
        )
        self.debug(
            lambda: f"Creating inference client for {self.model_endpoint.endpoint.type}, "
            f"class: {self.inference_client.__class__.__name__}",
        )
        self.attach_child_lifecycle(self.inference_client)

    @on_pull_message(MessageType.CREDIT_DROP)
    async def _credit_drop_callback(self, message: CreditDropMessage) -> None:
        """Handle an incoming credit drop message from the timing manager. Every credit must be returned after processing."""

        try:
            # NOTE: This must be awaited to ensure that the max concurrency is respected
            await self._process_credit_drop_internal(message)
        except Exception as e:
            self.error(f"Error processing credit drop: {e!r}")

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
        return_message = CreditReturnMessage(
            service_id=self.service_id,
            phase=message.phase,
            credit_drop_id=message.request_id,
            delayed_ns=None,  # TODO: set this properly (from record if available?)
            requests_sent=0,
        )

        try:
            if self.is_trace_enabled:
                self.trace(f"Processing credit drop: {message}")

            await self._execute_single_credit_internal(message, return_message)
        finally:
            # Need to return the credit here to ensure it is always returned
            if self.is_trace_enabled:
                self.trace(f"Returning credit {return_message}")
            # NOTE: Do not do this execute_async, as we want to give the credit back as soon as possible.
            await self.credit_return_push_client.push(return_message)

    async def _execute_single_credit_internal(
        self, message: CreditDropMessage, return_message: CreditReturnMessage
    ) -> None:
        """Run a credit task for a single credit.

        For multi-turn conversations, this method simulates realistic user interaction
        by applying turn delays between subsequent turns. The flow follows real-world
        conversation behavior:

        1. Turn 0 (first turn): User sends initial message → AI responds (no delay)
        2. DELAY: User reads AI's response and thinks about next message
        3. Turn 1 (second turn): User sends follow-up message → AI responds
        4. DELAY: User reads AI's response and thinks about next message
        5. Turn 2 (third turn): User sends next message → AI responds
        ... and so on

        Turn delays are configured via:
        - --conversation-turn-delay-mean: Average delay between turns (milliseconds)
        - --conversation-turn-delay-stddev: Standard deviation of delay (milliseconds)
        - --conversation-turn-delay-ratio: Ratio to scale delays
        """
        drop_perf_ns = time.perf_counter_ns()  # The time the credit was received

        if not self.inference_client:
            raise NotInitializedError("Inference server client not initialized.")

        conversation = await self._retrieve_conversation_response(
            service_id=self.service_id,
            conversation_id=message.conversation_id,
            phase=message.phase,
        )

        turn_list = []
        for turn_index in range(len(conversation.turns)):
            # Apply turn delay BEFORE sending the turn (simulating user thinking time)
            # Skip delay for the first turn
            turn = conversation.turns[turn_index]
            if turn_index > 0 and turn.delay is not None and turn.delay > 0:
                delay_seconds = (
                    turn.delay / MILLIS_PER_SECOND
                )  # Convert milliseconds to seconds
                if self.is_trace_enabled:
                    self.trace(
                        f"Applying turn delay of {turn.delay}ms before sending turn {turn_index}"
                    )
                await asyncio.sleep(delay_seconds)

            self.task_stats.total += 1
            turn_list.append(turn)

            request_info = RequestInfo(
                model_endpoint=self.model_endpoint,
                credit_num=message.credit_num,
                credit_phase=message.phase,
                should_cancel=message.should_cancel,
                cancel_after_ns=message.cancel_after_ns,
                x_request_id=str(uuid.uuid4()),
                x_correlation_id=message.request_id,  # CreditDropMessage request_id is the X-Correlation-ID header
                conversation_id=message.conversation_id,
                turn_index=turn_index,
                turns=turn_list,
                system_message=conversation.system_message,
                user_context_message=conversation.user_context_message,
            )

            return_message.requests_sent += 1
            record = await self._build_response_record(
                request_info=request_info,
                drop_perf_ns=drop_perf_ns,
            )
            await self._send_inference_result_message(record)

            if resp_turn := await self._process_response(record):
                turn_list.append(resp_turn)

    async def _retrieve_conversation_response(
        self,
        *,
        service_id: str,
        conversation_id: str | None,
        phase: CreditPhase,
    ) -> Conversation:
        """Retrieve the conversation from the dataset manager. If a conversation
        cannot be retrieved, an error message will be sent to the
        inference results client and an Exception is raised.
        """
        # retrieve the prompt from the dataset
        conversation_response: ConversationResponseMessage = (
            await self.conversation_request_client.request(
                ConversationRequestMessage(
                    service_id=service_id,
                    conversation_id=conversation_id,
                    credit_phase=phase,
                )
            )
        )
        if self.is_trace_enabled:
            self.trace(f"Received response message: {conversation_response}")

        # Check for error in conversation response
        if isinstance(conversation_response, ErrorMessage):
            await self._send_inference_result_message(
                RequestRecord(
                    request_headers=None,
                    model_name=self.model_endpoint.primary_model_name,
                    conversation_id=conversation_id,
                    turn_index=0,
                    turns=None,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=conversation_response.error,
                )
            )
            raise ValueError("Failed to retrieve conversation response")

        return conversation_response.conversation

    async def _build_response_record(
        self,
        *,
        request_info: RequestInfo,
        drop_perf_ns: int,
    ) -> RequestRecord:
        """Build a RequestRecord from an inference API call for the given turn."""
        record = await self._call_inference_api_internal(request_info, drop_perf_ns)
        record.model_name = (
            request_info.turns[request_info.turn_index].model
            or self.model_endpoint.primary_model_name
        )
        record.conversation_id = request_info.conversation_id
        record.turn_index = request_info.turn_index
        record.credit_phase = request_info.credit_phase
        record.cancel_after_ns = request_info.cancel_after_ns
        record.x_request_id = request_info.x_request_id
        record.x_correlation_id = request_info.x_correlation_id
        record.credit_num = request_info.credit_num
        # If this is the first turn, calculate the credit drop latency
        if request_info.turn_index == 0:
            record.credit_drop_latency = record.start_perf_ns - drop_perf_ns
        # Preserve headers set by transport; only use endpoint headers if not set
        if record.request_headers is None:
            record.request_headers = request_info.endpoint_headers
        return record

    async def _process_response(self, record: RequestRecord) -> Turn | None:
        """Process the response from the inference API call and convert it to a Turn object."""
        resp = self.inference_client.endpoint.extract_response_data(record)
        # Skip reasoning responses in multi-turn conversations
        output_texts = []
        for response in resp:
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())
        resp_text = "".join(output_texts)

        return (
            Turn(role="assistant", texts=[Text(contents=[resp_text])])
            if resp_text
            else None
        )

    async def _call_inference_api_internal(
        self,
        request_info: RequestInfo,  # NOTE: RequestInfo is used to pass the request info to the inference client
        credit_drop_ns: int,
    ) -> RequestRecord:
        """Make a single call to the inference API. Will return an error record if the call fails."""
        if self.is_trace_enabled:
            self.trace(
                f"Calling inference API for turn: {request_info.turns[request_info.turn_index]}"
            )
        pre_send_perf_ns = None
        timestamp_ns = None
        try:
            # NOTE: Current implementation of the TimingManager bypasses this, it is for future use.
            # Wait for the credit drop time if it is in the future.
            # Note that we check this after we have retrieved the data from the dataset, to ensure
            # that we are fully ready to go.
            delayed_ns = None
            drop_ns = credit_drop_ns
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
                request_info=request_info,
            )

            maybe_result: RequestRecord | None = await self._send_with_optional_cancel(
                send_coroutine=send_coroutine,
                should_cancel=request_info.should_cancel,
                cancel_after_ns=request_info.cancel_after_ns,
            )

            if maybe_result is not None:
                result = maybe_result
                if self.is_debug_enabled:
                    self.debug(
                        f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
                    )
                result.delayed_ns = delayed_ns
                result.turns = request_info.turns
                return result
            else:
                cancellation_perf_ns = time.perf_counter_ns()
                if self.is_debug_enabled:
                    delay_s = request_info.cancel_after_ns / NANOS_PER_SECOND
                    self.debug(f"Request cancelled after {delay_s:.3f}s")
                # TODO what do i do with the turn here?
                return RequestRecord(
                    # TODO: This should be handled by the transport, but we need to handle it here for now.
                    request_headers=request_info.endpoint_headers,
                    turns=request_info.turns,
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
                            f"{request_info.cancel_after_ns / NANOS_PER_SECOND:.3f} seconds"
                        ),
                        code=499,  # Client Closed Request
                    ),
                )
        except Exception as e:
            self.error(
                f"Error calling inference server API at {self.model_endpoint.endpoint.base_url}: {e!r}"
            )
            return RequestRecord(
                request_headers=request_info.endpoint_headers,
                turns=request_info.turns,
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
