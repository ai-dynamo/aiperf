# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    CreditDropMessage,
    CreditReturnMessage,
    ErrorMessage,
    InferenceResultsMessage,
)
from aiperf.common.models import ErrorDetails, RequestRecord, Turn, WorkerTaskStats
from aiperf.common.protocols import (
    InferenceClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
    RequestConverterProtocol,
    TaskManagerProtocol,
)


@runtime_checkable
class CreditProcessorProtocol(Protocol):
    """CreditProcessorProtocol is a protocol that provides a method to process credit drops."""

    async def _process_credit_drop_internal(
        self, message: CreditDropMessage
    ) -> CreditReturnMessage:
        """Process a credit drop message. Return the CreditReturnMessage."""
        ...


@runtime_checkable
class CreditProcessorMixinRequirements(TaskManagerProtocol, Protocol):
    """CreditProcessorMixinRequirements is a protocol that provides the requirements needed for the CreditProcessorMixin."""

    service_id: str
    inference_client: InferenceClientProtocol
    conversation_request_client: RequestClientProtocol
    inference_results_push_client: PushClientProtocol
    credit_return_push_client: PushClientProtocol
    request_converter: RequestConverterProtocol
    model_endpoint: ModelEndpointInfo
    task_stats: WorkerTaskStats

    async def _process_credit_drop_internal(self, message: CreditDropMessage) -> None:
        """Process a credit drop message. Return the credit as soon as possible."""
        ...

    async def _execute_single_credit_internal(
        self, message: CreditDropMessage
    ) -> RequestRecord:
        """Execute a single credit drop. Return the RequestRecord."""
        ...

    async def _call_inference_api_internal(
        self,
        message: CreditDropMessage,
        turn: Turn,
    ) -> RequestRecord:
        """Make a single call to the inference API. Will return an error record if the call fails."""
        ...


class CreditProcessorMixin(CreditProcessorMixinRequirements):
    """CreditProcessorMixin is a mixin that provides a method to process credit drops."""

    def __init__(self, **kwargs):
        if not isinstance(self, CreditProcessorMixinRequirements):
            raise ValueError(
                "CreditProcessorMixin must be used with CreditProcessorMixinRequirements"
            )

    async def _process_credit_drop_internal(self, message: CreditDropMessage) -> None:
        """Process a credit drop message. Make sure to return the credit as soon as possible.

        - Every credit must be returned after processing
        - All results or errors should be converted to a RequestRecord and pushed to the inference results client.

        NOTE: This function MUST NOT return until the credit drop is fully processed.
        This is to ensure that the max concurrency is respected via the semaphore of the pull client.
        The way this is enforced is by requiring that this method returns a CreditReturnMessage.
        """
        # TODO: Add tests to ensure that the above note is never violated in the future

        if self.is_trace_enabled:
            self.trace(f"Processing credit drop: {message}")
        drop_perf_ns = time.perf_counter_ns()  # The time the credit was received

        self.task_stats.total += 1

        record: RequestRecord = RequestRecord()
        try:
            record = await self._execute_single_credit_internal(message)

        except Exception as e:
            self.exception(f"Error processing credit drop: {e}")
            record.error = ErrorDetails.from_exception(e)
            record.end_perf_ns = time.perf_counter_ns()

        finally:
            return_message = CreditReturnMessage(
                service_id=self.service_id,
                phase=message.phase,
                delayed_ns=record.delayed_ns,
            )
            if self.is_trace_enabled:
                self.trace(f"Returning credit {return_message}")
            # NOTE: Do not do this execute_async, as we want to give the credit back as soon as possible.
            await self.credit_return_push_client.push(return_message)

            self.task_stats.task_finished(record.valid)

            record.credit_phase = message.phase
            # Calculate the latency of the credit drop (from when the credit drop was first received to when the request was sent)
            record.credit_drop_latency = record.start_perf_ns - drop_perf_ns

            record.cancel_after_ns = message.cancel_after_ns

            msg = InferenceResultsMessage(
                service_id=self.service_id,
                record=record,
            )
            self.execute_async(self.inference_results_push_client.push(msg))

    async def _execute_single_credit_internal(
        self, message: CreditDropMessage
    ) -> RequestRecord:
        """Run a credit task for a single credit."""

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

        turn_index = 0
        turn = conversation_response.conversation.turns[turn_index]

        if isinstance(conversation_response, ErrorMessage):
            return RequestRecord(
                model_name=turn.model or self.model_endpoint.primary_model_name,
                conversation_id=message.conversation_id,
                turn_index=turn_index,
                turn=turn,
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=conversation_response.error,
            )

        record = await self._call_inference_api_internal(message, turn)
        record.model_name = turn.model or self.model_endpoint.primary_model_name
        record.conversation_id = conversation_response.conversation.session_id
        record.turn_index = turn_index
        return record

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
                cancellation_time_ns = time.perf_counter_ns()
                if self.is_debug_enabled:
                    delay_s = message.cancel_after_ns / NANOS_PER_SECOND
                    self.debug(f"Request cancelled after {delay_s:.3f}s")

                return RequestRecord(
                    turn=turn,
                    timestamp_ns=timestamp_ns,
                    start_perf_ns=pre_send_perf_ns,
                    end_perf_ns=cancellation_time_ns,
                    was_cancelled=True,
                    cancellation_time_ns=cancellation_time_ns,
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
