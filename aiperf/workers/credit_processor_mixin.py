# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Protocol, runtime_checkable

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    CreditDropMessage,
    CreditReturnMessage,
    ErrorMessage,
    InferenceResultsMessage,
)
from aiperf.common.models import ErrorDetails, RequestRecord, Turn, WorkerPhaseTaskStats
from aiperf.common.protocols import (
    AIPerfLoggerProtocol,
    InferenceClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
    RequestConverterProtocol,
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
class CreditProcessorMixinRequirements(AIPerfLoggerProtocol, Protocol):
    """CreditProcessorMixinRequirements is a protocol that provides the requirements needed for the CreditProcessorMixin."""

    service_id: str
    inference_client: InferenceClientProtocol
    conversation_request_client: RequestClientProtocol
    inference_results_push_client: PushClientProtocol
    request_converter: RequestConverterProtocol
    model_endpoint: ModelEndpointInfo
    task_stats: dict[CreditPhase, WorkerPhaseTaskStats]

    async def _process_credit_drop_internal(
        self, message: CreditDropMessage
    ) -> CreditReturnMessage:
        """Process a credit drop message. Return the CreditReturnMessage."""
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

    async def _process_credit_drop_internal(
        self, message: CreditDropMessage
    ) -> CreditReturnMessage:
        """Process a credit drop message. Return the CreditReturnMessage.

        - Every credit must be returned after processing
        - All results or errors should be converted to a RequestRecord and pushed to the inference results client.

        NOTE: This function MUST NOT return until the credit drop is fully processed.
        This is to ensure that the max concurrency is respected via the semaphore of the pull client.
        The way this is enforced is by requiring that this method returns a CreditReturnMessage.
        """
        # TODO: Add tests to ensure that the above note is never violated in the future

        self.trace(lambda: f"Processing credit drop: {message}")
        drop_perf_ns = time.perf_counter_ns()  # The time the credit was received

        if message.phase not in self.task_stats:
            self.task_stats[message.phase] = WorkerPhaseTaskStats()
        self.task_stats[message.phase].total += 1

        record: RequestRecord = RequestRecord()
        try:
            record = await self._execute_single_credit_internal(message)

        except Exception as e:
            self.exception(f"Error processing credit drop: {e}")
            record.error = ErrorDetails.from_exception(e)
            record.end_perf_ns = time.perf_counter_ns()

        finally:
            record.credit_phase = message.phase
            # Calculate the latency of the credit drop (from when the credit drop was first received to when the request was sent)
            record.credit_drop_latency = record.start_perf_ns - drop_perf_ns

            msg = InferenceResultsMessage(
                service_id=self.service_id,
                record=record,
            )

            # Note that we already ensured that the phase exists in the task_stats dict in the above code.
            if not record.valid:
                self.task_stats[message.phase].failed += 1
            else:
                self.task_stats[message.phase].completed += 1

            try:
                await self.inference_results_push_client.push(msg)
            except Exception as e:
                # If we fail to push the record, log the error and continue
                self.exception(f"Error pushing request record: {e}")
            finally:
                # Always return the credits
                return_message = CreditReturnMessage(
                    service_id=self.service_id,
                    delayed_ns=record.delayed_ns,
                    phase=message.phase,
                )
                self.trace(lambda: f"Returning credit {return_message}")
                return return_message  # noqa: B012

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
        self.trace(lambda: f"Received response message: {conversation_response}")

        if isinstance(conversation_response, ErrorMessage):
            return RequestRecord(
                model_name=self.model_endpoint.primary_model_name,
                conversation_id=message.conversation_id,
                turn_index=0,
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=conversation_response.error,
            )

        record = await self._call_inference_api_internal(
            message, conversation_response.conversation.turns[0]
        )
        record.model_name = self.model_endpoint.primary_model_name
        record.conversation_id = conversation_response.conversation.session_id
        record.turn_index = 0
        return record

    async def _call_inference_api_internal(
        self,
        message: CreditDropMessage,
        turn: Turn,
    ) -> RequestRecord:
        """Make a single call to the inference API. Will return an error record if the call fails."""
        self.trace(lambda: f"Calling inference API for turn: {turn}")
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
                self.trace(
                    lambda: f"Waiting for credit drop expected time: {(drop_ns - now_ns) / NANOS_PER_SECOND:.2f} s"
                )
                await asyncio.sleep((drop_ns - now_ns) / NANOS_PER_SECOND)
            elif drop_ns and drop_ns < now_ns:
                delayed_ns = now_ns - drop_ns

            # Save the current perf_ns before sending the request so it can be used to calculate
            # the start_perf_ns of the request in case of an exception.
            pre_send_perf_ns = time.perf_counter_ns()
            timestamp_ns = time.time_ns()

            # Send the request to the Inference Server API and wait for the response
            result: RequestRecord = await self.inference_client.send_request(
                model_endpoint=self.model_endpoint,
                payload=formatted_payload,
            )

            self.debug(
                lambda: f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
            )

            result.delayed_ns = delayed_ns
            return result

        except Exception as e:
            self.exception(
                f"Error calling inference server API at {self.model_endpoint.url}: {e}"
            )
            return RequestRecord(
                request=formatted_payload,
                timestamp_ns=timestamp_ns or time.time_ns(),
                # Try and use the pre_send_perf_ns if it is available, otherwise use the current time.
                start_perf_ns=pre_send_perf_ns or time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
