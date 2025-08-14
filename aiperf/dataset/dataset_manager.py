# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import random
import time

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    CustomDatasetType,
    DatasetType,
    MessageType,
    ServiceType,
)
from aiperf.common.factories import (
    CustomDatasetFactory,
    ServiceFactory,
)
from aiperf.common.hooks import on_command, on_pull_message, on_request
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    DatasetTimingRequest,
    DatasetTimingResponse,
    ProcessDatasetMessage,
    ProcessDatasetResponseMessage,
    ProcessMooncakeTraceDatasetMessage,
    ProcessMultiTurnDatasetMessage,
    ProcessRandomPoolDatasetMessage,
    ProcessSingleTurnDatasetMessage,
    ProcessSyntheticDatasetMessage,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import PullClientMixin, ReplyClientMixin
from aiperf.common.models import Conversation
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.tokenizer import Tokenizer

DATASET_CONFIGURATION_TIMEOUT = 300.0


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.DATASET_MANAGER)
class DatasetManager(ReplyClientMixin, PullClientMixin, BaseComponentService):
    """
    The DatasetManager primary responsibility is to manage the data generation or acquisition.
    For synthetic generation, it contains the code to generate the prompts or tokens.
    It will have an API for dataset acquisition of a dataset if available in a remote repository or database.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            reply_client_address=CommAddress.DATASET_MANAGER_PROXY_BACKEND,
            reply_client_bind=False,
            pull_client_address=CommAddress.DATASET_RESULT,
            pull_client_bind=True,
        )
        self.debug("Dataset manager __init__")
        self.user_config = user_config
        self.tokenizer: Tokenizer | None = None
        self.dataset: dict[str, Conversation] = {}  # session ID -> Conversation mapping
        self._session_ids_cache: list[str] = []
        self._conversation_query_random = random.Random(
            self.user_config.input.random_seed
        )
        self.dataset_configured = asyncio.Event()

        self.jobs_push_client = self.comms.create_push_client(
            address=CommAddress.DATASET_JOB,
            bind=True,
        )
        self.num_processors = self.service_config.dataset_processor_service_count
        self._custom_dataset_type = self.user_config.input.custom_dataset_type
        self.total_dataset_size: int | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""
        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()

        if self.user_config.input.dataset_type == DatasetType.SYNTHETIC:
            await self._configure_synthetic_dataset()
        elif self.user_config.input.dataset_type == DatasetType.CUSTOM:
            await self._configure_custom_dataset()
        else:
            raise NotImplementedError(
                f"Dataset type {self.user_config.input.dataset_type} is not supported yet."
            )

        await self._wait_for_dataset_configuration()
        self._session_ids_cache = list(self.dataset.keys())
        duration = time.perf_counter() - begin
        self.info(
            lambda: f"Dataset configured in {duration:.2f} seconds with {len(self.dataset)} conversations"
        )

    async def _configure_synthetic_dataset(self) -> None:
        """Configure a synthetic dataset.

        This will generate a synthetic dataset and distribute the work across the
        dataset processors.
        """
        self.total_dataset_size = self.user_config.input.conversation.num
        self.info(
            f"Distributing {self.total_dataset_size} conversations "
            f"across {self.num_processors} processors "
        )

        remaining_dataset_size = self.total_dataset_size
        chunk_size = max(self.total_dataset_size // self.num_processors, 1)
        while remaining_dataset_size > 0:
            await self.jobs_push_client.push(
                message=ProcessSyntheticDatasetMessage(
                    service_id=self.service_id,
                    num_conversations=chunk_size,
                )
            )

            remaining_dataset_size -= chunk_size
            if remaining_dataset_size < chunk_size:
                chunk_size = remaining_dataset_size

    async def _configure_custom_dataset(self) -> None:
        """Configure a custom dataset.

        This will load a custom dataset from a file and distribute the work across the
        dataset processors.
        """
        process_messages: dict[CustomDatasetType, type[ProcessDatasetMessage]] = {
            CustomDatasetType.MOONCAKE_TRACE: ProcessMooncakeTraceDatasetMessage,
            CustomDatasetType.MULTI_TURN: ProcessMultiTurnDatasetMessage,
            CustomDatasetType.SINGLE_TURN: ProcessSingleTurnDatasetMessage,
            CustomDatasetType.RANDOM_POOL: ProcessRandomPoolDatasetMessage,
        }
        process_message = process_messages[self._custom_dataset_type]

        custom_kwargs = {}
        if self._custom_dataset_type == CustomDatasetType.RANDOM_POOL:
            custom_kwargs["num_conversations"] = self.user_config.input.conversation.num

        loader = CustomDatasetFactory.create_instance(
            self._custom_dataset_type,
            user_config=self.user_config,
        )
        dataset_list = list(loader.load_dataset().items())
        self.total_dataset_size = len(dataset_list)

        self.info(
            f"Distributing {self.total_dataset_size} dataset items "
            f"across {self.num_processors} processors"
        )

        remaining_dataset_size = self.total_dataset_size
        chunk_size = max(self.total_dataset_size // self.num_processors, 1)
        start = 0
        while remaining_dataset_size > 0:
            await self.jobs_push_client.push(
                message=process_message(
                    service_id=self.service_id,
                    dataset=dataset_list[start : start + chunk_size],
                    **custom_kwargs,
                )
            )
            start += chunk_size
            remaining_dataset_size -= chunk_size
            if remaining_dataset_size < chunk_size:
                chunk_size = remaining_dataset_size

    @on_pull_message(MessageType.DATASET_RESULT)
    async def _on_result(self, message: ProcessDatasetResponseMessage) -> None:
        """Handle a dataset job result."""
        self.debug(
            lambda: f"Received processed dataset response from {message.service_id}"
        )

        for conversation in message.generated_data:
            self.dataset[conversation.session_id] = conversation

        if len(self.dataset) == self.total_dataset_size:
            self.debug(
                lambda: f"Dataset configured with {len(self.dataset)} conversations"
            )
            self.dataset_configured.set()
            await self.publish(
                DatasetConfiguredNotification(
                    service_id=self.service_id,
                ),
            )

    @on_request(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        self.debug(lambda: f"Handling conversation request: {message}")

        await self._wait_for_dataset_configuration()

        if not self.dataset:
            raise self._service_error(
                "Dataset is empty and must be configured before handling requests.",
            )

        if message.conversation_id is None:
            return self._return_any_conversation(
                request_id=message.request_id,
            )
        else:
            return self._return_conversation_by_id(
                request_id=message.request_id,
                conversation_id=message.conversation_id,
            )

    def _return_any_conversation(
        self, request_id: str | None
    ) -> ConversationResponseMessage:
        """Return any conversation from the dataset based on the user specified method."""

        # TODO: Implement the user specified method (random, round robin, etc.)
        session_id = self._conversation_query_random.choice(self._session_ids_cache)
        conversation = self.dataset[session_id]
        self.trace_or_debug(
            lambda: f"Sending random conversation response: {conversation}",
            lambda: f"Sending random conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    def _return_conversation_by_id(
        self, request_id: str | None, conversation_id: str
    ) -> ConversationResponseMessage:
        """Return a conversation if it exists, otherwise raise an error."""

        if conversation_id not in self.dataset:
            raise self._service_error(
                f"Conversation {conversation_id} not found in dataset.",
            )

        conversation = self.dataset[conversation_id]
        self.trace_or_debug(
            lambda: f"Sending conversation response: {conversation}",
            lambda: f"Sending conversation response with id: {conversation.session_id}",
        )
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    @on_request(MessageType.CONVERSATION_TURN_REQUEST)
    async def _handle_conversation_turn_request(
        self, message: ConversationTurnRequestMessage
    ) -> ConversationTurnResponseMessage:
        """Handle a turn request."""
        self.debug(lambda: f"Handling turn request: {message}")

        if message.conversation_id not in self.dataset:
            raise self._service_error(
                f"Conversation {message.conversation_id} not found in dataset.",
            )

        conversation = self.dataset[message.conversation_id]
        if message.turn_index >= len(conversation.turns):
            raise self._service_error(
                f"Turn index {message.turn_index} is out of range for conversation {message.conversation_id}.",
            )

        turn = conversation.turns[message.turn_index]

        self.trace_or_debug(
            lambda: f"Sending turn response: {turn}",
            "Sending turn response",
        )
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=turn,
        )

    @on_request(MessageType.DATASET_TIMING_REQUEST)
    async def _handle_dataset_timing_request(
        self, message: DatasetTimingRequest
    ) -> DatasetTimingResponse:
        """Handle a dataset timing request."""
        self.trace_or_debug(
            lambda: f"Handling dataset timing request: {message}",
            "Handling dataset timing request",
        )

        await self._wait_for_dataset_configuration()

        if not self.dataset:
            raise self._service_error(
                "Dataset is empty and must be configured before handling timing requests.",
            )

        timing_dataset = []
        for conversation_id, conversation in self.dataset.items():
            for turn in conversation.turns:
                timing_dataset.append((turn.timestamp, conversation_id))

        return DatasetTimingResponse(
            service_id=self.service_id,
            request_id=message.request_id,
            timing_data=timing_dataset,
        )

    async def _wait_for_dataset_configuration(self) -> None:
        """Wait for the dataset to be configured if it is not already."""
        if not self.dataset_configured.is_set():
            self.debug(
                "Dataset not configured. Waiting for dataset to be configured..."
            )
            await asyncio.wait_for(
                self.dataset_configured.wait(), timeout=DATASET_CONFIGURATION_TIMEOUT
            )


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    main()
