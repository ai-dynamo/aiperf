# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import random
import time

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ComposerType,
    MessageType,
    ServiceType,
)
from aiperf.common.enums.dataset_enums import CustomDatasetType
from aiperf.common.factories import ComposerFactory, ServiceFactory
from aiperf.common.hooks import on_command, on_init, on_request
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    DatasetTimingRequest,
    DatasetTimingResponse,
    ProfileConfigureCommand,
)
from aiperf.common.mixins import ReplyClientMixin
from aiperf.common.models import Conversation
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader import ShareGPTLoader

DATASET_CONFIGURATION_TIMEOUT = 300.0
_logger = AIPerfLogger(__name__)


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.DATASET_MANAGER)
class DatasetManager(ReplyClientMixin, BaseComponentService):
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
        self._sequential_iterator_index = 0
        self._use_sequential_iteration = False

    @on_init
    async def _initialize(self) -> None:
        """Initialize the dataset manager."""
        self.debug("Initializing dataset manager")
        tokenizer_name = self.user_config.tokenizer.name
        if tokenizer_name is None:
            # TODO: What do we do if there are multiple models?
            # How will we know which tokenizer to use?
            tokenizer_name = self.user_config.endpoint.model_names[0]

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the dataset."""
        self.info(lambda: f"Configuring dataset for {self.service_id}")
        begin = time.perf_counter()
        await self._configure_dataset()
        duration = time.perf_counter() - begin
        self.info(lambda: f"Dataset configured in {duration:.2f} seconds")

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        self.dataset_configured.clear()

        # Temporary as this will change with the following dataset processor service PR
        if self.user_config.input.public_dataset is not None:
            loader = ShareGPTLoader(self.user_config, self.tokenizer)
            dataset = await loader.load_dataset()
            conversations = await loader.convert_to_conversations(dataset)
        elif self.user_config.input.custom_dataset_type is not None:
            composer = ComposerFactory.create_instance(
                ComposerType.CUSTOM,
                config=self.user_config,
                tokenizer=self.tokenizer,
            )
            conversations = composer.create_dataset()
            if (
                self.user_config.input.custom_dataset_type
                == CustomDatasetType.MOONCAKE_TRACE
            ):
                self._use_sequential_iteration = True
        else:
            composer = ComposerFactory.create_instance(
                ComposerType.SYNTHETIC,
                config=self.user_config,
                tokenizer=self.tokenizer,
            )
            conversations = composer.create_dataset()

        self.dataset = {conv.session_id: conv for conv in conversations}
        self._session_ids_cache = list(self.dataset.keys())

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

        if self._use_sequential_iteration:
            if self._sequential_iterator_index >= len(self._session_ids_cache):
                # Reset iterator if we've gone through all conversations
                _logger.warning(
                    "All conversations have been used. Resetting sequential iterator to start over."
                )
                self._sequential_iterator_index = 0

            session_id = self._session_ids_cache[self._sequential_iterator_index]
            self._sequential_iterator_index += 1

            conversation = self.dataset[session_id]

            self.trace_or_debug(
                lambda: f"Sending sequential conversation response: {conversation}",
                lambda: f"Sending sequential conversation response with id: {conversation.session_id}",
            )
        else:
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
