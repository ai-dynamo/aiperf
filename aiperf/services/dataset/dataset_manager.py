# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import random

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommAddress,
    ComposerType,
    MessageType,
    ServiceType,
)
from aiperf.common.factories import ComposerFactory, ServiceFactory
from aiperf.common.hooks import (
    on_init,
    request_handler,
)
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    DatasetTimingRequest,
    DatasetTimingResponse,
)
from aiperf.common.mixins.reply_client_mixin import ReplyClientMixin
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.base_component_service import BaseComponentService

DATASET_CONFIGURATION_TIMEOUT = 30.0


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
        self.dataset_configured = asyncio.Event()

    @on_init
    async def _initialize(self) -> None:
        """Initialize dataset manager-specific components."""
        self.debug(lambda: f"Initializing dataset manager {self.service_id}")
        self.dataset_configured.clear()
        await self._configure_dataset()
        self.debug(lambda: f"Dataset manager {self.service_id} initialized")

    async def _configure_dataset(self) -> None:
        if self.user_config is None:
            raise self._service_error("User config is required for dataset manager")

        if self.user_config.input.file:
            composer_type = ComposerType.CUSTOM
            self.debug(
                lambda: f"Detected input file '{self.user_config.input.file}'. Setting the composer type to {ComposerType.CUSTOM}."
            )
        else:
            composer_type = ComposerType.SYNTHETIC
            self.debug(
                lambda: f"No input file detected. Setting the composer type to {ComposerType.SYNTHETIC}."
            )

        tokenizer_name = self.user_config.tokenizer.name
        if tokenizer_name is None:
            # TODO: What do we do if there are multiple models?
            # How will we know which tokenizer to use?
            tokenizer_name = self.user_config.model_names[0]

        tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )
        composer = ComposerFactory.create_instance(
            composer_type,
            config=self.user_config.input,
            tokenizer=tokenizer,
        )
        conversations = composer.create_dataset()
        self.dataset = {conv.session_id: conv for conv in conversations}

        self.dataset_configured.set()
        await self.pub_client.publish(
            DatasetConfiguredNotification(
                service_id=self.service_id,
            ),
        )

    @request_handler(MessageType.CONVERSATION_REQUEST)
    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        self.debug(lambda: f"Handling conversation request: {message}")

        # Wait for the dataset to be configured if it is not already
        if not self.dataset_configured.is_set():
            self.debug(
                "Dataset not configured. Waiting for dataset to be configured..."
            )
            await asyncio.wait_for(
                self.dataset_configured.wait(), timeout=DATASET_CONFIGURATION_TIMEOUT
            )

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
        conversation = random.choice(list(self.dataset.values()))
        self.debug(lambda: f"Sending random conversation response: {conversation}")
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
        self.debug(lambda: f"Sending conversation response: {conversation}")
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    @request_handler(MessageType.CONVERSATION_TURN_REQUEST)
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

        self.debug(lambda: f"Sending turn response: {turn}")
        return ConversationTurnResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            turn=turn,
        )

    @request_handler(MessageType.DATASET_TIMING_REQUEST)
    async def _handle_dataset_timing_request(
        self, message: DatasetTimingRequest
    ) -> DatasetTimingResponse:
        """Handle a dataset timing request."""
        self.debug(lambda: f"Handling dataset timing request: {message}")
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


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    main()
