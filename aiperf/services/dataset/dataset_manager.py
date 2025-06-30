# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import random
import sys

from pydantic import BaseModel, ConfigDict

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.dataset_models import Conversation
from aiperf.common.enums import (
    ComposerType,
    CustomDatasetType,
    MessageType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    DatasetTimingRequest,
    DatasetTimingResponse,
    Message,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.composer import ComposerFactory
from aiperf.services.dataset.config import DatasetConfig, PromptConfig


################################################################################
# TODO: Temporary (remove when command config is ready)
class MockConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str | None = None
    tokenizer: Tokenizer | None = None
    custom_dataset_type: CustomDatasetType | None = None
    public_dataset: str | None = None
    prompt: PromptConfig | None = None


################################################################################


@ServiceFactory.register(ServiceType.DATASET_MANAGER)
class DatasetManager(BaseComponentService):
    """
    The DatasetManager primary responsibility is to manage the data generation or acquisition.
    For synthetic generation, it contains the code to generate the prompts or tokens.
    It will have an API for dataset acquisition of a dataset if available in a remote repository or database.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.dataset: dict[str, Conversation] = {}  # session ID -> Conversation mapping

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.DATASET_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize dataset manager-specific components."""
        self.logger.debug("Initializing dataset manager")

        await self.comms.register_request_handler(
            service_id=self.service_id,
            message_type=MessageType.CONVERSATION_REQUEST,
            handler=self._handle_conversation_request,
        )

        await self.comms.register_request_handler(
            service_id=self.service_id,
            message_type=MessageType.DATASET_TIMING_REQUEST,
            handler=self._handle_dataset_timing_request,
        )

    @on_start
    async def _start(self) -> None:
        """Start the dataset manager."""
        self.logger.debug("Starting dataset manager")
        # TODO: Implement dataset manager start

    @on_stop
    async def _stop(self) -> None:
        """Stop the dataset manager."""
        self.logger.debug("Stopping dataset manager")
        # TODO: Implement dataset manager stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up dataset manager-specific components."""
        self.logger.debug("Cleaning up dataset manager")
        # TODO: Implement dataset manager cleanup

    @on_configure
    async def _configure(self, message: Message) -> None:
        """Configure the dataset manager."""
        self.logger.debug(f"Configuring dataset manager with message: {message}")

        # TODO: remove this mock config
        # mocks config inside the message
        config = MockConfig()
        config.filename = os.getenv("AIPERF_DATASET_FILENAME", None)  # "trace1.jsonl"
        config.tokenizer = Tokenizer.from_pretrained(os.getenv("AIPERF_MODEL", "gpt2"))

        if config.filename:
            composer_type = ComposerType.CUSTOM
            config.custom_dataset_type = CustomDatasetType.TRACE
        else:
            composer_type = ComposerType.SYNTHETIC
            config.custom_dataset_type = CustomDatasetType.SINGLE_TURN  # ignored

        # TODO: update once we integrate with command config
        dataset_config = DatasetConfig(
            filename=config.filename,
            tokenizer=config.tokenizer,
            custom_dataset_type=config.custom_dataset_type,
            prompt=PromptConfig(mean=10, stddev=2),
        )

        composer = ComposerFactory.create_instance(composer_type, config=dataset_config)
        conversations = composer.create_dataset()
        self.dataset = {conv.session_id: conv for conv in conversations}

    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        self.logger.debug("Handling conversation request: %s", message)

        if not self.dataset:
            raise self._service_error(
                "Dataset is empty and must be configured before handling requests.",
            )

        if message.conversation_id is None:
            return await self._return_any_conversation(
                request_id=message.request_id,
            )
        else:
            return await self._return_conversation_by_id(
                request_id=message.request_id,
                conversation_id=message.conversation_id,
            )

    async def _return_any_conversation(
        self, request_id: str | None
    ) -> ConversationResponseMessage:
        """Return any conversation from the dataset based on the user specified method."""

        # TODO: Implement the user specified method (random, round robin, etc.)
        conversation = random.choice(list(self.dataset.values()))
        self.logger.debug("Sending random conversation response: %s", conversation)
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    async def _return_conversation_by_id(
        self, request_id: str | None, conversation_id: str
    ) -> ConversationResponseMessage:
        """Return a conversation if it exists, otherwise raise an error."""

        if conversation_id not in self.dataset:
            raise self._service_error(
                f"Conversation {conversation_id} not found in dataset.",
            )

        conversation = self.dataset[conversation_id]
        self.logger.debug("Sending conversation response: %s", conversation)
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=request_id,
            conversation=conversation,
        )

    async def _handle_dataset_timing_request(
        self, message: DatasetTimingRequest
    ) -> DatasetTimingResponse:
        """Handle a dataset timing request."""
        self.logger.debug("Handling dataset timing request: %s", message)
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
    sys.exit(main())
