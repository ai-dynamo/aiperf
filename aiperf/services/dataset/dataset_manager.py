# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

from pydantic import BaseModel, ConfigDict

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.dataset_models import Conversation
from aiperf.common.enums import CustomDatasetType, ServiceType
from aiperf.common.exceptions import ServiceConfigurationError
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
    Message,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.composer import ComposerFactory
from aiperf.services.dataset.config import PromptConfig
from aiperf.services.dataset.enums import ComposerType


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
        # TODO: wait until Anthony's ZMQRouterRepClient is merged
        # await self.comms.register_request_handler(
        #    service_id=self.service_id,
        #    topic=Topic.CONVERSATION_DATA,
        #    message_type=MessageType.CONVERSATION_REQUEST,
        #    handler=self._handle_conversation_request,
        # )

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
        config.filename = "trace1.jsonl"
        config.tokenizer = Tokenizer.from_pretrained("gpt2")

        if config.filename:
            composer_type = ComposerType.CUSTOM
            config.custom_dataset_type = CustomDatasetType.TRACE
        else:
            composer_type = ComposerType.SYNTHETIC
            config.custom_dataset_type = CustomDatasetType.NONE  # default

        composer = ComposerFactory.create_instance(composer_type, config)
        conversations = composer.create_dataset()
        self.dataset = {conv.session_id: conv for conv in conversations}

    async def _handle_conversation_request(
        self, message: ConversationRequestMessage
    ) -> ConversationResponseMessage:
        """Handle a conversation request."""
        if not self.dataset:
            raise ServiceConfigurationError(
                "Dataset is empty and must be loaded before handling requests."
            )
        if message.conversation_id not in self.dataset:
            raise ServiceConfigurationError(
                "Conversation %s not found in dataset.", message.conversation_id
            )

        self.logger.debug("Handling conversation request: %s", message)
        conversation = self.dataset[message.conversation_id]
        return ConversationResponseMessage(
            service_id=self.service_id,
            request_id=message.request_id,
            conversation_id=message.conversation_id,
            conversation=conversation,
        )


def main() -> None:
    """Main entry point for the dataset manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetManager)


if __name__ == "__main__":
    sys.exit(main())
