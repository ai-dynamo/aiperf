# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.comms.base_comms import PushClientProtocol
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import DEFAULT_PULL_CLIENT_MAX_CONCURRENCY
from aiperf.common.enums import CommAddress, MessageType, ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_init, on_pull_message
from aiperf.common.messages import (
    DatasetJobMessage,
    DatasetJobResult,
    DatasetResultMessage,
)
from aiperf.common.mixins import PullClientMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer


@ServiceFactory.register(ServiceType.DATASET_PROCESSOR)
class DatasetProcessor(PullClientMixin, BaseComponentService):
    """
    DatasetProcessor is responsible for generating dataset conversations in parallel.
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
            pull_client_address=CommAddress.DATASET_JOB,
            pull_client_bind=False,
            pull_client_max_concurrency=DEFAULT_PULL_CLIENT_MAX_CONCURRENCY,
        )
        self.debug("Initializing dataset processor service")
        self.results_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.DATASET_RESULT
        )
        self.tokenizer: Tokenizer | None = None
        self.user_config = user_config

    @on_init
    async def _initialize(self) -> None:
        """Initialize dataset processor service-specific components."""
        self.debug("Initializing dataset processor service")
        tokenizer_name = self.user_config.tokenizer.name
        if tokenizer_name is None:
            tokenizer_name = self.user_config.endpoint.model_names[0]

        self.tokenizer = Tokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.user_config.tokenizer.trust_remote_code,
            revision=self.user_config.tokenizer.revision,
        )

    @on_pull_message(MessageType.DATASET_JOB)
    async def _on_dataset_job(self, message: DatasetJobMessage) -> None:
        """Handle a dataset generation job."""
        self.debug(lambda: f"Received dataset generation job: {message}")

        # if self.user_config.input.file:
        #    composer_type = ComposerType.CUSTOM
        # else:
        #    composer_type = ComposerType.SYNTHETIC

        # composer = ComposerFactory.create_instance(
        #    composer_type,
        #    config=self.user_config,
        #    tokenizer=self.tokenizer,
        # )

        # TODO: Generate conversation
        conversation = Conversation(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            turns=[],
        )

        for _ in range(message.info.num_turns):
            conversation.turns.append(Turn(role="user", content="Hello, how are you?"))

        await self.results_push_client.push(
            DatasetResultMessage(
                service_id=self.service_id,
                result=DatasetJobResult(
                    conversation=conversation,
                ),
            )
        )


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(DatasetProcessor)


if __name__ == "__main__":
    main()
