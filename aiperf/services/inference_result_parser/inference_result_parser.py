# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import sys

from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_configure,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.models.messages import (
    CommandMessage,
    InferenceResultsMessage,
    ParsedInferenceResultsMessage,
)
from aiperf.common.models.record_models import ErrorDetails, ParsedResponseRecord
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.tokenizer import Tokenizer
from aiperf.parsers.openai_parsers import OpenAIResponseExtractor


@ServiceFactory.register(ServiceType.INFERENCE_RESULT_PARSER)
class InferenceResultParser(BaseComponentService):
    """InferenceResultParser is responsible for parsing the inference results
    and pushing them to the RecordsManager.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing inference result parser")
        # TODO: Enable after ZMQ client refactor
        # self.inference_results_client: PullClient = self.comms.create_pull_client(
        #     ClientAddressType.PUSH_PULL_BACKEND,
        # )
        # self.response_results_client: PushClient = self.comms.create_push_client(
        #     ClientAddressType.INFERENCE_RESULTS_PUSH_PULL,
        # )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.user_config: UserConfig | None = None
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.extractor = OpenAIResponseExtractor()

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.INFERENCE_RESULT_PARSER

    @on_init
    async def _initialize(self) -> None:
        """Initialize inference result parser-specific components."""
        self.logger.debug("Initializing inference result parser")
        # TODO: Enable after ZMQ client refactor
        # await self.inference_results_client.register_pull_callback(
        #     message_type=MessageType.INFERENCE_RESULTS,
        #     callback=self._on_inference_results,
        #     max_concurrency=1000000,
        # )

    @on_start
    async def _start(self) -> None:
        """Start the inference result parser."""
        self.logger.debug("Starting inference result parser")
        # TODO: Implement inference result parser start

    @on_stop
    async def _stop(self) -> None:
        """Stop the inference result parser."""
        self.logger.debug("Stopping inference result parser")
        # TODO: Implement inference result parser stop

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up inference result parser-specific components."""
        self.logger.debug("Cleaning up inference result parser")
        # TODO: Implement inference result parser cleanup

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                self.tokenizers[model] = Tokenizer.from_pretrained(model)
            return self.tokenizers[model]

    @on_configure
    async def _configure(self, message: CommandMessage) -> None:
        """Configure the inference result parser."""
        self.logger.debug(
            f"Configuring inference result parser with message: {message}"
        )
        self.user_config = (
            message.data if isinstance(message.data, UserConfig) else None
        )

        # TODO: This is a hack to get the tokenizer for the default model.
        # We should remove this once we have a better way to get the tokenizer from the user config.
        await self.get_tokenizer(
            os.getenv("AIPERF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        )

        if self.user_config:
            # TODO: Does this code actually work as intended? Maybe refactor this to use a loop.
            await asyncio.gather(
                *[self.get_tokenizer(model) for model in self.user_config.model_names]
            )
            self.logger.info(
                "Initialized tokenizers for %d models", len(self.tokenizers)
            )

    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        self.logger.debug(f"Received inference results message: {message}")

        if message.record.has_error:
            await self.response_results_client.push(
                ParsedInferenceResultsMessage(
                    service_id=self.service_id,
                    record=ParsedResponseRecord(
                        worker_id=message.service_id,
                        request=message.record,
                        responses=[],
                    ),
                )
            )

        elif message.record.valid:
            tokenizer = await self.get_tokenizer(message.record.request["model"])
            resp = await self.extractor.extract_response_data(message.record, tokenizer)

            result = ParsedInferenceResultsMessage(
                service_id=self.service_id,
                record=ParsedResponseRecord(
                    worker_id=message.service_id,
                    request=message.record,
                    responses=resp,
                ),
            )
            self.logger.debug(
                "Received %d responses, %d total tokens",
                len(resp),
                result.record.token_count,
            )
            await self.response_results_client.push(result)
        else:
            self.logger.warning(
                "Received invalid inference results: %s", message.record
            )
            message.record.error = ErrorDetails(
                code=None,
                message="Invalid inference results",
                type="InvalidInferenceResults",
            )
            await self.response_results_client.push(
                ParsedInferenceResultsMessage(
                    service_id=self.service_id,
                    record=ParsedResponseRecord(
                        worker_id=message.service_id,
                        request=message.record,
                        responses=[],
                    ),
                )
            )


def main() -> None:
    """Main entry point for the inference result parser."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(InferenceResultParser)


if __name__ == "__main__":
    sys.exit(main())
