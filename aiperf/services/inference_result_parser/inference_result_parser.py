# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import sys

from aiperf.clients.client_interfaces import ResponseExtractorFactory
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.comms.base import (
    PullClientProtocol,
    PushClientProtocol,
    RequestClientProtocol,
)
from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CommunicationClientAddressType, MessageType, ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_configure,
    on_init,
)
from aiperf.common.messages import (
    CommandMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    ErrorMessage,
    InferenceResultsMessage,
    ParsedInferenceResultsMessage,
)
from aiperf.common.record_models import (
    ErrorDetails,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.tokenizer import Tokenizer


@ServiceFactory.register(ServiceType.INFERENCE_RESULT_PARSER)
class InferenceResultParser(BaseComponentService):
    """InferenceResultParser is responsible for parsing the inference results
    and pushing them to the RecordsManager.
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
        )
        self.debug("Initializing inference result parser")
        self.inference_results_client: PullClientProtocol = (
            self.comms.create_pull_client(
                CommunicationClientAddressType.RAW_INFERENCE_PROXY_BACKEND,
            )
        )
        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommunicationClientAddressType.RECORDS,
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommunicationClientAddressType.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.user_config: UserConfig = user_config
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_user_config(
            user_config
        )

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.INFERENCE_RESULT_PARSER

    @on_init
    async def _initialize(self) -> None:
        """Initialize inference result parser-specific components."""
        self.debug("Initializing inference result parser")

        await self.inference_results_client.register_pull_callback(
            message_type=MessageType.INFERENCE_RESULTS,
            callback=self._on_inference_results,
            # TODO: Support for unbounded concurrency in the future by setting to None or 0?
            max_concurrency=1_000_000,
        )

        self.extractor = ResponseExtractorFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )

        async with self.tokenizer_lock:
            self.tokenizers = {
                model.name: Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model.name,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
                for model in self.model_endpoint.models.models
            }
            self.info("Initialized tokenizers for %d models", len(self.tokenizers))

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                self.tokenizers[model] = Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
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

    async def _compute_input_token_count(
        self, request_record: RequestRecord, tokenizer: Tokenizer
    ) -> int:
        "Extract the input prompt text from the request record and return token count."
        input_data = request_record.request
        prompt = None
        if "messages" in input_data:
            prompt = " ".join(
                m["content"] for m in input_data["messages"] if "content" in m
            )
        elif "input" in input_data:
            if isinstance(input_data["input"], list):
                prompt = " ".join(input_data["input"])
            else:
                prompt = input_data["input"]
        elif "prompt" in input_data:
            if isinstance(input_data["prompt"], list):
                prompt = " ".join(input_data["prompt"])
            else:
                prompt = input_data["prompt"]
        elif "inputs" in input_data:
            if isinstance(input_data["inputs"], list):
                prompt = " ".join(input_data["inputs"])
            else:
                prompt = input_data["inputs"]

        if prompt is None:
            raise ValueError("Unable to extract prompt from input.")

        return len(tokenizer.encode(prompt))

    async def _on_inference_results(self, message: InferenceResultsMessage) -> None:
        """Handle an inference results message."""
        self.debug(lambda: f"Received inference results message: {message}")

        if message.record.has_error:
            await self.records_push_client.push(
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
            try:
                record = await self.process_valid_record(message)
                self.debug(
                    lambda: f"Received {len(record.request.responses)} responses, isl: {record.isl}, osl: {record.token_count}"
                )
                await self.records_push_client.push(
                    ParsedInferenceResultsMessage(
                        service_id=self.service_id,
                        record=record,
                    )
                )
            except Exception as e:
                self.exception(f"Error processing valid record: {e}")
                await self.records_push_client.push(
                    ParsedInferenceResultsMessage(
                        service_id=self.service_id,
                        record=ParsedResponseRecord(
                            worker_id=message.service_id,
                            request=message.record,
                            responses=[],
                        ),
                    )
                )
        else:
            self.warning(f"Received invalid inference results: {message.record}")
            message.record.error = ErrorDetails(
                code=None,
                message="Invalid inference results",
                type="InvalidInferenceResults",
            )
            await self.records_push_client.push(
                ParsedInferenceResultsMessage(
                    service_id=self.service_id,
                    record=ParsedResponseRecord(
                        worker_id=message.service_id,
                        request=message.record,
                        responses=[],
                    ),
                )
            )

    async def process_valid_record(
        self, message: InferenceResultsMessage
    ) -> ParsedResponseRecord:
        """Process a valid request record."""
        if message.record.model_name is None:
            self.warning(
                lambda: f"Model name is None, unable to process record: {message.record}"
            )
            return ParsedResponseRecord(
                worker_id=message.service_id,
                request=message.record,
                responses=[],
                isl=None,
            )

        tokenizer = await self.get_tokenizer(message.record.model_name)
        resp = await self.extractor.extract_response_data(message.record, tokenizer)
        isl = await self.compute_isl(message.record, tokenizer)

        return ParsedResponseRecord(
            worker_id=message.service_id,
            request=message.record,
            responses=resp,
            isl=isl,
        )

    async def compute_isl(
        self, record: RequestRecord, tokenizer: Tokenizer
    ) -> int | None:
        """Compute the ISL for a given request record."""
        if record.conversation_id is None or record.turn_index is None:
            self.warning(
                lambda: f"Conversation ID or turn index is None: {record.conversation_id=} {record.turn_index=}"
            )
            return None

        turn_response: ConversationTurnResponseMessage = (
            await self.conversation_request_client.request(
                ConversationTurnRequestMessage(
                    service_id=self.service_id,
                    conversation_id=record.conversation_id,
                    turn_index=record.turn_index,
                )
            )
        )
        if isinstance(turn_response, ErrorMessage):
            self.error(lambda: f"Error getting turn response: {turn_response}")
            return None

        turn = turn_response.turn
        return sum(
            len(tokenizer.encode(content))
            for text in turn.text
            for content in text.content
        )


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(InferenceResultParser)


if __name__ == "__main__":
    sys.exit(main())
