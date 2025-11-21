# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.factories import EndpointFactory
from aiperf.common.hooks import on_init
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.models import ErrorDetails, ParsedResponseRecord, RequestRecord
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import ReasoningResponseData
from aiperf.common.protocols import EndpointProtocol
from aiperf.common.tokenizer import Tokenizer


# TODO: Should we create non-tokenizer based parsers?
class InferenceResultParser(CommunicationMixin):
    """InferenceResultParser is responsible for parsing the inference results."""

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.user_config: UserConfig = user_config
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_user_config(
            user_config
        )
        self.endpoint: EndpointProtocol = EndpointFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )
        self.debug(
            lambda: f"Created endpoint for {self.model_endpoint.endpoint.type}, "
            f"class: {self.endpoint.__class__.__name__}",
        )

    @on_init
    async def _initialize(self) -> None:
        """Initialize inference result parser-specific components."""
        self.debug("Initializing inference result parser")

    async def configure(self) -> None:
        """Configure the tokenizers."""
        self.info("Configuring tokenizers for inference result parser")
        begin = time.perf_counter()
        async with self.tokenizer_lock:
            self.tokenizers = {
                model.name: Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model.name,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
                for model in self.model_endpoint.models.models
            }
        duration = time.perf_counter() - begin
        tokenizer_info = {
            model: {
                "class": tokenizer._tokenizer.__class__.__name__,
                "name_or_path": getattr(tokenizer._tokenizer, "name_or_path", ""),
            }
            for model, tokenizer in self.tokenizers.items()
        }
        self.info(f"Initialized tokenizers: {tokenizer_info} in {duration:.2f} seconds")

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model or create it if it doesn't exist."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                self.tokenizers[model] = Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
            return self.tokenizers[model]

    async def parse_request_record(
        self, request_record: RequestRecord
    ) -> ParsedResponseRecord:
        """Handle an inference results message."""
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: f"Received inference results for credit '{request_record.credit_num}' (id: {request_record.x_request_id})",
        )

        # Make sure any invalid request records are converted to error records for combined processing.
        request_record.create_error_from_invalid()

        if request_record.has_error:
            # Even for error records, compute input token count if possible
            try:
                input_token_count = await self.compute_input_token_count(request_record)
            except Exception as e:
                self.warning(f"Error computing input token count for error record: {e}")
                input_token_count = None

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                input_token_count=input_token_count,
            )

        else:
            try:
                record = await self.process_valid_record(request_record)

                # Check if the parsed record is actually valid (e.g., has content responses)
                record.create_error_from_invalid()

                if record.has_error:
                    # Parsed record was invalid, return as error record
                    return ParsedResponseRecord(
                        request=record.request,
                        responses=[],
                        input_token_count=record.input_token_count,
                    )

                self.debug(
                    lambda: f"Received {len(record.request.responses)} response packet(s), input_token_count: {record.input_token_count}, "
                    f"output_token_count: {record.output_token_count}, reasoning_token_count: {record.reasoning_token_count}"
                )
                return record
            except Exception as e:
                # TODO: We should add an ErrorDetails to the response record and not the request record.
                self.exception(f"Error processing valid record: {e}")
                request_record.error = ErrorDetails.from_exception(e)

                try:
                    input_token_count = await self.compute_input_token_count(
                        request_record
                    )
                except Exception:
                    input_token_count = None

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    input_token_count=input_token_count,
                )

    async def process_valid_record(
        self, request_record: RequestRecord
    ) -> ParsedResponseRecord:
        """Process a valid request record."""
        if request_record.model_name is None:
            self.warning(
                lambda: f"Model name is None, unable to process record: {request_record}"
            )
            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                input_token_count=None,
                output_token_count=None,
            )

        resp = self.endpoint.extract_response_data(request_record)
        input_token_count = await self.compute_input_token_count(request_record)

        output_texts: list[str] = []
        reasoning_texts: list[str] = []
        for response in resp:
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.reasoning:
                    reasoning_texts.append(response.data.reasoning)
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())

        tokenizer = await self.get_tokenizer(request_record.model_name)
        output_token_count = (
            len(tokenizer.encode("".join(output_texts))) if output_texts else None
        )
        reasoning_token_count = (
            len(tokenizer.encode("".join(reasoning_texts))) if reasoning_texts else None
        )

        return ParsedResponseRecord(
            request=request_record,
            responses=resp,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            reasoning_token_count=reasoning_token_count,
        )

    async def compute_input_token_count(
        self, request_record: RequestRecord
    ) -> int | None:
        """Compute the number of tokens in the input for a given request record."""
        turns = request_record.turns
        if turns is None:
            self.warning(
                "Turns are not set for request record, unable to calculate input token count"
            )
            return None

        tokenizer = await self.get_tokenizer(request_record.model_name)
        input_token_count = 0
        # TODO: We need to handle images, audios, videos, etc.
        for turn in turns:
            for text in turn.texts:
                input_token_count += len(tokenizer.encode("".join(text.contents)))
        return input_token_count
