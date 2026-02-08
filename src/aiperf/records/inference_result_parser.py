# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import suppress

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.hooks import on_init
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import ReasoningResponseData, TokenCounts
from aiperf.common.tokenizer import Tokenizer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


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
        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, self.model_endpoint.endpoint.type
        )
        self.endpoint = EndpointClass(model_endpoint=self.model_endpoint)
        endpoint_meta = plugins.get_endpoint_metadata(self.model_endpoint.endpoint.type)
        # Disable tokenization if the endpoint doesn't produce tokens and doesn't tokenize input, or
        # if the user config is set to use server token counts.
        self.disable_tokenization: bool = (
            user_config.endpoint.use_server_token_count
            or (not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input)
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
        if self.disable_tokenization:
            self.info(
                "Tokenization is disabled for this endpoint, skipping tokenizer configuration"
            )
            return

        self.info("Configuring tokenizers for inference result parser")
        begin = time.perf_counter()
        tokenizer_config = self.user_config.tokenizer

        async with self.tokenizer_lock:
            self.tokenizers = {}
            for model in self.model_endpoint.models.models:
                self.tokenizers[model.name] = await asyncio.to_thread(
                    Tokenizer.from_pretrained,
                    tokenizer_config.get_tokenizer_name_for_model(model.name),
                    trust_remote_code=tokenizer_config.trust_remote_code,
                    revision=tokenizer_config.revision,
                    resolve_alias=tokenizer_config.should_resolve_alias,
                )

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
                tokenizer_config = self.user_config.tokenizer
                self.tokenizers[model] = await asyncio.to_thread(
                    Tokenizer.from_pretrained,
                    tokenizer_config.get_tokenizer_name_for_model(model),
                    trust_remote_code=tokenizer_config.trust_remote_code,
                    revision=tokenizer_config.revision,
                    resolve_alias=tokenizer_config.should_resolve_alias,
                )
            return self.tokenizers[model]

    async def parse_request_record(
        self, request_record: RequestRecord
    ) -> ParsedResponseRecord:
        """Handle an inference results message."""
        request_info = request_record.request_info
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: f"Received inference results for credit '{request_info.credit_num}' (id: {request_info.x_request_id})"
            if request_info
            else "Received inference results (no request_info)",
        )

        # Make sure any invalid request records are converted to error records for combined processing.
        request_record.create_error_from_invalid()

        if request_record.has_error:
            # Even for error records, compute input token count if possible
            input_token_count = None
            if not self.disable_tokenization:
                # Suppress exceptions during token counting for error records to avoid masking the original error.
                # If token counting fails, we still return the error record with token_counts.input=None.
                with suppress(Exception):
                    input_token_count = await self.compute_input_token_count(
                        request_record
                    )

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                token_counts=TokenCounts(
                    input=input_token_count,
                ),
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
                        token_counts=TokenCounts(
                            input=record.token_counts.input
                            if record.token_counts
                            else None
                        ),
                    )
                else:
                    # Success path: valid record with no errors
                    self.debug(
                        lambda: f"Received {len(record.request.responses)} response packet(s), token counts: {record.token_counts}"
                    )
                    return record

            except Exception as e:
                # TODO: We should add an ErrorDetails to the response record and not the request record.
                self.exception(f"Error processing valid record: {e}")
                request_record.error = ErrorDetails.from_exception(e)
                input_token_count = None

                if not self.disable_tokenization:
                    # Suppress exceptions during token counting for error records to avoid masking the original error.
                    # If token counting fails, we still return the error record with token_counts.input=None.
                    with suppress(Exception):
                        input_token_count = await self.compute_input_token_count(
                            request_record
                        )

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    token_counts=TokenCounts(
                        input=input_token_count,
                    ),
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
            )

        resp = self.endpoint.extract_response_data(request_record)

        # Compute token counts based on configuration
        if self.user_config.endpoint.use_server_token_count:
            token_counts = await self._compute_server_token_counts(resp)
        elif not self.disable_tokenization:
            token_counts = await self._compute_client_side_token_counts(
                request_record, resp
            )
        else:
            token_counts = TokenCounts()

        return ParsedResponseRecord(
            request=request_record,
            responses=resp,
            token_counts=token_counts,
        )

    async def compute_input_token_count(
        self, request_record: RequestRecord
    ) -> int | None:
        """Compute the number of tokens in the input for a given request record.

        This includes:
        - system_message (shared system prompt)
        - user_context_message (per-conversation user context)
        - All turns' text content
        """
        turns = request_record.turns
        if turns is None:
            self.warning(
                "Turns are not set for request record, unable to calculate input token count"
            )
            return None

        tokenizer = await self.get_tokenizer(request_record.model_name)
        prompt_texts: list[str] = []

        # Include system_message if present (shared system prompt)
        if request_record.request_info and request_record.request_info.system_message:
            prompt_texts.append(request_record.request_info.system_message)

        # Include user_context_message if present (per-conversation user context)
        if (
            request_record.request_info
            and request_record.request_info.user_context_message
        ):
            prompt_texts.append(request_record.request_info.user_context_message)

        # Include all turns' text content
        for turn in turns:
            for text in turn.texts:
                prompt_texts.append("".join(text.contents))

        if not prompt_texts:
            return None

        # NOTE: We combine all the prompt texts with a space separator to create a single prompt string.
        # This will get us the most accurate token count for the prompt by avoiding any potential
        # boundary issues that could occur if we were to tokenize each text individually.
        return self._compute_token_count(tokenizer, prompt_texts, separator=" ")

    async def _compute_server_token_counts(
        self, responses: list[ParsedResponse]
    ) -> TokenCounts:
        """Compute token counts using server-provided usage fields.

        Args:
            responses: List of parsed responses from the server

        Returns:
            TokenCounts populated with server-reported values
        """
        input_token_count = self._extract_server_input_token_count(responses)
        reasoning_token_count = self._extract_server_reasoning_token_count(responses)
        output_token_count = self._extract_server_output_token_count(
            responses, reasoning_token_count
        )

        token_counts = TokenCounts(
            input=input_token_count,
            reasoning=reasoning_token_count,
            output=output_token_count,
        )

        # Warn if server provided no usage information
        if (
            token_counts.input is None
            and token_counts.output is None
            and token_counts.reasoning is None
        ):
            self.warning(
                "Server did not provide token usage information. Token count metrics will be unavailable. "
                "Verify that your API endpoint supports usage reporting (stream_options are automatically configured for OpenAI-compatible endpoints)."
            )

        return token_counts

    def _parse_output_and_reasoning_texts(
        self, responses: list[ParsedResponse]
    ) -> tuple[list[str], list[str]]:
        """Parse all the output and reasoning texts from the responses.

        Args:
            responses: List of parsed responses from the server

        Returns:
            Tuple of lists of output and reasoning texts
        """
        output_texts: list[str] = []
        reasoning_texts: list[str] = []
        for response in responses:
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.reasoning:
                    reasoning_texts.append(response.data.reasoning)
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())

        return output_texts, reasoning_texts

    def _compute_token_count(
        self, tokenizer: Tokenizer, texts: list[str], separator: str = ""
    ) -> int | None:
        """Compute the number of tokens in the texts by joining them with an optional separator (default none) and encoding with the tokenizer.

        Args:
            tokenizer: The tokenizer to use
            texts: List of texts to compute the token count for
            separator: The separator to use between the texts

        Returns:
            The number of tokens in the texts, or None if the texts are empty
        """
        if not texts:
            return None
        return len(tokenizer.encode(separator.join(texts)))

    async def _compute_client_side_token_counts(
        self, request_record: RequestRecord, responses: list[ParsedResponse]
    ) -> TokenCounts:
        """Compute token counts using client-side tokenization.

        Args:
            request_record: The request record containing input data
            responses: List of parsed responses from the server

        Returns:
            TokenCounts populated with client-side tokenized values
        """
        input_token_count = await self.compute_input_token_count(request_record)

        tokenizer = await self.get_tokenizer(request_record.model_name)
        output_texts, reasoning_texts = self._parse_output_and_reasoning_texts(
            responses
        )
        output_token_count = self._compute_token_count(tokenizer, output_texts)
        reasoning_token_count = self._compute_token_count(tokenizer, reasoning_texts)

        return TokenCounts(
            input=input_token_count,
            reasoning=reasoning_token_count,
            output=output_token_count,
        )

    def _extract_server_input_token_count(
        self, responses: list[ParsedResponse]
    ) -> int | None:
        """Extract input token count from server usage field.

        Searches backwards through responses for the last non-None value.
        This handles streaming where usage appears in the final chunk.

        Args:
            responses: List of parsed responses from the server

        Returns:
            Server-reported prompt token count, or None if unavailable
        """
        for response in reversed(responses):
            if response.usage and response.usage.prompt_tokens is not None:
                return response.usage.prompt_tokens
        return None

    def _extract_server_reasoning_token_count(
        self, responses: list[ParsedResponse]
    ) -> int | None:
        """Extract reasoning token count from server usage field.

        Reasoning tokens are nested in completion_tokens_details.reasoning_tokens
        per the OpenAI API specification.

        Args:
            responses: List of parsed responses from the server

        Returns:
            Server-reported reasoning tokens, or None if unavailable
        """
        for response in reversed(responses):
            if response.usage and response.usage.reasoning_tokens is not None:
                return response.usage.reasoning_tokens
        return None

    def _extract_server_output_token_count(
        self, responses: list[ParsedResponse], reasoning_token_count: int | None
    ) -> int | None:
        """Extract output token count from server usage field.

        Returns ONLY non-reasoning completion tokens. The server's completion_tokens
        includes both reasoning and output, so we subtract reasoning_tokens to get
        the pure output count (matching our client-side semantics).

        Args:
            responses: List of parsed responses from the server
            reasoning_token_count: The reasoning token count to subtract from completion tokens

        Returns:
            Server-reported output tokens (excluding reasoning), or None if unavailable
        """
        for response in reversed(responses):
            if response.usage:
                completion_tokens = response.usage.completion_tokens
                if completion_tokens is not None:
                    reasoning_tokens = reasoning_token_count or 0
                    result = completion_tokens - reasoning_tokens
                    if result < 0:
                        self.warning(
                            f"Server reported inconsistent token counts: completion_tokens={completion_tokens}, "
                            f"reasoning_tokens={reasoning_tokens}. Clamping output tokens to 0."
                        )
                        return 0
                    return result
        return None
