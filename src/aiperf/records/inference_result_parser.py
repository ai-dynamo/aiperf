# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import suppress

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ExportLevel
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
        self.disable_tokenization: bool = (
            not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input
        )
        self.tokenize_output: bool = user_config.tokenizer.tokenize_output
        self.tokenize_input: bool = user_config.tokenizer.tokenize_input
        self._explicit_no_tokenize_input: bool = (
            "tokenize_input" in user_config.tokenizer.model_fields_set
            and not user_config.tokenizer.tokenize_input
        )
        self._warned_no_usage: bool = False
        if (
            self.model_endpoint.endpoint.streaming
            and self.model_endpoint.endpoint.stream_usage
        ):
            self.info(
                "stream_options.include_usage is enabled for streaming requests. "
                "Server-reported token counts will be requested. "
                "Use --no-stream-usage if the server does not support stream_options."
            )
        if not self.disable_tokenization and self._explicit_no_tokenize_input:
            self.info(
                "Input tokenization is disabled (--no-tokenize-input). "
                "Client-side input token counts will not be computed, even as fallback. "
                "Use --tokenize-input to enable."
            )
        elif not self.disable_tokenization and not self.tokenize_input:
            self.info(
                "Always-on input tokenization is disabled. "
                "Client-side input tokenization will still occur as a fallback "
                "when the server does not report prompt tokens. "
                "Use --tokenize-input to enable for all requests."
            )
        if not self.disable_tokenization and not self.tokenize_output:
            self.info(
                "Output tokenization is disabled. "
                "Usage output and reasoning token diff metrics will not be available. "
                "Use --tokenize-output to enable."
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
            input_local = None
            if not self.disable_tokenization and not self._explicit_no_tokenize_input:
                # Suppress exceptions during token counting for error records to avoid masking the original error.
                # If token counting fails, we still return the error record with token_counts.input_local=None.
                with suppress(Exception):
                    input_local = await self.compute_input_token_count(request_record)

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                token_counts=TokenCounts(
                    input_local=input_local,
                ),
            )

        else:
            try:
                raw_response_count = len(request_record.responses)
                record = await self.process_valid_record(request_record)

                # Check if the parsed record is actually valid (e.g., has content responses)
                record.create_error_from_invalid()

                if record.has_error:
                    # Parsed record was invalid, return as error record
                    return ParsedResponseRecord(
                        request=record.request,
                        responses=[],
                        token_counts=TokenCounts(
                            input_local=record.token_counts.input_local
                            if record.token_counts
                            else None
                        ),
                    )
                else:
                    # Success path: valid record with no errors
                    self.debug(
                        lambda: f"Received {raw_response_count} response packet(s), token counts: {record.token_counts}"
                    )
                    return record

            except Exception as e:
                # TODO: We should add an ErrorDetails to the response record and not the request record.
                self.exception(f"Error processing valid record: {e}")
                request_record.error = ErrorDetails.from_exception(e)
                input_local = None

                if (
                    not self.disable_tokenization
                    and not self._explicit_no_tokenize_input
                ):
                    # Suppress exceptions during token counting for error records to avoid masking the original error.
                    # If token counting fails, we still return the error record with token_counts.input_local=None.
                    with suppress(Exception):
                        input_local = await self.compute_input_token_count(
                            request_record
                        )

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    token_counts=TokenCounts(
                        input_local=input_local,
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

        # Free the raw responses list after extraction.
        # Skip when RAW export needs the original responses for serialization.
        if self.user_config.output.export_level != ExportLevel.RAW:
            request_record.responses = None

        token_counts = await self._compute_token_counts(request_record, resp)

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
        return await self._compute_token_count(tokenizer, prompt_texts, separator=" ")

    async def _compute_token_counts(
        self, request_record: RequestRecord, responses: list[ParsedResponse]
    ) -> TokenCounts:
        """Compute token counts using server usage for input/output/reasoning and client-side tokenization for input_local.

        Server-reported usage fields are used for input, output, and reasoning counts.
        Client-side tokenization is computed for input and stored as `input_local`.
        Client-side output/reasoning tokenization is stored in `output_local`/`reasoning_local`
        either as a fallback (when server doesn't report) or when `--tokenize-output` is enabled.

        Args:
            request_record: The request record containing input data
            responses: List of parsed responses from the server

        Returns:
            TokenCounts with both client and server values populated
        """
        # Server-reported counts (extracted first to enable fallback logic)
        input_token_count = self._extract_server_input_token_count(responses)
        reasoning_server = self._extract_server_reasoning_token_count(responses)
        output_server = self._extract_server_output_token_count(
            responses, reasoning_server
        )

        # Warn once if server provided no usage information at all
        if (
            not self._warned_no_usage
            and input_token_count is None
            and output_server is None
            and reasoning_server is None
        ):
            self._warned_no_usage = True
            self.warning(
                "Server did not provide token usage information. Token count metrics will be unavailable. "
                "Verify that your API endpoint supports usage reporting (stream_options are automatically configured for OpenAI-compatible endpoints)."
            )

        # Client-side input tokenization
        input_local: int | None = None
        if (
            not self.disable_tokenization
            and not self._explicit_no_tokenize_input
            and (self.tokenize_input or input_token_count is None)
        ):
            try:
                input_local = await self.compute_input_token_count(request_record)
            except Exception as e:
                self.warning(f"Client-side input tokenization failed: {e}")

        # Client-side output/reasoning tokenization
        output_local: int | None = None
        reasoning_local: int | None = None
        need_local = (
            self.tokenize_output or output_server is None or reasoning_server is None
        )
        if not self.disable_tokenization and need_local:
            try:
                tokenizer = await self.get_tokenizer(request_record.model_name)
                output_texts, reasoning_texts = self._parse_output_and_reasoning_texts(
                    responses
                )
                if self.tokenize_output or output_server is None:
                    output_local = await self._compute_token_count(
                        tokenizer, output_texts
                    )
                if self.tokenize_output or reasoning_server is None:
                    reasoning_local = await self._compute_token_count(
                        tokenizer, reasoning_texts
                    )
            except Exception as e:
                self.warning(f"Client-side output/reasoning tokenization failed: {e}")

        return TokenCounts(
            input=input_token_count,
            input_local=input_local,
            output=output_server,
            output_local=output_local,
            reasoning=reasoning_server,
            reasoning_local=reasoning_local,
        )

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

    async def _compute_token_count(
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
        text = separator.join(texts)
        tokens = await asyncio.to_thread(tokenizer.encode, text)
        return len(tokens)

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
