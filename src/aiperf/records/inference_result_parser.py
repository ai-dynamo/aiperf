# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.hooks import on_init
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.models import (
    ErrorDetails,
    ExtractedPayload,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
    TurnMetadata,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    TokenCounts,
    ToolCallResponseData,
    find_last_non_empty_usage,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


# Failures from DatasetClientStoreProtocol.get_payload_bytes that the parser demotes to a warn + None record rather than escaping parse_request_record.  # noqa: E501
_DATASET_CLIENT_RECOVERABLE = (KeyError, IndexError, RuntimeError, OSError, MemoryError, ValueError)  # noqa: E501  # fmt: skip


# TODO: Should we create non-tokenizer based parsers?
class InferenceResultParser(CommunicationMixin):
    """InferenceResultParser is responsible for parsing the inference results."""

    def __init__(
        self,
        run: "BenchmarkRun",
    ) -> None:
        super().__init__(
            run=run,
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_run(run)
        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, self.model_endpoint.endpoint.type
        )
        self.endpoint = EndpointClass(model_endpoint=self.model_endpoint)
        endpoint_meta = plugins.get_endpoint_metadata(self.model_endpoint.endpoint.type)
        # Disable tokenization if the endpoint doesn't produce tokens and doesn't tokenize input, or
        # if the user config is set to use server token counts.
        self.disable_tokenization: bool = run.cfg.endpoint.use_server_token_count or (
            not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input
        )
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._turn_metadata_by_conversation: dict[str, tuple[TurnMetadata, ...]] = {}
        self.debug(
            lambda: f"Created endpoint for {self.model_endpoint.endpoint.type}, "
            f"class: {self.endpoint.__class__.__name__}",
        )

    @on_init
    async def _initialize(self) -> None:
        """Initialize inference result parser-specific components."""
        self.debug("Initializing inference result parser")

    def on_dataset_configured(
        self,
        *,
        turn_metadata_by_conversation: dict[str, tuple[TurnMetadata, ...]],
        dataset_client: DatasetClientStoreProtocol | None,
    ) -> None:
        """Receive the records-side dataset wiring from RecordProcessor.

        Stashes the conversation_id -> TurnMetadata sequence index and the
        optional PAYLOAD_BYTES mmap client. ``_extract_payload_inputs`` uses
        both to resolve payloads once per record.
        """
        self._turn_metadata_by_conversation = turn_metadata_by_conversation
        self._dataset_client = dataset_client

    async def configure(self) -> None:
        """Configure the tokenizers."""
        if self.disable_tokenization:
            self.info(
                "Tokenization is disabled for this endpoint, skipping tokenizer configuration"
            )
            return

        tokenizer_config = self.run.cfg.tokenizer
        self.info(
            f"Configuring tokenizers for inference result parser (resolve_alias: {tokenizer_config.should_resolve_alias})"
        )
        begin = time.perf_counter()
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
                tokenizer_config = self.run.cfg.tokenizer
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
        """Parse a wire ``RequestRecord`` into a ``ParsedResponseRecord``.

        Sole chokepoint for payload IO + JSON decode in the records
        pipeline: resolves bytes (inline or PAYLOAD_BYTES mmap), runs
        ``endpoint.extract_payload_inputs`` once, looks up ``TurnMetadata``
        once, and stashes the results on the returned record. Downstream
        consumers read structured fields -- no metric does IO or parses JSON.
        """
        mi = request_record.metric_inputs
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: f"Received inference results for credit '{mi.credit_num}' (id: {mi.x_request_id})"
            if mi
            else "Received inference results (no metric_inputs)",
        )

        # Single chokepoint -- see docstring.
        (
            payload_inputs,
            turn_metadata,
            payload_dict,
        ) = await self._extract_payload_inputs(request_record)

        # Make sure any invalid request records are converted to error records for combined processing.
        request_record.create_error_from_invalid()

        if request_record.has_error:
            # Even for error records, compute input token count if possible
            input_token_count = None
            if not self.disable_tokenization:
                # Suppress exceptions during token counting for error records to avoid masking the original error.
                # If token counting fails, we still return the error record with token_counts.input=None.
                with suppress(Exception):
                    input_token_count = await self._compute_input_token_count(
                        request_record, payload_inputs
                    )

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                token_counts=TokenCounts(
                    input=input_token_count,
                ),
                payload_inputs=payload_inputs,
                turn_metadata=turn_metadata,
                payload_dict=payload_dict,
            )

        else:
            try:
                raw_response_count = len(request_record.responses)
                record = await self.process_valid_record(
                    request_record,
                    payload_inputs=payload_inputs,
                    turn_metadata=turn_metadata,
                    payload_dict=payload_dict,
                )

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
                        payload_inputs=payload_inputs,
                        turn_metadata=turn_metadata,
                        payload_dict=payload_dict,
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
                input_token_count = None

                if not self.disable_tokenization:
                    # Suppress exceptions during token counting for error records to avoid masking the original error.
                    # If token counting fails, we still return the error record with token_counts.input=None.
                    with suppress(Exception):
                        input_token_count = await self._compute_input_token_count(
                            request_record, payload_inputs
                        )

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    token_counts=TokenCounts(
                        input=input_token_count,
                    ),
                    payload_inputs=payload_inputs,
                    turn_metadata=turn_metadata,
                    payload_dict=payload_dict,
                )

    async def process_valid_record(
        self,
        request_record: RequestRecord,
        *,
        payload_inputs: ExtractedPayload | None = None,
        turn_metadata: TurnMetadata | None = None,
        payload_dict: dict[str, Any] | None = None,
    ) -> ParsedResponseRecord:
        """Process a valid request record."""
        if request_record.model_name is None:
            self.warning(
                lambda: f"Model name is None, unable to process record: {request_record}"
            )
            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                payload_inputs=payload_inputs,
                turn_metadata=turn_metadata,
                payload_dict=payload_dict,
            )

        resp = self.endpoint.extract_response_data(request_record)

        # Compute token counts based on configuration
        if self.run.cfg.endpoint.use_server_token_count:
            token_counts = await self._compute_server_token_counts(resp)
        elif not self.disable_tokenization:
            token_counts = await self._compute_client_side_token_counts(
                request_record, resp, payload_inputs
            )
        else:
            token_counts = TokenCounts()

        return ParsedResponseRecord(
            request=request_record,
            responses=resp,
            token_counts=token_counts,
            payload_inputs=payload_inputs,
            turn_metadata=turn_metadata,
            payload_dict=payload_dict,
        )

    async def _extract_payload_inputs(
        self, request_record: RequestRecord
    ) -> tuple[ExtractedPayload | None, TurnMetadata | None, dict[str, Any] | None]:
        """THE single chokepoint for payload IO + JSON decode. Tokenization,
        raw export, and every metric read stashed results off
        ``ParsedResponseRecord`` -- they do not re-resolve."""
        mi = request_record.metric_inputs
        if mi is None:
            return None, None, None

        turns = self._turn_metadata_by_conversation.get(mi.conversation_id)
        tmeta = None
        if turns is not None and 0 <= mi.turn_index < len(turns):
            tmeta = turns[mi.turn_index]

        raw = mi.payload_bytes_or_none
        if raw is None and self._dataset_client is not None:
            try:
                raw = await self._dataset_client.get_payload_bytes(
                    mi.conversation_id, mi.turn_index
                )
            except _DATASET_CLIENT_RECOVERABLE as e:
                self.warning(
                    f"mmap payload resolution failed for {mi.conversation_id}[{mi.turn_index}]: {e!r}"
                )  # noqa: E501
                return None, tmeta, None
        if raw is None:
            return None, tmeta, None
        try:
            payload = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            self.warning(
                f"payload_bytes is not valid JSON for {mi.conversation_id}[{mi.turn_index}]: {e!r}"
            )  # noqa: E501
            return None, tmeta, None
        if not isinstance(payload, dict):
            self.warning(
                f"payload_bytes parsed but is not a JSON object (got {type(payload).__name__}) for {mi.conversation_id}[{mi.turn_index}]"
            )  # noqa: E501
            return None, tmeta, None
        return self.endpoint.extract_payload_inputs(payload), tmeta, payload

    async def _compute_input_token_count(
        self,
        request_record: RequestRecord,
        payload_inputs: ExtractedPayload | None,
    ) -> int | None:
        if payload_inputs is None:
            return None

        input_token_count = payload_inputs.pretokenised_token_count or None
        if payload_inputs.texts:
            tokenizer = await self.get_tokenizer(request_record.model_name)
            text_token_count = await self._compute_token_count(
                tokenizer, payload_inputs.texts, separator=" "
            )
            if text_token_count is not None:
                input_token_count = (input_token_count or 0) + text_token_count

        return input_token_count

    async def _compute_server_token_counts(
        self, responses: list[ParsedResponse]
    ) -> TokenCounts:
        """Compute token counts using server-provided usage fields.

        Walks `responses` ONCE to find the last chunk with usage and reads
        all token counts from that single Usage. This guarantees the input,
        reasoning, and output counts are mutually consistent (all from the
        same chunk), and it avoids three redundant walks of the same list.

        Args:
            responses: List of parsed responses from the server

        Returns:
            TokenCounts populated with server-reported values. All fields
            are None if no chunk had usage at all.
        """
        usage = find_last_non_empty_usage(responses)
        if usage is None:
            input_token_count = None
            reasoning_token_count = None
            output_token_count = None
        else:
            input_token_count = usage.prompt_tokens
            reasoning_token_count = usage.reasoning_tokens
            output_token_count = self._server_output_minus_reasoning(
                usage.completion_tokens, reasoning_token_count
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

    def _server_output_minus_reasoning(
        self,
        completion_tokens: int | None,
        reasoning_token_count: int | None,
    ) -> int | None:
        """Return server-reported output tokens with reasoning subtracted out.

        The server's `completion_tokens` includes both reasoning and output;
        we subtract reasoning_tokens to match the client-side semantic of
        "output tokens" (text the user sees). Clamps to 0 if the subtraction
        would go negative (server reported inconsistent counts).
        """
        if completion_tokens is None:
            return None
        reasoning = reasoning_token_count or 0
        result = completion_tokens - reasoning
        if result < 0:
            self.warning(
                f"Server reported inconsistent token counts: completion_tokens={completion_tokens}, "
                f"reasoning_tokens={reasoning}. Clamping output tokens to 0."
            )
            return 0
        return result

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
            elif isinstance(response.data, ToolCallResponseData):
                output_texts.append(response.data.tool_call_text)
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

    async def _compute_client_side_token_counts(
        self,
        request_record: RequestRecord,
        responses: list[ParsedResponse],
        payload_inputs: ExtractedPayload | None,
    ) -> TokenCounts:
        """Compute token counts using client-side tokenization.

        ``payload_inputs`` is the already-extracted ExtractedPayload from
        ``_extract_payload_inputs``; passed in to avoid re-resolution.
        """
        input_token_count = await self._compute_input_token_count(
            request_record, payload_inputs
        )

        tokenizer = await self.get_tokenizer(request_record.model_name)
        output_texts, reasoning_texts = self._parse_output_and_reasoning_texts(
            responses
        )
        output_token_count = await self._compute_token_count(tokenizer, output_texts)
        reasoning_token_count = await self._compute_token_count(
            tokenizer, reasoning_texts
        )

        return TokenCounts(
            input=input_token_count,
            reasoning=reasoning_token_count,
            output=output_token_count,
        )
