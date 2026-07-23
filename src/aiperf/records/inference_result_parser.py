# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import suppress
from typing import TYPE_CHECKING

import orjson

from aiperf.common.enums import ExportLevel
from aiperf.common.hooks import on_init
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.models import (
    ErrorDetails,
    ExtractedPayload,
    MediaCounts,
    ParsedResponse,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    TokenCounts,
    ToolCallResponseData,
    find_last_non_empty_usage,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.payload_retention import resolve_disable_tokenization

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


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
        # Single source of truth shared with the worker's payload-strip
        # auto-detector (resolve_strip_record_payload_bytes -> this helper), so
        # the parser's ISL path and the worker's payload retention can never
        # disagree about whether client-side tokenization runs.
        self.disable_tokenization: bool = resolve_disable_tokenization(
            run.cfg, endpoint_meta
        )
        self.debug(
            lambda: (
                f"Created endpoint for {self.model_endpoint.endpoint.type}, "
                f"class: {self.endpoint.__class__.__name__}"
            ),
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
        self,
        request_record: RequestRecord,
        *,
        parsed_responses: list[ParsedResponse] | None = None,
        raw_response_count: int | None = None,
        responses_validated: bool = False,
    ) -> ParsedResponseRecord:
        """Handle an inference results message.

        Single-pass payload extraction: decode ``payload_bytes`` once, run it
        through the endpoint's ``extract_payload_inputs`` hook to yield
        tokenisable text + multimodal counts, then feed both into downstream
        scoring (ISL tokeniser + ``MediaCounts`` on the returned
        ``ParsedResponseRecord``). Consumers of the parsed record never re-parse
        ``payload_bytes`` -- the metric classes read ``token_counts`` and
        ``media_counts`` directly.
        """
        request_info = request_record.request_info
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: (
                f"Received inference results for credit '{request_info.credit_num}' (id: {request_info.x_request_id})"
                if request_info
                else "Received inference results (no request_info)"
            ),
        )

        # Make sure any invalid request records are converted to error records for combined processing.
        if not responses_validated:
            request_record.create_error_from_invalid()

        # One payload decode + walk per record, shared by the ISL tokeniser and
        # the MediaCounts builder. Both valid and error records go through this.
        inputs = self._extract_payload_inputs_for_record(request_record)
        media_counts = (
            MediaCounts(
                images=inputs.image_count,
                audios=inputs.audio_count,
                videos=inputs.video_count,
            )
            if inputs is not None
            else MediaCounts()
        )

        if request_record.has_error:
            # Even for error records, compute input token count if possible
            input_token_count = None
            if not self.disable_tokenization:
                # Suppress exceptions during token counting for error records to avoid masking the original error.
                # If token counting fails, we still return the error record with token_counts.input=None.
                with suppress(Exception):
                    input_token_count = await self.compute_input_token_count(
                        request_record, inputs=inputs
                    )

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                token_counts=TokenCounts(
                    input=input_token_count,
                ),
                media_counts=media_counts,
            )

        else:
            try:
                if raw_response_count is None:
                    raw_response_count = len(request_record.responses)
                record = await self.process_valid_record(
                    request_record,
                    inputs=inputs,
                    media_counts=media_counts,
                    parsed_responses=parsed_responses,
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
                        media_counts=media_counts,
                    )
                else:
                    # Success path: valid record with no errors
                    self.debug(
                        lambda: (
                            f"Received {raw_response_count} response packet(s), token counts: {record.token_counts}"
                        )
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
                            request_record, inputs=inputs
                        )

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    token_counts=TokenCounts(
                        input=input_token_count,
                    ),
                    media_counts=media_counts,
                )

    def _extract_payload_inputs_for_record(
        self, request_record: RequestRecord
    ) -> ExtractedPayload | None:
        """Decode ``request_info.payload_bytes`` once and hand it to the
        endpoint's single-pass extractor.

        ``payload_bytes`` is the canonical wire body, always JSON-encoded by
        ``inference_client`` (even for multipart endpoints, whose dict is
        ``orjson.dumps``-ed onto ``payload_bytes`` before the FormData is built).
        Returns ``None`` when the payload is absent or not a JSON object --
        callers then treat ISL as unavailable and use zero-valued
        ``MediaCounts``.
        """
        request_info = request_record.request_info
        if request_info is None or not request_info.payload_bytes:
            return None
        try:
            payload = orjson.loads(request_info.payload_bytes)
        except orjson.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return self.endpoint.extract_payload_inputs(payload)

    async def process_valid_record(
        self,
        request_record: RequestRecord,
        *,
        inputs: ExtractedPayload | None = None,
        media_counts: MediaCounts | None = None,
        parsed_responses: list[ParsedResponse] | None = None,
    ) -> ParsedResponseRecord:
        """Process a valid request record.

        ``inputs`` and ``media_counts`` are passed through from
        ``parse_request_record`` (single payload decode shared with the token
        counter). When called directly (tests, other entry points) both default
        to ``None`` and the tokeniser falls back to a per-call decode.
        ``parsed_responses`` bypasses endpoint extraction when the worker
        supplied an internal IPC representation.
        """
        if request_record.model_name is None:
            self.warning(
                lambda: (
                    f"Model name is None, unable to process record: {request_record}"
                )
            )
            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                media_counts=media_counts or MediaCounts(),
            )

        resp = (
            parsed_responses
            if parsed_responses is not None
            else self.endpoint.extract_response_data(request_record)
        )

        # Free the raw responses list after extraction.
        # Skip when RAW export needs the original responses for serialization.
        if self.run.cfg.artifacts.export_level != ExportLevel.RAW:
            request_record.responses = None

        # Compute token counts based on configuration
        if self.run.cfg.endpoint.use_server_token_count:
            token_counts = await self._compute_server_token_counts(resp)
        elif not self.disable_tokenization:
            token_counts = await self._compute_client_side_token_counts(
                request_record, resp, inputs=inputs
            )
        else:
            token_counts = TokenCounts()

        return ParsedResponseRecord(
            request=request_record,
            responses=resp,
            token_counts=token_counts,
            media_counts=media_counts or MediaCounts(),
        )

    async def compute_input_token_count(
        self,
        request_record: RequestRecord,
        *,
        inputs: ExtractedPayload | None = None,
    ) -> int | None:
        """Compute the number of input (prompt) tokens for a request record.

        Source of truth is ``request_info.payload_bytes`` -- the exact JSON the
        endpoint sent on the wire -- decoded once and walked by
        ``extract_payload_inputs`` into tokenisable ``texts`` (plus a chat-shape
        ``messages`` view and any ``pretokenised_token_count``). ``turns`` are
        never read here: on the payload-bytes fast path they are a content-free
        stub, and the canonical body is the bytes.

        ``system_message`` / ``user_context_message`` are NOT added here -- the
        endpoint's ``format_payload`` already inlined them into ``payload_bytes``,
        so re-adding would double-count.

        When ``--apply-chat-template`` is set AND the payload carries a chat
        ``messages`` array AND the HF tokenizer has a chat template, the templated
        count (role/header wrapping + generation prompt) is returned; otherwise
        the bare ``texts`` are encoded with a space separator. Any
        ``pretokenised_token_count`` (e.g. token-id embeddings input) is added on
        top. Returns ``None`` when no ``payload_bytes`` is available or the
        endpoint reports nothing tokenisable.
        """
        if inputs is None:
            inputs = self._extract_payload_inputs_for_record(request_record)

        if inputs is None:
            if not (
                request_record.request_info
                and request_record.request_info.payload_bytes
            ):
                self.warning(
                    "payload_bytes not set on request_info; cannot compute "
                    "input token count"
                )
            return None

        pretokenised = inputs.pretokenised_token_count

        tokenizer = None
        # Chat-template path: count role/template wrapping when the user opted in
        # and the payload is chat-shaped. Runs even when ``texts`` is empty (e.g.
        # an image-only message still contributes role/header tokens).
        tokenizer_cfg = self.run.cfg.tokenizer
        if (
            inputs.messages
            and tokenizer_cfg is not None
            and tokenizer_cfg.apply_chat_template
        ):
            tokenizer = await self.get_tokenizer(request_record.model_name)
            templated = await self._compute_chat_template_token_count(
                tokenizer, inputs.messages
            )
            if templated is not None:
                # Tool text (replayed tool_calls / function_call items and
                # tools schemas) is absent from the role/content ``messages``
                # view, so it must be tokenised on top of the templated count.
                tool_count = (
                    await self._compute_token_count(
                        tokenizer, inputs.tool_texts, separator=" "
                    )
                    or 0
                )
                return templated + tool_count + pretokenised

        # Bare-text path: join the extracted texts with a space separator.
        if inputs.texts:
            if tokenizer is None:
                tokenizer = await self.get_tokenizer(request_record.model_name)
            text_count = await self._compute_token_count(
                tokenizer, inputs.texts, separator=" "
            )
            if text_count is not None:
                return text_count + pretokenised

        # Pure pre-tokenised input (e.g. token-id embeddings) carries no text.
        return pretokenised if pretokenised > 0 else None

    async def _compute_chat_template_token_count(
        self,
        tokenizer: Tokenizer,
        messages: list[dict[str, str]],
    ) -> int | None:
        """Tokenize ``messages`` through the HF tokenizer's chat template.

        ``messages`` is the chat-shape view produced by ``extract_payload_inputs``
        from the wire ``payload_bytes`` -- so the count reflects exactly what the
        server tokenizes (role/header wrapping + generation prompt) rather than a
        re-render of ``turns``. ``add_generation_prompt=True`` mirrors the server
        begin-generation suffix. Returns ``None`` (caller falls back to bare-text
        encoding) when the tokenizer has no chat template configured, is not
        HF-backed, or rendering fails.
        """
        inner = getattr(tokenizer, "_tokenizer", None)
        apply = getattr(inner, "apply_chat_template", None)
        if apply is None or not messages:
            return None
        # Short-circuit HF tokenizers that explicitly carry no chat template.
        if getattr(inner, "chat_template", "_unset") is None:
            return None
        try:
            tokens = await asyncio.to_thread(
                apply,
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; fall back to bare encode
            self.debug(
                lambda exc=exc: f"Chat-template tokenization unavailable, "
                f"falling back to bare-text encode: {exc!r}"
            )
            return None
        if not isinstance(tokens, list):
            return None
        return len(tokens)

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
        *,
        inputs: ExtractedPayload | None = None,
    ) -> TokenCounts:
        """Compute token counts using client-side tokenization.

        Args:
            request_record: The request record containing input data
            responses: List of parsed responses from the server
            inputs: Pre-extracted payload inputs shared with the media-counts
                path in ``parse_request_record`` (single payload decode).
                ``None`` triggers a fresh decode inside
                ``compute_input_token_count``.

        Returns:
            TokenCounts populated with client-side tokenized values
        """
        input_token_count = await self.compute_input_token_count(
            request_record, inputs=inputs
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
