# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from contextlib import suppress

import orjson

from aiperf.common.config import ServiceConfig, UserConfig
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
from aiperf.common.scenario import is_context_overflow_response
from aiperf.common.tokenizer import Tokenizer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.payload_retention import resolve_disable_tokenization


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
        # if the user config is set to use server token counts. Shared with the
        # worker's payload-strip auto-detection so both agree on whether
        # client-side tokenization reads payload_bytes.
        self.disable_tokenization: bool = resolve_disable_tokenization(
            user_config, endpoint_meta
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

        tokenizer_config = self.user_config.tokenizer
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
        """Handle an inference results message.

        Single-pass payload extraction: decode ``payload_bytes`` once, run
        it through the endpoint's ``extract_payload_inputs`` hook to yield
        tokenisable text + multimodal counts, then feed both into
        downstream scoring (ISL tokeniser + ``MediaCounts`` on the returned
        ``ParsedResponseRecord``). Consumers of the parsed record never
        re-parse ``payload_bytes`` — the metric classes read ``token_counts``
        and ``media_counts`` directly.
        """
        request_info = request_record.request_info
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: f"Received inference results for credit '{request_info.credit_num}' (id: {request_info.x_request_id})"
            if request_info
            else "Received inference results (no request_info)",
        )

        # Make sure any invalid request records are converted to error records for combined processing.
        request_record.create_error_from_invalid()

        # Classify context-overflow errors per InferenceX AgentX RFC §7.
        # Runs unconditionally (cheap substring scan, allowlist-driven) so the
        # ``context_overflow_count`` metric can aggregate even outside scenario
        # mode -- useful for diagnostics. The boolean is consumed by
        # ``ContextOverflowCountMetric`` via ``record.request.context_overflow``.
        if request_record.has_error and request_record.error is not None:
            try:
                request_record.context_overflow = is_context_overflow_response(
                    body=request_record.error.message,
                )
            except Exception:
                # Detection is best-effort -- never let it surface as a parse error.
                request_record.context_overflow = False

        # One payload decode + walk per record. Shared downstream by the
        # tokeniser and the MediaCounts builder; both valid and error
        # records go through this.
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
                token_counts=TokenCounts(input=input_token_count),
                media_counts=media_counts,
            )

        else:
            try:
                raw_response_count = len(request_record.responses)
                record = await self.process_valid_record(
                    request_record, inputs=inputs, media_counts=media_counts
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
                        input_token_count = await self.compute_input_token_count(
                            request_record, inputs=inputs
                        )

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    token_counts=TokenCounts(input=input_token_count),
                    media_counts=media_counts,
                )

    def _extract_payload_inputs_for_record(
        self, request_record: RequestRecord
    ) -> ExtractedPayload | None:
        """Decode ``request_info.payload_bytes`` once and hand it to the
        endpoint's single-pass extractor.

        Returns ``None`` when the payload is absent or non-decodable —
        callers then know to fall back to ``request_info.turns`` for
        tokenisation and to zero-valued ``MediaCounts`` for metrics.
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
    ) -> ParsedResponseRecord:
        """Process a valid request record.

        ``inputs`` and ``media_counts`` are passed through from
        ``parse_request_record`` which decoded ``payload_bytes`` once and
        shared the result with the token counter. When called directly
        (tests, other entry points) both default to ``None`` and the
        tokeniser falls back to per-call decoding.
        """
        if request_record.model_name is None:
            self.warning(
                lambda: f"Model name is None, unable to process record: {request_record}"
            )
            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                media_counts=media_counts or MediaCounts(),
            )

        resp = self.endpoint.extract_response_data(request_record)

        # Free the raw responses list after extraction.
        # Skip when RAW export needs the original responses for serialization.
        if self.user_config.output.export_level != ExportLevel.RAW:
            request_record.responses = None

        # Compute token counts based on configuration
        if self.user_config.endpoint.use_server_token_count:
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
        """Compute the number of tokens in the input for a given request record.

        Source of truth is ``request_info.payload_bytes`` — the exact JSON
        bytes that went on the wire to the server, stashed by
        ``inference_client._send_request_to_transport``. Each endpoint owns
        its own payload shape and implements ``extract_payload_inputs`` to
        pull tokenisable text out; see
        ``BaseEndpoint.extract_payload_inputs``.

        ``system_message`` and ``user_context_message`` from ``RequestInfo``
        are NOT tokenised additively — the endpoint's ``format_payload``
        already inlines them into ``payload_bytes`` before the transport
        call, and tokenising them again would double-count. Tests that
        exercise those scalars should regenerate ``payload_bytes`` via the
        endpoint after mutating the fields.

        ``inputs`` is the already-extracted payload result from
        ``parse_request_record`` (single-pass sharing with the
        ``MediaCounts`` path). Callers that don't pre-extract trigger a
        fresh decode here.

        When the payload carries a chat-shape ``messages`` list AND the
        underlying HF tokenizer exposes ``apply_chat_template`` with a
        template configured AND the user passed ``--apply-chat-template``,
        the templated token count (role/header wrapping + assistant-prompt
        suffix included) is returned. Other payload shapes — completions,
        embeddings, rankings, HF inputs — and runs without the opt-in
        flag fall back to bare text encoding with a space separator.

        Returns ``None`` when no ``payload_bytes`` is available (pre-
        transport error path) or the endpoint reports no extractable text.
        Callers treat ``None`` as "metric unavailable" and skip.
        """
        if inputs is None:
            inputs = self._extract_payload_inputs_for_record(request_record)

        if inputs is None or not inputs.texts:
            if not (
                request_record.request_info
                and request_record.request_info.payload_bytes
            ):
                self.warning(
                    "payload_bytes not set on request_info; cannot compute "
                    "input token count"
                )
            return None

        tokenizer = await self.get_tokenizer(request_record.model_name)

        # Prefer chat-template tokenization when both are available AND
        # the user opted in via ``--apply-chat-template``: the payload
        # carries a chat-shape ``messages`` list AND the underlying HF
        # tokenizer has a chat template configured. This makes the
        # reported ISL match the wrapped wire payload (template/role
        # tokens + cache-bust marker + bare prompt) instead of just the
        # bare text. Without the flag, ISL reports the bare prompt token
        # count -- matching the user's ``--isl`` value rather than what
        # the model actually sees on the wire.
        if inputs.messages and self.user_config.tokenizer.apply_chat_template:
            templated = await self._compute_chat_template_token_count(
                tokenizer, inputs.messages
            )
            if templated is not None:
                return templated

        # NOTE: We combine all the prompt texts with a space separator to create a single prompt string.
        # This will get us the most accurate token count for the prompt by avoiding any potential
        # boundary issues that could occur if we were to tokenize each text individually.
        return await self._compute_token_count(tokenizer, inputs.texts, separator=" ")

    async def _compute_chat_template_token_count(
        self,
        tokenizer: Tokenizer,
        messages: list[dict[str, str]],
    ) -> int | None:
        """Tokenize ``messages`` through the HF tokenizer's chat template.

        Returns the templated token count, or ``None`` when chat-template
        tokenization is unavailable or fails (model has no template
        configured, tokenizer is not HF-backed, etc.). Callers fall back
        to bare text encoding in that case.

        ``add_generation_prompt=True`` mirrors the actual server-side
        behavior: the assistant-prompt suffix is what the model sees on
        the wire when it begins generating.
        """
        inner = getattr(tokenizer, "_tokenizer", None)
        apply = getattr(inner, "apply_chat_template", None)
        if apply is None or not messages:
            return None
        # Short-circuit for HF tokenizers that explicitly carry no chat
        # template (``chat_template is None``). Saves a per-record raise +
        # f-string format on the bare-text fallback path when the user
        # is benchmarking a base/un-templated model.
        if getattr(inner, "chat_template", "_unset") is None:
            return None
        try:
            tokens = await asyncio.to_thread(
                apply,
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
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
                output_texts.append(response.data.text)
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
            inputs: Pre-extracted payload inputs shared with the
                media-counts path in ``parse_request_record`` (single-pass
                decode). ``None`` triggers a fresh decode inside
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
