# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from aiperf.common.enums import MediaType
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    BaseResponseData,
    EmbeddingResponseData,
    ExtractedPayload,
    InferenceServerResponse,
    Media,
    ModelEndpointInfo,
    ParsedResponse,
    RankingsResponseData,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    Text,
    TextResponseData,
    Turn,
)
from aiperf.common.types import RequestOutputT


class BaseEndpoint(AIPerfLoggerMixin, ABC):
    """Base for all endpoints.

    Endpoints handle API-specific formatting and parsing.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint headers (auth + user custom). Override to customize."""
        cfg = self.model_endpoint.endpoint
        headers = dict(cfg.headers) if cfg.headers else {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        return headers

    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint URL query params (e.g., api-version). Override to customize."""
        cfg = self.model_endpoint.endpoint
        return dict(cfg.url_params) if cfg.url_params else {}

    @abstractmethod
    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format request payload from RequestInfo.

        Uses request_info.turns[0] as the turn data (currently hardcoded to first turn).
        """

    @abstractmethod
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        """Extract parsed data from record.

        Args:
            record: Request record containing responses to parse

        Returns:
            List of successfully parsed responses
        """
        return [
            parsed
            for response in record.responses
            if (parsed := self.parse_response(response))
        ]

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Build a Turn representing the assistant response for context replay.

        Used by the worker after each request to capture the model's reply
        so subsequent turns in the same session — and FORK-mode DAG children
        that inherit the parent's history — see what the model actually said.

        The default implementation captures plain text (and the ``content``
        field of any ``ReasoningResponseData``); ``reasoning`` itself is
        deliberately dropped, since most chat templates do not round-trip it.
        Endpoints whose responses carry structured fields beyond plain text
        (e.g. chat ``tool_calls`` / ``function_call``) should override this
        to preserve those fields verbatim by returning a Turn whose
        ``raw_messages`` re-renders as the same assistant message — that way
        ``build_messages`` extends them back into the wire body unchanged.

        Returns ``None`` when the record has no replayable assistant content
        (error response, empty body, etc.).
        """
        output_texts: list[str] = []
        for response in self.extract_response_data(record):
            if not response.data:
                continue
            if isinstance(response.data, ReasoningResponseData):
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())
        resp_text = "".join(output_texts)
        if not resp_text:
            return None
        return Turn(role="assistant", texts=[Text(contents=[resp_text])])

    # -------------------------------------------------------------------------
    # Generic turn→messages building
    # -------------------------------------------------------------------------
    #
    # AIPerf's chat-like endpoints (``openai_chat``, ``openai_responses``, and
    # any plugin that emits a role/content message array) share a fixed
    # flatten-and-merge skeleton:
    #
    #   1. iterate ``request_info.turns`` in order
    #   2. if the turn carries ``raw_messages`` (author-provided OpenAI-shape
    #      entries — ``dag_jsonl``, ``mooncake_trace`` payload mode, or a
    #      captured live assistant turn), splice them in verbatim
    #   3. otherwise synthesise a single role/content message from the
    #      structured ``Turn`` fields (``role``, ``texts``, ``images``,
    #      ``audios``, ``videos``).
    #
    # Only step 3 depends on the endpoint's wire shape — OpenAI chat uses
    # ``{"type": "text"}`` / ``{"type": "image_url"}`` parts, the Responses
    # API uses ``{"type": "input_text"}`` / ``{"type": "input_image"}``,
    # future plugins may use something else entirely. The iteration and
    # merge logic is universal, so it lives here; the part-rendering hooks
    # below are what endpoint subclasses override.
    #
    # Endpoints that don't emit a message array (``openai_completions``,
    # ``openai_embeddings``, rankings, image/video generation, raw payload
    # replay) simply never call ``build_messages`` — they format their
    # payload directly.

    DEFAULT_TURN_ROLE: str = "user"
    """Default role for a synthesised turn message when ``turn.role`` is None."""

    def build_messages(self, turns: list[Turn]) -> list[dict[str, Any]]:
        """Flatten ``turns`` into a wire-ready role/content message array.

        Turns carrying ``raw_messages`` extend the array verbatim; every
        other turn renders through ``_render_turn_message``. The result is
        ``payload["messages"]`` for chat endpoints, ``payload["input"]``
        for the Responses API, and any similar shape for plugins.

        Does NOT prepend shared ``system_message`` or
        ``user_context_message`` — those live on ``RequestInfo`` and are
        placed wherever the endpoint's wire contract dictates (e.g. a
        leading ``system`` role in chat; a top-level ``instructions`` field
        in Responses). Callers handle that in their ``format_payload``.
        """
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if turn.raw_messages is not None:
                messages.extend(turn.raw_messages)
                continue
            messages.append(self._render_turn_message(turn))
        return messages

    def _render_turn_message(self, turn: Turn) -> dict[str, Any]:
        """Render a single synthetic turn as a role/content message.

        Default emits chat-shape ``{"role": ..., "content": ...}``.
        Endpoints with a different envelope (e.g. Responses input items with
        additional fields) override this.
        """
        return {
            "role": turn.role or self.DEFAULT_TURN_ROLE,
            "content": self._render_turn_content(turn),
        }

    def _render_turn_content(self, turn: Turn) -> str | list[dict[str, Any]]:
        """Render the ``content`` side of a synthetic turn message.

        Single-text turns return the raw string (OpenAI Dynamo compatibility
        hotfix — some servers reject list-of-parts content when only one
        text is present). Multi-modal or multi-text turns return a list of
        content parts built via the ``_render_*_part`` hooks.

        Endpoints override the ``_render_*_part`` hooks to change content-
        part type names (e.g. ``text`` → ``input_text`` for Responses API).
        """
        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and not turn.images
            and not turn.audios
            and not turn.videos
        ):
            return turn.texts[0].contents[0] or ""

        parts: list[dict[str, Any]] = []
        for text in turn.texts:
            for content in text.contents:
                if not content:
                    continue
                parts.append(self._render_text_part(content))
        for image in turn.images:
            for content in image.contents:
                if not content:
                    continue
                parts.append(self._render_image_part(content))
        for audio in turn.audios:
            for content in audio.contents:
                if not content:
                    continue
                parts.append(self._render_audio_part(content))
        for video in turn.videos:
            for content in video.contents:
                if not content:
                    continue
                parts.append(self._render_video_part(content))
        return parts

    # --- Content-part hooks: override per endpoint to change type names ------

    def _render_text_part(self, text: str) -> dict[str, Any]:
        """Render one text content part. Default: OpenAI chat shape."""
        return {"type": "text", "text": text}

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Render one image content part. Default: OpenAI chat shape."""
        return {"type": "image_url", "image_url": {"url": url_or_data_uri}}

    def _render_audio_part(self, format_and_b64: str) -> dict[str, Any]:
        """Render one audio content part. Default: OpenAI chat shape.

        ``format_and_b64`` is the comma-joined ``"<format>,<base64 data>"``
        string AIPerf uses to carry audio payloads through Turn media lists.
        """
        if "," not in format_and_b64:
            raise ValueError("Audio content must be in the format 'format,b64_audio'.")
        fmt, b64 = format_and_b64.split(",", 1)
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}

    def _render_video_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Render one video content part. Default: OpenAI chat shape."""
        return {"type": "video_url", "video_url": {"url": url_or_data_uri}}

    # --- Payload → inputs extraction (single-pass read side) ----------------

    #: Content-part ``type`` values keyed by ``MediaType`` that this endpoint
    #: emits. ``extract_payload_inputs`` uses this map to dispatch each part
    #: it encounters to text / image / audio / video accumulators. Endpoints
    #: override by assigning a different dict (cheapest) or by subclassing
    #: ``extract_payload_inputs`` directly.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"text"},
        MediaType.IMAGE: {"image_url"},
        MediaType.AUDIO: {"input_audio"},
        MediaType.VIDEO: {"video_url"},
    }

    def extract_payload_inputs(self, payload: dict) -> ExtractedPayload:
        """Single-pass extraction of tokenisable text + media counts from a
        wire-ready payload.

        One ``orjson.loads`` plus one O(n) walk yields everything downstream
        consumes (ISL tokenisation via ``texts``; ``image_throughput`` /
        ``image_latency`` / ``num_images`` via ``image_count``; future
        audio/video metrics via the remaining counts).

        Default implementation covers every payload shape AIPerf emits
        today:

        - chat / Responses ``messages`` or ``input`` items arrays
          (dispatch each content part against ``PART_TYPES``)
        - completions ``prompt`` (string or list of strings)
        - embeddings ``input`` (string or list of strings)
        - rankings ``query`` + ``passages``
        - HuggingFace ``inputs``

        Endpoints with a non-standard payload shape (e.g. Responses API's
        top-level ``instructions``) override this method; endpoints that
        share a shape but emit different part type names just set
        ``PART_TYPES`` and inherit the walk.
        """
        result = ExtractedPayload()
        # Reverse index the part-type set: ``{"text": MediaType.TEXT,
        # "image_url": MediaType.IMAGE, ...}``. Built per-call — the map is
        # small and per-part lookup is O(1).
        type_to_media: dict[str, MediaType] = {
            type_name: media_type
            for media_type, type_names in self.PART_TYPES.items()
            for type_name in type_names
        }

        found_items_shape = False
        for items_field in ("messages", "input"):
            items = payload.get(items_field)
            if not isinstance(items, list) or not items:
                continue
            # Disambiguate Responses/chat message arrays from embeddings
            # ``input: [str, ...]``: the former always carries dicts with
            # a ``role`` key OR a Responses-style item ``type`` key
            # (``function_call``, ``function_call_output``, ``message``,
            # ``reasoning``, ...). Pure embedding strings have neither.
            if not any(
                isinstance(i, dict) and ("role" in i or "type" in i) for i in items
            ):
                continue
            found_items_shape = True
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Chat-shape ``content`` (string or content-parts array).
                content = item.get("content")
                if isinstance(content, str):
                    result.texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        media = type_to_media.get(part.get("type"))
                        if media is MediaType.TEXT:
                            text = part.get("text")
                            if isinstance(text, str):
                                result.texts.append(text)
                        elif media is MediaType.IMAGE:
                            result.image_count += 1
                        elif media is MediaType.AUDIO:
                            result.audio_count += 1
                        elif media is MediaType.VIDEO:
                            result.video_count += 1
                # Chat-shape assistant message replaying earlier
                # ``tool_calls``: each call's ``function.name`` and
                # ``function.arguments`` are tokens the model previously
                # generated, and the server tokenises them on input
                # replay too. Without this the ISL of agent-history
                # replays is undercounted by everything in those calls.
                tool_calls = item.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        if not isinstance(fn, dict):
                            continue
                        for key in ("name", "arguments"):
                            value = fn.get(key)
                            if isinstance(value, str) and value:
                                result.texts.append(value)
                # Responses-shape replayed items ``function_call`` and
                # ``function_call_output``: same logic — the model
                # generated those tokens, the server retokenises them on
                # replay.
                item_type = item.get("type")
                if item_type == "function_call":
                    for key in ("name", "arguments"):
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            result.texts.append(value)
                elif item_type == "function_call_output":
                    output_text = item.get("output")
                    if isinstance(output_text, str) and output_text:
                        result.texts.append(output_text)

        # Top-level ``tools`` (chat) or ``tools`` (Responses) — schema
        # text the server tokenises into the prefix of every request.
        # Walks ``function.name``, ``function.description``,
        # ``function.parameters`` (serialised), plus any future tool
        # shape's ``name``/``description``. Counts toward ISL.
        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                fn = tool.get("function") or {}
                # Chat-shape: nested under ``function``. Responses-shape:
                # fields can sit at the top level. Walk both.
                for source in (fn, tool):
                    if not isinstance(source, dict):
                        continue
                    for key in ("name", "description"):
                        value = source.get(key)
                        if isinstance(value, str) and value:
                            result.texts.append(value)
                    parameters = source.get("parameters")
                    if isinstance(parameters, dict):
                        # Serialise once; the tokeniser sees the same
                        # JSON the server would when prepending the tool
                        # schema to the prompt.
                        try:
                            import orjson

                            result.texts.append(orjson.dumps(parameters).decode())
                        except (ImportError, TypeError):
                            pass

        if found_items_shape:
            return result

        # Flat-field fallback shapes (completions / embeddings / rankings /
        # HuggingFace). Only consulted when no items-array was found so
        # embeddings ``input: [str, ...]`` doesn't get double-counted with
        # the chat/Responses walk above. Each shape early-returns so a
        # plugin that accidentally emitted two shapes doesn't silently
        # double-count.
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            result.texts.append(prompt)
            return result
        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            result.texts.extend(prompt)
            return result

        inp = payload.get("input")
        if isinstance(inp, str):
            result.texts.append(inp)
            return result
        if isinstance(inp, list) and all(isinstance(s, str) for s in inp):
            result.texts.extend(inp)
            return result

        query = payload.get("query")
        passages = payload.get("passages")
        if isinstance(query, str) and isinstance(passages, list):
            result.texts.append(query)
            for p in passages:
                if isinstance(p, str):
                    result.texts.append(p)
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    result.texts.append(p["text"])
            return result

        hf = payload.get("inputs")
        if isinstance(hf, str):
            result.texts.append(hf)
            return result

        return result

    @staticmethod
    def make_text_response_data(text: str | None) -> TextResponseData | None:
        """Make a TextResponseData object from a string or return None if the text is empty."""
        return TextResponseData(text=text) if text else None

    def auto_detect_and_extract(self, json_obj: dict) -> BaseResponseData | None:
        """Optional utility: Auto-detect response type and extract relevant data.

        Tries to extract data in this order: embeddings, rankings, text.
        Endpoints can use this as a fallback or for flexible response handling.

        Args:
            json_obj: JSON response object

        Returns:
            Typed response data object or None if not found
        """
        if data := self.try_extract_embeddings(json_obj):
            return data

        if data := self.try_extract_rankings(json_obj):
            return data

        if data := self.try_extract_text(json_obj):
            return data

        return None

    def try_extract_embeddings(self, json_obj: dict) -> EmbeddingResponseData | None:
        """Optional utility: Try to extract embeddings from common response formats.

        Supports:
        - OpenAI format: {"data": [{"embedding": [...], "object": "embedding"}, ...]}
        - Simple formats: {"embeddings": [[...], ...]} or {"embedding": [...]}

        Args:
            json_obj: JSON response object

        Returns:
            EmbeddingResponseData with extracted embeddings or None if not found
        """
        data = json_obj.get("data")
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and data[0].get("object") == "embedding"
        ):
            embeddings = [item["embedding"] for item in data if "embedding" in item]
            if embeddings:
                return EmbeddingResponseData(embeddings=embeddings)

        for field in ("embeddings", "embedding"):
            value = json_obj.get(field)
            if not (isinstance(value, list) and value):
                continue
            if isinstance(value[0], int | float):
                return EmbeddingResponseData(embeddings=[value])
            if isinstance(value[0], list):
                return EmbeddingResponseData(embeddings=value)

        return None

    def try_extract_rankings(self, json_obj: dict) -> RankingsResponseData | None:
        """Optional utility: Try to extract rankings from common response formats.

        Supports formats with "rankings" or "results" fields containing a list.

        Args:
            json_obj: JSON response object

        Returns:
            RankingsResponseData with extracted rankings or None if not found
        """
        for field in ("rankings", "results"):
            value = json_obj.get(field)
            if isinstance(value, list):
                return RankingsResponseData(rankings=value)
        return None

    def try_extract_text(self, json_obj: dict) -> TextResponseData | None:
        """Optional utility: Try to extract text from common response formats.

        Supports:
        - Simple fields: text, content, response, output, result
        - List of strings (joined without separator): {"text": ["A", "B", "C"]} -> "ABC"
        - OpenAI completions: {"choices": [{"text": "..."}]}
        - OpenAI chat (non-streaming): {"choices": [{"message": {"content": "..."}}]}
        - OpenAI chat (streaming): {"choices": [{"delta": {"content": "..."}}]}

        Args:
            json_obj: JSON response object

        Returns:
            TextResponseData with extracted text or None if not found
        """
        for field in ("text", "content", "response", "output", "result"):
            value = json_obj.get(field)
            if isinstance(value, str):
                return self.make_text_response_data(value)
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, str) for item in value)
            ):
                return self.make_text_response_data("".join(value))

        choices = json_obj.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if text := choice.get("text"):
                return self.make_text_response_data(text)
            # Non-streaming chat format
            message = choice.get("message")
            if message and (content := message.get("content")):
                return self.make_text_response_data(content)
            # Streaming chat format (delta)
            delta = choice.get("delta")
            if delta and (content := delta.get("content")):
                return self.make_text_response_data(content)

        return None

    def convert_to_response_data(self, value: Any) -> BaseResponseData | None:
        """Optional utility: Convert extracted value to appropriate response data type.

        Automatically determines the type based on the value structure:
        - list[list[float]] or list[float] -> EmbeddingResponseData
        - list[dict] -> RankingsResponseData
        - str -> TextResponseData

        Args:
            value: Extracted value from response

        Returns:
            Typed response data or None if conversion not possible
        """
        if value is None:
            return None

        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, list) and first and isinstance(first[0], int | float):
                return EmbeddingResponseData(embeddings=value)
            if isinstance(first, int | float):
                return EmbeddingResponseData(embeddings=[value])
            if isinstance(first, dict):
                return RankingsResponseData(rankings=value)

        if isinstance(value, str):
            return self.make_text_response_data(value)

        return None

    def extract_named_contents(
        self,
        content_items: list[Media],
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Extract contents and organize by name.

        Args:
            content_items: List of content items (texts, images, audios, videos)

        Returns:
            Tuple of (all_contents, contents_by_name)
        """
        all_contents = []
        by_name: dict[str, list[str]] = {}

        for item in content_items:
            if not item.contents:
                continue
            all_contents.extend(item.contents)
            if item.name:
                by_name.setdefault(item.name, []).extend(item.contents)

        return all_contents, by_name
