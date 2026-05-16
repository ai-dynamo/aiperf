# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    Text,
    ToolCallResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"
_FAST_PARSE_FALLBACK = object()


class ChatEndpoint(BaseEndpoint):
    _FAST_PARSE_FALLBACK = _FAST_PARSE_FALLBACK
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Chat Completions API payload
        """
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = self.run.cfg

        if turns[-1].raw_messages is not None:
            messages = turns[-1].raw_messages
        else:
            messages = self._create_messages(
                turns, request_info.system_message, request_info.user_context_message
            )

        payload = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.get_model_names()[0],
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].raw_tools is not None:
            payload["tools"] = turns[-1].raw_tools
        else:
            # Walk back through prior turns so DAG FORK children inherit
            # ``raw_tools`` from the parent turn that declared them. Stop on
            # the first non-None value (closest ancestor wins).
            for prior in reversed(turns[:-1]):
                if prior.raw_tools is not None:
                    payload["tools"] = prior.raw_tools
                    break

        if turns[-1].max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        # Per-turn ``extra_body`` overrides endpoint-level extra and even the
        # built-in keys (model, messages, tools, stream, etc.). This matches
        # the OpenAI client contract where ``extra_body`` is the last writer.
        # Circular references are handled by ``dict.update`` (no copy needed).
        if turns[-1].extra_body:
            try:
                payload.update(turns[-1].extra_body)
            except Exception as e:
                self.warning(
                    lambda exc=e: f"Failed to merge extra_body into payload: {exc}"
                )

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
            # Automatically set stream_options to include usage when using server token counts
            if "stream_options" not in payload:
                payload["stream_options"] = {"include_usage": True}
            elif (
                isinstance(payload["stream_options"], dict)
                and "include_usage" not in payload["stream_options"]
            ):
                payload["stream_options"]["include_usage"] = True

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(
        self,
        turns: list[Turn],
        system_message: str | None,
        user_context_message: str | None,
    ) -> list[dict[str, Any]]:
        """Create messages from turns for OpenAI Chat Completions.

        Args:
            turns: List of turns in the request
            system_message: Optional shared system message to prepend
            user_context_message: Optional per-conversation user context to prepend

        Returns:
            List of formatted message dicts for OpenAI Chat Completions API
        """
        messages = []

        # Prepend system_message and user_context_message if present
        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

        if user_context_message:
            messages.append(
                {
                    "role": "user",
                    "content": user_context_message,
                }
            )

        for turn in turns:
            message = {
                "role": turn.role or _DEFAULT_ROLE,
            }
            self._set_message_content(message, turn)
            messages.append(message)
        return messages

    def _set_message_content(self, message: dict[str, Any], turn: Turn) -> None:
        """Create message content from turn for OpenAI Chat Completions."""
        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and len(turn.images) == 0
            and len(turn.audios) == 0
            and len(turn.videos) == 0
        ):
            # Hotfix for Dynamo API which does not yet support a list of messages
            message["content"] = (
                turn.texts[0].contents[0] if turn.texts[0].contents else ""
            )
            return

        message_content: list[dict[str, Any]] = []
        self._append_text_parts(message_content, turn)
        self._append_image_parts(message_content, turn)
        self._append_audio_parts(message_content, turn)
        self._append_video_parts(message_content, turn)
        message["content"] = message_content

    @staticmethod
    def _append_text_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for text in turn.texts:
            for content in text.contents:
                if not content:
                    continue
                parts.append({"type": "text", "text": content})

    @staticmethod
    def _append_image_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for image in turn.images:
            for content in image.contents:
                if not content:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": content}})

    @staticmethod
    def _append_audio_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for audio in turn.audios:
            for content in audio.contents:
                if not content:
                    continue
                if "," not in content:
                    raise ValueError(
                        "Audio content must be in the format 'format,b64_audio'."
                    )
                format, b64_audio = content.split(",", 1)
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": format,
                        },
                    }
                )

    @staticmethod
    def _append_video_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for video in turn.videos:
            for content in video.contents:
                if not content:
                    continue
                parts.append({"type": "video_url", "video_url": {"url": content}})

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture text + ``tool_calls`` from a chat response for replay.

        Walks the raw responses on ``record``, accumulating ``content`` and
        any ``tool_calls`` (reassembling streaming deltas keyed by ``index``,
        with a fallback when ``index`` is missing so parallel tool calls
        don't collapse), then returns a Turn whose ``raw_messages``
        re-renders as the full assistant message — ``content`` plus
        ``tool_calls`` — verbatim through ``_create_messages``. This means a
        FORK-mode DAG child that inherits the parent's history sees the
        parent's complete assistant message, not just the text.

        Legacy non-streaming ``function_call`` (Chat Completions <2023, plus
        LiteLLM / llama.cpp / older vLLM wrappers) and streaming
        ``function_call`` deltas are normalised into the same index-keyed
        accumulator as a synthesised function-type tool_call so downstream
        replay sees a single shape.

        Falls back to the base text-only behaviour when no ``tool_calls``
        are present, so callers that don't care about tools see no
        behavioural change.
        """
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for response in record.responses or []:
            json_obj = response.get_json() if hasattr(response, "get_json") else None
            if not json_obj:
                continue
            choices = json_obj.get("choices") or []
            if not choices:
                continue
            self._absorb_chat_choice(
                json_obj.get("object"),
                choices[0],
                content_parts,
                tool_calls_by_index,
            )

        if not tool_calls_by_index:
            return super().build_assistant_turn(record)

        text = "".join(content_parts)
        tool_calls = [tool_calls_by_index[k] for k in sorted(tool_calls_by_index)]
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text if text else None,
            "tool_calls": tool_calls,
        }
        return Turn(role="assistant", raw_messages=[assistant_msg])

    @staticmethod
    def _absorb_chat_choice(
        obj_type: str | None,
        choice: dict[str, Any],
        content_parts: list[str],
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> None:
        """Fold one ``choices[0]`` entry into the running assistant accumulators."""
        if obj_type == "chat.completion":
            msg = choice.get("message") or {}
            if isinstance(msg.get("content"), str):
                content_parts.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                idx = tc.get("index", len(tool_calls_by_index))
                tool_calls_by_index[idx] = {
                    k: v for k, v in tc.items() if k != "index"
                }
            ChatEndpoint._absorb_legacy_function_call(
                msg.get("function_call"), tool_calls_by_index
            )
            return

        if obj_type == "chat.completion.chunk":
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            for tc_delta in delta.get("tool_calls") or []:
                ChatEndpoint._merge_tool_call_delta(tc_delta, tool_calls_by_index)
            ChatEndpoint._merge_legacy_function_call_delta(
                delta.get("function_call"), tool_calls_by_index
            )

    @staticmethod
    def _absorb_legacy_function_call(
        function_call: Any,
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> None:
        """Fold a legacy non-streaming ``function_call`` into a synthesised tool_call slot."""
        if not isinstance(function_call, dict):
            return
        idx = len(tool_calls_by_index)
        tool_calls_by_index[idx] = {
            "type": "function",
            "function": {
                "name": function_call.get("name", ""),
                "arguments": function_call.get("arguments", ""),
            },
        }

    @staticmethod
    def _merge_legacy_function_call_delta(
        fn_delta: Any,
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> None:
        """Concatenate streaming ``function_call`` delta into a synthesised slot.

        Legacy chunks emit ``delta.function_call={"name": ..., "arguments": ...}``
        without an ``index``. Concatenate into a single slot keyed at index 0
        so name/arguments fragments accumulate correctly across chunks.
        """
        if not isinstance(fn_delta, dict):
            return
        existing = tool_calls_by_index.setdefault(
            0, {"type": "function", "function": {}}
        )
        existing.setdefault("type", "function")
        fn = existing.setdefault("function", {})
        if fn_delta.get("name"):
            fn["name"] = fn.get("name", "") + fn_delta["name"]
        if "arguments" in fn_delta:
            fn["arguments"] = fn.get("arguments", "") + (fn_delta["arguments"] or "")

    @staticmethod
    def _merge_tool_call_delta(
        tc_delta: dict[str, Any],
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> None:
        """Merge one streaming ``tool_calls`` delta into the index-keyed accumulator.

        Falls back to ``len(tool_calls_by_index)`` when the server omits
        ``index`` — defaulting to ``0`` would collapse parallel tool calls
        into a single slot, overwriting names and concatenating arguments
        into a Frankenstein call. Some Azure proxies and older vLLM
        tool-call patches drop ``index`` even though the OpenAI streaming
        spec requires it.
        """
        idx = tc_delta.get("index", len(tool_calls_by_index))
        existing = tool_calls_by_index.setdefault(idx, {})
        if tc_delta.get("id"):
            existing["id"] = tc_delta["id"]
        if tc_delta.get("type"):
            existing["type"] = tc_delta["type"]
        fn_delta = tc_delta.get("function") or {}
        if not fn_delta:
            return
        fn = existing.setdefault("function", {})
        if fn_delta.get("name"):
            fn["name"] = fn_delta["name"]
        if "arguments" in fn_delta:
            fn["arguments"] = fn.get("arguments", "") + (fn_delta["arguments"] or "")

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Chat Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        fast_parsed = self._fast_parse_response(json_obj, response.perf_ns)
        if fast_parsed is not self._FAST_PARSE_FALLBACK:
            return fast_parsed

        data = self.extract_chat_response_data(json_obj)
        usage = json_obj.get("usage") or None

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def _fast_parse_response(
        self,
        json_obj: JsonObject,
        perf_ns: int,
    ) -> ParsedResponse | None | object:
        """Fast-path the common OpenAI chat shapes and fall back for anything unusual."""
        try:
            data_key = self._fast_parse_data_key(json_obj)
            if data_key is None:
                return self._FAST_PARSE_FALLBACK

            choices = json_obj.get("choices")
            if not choices:
                return self._usage_only_response(json_obj, perf_ns)

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return self._FAST_PARSE_FALLBACK

            data = first_choice.get(data_key)
            if not isinstance(data, dict):
                return None

            return self._build_fast_parsed(data, json_obj, perf_ns)
        except (IndexError, KeyError, TypeError):
            return self._FAST_PARSE_FALLBACK

    @staticmethod
    def _fast_parse_data_key(json_obj: JsonObject) -> str | None:
        object_type = json_obj.get("object")
        if object_type == "chat.completion":
            return "message"
        if object_type == "chat.completion.chunk":
            return "delta"
        return None

    @staticmethod
    def _usage_only_response(
        json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        # Final usage-only chunk (stream_options.include_usage=true)
        # has empty choices but carries the cumulative usage totals.
        usage = json_obj.get("usage") or None
        if usage is not None:
            return ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
        return None

    def _build_fast_parsed(
        self, data: dict[str, Any], json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")
        usage = json_obj.get("usage") or None

        # When tool_calls are present in the same delta as ``content``, both
        # must round-trip — reassemble via ToolCallResponseData(content=...).
        tool_call_text = "" if reasoning else _extract_tool_call_text(data)

        if not content and not reasoning and not tool_call_text and not usage:
            return None

        if reasoning:
            response_data: BaseResponseData | None = ReasoningResponseData(
                content=content,
                reasoning=reasoning,
            )
        elif tool_call_text:
            response_data = ToolCallResponseData(
                tool_call_text=tool_call_text,
                content=content if isinstance(content, str) and content else None,
            )
        elif content:
            response_data = self.make_text_response_data(content)
        else:
            response_data = None

        if response_data or usage:
            return ParsedResponse(perf_ns=perf_ns, data=response_data, usage=usage)
        return None

    def extract_chat_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Priority order: reasoning > content > tool_calls > None. Tool-call
        deltas are concatenated as ``function.name + function.arguments``
        across every tool call in the chunk so the tokenizer can count them
        and TTFT can include them as a non-reasoning first-token boundary.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted response data or None if no content
        """
        match json_obj.get("object"):
            case "chat.completion":
                data_key = "message"
            case "chat.completion.chunk":
                data_key = "delta"
            case _:
                object_type = json_obj.get("object")
                raise ValueError(f"Unsupported OpenAI object type: {object_type!r}")

        choices = json_obj.get("choices")
        if not choices:
            self.debug(lambda: f"No choices found in response: {json_obj}")
            return None

        data = choices[0].get(data_key)
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")

        if reasoning:
            return ReasoningResponseData(content=content, reasoning=reasoning)

        tool_call_text = _extract_tool_call_text(data)
        if tool_call_text:
            # Mixed-content chunk: prose lives in ``content`` and the tool
            # call shape is preserved verbatim. ``ToolCallResponseData`` is
            # the union shape that carries both.
            return ToolCallResponseData(
                tool_call_text=tool_call_text,
                content=content if isinstance(content, str) and content else None,
            )

        if content:
            return self.make_text_response_data(content)

        return None


def _extract_tool_call_text(data: dict[str, Any]) -> str:
    """Concatenate ``function.name + function.arguments`` across all tool calls.

    Used for both streaming chunks (``delta.tool_calls``) and non-streaming
    responses (``message.tool_calls``). Empty strings are skipped so partial
    deltas don't introduce gaps. Returns an empty string if no tool calls.
    """
    tool_calls = data.get("tool_calls") or []
    parts: list[str] = []
    for tc in tool_calls:
        func = tc.get("function") or {}
        name = func.get("name") or ""
        arguments = func.get("arguments") or ""
        if name:
            parts.append(name)
        if arguments:
            parts.append(arguments)
    return "".join(parts)
