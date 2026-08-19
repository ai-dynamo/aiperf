# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.constants import SYSTEM_PROMPT_JOIN_SEP
from aiperf.common.models import (
    BaseResponseData,
    Image,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    ToolCallResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint


def _prepend_system_text(prefix: str, content: Any) -> Any:
    """Return ``content`` with ``prefix`` prepended, preserving its shape.

    System-message content is normally a plain string, but the OpenAI schema
    also permits a list of content parts; a raw-payload dataset may author
    either. Returns a new object in both cases -- callers pass content that
    aliases reusable turn state.
    """
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    if isinstance(content, str) and content:
        return f"{prefix}{SYSTEM_PROMPT_JOIN_SEP}{content}"
    return prefix


class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses. Message-array construction
    uses the generic ``BaseEndpoint.build_messages`` flow. Image UUIDs
    use a narrow image-rendering hook without changing audio or video.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo."""
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        messages = self._format_messages(request_info, self.build_messages(turns))

        # Conversation-level fields walk from the end and pick the most recent
        # non-None value. Per-request overrides stay scoped to the dispatching
        # turn so DAG children do not inherit parent limits or vendor knobs.
        raw_tools = self._latest_turn_attr(turns, "raw_tools")
        max_tokens = turns[-1].max_tokens
        extra_body = turns[-1].extra_body
        model_name = turns[-1].model

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model_name or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if raw_tools is not None:
            payload["tools"] = raw_tools

        if max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if extra_body:
            payload.update(extra_body)

        if model_endpoint.endpoint.streaming:
            # Requested for every streaming run, not just server-token-count
            # ones: vLLM rides per-request metrics (including
            # metrics.speculative_decoding) on the trailing usage chunk and
            # only emits that chunk when include_usage is set, so gating it on
            # an unrelated flag would silently drop those metrics. Authors who
            # want it off can set stream_options.include_usage explicitly.
            # continuous_usage_stats stays opt-in regardless: per_chunk_usage
            # implies use_server_token_count (enforced by the endpoint config
            # validator), so widening this gate cannot turn it on by itself.
            self._ensure_include_usage(
                payload, continuous=model_endpoint.endpoint.per_chunk_usage
            )

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    @staticmethod
    def _format_messages(
        request_info: RequestInfo, rendered: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build chat messages with RequestInfo-level prompts applied.

        When the dataset already rendered a leading system message, the
        conversation-level ``system_message`` is merged into it rather than
        dropped, so a verbatim ``--system-prompt`` still reaches the wire on
        raw-payload/DAG datasets. The two are joined into one message because
        repeated system roles are mishandled by many OpenAI-compatible servers.
        """
        messages: list[dict[str, Any]] = []
        first_is_system = (
            bool(rendered)
            and isinstance(rendered[0], dict)
            and rendered[0].get("role") == "system"
        )
        if request_info.system_message:
            if first_is_system:
                # Copy rather than mutate: ``rendered`` aliases the turn's
                # raw_messages, which are reused across credits in a session, so
                # an in-place edit would restack the prefix on every replay.
                merged = dict(rendered[0])
                merged["content"] = _prepend_system_text(
                    request_info.system_message, merged.get("content")
                )
                messages.append(merged)
                rendered = rendered[1:]
            else:
                messages.append(
                    {"role": "system", "content": request_info.system_message}
                )
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(rendered)
        return messages

    def _extend_image_parts(
        self, parts: list[dict[str, Any]], images: list[Image]
    ) -> None:
        """Append image parts, always passing through authored cache UUIDs."""
        for image in images:
            uuids = getattr(image, "uuids", None)
            if not uuids:
                self._extend_parts(parts, [image], self._render_image_part)
                continue
            for content, uuid in zip(image.contents, uuids, strict=True):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": content},
                        "uuid": uuid,
                    }
                )

    @staticmethod
    def _ensure_include_usage(
        payload: dict[str, Any], *, continuous: bool = False
    ) -> None:
        """Force ``stream_options.include_usage = True`` (and, when ``continuous``,
        ``stream_options.continuous_usage_stats = True``) while preserving any
        author-supplied stream_options keys (and any explicit values the author
        already set).

        ``continuous_usage_stats`` asks the server to report cumulative usage on
        every streamed chunk, not just the final one. It is a vLLM/TRT-LLM
        extension (strict OpenAI rejects it), so it is only injected when the
        caller opts in via ``--per-chunk-usage``.
        """
        stream_options = payload.get("stream_options")
        if stream_options is None:
            stream_options = {}
        elif not isinstance(stream_options, dict):
            return
        # Copy rather than mutate: the payload merge aliases endpoint.extra /
        # turn.extra_body, which are long-lived config reused across every
        # request (and endpoint.extra feeds the mmap cache key), so an in-place
        # edit would rewrite the author's config.
        merged = {**stream_options}
        merged.setdefault("include_usage", True)
        if continuous:
            merged.setdefault("continuous_usage_stats", True)
        payload["stream_options"] = merged

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

        return self._parse_json_response(response.perf_ns, json_obj)

    def _parse_json_response(
        self, perf_ns: int, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse one already-decoded Chat Completions response object."""

        data = self.extract_chat_response_data(json_obj)
        usage = json_obj.get("usage") or None
        spec_decode_stats = self.extract_spec_decode_stats(json_obj)

        if data or usage or spec_decode_stats:
            return ParsedResponse(
                perf_ns=perf_ns,
                data=data,
                usage=usage,
                spec_decode_stats=spec_decode_stats,
            )

        return None

    def process_responses(
        self,
        record: RequestRecord,
        *,
        capture_assistant_turn: bool,
    ) -> tuple[list[ParsedResponse], Turn | None]:
        """Parse chat responses and collect replay fields in one JSON pass."""
        parsed_responses: list[ParsedResponse] = []
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for response in record.responses:
            json_obj = response.get_json()
            if not json_obj:
                continue

            if parsed := self._parse_json_response(response.perf_ns, json_obj):
                parsed_responses.append(parsed)

            if not capture_assistant_turn:
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

        record._parsed_responses_cache = parsed_responses
        if not capture_assistant_turn:
            return parsed_responses, None

        if not tool_calls_by_index:
            return parsed_responses, self._build_assistant_turn_from_parsed(
                parsed_responses
            )

        text = "".join(content_parts)
        tool_calls = [tool_calls_by_index[k] for k in sorted(tool_calls_by_index)]
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text if text else None,
            "tool_calls": tool_calls,
        }
        return parsed_responses, Turn(role="assistant", raw_messages=[assistant_msg])

    def extract_chat_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Surfaces ``tool_calls`` as ``ToolCallResponseData`` for tool-only
        chunks/messages so client-side TTFT and OSL include the tokens
        the model generated for the dispatch (function name + arguments).
        Precedence is ``reasoning > content+tool_calls > tool_calls > content``.
        A chunk that carries both prose ``content`` and a ``tool_calls``
        delta returns a ``ToolCallResponseData`` with both fields set
        (~18% of agentic turns) so client-side OSL counts both portions
        and matches the server's ``usage.completion_tokens``.

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
                # Unrecognized object: the server can return arbitrary bodies
                # (error JSON, proxy pages, truncated streams on crash). Degrade
                # to None like the no-choices/no-data cases below rather than
                # raising, so the worker records a failure and keeps going.
                return None

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

        # Extract tool-call text first so we can emit either a pure
        # ``ToolCallResponseData`` (tool-only chunk) OR a mixed one with
        # ``content`` populated (model talked AND dispatched a tool -
        # ~18% of agentic turns in production traffic). Dropping content
        # when tool_calls are present would silently undercount client-OSL
        # for those mixed chunks since the server's ``usage.completion_tokens``
        # counts both portions.
        tool_calls = data.get("tool_calls") or []
        tool_call_parts: list[str] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "")
            if name:
                tool_call_parts.append(name)
            if arguments:
                tool_call_parts.append(arguments)
        tool_call_text = "".join(tool_call_parts)

        if tool_call_text:
            return ToolCallResponseData(
                tool_call_text=tool_call_text,
                content=content if isinstance(content, str) and content else None,
            )

        if content:
            return self.make_text_response_data(content)

        return None

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture text + ``tool_calls`` from a chat response for replay.

        Walks the raw responses on ``record``, accumulating ``content`` and
        any ``tool_calls`` (reassembling streaming deltas keyed by
        ``index``), then returns a Turn whose ``raw_messages`` re-renders as
        the full assistant message - ``content`` plus ``tool_calls`` -
        verbatim through ``build_messages``. This means a FORK-mode DAG
        child that inherits the parent's history sees the parent's complete
        assistant message, not just the text.

        Falls back to the base text-only behaviour when no ``tool_calls``
        are present, so callers that don't care about tools see no
        behavioural change.
        """
        return self.process_responses(record, capture_assistant_turn=True)[1]

    @staticmethod
    def _absorb_chat_choice(
        obj_type: str | None,
        choice: dict[str, Any],
        content_parts: list[str],
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> None:
        """Fold one ``choices[0]`` entry into the running assistant accumulators.

        Handles both the modern ``tool_calls`` array and the legacy
        singular ``function_call`` (Chat Completions <2023, plus several
        wrappers — LiteLLM, llama.cpp, llama-cpp-python, older vLLM —
        that still emit it). The legacy form is normalised into the
        same index-keyed accumulator as a synthesised function-type
        tool_call so downstream replay sees a single shape.
        """
        if obj_type == "chat.completion":
            msg = choice.get("message") or {}
            if isinstance(msg.get("content"), str):
                content_parts.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                idx = tc.get("index", len(tool_calls_by_index))
                tool_calls_by_index[idx] = {k: v for k, v in tc.items() if k != "index"}
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
        """Merge a legacy streaming ``function_call`` delta into a synthesised slot.

        Legacy chunks emit ``delta.function_call={"name": ..., "arguments": ...}``
        without an ``index``. Concatenate into a single slot keyed at
        index 0 so name/arguments fragments accumulate correctly across
        chunks, matching the assembly behaviour of ``_merge_tool_call_delta``.
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

        Falls back to ``len(tool_calls_by_index)`` (matching the
        non-streaming path) when the server omits ``index`` - defaulting
        to ``0`` would collapse parallel tool calls into a single slot,
        overwriting names and concatenating arguments into a Frankenstein
        call. Some Azure proxies and older vLLM tool-call patches drop
        ``index`` even though the OpenAI streaming spec requires it.
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
