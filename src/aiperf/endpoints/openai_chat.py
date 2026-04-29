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
    ToolCallResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint


class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses. Message-array construction
    uses the generic ``BaseEndpoint.build_messages`` flow — the default
    ``_render_*_part`` hooks already emit OpenAI chat shape, so nothing
    needs overriding here.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo."""
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        # Prepend the shared system + per-conversation user-context prompts
        # (both live on RequestInfo), then flatten turns via the generic
        # build_messages skeleton.
        messages: list[dict[str, Any]] = []
        if request_info.system_message:
            messages.append({"role": "system", "content": request_info.system_message})
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(self.build_messages(turns))

        payload: dict[str, Any] = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].raw_tools is not None:
            payload["tools"] = turns[-1].raw_tools

        if turns[-1].max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if turns[-1].extra_body:
            payload.update(turns[-1].extra_body)

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

        data = self.extract_chat_response_data(json_obj)
        usage = json_obj.get("usage") or None

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def extract_chat_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Surfaces ``tool_calls`` as ``ToolCallResponseData`` for tool-only
        chunks/messages so client-side TTFT and OSL include the tokens
        the model generated for the dispatch (function name + arguments).
        Precedence is ``reasoning > content > tool_calls`` — the first
        non-empty field wins. A chunk that carries both prose ``content``
        and a tool-call delta returns the prose ``content`` only; the
        server's ``usage.completion_tokens`` is the source of truth for
        the combined token count when needed.

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

        # Extract tool-call text first so we can emit either a pure
        # ``ToolCallResponseData`` (tool-only chunk) OR a mixed one with
        # ``content`` populated (model talked AND dispatched a tool —
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
        the full assistant message — ``content`` plus ``tool_calls`` —
        verbatim through ``build_messages``. This means a FORK-mode DAG
        child that inherits the parent's history sees the parent's complete
        assistant message, not just the text.

        Falls back to the base text-only behaviour when no ``tool_calls``
        are present, so callers that don't care about tools see no
        behavioural change.
        """
        content_parts: list[str] = []
        # OpenAI streams tool_calls as deltas keyed by ``index``; each delta
        # may carry a partial id, type, function.name, or function.arguments
        # fragment that must be concatenated in order.
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for response in record.responses:
            json_obj = response.get_json()
            if not json_obj:
                continue
            choices = json_obj.get("choices") or []
            if not choices:
                continue

            obj_type = json_obj.get("object")
            if obj_type == "chat.completion":
                msg = choices[0].get("message") or {}
                if isinstance(msg.get("content"), str):
                    content_parts.append(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    idx = tc.get("index", len(tool_calls_by_index))
                    tool_calls_by_index[idx] = {
                        k: v for k, v in tc.items() if k != "index"
                    }
            elif obj_type == "chat.completion.chunk":
                delta = choices[0].get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    existing = tool_calls_by_index.setdefault(idx, {})
                    if tc_delta.get("id"):
                        existing["id"] = tc_delta["id"]
                    if tc_delta.get("type"):
                        existing["type"] = tc_delta["type"]
                    fn_delta = tc_delta.get("function") or {}
                    if fn_delta:
                        fn = existing.setdefault("function", {})
                        if fn_delta.get("name"):
                            fn["name"] = fn_delta["name"]
                        if "arguments" in fn_delta:
                            fn["arguments"] = fn.get("arguments", "") + (
                                fn_delta["arguments"] or ""
                            )

        if not tool_calls_by_index:
            # No structured fields to preserve — fall back to base behaviour.
            return super().build_assistant_turn(record)

        text = "".join(content_parts)
        tool_calls = [tool_calls_by_index[k] for k in sorted(tool_calls_by_index)]
        # OpenAI requires ``content`` on assistant messages; it is permitted
        # to be ``null`` when the message carries ``tool_calls`` instead.
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text if text else None,
            "tool_calls": tool_calls,
        }
        return Turn(role="assistant", raw_messages=[assistant_msg])
