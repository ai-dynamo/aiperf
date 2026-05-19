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
    uses the generic ``BaseEndpoint.build_messages`` flow - the default
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
        rendered = self.build_messages(turns)
        # If the first authored message is already a system role (common
        # in dag_jsonl / mooncake_trace traces), skip prepending the
        # RequestInfo.system_message - some servers concatenate the two,
        # others take the last, neither matches user intent. The authored
        # one wins.
        first_is_system = (
            bool(rendered)
            and isinstance(rendered[0], dict)
            and rendered[0].get("role") == "system"
        )
        if request_info.system_message and not first_is_system:
            messages.append({"role": "system", "content": request_info.system_message})
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(rendered)

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

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
            self._ensure_include_usage(payload)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    @staticmethod
    def _ensure_include_usage(payload: dict[str, Any]) -> None:
        """Force ``stream_options.include_usage = True`` while preserving any
        author-supplied stream_options keys (and any explicit ``include_usage``
        the author already set)."""
        if "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
            return
        if (
            isinstance(payload["stream_options"], dict)
            and "include_usage" not in payload["stream_options"]
        ):
            payload["stream_options"]["include_usage"] = True

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
            self._absorb_chat_choice(
                json_obj.get("object"),
                choices[0],
                content_parts,
                tool_calls_by_index,
            )

        if not tool_calls_by_index:
            # No structured fields to preserve - fall back to base behaviour.
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
