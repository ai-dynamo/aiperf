# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.enums import CreditPhase
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

        # Flatten ALL turns (raw_messages spliced verbatim, structured turns
        # rendered) so accumulated session history — prior authored turns,
        # captured assistant replies, and DAG FORK-seeded parent context —
        # reaches the wire. A previous last-turn-only shortcut
        # (``turns[-1].raw_messages``) silently dropped every prior turn for
        # raw_messages datasets (dag_jsonl), severing multi-turn and FORK
        # context inheritance.
        messages = self._format_messages(request_info, self.build_messages(turns))

        payload = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.get_model_names()[0],
            "stream": model_endpoint.endpoint.streaming,
        }

        self._apply_raw_tools(payload, turns)

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
            self._ensure_stream_usage(payload)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    @staticmethod
    def _apply_raw_tools(payload: dict[str, Any], turns: list[Turn]) -> None:
        """Set ``payload["tools"]`` from the dispatch turn's ``raw_tools``.

        Walks back through prior turns so DAG FORK children inherit
        ``raw_tools`` from the parent turn that declared them. Stops on the
        first non-None value (closest ancestor wins). Leaves ``payload``
        untouched when no turn declares tools.
        """
        if turns[-1].raw_tools is not None:
            payload["tools"] = turns[-1].raw_tools
            return
        for prior in reversed(turns[:-1]):
            if prior.raw_tools is not None:
                payload["tools"] = prior.raw_tools
                break

    @staticmethod
    def _ensure_stream_usage(payload: dict[str, Any]) -> None:
        """Set ``stream_options.include_usage`` so streamed responses carry usage.

        Server token counts require the final usage-bearing chunk; user-supplied
        ``stream_options`` (including non-dict shapes from ``extra_body``) are
        preserved, only a missing ``include_usage`` key is filled in.
        """
        if "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
        elif (
            isinstance(payload["stream_options"], dict)
            and "include_usage" not in payload["stream_options"]
        ):
            payload["stream_options"]["include_usage"] = True

    @staticmethod
    def _format_messages(
        request_info: RequestInfo, rendered: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply RequestInfo-level prompts to pre-rendered (raw) messages.

        Authored ``raw_messages`` own the wire shape: an author-supplied leading
        system message wins over ``request_info.system_message`` EXCEPT during
        the WARMUP phase, where the warmup marker is merged into the leading
        system message so the server can still identify warmup traffic. When
        there is no leading system message, ``request_info.system_message`` is
        prepended as its own system message. The original ``rendered`` list and
        its dicts are never mutated.
        """
        messages: list[dict[str, Any]] = []
        first_is_system = (
            bool(rendered)
            and isinstance(rendered[0], dict)
            and rendered[0].get("role") == "system"
        )
        if request_info.system_message:
            if first_is_system and request_info.credit_phase == CreditPhase.WARMUP:
                rendered = ChatEndpoint._prepend_system_message(
                    rendered, request_info.system_message
                )
            elif not first_is_system:
                messages.append(
                    {"role": "system", "content": request_info.system_message}
                )
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(rendered)
        return messages

    @staticmethod
    def _prepend_system_message(
        rendered: list[dict[str, Any]], system_message: str
    ) -> list[dict[str, Any]]:
        """Prepend ``system_message`` to the leading rendered system message
        without mutating the caller's list/dicts."""
        first = dict(rendered[0])
        content = first.get("content")
        if isinstance(content, str):
            first["content"] = (
                f"{system_message}\n{content}" if content else system_message
            )
        elif isinstance(content, list):
            first["content"] = [{"type": "text", "text": system_message}, *content]
        elif content is None:
            first["content"] = system_message
        else:
            first["content"] = f"{system_message}\n{content}"
        return [first, *rendered[1:]]

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture text + ``tool_calls`` from a chat response for replay.

        Walks the raw responses on ``record``, accumulating ``content`` and
        any ``tool_calls`` (reassembling streaming deltas keyed by ``index``,
        with a fallback when ``index`` is missing so parallel tool calls
        don't collapse), then returns a Turn whose ``raw_messages``
        re-renders as the full assistant message — ``content`` plus
        ``tool_calls`` — verbatim through ``build_messages``. This means a
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
            if not isinstance(json_obj, dict):
                # Same top-level guard as parse_response: a bare-list/string/int
                # body would crash the ``json_obj.get("choices")`` below.
                continue
            choices = json_obj.get("choices") or []
            # Same malformed-``choices`` degradation as the parse path: an empty
            # list, a non-list value, or a first entry that isn't a dict would
            # crash ``_absorb_chat_choice``'s ``choice.get(...)``. Skip it.
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
            ):
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

        # A bare-list/string/int 200-OK body (e.g. a chat endpoint pointed at a
        # TGI/HF server, which returns ``[{...}]``) would crash
        # ``_fast_parse_data_key``'s ``json_obj.get("object")`` — the fast-path
        # try/except catches only (IndexError, KeyError, TypeError), NOT
        # AttributeError, so it propagates through the worker's unconditional
        # post-response parse and drops every record. Degrade to a clean
        # no-content error record instead (mirrors huggingface_generate.py).
        if not isinstance(json_obj, dict):
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
                # No delta/message dict (e.g. finish_reason-only or
                # usage-carrying final chunks): fall through to usage handling
                # so server-reported usage is not dropped — mirrors the slow
                # path's ``if data or usage`` semantics.
                return self._usage_only_response(json_obj, perf_ns)

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
        # has empty choices — or a choice without a delta/message dict —
        # but carries the cumulative usage totals.
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
                # Unrecognized / missing object type: the server can return
                # error bodies, proxy pages, or truncated streams on crash.
                # Degrade to None so the worker records a request failure rather
                # than crashing the parser (see the malformed-response contract
                # in tests/unit/records/test_inference_result_parser.py).
                self.debug(
                    lambda: f"Unsupported OpenAI object type: {json_obj.get('object')!r}"
                )
                return None

        choices = json_obj.get("choices")
        if not choices:
            self.debug(lambda: f"No choices found in response: {json_obj}")
            return None

        # Malformed ``choices`` shapes — a non-list value or a first entry that
        # isn't a dict (``[None]``, ``['x']``, ``[5]``, ``'oops'``, ``{...}``) —
        # degrade to None rather than crashing the parser, mirroring the fast
        # path's ``isinstance(first_choice, dict)`` guard so both paths agree on
        # every malformed body (see the contract comment above).
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            self.debug(lambda: f"Malformed choices in response: {json_obj}")
            return None

        data = choices[0].get(data_key)
        if not isinstance(data, dict):
            # A truthy non-dict ``delta``/``message`` (e.g. ``message: 'hello'``)
            # would crash ``data.get(...)`` below; the fast path routes this to
            # ``_usage_only_response``, so degrade to None here and let
            # ``parse_response`` surface any usage exactly as the fast path does.
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
