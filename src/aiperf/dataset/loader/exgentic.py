# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import orjson
from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.exgentic_filters import (
    V1_UNSUPPORTED_FILTER_PAIRS,
    ExgenticDatasetFilters,
    ExgenticHarness,
    ExgenticSourceModel,
    available_filter_values,
)
from aiperf.plugin.enums import PhaseType


def canonical_source_model(value: str) -> str:
    """Strip Exgentic's provider prefixes from a recorded model name."""
    lowered = value.casefold()
    for prefix in ("openai/azure/", "azure/", "aws/", "gcp/"):
        if lowered.startswith(prefix):
            return value[len(prefix) :]
    return value


def _validated_row_lists(
    row_index: int, row: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    models = row.get("models")
    if not isinstance(models, list) or not all(
        isinstance(value, str) and value for value in models
    ):
        raise DatasetLoaderError(
            f"Exgentic row {row_index} models must be a list of non-empty strings"
        )
    spans = row.get("spans")
    if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
        raise DatasetLoaderError(
            f"Exgentic row {row_index} spans must be a list of objects"
        )
    return models, spans


def _timestamp_ms(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() * 1000.0


def _json_string(value: Any) -> str:
    return value if isinstance(value, str) else orjson.dumps(value).decode()


def _parse_json_list(value: Any, *, field: str) -> list[dict[str, Any]]:
    try:
        parsed = orjson.loads(value) if isinstance(value, str | bytes) else value
    except orjson.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON: {error}") from error
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) for item in parsed
    ):
        raise ValueError(f"{field} must be a JSON array of objects")
    return parsed


def _normalize_part(
    part: dict[str, Any],
    *,
    role: str,
    content: list[str],
    reasoning: list[str],
    tool_calls: list[dict[str, Any]],
) -> None:
    part_type = part.get("type")
    if part_type == "text":
        content.append(str(part.get("content") or ""))
    elif part_type == "thinking":
        reasoning.append(str(part.get("thinking") or ""))
    elif part_type == "tool_call":
        tool_calls.append(
            {
                "id": part.get("id"),
                "type": "function",
                "function": {
                    "name": part.get("name"),
                    "arguments": _json_string(part.get("arguments")),
                },
            }
        )
    else:
        raise ValueError(f"unsupported {role!r} message part {part_type!r}")


def _normalize_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    parts = message.get("parts")
    if not isinstance(role, str) or not isinstance(parts, list):
        raise ValueError("each input message requires a string role and parts array")
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []

    def flush_message() -> None:
        output: dict[str, Any] = {
            "role": "system" if role == "developer" else role,
            "content": "".join(content),
        }
        if reasoning:
            output["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            output["tool_calls"] = list(tool_calls)
        normalized.append(output)
        content.clear()
        reasoning.clear()
        tool_calls.clear()

    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("message parts must be objects")
        if part.get("type") == "tool_call_response":
            if content or reasoning or tool_calls:
                flush_message()
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": part.get("id"),
                    "content": _json_string(part.get("result")),
                }
            )
            continue
        _normalize_part(
            part,
            role=role,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    if content or reasoning or tool_calls or not normalized:
        flush_message()
    return normalized


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    messages = _parse_json_list(value, field="gen_ai.input.messages")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        normalized.extend(_normalize_message(message))
    if not normalized:
        raise ValueError("gen_ai.input.messages must contain at least one message")
    return normalized


def _normalize_system_instructions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, str):
        raise ValueError("gen_ai.system_instructions must be a string")
    if not value:
        return []
    try:
        parts = _parse_json_list(value, field="gen_ai.system_instructions")
    except ValueError:
        return [{"role": "system", "content": value}]
    return _normalize_message({"role": "system", "parts": parts})


def _normalize_tools(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    tools = _parse_json_list(value, field="gen_ai.tool.definitions")
    normalized = []
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError(f"unsupported tool type {tool.get('type')!r}")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters"),
                },
            }
        )
    return normalized or None


def _request_extra_body(attributes: dict[str, Any]) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}
    if (temperature := attributes.get("gen_ai.request.temperature")) is not None:
        if (
            not isinstance(temperature, int | float)
            or isinstance(temperature, bool)
            or not isfinite(temperature)
        ):
            raise ValueError("gen_ai.request.temperature must be finite")
        extra_body["temperature"] = temperature
    if (stop_sequences := attributes.get("gen_ai.request.stop_sequences")) is not None:
        if not isinstance(stop_sequences, list) or not all(
            isinstance(sequence, str) for sequence in stop_sequences
        ):
            raise ValueError("gen_ai.request.stop_sequences must be a list of strings")
        if stop_sequences:
            extra_body["stop"] = stop_sequences
    return extra_body or None


def _is_replayable_span(span: dict[str, Any], attributes: dict[str, Any]) -> bool:
    return span.get("type") == "llm_call" or (
        span.get("type") is None and attributes.get("gen_ai.operation.name") == "chat"
    )


class ExgenticDatasetLoader(BaseHFDatasetLoader):
    """Replay complete Exgentic LLM request snapshots as AIPerf sessions."""

    hf_revision = "70036b93a04e61b0ea2706a68b962f4f26774587"
    unsupported_filter_pairs = V1_UNSUPPORTED_FILTER_PAIRS
    supports_benchmark_filter = False

    def __init__(
        self,
        *,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_filters = dict(filters or {})
        if isinstance(source_model := normalized_filters.get("source_model"), str):
            normalized_filters["source_model"] = canonical_source_model(source_model)
        try:
            self.filters = ExgenticDatasetFilters.model_validate(normalized_filters)
        except ValidationError as error:
            available = available_filter_values(
                self.unsupported_filter_pairs,
                supports_benchmark_filter=self.supports_benchmark_filter,
            )
            raise DatasetLoaderError(
                f"Invalid Exgentic dataset filters: {error}; available filters: {available}"
            ) from error
        if self.filters.benchmark is not None and not self.supports_benchmark_filter:
            raise DatasetLoaderError(
                "Exgentic benchmark filter is only supported for Exgentic v2 traces"
            )
        if (
            self.filters.harness is not None
            and self.filters.source_model is not None
            and (self.filters.harness, self.filters.source_model)
            in self.unsupported_filter_pairs
        ):
            available_models = ", ".join(
                model.value
                for model in ExgenticSourceModel
                if (self.filters.harness, model) not in self.unsupported_filter_pairs
            )
            raise DatasetLoaderError(
                "Unsupported Exgentic filter combination "
                f"harness={self.filters.harness.value!r}, "
                f"source_model={self.filters.source_model.value!r}; "
                f"available source models for this harness: {available_models}"
            )
        if self.filters.source_model is not None and all(
            (harness, self.filters.source_model) in self.unsupported_filter_pairs
            for harness in ExgenticHarness
        ):
            raise DatasetLoaderError(
                "Unsupported Exgentic source_model="
                f"{self.filters.source_model.value!r}; no harness supports it"
            )
        super().__init__(**kwargs)
        self._fixed_schedule = any(
            phase.type == PhaseType.FIXED_SCHEDULE
            for phase in self.run.cfg.get_profiling_phases()
        )

    def _max_conversations(self) -> int:
        dataset = self.run.cfg.get_default_dataset()
        if entries := getattr(dataset, "entries", None):
            return entries
        if limit := super()._max_conversations():
            return limit
        raise DatasetLoaderError(
            "Exgentic requires a finite entry or request count; set "
            "--num-conversations, --num-dataset-entries, or --request-count"
        )

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        return await asyncio.to_thread(self._convert_rows, data["dataset"])

    def _matches_filters(
        self, harness: str, source_models: set[str], benchmark: str | None
    ) -> bool:
        if self.filters.harness is not None and harness != self.filters.harness:
            return False
        if self.filters.source_model is not None and not any(
            model.casefold() == str(self.filters.source_model).casefold()
            for model in source_models
        ):
            return False
        return self.filters.benchmark is None or benchmark == self.filters.benchmark

    @staticmethod
    def _parse_span(
        session_id: str,
        span_index: int,
        span: dict[str, Any],
        stats: Counter[str],
    ) -> tuple[float, int, float, Turn] | None:
        attributes = span.get("attributes") or {}
        if not _is_replayable_span(span, attributes):
            stats["non_llm"] += 1
            return None
        if (span.get("status") or {}).get("code") == 2:
            stats["failed"] += 1
            return None
        input_tokens = attributes.get("gen_ai.usage.input_tokens")
        output_tokens = attributes.get("gen_ai.usage.output_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            raise DatasetLoaderError(
                f"Exgentic session {session_id!r} span {span_index} has "
                "non-integer token counts"
            )
        if input_tokens <= 0 or output_tokens <= 0:
            stats["zero_token"] += 1
            return None
        try:
            start_ms = _timestamp_ms(span["start_time"])
            end_ms = _timestamp_ms(span["end_time"])
            if end_ms < start_ms:
                raise ValueError("span ends before it starts")
            requested_max_tokens = attributes.get("gen_ai.request.max_tokens")
            if requested_max_tokens is not None and (
                not isinstance(requested_max_tokens, int)
                or isinstance(requested_max_tokens, bool)
                or requested_max_tokens < 1
            ):
                raise ValueError("gen_ai.request.max_tokens must be a positive integer")
            turn = Turn(
                max_tokens=requested_max_tokens or output_tokens,
                raw_messages=_normalize_system_instructions(
                    attributes.get("gen_ai.system_instructions")
                )
                + _normalize_messages(attributes.get("gen_ai.input.messages")),
                raw_tools=_normalize_tools(attributes.get("gen_ai.tool.definitions")),
                extra_body=_request_extra_body(attributes),
                extra_headers={"x-dynamo-session-id": session_id},
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetLoaderError(
                f"Exgentic session {session_id!r} span {span_index}: {error}"
            ) from error
        return start_ms, span_index, end_ms, turn

    @staticmethod
    def _build_turns(
        spans: list[tuple[float, int, float, Turn]], stats: Counter[str]
    ) -> list[Turn]:
        previous_end_ms: float | None = None
        turns = []
        for start_ms, _, end_ms, turn in spans:
            if previous_end_ms is not None:
                turn.delay = max(0.0, start_ms - previous_end_ms)
                if start_ms < previous_end_ms:
                    stats["overlap"] += 1
            turns.append(turn)
            previous_end_ms = end_ms
        return turns

    @staticmethod
    def _build_fixed_schedule_conversations(
        session_id: str,
        spans: list[tuple[float, int, float, Turn]],
        stats: Counter[str],
    ) -> list[Conversation]:
        session_start_ms = spans[0][0]
        previous_end_ms: float | None = None
        conversations = []
        for start_ms, span_index, end_ms, turn in spans:
            turn.timestamp = start_ms - session_start_ms
            if previous_end_ms is not None and start_ms < previous_end_ms:
                stats["overlap"] += 1
            conversations.append(
                Conversation(
                    session_id=f"{session_id}:{span_index}",
                    context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                    turns=[turn],
                )
            )
            previous_end_ms = end_ms
        return conversations

    def _convert_row(
        self,
        *,
        row_index: int,
        row: dict[str, Any],
        combinations: set[tuple[str, str]],
        seen_sessions: set[str],
        stats: Counter[str],
    ) -> list[Conversation]:
        harness = row.get("harness")
        session_id = row.get("session_id")
        if not isinstance(harness, str) or not harness:
            raise DatasetLoaderError(f"Exgentic row {row_index} has no harness")
        if not isinstance(session_id, str) or not session_id:
            raise DatasetLoaderError(f"Exgentic row {row_index} has no session_id")
        models, row_spans = _validated_row_lists(row_index, row)
        source_models = {canonical_source_model(value) for value in models}
        combinations.update((harness, model) for model in source_models)
        benchmark = row.get("benchmark")
        if not self._matches_filters(
            harness,
            source_models,
            benchmark if isinstance(benchmark, str) else None,
        ):
            return []
        spans = []
        for span_index, span in enumerate(row_spans):
            parsed = self._parse_span(session_id, span_index, span, stats)
            if parsed is not None:
                spans.append(parsed)
        spans.sort(key=lambda item: (item[0], item[1]))
        if not spans:
            return []
        if session_id in seen_sessions:
            raise DatasetLoaderError(f"Duplicate Exgentic session_id {session_id!r}")
        seen_sessions.add(session_id)
        stats["sessions"] += 1
        stats["requests"] += len(spans)
        if self._fixed_schedule:
            return self._build_fixed_schedule_conversations(session_id, spans, stats)

        return [
            Conversation(
                session_id=session_id,
                context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
                turns=self._build_turns(spans, stats),
            )
        ]

    def _convert_rows(self, rows: Iterable[dict[str, Any]]) -> list[Conversation]:
        conversations: list[Conversation] = []
        combinations: set[tuple[str, str]] = set()
        seen_sessions: set[str] = set()
        stats: Counter[str] = Counter()
        max_conversations = self._max_conversations()

        for row_index, row in enumerate(rows, 1):
            row_conversations = self._convert_row(
                row_index=row_index,
                row=row,
                combinations=combinations,
                seen_sessions=seen_sessions,
                stats=stats,
            )
            conversations.extend(row_conversations)
            if max_conversations is not None and stats["sessions"] >= max_conversations:
                break

        if not conversations:
            available = ", ".join(
                f"{harness}/{model}" for harness, model in sorted(combinations)
            )
            raise DatasetLoaderError(
                "No replayable Exgentic spans matched "
                f"harness={self.filters.harness!s}, "
                f"source_model={self.filters.source_model!s}, "
                f"benchmark={self.filters.benchmark!s}; "
                f"available combinations: {available or 'none'}"
            )

        self.info(
            f"Loaded {stats['sessions']} Exgentic sessions / {stats['requests']} "
            f"requests; skipped failed={stats['failed']}, "
            f"non_llm={stats['non_llm']}, zero_token={stats['zero_token']}; "
            f"overlapping_calls={stats['overlap']}"
        )
        return conversations
