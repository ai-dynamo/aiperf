# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay OpenHands SWE-rebench trajectories as multi-turn sessions.

Every recorded assistant message becomes one request. The request for assistant
message k carries the recorded history before it, so DELTAS_WITH_RESPONSES
accumulates the recorded prefix and never appends live responses.
"""

import asyncio
from collections import Counter
from collections.abc import Iterable
from typing import Any

import orjson
from pydantic import ConfigDict, Field, PositiveInt, ValidationError

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import AIPerfBaseModel, Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class SweRebenchDatasetFilters(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved: bool | None = Field(
        default=None,
        description="Keep only trajectories whose recorded resolved flag matches.",
    )
    max_turns: PositiveInt | None = Field(
        default=None,
        description="Keep only the first N recorded requests of each session.",
    )


def _strip_none(value: Any) -> Any:
    """Drop None-valued keys recursively.

    Parquet unifies struct schemas across rows, so every message carries every
    key and every tool schema carries the properties of every other tool, padded
    with None. Lists are kept intact because a None inside a list (for example
    a JSON Schema enum) is a real value.
    """
    if isinstance(value, dict):
        return {
            key: _strip_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def _assistant_output_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    text = content
    if tool_calls := message.get("tool_calls"):
        text += orjson.dumps(tool_calls).decode()
    return text


class SweRebenchOpenHandsDatasetLoader(BaseHFDatasetLoader):
    """Replay recorded OpenHands SWE-rebench agent trajectories."""

    hf_revision = "35455389ab51bf5e2306bfd436ef72d0f98bf882"

    def __init__(
        self,
        *,
        filters: dict[str, str] | None = None,
        tokenizer: Tokenizer | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self.filters = SweRebenchDatasetFilters.model_validate(dict(filters or {}))
        except ValidationError as error:
            raise DatasetLoaderError(
                f"Invalid SWE-rebench OpenHands dataset filters: {error}; "
                "available filters: resolved=[true, false], "
                "max_turns=<positive integer>"
            ) from error
        self.tokenizer = tokenizer
        super().__init__(**kwargs)

    def _max_conversations(self) -> int:
        return self._require_finite_max_conversations("SWE-rebench OpenHands")

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        return await asyncio.to_thread(self._convert_rows, data["dataset"])

    def _convert_rows(self, rows: Iterable[dict[str, Any]]) -> list[Conversation]:
        max_conversations = self._max_conversations()
        conversations: list[Conversation] = []
        stats: Counter[str] = Counter()
        seen_sessions: set[str] = set()

        for row_index, row in enumerate(rows, 1):
            if (
                conversation := self._convert_row(row_index, row, stats, seen_sessions)
            ) is not None:
                conversations.append(conversation)
                if len(conversations) >= max_conversations:
                    break

        if not conversations:
            raise DatasetLoaderError(
                "No SWE-rebench OpenHands trajectories matched the filters "
                f"resolved={self.filters.resolved}, max_turns={self.filters.max_turns}; "
                f"skipped filtered={stats['filtered']}, no_assistant={stats['no_assistant']}"
            )

        self.info(
            f"Loaded {len(conversations)} SWE-rebench OpenHands sessions / "
            f"{stats['requests']} requests; skipped filtered={stats['filtered']}, "
            f"no_assistant={stats['no_assistant']}"
        )
        return conversations

    def _convert_row(
        self,
        row_index: int,
        row: dict[str, Any],
        stats: Counter[str],
        seen_sessions: set[str],
    ) -> Conversation | None:
        trajectory_id = row.get("trajectory_id")
        location = f"row {row_index} trajectory_id={trajectory_id!r}"
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise DatasetLoaderError(
                f"{location}: trajectory_id must be a non-empty string"
            )

        if self.filters.resolved is not None:
            resolved = row.get("resolved")
            if resolved is None:
                stats["filtered"] += 1
                return None
            if resolved not in (0, 1):
                raise DatasetLoaderError(f"{location}: resolved must be 0 or 1")
            if bool(resolved) != self.filters.resolved:
                stats["filtered"] += 1
                return None

        messages = self._row_messages(location, row.get("trajectory"))
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        if not assistant_indices:
            stats["no_assistant"] += 1
            return None
        if assistant_indices[0] == 0:
            raise DatasetLoaderError(
                f"{location}: trajectory must start with a non-assistant message"
            )

        tools = self._row_tools(location, row.get("tools"))
        if trajectory_id in seen_sessions:
            raise DatasetLoaderError(f"{location}: duplicate trajectory_id")
        seen_sessions.add(trajectory_id)

        requests = self._split_requests(messages, assistant_indices)
        if self.filters.max_turns is not None:
            requests = requests[: self.filters.max_turns]

        answers = [answer for _, answer in requests]
        max_tokens = self._output_token_limits(answers)

        turns = [
            Turn(
                raw_messages=raw_messages,
                raw_tools=tools if index == 0 else None,
                max_tokens=max_tokens[index] if max_tokens is not None else None,
            )
            for index, (raw_messages, _) in enumerate(requests)
        ]
        stats["requests"] += len(turns)
        return Conversation(
            session_id=trajectory_id,
            context_mode=ConversationContextMode.DELTAS_WITH_RESPONSES,
            turns=turns,
        )

    @staticmethod
    def _row_messages(location: str, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise DatasetLoaderError(
                f"{location}: trajectory must be a list of objects"
            )
        messages = [_strip_none(item) for item in value]
        for index, message in enumerate(messages):
            if not isinstance(message.get("role"), str):
                raise DatasetLoaderError(
                    f"{location}: message {index} must have a string role"
                )
            if not isinstance(message.get("content", ""), str):
                raise DatasetLoaderError(
                    f"{location}: message {index} content must be a string"
                )
        for message in messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls") or []:
                    call.setdefault("function", {}).setdefault("arguments", "{}")
        return messages

    @staticmethod
    def _row_tools(location: str, value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise DatasetLoaderError(f"{location}: tools must be a list of objects")
        return [_strip_none(item) for item in value] or None

    @staticmethod
    def _split_requests(
        messages: list[dict[str, Any]], assistant_indices: list[int]
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        """Pair each request's new messages with the recorded assistant message it answers.

        Request k covers the messages from the previous recorded assistant message
        (inclusive) up to assistant message k (exclusive); request 0 covers the
        messages before the first assistant message.
        """
        requests = []
        previous = 0
        for index in assistant_indices:
            requests.append((messages[previous:index], messages[index]))
            previous = index
        return requests

    def _output_token_limits(self, answers: list[dict[str, Any]]) -> list[int] | None:
        if self.tokenizer is None:
            return None
        lengths = self.tokenizer.encode_lengths_batch(
            [_assistant_output_text(answer) for answer in answers]
        )
        return [max(1, length) for length in lengths]
