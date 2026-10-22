# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load complete Hugging Face agent trajectories through recorded Mooncake replay."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class HFMooncakeTraceDatasetLoader(BaseHFDatasetLoader):
    """Load HF trajectories and delegate replay construction to Mooncake.

    ``load_dataset`` selects and maps HF rows into recorded Mooncake requests,
    then uses the Mooncake loader to parse and group them. Conversation
    construction shares the same implementation as local Mooncake files.
    """

    def __init__(
        self,
        *,
        run: BenchmarkRun | None = None,
        hf_dataset_name: str,
        hf_split: str = "test",
        hf_subset: str | None = None,
        prompt_generator: PromptGenerator,
        field_mapping: dict[str, str] | None = None,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.field_mapping = field_mapping or {}
        self.filters = filters or {}
        super().__init__(
            run=run,
            hf_dataset_name=hf_dataset_name,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        self._mooncake = MooncakeTraceDatasetLoader(
            run=self.run, prompt_generator=prompt_generator, inline_records=[]
        )

    async def load_dataset(self) -> dict[str, list[MooncakeTrace]]:
        raw = await super().load_dataset()
        return await asyncio.to_thread(self._rows_to_traces, raw["dataset"])

    def _rows_to_traces(
        self, rows: Iterable[dict[str, Any]]
    ) -> dict[str, list[MooncakeTrace]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        max_conversations = self._max_conversations()
        for index, row in enumerate(rows):
            try:
                if not all(
                    _matches_filter(_source_value(row, path), value)
                    for path, value in self.filters.items()
                ):
                    continue
                mapped = _map_row(row, self.field_mapping)
                session_id = mapped.get("session_id")
                if session_id is None:
                    session_id = f"hf-row-{index}"
                if isinstance(session_id, int) and not isinstance(session_id, bool):
                    session_id = str(session_id)
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("session_id must be a nonempty string or integer")
                if session_id in seen:
                    raise ValueError(f"session {session_id!r}: duplicate session_id")
                seen.add(session_id)
                mapped["session_id"] = session_id
                records.extend(
                    trace.model_dump(exclude_none=True)
                    for trace in trajectory_to_mooncake(mapped)
                )
                if max_conversations is not None and len(seen) >= max_conversations:
                    break
            except (TypeError, ValueError) as exc:
                raise DatasetLoaderError(f"mooncake_hf row {index}: {exc}") from exc
        if not records:
            raise DatasetLoaderError(
                f"mooncake_hf: no trajectories matched filters {self.filters!r} "
                f"in {self.hf_dataset_name!r}"
            )
        self._mooncake.inline_records = records
        try:
            return self._mooncake.load_dataset()
        finally:
            self._mooncake.inline_records = []

    async def convert_to_conversations(
        self, data: dict[str, list[MooncakeTrace]]
    ) -> list[Conversation]:
        return await asyncio.to_thread(self._mooncake.convert_to_conversations, data)


def _drop_null_fields(value: Any) -> Any:
    """Omit null object fields from Arrow, preserving null list items and JSON strings."""
    if isinstance(value, dict):
        return {k: _drop_null_fields(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_null_fields(item) for item in value]
    return value


def _matches_filter(value: Any, expected: str) -> bool:
    try:
        return str(value) == expected or orjson.dumps(value).decode() == expected
    except orjson.JSONEncodeError:
        return False


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a nonempty list of messages")
    messages = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"message {index}: expected an object")
        message = _drop_null_fields(raw)
        if calls := message.get("tool_calls"):
            if not isinstance(calls, list) or not all(
                isinstance(c, dict) for c in calls
            ):
                raise ValueError(
                    f"message {index}: tool_calls must be an array of objects"
                )
            message["tool_calls"] = [
                {k: v for k, v in call.items() if k != "index"} for call in calls
            ]
        messages.append(message)
    return messages


def _decode_json(value: Any) -> Any:
    return orjson.loads(value) if isinstance(value, str) else value


def _source_value(row: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path, decoding JSON objects while traversing them."""
    value: Any = row
    for part in path.split("."):
        value = _decode_json(value)
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"source path {path!r} does not exist")
        value = value[part]
    return value


def _map_row(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Use canonical columns by default; explicit mappings override each field."""
    mapped = {
        target: _source_value(row, mapping[target])
        if target in mapping
        else row.get(target)
        for target in ("session_id", "messages", "tools", "output_length", "extra")
    }
    extra = _decode_json(mapped["extra"])
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra must decode to an object")
    extra = dict(extra or {})
    for target, source in mapping.items():
        if target.startswith("extra."):
            value = _source_value(row, source)
            if value is not None:
                extra[target[6:]] = value
    mapped["extra"] = extra
    return mapped


def _parameters(row: dict[str, Any]) -> tuple[Any, int | None, dict[str, Any]]:
    params = _decode_json(row.get("extra"))
    if params is not None and not isinstance(params, dict):
        raise ValueError("extra must decode to an object")
    params = _drop_null_fields(dict(params or {}))
    if keys := {"messages", "input", "model", "tools"}.intersection(params):
        raise ValueError(f"extra must not override {sorted(keys)}")
    output_length = row.get("output_length")
    if output_length is None:
        output_length = params.get("max_tokens")
    if output_length is None:
        output_length = params.get("max_completion_tokens")
    if output_length is not None and (
        not isinstance(output_length, int)
        or isinstance(output_length, bool)
        or output_length <= 0
    ):
        raise ValueError("output_length must be a positive integer")
    for key in ("max_tokens", "max_completion_tokens", "stream", "stream_options"):
        params.pop(key, None)
    return _drop_null_fields(_decode_json(row.get("tools"))), output_length, params


def trajectory_to_mooncake(row: dict[str, Any]) -> list[MooncakeTrace]:
    """Create one complete recorded input prefix before each assistant message.

    The target assistant and any trailing tool results are never sent as a
    separate request. Source replies remain in later prefixes; live responses
    are discarded by Mooncake's existing recorded context mode.
    """
    session_id = row.get("session_id")
    try:
        messages = _messages(_decode_json(row.get("messages")))
        tools, output_length, extra = _parameters(row)
        # Validate even trailing messages, which do not belong to a request prefix.
        MooncakeTrace(
            messages=messages, tools=tools, output_length=output_length, extra=extra
        )
        boundaries = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if not boundaries:
            raise ValueError("messages contains no assistant generation boundaries")
        if boundaries[0] == 0:
            raise ValueError("message 0: assistant has an empty input prefix")
        return [
            MooncakeTrace(
                session_id=session_id,
                messages=messages[:index],
                tools=tools,
                output_length=output_length,
                extra=extra,
                assistant_responses="recorded",
            )
            for index in boundaries
        ]
    except (TypeError, ValueError) as exc:
        raise DatasetLoaderError(f"session {session_id!r}: {exc}") from exc
