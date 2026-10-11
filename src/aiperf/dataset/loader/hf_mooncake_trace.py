# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load complete Hugging Face agent trajectories through recorded Mooncake replay."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import orjson

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
from aiperf.dataset.loader.models import MooncakeTrace
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.plugin.enums import EndpointType, PhaseType


class HFMooncakeTraceDatasetLoader(BaseHFDatasetLoader):
    """Map Hugging Face trajectory rows to recorded Mooncake request snapshots."""

    def __init__(
        self,
        *,
        prompt_generator: PromptGenerator,
        hf_revision: str | None = None,
        field_mapping: dict[str, str] | None = None,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.hf_revision = hf_revision
        self.field_mapping = field_mapping or {}
        self.filters = filters or {}
        _validate_mapping(self.field_mapping)
        super().__init__(**kwargs)
        if self.run.cfg.endpoint.type != EndpointType.CHAT:
            raise DatasetLoaderError("mooncake_hf requires --endpoint-type chat")
        if any(p.type == PhaseType.FIXED_SCHEDULE for p in self.run.cfg.phases):
            raise DatasetLoaderError(
                "mooncake_hf has no recorded timing; use concurrency or request-rate, "
                "not --fixed-schedule"
            )
        if keys := {"messages", "input", "model"}.intersection(
            self.run.cfg.endpoint.extra
        ):
            raise DatasetLoaderError(
                f"mooncake_hf: --extra-inputs must not override {sorted(keys)}"
            )
        self._mooncake = MooncakeTraceDatasetLoader(
            run=self.run, prompt_generator=prompt_generator, inline_records=[]
        )

    def _max_conversations(self) -> int:
        """Bound remote materialization by explicit entries or profiling limits."""
        if entries := getattr(self.run.cfg.get_default_dataset(), "entries", None):
            return entries
        limits = [
            limit
            for phase in self.run.cfg.get_profiling_phases()
            for limit in (
                getattr(phase, "sessions", None),
                getattr(phase, "requests", None),
            )
            if limit is not None
        ]
        if limits:
            return max(limits)
        raise DatasetLoaderError(
            "mooncake_hf requires a finite materialization limit: "
            "--num-dataset-entries, --num-sessions, or --request-count"
        )

    async def load_dataset(self) -> dict[str, Any]:
        """Validate the materialization bound before opening the HF stream."""
        self._max_conversations()
        return await super().load_dataset()

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Iterate the remote stream and normalize rows outside the event loop."""
        return await asyncio.to_thread(self._convert_rows, data["dataset"])

    def _convert_rows(self, rows: Iterable[dict[str, Any]]) -> list[Conversation]:
        limit = self._max_conversations()
        conversations: list[Conversation] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            try:
                if not all(
                    str(_source_value(row, path)) == value
                    for path, value in self.filters.items()
                ):
                    continue
                mapped = _map_row(row, self.field_mapping)
                session_id = mapped.get("session_id")
                if session_id is None:
                    session_id = f"hf-row-{index}"
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError("session_id must be a nonempty string")
                if session_id in seen:
                    raise ValueError(f"session {session_id!r}: duplicate session_id")
                seen.add(session_id)
                mapped["session_id"] = session_id
                traces = trajectory_to_mooncake(mapped)
            except (TypeError, ValueError) as exc:
                raise DatasetLoaderError(f"mooncake_hf row {index}: {exc}") from exc
            conversations.extend(
                self._mooncake.convert_to_conversations({session_id: traces})
            )
            if len(conversations) >= limit:
                break
        if not conversations:
            raise DatasetLoaderError(
                f"mooncake_hf: no trajectories matched filters {self.filters!r} "
                f"in {self.hf_dataset_name!r}"
            )
        self.info(
            f"Loaded {len(conversations)} agent trajectories / "
            f"{sum(len(c.turns) for c in conversations)} recorded request steps"
        )
        return conversations


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a nonempty list of messages")
    messages = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"message {index}: expected an object")
        message = dict(raw)
        for key in ("reasoning_content", "tool_calls", "tool_call_id", "name"):
            if message.get(key) is None:
                message.pop(key, None)
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


def _validate_mapping(mapping: dict[str, str]) -> None:
    for target, source in mapping.items():
        if target not in {
            "session_id",
            "messages",
            "tools",
            "output_length",
            "extra",
        } and (not target.startswith("extra.") or not target[6:] or "." in target[6:]):
            raise DatasetLoaderError(
                f"--dataset-field target {target!r} is unsupported; use session_id, "
                "messages, tools, output_length, extra, or extra.PARAMETER"
            )
        if not source or any(not part for part in source.split(".")):
            raise DatasetLoaderError(
                f"--dataset-field {target}: invalid source path {source!r}"
            )


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
    params = dict(params or {})
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
    return _decode_json(row.get("tools")), output_length, params


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
