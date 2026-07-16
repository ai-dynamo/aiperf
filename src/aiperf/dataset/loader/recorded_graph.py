# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-only loader identities for Rust-native recorded graphs.

The Python frontend owns plugin names and Config-v2 validation, but it does not
parse or lower these sources. The selected ``aiperf`` adapter acquires,
validates, reconstructs, and compiles each trace exactly once.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any

import orjson

from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class _RecordedGraphNativeLoader(BaseFileLoader):
    """Shared fail-closed legacy surface for a runner-owned graph format."""

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[Any]]:
        raise RuntimeError(
            f"{type(self).__name__} is a Rust-native Graph-IR input; "
            "it must be loaded by aiperf runner"
        )

    def convert_to_conversations(self, custom_data: dict[str, list[Any]]) -> list[Any]:
        raise RuntimeError(
            f"{type(self).__name__} cannot enter the legacy linear conversation path"
        )


class WekaTraceNativeLoader(_RecordedGraphNativeLoader):
    """Plugin identity and bounded structural detector for ``weka_trace``."""

    hf_revision = "23f152f6f0f9399a85901b89a6458def0ef16729"

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> bool:
        if _weka_value_matches(data):
            return True
        if filename is None:
            return False
        path = Path(filename)
        if path.is_dir():
            try:
                candidates = sorted(
                    child
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() == ".json"
                )
            except OSError:
                return False
            return bool(candidates and _weka_file_matches(candidates[0]))
        return path.suffix.lower() == ".json" and _weka_file_matches(path)


class DynamoTraceNativeLoader(_RecordedGraphNativeLoader):
    """Plugin identity and first-record detector for ``dynamo_trace``."""

    @classmethod
    def can_load(
        cls,
        data: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> bool:
        if _dynamo_value_matches(data):
            return True
        if filename is None:
            return False
        path = Path(filename)
        if path.is_dir():
            try:
                candidates = sorted(
                    (
                        child
                        for child in path.iterdir()
                        if child.is_file()
                        and child.name.endswith((".jsonl", ".jsonl.gz"))
                    ),
                    key=_dynamo_segment_sort_key,
                )
            except OSError:
                return False
            return bool(candidates and _dynamo_file_matches(candidates[0]))
        if path.suffix.lower() not in {".gz", ".jsonl"}:
            return False
        return _dynamo_file_matches(path)


_WEKA_ALLOWED_KEYS = {
    "id",
    "models",
    "block_size",
    "hash_id_scope",
    "tool_tokens",
    "system_tokens",
    "requests",
    "totals",
    "kind",
}
_DYNAMO_EVENT_TYPES = {"request_end", "tool_start", "tool_end", "tool_error"}
_DYNAMO_SEGMENT = re.compile(r"^(.+?)\.(\d{6,})\.jsonl\.gz$")


def _weka_value_matches(data: object) -> bool:
    return bool(
        isinstance(data, dict)
        and set(data).issubset(_WEKA_ALLOWED_KEYS)
        and isinstance(data.get("id"), str)
        and isinstance(data.get("models"), list)
        and isinstance(data.get("block_size"), int)
        and not isinstance(data.get("block_size"), bool)
        and data.get("hash_id_scope") in {"local", "global"}
        and isinstance(data.get("requests"), list)
    )


def _weka_file_matches(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            head = stream.read(4096)
        if not head:
            return False
        try:
            value = orjson.loads(head)
        except orjson.JSONDecodeError:
            if not all(
                f'"{name}"'.encode() in head
                for name in ("id", "models", "block_size", "hash_id_scope", "requests")
            ):
                return False
            value = orjson.loads(path.read_bytes())
        return _weka_value_matches(value)
    except (OSError, orjson.JSONDecodeError):
        return False


def _dynamo_value_matches(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if "schema" not in data and isinstance(data.get("event"), dict):
        data = data["event"]
    context = data.get("agent_context")
    return bool(
        data.get("schema") == "dynamo.request.trace.v1"
        and data.get("event_type") in _DYNAMO_EVENT_TYPES
        and (context is None or isinstance(context, dict))
    )


def _dynamo_file_matches(path: Path) -> bool:
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line := line.strip():
                    return _dynamo_value_matches(orjson.loads(line))
    except (OSError, UnicodeError, EOFError, orjson.JSONDecodeError):
        return False
    return False


def _dynamo_segment_sort_key(path: Path) -> tuple[str, int]:
    match = _DYNAMO_SEGMENT.match(path.name)
    if match is None:
        return path.name, -1
    return match.group(1), int(match.group(2))


__all__ = ["DynamoTraceNativeLoader", "WekaTraceNativeLoader"]
