# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.dataset.loader.base_trace_loader import BaseTraceDatasetLoader
from aiperf.dataset.loader.models import BailianTrace


class BailianTraceDatasetLoader(BaseTraceDatasetLoader[BailianTrace]):
    """A dataset loader for Alibaba Bailian trace data.

    See https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon

    Loads Bailian trace data from a JSONL file and converts it into
    conversations for the dataset manager. Multi-turn conversations are
    linked via `chat_id` / `parent_chat_id` (`-1` = root) and
    ordered by `turn`.

    Timestamps are **seconds since request arrival** and are converted to
    **milliseconds** internally.

    The 16-token SipHash block size is declared in `plugins.yaml` metadata
    and applied automatically—no need to pass `--isl-block-size`.

    Example JSONL entry::

        {"chat_id": 159, "parent_chat_id": -1, "timestamp": 61.114,
         "input_length": 521, "output_length": 132, "type": "text",
         "turn": 1, "hash_ids": [1089, 1090, 1091, 6326, 13148]}
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        Detects Bailian traces by the presence of `chat_id`,
        `parent_chat_id`, and `turn` fields.
        """
        if data is None:
            return False

        try:
            BailianTrace.model_validate(data)
            return True
        except ValidationError:
            return False

    # ------------------------------------------------------------------
    # Template-method hooks (see BaseTraceDatasetLoader.load_dataset)
    # ------------------------------------------------------------------

    def _parse_trace(self, line: str) -> BailianTrace:
        return BailianTrace.model_validate_json(line)

    def _preprocess_trace(self, trace: BailianTrace) -> None:
        """Convert timestamp from seconds to milliseconds."""
        trace.timestamp = trace.timestamp * 1000.0

    def _group_traces(self, items: list[BailianTrace]) -> dict[str, list[BailianTrace]]:
        return self._group_into_sessions(items)

    def _group_into_sessions(
        self, items: list[BailianTrace]
    ) -> dict[str, list[BailianTrace]]:
        """Group flat trace entries into sessions using parent-child links.

        Builds a union-find over `chat_id` → `parent_chat_id` to identify
        session roots, then groups entries by root and sorts each session by
        `turn`.  Root requests have `parent_chat_id == -1`.
        """
        if not items:
            return {}

        # Build lookup: chat_id → trace
        by_chat_id: dict[int, BailianTrace] = {t.chat_id: t for t in items}

        # Find root chat_id for each entry by walking parent links
        root_cache: dict[int, int] = {}

        def find_root(chat_id: int) -> int:
            if chat_id in root_cache:
                return root_cache[chat_id]

            path: list[int] = []
            seen: set[int] = set()
            current = chat_id
            while current in by_chat_id and by_chat_id[current].parent_chat_id != -1:
                if current in seen:
                    break
                seen.add(current)
                parent = by_chat_id[current].parent_chat_id
                if parent == current or parent not in by_chat_id:
                    break
                path.append(current)
                current = parent

            # Path compression
            for node in path:
                root_cache[node] = current
            root_cache[chat_id] = current
            return current

        groups: dict[str, list[BailianTrace]] = defaultdict(list)
        for trace in items:
            root = find_root(trace.chat_id)
            session_id = str(root)
            groups[session_id].append(trace)

        # Sort each session by turn number
        for traces in groups.values():
            traces.sort(key=lambda t: t.turn)

        return dict(groups)

    # ------------------------------------------------------------------
    # Synthesis hooks
    # ------------------------------------------------------------------

    _BAILIAN_ONLY_FIELDS = frozenset(
        {
            "chat_id",
            "parent_chat_id",
            "request_type",
            "turn",
        }
    )

    def _synthesis_exclude_fields(self) -> frozenset[str]:
        return self._BAILIAN_ONLY_FIELDS

    def _synthesis_dump_kwargs(self) -> dict[str, Any]:
        return {"by_alias": True}

    def _reconstruct_traces(
        self, originals: list[BailianTrace], synth_dicts: list[dict[str, Any]]
    ) -> list[BailianTrace]:
        result: list[BailianTrace] = []
        for i, synth_dict in enumerate(synth_dicts):
            original = originals[i] if i < len(originals) else originals[-1]
            result.append(
                BailianTrace(
                    chat_id=original.chat_id,
                    parent_chat_id=original.parent_chat_id,
                    timestamp=synth_dict.get("timestamp", original.timestamp),
                    input_length=synth_dict["input_length"],
                    output_length=synth_dict["output_length"],
                    request_type=original.request_type,
                    turn=original.turn,
                    hash_ids=synth_dict.get("hash_ids", original.hash_ids),
                )
            )
        return result
