# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import statistics
from collections.abc import Iterator
from itertools import groupby, islice
from operator import itemgetter
from pathlib import Path
from typing import Any, TextIO

import zstandard
from pydantic import ConfigDict, Field, model_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import AIPerfBaseModel

_logger = AIPerfLogger(__name__)

# Mirrors the dataset's trace_processor.py (published beside the data); keep the two in sync.

IMAGE_PLACEHOLDER = "[Image omitted by context cleaning]"
"""Marker the agent leaves in place of a screenshot it dropped from its context."""

IMAGE_OMITTED_TEXT = f"\n\n{IMAGE_PLACEHOLDER}\n\n\n\n"

SCREENSHOT_ROLES = frozenset({"user", "tool"})
"""Observations carry the screenshots: user messages for GUI agents, tool results for tool-calling ones."""

_BISECTION_STEPS = 32


class HCuaPerfFilters(AIPerfBaseModel):
    """``--dataset-filter`` keys accepted by ``h_cua_perf``.

    Names and defaults match the options of the dataset's ``trace_processor.py``;
    its ``--num-traces`` / ``--seed`` sampling is replaced by
    ``--num-dataset-entries`` (first N eligible trajectories).
    """

    model_config = ConfigDict(extra="forbid")

    n_screenshots: int | None = Field(
        default=None,
        ge=1,
        description="Screenshots each request keeps: its own plus the preceding "
        "ones of the trajectory, restored from the records that carry them; "
        "older ones stay as the agent's placeholder text. None leaves the "
        "published single screenshot per request.",
    )
    min_trace_length: int = Field(
        default=1,
        ge=1,
        description="Drop trajectories with fewer requests; also the floor of every truncation.",
    )
    max_trace_length: int | None = Field(
        default=None,
        ge=1,
        description="Truncate trajectories to their first N requests.",
    )
    avg_trace_length: float | None = Field(
        default=None,
        gt=0,
        description="Target mean requests per trajectory, reached by truncating "
        "every trajectory by the same factor.",
    )

    @model_validator(mode="after")
    def _max_not_below_min(self) -> HCuaPerfFilters:
        if (
            self.max_trace_length is not None
            and self.max_trace_length < self.min_trace_length
        ):
            raise ValueError(
                f"max_trace_length={self.max_trace_length} is below "
                f"min_trace_length={self.min_trace_length}"
            )
        return self


def supported_filter_keys() -> str:
    """Comma-separated filter names, for error messages."""
    return ", ".join(HCuaPerfFilters.model_fields)


def select_trace_lengths(
    filters: HCuaPerfFilters,
    turns: dict[str, int],
    *,
    first_n: int | None = None,
) -> dict[str, int]:
    """Requests to keep per selected trajectory, in file order.

    Filtered on ``min_trace_length``, capped at ``max_trace_length``, cut to the
    first ``first_n`` eligible trajectories, then fitted to ``avg_trace_length``.
    """
    selected = {
        session_id: n
        if filters.max_trace_length is None
        else min(n, filters.max_trace_length)
        for session_id, n in turns.items()
        if n >= filters.min_trace_length
    }
    if first_n is not None:
        selected = dict(islice(selected.items(), first_n))
    if not selected:
        raise ValueError(f"No trajectory has {filters.min_trace_length}+ requests")
    if filters.avg_trace_length is not None:
        lengths = fit_lengths(
            list(selected.values()),
            filters.avg_trace_length,
            filters.min_trace_length,
        )
        selected = dict(zip(selected, lengths, strict=True))
    return selected


def fit_lengths(lengths: list[int], avg: float, floor: int) -> list[int]:
    """Scale every length by the smallest common factor, never below floor, whose mean reaches avg."""
    if statistics.mean(lengths) <= avg:
        _logger.warning(
            f"Mean trajectory length {statistics.mean(lengths):.1f} is already "
            f"below {avg}; nothing truncated"
        )
        return lengths
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2
        low, high = (
            (mid, high)
            if statistics.mean(_scaled(lengths, mid, floor)) < avg
            else (low, mid)
        )
    return _scaled(lengths, high, floor)


def _scaled(lengths: list[int], factor: float, floor: int) -> list[int]:
    return [max(floor, round(factor * length)) for length in lengths]


def open_dataset(path: Path) -> TextIO:
    """Open a dataset as text, decompressing on the fly when it is stored as ``.zst``."""
    if path.suffix != ".zst":
        return open(path, encoding="utf-8")
    return zstandard.open(path, "rt", encoding="utf-8")


def iter_selected_records(
    records: Iterator[dict[str, Any]],
    plan: dict[str, int],
    n_screenshots: int | None,
) -> Iterator[dict[str, Any]]:
    """Yield the planned prefix of every selected trajectory, re-windowed, in file order.

    The plan comes from the manifest and the file is verified against the
    manifest's sha256 before this runs, so the plan is trusted. Trajectories
    are contiguous in the source, so each is consumed as a group and iteration
    stops after the last planned one; the rest of the file is never read.
    """
    remaining = dict(plan)
    for session_id, session in groupby(records, key=itemgetter("session_id")):
        kept_n = remaining.pop(session_id, None)
        if kept_n is None:
            continue
        kept = list(islice(session, kept_n))
        if n_screenshots is not None:
            apply_screenshot_window(kept, n_screenshots)
        yield from kept
        if not remaining:
            return


def apply_screenshot_window(records: list[dict[str, Any]], n_screenshots: int) -> None:
    """Resize the screenshot window of every record of one trajectory, in place, to its last n_screenshots slots."""
    slots_per_record = [screenshot_slots(record["messages"]) for record in records]
    for start, stop in _history_segments(records, slots_per_record):
        _window_segment(slots_per_record[start:stop], n_screenshots)


def _history_segments(
    records: list[dict[str, Any]],
    slots_per_record: list[list[tuple[list[dict[str, Any]], int]]],
) -> Iterator[tuple[int, int]]:
    """Spans of records whose slot indexes refer to the same observations.

    Message and slot counts only grow while a trajectory's history is
    append-only, so a drop in either means the history was reset: the agent
    compacted its context, or an auxiliary model was invoked under the same
    session id. Slot indexes restart there, so windowing across the boundary
    would restore a screenshot from before the reset into a slot that now
    stands for a different observation.
    """
    start = 0
    for idx in range(1, len(records)):
        if len(slots_per_record[idx]) < len(slots_per_record[idx - 1]) or len(
            records[idx]["messages"]
        ) < len(records[idx - 1]["messages"]):
            yield start, idx
            start = idx
    yield start, len(records)


def _window_segment(
    slots_per_record: list[list[tuple[list[dict[str, Any]], int]]],
    n_screenshots: int,
) -> None:
    images_by_slot: dict[int, dict[str, Any]] = {}
    for slots in slots_per_record:
        for slot_idx, (parts, part_idx) in enumerate(slots):
            if _is_image(parts[part_idx]):
                images_by_slot.setdefault(slot_idx, parts[part_idx])

    for slots in slots_per_record:
        first_kept = max(0, len(slots) - n_screenshots)
        for slot_idx, (parts, part_idx) in enumerate(slots):
            if _is_image(parts[part_idx]) and slot_idx < first_kept:
                drop_screenshot(parts, part_idx)
            elif (
                not _is_image(parts[part_idx])
                and slot_idx >= first_kept
                and slot_idx in images_by_slot
            ):
                _restore_screenshot(parts, part_idx, images_by_slot[slot_idx])


def screenshot_slots(
    messages: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], int]]:
    """A slot is a screenshot part, or the placeholder standing for a dropped one, in any observation message."""
    return [
        (message["content"], part_idx)
        for message in messages
        if message.get("role") in SCREENSHOT_ROLES
        and isinstance(message.get("content"), list)
        for part_idx, part in enumerate(message["content"])
        if _is_image(part) or IMAGE_PLACEHOLDER in part.get("text", "")
    ]


def drop_screenshot(parts: list[dict[str, Any]], part_idx: int) -> None:
    """Replace a screenshot by the placeholder, merged into the text parts surrounding it."""
    before, after = _text(parts, part_idx - 1), _text(parts, part_idx + 1)
    start = part_idx if before is None else part_idx - 1
    end = part_idx + 1 if after is None else part_idx + 2
    parts[start:end] = [
        {"type": "text", "text": (before or "") + IMAGE_OMITTED_TEXT + (after or "")}
    ]


def _restore_screenshot(
    parts: list[dict[str, Any]], part_idx: int, image: dict[str, Any]
) -> None:
    """Split a placeholder text part back into text, screenshot and text."""
    text = parts[part_idx]["text"]
    marker = IMAGE_OMITTED_TEXT if IMAGE_OMITTED_TEXT in text else IMAGE_PLACEHOLDER
    before, _, after = text.partition(marker)
    # Empty sides are dropped: the placeholder often begins or ends its text
    # part, and a zero-length text part is a malformed content part on the wire.
    parts[part_idx : part_idx + 1] = [
        *([{"type": "text", "text": before}] if before else []),
        image,
        *([{"type": "text", "text": after}] if after else []),
    ]


def _is_image(part: dict[str, Any]) -> bool:
    return part.get("type") == "image_url"


def _text(parts: list[dict[str, Any]], part_idx: int) -> str | None:
    if not 0 <= part_idx < len(parts):
        return None
    return parts[part_idx]["text"] if parts[part_idx].get("type") == "text" else None
