# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Light reader: weka JSON file/dir -> ParsedTurn sessions for HTML reports.

Reuses the WekaTrace pydantic models from `weka_trace_models.py` and skips
the heavy WekaTraceLoader path entirely (no tokenizer, no UserConfig, no
PromptGenerator). Output shape matches what the existing reporting pipeline
already consumes: `dict[session_id, list[ParsedTurn]]`.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaTrace,
)


def _enumerate_files(path: Path) -> list[Path]:
    """Mirror WekaTraceLoader._enumerate_files: file or sorted *.json dir."""
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [path]


def _load_weka_traces(path: Path) -> list[WekaTrace]:
    """Parse every *.json under `path` (file or dir) into WekaTrace models."""
    traces: list[WekaTrace] = []
    for file_path in _enumerate_files(path):
        blob = orjson.loads(file_path.read_bytes())
        traces.append(WekaTrace.model_validate(blob))
    return traces


def _parent_session_turns(trace: WekaTrace) -> list[ParsedTurn]:
    """Build the ParsedTurn list for a parent trace's normal/streaming requests.

    delay_ms is computed between consecutive normal requests using their
    seconds-valued `t` field (subagent entries between them do not advance
    the previous-normal pointer; their `t` is on the parent's clock and what
    matters for report distributions is the gap between consecutive normals).
    """
    turns: list[ParsedTurn] = []
    prev_t: float | None = None
    for req in trace.requests:
        if not isinstance(req, WekaNormalRequest | WekaStreamingRequest):
            continue
        delay_ms = 0.0 if prev_t is None else (req.t - prev_t) * 1000.0
        turns.append(
            ParsedTurn(
                session_id=trace.id,
                input_length=req.input_length,
                output_length=req.output_length,
                hash_ids=req.hash_ids,
                delay_ms=delay_ms,
                group_id=None,
                is_restart=False,
            )
        )
        prev_t = req.t
    return turns


def load_weka_as_parsed(path: Path) -> dict[str, list[ParsedTurn]]:
    """Read a weka trace file or directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session keyed by `trace.id`. Subagent
    handling, max_context_length filtering, and other knobs are added in
    later tasks.
    """
    traces = _load_weka_traces(path)
    parsed: dict[str, list[ParsedTurn]] = {}
    for trace in traces:
        if trace.id in parsed:
            raise ValueError(f"Duplicate trace id '{trace.id}' across input files")
        parsed[trace.id] = _parent_session_turns(trace)
    return parsed
