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
    WekaSubagentEntry,
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


def _subagent_session_turns(
    trace_id: str, entry: WekaSubagentEntry
) -> tuple[str, list[ParsedTurn]]:
    """Build (session_id, turns) for one subagent entry's nested normal requests.

    delay_ms is computed within the subagent's own request list, so the
    subagent's first turn always has delay_ms=0.0 (matches the convention
    used for parent-session turn 0).
    """
    session_id = f"{trace_id}::sa:{entry.agent_id}"
    turns: list[ParsedTurn] = []
    prev_t: float | None = None
    for req in entry.requests:
        delay_ms = 0.0 if prev_t is None else (req.t - prev_t) * 1000.0
        turns.append(
            ParsedTurn(
                session_id=session_id,
                input_length=req.input_length,
                output_length=req.output_length,
                hash_ids=req.hash_ids,
                delay_ms=delay_ms,
                group_id=None,
                is_restart=False,
            )
        )
        prev_t = req.t
    return session_id, turns


def _trace_peak_input_length(trace: WekaTrace) -> int:
    """Peak `input_length` across parent and subagent requests.

    Mirrors WekaTraceLoader._filter_traces_by_max_context's rule.
    """
    peak = 0
    for req in trace.requests:
        if (
            isinstance(req, WekaNormalRequest | WekaStreamingRequest)
            and req.input_length > peak
        ):
            peak = req.input_length
        elif isinstance(req, WekaSubagentEntry):
            for child_req in req.requests:
                if child_req.input_length > peak:
                    peak = child_req.input_length
    return peak


def load_weka_as_parsed(
    path: Path,
    *,
    include_subagents: bool = True,
    max_context_length: int | None = None,
) -> dict[str, list[ParsedTurn]]:
    """Read a weka trace file or directory of *.json into ParsedTurn sessions.

    Each parent trace becomes one session keyed by `trace.id`. When
    include_subagents=True (default), each `WekaSubagentEntry` in the parent's
    request list also becomes a session keyed by `f"{trace.id}::sa:{agent_id}"`.

    When max_context_length is set, traces whose peak input_length
    exceeds the cap are dropped entirely (parent and subagents).
    """
    traces = _load_weka_traces(path)
    parsed: dict[str, list[ParsedTurn]] = {}
    for trace in traces:
        if (
            max_context_length is not None
            and _trace_peak_input_length(trace) > max_context_length
        ):
            continue
        if trace.id in parsed:
            raise ValueError(f"Duplicate trace id '{trace.id}' across input files")
        parsed[trace.id] = _parent_session_turns(trace)
        if include_subagents:
            for req in trace.requests:
                if isinstance(req, WekaSubagentEntry):
                    sid, turns = _subagent_session_turns(trace.id, req)
                    if sid in parsed:
                        raise ValueError(
                            f"Duplicate subagent session id '{sid}' in trace "
                            f"'{trace.id}'"
                        )
                    parsed[sid] = turns
    return parsed


def infer_weka_block_size(path: Path, max_context_length: int | None = None) -> int:
    """Return the single block_size used by matching weka trace files."""
    block_sizes: set[int] = set()
    for trace in _load_weka_traces(path):
        if (
            max_context_length is not None
            and _trace_peak_input_length(trace) > max_context_length
        ):
            continue
        block_sizes.add(trace.block_size)
    if not block_sizes:
        raise ValueError("No weka traces matched the input")
    if len(block_sizes) > 1:
        values = ", ".join(str(v) for v in sorted(block_sizes))
        raise ValueError(f"Weka traces use multiple block sizes: {values}")
    return next(iter(block_sizes))


def parsed_to_sim_sessions(
    parsed: dict[str, list[ParsedTurn]],
) -> list[dict]:
    """Convert ParsedTurn sessions to the dict shape `render_simulation` expects.

    Weka trace input_length is already cumulative context at that turn. The
    simulation shape also includes per-turn incremental input_length, so derive
    the delta from the previous cumulative input and output.
    """
    result: list[dict] = []
    for session_id, turns in parsed.items():
        prev_input_length = 0
        prev_output_length = 0
        sim_turns: list[dict] = []
        for i, turn in enumerate(turns):
            input_delta = (
                turn.input_length
                if i == 0
                else max(turn.input_length - prev_input_length - prev_output_length, 0)
            )
            sim_turns.append(
                {
                    "input_length": input_delta,
                    "output_length": turn.output_length,
                    "delay_ms": turn.delay_ms,
                    "hash_ids": turn.hash_ids,
                    "cumulative_input_length": turn.input_length,
                }
            )
            prev_input_length = turn.input_length
            prev_output_length = turn.output_length

        first = turns[0] if turns else None
        result.append(
            {
                "session_id": session_id,
                "group_id": first.group_id
                if first and first.group_id is not None
                else 0,
                "is_restart": first.is_restart if first else False,
                "turns": sim_turns,
            }
        )
    return result
