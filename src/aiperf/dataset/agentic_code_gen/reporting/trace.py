# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace parsing helpers for Agentic Code reports and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import Field, ValidationError

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.agentic_code_gen.models import SynthesizedSession
from aiperf.dataset.loader.models import MooncakeTrace


class ParsedTurn(AIPerfBaseModel):
    """A parsed request turn from dataset.jsonl."""

    session_id: str = Field(description="Session identifier")
    input_length: int = Field(description="Input token count for this JSONL row")
    output_length: int = Field(description="Output token count for this turn")
    hash_ids: list[int] = Field(description="KV cache block hash IDs")
    delay_ms: float = Field(
        description="Inter-turn delay in milliseconds, 0.0 for first turn"
    )
    group_id: int | None = Field(
        default=None, description="Shared-prefix group id, emitted on turn 0"
    )
    is_restart: bool = Field(
        default=False, description="Whether this row starts a restart continuation"
    )


def iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows as dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(orjson.loads(line))
    return rows


def load_jsonl(path: Path) -> list[ParsedTurn]:
    """Load dataset.jsonl rows into ParsedTurn objects."""
    turns: list[ParsedTurn] = []
    for row in iter_jsonl_rows(path):
        turns.append(
            ParsedTurn(
                session_id=row["session_id"],
                input_length=row["input_length"],
                output_length=row["output_length"],
                hash_ids=row.get("hash_ids", []),
                delay_ms=row.get("delay", row.get("timestamp", 0.0)),
                group_id=row.get("group_id"),
                is_restart=row.get("is_restart", False),
            )
        )
    return turns


def group_sessions(turns: list[ParsedTurn]) -> dict[str, list[ParsedTurn]]:
    """Group ParsedTurn rows by session id while preserving input order."""
    sessions: dict[str, list[ParsedTurn]] = {}
    for turn in turns:
        sessions.setdefault(turn.session_id, []).append(turn)
    return sessions


def synthesized_sessions_to_parsed(
    sessions: list[SynthesizedSession],
) -> dict[str, list[ParsedTurn]]:
    """Convert generated sessions into the parsed trace shape used by reports."""
    parsed: dict[str, list[ParsedTurn]] = {}
    for session in sessions:
        parsed[session.session_id] = [
            ParsedTurn(
                session_id=session.session_id,
                input_length=t.new_tokens,
                output_length=t.output_length,
                hash_ids=t.hash_ids,
                delay_ms=t.delay_ms,
                group_id=session.group_id if t.turn_index == 0 else None,
                is_restart=session.is_restart_continuation
                if t.turn_index == 0
                else False,
            )
            for t in session.turns
        ]
    return parsed


def validate_mooncake_trace(path: Path, max_errors: int = 10) -> tuple[int, list[str]]:
    """Validate a dataset JSONL file against the Mooncake trace row model."""
    errors: list[str] = []
    line_count = 0

    with path.open("rb") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                MooncakeTrace(**orjson.loads(line))
            except (orjson.JSONDecodeError, ValidationError) as e:
                errors.append(f"Line {line_num}: {e}")
                if len(errors) >= max_errors:
                    break

    return line_count, errors


def load_simulation_sessions(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load JSONL into the dictionary shape consumed by simulation rendering.

    JSONL rows contain incremental input_length values. The returned structure
    reconstructs cumulative_input_length so the simulation can track total
    in-flight ISL for each turn.
    """
    grouped = group_sessions(load_jsonl(jsonl_path))
    result: list[dict[str, Any]] = []
    for session_id, turns in grouped.items():
        cumulative = 0
        sim_turns: list[dict[str, Any]] = []
        for turn in turns:
            cumulative += turn.input_length
            sim_turns.append(
                {
                    "input_length": turn.input_length,
                    "output_length": turn.output_length,
                    "delay_ms": turn.delay_ms,
                    "hash_ids": turn.hash_ids,
                    "cumulative_input_length": cumulative,
                }
            )
            cumulative += turn.output_length

        first = turns[0]
        result.append(
            {
                "session_id": session_id,
                "group_id": first.group_id if first.group_id is not None else 0,
                "is_restart": first.is_restart,
                "turns": sim_turns,
            }
        )
    return result
