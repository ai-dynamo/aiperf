# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo adapter output locked through the full graph validator.

Defeats the adapter-tests-skip-validator trap (the weka adapter locks its
output the same way, ``test_start_anchor_weka_stamping``): structural
regressions in the trie lowering -- dangling rule-56 edge endpoints, rule-1
cycles, rule-55 anchor-shape violations -- would otherwise pass a suite that
only asserts node-level fields. Every ``from_dynamo_trace`` shape (linear
recorded, virtual-hash fallback, nested/subagent, parallel, multi-root) must
produce ZERO ERROR-severity issues from :func:`validate`. WARNING-severity
issues are allowed: dynamo stamps ``expected.cache_read_tokens`` from the
recorded ``cached_tokens``, which rule-15 flags as informational.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.validator import (
    ValidationIssue,
    ValidationSeverity,
    validate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dynamo_nested"


def _blocking(parsed: ParsedGraph) -> list[ValidationIssue]:
    return [i for i in validate(parsed) if i.severity is ValidationSeverity.ERROR]


def _rec(
    *,
    ts: int,
    sid: str,
    input_tokens: int,
    output_tokens: int,
    hashes: list[int] | None = None,
) -> dict:
    req: dict = {
        "request_id": f"r{ts}",
        "model": "m",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
    }
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": 16,
            "input_length": input_tokens,
            "input_sequence_hashes": hashes,
        }
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": req,
    }


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


@pytest.mark.parametrize(
    "fixture_name",
    [
        param("nested_2_level.jsonl.gz", id="nested_2_level"),
        param("nested_3_level.jsonl.gz", id="nested_3_level"),
        param("parallel_subagents.jsonl.gz", id="parallel_subagents"),
        param("parallel_two_root.jsonl.gz", id="parallel_two_root_multi_root"),
        param("mixed_turn.jsonl.gz", id="mixed_turn_tool_events"),
        param("tool_call_id_linkage.jsonl.gz", id="tool_call_id_linkage"),
    ],
)  # fmt: skip
def test_nested_fixture_output_has_no_validator_errors(fixture_name: str) -> None:
    parsed = from_dynamo_trace(FIXTURES / fixture_name)
    blocking = _blocking(parsed)
    assert blocking == [], blocking


def test_linear_recorded_trace_output_has_no_validator_errors(tmp_path: Path) -> None:
    """Trie lowering on recorded replay hashes (turn 2 extends turn 1)."""
    p = _write(
        tmp_path,
        "linear_recorded.jsonl",
        [
            _rec(
                ts=1000, sid="s1", input_tokens=32, output_tokens=8, hashes=[111, 222]
            ),
            _rec(
                ts=2000,
                sid="s1",
                input_tokens=64,
                output_tokens=12,
                hashes=[111, 222, 333, 444],
            ),
        ],
    )
    parsed = from_dynamo_trace(p)
    blocking = _blocking(parsed)
    assert blocking == [], blocking


def test_linear_virtual_fallback_output_has_no_validator_errors(
    tmp_path: Path,
) -> None:
    """Trie lowering on the virtual-hash fallback (no replay metadata)."""
    p = _write(
        tmp_path,
        "linear_virtual.jsonl",
        [
            _rec(ts=1000, sid="s1", input_tokens=32, output_tokens=8),
            _rec(ts=2000, sid="s1", input_tokens=64, output_tokens=12),
        ],
    )
    parsed = from_dynamo_trace(p)
    assert "virtual-hash-fallback" in parsed.traces[0].tags
    blocking = _blocking(parsed)
    assert blocking == [], blocking


def test_replayed_delays_output_has_no_validator_errors(tmp_path: Path) -> None:
    """The default replay keeps recorded edge delays; rule-54/57 must still pass."""
    p = _write(
        tmp_path,
        "linear_cadence.jsonl",
        [
            _rec(
                ts=1000, sid="s1", input_tokens=32, output_tokens=8, hashes=[111, 222]
            ),
            _rec(
                ts=2000,
                sid="s1",
                input_tokens=64,
                output_tokens=12,
                hashes=[111, 222, 333, 444],
            ),
        ],
    )
    parsed = from_dynamo_trace(p)
    blocking = _blocking(parsed)
    assert blocking == [], blocking
