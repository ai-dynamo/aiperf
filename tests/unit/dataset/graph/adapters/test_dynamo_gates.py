# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-loud gates raised by ``from_dynamo_trace`` itself: block alignment and mixed ``trace_block_size``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
)
from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
    DynamoISLMismatchError,
)

from .conftest import write_jsonl


def _rec(
    *,
    ts: int,
    sid: str,
    input_tokens: int,
    output_tokens: int,
    hashes: list[int] | None = None,
    block_size: int = 16,
    input_length: int | None = None,
) -> dict:
    """One current-schema ``dynamo.request.trace.v1`` ``request_end`` record."""
    req: dict = {
        "request_id": f"r{sid}{ts}",
        "model": "m",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
    }
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": block_size,
            "input_length": input_length if input_length is not None else input_tokens,
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


@pytest.mark.parametrize(
    "name, records, expected_error, match",
    [
        # input_length=100 cannot be spanned by 2 hashes at bs=16 (16 < 100 > 32),
        # so no reconstruction can honor both fields.
        param(
            "bad_isl.jsonl",
            [
                _rec(
                    ts=1000,
                    sid="s1",
                    input_tokens=100,
                    output_tokens=8,
                    hashes=[111, 222],
                    block_size=16,
                    input_length=100,
                )
            ],
            DynamoISLMismatchError,
            None,
            id="isl_not_spanned_by_hashes_at_recorded_block_size",
        ),
        param(
            "mixed_bs.jsonl",
            [
                _rec(
                    ts=1000, sid="s1", input_tokens=32, output_tokens=8, hashes=[1, 2]
                ),
                _rec(
                    ts=2000,
                    sid="s1",
                    input_tokens=64,
                    output_tokens=8,
                    hashes=[1, 2, 3, 4],
                    block_size=32,
                    input_length=128,
                ),
            ],
            DynamoTraceAdapterError,
            "trace_block_size",
            id="two_replay_turns_disagree_on_trace_block_size",
        ),
    ],
)  # fmt: skip
def test_malformed_replay_metadata_aborts_parse(
    tmp_path: Path,
    name: str,
    records: list[dict[str, Any]],
    expected_error: type[Exception],
    match: str | None,
) -> None:
    """Unreconstructable replay metadata aborts the parse instead of guessing (the segment trie has no separate build pass)."""
    p = write_jsonl(tmp_path / name, records)
    with pytest.raises(expected_error, match=match):
        from_dynamo_trace(p)
