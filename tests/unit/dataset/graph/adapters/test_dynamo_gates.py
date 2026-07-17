# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-loud gates of the dynamo trie parse.

Two gates, both raised from :func:`from_dynamo_trace` itself (the trie IR has
no separate build pass):

* Block alignment: a recorded ``input_length`` not spanned by its replay
  hashes at the recorded ``trace_block_size`` raises
  :class:`DynamoISLMismatchError` -- no reconstruction can honor both fields.
* Mixed block sizes: two replay turns recording different ``trace_block_size``
  values raise :class:`DynamoTraceAdapterError`.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
)
from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
    DynamoISLMismatchError,
)


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


def _write(tmp_path: Path, name: str, records: list[dict]) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


def test_isl_mismatch_aborts_parse(tmp_path):
    """input_length=100 cannot be spanned by 2 hashes at bs=16 (16 < 100 > 32)."""
    p = _write(
        tmp_path,
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
    )
    with pytest.raises(DynamoISLMismatchError):
        from_dynamo_trace(p)


def test_mixed_block_size_aborts_parse(tmp_path):
    p = _write(
        tmp_path,
        "mixed_bs.jsonl",
        [
            _rec(ts=1000, sid="s1", input_tokens=32, output_tokens=8, hashes=[1, 2]),
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
    )
    with pytest.raises(DynamoTraceAdapterError, match="trace_block_size"):
        from_dynamo_trace(p)
