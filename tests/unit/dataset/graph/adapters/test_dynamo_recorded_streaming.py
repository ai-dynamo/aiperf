# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo recorded-mode streaming derivation through ``from_dynamo_trace``.

A recorded ``ttft_ms`` proves the original request streamed, so the lowered
``LlmNode.streaming`` must be ``ttft is not None`` (mirroring weka's recorded
``"n"``/``"s"`` discriminator); the build-plane envelope carries the mode to
the worker's per-request stream override from this native field.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace


def _rec(*, ts: int, sid: str, ttft_ms: float | None = None) -> dict:
    req: dict = {
        "request_id": f"{sid}-{ts}",
        "model": "m",
        "input_tokens": 32,
        "output_tokens": 16,
        "cached_tokens": 0,
        "request_received_ms": ts - 100,
        "total_time_ms": 100.0,
    }
    if ttft_ms is not None:
        req["ttft_ms"] = ttft_ms
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": req,
    }


def _write(tmp_path: Path, record: dict) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_bytes(orjson.dumps(record))
    return p


def test_ttft_bearing_turn_is_streaming(tmp_path: Path) -> None:
    """A recorded ``ttft_ms`` lowers to a streaming node + stream override."""
    p = _write(tmp_path, _rec(ts=1000, sid="s1", ttft_ms=250.0))
    pb = from_dynamo_trace(p)
    node = pb.graph.nodes["s1:0"]
    assert node.streaming is True


def test_ttft_absent_turn_is_not_streaming(tmp_path: Path) -> None:
    """No recorded ``ttft_ms`` lowers to a non-streaming node + stream override."""
    p = _write(tmp_path, _rec(ts=1000, sid="s1"))
    pb = from_dynamo_trace(p)
    node = pb.graph.nodes["s1:0"]
    assert node.streaming is False


def test_ttft_zero_is_streaming(tmp_path: Path) -> None:
    """``ttft_ms=0.0`` still streamed -- derivation is is-not-None, not truthiness."""
    p = _write(tmp_path, _rec(ts=1000, sid="s1", ttft_ms=0.0))
    pb = from_dynamo_trace(p)
    node = pb.graph.nodes["s1:0"]
    assert node.streaming is True
