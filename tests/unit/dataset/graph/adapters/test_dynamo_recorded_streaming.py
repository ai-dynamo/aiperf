# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo replay streaming follows recorded TTFT unless the run overrides it."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from tests.unit.dataset.graph.adapters.conftest import write_jsonl


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


@pytest.mark.parametrize(
    "ttft_ms,expected_streaming",
    [
        param(250.0, True, id="ttft_present_streams"),
        param(None, False, id="ttft_absent_does_not_stream"),
        # A recorded 0.0 still streamed: the derivation is is-not-None, never truthiness.
        param(0.0, True, id="ttft_zero_still_streams"),
    ],
)  # fmt: skip
def test_streaming_derived_from_recorded_ttft(
    tmp_path: Path, ttft_ms: float | None, expected_streaming: bool
) -> None:
    """The lowered node's ``streaming`` mode follows presence of a recorded ``ttft_ms``, and the build-plane envelope carries it to the worker's per-request stream override."""
    p = write_jsonl(
        tmp_path / "trace.jsonl", [_rec(ts=1000, sid="s1", ttft_ms=ttft_ms)]
    )

    pb = from_dynamo_trace(p)

    assert pb.graph.nodes["s1:0"].streaming is expected_streaming


def test_streaming_override_forces_all_dynamo_requests(tmp_path: Path) -> None:
    """The run-level streaming flag overrides missing recorded TTFT data."""
    p = write_jsonl(tmp_path / "trace.jsonl", [_rec(ts=1000, sid="s1")])

    pb = from_dynamo_trace(p, streaming=True)

    assert pb.graph.nodes["s1:0"].streaming is True
