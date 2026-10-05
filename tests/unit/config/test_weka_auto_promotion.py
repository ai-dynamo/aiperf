# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-promotion and timing validation must agree on what carries timing.

They did not. Validation consulted ``_implicit_timing_types`` -- loaders whose
timing is nested where a first-record probe cannot see it -- while
auto-promotion used a probe that only looked for a top-level ``timestamp`` key
and returned False for a directory. A weka_trace keeps timing in
``requests[].t`` and is documented as a *directory* of files, so it failed that
probe, never auto-promoted, and replayed as plain concurrency: a green run that
silently discarded the recorded timeline, which is the entire reason to replay
a trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset.resolver import _implicit_timing_types
from aiperf.config.flags._converter_profiling import _trace_carries_timing

BLOCK = 64


def _weka_trace(tmp_path: Path) -> Path:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "trace_0001.json").write_text(
        json.dumps(
            {
                "id": "demo",
                "models": ["m"],
                "block_size": BLOCK,
                "hash_id_scope": "local",
                "tool_tokens": 0,
                "system_tokens": 0,
                "requests": [
                    {
                        "t": 0.0,
                        "type": "n",
                        "model": "m",
                        "in": 4 * BLOCK,
                        "out": 16,
                        "hash_ids": [1, 2, 3, 4],
                        "stop": "end_turn",
                    }
                ],
            }
        )
    )
    return traces


def test_weka_trace_directory_is_recognised_as_carrying_timing(tmp_path) -> None:
    """The documented input shape: a directory, with timing nested per request."""
    assert _trace_carries_timing("weka_trace", _weka_trace(tmp_path)) is True


def test_weka_trace_single_file_is_recognised_too(tmp_path) -> None:
    traces = _weka_trace(tmp_path)
    assert _trace_carries_timing("weka_trace", traces / "trace_0001.json") is True


@pytest.mark.parametrize(
    "dataset_type",
    [param(str(t), id=str(t)) for t in sorted(str(x) for x in _implicit_timing_types())],
)  # fmt: skip
def test_every_implicit_timing_type_auto_promotes(dataset_type: str, tmp_path) -> None:
    """Whatever validation accepts as timed, auto-promotion must also accept.

    Pinning the whole set stops the two notions drifting apart again, which is
    the defect this guards.
    """
    assert _trace_carries_timing(dataset_type, tmp_path / "does-not-matter") is True


def test_an_untimed_dataset_still_falls_back_to_the_probe(tmp_path) -> None:
    """Scoping: a format with no implicit timing must not be promoted blindly."""
    plain = tmp_path / "plain.jsonl"
    plain.write_text(json.dumps({"text": "hello"}) + "\n")
    assert _trace_carries_timing("mooncake_trace", plain) is False


def test_a_timestamped_record_still_promotes(tmp_path) -> None:
    timed = tmp_path / "timed.jsonl"
    timed.write_text(json.dumps({"timestamp": 0, "text": "hello"}) + "\n")
    assert _trace_carries_timing("mooncake_trace", timed) is True
