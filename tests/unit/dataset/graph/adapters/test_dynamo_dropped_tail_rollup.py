# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dropped partial block tails must reach the operator from the PARALLEL paths.

The rollup was logged inside ``_build_trees_sequential``, which pool workers
call. A worker's root logger has no handler (the forkserver preload carries no
logging setup), so on every parallel build -- the default for dynamo -- the
line was discarded and prompts were silently shorter than the recording. The
counts now ride back in the worker's result and the parent logs once.
"""

from __future__ import annotations

import inspect

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo import trace_parallel
from aiperf.dataset.graph.adapters.dynamo.trace import (
    _build_trees_sequential,
    format_dropped_tail_rollup,
)


def test_rollup_text_names_turns_trees_and_tokens() -> None:
    """The one operator-facing line carries all three counts."""
    msg = format_dropped_tail_rollup(
        nodes=14_792, tokens=469_017, trees=12_507, recorded_tokens=817_982_749
    )
    # EVERY count is thousands-separated -- these are six- and nine-digit
    # figures on a real corpus, and an unseparated 817982749 is unreadable at
    # a glance in a log line.
    assert "14,792 turn(s)" in msg
    assert "12,507 tree(s)" in msg
    assert "469,017 tokens" in msg
    assert "817,982,749 recorded input tokens" in msg
    assert "31.7/turn" in msg
    assert "0.06%" in msg


def test_rollup_separates_a_four_digit_per_turn_mean() -> None:
    """The per-turn mean is separated too; a huge block size makes it four digits."""
    msg = format_dropped_tail_rollup(
        nodes=2, tokens=9_000, trees=1, recorded_tokens=100_000
    )
    assert "4,500.0/turn" in msg


@pytest.mark.parametrize(
    "nodes,tokens,recorded",
    [
        param(0, 0, 0, id="nothing-dropped"),
        param(3, 60, 0, id="zero-denominator"),
    ],
)  # fmt: skip
def test_rollup_text_survives_degenerate_counts(
    nodes: int, tokens: int, recorded: int
) -> None:
    """No ZeroDivisionError when a corpus drops nothing or records nothing.

    The formatter runs on whatever the workers summed; a guard here is cheaper
    than a crash in the middle of a long build.
    """
    assert format_dropped_tail_rollup(nodes, tokens, 1, recorded)


def test_builder_accepts_a_tails_out_channel() -> None:
    """``_build_trees_sequential`` exposes the out-param workers report through.

    Signature guard: the worker paths pass ``tails_out``, so losing the
    parameter silently reverts them to the discarded-log behavior.
    """
    assert "tails_out" in inspect.signature(_build_trees_sequential).parameters


@pytest.mark.parametrize(
    "fn_name",
    [
        param("_build_batch_file_to_blob", id="fused-blob-worker"),
        param("_build_batch_file_to_payloads", id="streaming-payload-worker"),
    ],
)  # fmt: skip
def test_worker_reports_tails_instead_of_logging(fn_name: str) -> None:
    """Both pool workers must request the counts and must NOT log them.

    Two separate worker entries with two different return contracts; a fix
    applied to one only would leave the other silent.
    """
    src = inspect.getsource(getattr(trace_parallel, fn_name))
    assert "tails_out=tails" in src, f"{fn_name} must collect dropped-tail counts"
    assert "_logger" not in src, (
        f"{fn_name} runs in a pool worker with no log handler; it must return "
        "counts, never log them"
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        param("stream_dynamo_trace_segment_payloads", id="streaming-parent"),
        param("_build_fused_parallel", id="fused-parent"),
    ],
)  # fmt: skip
def test_parent_emits_the_rollup(fn_name: str) -> None:
    """Each parent-side consumer sums worker counts and emits the rollup once."""
    src = inspect.getsource(getattr(trace_parallel, fn_name))
    assert (
        "_log_dropped_tail_rollup(tail_nodes, tail_tokens, tail_trees, tail_recorded)"
        in src
    )


@pytest.mark.parametrize(
    "nodes,expect_logged",
    [
        param(3, True, id="tails-dropped"),
        param(0, False, id="nothing-dropped"),
    ],
)  # fmt: skip
def test_rollup_is_silent_when_nothing_was_dropped(
    nodes: int, expect_logged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block-aligned corpus logs nothing; a zero line would train readers to skip it."""
    emitted: list[str] = []
    monkeypatch.setattr(
        trace_parallel._logger, "info", lambda msg: emitted.append(str(msg))
    )
    trace_parallel._log_dropped_tail_rollup(nodes, 128, 2, 4096)
    assert bool(emitted) is expect_logged


def test_encode_decode_round_trips_the_counts() -> None:
    """The cross-process frame preserves the counts the parent rolls up."""
    blob = trace_parallel._encode_batch_result([], (4, 512, 2, 8192))
    per_tree, tails = trace_parallel._decode_batch_result(blob)
    assert per_tree == []
    assert tails == (4, 512, 2, 8192)
