# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A build whose KV hints were synthesized must say so at run time.

``VIRTUAL_HASH_FALLBACK_TAG`` was written to ``TraceRecord.tags`` and read by
nothing outside ``aiperf dynamo trace-report``, so a run on a partly-unrecorded
corpus was indistinguishable from a fully recorded one.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
    VIRTUAL_HASH_FALLBACK_TAG,
)
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.dataset.graph.store_build import GraphStoreBuilder


def _parsed(tagged: int, total: int) -> ParsedGraph:
    """A ParsedGraph with `tagged` of `total` traces carrying the fallback tag."""
    traces = [
        TraceRecord(
            id=f"t-{i}",
            tags=[VIRTUAL_HASH_FALLBACK_TAG] if i < tagged else [],
        )
        for i in range(total)
    ]
    return ParsedGraph(graph=GraphRecord(nodes={}, edges=[]), traces=traces)


def _advise(parsed: ParsedGraph) -> list[str]:
    """Run the advisory against a stub builder; return emitted notice strings."""
    builder = MagicMock(spec=GraphStoreBuilder)
    emitted: list[str] = []
    builder.notice = lambda msg: emitted.append(msg() if callable(msg) else str(msg))
    GraphStoreBuilder._advise_virtual_hash_fallback(builder, parsed)
    return emitted


@pytest.mark.parametrize(
    "tagged,total",
    [
        param(1, 4, id="some-tagged"),
        param(4, 4, id="all-tagged"),
    ],
)  # fmt: skip
def test_notice_reports_tagged_and_total(tagged: int, total: int) -> None:
    """The notice names how many traces of how many fell back."""
    emitted = _advise(_parsed(tagged, total))
    assert len(emitted) == 1
    assert f"{tagged} of {total}" in emitted[0]


def test_silent_when_every_trace_carries_recorded_hashes() -> None:
    """A fully recorded corpus emits nothing -- this is not a per-run banner."""
    assert _advise(_parsed(0, 4)) == []


def test_sidecar_writer_calls_the_advisory() -> None:
    """The advisory must be WIRED, not merely correct.

    The tests above drive the method directly, so they pass whether or not
    anything calls it. ``_write_graph_sidecar`` is the one chokepoint every
    graph route lands on; if the call is dropped there, the tag goes back to
    being invisible at run time.
    """
    import inspect

    src = inspect.getsource(GraphStoreBuilder._write_graph_sidecar)
    assert "_advise_virtual_hash_fallback(parsed)" in src


def test_notice_does_not_claim_the_metrics_are_wrong() -> None:
    """It must describe synthesized reuse, not assert the run is invalid.

    A virtual-hash run still replays; only its prefix reuse is synthetic. The
    wording has to keep that distinction or it becomes the same overclaiming
    this branch spent its time removing from the replay-wait advisory.
    """
    text = _advise(_parsed(1, 4))[0].lower()
    assert "synthesi" in text
    assert "invalid" not in text and "wrong" not in text
