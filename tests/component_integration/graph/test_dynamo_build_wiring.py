# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DatasetManager unconditionally routes a dynamo graph's stamped parse into the unified segment store."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from tests.component_integration.graph.conftest import (
    dynamo_request_end,
    dynamo_run,
    write_dynamo_jsonl,
)

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
def dyn_fixture(tmp_path: Path) -> Path:
    """A minimal single-session two-turn dynamo trace with recorded replay hashes."""
    return write_dynamo_jsonl(
        tmp_path / "dyn_min.jsonl",
        [
            dynamo_request_end(
                ts=1000, session_id="s1", hashes=[111, 222], output_tokens=8
            ),
            dynamo_request_end(
                ts=2000,
                session_id="s1",
                hashes=[111, 222, 333, 444],
                output_tokens=8,
            ),
        ],
    )


async def test_dynamo_build_populates_unified_store(
    dyn_fixture: Path,
    mmap_base_path: Path,
) -> None:
    """A known node's profiling manifest materializes from the store the dynamo build wrote."""
    dm = DatasetManager(run=dynamo_run(dyn_fixture), service_id="test")

    convs = await dm._configure_graph_workload(dyn_fixture)
    assert convs.trace_ids, "dynamo build produced no graph traces"

    # Known node: trace 's1', ordinal 0 (the first LlmNode). Its profiling manifest
    # must materialize from the unified store -- proving the dynamo build wrote the
    # replay-derived content pool + node manifests, never left empty.
    with GraphSegmentUnifiedClient(mmap_base_path, dm.run.benchmark_id).open() as c:
        raw = c.get_node_envelope("s1", 0)
        assert raw is not None, "node (s1, 0) missing from unified store"
        envelope = orjson.loads(raw)
        assert "handles" in envelope, f"unexpected manifest envelope: {envelope!r}"
        assert envelope["handles"], "node manifest has no segment handles"
        msgs = c.materialize_handles(envelope["handles"])
        assert msgs and all(set(m) == {"role", "content"} for m in msgs)
