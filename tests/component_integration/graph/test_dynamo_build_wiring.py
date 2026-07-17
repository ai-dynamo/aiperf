# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DatasetManager routes a dynamo graph into the unified store (unconditional).

The dynamo trie parse stamps per-node ``prompt_segment_ids`` + a
``segment_pool`` at parse time (shared LCP segment-trie core), so the store
build streams the stamped parse's per-trace payloads into a
:class:`GraphSegmentUnifiedBackingStore` via
:func:`build_unified_trie_store_from_payloads` (the same drain the weka pool
path uses). This test drives the whole path through
``_configure_graph_workload`` and asserts a known node's manifest materializes
from the unified store (no ``GraphEnvelopeMissing`` / empty store).
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    """One current-schema ``dynamo.request.trace.v1`` ``request_end`` record."""
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": {
            "request_id": f"r{ts}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": 8,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_tokens,
                "input_sequence_hashes": hashes,
            },
        },
    }


@pytest.fixture
def dyn_fixture(tmp_path: Path) -> Path:
    """A minimal single-session two-turn dynamo trace (recorded replay hashes)."""
    p = tmp_path / "dyn_min.jsonl"
    records = [
        _dynamo_record(1000, "s1", 32, [111, 222]),
        _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


async def test_dynamo_build_populates_unified_store(
    dyn_fixture: Path,
    mmap_base_path: Path,
) -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
            input_file=str(dyn_fixture),
            random_seed=1234,
        )
    )
    dm = DatasetManager(run=run, service_id="test")

    convs = await dm._configure_graph_workload(dyn_fixture)
    assert convs.trace_ids, "dynamo build produced no graph traces"

    # Known node: trace 's1', ordinal 0 (the first LlmNode). Its profiling manifest
    # must materialize from the unified store -- proving the dynamo build wrote the
    # replay-derived content pool + node manifests, never left empty.
    with GraphSegmentUnifiedClient(mmap_base_path, dm.run.benchmark_id).open() as c:
        raw = c.get_node_envelope("s1", 0, "profiling")
        assert raw is not None, "node (s1, 0) missing from unified store"
        envelope = orjson.loads(raw)
        assert "handles" in envelope, f"unexpected manifest envelope: {envelope!r}"
        assert envelope["handles"], "node manifest has no segment handles"
        msgs = c.materialize_handles(envelope["handles"])
        assert msgs and all(set(m) == {"role", "content"} for m in msgs)
