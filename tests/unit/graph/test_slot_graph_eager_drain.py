# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every non-weka graph takes the in-process interned drain.

``GraphStoreBuilder._build_graph_store_streaming`` routes only ``weka_trace`` through the
worker-pool payload/trie drain; every other format (here: ``native``) parses
once in-process and drains that SAME parse through
``build_unified_trie_store_interned`` -- the interned drain. That is true for
slot-carrying ``@channel`` graphs (whose assembly items/capture the streaming
payload envelope cannot carry) AND for plain slot-free graphs (for which the
payload round trip is pure overhead in-process). These tests pin that route:
both shapes return the FULL parse and never touch the weka trie drain, and the
persisted store carries the real (including dynamic-slot) envelopes.
``graph_carries_assembly_slots`` is retained for the ``workload_detect``
t*-gate, so its detection is pinned here too.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import orjson
import pytest

from aiperf.dataset.graph.parser import parse_native
from aiperf.dataset.graph.segment_ir.store_builder import (
    graph_carries_assembly_slots,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import GraphSegmentUnifiedClient

SLOT_GRAPH = """
graph:
  nodes:
    plan:
      prompt: [{role: user, content: "Make a plan."}]
      output: plan_out
    review:
      prompt:
        - role: user
          content: ["Review this plan: ", "@plan_out"]
      output: review_out
  edges:
    - {source: START, target: plan}
    - {source: plan, target: review}
    - {source: review, target: END}
traces:
  - id: t1
"""

STATIC_GRAPH = """
graph:
  nodes:
    plan:
      prompt: [{role: user, content: "Make a plan."}]
      output: plan_out
  edges:
    - {source: START, target: plan}
    - {source: plan, target: END}
traces:
  - id: t1
"""


def _parse(tmp_path: Path, yaml_text: str):
    p = tmp_path / "workload.yaml"
    p.write_text(yaml_text)
    return p, parse_native(p)


def _interned_stub() -> SimpleNamespace:
    """Just the attributes the non-weka interned branch reads from self, with
    the REAL interned-drain helpers bound and a weka trie drain that fails
    loudly if reached (no non-weka format may ever take it)."""
    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id="bench"),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    stub._build_interned_unified_store = MethodType(
        GraphStoreBuilder._build_interned_unified_store, stub
    )
    stub._write_graph_sidecar = MethodType(GraphStoreBuilder._write_graph_sidecar, stub)

    async def _fail_trie(payloads: Any, base_path: Path) -> None:
        raise AssertionError("non-weka graph must not take the weka trie payload drain")

    stub._build_graph_store_streaming_trie = _fail_trie
    return stub


def test_graph_carries_assembly_slots_true_for_slot_graph(tmp_path: Path) -> None:
    _, parsed = _parse(tmp_path, SLOT_GRAPH)
    assert graph_carries_assembly_slots(parsed) is True


def test_graph_carries_assembly_slots_false_for_static_graph(tmp_path: Path) -> None:
    _, parsed = _parse(tmp_path, STATIC_GRAPH)
    assert graph_carries_assembly_slots(parsed) is False


@pytest.mark.asyncio
async def test_build_graph_store_streaming_slot_graph_takes_interned_drain(
    tmp_path: Path, monkeypatch
) -> None:
    """A slot-carrying native parse drains through the interned builder (slot
    envelope persisted) and the SAME parse is returned -- never the trie drain."""
    from aiperf.dataset.graph import workload_detect

    graph_path, parsed = _parse(tmp_path, SLOT_GRAPH)
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )
    stub = _interned_stub()

    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, graph_path, tmp_path, "native"
    )

    assert returned is parsed
    assert set(catalog) == {"t1"}
    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as client:
        review_bytes = client.get_node_envelope(
            "t1", catalog["t1"]["review"], "profiling"
        )
        plan_bytes = client.get_node_envelope("t1", catalog["t1"]["plan"], "profiling")
    assert review_bytes is not None and plan_bytes is not None
    # The dynamic-slot envelopes only the interned drain persists: the reader's
    # composed message with the plan node's response slot, and the producer's
    # capture flag.
    review_envelope = orjson.loads(review_bytes)
    assert review_envelope["items"] == [
        {
            "m": {
                "role": "user",
                "parts": [
                    {"t": "Review this plan: "},
                    {"sv": catalog["t1"]["plan"]},
                ],
            }
        }
    ]
    assert orjson.loads(plan_bytes).get("capture") is True


@pytest.mark.asyncio
async def test_build_graph_store_streaming_static_graph_takes_interned_drain(
    tmp_path: Path, monkeypatch
) -> None:
    """A plain (slot-free) native parse ALSO takes the interned drain now: the
    weka trie payload drain is weka-only, so the flip routes every non-weka
    format -- slot-carrying or not -- through the interned builder and returns
    the FULL parse."""
    from aiperf.dataset.graph import workload_detect

    graph_path, parsed = _parse(tmp_path, STATIC_GRAPH)
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )
    stub = _interned_stub()

    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, graph_path, tmp_path, "native"
    )

    assert returned is parsed
    assert set(catalog) == {"t1"}
    # The interned store the worker opens carries the plain node's envelope.
    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as client:
        plan_bytes = client.get_node_envelope("t1", catalog["t1"]["plan"], "profiling")
    assert plan_bytes is not None
