from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    _prompt_segment_ids,
    _trie_llm_nodes,
    build_unified_trie_store_interned,
    trie_node_ordinals,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)

FIXTURES = Path(__file__).parent.parent / "graph" / "fixtures"


@pytest.fixture
def one_trie_trace(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    parsed = from_weka_trace(FIXTURES / "weka_subagent.json")
    assert parsed.segment_pool is not None
    known_path = None
    known_ordinal = None
    known_trace = None
    for trace in parsed.traces:
        nodes = _trie_llm_nodes(parsed, trace)
        # Assign ordinals via the SAME builder helper (sort key
        # (arrival_offset_us or 0, node_id)) so known_ordinal matches the
        # ordinal build_unified_trie_store_interned stores; a hand-rolled
        # enumerate(sorted(nodes)) would query a different node's manifest
        # whenever arrival order differs from lexical node-id order.
        ordinals = trie_node_ordinals(nodes)
        for nid, node in nodes.items():
            path = _prompt_segment_ids(node)
            if path and (known_path is None or len(path) > len(known_path)):
                known_path, known_ordinal, known_trace = path, ordinals[nid], trace.id
    assert known_path
    return parsed, known_trace, known_ordinal, known_path


def _dyn_subagent_fixture(tmp_path: Path) -> Path:
    """Current-schema dynamo trace: one root session spawning one subagent.

    ``s_root`` makes several LLM calls; ``s_child`` (``parent_session_id=s_root``)
    is a subagent session. The trie adapter flattens BOTH into one graph of
    ``{agent_id}_a{k}`` LlmNodes (no ``parsed.subgraphs``), so the built store
    must carry a bare-id manifest for every node -- child-session nodes included.
    """

    def _rec(ts: int, sid: str, parent: str | None = None) -> dict:
        ctx: dict = {"session_id": sid}
        if parent is not None:
            ctx["parent_session_id"] = parent
        return {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": ts,
            "event_source": "dynamo",
            "agent_context": ctx,
            "request": {
                "request_id": f"{sid}-r{ts}",
                "model": "m",
                "input_tokens": 32,
                "output_tokens": 16,
                "cached_tokens": 0,
            },
        }

    records = [
        _rec(1000, "s_root"),
        _rec(1100, "s_root"),
        _rec(1150, "s_child", "s_root"),
        _rec(1170, "s_child", "s_root"),
        _rec(1200, "s_root"),
    ]
    p = tmp_path / "dyn_subagent.jsonl"
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


@pytest.mark.asyncio
async def test_unified_store_includes_child_session_manifests(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace

    p = _dyn_subagent_fixture(tmp_path)
    parsed = from_dynamo_trace(p, content_root_seed=1234, content_tokenizer="builtin")
    assert parsed.segment_pool is not None

    store = GraphSegmentUnifiedBackingStore(tmp_path, "bench")
    catalog = await build_unified_trie_store_interned(parsed, store)

    # Child-session nodes carry bare-id catalog keys next to the root's.
    keys = {k for trace_keys in catalog.values() for k in trace_keys}
    child_keys = {k for k in keys if k.startswith("s_child:")}
    assert len(child_keys) == 2, f"missing child-session keys in {sorted(keys)}"

    # Every node's manifest round-trips out of the store at its ordinal.
    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as c:
        for trace_id, trace_keys in catalog.items():
            for _key, ordinal in trace_keys.items():
                env = orjson.loads(c.get_node_envelope(trace_id, ordinal, "profiling"))
                handles = env["handles"]
                msgs = c.materialize_handles(handles)
                assert msgs and all(set(m) == {"role", "content"} for m in msgs)


@pytest.mark.asyncio
async def test_build_unified_trie_store_interned_round_trips(tmp_path, one_trie_trace):
    parsed, trace_id, ordinal, path = one_trie_trace
    store = GraphSegmentUnifiedBackingStore(tmp_path, "bench")
    catalog = await build_unified_trie_store_interned(parsed, store)
    assert trace_id in catalog

    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as c:
        env = orjson.loads(c.get_node_envelope(trace_id, ordinal, "profiling"))
        handles = env["handles"]
        assert len(handles) == len(path)
        msgs = c.materialize_handles(handles)
        assert all(set(m) == {"role", "content"} for m in msgs)
