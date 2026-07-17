# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dynamo -> shared-trie normalization layer."""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import _Chain, _Turn
from aiperf.dataset.graph.adapters.dynamo.trace_reader import AgentTraceRecord


def _rec_obj(
    ts, sid, itok, otok, hashes=None, bs=16, ilen=None, received=None, total=None
):
    req = {
        "request_id": f"r{ts}",
        "model": "m",
        "input_tokens": itok,
        "output_tokens": otok,
        "cached_tokens": 0,
    }
    if received is not None:
        req["request_received_ms"] = received
    if total is not None:
        req["total_time_ms"] = total
    if hashes is not None:
        req["replay"] = {
            "trace_block_size": bs,
            "input_length": ilen if ilen is not None else itok,
            "input_sequence_hashes": hashes,
        }
    return AgentTraceRecord.model_validate(
        {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": ts,
            "event_source": "dynamo",
            "agent_context": {"session_id": sid},
            "request": req,
        }
    )


def _chain(sid, turns, parent=None):
    return _Chain(
        sid,
        parent_session_id=parent,
        turns=[_Turn(record=r) for r in turns],
    )


def test_dynamo_recon_callbacks_offset_cache_smoke_and_shape():
    """``dynamo_recon_callbacks`` decodes through the offset cache and returns
    tokens byte-identical to a hand-built list-cache ``_decode_block_tokens`` at
    the same block size; the offset cache stores plain int offsets keyed by hash
    id (one int per unique hash, not the decoded block list)."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_recon_callbacks,
    )
    from aiperf.dataset.graph.adapters.shared.content import (
        CorpusContentSynthesizer,
        get_or_build_synthesizer,
    )

    CorpusContentSynthesizer.reset_worker_cache()
    bs = 16
    hash_ids = [111, 222, 333, 111, 222]  # repeats exercise the hit path

    cb = dynamo_recon_callbacks(
        "builtin", "coding", 1234, block_size=bs, trace_scope="t"
    )
    got = cb.decode_block_tokens(hash_ids)

    # Hand-built oracle: the SAME shared synth via the list-cache path.
    synth = get_or_build_synthesizer("builtin", prompt_corpus="coding", root_seed=1234)
    expected = synth._decode_block_tokens(hash_ids, block_size=bs, cache={})
    assert got == expected
    assert len(got) == len(hash_ids) * bs

    offsets: dict[int, int] = {}
    synth._decode_block_tokens_offset_cached(
        hash_ids, block_size=bs, offset_cache=offsets
    )
    assert offsets and all(type(v) is int for v in offsets.values())
    assert set(offsets) == set(hash_ids)


def test_recorded_hashes_used_when_present_and_relative_seconds():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(
                    2000, "s1", 32, 8, hashes=[111, 222], received=1000, total=1000
                ),
                _rec_obj(
                    4000,
                    "s1",
                    64,
                    12,
                    hashes=[111, 222, 333, 444],
                    received=3000,
                    total=1000,
                ),
            ],
        )
    }
    nodes, bs, _tags = dynamo_trie_nodes(chains)
    assert bs == 16
    assert [n.request.hash_ids for n in nodes] == [[111, 222], [111, 222, 333, 444]]
    assert nodes[0].request.t == 0.0 and nodes[1].request.t == 2.0
    assert nodes[0].request.api_time == 1.0


def test_virtual_hashes_extend_per_session_and_never_collide_with_recorded():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8),
                _rec_obj(2000, "s1", 64, 8),
            ],
        )
    }
    nodes, bs, _ = dynamo_trie_nodes(chains)
    h1, h2 = nodes[0].request.hash_ids, nodes[1].request.hash_ids
    assert len(h1) == 32 // bs and len(h2) == 64 // bs
    assert h2[: len(h1)] == h1
    assert all(h < 0 for h in h1 + h2)


def test_mixed_block_size_raises():
    from aiperf.dataset.graph.adapters.dynamo.trace import (
        DynamoTraceAdapterError,
    )
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8, hashes=[1, 2], bs=16),
                _rec_obj(2000, "s1", 64, 8, hashes=[1, 2, 3, 4], bs=32),
            ],
        )
    }
    with pytest.raises(DynamoTraceAdapterError, match="trace_block_size"):
        dynamo_trie_nodes(chains)


def test_misaligned_input_length_raises():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        DynamoISLMismatchError,
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain("s1", [_rec_obj(1000, "s1", 100, 8, hashes=[1, 2], ilen=100)])
    }
    with pytest.raises(DynamoISLMismatchError):
        dynamo_trie_nodes(chains)


def test_kv_fallback_turn_gets_virtual_hashes_and_tag():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8, hashes=[111, 222]),
                _rec_obj(2000, "s1", 64, 8),  # no replay metadata
            ],
        )
    }
    nodes, _bs, tags = dynamo_trie_nodes(chains)
    assert "virtual-hash-fallback" in tags
    h2 = nodes[1].request.hash_ids
    assert h2[:2] == [111, 222] and all(h < 0 for h in h2[2:])


def test_node_ids_and_session_header_metadata():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "root": _chain(
            "root", [_rec_obj(1000, "root", 32, 8), _rec_obj(2000, "root", 64, 8)]
        ),
        "child": _chain("child", [_rec_obj(1500, "child", 16, 4)], parent="root"),
    }
    nodes, _bs, _t = dynamo_trie_nodes(chains)
    ids = {n.node_id for n in nodes}
    assert "root:0" in ids and "child:0" in ids
    root_a1 = next(n for n in nodes if n.node_id == "root:0")
    h1 = root_a1.dynamo_meta["extra_headers"]
    assert h1["x-dynamo-session-id"] == "root"
    assert "x-dynamo-session-final" not in h1
    root_a2 = next(n for n in nodes if n.node_id == "root:1")
    assert root_a2.dynamo_meta["extra_headers"]["x-dynamo-session-final"] == "true"
    child = next(n for n in nodes if n.node_id == "child:0")
    ch = child.dynamo_meta["extra_headers"]
    assert ch["x-dynamo-parent-session-id"] == "root"
    assert ch["x-dynamo-session-final"] == "true"  # single-turn session


def test_session_id_with_colon_raises():
    """Node ids are ``{session_id}:{k}`` with the session id VERBATIM, so a
    recorded session id containing the reserved ``:`` separator (or empty)
    cannot form a node id and must fail loud at lowering."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {"a:b": _chain("a:b", [_rec_obj(1000, "a:b", 32, 8)])}
    with pytest.raises(ValueError, match="cannot form node ids"):
        dynamo_trie_nodes(chains)


def test_same_ms_turn_tiebreak_is_numeric_past_index_nine():
    """12 turns at the same start_ms must order :0.. :11, not the lexicographic
    :0, :1, :10, :11, :2, ... (order is the trie's content-parent ground truth)."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    n_turns = 12
    chains = {"s1": _chain("s1", [_rec_obj(1000, "s1", 32, 8) for _ in range(n_turns)])}
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    assert [n.node_id for n in nodes] == [f"s1:{k}" for k in range(n_turns)]
    assert [n.order for n in nodes] == list(range(n_turns))


def test_virtual_hashes_shrinking_input_reuses_prefix_without_new_ids():
    """prev[:m] truncation branch: a turn with FEWER covered blocks than the
    session's prior turn reuses the leading virtual ids and allocates none."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 64, 8),  # 4 covered blocks at bs=16
                _rec_obj(2000, "s1", 32, 8),  # shrinks to 2 covered blocks
                _rec_obj(3000, "s1", 64, 8),  # grows again past the truncation
            ],
        )
    }
    nodes, bs, _ = dynamo_trie_nodes(chains)
    h1, h2, h3 = (n.request.hash_ids for n in nodes)
    assert len(h1) == 64 // bs and len(h2) == 32 // bs
    assert h2 == h1[: len(h2)]  # pure truncation, no fresh ids
    # Regrowth extends the TRUNCATED prefix with fresh ids (the counter never
    # rewinds), so the shared prefix survives but the tail is new.
    assert h3[: len(h2)] == h2
    assert set(h3[len(h2) :]).isdisjoint(set(h1))
    assert all(h < 0 for h in h1 + h2 + h3)


def _built_nodes(chains):
    """Run the shared trie build with cheap stub callbacks; return (nodes, pool, result)."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )
    from aiperf.dataset.graph.segment_ir.pool import SegmentPool
    from aiperf.dataset.graph.segment_ir.trie_content import (
        ReconCallbacks,
        build_trie_ir,
    )

    nodes, bs, _tags = dynamo_trie_nodes(chains)
    pool = SegmentPool()
    result = build_trie_ir(
        nodes,
        block_size=bs,
        callbacks=ReconCallbacks(
            # One block decodes to EXACTLY bs tokens (the ISL gate checks the
            # actual assembled token count, so a mis-sized stub aborts the build).
            decode_block_tokens=lambda hash_ids: [
                t for h in hash_ids for t in [abs(h) % 9973] * bs
            ],
            sample_partial_tail_tokens=lambda n, seed: list(range(n)),
            decode_tokens_to_text=lambda toks: " ".join(map(str, toks)),
        ),
        pool=pool,
        idle_gap_cap_seconds=None,
        small_prompt_fallback=True,
    )
    return nodes, pool, result


def test_build_dynamo_llm_node_overrides_metadata_and_stamp():
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        build_dynamo_llm_node,
    )

    chains = {
        "root": _chain(
            "root",
            [
                _rec_obj(1000, "root", 32, 8, received=1000, total=500),
                _rec_obj(2000, "root", 64, 12, received=2000, total=500),
            ],
        )
    }
    nodes, pool, result = _built_nodes(chains)
    node = nodes[1]
    llm = build_dynamo_llm_node(
        node,
        build=result.builds[node.node_id],
    )
    assert llm.output == f"{node.node_id}_out"
    assert llm.streaming is False
    # arrival stamps the warped start assigned by the trie build.
    assert llm.arrival_offset_us == int(round(node.start * 1_000_000.0))
    # Body params ride the native node fields (no extra_body);
    # session identity is header-borne, never body nvext.
    assert llm.extra_body is None
    headers = llm.extra_headers
    assert headers["x-dynamo-session-id"] == "root"
    assert headers["x-dynamo-session-final"] == "true"
    assert llm.model == "m"
    assert llm.max_tokens == 12
    # expected tokens mirror the recorded request.
    assert llm.expected is not None and llm.expected.output_tokens == 12
    # metadata shape: observed round-trip + dynamo identity + trie stamp.
    assert "observed" not in llm.metadata
    dyn = llm.metadata["dynamo"]
    assert dyn["session_id"] == "root"
    assert dyn["parent_session_id"] is None
    assert dyn["turn_index"] == 1  # 0-based; nodes[1] is the second turn
    assert "tool_breakdown" not in dyn
    assert dyn["small_prompt"] is False
    trie = llm.metadata["trie"]
    assert trie["prompt_segment_ids"] == result.builds[node.node_id].prompt_path
    # Dynamo stamps ONLY prompt_segment_ids -- the build-synthesis response_id /
    # hash_ids extras are dropped (nothing on the build, sidecar, store, or
    # dispatch plane reads them; weka is the sole remaining extras author).
    assert set(trie) == {"prompt_segment_ids"}
    assert pool.materialize(trie["prompt_segment_ids"])


def test_build_dynamo_llm_node_omits_inline_prompt():
    """The trie-route LlmNode carries NO inline prompt: content lives only in the
    segment pool, addressed via ``metadata["trie"]["prompt_segment_ids"]``.

    ``node.prompt`` is dead weight on this route (the store, sidecar, and worker
    all go through the segment pool), so the adapter stamps ``prompt=[]`` and the
    pool path still materializes the real conversation content."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        build_dynamo_llm_node,
    )

    chains = {
        "root": _chain(
            "root",
            [
                _rec_obj(1000, "root", 32, 8, received=1000, total=500),
                _rec_obj(2000, "root", 64, 12, received=2000, total=500),
            ],
        )
    }
    nodes, pool, result = _built_nodes(chains)
    node = nodes[1]
    llm = build_dynamo_llm_node(
        node,
        build=result.builds[node.node_id],
    )
    assert llm.prompt == []
    # Content is still reachable through the segment pool via the trie envelope.
    seg_ids = llm.metadata["trie"]["prompt_segment_ids"]
    messages = pool.materialize(seg_ids)
    assert messages, "the pool path must still materialize the real prompt"
    for msg in messages:
        assert msg["role"] in {"system", "user", "assistant", "tool"}
        assert isinstance(msg["content"], str)


def test_node_meta_carries_no_recorded_scalar_round_trip():
    """No recorded-scalar round-trip rides the node meta: node metadata
    survives the graph_meta sidecar strip (which clears only prompt and
    metadata["trie"]), so every extra key would bloat the content-free
    structural plane. Hashes are consumed via TrieRequest.hash_ids; scalars
    live on the native fields (model / max_tokens / expected)."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = {"s1": _chain("s1", [_rec_obj(1000, "s1", 32, 8, hashes=[111, 222])])}
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    assert set(nodes[0].dynamo_meta) == {
        "extra_headers",
        "session_id",
        "parent_session_id",
        "turn_index",
        "expected",
    }
    assert nodes[0].request.hash_ids == [111, 222]


def test_build_dynamo_llm_node_zero_output_upgrades_cap_to_one():
    """A recorded output_tokens == 0 must NOT reach the wire as
    max_output_tokens=0 (a meaningless cap); it upgrades to 1 with a warning
    (wire_output_cap) so the turn stays pinned."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        build_dynamo_llm_node,
    )

    chains = {"s1": _chain("s1", [_rec_obj(1000, "s1", 32, 0)])}
    nodes, _pool, result = _built_nodes(chains)
    llm = build_dynamo_llm_node(
        nodes[0],
        build=result.builds[nodes[0].node_id],
    )
    assert llm.max_tokens == 1


def _replay_release_chains():
    """Two recorded-replay sessions, freshly built per call so a release
    mutation in one lowering never leaks into another test's chains."""
    return {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8, hashes=[111, 222]),
                _rec_obj(2000, "s1", 64, 12, hashes=[111, 222, 333, 444]),
            ],
        ),
        "s2": _chain("s2", [_rec_obj(3000, "s2", 48, 8, hashes=[555, 666, 777])]),
    }


def _all_records(chains):
    return [t.record for c in chains.values() for t in c.turns]


def test_dynamo_trie_nodes_release_replay_off_keeps_recorded_replay():
    """Default (release_replay=False): every record's replay stays intact -- the
    re-lowering safety pin. A second lowering of the same chains must still read
    recorded hashes, never degrade to the virtual-hash fallback."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    chains = _replay_release_chains()
    dynamo_trie_nodes(chains)
    assert all(r.request.replay is not None for r in _all_records(chains))


def test_dynamo_trie_nodes_release_replay_frees_records_without_changing_nodes():
    """release_replay=True nulls each recorded req.replay AFTER its hashes are
    copied into the TrieRequest, so the emitted nodes are identical (same
    hashes, ISL, order, causal parents, block size, tags) to the default."""
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        dynamo_trie_nodes,
    )

    keep = _replay_release_chains()
    release = _replay_release_chains()
    nodes_keep, bs_keep, tags_keep = dynamo_trie_nodes(keep)
    nodes_rel, bs_rel, tags_rel = dynamo_trie_nodes(release, release_replay=True)

    assert all(r.request.replay is None for r in _all_records(release))
    assert all(r.request.replay is not None for r in _all_records(keep))

    def _ident(nodes):
        return [
            (
                n.node_id,
                n.request.hash_ids,
                n.request.input_length,
                n.order,
                n.causal_parent_id,
            )
            for n in nodes
        ]

    assert _ident(nodes_rel) == _ident(nodes_keep)
    assert bs_rel == bs_keep and tags_rel == tags_keep


def _parse_route_sids(path, *, direct_store):
    """Parse a capture and return ``(parsed, [every prompt_segment_id str, in order])``.

    Reads each ``LlmNode``'s ``read_prompt_segment_ids`` path across the whole
    graph; the returned strings preserve their object identity so a caller can
    compare ``id()``-set cardinality against distinct VALUE cardinality.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
    from aiperf.dataset.graph.models import LlmNode
    from aiperf.dataset.graph.segment_ir.envelope import read_prompt_segment_ids

    parsed = from_dynamo_trace(
        path,
        content_root_seed=1234,
        content_tokenizer="builtin",
        direct_store=direct_store,
    )
    sids: list[str] = []
    for node in parsed.graph.nodes.values():
        if isinstance(node, LlmNode):
            sids.extend(read_prompt_segment_ids(node) or [])
    return parsed, sids


@pytest.mark.parametrize("route", ["eager", "direct"])  # fmt: skip
def test_prompt_segment_ids_are_canonical_objects_on_both_routes(route, tmp_path):
    """Every equal ``prompt_segment_ids`` string across all nodes is ONE object.

    Each turn re-lists its whole message chain, so absent interning the same
    segment value appears as ~``H/N`` distinct fresh hexdigest strings per node.
    Pool interning collapses those to one canonical object per
    unique value on BOTH dynamo routes -- so the count of distinct ``id()``s
    equals the count of distinct VALUES. On the eager route each stamped sid is
    additionally the very object the ``SegmentPool`` first-born for that value.
    """
    from tests.harness.dynamo_synth_corpus import write_synthetic_dynamo_capture

    capture = tmp_path / "route.jsonl"
    write_synthetic_dynamo_capture(
        capture,
        sessions=3,
        turns_per_session=6,
        new_blocks_per_turn=2,
        block_size=16,
        seed=7,
    )

    if route == "eager":
        parsed, sids = _parse_route_sids(capture, direct_store=None)
    else:
        from aiperf.dataset.graph_segment_unified_store import (
            GraphSegmentUnifiedBackingStore,
        )

        store = GraphSegmentUnifiedBackingStore(
            base_path=tmp_path, benchmark_id="route-direct"
        )
        try:
            parsed, sids = _parse_route_sids(capture, direct_store=store)
        finally:
            store.abort()

    assert sids, "the capture produced no prompt_segment_ids"
    # Duplicates across turns/nodes exist (each turn re-lists its chain)...
    assert len(sids) > len(set(sids)), "test corpus too small to exercise dedup"
    # ...but every equal-valued sid is the SAME object: distinct objects == values.
    assert len({id(s) for s in sids}) == len(set(sids))

    if route == "eager":
        by_id = parsed.segment_pool.by_id
        assert all(s is by_id[s].id for s in sids), (
            "eager route must stamp the first-born canonical Segment.id object"
        )
