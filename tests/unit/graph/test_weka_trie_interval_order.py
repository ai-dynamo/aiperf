# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural invariants of the interval-order + message-unit Weka trie builder.

Covers the two reconstruction contracts (with deterministic stub callbacks):

* **Timing** -- global interval-order edges: ``async_ancestors`` stamping, a
  time-consistent ``rank``, the finished-before frontier (transitive reduction),
  async-subtree exclusion, coincident/concurrent handling, and START-rooting.
* **Content** -- per-block frozen ``(role, starts_new_message)`` tags
  (creator-frozen, inherited verbatim, trailing-user at creation), message-unit
  content-addressed emission, shared-prefix identical message-id chains, boundary
  preservation, and the block-aligned covered-count ISL gate.

Scope: these assert the *structural* invariant only -- properties of the
reconstructed IR. The end-to-end cache-hit claim (a shared block prefix yielding
a real KV prefix-cache hit) is provable only against a real prefix-caching engine
(vLLM/SGLang/real Dynamo); AIPerf's mock server is throughput-only and cannot
validate it. See ``docs/reference/graph-ingest-build-pipeline.md`` ->
"Validation boundary".
"""

import pytest

from aiperf.dataset.graph.adapters.weka.trace_models import (
    WekaNormalRequest,
    WekaSubagentEntry,
)
from aiperf.dataset.graph.adapters.weka.trie_build import (
    _flatten_requests,
)
from aiperf.dataset.graph.segment_ir.interval_order import (
    build_interval_edges as _build_interval_edges,
)
from aiperf.dataset.graph.segment_ir.interval_order import (
    compute_ranks as _compute_ranks,
)


def _n(t, hid, out=0, api=0.0, in_=0):
    return WekaNormalRequest(
        t=t, type="n", model="m", **{"in": in_}, out=out, hash_ids=hid, api_time=api
    )


def test_flatten_stamps_async_ancestors():
    reqs = [
        _n(0.0, [1]),
        WekaSubagentEntry(
            t=1.0,
            type="subagent",
            agent_id="a",
            subagent_type="X",
            status="completed",
            requests=[_n(1.1, [1, 2])],
            models=["m"],
        ),
        WekaSubagentEntry(
            t=2.0,
            type="subagent",
            agent_id="b",
            subagent_type="X",
            status="async_launched",
            requests=[_n(2.1, [1, 3])],
            models=["m"],
        ),
        _n(3.0, [1, 4]),
    ]
    by = {
        tuple(n.request.hash_ids): n for n in _flatten_requests(reqs, root_scope="tr")
    }
    assert by[(1, 3)].async_ancestors and not by[(1, 2)].async_ancestors
    assert not by[(1,)].async_ancestors and not by[(1, 4)].async_ancestors


def test_rank_total_order_and_coincident_tiebreak():
    nodes = _flatten_requests(
        [_n(0.0, [1], api=0.0), _n(0.0, [2], api=0.0)], root_scope="tr"
    )  # coincident zero-duration
    for x in nodes:
        x.warped_start = x.request.t
    _compute_ranks(nodes)
    assert sorted(n.rank for n in nodes) == [0, 1]  # unique, total order
    a, b = sorted(nodes, key=lambda n: n.node_id)
    assert a.rank < b.rank  # lower node_id -> lower rank on a full tie


def _prep(reqs):
    nodes = _flatten_requests(reqs, root_scope="tr")
    for x in nodes:
        x.warped_start = x.request.t
    _compute_ranks(nodes)
    return nodes


def test_overlap_concurrent_finishedbefore_andjoin():
    nodes = _prep(
        [
            _n(0.0, [1], api=1.0),  # P0 [0,1]
            _n(1.2, [1, 2], api=2.8),  # A  [1.2,4]
            _n(1.3, [1, 3], api=3.7),  # B  [1.3,5]
            _n(5.2, [1, 4], api=1.8),  # C  [5.2,7]
        ]
    )
    edges = _build_interval_edges(nodes)

    def srcs(n):
        return {e.source for e in edges[n.node_id]}

    a, b, c = nodes[1], nodes[2], nodes[3]
    assert b.node_id not in srcs(a) and a.node_id not in srcs(b)  # racers concurrent
    assert srcs(c) == {a.node_id, b.node_id}  # AND-join both
    ce = {e.source: e for e in edges[c.node_id]}
    assert ce[b.node_id].delay_after_predecessor_us == pytest.approx(
        (5.2 - 5.0) * 1e6
    )  # binding=B (latest end)
    assert ce[a.node_id].delay_after_predecessor_us == 0.0


def test_empty_frontier_roots_at_start():
    nodes = _prep([_n(0.0, [1], api=1.0)])
    e = _build_interval_edges(nodes)[nodes[0].node_id]
    assert (
        len(e) == 1
        and e[0].source == "START"
        and e[0].min_start_delay_us == pytest.approx(0.0)
    )


def test_async_excluded_cross_subtree():
    nodes = _prep(
        [
            _n(0.0, [1], api=0.5),
            WekaSubagentEntry(
                t=0.6,
                type="subagent",
                agent_id="b",
                subagent_type="X",
                status="async_launched",
                requests=[_n(0.6, [1, 9], api=0.2)],
                models=["m"],
            ),
            _n(3.0, [1, 4], api=0.5),  # P1 starts after the async child ended
        ]
    )
    edges = _build_interval_edges(nodes)
    p1 = next(n for n in nodes if tuple(n.request.hash_ids) == (1, 4))
    b = next(n for n in nodes if tuple(n.request.hash_ids) == (1, 9))
    assert b.node_id not in {
        e.source for e in edges[p1.node_id]
    }  # fire-and-forget child not a pred


# --- Task 4B: block geometry + role split + trailing-user caps (foundation) ---

from aiperf.dataset.graph.segment_ir.trie_content import (  # noqa: E402
    block_role_split,
    compute_asst_caps,
    compute_turn_block_geometry,
    resolve_content_parents,
)


def test_geometry_block_aligned_no_partial_tail():
    # in=5, bs=2 -> in//bs=2 covered blocks; 3 hash ids but only 2 covered; NO in%bs tail
    g = compute_turn_block_geometry([1], [1, 2, 3], 5, 2)
    assert (
        g.lcp == 1
        and g.m_curr_covered == 2
        and g.new_blocks_count == 1
        and g.synth_tail_n == 0
    )
    # missing whole blocks: in//bs=3 but only 2 hash ids -> 1 missing block = 2 tokens
    g2 = compute_turn_block_geometry([], [1, 2], 6, 2)
    assert g2.m_curr_covered == 2 and g2.synth_tail_n == 2


def test_block_role_split_trailing_user_at_creation():
    # prev_out large: ceil(6/2)=3 asst, clamped to 2 new blocks -> asst==new_n, so the
    # last new block flips to user (trailing-user frozen at creation).
    inherited, roles = block_role_split(
        prev_hash_ids=[1],
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=6,
        prev_out_tokens=6,
        block_size=2,
        max_asst_blocks=None,
        parent_has_user=True,
    )
    # last new block flipped to user
    assert inherited == 1
    assert roles == ["assistant", "user"]
    # context-loss: parent_has_user False -> all user
    _, roles2 = block_role_split(
        prev_hash_ids=[1],
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=6,
        prev_out_tokens=6,
        block_size=2,
        max_asst_blocks=None,
        parent_has_user=False,
    )
    assert roles2 == ["user", "user"]
    # single all-assistant new block flips wholesale to user (no assistant survives)
    _, roles3 = block_role_split(
        prev_hash_ids=[1],
        curr_hash_ids=[1, 2],
        curr_in_tokens=4,
        prev_out_tokens=10,
        block_size=2,
        max_asst_blocks=None,
        parent_has_user=True,
    )
    assert roles3 == ["user"]


def test_compute_asst_caps_skips_root_owner():
    # single root node, degenerate: no cap should be recorded on a root owner
    nodes = _flatten_requests(
        [_n(0.0, [1, 2], in_=4), _n(0.2, [1, 2], in_=4)], root_scope="tr"
    )
    resolve_content_parents(nodes)
    caps = compute_asst_caps(nodes, 2)
    roots = [n for n in nodes if n.content_parent is None]
    for r in roots:
        assert caps.get(r.node_id) is None  # root owner never capped


# --- Task 4: per-block (role, starts_new_message) tag pass, frozen at creation ---

from aiperf.dataset.graph.segment_ir.trie_content import (  # noqa: E402
    assign_block_tags,
)


def _tags_for(reqs, bs):
    nodes = _flatten_requests(reqs, root_scope="tr")
    resolve_content_parents(nodes)
    caps = compute_asst_caps(nodes, bs)
    return nodes, assign_block_tags(nodes, bs, caps)


def test_shared_prefix_tags_identical_and_no_relabel():
    # A long parent chain that per-turn would relabel a shared block, + a fork inheriting it.
    reqs = [
        _n(0.0, [10, 11, 12], in_=6, out=0, api=0.1),  # P0 user blocks
        _n(0.2, [10, 11], in_=4, out=400, api=0.1),  # P1 pull-back w/ big prev_out
        _n(0.4, [10, 11, 20], in_=6, out=0, api=0.1),  # P2 continues, shares [10,11]
        WekaSubagentEntry(
            t=1.0,
            type="subagent",
            agent_id="a",
            subagent_type="X",
            status="completed",
            requests=[_n(1.1, [10, 11, 30], in_=6, out=0, api=0.1)],
            models=["m"],
        ),  # fork inherits [10,11]
    ]
    nodes, tags = _tags_for(reqs, 2)
    by = lambda h: next(n for n in nodes if tuple(n.request.hash_ids) == h)  # noqa: E731
    fork, p2 = by((10, 11, 30)), by((10, 11, 20))
    # blocks [10,11] (indices 0,1) have IDENTICAL (role, starts_new_message) in fork and P2
    assert tags[fork.node_id][:2] == tags[p2.node_id][:2]


def test_contiguous_same_role_turns_keep_boundary():
    # two consecutive user turns (prev_out=0) -> the 2nd turn's first new block starts a new message
    reqs = [_n(0.0, [1], in_=2, out=0, api=0.1), _n(0.2, [1, 2], in_=4, out=0, api=0.1)]
    nodes, tags = _tags_for(reqs, 2)
    p = next(n for n in nodes if tuple(n.request.hash_ids) == (1, 2))
    # block 0 (user, starts), block 1 is the new turn's block -> user, starts_new_message True
    assert tags[p.node_id][1] == ("user", True)


# --- Task 5: message-unit content emission from frozen per-block tags ---

from aiperf.dataset.graph.segment_ir.pool import (  # noqa: E402
    SegmentPool,
)
from aiperf.dataset.graph.segment_ir.trie_content import (  # noqa: E402
    assemble_messages,
)


def _decode(hids):
    return [h * 1000 + k for h in hids for k in range(2)]  # bs=2, collision-free


def _text(toks):
    return " ".join(map(str, toks))


def test_shared_prefix_identical_message_ids():
    reqs = [
        _n(0.0, [10, 11], in_=4, out=0, api=0.1),  # P0
        WekaSubagentEntry(
            t=1.0,
            type="subagent",
            agent_id="a",
            subagent_type="X",
            status="completed",
            requests=[_n(1.1, [10, 11, 30], in_=6, out=0, api=0.1)],
            models=["m"],
        ),  # fork inherits [10,11]
        _n(2.0, [10, 11, 20], in_=6, out=0, api=0.1),  # P1 shares [10,11]
    ]
    nodes, tags = _tags_for(reqs, 2)
    by = lambda h: next(  # noqa: E731
        n for n in nodes if tuple(n.request.hash_ids) == h
    )

    def ids(node):
        return assemble_messages(
            node.request.hash_ids, tags[node.node_id], pool, _decode, _text
        )[0]

    pool = SegmentPool()
    fork_ids = ids(by((10, 11, 30)))
    p1_ids = ids(by((10, 11, 20)))
    p0_ids = ids(by((10, 11)))
    # the message(s) covering the shared [10,11] prefix are byte-identical ids
    # across all three
    assert fork_ids[: len(p0_ids)] == p0_ids == p1_ids[: len(p0_ids)]


def test_contiguous_same_role_not_coalesced():
    # two user turns -> block1 has starts_new_message True -> two user messages
    reqs = [
        _n(0.0, [1], in_=2, out=0, api=0.1),
        _n(0.2, [1, 2], in_=4, out=0, api=0.1),
    ]
    nodes, tags = _tags_for(reqs, 2)
    pool = SegmentPool()
    p = next(n for n in nodes if tuple(n.request.hash_ids) == (1, 2))
    msg_ids, _ = assemble_messages(
        p.request.hash_ids, tags[p.node_id], pool, _decode, _text
    )
    roles = [pool.materialize([mid])[0]["role"] for mid in msg_ids]
    assert roles == ["user", "user"] and len(msg_ids) == 2  # boundary preserved


def test_last_message_is_user():
    nodes, tags = _tags_for([_n(0.0, [1, 2], in_=4, out=0, api=0.1)], 2)
    pool = SegmentPool()
    n0 = nodes[0]
    ids, _ = assemble_messages(
        n0.request.hash_ids, tags[n0.node_id], pool, _decode, _text
    )
    # trailing-user (frozen in tags)
    assert pool.materialize([ids[-1]])[0]["role"] == "user"


# --- Task 6: block-aligned covered-count ISL build-abort gate ---

from aiperf.dataset.graph.segment_ir.trie_content import (  # noqa: E402
    TrieISLMismatchError,
    assert_covered_isl,
)


def test_isl_covered_count_pass_and_fail():
    node = _flatten_requests([_n(0.0, [1, 2], in_=40)], root_scope="tr")[
        0
    ]  # 2 hash blocks, in//16=2 -> covered 2 -> 32
    assert_covered_isl(node, 32, 16)  # exact covered-count -> no raise
    with pytest.raises(TrieISLMismatchError):
        assert_covered_isl(node, 40, 16)  # 40 != 32


def test_isl_truncated_hash_does_not_abort():
    node = _flatten_requests([_n(0.0, [1], in_=40)], root_scope="tr")[
        0
    ]  # only 1 hash block, in//16=2 -> covered min(1,2)=1 -> 16
    assert_covered_isl(node, 16, 16)  # covered-count 16, NOT 32 -> must NOT raise
    with pytest.raises(TrieISLMismatchError):
        assert_covered_isl(
            node, 32, 16
        )  # demanding (in//bs)*bs would be wrong; 32 != 16


# --- Task 8: end-to-end build_trie_graph switch (interval-order + message-unit) ---

from aiperf.dataset.graph.adapters.weka.trace_models import (  # noqa: E402
    WekaTrace,
)
from aiperf.dataset.graph.adapters.weka.trie_build import (  # noqa: E402
    ReconCallbacks,
    build_trie_graph,
)

# Collision-free stub callbacks at bs=2 (no tokenizer / corpus build). Each hash
# id decodes to two distinct tokens (h*1000, h*1000+1) so shared-prefix blocks
# yield byte-identical tokens -> identical content-addressed pool ids.
_STUB_CB = ReconCallbacks(
    decode_block_tokens=lambda hids: [h * 1000 + k for h in hids for k in range(2)],
    sample_partial_tail_tokens=lambda n, s: [0] * n,
    decode_tokens_to_text=lambda t: " ".join(map(str, t)),
)


def _trace(reqs, block_size=2):
    return WekaTrace(
        id="t",
        models=["m"],
        block_size=block_size,
        hash_id_scope="local",
        requests=reqs,
    )


def test_e2e_racers_fork_and_dynamic_join():
    reqs = [
        _n(0.0, [1], in_=2, out=0, api=1.0),
        WekaSubagentEntry(
            t=1.1,
            type="subagent",
            agent_id="a",
            subagent_type="X",
            status="completed",
            requests=[_n(1.2, [1, 2], in_=4, out=0, api=2.8)],
            models=["m"],
        ),
        WekaSubagentEntry(
            t=1.1,
            type="subagent",
            agent_id="b",
            subagent_type="X",
            status="completed",
            requests=[_n(1.3, [1, 3], in_=4, out=0, api=3.7)],
            models=["m"],
        ),
        WekaSubagentEntry(
            t=5.1,
            type="subagent",
            agent_id="c",
            subagent_type="X",
            status="completed",
            requests=[_n(5.2, [1, 5], in_=4, out=0, api=1.8)],
            models=["m"],
        ),
    ]
    trace = _trace(reqs, block_size=2)
    parsed, _ = build_trie_graph(trace, callbacks=_STUB_CB, idle_gap_cap_seconds=None)
    # Key nodes by their recorded hashes via the flattener (same walk the
    # build ran), not node metadata -- the trie stamp carries only the path.
    ids = {
        tuple(t.request.hash_ids): t.node_id
        for t in _flatten_requests(trace.requests, root_scope=trace.id)
    }
    preds = lambda h: {  # noqa: E731
        e.source for e in parsed.graph.edges if e.target == ids[h]
    }
    assert ids[(1, 3)] not in preds((1, 2)) and ids[(1, 2)] not in preds(
        (1, 3)
    )  # racers concurrent
    assert preds((1, 5)) == {ids[(1, 2)], ids[(1, 3)]}  # dynamic join, no parent turn


def test_e2e_shared_prefix_identical_prompt_ids():
    reqs = [
        _n(0.0, [10, 11], in_=4, out=0, api=0.5),
        WekaSubagentEntry(
            t=1.0,
            type="subagent",
            agent_id="a",
            subagent_type="X",
            status="completed",
            requests=[_n(1.1, [10, 11, 30], in_=6, out=0, api=0.5)],
            models=["m"],
        ),
        _n(2.0, [10, 11, 20], in_=6, out=0, api=0.5),
    ]
    trace = _trace(reqs, block_size=2)
    parsed, _ = build_trie_graph(trace, callbacks=_STUB_CB, idle_gap_cap_seconds=None)
    paths = {
        tuple(t.request.hash_ids): parsed.graph.nodes[t.node_id].metadata["trie"][
            "prompt_segment_ids"
        ]
        for t in _flatten_requests(trace.requests, root_scope=trace.id)
    }
    lead = paths[(10, 11)]
    for h, p in paths.items():
        if h[:2] == (10, 11):
            assert (
                p[: len(lead)] == lead
            )  # shared [10,11] prefix -> identical leading message ids


def test_e2e_coincident_zero_duration_single_edge():
    parsed, _ = build_trie_graph(
        _trace([_n(0.0, [1], in_=2), _n(0.0, [2], in_=2)], block_size=2),
        callbacks=_STUB_CB,
        idle_gap_cap_seconds=None,
    )
    assert len([e for e in parsed.graph.edges if e.source != "START"]) <= 1


# --- Task 8 fix: content-parent under-covers, child must tag ALL covered blocks ---


def test_under_covering_parent_child_fully_tagged():
    # parent: 3 hash blocks but in=4 (in//2=2) -> covers only 2 blocks (under-covers block index 2).
    # child: shares all 3 (lcp=3) and in=6 (covers 3). Child must tag all 3 covered blocks.
    reqs = [
        _n(
            0.0, [1, 2, 3], in_=4, out=0, api=0.1
        ),  # parent under-covers (2 of 3 blocks)
        _n(0.2, [1, 2, 3], in_=6, out=0, api=0.1),  # child covers all 3
    ]
    nodes, tags = _tags_for(reqs, 2)
    child = nodes[1]
    covered = min(len(child.request.hash_ids), child.request.input_length // 2)  # = 3
    assert len(tags[child.node_id]) == covered
    # end-to-end: no ISL abort
    parsed, _ = build_trie_graph(
        _trace(reqs, block_size=2), callbacks=_STUB_CB, idle_gap_cap_seconds=None
    )
    assert parsed is not None


def test_over_inheriting_child_clamped_to_covered_count():
    # Symmetric to the under-covering case: the child shares a FULL block prefix
    # (lcp=3) with its parent but declares a smaller in (in//2=1 < lcp). inherited
    # must clamp to the child's OWN covered count (1), not the lcp (3) -- else it
    # would carry 3 tags for a 1-block prompt and abort the build on assert_covered_isl.
    reqs = [
        _n(0.0, [1, 2, 3], in_=6, out=0, api=0.1),  # parent covers all 3 blocks
        _n(0.2, [1, 2, 3], in_=2, out=0, api=0.1),  # child covers only 1 (in//2=1)
    ]
    nodes, tags = _tags_for(reqs, 2)
    child = nodes[1]
    covered = min(len(child.request.hash_ids), child.request.input_length // 2)  # = 1
    assert len(tags[child.node_id]) == covered
    # end-to-end: no ISL abort, and the child's single tag is the parent's first
    # tag (shared-prefix identity preserved for the blocks the child does cover).
    assert tags[child.node_id][0] == tags[nodes[0].node_id][0]
    parsed, _ = build_trie_graph(
        _trace(reqs, block_size=2), callbacks=_STUB_CB, idle_gap_cap_seconds=None
    )
    assert parsed is not None
