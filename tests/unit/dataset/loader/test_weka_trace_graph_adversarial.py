# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial subagent-graph-pathology tests for WekaTraceLoader."""

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(*, max_isl=None, max_osl=None, start=None, end=None):
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = start
    uc.input.fixed_schedule_end_offset = end
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
    uc.input.synthesis.max_isl = max_isl
    uc.input.synthesis.max_osl = max_osl
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "t"
    uc.endpoint.model_names = ["m"]
    return uc


def _make_loader(filename, uc, monkeypatch):
    loader = WekaTraceLoader(filename=str(filename), user_config=uc)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = (
        lambda toks: f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def _subagent(
    agent_id,
    *,
    t=1.0,
    inner_model="m",
    inner=(("n", 0.0, 10, 1),),
    models=("m",),
    status="completed",
    duration_ms=1,
    total_tokens=0,
    tool_use_count=0,
):
    inner_reqs = [
        {"t": it, "type": "n", "model": inner_model, "in": ins, "out": outs}
        for _ty, it, ins, outs in inner
    ]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "X",
        "duration_ms": duration_ms,
        "total_tokens": total_tokens,
        "tool_use_count": tool_use_count,
        "status": status,
        "requests": inner_reqs,
        "models": list(models),
    }


def _normal(t=0.0, model="m", in_=10, out=1):
    return {"t": t, "type": "n", "model": model, "in": in_, "out": out}


def _build_trace(trace_id, requests, models=("m",)):
    return {
        "id": trace_id,
        "models": list(models),
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def test_terminal_subagent_at_trace_start_with_no_parents_dropped(
    tmp_path, monkeypatch
):
    """A trace with only a single subagent and no parent normals drops the
    branch (preceding and following both None). The parent conversation is
    empty AND the orphan child conversation is pruned (post-Task-7 fix), so
    only the empty parent remains.
    """
    data = _build_trace("t1", [_subagent("a1", t=0.0)])
    path = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {c.session_id for c in convs} == {"t1"}
    parent = next(c for c in convs if c.session_id == "t1")
    assert parent.turns == []
    assert parent.branches == []


def test_three_subagents_between_same_parent_turn_pair_collapse_to_one_multi_child_branch(
    tmp_path, monkeypatch
):
    """Three subagents sandwiched between the same preceding/following parent
    turn pair collapse into a single SPAWN branch with three
    child_conversation_ids and a single SPAWN_JOIN prereq on the following
    turn, so the v1 orchestrator validator accepts the topology.
    """
    requests = [
        _normal(t=0.0),
        _subagent("a1", t=1.0),
        _subagent("a2", t=2.0),
        _subagent("a3", t=3.0),
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert len(branch.child_conversation_ids) == 3
    assert set(branch.child_conversation_ids) == {
        "t1::sa:a1",
        "t1::sa:a2",
        "t1::sa:a3",
    }
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert len(parent.turns[1].prerequisites) == 1
    assert parent.turns[1].prerequisites[0].branch_id == branch.branch_id
    assert {c.session_id for c in convs} == {
        "t1",
        "t1::sa:a1",
        "t1::sa:a2",
        "t1::sa:a3",
    }


def test_multiple_terminal_subagents_collapse_to_one_background_branch(
    tmp_path, monkeypatch
):
    """Two terminal subagents after the final parent turn share the same
    (preceding, following=None) anchor pair and collapse into ONE background
    branch with two child_conversation_ids. No prereqs are emitted.
    """
    requests = [
        _normal(t=0.0),
        _subagent("a1", t=1.0),
        _subagent("a2", t=2.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is True
    assert set(branch.child_conversation_ids) == {"t1::sa:a1", "t1::sa:a2"}
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert parent.turns[0].prerequisites == []


def test_subagent_with_empty_inner_requests_emits_empty_child_conversation(
    tmp_path, monkeypatch
):
    """A subagent with an empty ``requests`` list currently produces a child
    conversation with zero turns. Documents current behavior (a downstream
    orchestrator consuming zero-turn children would be notable).
    """
    requests = [
        _normal(t=0.0),
        _subagent("a1", t=1.0, inner=()),
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "t1::sa:a1")
    assert len(child.turns) == 0


def test_parent_has_only_subagents_no_normals_emits_no_turns(tmp_path, monkeypatch):
    """A trace consisting exclusively of subagent entries (no parent normals)
    yields a parent conversation with empty turns and empty branches (both
    anchors None -> dropped). Post-Task-7 fix, the orphan child conversations
    are also pruned, so only the empty parent remains.
    """
    requests = [
        _subagent("a1", t=1.0),
        _subagent("a2", t=2.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert parent.turns == []
    assert parent.branches == []
    session_ids = {c.session_id for c in convs}
    assert session_ids == {"t1"}


def test_subagent_status_async_launched_with_null_telemetry_parses_and_converts(
    tmp_path, monkeypatch
):
    """A subagent with ``status='async_launched'`` and ``duration_ms``,
    ``total_tokens``, ``tool_use_count`` all None (telemetry not captured)
    plus an empty inner-requests list parses successfully and still emits a
    SPAWN branch on the parent conversation.
    """
    requests = [
        _normal(t=0.0),
        _subagent(
            "a1",
            t=1.0,
            inner=(),
            status="async_launched",
            duration_ms=None,
            total_tokens=None,
            tool_use_count=None,
        ),
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1
    assert parent.branches[0].mode == ConversationBranchMode.SPAWN


def test_subagent_inner_decreasing_timestamps_produce_negative_delay(
    tmp_path, monkeypatch
):
    """A subagent whose inner requests appear in the trace with decreasing
    ``t`` (5.0 then 3.0) is sorted by ``t`` during stream-packing, so the
    child turns end up in monotonic order with a positive +2s delay
    (5.0 - 3.0). Documents the post-stream-packing contract: inner requests
    are reordered by ``t`` rather than preserved in raw insertion order.
    """
    requests = [
        _normal(t=0.0),
        _subagent(
            "a1",
            t=1.0,
            inner=(("n", 5.0, 10, 1), ("n", 3.0, 10, 1)),
        ),
        _normal(t=10.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "t1::sa:a1")
    assert child.turns[1].delay == pytest.approx(2000.0)


def test_subagent_inner_models_mismatch_declared_models_no_error(tmp_path, monkeypatch):
    """A subagent's declared ``models`` list is not cross-checked against the
    model field of its inner requests. Both models appear in the endpoint
    allow-list so validation succeeds. Documents the lack of cross-check.
    """
    requests = [
        _normal(t=0.0),
        _subagent(
            "a1",
            t=1.0,
            inner_model="m",
            models=("declared",),
        ),
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests, models=("m",)))
    uc = _mk_user_config()
    uc.endpoint.model_names = ["declared", "m"]
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1


def test_subagent_with_hundred_inner_turns_scales(tmp_path, monkeypatch):
    """A subagent with 100 inner normal requests produces a child
    conversation with exactly 100 turns. Smoke test for large inner fanout.
    """
    inner = tuple(("n", float(i), 10, 1) for i in range(100))
    requests = [
        _normal(t=0.0),
        _subagent("a1", t=1.0, inner=inner),
        _normal(t=500.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "t1::sa:a1")
    assert len(child.turns) == 100


def test_trace_with_hundred_subagents_collapse_to_single_branch(tmp_path, monkeypatch):
    """100 subagents sandwiched between two parent turns all share the same
    (preceding, following) anchor pair and collapse into a single SPAWN
    branch with 100 child_conversation_ids and ONE prereq on the following
    turn, so the v1 orchestrator validator accepts the topology.
    """
    subagents = [_subagent(f"a{i}", t=float(i + 1)) for i in range(100)]
    requests = [_normal(t=0.0), *subagents, _normal(t=200.0)]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert len(branch.child_conversation_ids) == 100
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert len(parent.turns[1].prerequisites) == 1
    assert parent.turns[1].prerequisites[0].branch_id == branch.branch_id


def test_subagent_duration_tokens_tool_count_all_none_non_async_accepted(
    tmp_path, monkeypatch
):
    """A subagent with status='completed' (non-async) but all three
    telemetry fields (duration_ms, total_tokens, tool_use_count) set to None
    parses and converts without error. Documents that the model does not
    enforce non-null telemetry for non-async subagents.
    """
    requests = [
        _normal(t=0.0),
        _subagent(
            "a1",
            t=1.0,
            status="completed",
            duration_ms=None,
            total_tokens=None,
            tool_use_count=None,
        ),
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1


def test_subagent_requests_ordering_preserved_in_child_conversation(
    tmp_path, monkeypatch
):
    """Inner request ordering is preserved, with timestamps on the root
    timeline: the spawn-relative inner ``t`` values 0.0, 1.0, 2.0 shift by
    the marker's t=5.0 (see ``_subagent_request_absolute_t``) but keep the
    inner-list order.
    """
    requests = [
        _normal(t=0.0),
        _subagent(
            "a1",
            t=5.0,
            inner=(("n", 0.0, 10, 1), ("n", 1.0, 10, 1), ("n", 2.0, 10, 1)),
        ),
        _normal(t=10.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    child = next(c for c in convs if c.session_id == "t1::sa:a1")
    assert [t.timestamp for t in child.turns] == [5000.0, 6000.0, 7000.0]


def test_terminal_subagent_after_filter_killed_final_turn_reanchors_to_earlier(
    tmp_path, monkeypatch
):
    """When max_isl filters out what was originally the subagent's following
    parent turn, and no later parent turn exists, the subagent still anchors
    to the earlier surviving parent and becomes a background branch.
    """
    requests = [
        _normal(t=0.0, in_=10),
        _normal(t=1.0, in_=500),  # filtered out by max_isl=100
        _subagent("a1", t=2.0),  # originally terminal; still terminal after filter
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    uc = _mk_user_config(max_isl=100)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.turns) == 1
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is True
    assert parent.turns[0].branch_ids == [branch.branch_id]


def test_two_subagents_around_filter_killed_middle_parent_both_reanchor(
    tmp_path, monkeypatch
):
    """With p0, p1(killed by max_isl), p2 and a subagent on each side of p1,
    both subagents re-anchor to the survivors with the same (preceding=p0,
    following=p2) pair, so they collapse into ONE branch with two
    child_conversation_ids and one SPAWN_JOIN prereq on p2.
    """
    requests = [
        _normal(t=0.0, in_=50),  # p0: outer 0
        _subagent("a1", t=0.5),  # outer 1
        _normal(t=1.0, in_=500),  # p1: outer 2, filtered
        _subagent("a2", t=1.5),  # outer 3
        _normal(t=2.0, in_=50),  # p2: outer 4
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    uc = _mk_user_config(max_isl=100)
    loader = _make_loader(path, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t1")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert set(branch.child_conversation_ids) == {"t1::sa:a1", "t1::sa:a2"}
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert len(parent.turns[1].prerequisites) == 1
    assert parent.turns[1].prerequisites[0].branch_id == branch.branch_id
    assert {c.session_id for c in convs} == {"t1", "t1::sa:a1", "t1::sa:a2"}


def test_subagent_inner_hash_id_collision_with_parent_does_not_raise(
    tmp_path, monkeypatch
):
    """Hash-id overlap between parent requests (hash_ids=[1,2,3]) and a
    subagent's inner request (hash_ids=[1,2]) does not raise; both parent
    and child conversations are emitted.
    """
    requests = [
        {
            "t": 0.0,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "hash_ids": [1, 2, 3],
        },
        {
            "t": 1.0,
            "type": "subagent",
            "agent_id": "a1",
            "subagent_type": "X",
            "duration_ms": 1,
            "total_tokens": 0,
            "tool_use_count": 0,
            "status": "completed",
            "requests": [
                {
                    "t": 0.0,
                    "type": "n",
                    "model": "m",
                    "in": 10,
                    "out": 1,
                    "hash_ids": [1, 2],
                }
            ],
            "models": ["m"],
        },
        _normal(t=5.0),
    ]
    path = _write_trace(tmp_path, _build_trace("t1", requests))
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {c.session_id for c in convs} == {"t1", "t1::sa:a1"}


def test_orphan_child_pruned_when_parent_has_only_subagent(tmp_path, monkeypatch):
    """Parent with zero normal requests and one subagent: subagent drops,
    child conversation must also drop.
    """
    data = _build_trace(
        "only_sa",
        [
            _subagent("a1", t=0.0, inner=(("n", 0.0, 10, 1),)),
        ],
    )
    p = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(p, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {c.session_id for c in convs} == {"only_sa"}


def test_subagent_at_trace_index_zero_dropped_with_info_log(
    tmp_path, monkeypatch, caplog
):
    """Subagent with no preceding parent turn is dropped, matching the symmetry
    of terminal-first subagents. Prior to the fix, a branch was created but no
    turn declared it in branch_ids, producing an orphan branch.
    """
    import logging

    data = _build_trace(
        "sa_first",
        [
            _subagent("a1", t=0.0, inner=(("n", 0.0, 10, 1),)),
            _normal(t=2.0, in_=5),
        ],
    )
    p = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(p, uc, monkeypatch)
    with caplog.at_level(logging.INFO):
        convs = loader.convert_to_conversations(loader.load_dataset())
    # Only parent with its single surviving turn; no child, no branch.
    assert {c.session_id for c in convs} == {"sa_first"}
    parent = convs[0]
    assert len(parent.turns) == 1
    assert parent.branches == []
    assert parent.turns[0].prerequisites == []
    assert any("Dropping subagent 'a1'" in rec.message for rec in caplog.records)


def test_three_adjacent_subagents_collapse_into_one_multi_child_branch(
    tmp_path, monkeypatch
):
    """3 back-to-back subagents between the same parent-turn pair must emit
    ONE branch with 3 child_conversation_ids and ONE SPAWN_JOIN prereq, so the
    topology passes validate_for_orchestrator_v1.
    """
    data = _build_trace(
        "collapse",
        [
            _normal(t=0.0, in_=10),
            _subagent("a1", t=1.0, inner=(("n", 0.0, 5, 1),)),
            _subagent("a2", t=2.0, inner=(("n", 0.0, 5, 1),)),
            _subagent("a3", t=3.0, inner=(("n", 0.0, 5, 1),)),
            _normal(t=5.0, in_=10),
        ],
    )
    p = _write_trace(tmp_path, data)
    uc = _mk_user_config()
    loader = _make_loader(p, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "collapse")
    assert len(parent.branches) == 1, (
        f"expected 1 collapsed branch, got {len(parent.branches)}"
    )
    branch = parent.branches[0]
    assert len(branch.child_conversation_ids) == 3
    assert set(branch.child_conversation_ids) == {
        "collapse::sa:a1",
        "collapse::sa:a2",
        "collapse::sa:a3",
    }
    assert parent.turns[0].branch_ids == [branch.branch_id]
    assert len(parent.turns[1].prerequisites) == 1
    assert parent.turns[1].prerequisites[0].branch_id == branch.branch_id
    assert {c.session_id for c in convs} - {"collapse"} == {
        "collapse::sa:a1",
        "collapse::sa:a2",
        "collapse::sa:a3",
    }
