# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async-subagent and parallel-inner-request replay in WekaTraceLoader.

Reuses the helpers from test_weka_trace_graph_adversarial.py: same
``_subagent``/``_normal``/``_build_trace``/``_make_loader`` pattern, same fixture
loader path.
"""

from unittest.mock import MagicMock

import orjson

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.dataset.loader.weka_trace import WekaTraceLoader


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
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


def _subagent(agent_id, *, t, duration_ms, inner):
    """inner: list of (t_offset_seconds, api_time_seconds_or_None)."""
    inner_reqs = [
        {
            "t": t + dt,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_t,
        }
        for dt, api_t in inner
    ]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "X",
        "duration_ms": duration_ms,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": inner_reqs,
        "models": ["m"],
    }


def _normal(t, model="m", in_=10, out=1):
    return {"t": t, "type": "n", "model": model, "in": in_, "out": out}


def _build_trace(trace_id, requests, models=("m",)):
    return {
        "id": trace_id,
        "models": list(models),
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def test_subagent_running_past_following_parent_is_background(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 > following_parent.t -> branch is_background=True,
    no SPAWN_JOIN prerequisite.
    """
    data = _build_trace(
        "t_async",
        [
            _normal(t=0.0),
            # sa starts at t=1, runs 100 seconds, ends at t=101.
            _subagent("a1", t=1.0, duration_ms=100_000, inner=[(0.0, 100.0)]),
            # following parent at t=2 - well before sa_end at t=101.
            _normal(t=2.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_async")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.is_background is True, (
        "Subagent runs past following parent turn - parent didn't wait. "
        "Expected is_background=True, got False."
    )
    # No SPAWN_JOIN prerequisite on any parent turn for this branch.
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert not (
                prereq.kind == PrerequisiteKind.SPAWN_JOIN
                and prereq.branch_id == branch.branch_id
            ), "background branch should not have a SPAWN_JOIN prerequisite"


def test_subagent_finishing_before_following_parent_keeps_join(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 < following_parent.t -> branch has SPAWN_JOIN,
    is_background=False (current behavior, regression guard).
    """
    data = _build_trace(
        "t_sync",
        [
            _normal(t=0.0),
            # sa runs 1s, ends at t=2.
            _subagent("a1", t=1.0, duration_ms=1000, inner=[(0.0, 1.0)]),
            # following parent at t=10 - well after sa_end at t=2.
            _normal(t=10.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_sync")
    branch = parent.branches[0]
    assert branch.is_background is False
    # SPAWN_JOIN must be on the following parent turn.
    following_turn = parent.turns[1]
    join_prereqs = [
        p
        for p in following_turn.prerequisites
        if p.kind == PrerequisiteKind.SPAWN_JOIN and p.branch_id == branch.branch_id
    ]
    assert len(join_prereqs) == 1


def test_subagent_duration_ms_none_falls_back_to_inner_api_time(tmp_path, monkeypatch):
    """When duration_ms is None (status='async_launched' style), end-time is
    inferred from max(inner.t + inner.api_time)."""
    data = _build_trace(
        "t_no_dur",
        [
            _normal(t=0.0),
            # duration_ms=None, but inner request runs from t=1 to t=51.
            _subagent("a1", t=1.0, duration_ms=None, inner=[(0.0, 50.0)]),
            _normal(t=2.0),  # well before sa_end at t=51.
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_no_dur")
    branch = parent.branches[0]
    assert branch.is_background is True
