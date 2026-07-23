# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.unit.dataset.loader._shared_helpers import _make_loader, _write_trace

"""Tests for async-subagent and parallel-inner-request replay in WekaTraceLoader."""

from pathlib import Path

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.environment import Environment

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config(model_names=None):
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=model_names or ["m"], tokenizer_name="t")


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


def test_subagent_running_past_following_parent_is_background(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 > following_parent.t -> branch is_background=True,"""
    data = _build_trace(
        "t_async",
        [
            _normal(t=0.0),
            _subagent("a1", t=1.0, duration_ms=100_000, inner=[(0.0, 100.0)]),
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
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert not (
                prereq.kind == PrerequisiteKind.SPAWN_JOIN
                and prereq.branch_id == branch.branch_id
            ), "background branch should not have a SPAWN_JOIN prerequisite"


def test_subagent_finishing_before_following_parent_keeps_join(tmp_path, monkeypatch):
    """sa.t + duration_ms/1000 < following_parent.t -> branch has SPAWN_JOIN,"""
    data = _build_trace(
        "t_sync",
        [
            _normal(t=0.0),
            _subagent("a1", t=1.0, duration_ms=1000, inner=[(0.0, 1.0)]),
            _normal(t=10.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_sync")
    branch = parent.branches[0]
    assert branch.is_background is False
    following_turn = parent.turns[1]
    join_prereqs = [
        p
        for p in following_turn.prerequisites
        if p.kind == PrerequisiteKind.SPAWN_JOIN and p.branch_id == branch.branch_id
    ]
    assert len(join_prereqs) == 1
    assert branch.start_timestamp_ms == 1000.0
    child = next(c for c in convs if c.session_id.startswith("t_sync::sa:"))
    assert child.turns[0].api_time_ms == 1000.0
    assert parent.turns[0].api_time_ms is None


def test_subagent_duration_ms_none_falls_back_to_inner_api_time(tmp_path, monkeypatch):
    """When duration_ms is None (status='async_launched' style), end-time is"""
    data = _build_trace(
        "t_no_dur",
        [
            _normal(t=0.0),
            _subagent("a1", t=1.0, duration_ms=None, inner=[(0.0, 50.0)]),
            _normal(t=2.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_no_dur")
    branch = parent.branches[0]
    assert branch.is_background is True


def test_overlapping_inner_requests_without_hash_evidence_stay_one_child(
    tmp_path, monkeypatch
):
    """Inner requests without hash evidence ride the main chain even when"""
    data = _build_trace(
        "t_par",
        [
            _normal(t=0.0),
            _subagent(
                "a1",
                t=1.0,
                duration_ms=100_000,
                inner=[(0.0, 100.0), (0.1, 100.0)],
            ),
            _normal(t=200.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_par")
    branch = parent.branches[0]
    assert branch.child_conversation_ids == ["t_par::sa:a1"]
    child = next(c for c in convs if c.session_id == "t_par::sa:a1")
    assert len(child.turns) == 2


def test_interleaved_inner_threads_split_into_lineage_chains(tmp_path, monkeypatch):
    """Interleaved inner context threads split by hash-prefix lineage."""

    def inner(t, api_time, hash_ids):
        return {
            "t": t,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_time,
            "hash_ids": hash_ids,
        }

    sa = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 20_000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": [
            inner(1.0, 10.0, [1]),
            inner(6.0, 3.0, [50]),
            inner(12.0, 0.5, [1, 2]),
            inner(13.0, 1.0, [50, 51]),
        ],
        "models": ["m"],
    }
    data = _build_trace("t_affinity", [_normal(t=0.0), sa, _normal(t=30.0)])
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    main = next(c for c in convs if c.session_id == "t_affinity::sa:a1")
    worker = next(c for c in convs if c.session_id == "t_affinity::sa:a1:fa:000")
    assert len(main.turns) == 2
    assert len(worker.turns) == 2


def test_subagent_one_shot_overflow_is_tagged_aux_sidecar(tmp_path, monkeypatch):
    """A single disjoint inner call is the subagent's own sidecar."""
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_MAX_REQUESTS", 1)

    def inner(t, api_time, hash_ids):
        return {
            "t": t,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_time,
            "hash_ids": hash_ids,
        }

    sa = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 20_000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": [
            inner(1.0, 10.0, [1]),
            inner(6.0, 1.0, [50]),
            inner(12.0, 0.5, [1, 2]),
        ],
        "models": ["m"],
    }
    data = _build_trace("t_aux", [_normal(t=0.0), sa, _normal(t=30.0)])
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = {
        c.session_id for c in loader.convert_to_conversations(loader.load_dataset())
    }
    assert "t_aux::sa:a1" in convs
    assert "t_aux::sa:a1:aux:000" in convs, sorted(convs)
    assert "t_aux::sa:a1:fa:000" not in convs, sorted(convs)


def test_nested_subagent_preamble_does_not_contaminate_main_model(
    tmp_path, monkeypatch
):
    """Regression: a leading prefix-disjoint preamble on a DIFFERENT model must"""
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_MAX_REQUESTS", 1)
    monkeypatch.setattr(Environment.DATASET, "WEKA_AUX_CROSS_MODEL", True)

    def inner(t, model, in_, out, hash_ids, api_time=1.0):
        return {
            "t": t,
            "type": "n",
            "model": model,
            "in": in_,
            "out": out,
            "api_time": api_time,
            "hash_ids": hash_ids,
        }

    sa = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 20_000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": [
            inner(1.0, "haiku", in_=200, out=10, hash_ids=[900]),
            inner(2.0, "opus", in_=64, out=10, hash_ids=[1]),
            inner(4.0, "opus", in_=128, out=10, hash_ids=[1, 2]),
            inner(3.0, "opus", in_=20000, out=5000, hash_ids=[50]),
        ],
        "models": ["opus", "haiku"],
    }
    uc = _mk_user_config(model_names=["opus", "haiku"])
    data = _build_trace(
        "t_preamble",
        [_normal(t=0.0, model="opus"), sa, _normal(t=30.0, model="opus")],
        models=("opus", "haiku"),
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, uc, monkeypatch)
    sids = {
        c.session_id for c in loader.convert_to_conversations(loader.load_dataset())
    }
    assert "t_preamble::sa:a1:fa:000" in sids, sorted(sids)
    assert "t_preamble::sa:a1:aux:000" not in sids, sorted(sids)


def test_subagent_with_sequential_inner_requests_emits_one_child_conversation(
    tmp_path, monkeypatch
):
    """Two non-overlapping inner requests stay in ONE child Conversation as two"""
    data = _build_trace(
        "t_seq",
        [
            _normal(t=0.0),
            _subagent(
                "a1",
                t=1.0,
                duration_ms=3000,
                inner=[(0.0, 1.0), (2.0, 1.0)],
            ),
            _normal(t=10.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_seq")
    branch = parent.branches[0]
    assert branch.child_conversation_ids == ["t_seq::sa:a1"], (
        "a single sequential chain keeps the legacy session-id shape (no :cNNN suffix)"
    )
    child = next(c for c in convs if c.session_id == "t_seq::sa:a1")
    assert len(child.turns) == 2


def _install_inproc_pool(monkeypatch, loader):
    """Replace multiprocessing Pool with synchronous in-process stub."""
    from aiperf.dataset.loader import weka_parallel_convert as wpc

    pg = loader.prompt_generator

    class _InProcPool:
        def __init__(self, num_workers, init_fn, init_args) -> None:
            init_fn(init_args[0])

        def imap(self, fn, items, chunksize=1):
            return [fn(it) for it in items]

        def close(self) -> None:
            return None

        def join(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    class _FakeCtx:
        Pool = _InProcPool

    monkeypatch.setattr(wpc, "get_loader_mp_context", lambda **kw: _FakeCtx())
    monkeypatch.setattr(wpc.Tokenizer, "from_pretrained", lambda *a, **kw: pg.tokenizer)


def _force_parallel(monkeypatch, loader):
    """Force ``convert_to_conversations`` onto the parallel reconstruction path."""
    from aiperf.common.environment import Environment
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator

    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
    loader.prompt_generator._hash_id_corpus_rng = HashIdRandomGenerator(
        12345, _internal=True
    )
    loader.prompt_generator._bpe_stable_terminator_tokens = []
    _install_inproc_pool(monkeypatch, loader)


def test_async_branch_detected_under_parallel_reconstruction(tmp_path, monkeypatch):
    """Same async-detection under the multiprocessing path."""
    data = _build_trace(
        "t_par_async",
        [
            _normal(t=0.0),
            _subagent("a1", t=1.0, duration_ms=100_000, inner=[(0.0, 100.0)]),
            _normal(t=2.0),
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    _force_parallel(monkeypatch, loader)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t_par_async")
    branch = parent.branches[0]
    assert branch.is_background is True
    for turn in parent.turns:
        for prereq in turn.prerequisites:
            assert not (
                prereq.kind == PrerequisiteKind.SPAWN_JOIN
                and prereq.branch_id == branch.branch_id
            ), "background branch should not have a SPAWN_JOIN prerequisite"


def test_parallel_inner_chains_under_parallel_reconstruction(tmp_path, monkeypatch):
    """Nested chain detection produces identical children under the"""

    def inner(t, api_time, hash_ids):
        return {
            "t": t,
            "type": "n",
            "model": "m",
            "in": 10,
            "out": 1,
            "api_time": api_time,
            "hash_ids": hash_ids,
        }

    sa = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 100_000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": [
            inner(1.0, 100.0, [1, 2]),
            inner(1.1, 100.0, [1, 3]),
        ],
        "models": ["m"],
    }
    data = _build_trace("t_par_split", [_normal(t=0.0), sa, _normal(t=200.0)])
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    _force_parallel(monkeypatch, loader)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t_par_split")
    branch = parent.branches[0]
    assert branch.child_conversation_ids == [
        "t_par_split::sa:a1",
        "t_par_split::sa:a1:fa:000",
    ]
    children = {
        c.session_id: c for c in convs if c.session_id.startswith("t_par_split::sa")
    }
    assert set(children.keys()) == set(branch.child_conversation_ids)
    for sid in children:
        assert len(children[sid].turns) == 1


def test_async_subagent_with_parallel_inner_real_trace(tmp_path, monkeypatch):
    """End-to-end regression against the real captured trace."""
    src = FIXTURES / "async_subagent_with_parallel_inner.json"
    assert src.exists(), f"regression fixture missing: {src}"
    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())

    uc = _mk_user_config()
    loader = _make_loader(dst, uc, monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(
        c for c in convs if c.session_id == "91a41301c26657b2500e2dc71141217dd11b"
    )
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.is_background is False
    assert set(branch.child_conversation_ids) == {
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001",
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:fa:000",
    }

    join_turns = [
        idx
        for idx, turn in enumerate(parent.turns)
        for prereq in turn.prerequisites
        if prereq.kind == PrerequisiteKind.SPAWN_JOIN
        and prereq.branch_id == branch.branch_id
    ]
    assert join_turns == [6]

    sid_main = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001"
    sid_fork = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:fa:000"
    children_by_sid = {c.session_id: c for c in convs}
    assert sid_main in children_by_sid
    assert sid_fork in children_by_sid
    assert len(children_by_sid[sid_main].turns) == 1
    assert len(children_by_sid[sid_fork].turns) == 1
