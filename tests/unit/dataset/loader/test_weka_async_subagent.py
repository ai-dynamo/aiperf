# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async-subagent and parallel-inner-request replay in WekaTraceLoader.

Reuses the helpers from test_weka_trace_graph_adversarial.py: same
``_subagent``/``_normal``/``_build_trace``/``_make_loader`` pattern, same fixture
loader path.
"""

from pathlib import Path
from unittest.mock import MagicMock

import orjson

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
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
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
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


def test_subagent_with_overlapping_inner_requests_emits_separate_child_conversations(
    tmp_path, monkeypatch
):
    """Two inner requests with overlapping [t, t+api_time] become two child
    Conversations under one multi-child SPAWN branch.
    """
    data = _build_trace(
        "t_par",
        [
            _normal(t=0.0),
            # Two inner requests at t=1 and t=1.1, both running 100s - overlap ~99.9s.
            _subagent(
                "a1",
                t=1.0,
                duration_ms=100_000,
                inner=[(0.0, 100.0), (0.1, 100.0)],
            ),
            _normal(t=200.0),  # well after both inner ends; SPAWN_JOIN-eligible.
        ],
    )
    path = _write_trace(tmp_path, data)
    loader = _make_loader(path, _mk_user_config(), monkeypatch)
    convs = loader.convert_to_conversations(loader.load_dataset())

    parent = next(c for c in convs if c.session_id == "t_par")
    branch = parent.branches[0]
    # Two streams -> two child conversations as siblings in the branch.
    assert len(branch.child_conversation_ids) == 2, (
        f"Expected 2 sibling child conversations, got {branch.child_conversation_ids}"
    )
    expected_sids = {"t_par::sa:a1:s0", "t_par::sa:a1:s1"}
    assert set(branch.child_conversation_ids) == expected_sids

    children = {c.session_id: c for c in convs if c.session_id.startswith("t_par::sa")}
    assert set(children.keys()) == expected_sids
    for sid in expected_sids:
        assert len(children[sid].turns) == 1, (
            f"each parallel stream is one inner request -> one turn; "
            f"{sid} has {len(children[sid].turns)} turns"
        )


def test_subagent_with_sequential_inner_requests_emits_one_child_conversation(
    tmp_path, monkeypatch
):
    """Two non-overlapping inner requests stay in ONE child Conversation as two
    sequential turns (regression: don't fragment serial inners).
    """
    data = _build_trace(
        "t_seq",
        [
            _normal(t=0.0),
            # Inner 0: t=1, runs 1s (ends t=2). Inner 1: t=3, runs 1s (ends t=4).
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
        "single sequential stream keeps the legacy session-id shape (no :s0 suffix)"
    )
    child = next(c for c in convs if c.session_id == "t_seq::sa:a1")
    assert len(child.turns) == 2


def _install_inproc_pool(monkeypatch, loader):
    """Replace multiprocessing Pool with synchronous in-process stub.

    Mirrors ``tests/component_integration/test_agentic_replay_e2e.py``'s
    ``_install_inproc_pool``. Lets unit tests drive ``_reconstruct_parallel``
    end-to-end without spawning real worker processes (which would re-import
    a real tokenizer the MagicMock fixtures don't carry).
    """
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

    # Conftest pins WORKERS=1 (forces serial); override for these tests.
    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
    monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
    # Parallel path reads pg._hash_id_corpus_rng.seed and ships it to workers;
    # a MagicMock's auto-attr is not a real int. Replace with a real RNG.
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


def test_parallel_inner_split_under_parallel_reconstruction(tmp_path, monkeypatch):
    """Two overlapping inner requests become two sibling child Conversations
    under the parallel reconstruction path."""
    data = _build_trace(
        "t_par_split",
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
    _force_parallel(monkeypatch, loader)
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "t_par_split")
    branch = parent.branches[0]
    assert set(branch.child_conversation_ids) == {
        "t_par_split::sa:a1:s0",
        "t_par_split::sa:a1:s1",
    }
    children = {
        c.session_id: c for c in convs if c.session_id.startswith("t_par_split::sa")
    }
    assert set(children.keys()) == {
        "t_par_split::sa:a1:s0",
        "t_par_split::sa:a1:s1",
    }
    for sid in children:
        assert len(children[sid].turns) == 1


def test_async_subagent_with_parallel_inner_real_trace(tmp_path, monkeypatch):
    """End-to-end regression against the real captured trace.

    Trace shape (verified by inspection):
      - 7 streaming parent turns at t=0, 13.01, 23.89, 32.36, 36.54, 271.10, 280.18
      - 1 subagent at outer index 4 (t=33.161, duration_ms=246584)
        with TWO overlapping inner requests (api_time ~237s each)

    Expected loader output:
      - 1 SPAWN branch with is_background=False because the subagent end
        joins the later parent turn at t=280.18
      - 2 sibling child conversations with session ids
        '<trace>::sa:codex_subagent_001:s0' and ':s1'
      - No SPAWN_JOIN prerequisite on the immediate t=36.54 parent turn
    """
    src = FIXTURES / "async_subagent_with_parallel_inner.json"
    assert src.exists(), f"regression fixture missing: {src}"
    # Loader requires a single file path or directory; copy into tmp_path
    # so we don't depend on the fixture location at runtime.
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
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s0",
        "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s1",
    }

    join_turns = [
        idx
        for idx, turn in enumerate(parent.turns)
        for prereq in turn.prerequisites
        if prereq.kind == PrerequisiteKind.SPAWN_JOIN
        and prereq.branch_id == branch.branch_id
    ]
    assert join_turns == [6]

    # Both children exist and each has exactly one turn.
    sid_s0 = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s0"
    sid_s1 = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:s1"
    children_by_sid = {c.session_id: c for c in convs}
    assert sid_s0 in children_by_sid
    assert sid_s1 in children_by_sid
    assert len(children_by_sid[sid_s0].turns) == 1
    assert len(children_by_sid[sid_s1].turns) == 1
