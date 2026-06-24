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
from aiperf.common.environment import Environment
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
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
    # start_timestamp_ms is the subagent's mapped spawn time (sa.t=1 -> 1000ms),
    # used to reconstruct in-flight subagents when sampling a mid-trace snapshot.
    assert branch.start_timestamp_ms == 1000.0
    # Per-turn api_time_ms is carried for happens-before gating: the subagent's
    # inner request recorded api_time=1.0s -> 1000.0ms on its child turn.
    child = next(c for c in convs if c.session_id.startswith("t_sync::sa:"))
    assert child.turns[0].api_time_ms == 1000.0
    # Top-level _normal rows carry no api_time -> None (no interval width).
    assert parent.turns[0].api_time_ms is None


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


def test_overlapping_inner_requests_without_hash_evidence_stay_one_child(
    tmp_path, monkeypatch
):
    """Inner requests without hash evidence ride the main chain even when
    their [t, t+api_time] intervals overlap.

    Nested LCP chain detection only splits on hash-prefix evidence; requests
    with empty ``hash_ids`` carry none, so they stay one sequential child
    conversation in time order (the legacy interval packing that split these
    by overlap alone is gone).
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
    assert branch.child_conversation_ids == ["t_par::sa:a1"]
    child = next(c for c in convs if c.session_id == "t_par::sa:a1")
    assert len(child.turns) == 2


def test_interleaved_inner_threads_split_into_lineage_chains(tmp_path, monkeypatch):
    """Interleaved inner context threads split by hash-prefix lineage.

    Two interleaved threads (A: blocks [1, ...], B: blocks [50, ...]) become
    the main chain (A, founded by the subagent's first request) and one
    spawned chain (B), each holding its own context lineage — the old
    time-interval packing would have stitched [A1, A2, B2] into one stream
    whenever they didn't overlap pairwise.
    """

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
            inner(1.0, 10.0, [1]),  # thread A founds the main chain
            inner(6.0, 3.0, [50]),  # thread B: disjoint context -> spawned chain
            inner(12.0, 0.5, [1, 2]),  # extends A's tail
            inner(13.0, 1.0, [50, 51]),  # extends B's tail
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
    """A single disjoint inner call is the subagent's own sidecar.

    Thread B here is one fresh-context request (not the 2-request chain above),
    so with aux classification on it is the subagent's one-shot sidecar:
    ::sa:a1:aux:000, not the :fa:000 agent tag.
    """
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
            inner(1.0, 10.0, [1]),  # founds the main chain
            inner(6.0, 1.0, [50]),  # single disjoint one-shot -> sidecar
            inner(12.0, 0.5, [1, 2]),  # extends the main chain
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
    """Regression: a leading prefix-disjoint preamble on a DIFFERENT model must
    not redefine the subagent's classification yardstick.

    Inner stream: a Haiku title-gen preamble (peeled by _split_off_preamble and
    re-attached to the main chain only for replay), an Opus main chain, and an
    Opus single-request spawned chain that is large (>= aux ISL floor) with a
    generative output (>= reduction OSL max). That spawned chain is same-model
    as the DETECTED main chain and neither small-fresh nor a reduction, so it is
    a genuine agent (:fa:). Deriving main_model from chains[0] -- which has the
    re-attached Haiku preamble sorted first -- would make it cross-model vs Haiku
    and mis-tag it as an :aux: sidecar (the cross-model arm fires regardless of
    size/output).
    """
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
            # Haiku title-gen preamble: earliest, prefix-disjoint, small output
            # -> peeled and re-attached to the main chain only for replay.
            inner(1.0, "haiku", in_=200, out=10, hash_ids=[900]),
            # Opus main chain (two requests sharing a prefix) -> founds the chain.
            inner(2.0, "opus", in_=64, out=10, hash_ids=[1]),
            inner(4.0, "opus", in_=128, out=10, hash_ids=[1, 2]),
            # Opus single-request spawned chain: large fresh context (>= aux ISL
            # floor 16384) with generative output (>= reduction OSL max 4000), so
            # only the cross-model arm could flip it -- a genuine nested agent.
            inner(3.0, "opus", in_=20000, out=5000, hash_ids=[50]),
        ],
        "models": ["opus", "haiku"],
    }
    uc = _mk_user_config()
    uc.endpoint.model_names = ["opus", "haiku"]
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
    # Same-model large spawned chain is a genuine agent, not a cross-model sidecar.
    assert "t_preamble::sa:a1:fa:000" in sids, sorted(sids)
    assert "t_preamble::sa:a1:aux:000" not in sids, sorted(sids)


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
        "a single sequential chain keeps the legacy session-id shape (no :cNNN suffix)"
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


def test_parallel_inner_chains_under_parallel_reconstruction(tmp_path, monkeypatch):
    """Nested chain detection produces identical children under the
    multiprocessing reconstruction path.

    Two overlapping inner requests forking the same context (shared [1]
    prefix, second still in flight when the first's continuation cannot
    extend) become the main chain plus one spawned chain sibling.
    """

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
            inner(1.0, 100.0, [1, 2]),  # main chain, in flight until t=101
            inner(1.1, 100.0, [1, 3]),  # parallel fork of the shared [1] prefix
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
    """End-to-end regression against the real captured trace.

    Trace shape (verified by inspection):
      - 7 streaming parent turns at t=0, 13.01, 23.89, 32.36, 36.54, 271.10, 280.18
      - 1 subagent at outer index 4 (t=33.161, duration_ms=246584)
        with TWO overlapping inner requests (api_time ~237s each)

    Expected loader output:
      - 1 SPAWN branch with is_background=False because the subagent end
        joins the later parent turn at t=280.18
      - 2 sibling child conversations: the main chain
        '<trace>::sa:codex_subagent_001' plus one spawned chain ':fa:000'
        (the second inner request forks the shared 548-block prefix while
        the first is still in flight)
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

    # Both children exist and each has exactly one turn.
    sid_main = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001"
    sid_fork = "91a41301c26657b2500e2dc71141217dd11b::sa:codex_subagent_001:fa:000"
    children_by_sid = {c.session_id: c for c in convs}
    assert sid_main in children_by_sid
    assert sid_fork in children_by_sid
    assert len(children_by_sid[sid_main].turns) == 1
    assert len(children_by_sid[sid_fork].turns) == 1
