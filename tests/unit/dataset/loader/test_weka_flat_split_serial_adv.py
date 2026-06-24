# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for WekaTraceLoader serial-path flattened-agent split.

Surface: ``weka_trace.py`` serial integration -
``_detect_and_split_flat_chains``, branch anchoring/grouping in
``_reconstruct_serial`` / ``_emit_flat_chain_conversation``, timing, filters,
and env gating. Each test encodes the design spec
(``2026-06-10-weka-flattened-agent-lcp-detection-design.md``) and throws a
hostile input at the loader. Findings that contradict the spec are kept and
marked ``xfail(strict=True)`` so the suite stays green while documenting the
gap.

Traces are built inline as dicts, validated through ``WekaTrace`` schema by
calling ``convert_to_conversations({tid: [trace]})`` on a loader constructed
with ``filename=None`` and a MagicMock user-config (copied here rather than
importing the reference suite's private helpers).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.environment import Environment
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace

_MODEL = "claude-opus-4-5-20251101"
_HAIKU = "claude-haiku-4-5-20251001"


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
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = [_MODEL, _HAIKU]
    return uc


def _stub_prompt_generator_for_reconstructor(loader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
    )


def _normal(
    t: float,
    hash_ids: list[int],
    *,
    in_tokens: int | None = None,
    out: int = 10,
    api_time: float = 1.0,
    think_time: float = 0.0,
    model: str = _MODEL,
) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": len(hash_ids) * 64 if in_tokens is None else in_tokens,
        "out": out,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": api_time,
        "think_time": think_time,
    }


def _trace(trace_id: str, requests: list[dict], **header) -> dict:
    base = {
        "id": trace_id,
        "models": [_MODEL, _HAIKU],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    }
    base.update(header)
    return base


def _build_loader(uc=None) -> WekaTraceLoader:
    loader = WekaTraceLoader(filename=None, user_config=uc or _mk_user_config())
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _convert(loader: WekaTraceLoader, trace_dict: dict):
    trace = WekaTrace.model_validate(trace_dict)
    convs = loader.convert_to_conversations({trace_dict["id"]: [trace]})
    return {c.session_id: c for c in convs}


def _retained_request_count(convs: dict) -> int:
    """Total turns across every conversation = retained-request count."""
    return sum(len(c.turns) for c in convs.values())


# --------------------------------------------------------------------------
# Branch anchoring: turn-0 fallback when filters drop the main chain's
# earliest turns so a worker's first outer_idx precedes every retained main
# outer_idx (spec §5.3 "Fallback ... anchor to main turn 0").
# --------------------------------------------------------------------------


def test_flat_chain_anchoring_turn_zero_fallback_passes_orchestrator_v1():
    """A worker chain whose first request's outer_idx is below EVERY main-chain
    turn's outer_idx must anchor to main turn 0 (spec §5.3 fallback) and pass
    orchestrator-v1 validation.

    The main chain owns the trace's earliest-``t`` request even though its rows
    sit at high outer indices (2, 3). The worker (a disjoint-namespace batch)
    sits at outer indices 0, 1 but starts later in time, so no main turn
    precedes it by outer_idx -> the loader's ``default=0`` fallback fires.
    """
    requests = [
        _normal(5.0, [900, 901], api_time=0.1, model=_HAIKU),  # worker founder, outer 0
        _normal(6.0, [900, 901, 902], api_time=0.1, model=_HAIKU),  # worker t1, outer 1
        _normal(0.0, [1, 2, 3], api_time=0.1),  # main founder (earliest t), outer 2
        _normal(1.0, [1, 2, 3, 4], api_time=0.1),  # main t1, outer 3
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_fb2", requests))

    root = convs["flt_fb2"]
    # Main chain = the [1,2,3...] rows (earliest t); worker = the [900,...] rows.
    assert "flt_fb2::fa:000" in convs
    # Worker's first outer_idx (0) is below both main outer indices (2, 3) ->
    # no preceding main turn by outer_idx -> fallback anchors to main turn 0.
    flat_branch = next(b for b in root.branches if ":flatspawn:" in b.branch_id)
    assert flat_branch.branch_id in root.turns[0].branch_ids
    # convert_to_conversations ran validate_for_orchestrator_v1 internally;
    # reaching here means it passed. Conserve retained-request count.
    assert _retained_request_count(convs) == 4  # 2 main + 2 worker


# --------------------------------------------------------------------------
# Join boundary epsilon: chain end EXACTLY equal to a main turn's t joins
# within 1e-6 (spec §5.3 "first main turn with t + eps >= end(tail(chain))").
# --------------------------------------------------------------------------


def test_flat_chain_join_at_exact_equality_gates_that_main_turn():
    """Worker ends at exactly t == a main turn's t. With the +eps slack the
    join must land on that main turn (SPAWN_JOIN), not be background."""
    # Main: outer0 t=0 [1,2,3], outer1 t=10 [1,2,3,4]. Worker forks at outer0,
    # one request ending exactly at t=10.0 (t=9.0 + api_time=1.0).
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder
        _normal(9.0, [1, 2, 700], api_time=1.0, model=_HAIKU),  # worker, ends t=10.0
        _normal(10.0, [1, 2, 3, 4], api_time=0.5),  # main t1 at exactly worker end
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_join_eq", requests))
    root = convs["flt_join_eq"]
    assert len(root.turns) == 2

    branches = {b.branch_id: b for b in root.branches}
    assert len(branches) == 1
    branch = next(iter(branches.values()))
    assert branch.is_background is False, (
        "exact-equality end should JOIN, not background"
    )
    join_prereqs = [p.branch_id for p in root.turns[1].prerequisites]
    assert join_prereqs == [branch.branch_id]


def test_flat_chain_join_within_epsilon_slack_gates_main_turn():
    """Worker ends a hair (< 1e-6 s) AFTER a main turn's t. The ``+ eps`` slack
    in the join rule must still gate that turn; without it the strict ``>=``
    would push the chain to background. Probes the exact 1e-6 epsilon (spec §3
    eps = 1e-6 s, §5.3 join rule ``t + eps >= end(tail(chain))``)."""
    # Main turn 1 at t=10.0; worker ends at 10.0 + 5e-7 (inside eps).
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder
        _normal(9.0, [1, 2, 701], api_time=1.0 + 5e-7, model=_HAIKU),  # ends 10.0+5e-7
        _normal(10.0, [1, 2, 3, 4], api_time=0.5),  # main t1 at t=10.0 (< worker end)
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_join_eps", requests))
    root = convs["flt_join_eps"]
    branch = next(b for b in root.branches if ":flatspawn:" in b.branch_id)
    assert branch.is_background is False, "within-eps overshoot must still JOIN"
    assert [p.branch_id for p in root.turns[1].prerequisites] == [branch.branch_id]


def test_flat_chain_just_past_last_main_turn_is_background_no_prereq():
    """A worker whose end is just past (by more than eps) the last main turn's
    t must become a background branch (is_background=True) with no prereq
    anywhere (spec §5.3 'None -> is_background=True')."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder
        _normal(5.0, [1, 2, 3, 4], api_time=0.5),  # main t1 (last main turn t=5)
        _normal(1.0, [1, 2, 800], api_time=4.001, model=_HAIKU),  # worker ends t=5.001
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_bg", requests))
    root = convs["flt_bg"]
    flatspawn = [b for b in root.branches if "flatspawn" in b.branch_id]
    assert len(flatspawn) == 1
    assert flatspawn[0].is_background is True
    assert flatspawn[0].branch_id not in {
        p.branch_id for t in root.turns for p in t.prerequisites
    }


# --------------------------------------------------------------------------
# Grouping: two chains sharing (preceding, join) collapse into one branch with
# both child ids; a third chain in a different group gets its own branch_id
# (spec §5.3 "Grouping ... branch_id = ...:flatspawn:{first_chain_index}").
# --------------------------------------------------------------------------


def test_two_flat_chains_sharing_spawn_and_join_collapse_to_one_branch():
    """Two worker chains with identical (preceding=turn0, join=turn1) anchoring
    must share ONE ConversationBranchInfo carrying both child ids."""
    # Two disjoint-namespace workers both forking off main turn 0 and both
    # ending before main turn 1 (t=20).
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder, outer 0
        _normal(1.0, [500, 501], api_time=0.5, model=_HAIKU),  # worker A, ends 1.5
        _normal(2.0, [600, 601], api_time=0.5, model=_HAIKU),  # worker B, ends 2.5
        _normal(20.0, [1, 2, 3, 4], api_time=0.5),  # main t1
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_group", requests))
    root = convs["flt_group"]
    flatspawn = [b for b in root.branches if "flatspawn" in b.branch_id]
    assert len(flatspawn) == 1, "two same-anchored chains must collapse to one branch"
    assert set(flatspawn[0].child_conversation_ids) == {
        "flt_group::fa:000",
        "flt_group::fa:001",
    }
    # One join prereq on turn 1 referencing that single branch.
    assert [p.branch_id for p in root.turns[1].prerequisites] == [
        flatspawn[0].branch_id
    ]


def test_three_flat_chains_distinct_groups_get_unique_branch_ids():
    """A third chain that joins a different main turn must land in its own
    branch with a distinct branch_id; branch_ids stay unique per turn so
    orchestrator-v1 validation passes."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.2),  # main founder, outer 0
        _normal(0.5, [500, 501], api_time=0.2, model=_HAIKU),  # A ends 0.7 -> join t1
        _normal(0.6, [600, 601], api_time=0.2, model=_HAIKU),  # B ends 0.8 -> join t1
        _normal(10.0, [1, 2, 3, 4], api_time=0.2),  # main t1 (t=10)
        _normal(11.0, [700, 701], api_time=0.2, model=_HAIKU),  # C ends 11.2 -> join t2
        _normal(20.0, [1, 2, 3, 4, 5], api_time=0.2),  # main t2 (t=20)
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_3", requests))
    root = convs["flt_3"]
    flatspawn = [b for b in root.branches if "flatspawn" in b.branch_id]
    # A+B share (turn0, turn1); C is (turn1, turn2) -> 2 branches.
    assert len(flatspawn) == 2
    ids = [b.branch_id for b in flatspawn]
    assert len(set(ids)) == 2, "branch_ids must be unique"
    # No turn declares the same branch_id twice (validator guardrail).
    for turn in root.turns:
        assert len(turn.branch_ids) == len(set(turn.branch_ids))


# --------------------------------------------------------------------------
# Interaction: real type:"subagent" entries coexist with flat-chain split.
# Subagent anchoring must use main-chain turns only; both branch kinds may
# share one turn (spec §5.3).
# --------------------------------------------------------------------------


def test_subagent_and_flat_branch_coexist_on_same_turn():
    """A subagent marker and a detected flat chain both spawn off main turn 0;
    their branch_ids coexist and stay distinct, and validation passes."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder, outer 0
        {
            "t": 0.5,
            "type": "subagent",
            "agent_id": "agent_x",
            "subagent_type": "Explore",
            "duration_ms": 500,
            "total_tokens": 10,
            "tool_use_count": 1,
            "status": "completed",
            "requests": [_normal(0.5, [40, 41], api_time=0.2, model=_HAIKU)],
            "models": [_HAIKU],
            "tool_tokens": 0,
            "system_tokens": 0,
        },  # subagent marker, outer 1, ends ~1.0
        _normal(1.0, [777, 778], api_time=0.3, model=_HAIKU),  # flat worker, outer 2
        _normal(10.0, [1, 2, 3, 4], api_time=0.5),  # main t1, outer 3
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_coexist", requests))
    root = convs["flt_coexist"]
    spawn_ids = [b.branch_id for b in root.branches if ":spawn:" in b.branch_id]
    flat_ids = [b.branch_id for b in root.branches if ":flatspawn:" in b.branch_id]
    assert len(spawn_ids) == 1, "subagent SPAWN branch present"
    assert len(flat_ids) == 1, "flat-chain SPAWN branch present"
    # Both anchored to turn 0; branch_ids on turn 0 unique.
    assert set(spawn_ids + flat_ids).issubset(set(root.turns[0].branch_ids))
    assert len(root.turns[0].branch_ids) == len(set(root.turns[0].branch_ids))
    # Child conversations exist for both.
    assert "flt_coexist::sa:agent_x" in convs
    assert "flt_coexist::fa:000" in convs


# --------------------------------------------------------------------------
# Disjoint-batch path: with no nonce-poison guard, a trace of mutually-
# disjoint requests is an independent-agent batch and splits into per-agent
# chains rather than being skipped.
# --------------------------------------------------------------------------


def test_disjoint_batch_splits_into_independent_chains(caplog):
    """A trace of mutually-disjoint requests (zero LCP between all) splits
    into independent per-agent chains, retaining every request, and logs no
    nonce-poison WARNING (the guard was removed)."""
    # 10 mutually-disjoint single-block requests -> independent founders.
    requests = [_normal(float(i), [1000 + i], api_time=0.01) for i in range(10)]
    loader = _build_loader()
    with caplog.at_level(logging.WARNING):
        convs = _convert(loader, _trace("flt_poison", requests))
    assert len(convs) > 1, "disjoint batch must split into per-agent chains"
    assert _retained_request_count(convs) == 10
    assert not any(
        "poison" in r.message.lower() or "nonce" in r.message.lower()
        for r in caplog.records
    ), "the nonce-poison guard was removed; no such warning should be logged"


# --------------------------------------------------------------------------
# Idle-gap warp interaction (spec §5.6): flat-chain warped timestamps/delays
# and the warp gap structure must match the unsplit trace shifted equivalently.
# --------------------------------------------------------------------------


def test_idle_gap_warp_flat_chain_gap_structure_matches_unsplit_run():
    """With trace_idle_gap_cap_seconds active, the warp collects the SAME
    request-start set whether or not chains split (spec §5.6 'unchanged
    inputs'). The main chain's warped timestamps must equal those produced
    with detection disabled (legacy single stream) over the same starts.
    """
    # Main founder at t=0, worker founder at t=20 (disjoint ns), main t2 at
    # t=220. The 20->220 request-start gap (200s) caps to 60s -> 140s shift.
    requests = [
        _normal(0.0, [1, 2, 3], api_time=1.0),  # main, t=0
        _normal(20.0, [900, 901], api_time=1.0, model=_HAIKU),  # worker, t=20
        _normal(220.0, [1, 2, 3, 4], api_time=1.0),  # main t2 -> warps to t=80
    ]
    uc = _mk_user_config()
    uc.loadgen.trace_idle_gap_cap_seconds = 60.0
    loader_split = _build_loader(uc)
    convs_split = _convert(loader_split, _trace("flt_warp", requests))

    uc2 = _mk_user_config()
    uc2.loadgen.trace_idle_gap_cap_seconds = 60.0
    loader_legacy = _build_loader(uc2)
    import unittest.mock as _m

    with _m.patch.object(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False):
        convs_legacy = _convert(loader_legacy, _trace("flt_warp", requests))

    root_split = convs_split["flt_warp"]
    legacy = convs_legacy["flt_warp"]
    # Main founder unwarped (gap is after it).
    assert root_split.turns[0].timestamp == 0.0
    # Worker request start (t=20) is also <= cap boundary so it is unwarped;
    # the legacy single stream sees it at t=20 too. The 200s gap 20->220
    # collapses to 60s, shifting t=220 -> t=80 in BOTH runs.
    assert root_split.turns[1].timestamp == pytest.approx(80_000.0)
    # Legacy keeps all 3 as one conversation; its turn at t=220 also warps to
    # 80s. The main chain's warped end timestamp must agree across runs.
    legacy_220 = next(t for t in legacy.turns if t.timestamp == pytest.approx(80_000.0))
    assert legacy_220.timestamp == pytest.approx(root_split.turns[1].timestamp)
    # Worker conversation's single turn warps to t=20 (20_000 ms).
    worker = convs_split["flt_warp::fa:000"]
    assert worker.turns[0].timestamp == pytest.approx(20_000.0)


def test_idle_gap_warp_flat_branch_start_uses_mapped_time_not_raw():
    """Regression: a multi-chain flat group whose workers begin AFTER a
    compressed idle gap must anchor its SPAWN branch on the WARPED first-request
    time, matching the workers' (also-warped) turn-0 timestamps.

    Using the raw first-request time leaves the branch start on the
    uncompressed timeline while the child turns live on the compressed one, so
    branch_orchestrator._child_dispatch_offset_ms (max(0, child_ts -
    branch_start)) goes negative and clamps to 0 -- silently collapsing the
    recorded inter-worker dispatch stagger for every flat worker-group fan-out
    whenever the (default-on for agentx) idle-gap cap is engaged.
    """
    # Main founder at t=0; a 1000s idle gap; two disjoint-namespace workers at
    # t=1000 and t=1002 (2s apart); main t1 at t=1003. Sorted request starts
    # [0, 1000, 1002, 1003]: the 0->1000 gap (1000s) caps to 60s (940s excess),
    # shifting everything at/after the gap left by 940s:
    #   worker A 1000 -> 60s, worker B 1002 -> 62s, main t1 1003 -> 63s.
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder, outer 0
        _normal(1000.0, [900, 901], api_time=0.5, model=_HAIKU),  # worker A
        _normal(1002.0, [910, 911], api_time=0.5, model=_HAIKU),  # worker B
        _normal(1003.0, [1, 2, 3, 4], api_time=0.5),  # main t1
    ]
    uc = _mk_user_config()
    uc.loadgen.trace_idle_gap_cap_seconds = 60.0
    loader = _build_loader(uc)
    convs = _convert(loader, _trace("flt_warp_start", requests))
    root = convs["flt_warp_start"]

    flatspawn = [b for b in root.branches if "flatspawn" in b.branch_id]
    assert len(flatspawn) == 1, "both workers share (preceding, join) -> one branch"
    branch = flatspawn[0]
    assert len(branch.child_conversation_ids) == 2

    # Branch start is the WARPED min worker start (60s), NOT the raw 1000s.
    assert branch.start_timestamp_ms == pytest.approx(60_000.0)

    # Worker turn-0 timestamps are warped (60s, 62s); the per-worker dispatch
    # offset from the branch start stays non-negative AND preserves the recorded
    # 2s stagger instead of collapsing both to 0.
    child_ts = sorted(
        convs[sid].turns[0].timestamp for sid in branch.child_conversation_ids
    )
    assert child_ts == [pytest.approx(60_000.0), pytest.approx(62_000.0)]
    offsets = sorted(ts - branch.start_timestamp_ms for ts in child_ts)
    assert offsets[0] == pytest.approx(0.0)
    assert offsets[1] == pytest.approx(2_000.0)


# --------------------------------------------------------------------------
# Timing: per-chain delays, think_time_only and ignore_delays on flat-chain
# turns (spec §5.6). Delays never negative.
# --------------------------------------------------------------------------


def test_flat_chain_delays_are_within_chain_and_nonnegative():
    """Flat-chain turn delays are computed within the chain only (spec §5.6),
    never against cross-agent neighbours, and are never negative."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main
        _normal(3.0, [900, 901], api_time=0.5, model=_HAIKU),  # worker t0
        _normal(4.0, [1, 2, 3, 4], api_time=0.5),  # main t1 interleaved
        _normal(8.0, [900, 901, 902], api_time=0.5, model=_HAIKU),  # worker t1
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_delay", requests))
    worker = convs["flt_delay::fa:000"]
    assert worker.turns[0].delay is None  # turn 0 always None
    # Within-chain delta: 8.0 - 3.0 = 5.0s -> 5000ms (NOT 8.0 - 4.0 cross-agent).
    assert worker.turns[1].delay == pytest.approx(5000.0)
    for c in convs.values():
        for turn in c.turns:
            if turn.delay is not None:
                assert turn.delay >= 0.0


def test_flat_chain_think_time_only_uses_recorded_think_time():
    """use_think_time_only=True: flat-chain turn delay equals recorded
    think_time*1000 (ms), falling back to the full within-chain delta when
    think_time is None (spec §5.6 'honoring think_time_only')."""
    requests = [
        # Real 2-turn main thread (shared [1,2,3] prefix) so its founder is not a
        # lone block-disjoint leader -- which would now be peeled as a one-shot
        # preamble. The [900,...] rows are then detected as the worker chain.
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main t0
        _normal(3.0, [900, 901], api_time=0.5, model=_HAIKU, think_time=0.0),  # w t0
        _normal(4.0, [1, 2, 3, 4], api_time=0.5),  # main t1 (shares prefix)
        _normal(
            8.0, [900, 901, 902], api_time=0.5, model=_HAIKU, think_time=2.0
        ),  # w t1
    ]
    uc = _mk_user_config()
    uc.input.use_think_time_only = True
    loader = _build_loader(uc)
    convs = _convert(loader, _trace("flt_tt", requests))
    worker = convs["flt_tt::fa:000"]
    assert worker.turns[0].delay is None
    assert worker.turns[1].delay == pytest.approx(2000.0)  # think_time, not (8-3)*1000


def test_flat_chain_ignore_delays_nulls_timestamp_and_delay():
    """ignore_trace_delays=True must null timestamp and delay on EVERY
    conversation including detected flat-chain children."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),
        _normal(3.0, [900, 901], api_time=0.5, model=_HAIKU),
        _normal(8.0, [1, 2, 3, 4], api_time=0.5),
        _normal(9.0, [900, 901, 902], api_time=0.5, model=_HAIKU),
    ]
    uc = _mk_user_config()
    uc.input.ignore_trace_delays = True
    loader = _build_loader(uc)
    convs = _convert(loader, _trace("flt_ign", requests))
    assert any("::fa:" in sid for sid in convs)
    for c in convs.values():
        for turn in c.turns:
            assert turn.timestamp is None
            assert turn.delay is None


# --------------------------------------------------------------------------
# --max-osl caps flat-chain max_tokens but NOT subagent children (spec §5.4
# diff: "max_tokens honors --max-osl like the top-level requests these rows
# used to be"; subagent children keep their own output_length).
# --------------------------------------------------------------------------


def test_max_osl_caps_flat_chain_but_not_subagent_child():
    """--max-osl caps a detected flat chain's max_tokens (it was a top-level
    row) but leaves a real subagent child's max_tokens uncapped."""
    requests = [
        _normal(0.0, [1, 2, 3], out=100, api_time=0.5),  # main, out 100
        {
            "t": 0.5,
            "type": "subagent",
            "agent_id": "agent_sa",
            "subagent_type": "Explore",
            "duration_ms": 300,
            "total_tokens": 10,
            "tool_use_count": 1,
            "status": "completed",
            "requests": [_normal(0.5, [40, 41], out=100, api_time=0.2, model=_HAIKU)],
            "models": [_HAIKU],
            "tool_tokens": 0,
            "system_tokens": 0,
        },
        _normal(1.0, [900, 901], out=100, api_time=0.3, model=_HAIKU),  # flat worker
        _normal(10.0, [1, 2, 3, 4], out=100, api_time=0.5),  # main t1
    ]
    uc = _mk_user_config()
    uc.input.synthesis.max_osl = 25
    loader = _build_loader(uc)
    convs = _convert(loader, _trace("flt_osl", requests))
    flat = convs["flt_osl::fa:000"]
    assert all(t.max_tokens <= 25 for t in flat.turns), (
        "flat chain must honor --max-osl"
    )
    child = convs["flt_osl::sa:agent_sa"]
    assert child.turns[0].max_tokens == 100, "subagent child max_tokens NOT capped"


# --------------------------------------------------------------------------
# Effective-prefix length guard (spec §5.4): observed > declared but turn-0
# hash list SHORTER than observed -> declared used, no crash.
# --------------------------------------------------------------------------


def test_zero_declared_fanout_keeps_shared_prefix_in_user_content():
    """The system role is never fabricated: a 0/0-declared fan-out trace
    keeps its observed shared prefix INSIDE the user content. Byte sharing
    across the group is content-based, not role-based, so turn 0 of the root
    and every worker chain is a single user message carrying the request's
    full token count.
    """
    # Main group: main founder [1,2,3] + two workers forking at depth 2 on
    # [1,2,...]. Observed group prefix = LCP over first requests = [1,2] = 2,
    # but with 0/0 declared it must NOT surface as a system segment.
    requests = [
        _normal(0.0, [1, 2, 3], api_time=1.0),  # main founder
        _normal(2.0, [1, 2, 50, 51], api_time=2.0, model=_HAIKU),  # worker A
        _normal(2.5, [1, 2, 60, 61], api_time=2.0, model=_HAIKU),  # worker B
        _normal(9.0, [1, 2, 3, 4], api_time=1.0),  # main t1
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_obs", requests))
    for sid, total in (
        ("flt_obs", 192),
        ("flt_obs::fa:000", 256),
        ("flt_obs::fa:001", 256),
    ):
        msgs0 = convs[sid].turns[0].raw_messages
        assert [m["role"] for m in msgs0] == ["user"], sid
        assert msgs0[0]["content"] == f"<dec:{total}>", sid
    assert _retained_request_count(convs) == 4


def test_flat_chain_prefix_blocks_zero_yields_all_user_turn0():
    """A singleton worker namespace group has P_observed=0 (spec §5.4
    'singleton group degrades to ... all-user when 0/0'); its turn 0 must be
    all-user (no system segment)."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.3),  # main
        _normal(1.0, [800, 801], api_time=0.3, model=_HAIKU),  # singleton worker
        _normal(10.0, [1, 2, 3, 4], api_time=0.3),  # main t1
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_zero_prefix", requests))
    worker = convs["flt_zero_prefix::fa:000"]
    roles0 = [m["role"] for m in worker.turns[0].raw_messages]
    assert "system" not in roles0, "singleton-group worker turn 0 must be all-user"


# --------------------------------------------------------------------------
# Invariant: every retained request appears in exactly one conversation
# exactly once (no duplication, no loss) across a fan-out split.
# --------------------------------------------------------------------------


def test_every_retained_request_appears_exactly_once_across_conversations():
    """Conservation invariant: the total turn count across root + worker
    children equals the number of retained top-level requests, and outer
    indices are partitioned (no request emitted twice)."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main
        _normal(1.0, [1, 2, 50], api_time=0.5, model=_HAIKU),  # worker A
        _normal(2.0, [1, 2, 60], api_time=0.5, model=_HAIKU),  # worker B
        _normal(9.0, [1, 2, 3, 4], api_time=0.5),  # main t1
        _normal(8.5, [1, 2, 50, 51], api_time=0.5, model=_HAIKU),  # worker A t1
        _normal(12.0, [1, 2, 3, 4, 5], api_time=0.5),  # main t2
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_conserve", requests))
    # 6 retained requests -> 6 turns total across all conversations.
    assert _retained_request_count(convs) == 6
    # Root + 2 workers.
    assert sum(1 for sid in convs if "::fa:" in sid) == 2


# --------------------------------------------------------------------------
# Zero api_time boundary: a worker whose requests all have api_time=0 has end
# == start; join derivation uses end so a same-t main turn still joins.
# --------------------------------------------------------------------------


def test_zero_api_time_worker_join_uses_request_start_as_end():
    """A worker with api_time=0 (or None) has end == start. The first main
    turn at/after that start (within eps) must gate it (spec §3 end(r) = t +
    max(api_time or 0, 0))."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),  # main founder
        _normal(2.0, [1, 2, 900], api_time=0.0, model=_HAIKU),  # worker ends t=2.0
        _normal(2.0, [1, 2, 3, 4], api_time=0.5),  # main t1 at exactly t=2.0
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_zero_api", requests))
    root = convs["flt_zero_api"]
    flatspawn = [b for b in root.branches if "flatspawn" in b.branch_id]
    assert len(flatspawn) == 1
    assert flatspawn[0].is_background is False
    assert [p.branch_id for p in root.turns[1].prerequisites] == [
        flatspawn[0].branch_id
    ]


# --------------------------------------------------------------------------
# Env gating: split disabled restores legacy single-stream on a fan-out trace
# (spec §6). All requests stay in one conversation.
# --------------------------------------------------------------------------


def test_split_disabled_keeps_single_conversation_on_fanout():
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.5),
        _normal(1.0, [1, 2, 50], api_time=0.5, model=_HAIKU),
        _normal(2.0, [1, 2, 3, 4], api_time=0.5),
        _normal(3.0, [1, 2, 50, 51], api_time=0.5, model=_HAIKU),
    ]
    uc = _mk_user_config()
    loader = _build_loader(uc)
    import unittest.mock as _m

    with _m.patch.object(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False):
        convs = _convert(loader, _trace("flt_disabled", requests))
    assert list(convs) == ["flt_disabled"]
    assert len(convs["flt_disabled"].turns) == 4


# --------------------------------------------------------------------------
# Branch invariant: every flat-chain branch's child_conversation_ids resolve
# to emitted conversations, and SPAWN_JOIN targets are never background.
# --------------------------------------------------------------------------


def test_flat_branch_child_ids_resolve_and_join_targets_not_background():
    """Every flatspawn branch's child ids must reference real conversations;
    any branch referenced by a SPAWN_JOIN prereq must be non-background
    (orchestrator-v1 invariants)."""
    requests = [
        _normal(0.0, [1, 2, 3], api_time=0.2),  # main
        _normal(0.5, [500, 501], api_time=0.2, model=_HAIKU),  # joins t1
        _normal(0.6, [600, 601], api_time=0.2, model=_HAIKU),  # background
        _normal(5.0, [1, 2, 3, 4], api_time=0.2),  # main t1
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_inv", requests))
    root = convs["flt_inv"]
    all_ids = set(convs)
    for b in root.branches:
        for cid in b.child_conversation_ids:
            assert cid in all_ids, f"dangling child id {cid}"
    branches_by_id = {b.branch_id: b for b in root.branches}
    for turn in root.turns:
        for p in turn.prerequisites:
            assert p.kind == PrerequisiteKind.SPAWN_JOIN
            assert branches_by_id[p.branch_id].is_background is False
            assert branches_by_id[p.branch_id].mode == ConversationBranchMode.SPAWN


# --------------------------------------------------------------------------
# Preamble split: a leading request that shares NO blocks with the rest of the
# trace is a one-shot preamble and must not found the main chain. Small ones
# (Claude Code title generation) and large fully-disjoint ones (observed on 4
# real 060826 traces: a 25-31k-token disjoint giant hijacked main_index into a
# 1-turn "main" while the real session split into dozens of fa:* chains).
# --------------------------------------------------------------------------


def _req(t: float, hash_ids: list[int], out: int):
    from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest

    return WekaNormalRequest.model_validate(_normal(t, hash_ids, out=out))


def test_split_off_preamble_small_disjoint_leader_is_split():
    """A small (out<=64), block-disjoint leading request (title-gen) is set
    aside; the rest reach detection in outer-index order."""
    from aiperf.dataset.loader.weka_trace import _split_off_preamble

    normals = [
        (0, _req(0.0, [900, 901], out=20)),  # title-gen: small, disjoint
        (1, _req(1.0, [1, 2, 3], out=200)),
        (2, _req(2.0, [1, 2, 3, 4], out=200)),
    ]
    preamble, rest = _split_off_preamble(normals)
    assert [oi for oi, _ in preamble] == [0]
    assert [oi for oi, _ in rest] == [1, 2]


def test_split_off_preamble_large_disjoint_leader_is_split():
    """A LARGE (out>64) leading request whose blocks are fully disjoint from
    every other request is still a one-shot preamble and must be set aside, so
    it cannot hijack main_index. This is the 060826 'disjoint giant' case."""
    from aiperf.dataset.loader.weka_trace import _split_off_preamble

    normals = [
        (0, _req(0.0, [900, 901, 902, 903], out=500)),  # disjoint giant
        (1, _req(1.0, [1, 2, 3], out=10)),
        (2, _req(2.0, [1, 2, 3, 4], out=10)),
    ]
    preamble, rest = _split_off_preamble(normals)
    assert [oi for oi, _ in preamble] == [0]
    assert [oi for oi, _ in rest] == [1, 2]


def test_split_off_preamble_large_leader_sharing_prefix_is_kept():
    """A large leading request whose blocks ARE reused by later turns is a
    genuine conversation root, not a preamble -- it must be kept (61/65 of the
    060826 out>64 leaders are this case)."""
    from aiperf.dataset.loader.weka_trace import _split_off_preamble

    normals = [
        (0, _req(0.0, [1, 2, 3], out=500)),  # real root: prefix reused below
        (1, _req(1.0, [1, 2, 3, 4], out=10)),
        (2, _req(2.0, [1, 2, 3, 4, 5], out=10)),
    ]
    preamble, rest = _split_off_preamble(normals)
    assert preamble == []
    assert [oi for oi, _ in rest] == [0, 1, 2]


def test_split_off_preamble_large_leader_partial_overlap_is_kept():
    """A large leader that shares SOME blocks (LCP>0) with the rest is not a
    preamble; only a FULLY block-disjoint large leader is set aside."""
    from aiperf.dataset.loader.weka_trace import _split_off_preamble

    normals = [
        (0, _req(0.0, [1, 2, 900], out=500)),  # shares prefix [1,2] with below
        (1, _req(1.0, [1, 2, 3], out=10)),
        (2, _req(2.0, [1, 2, 3, 4], out=10)),
    ]
    preamble, rest = _split_off_preamble(normals)
    assert preamble == []


def test_large_disjoint_leader_does_not_hijack_main_chain():
    """End-to-end: a large disjoint leading request must be peeled, not allowed
    to found a 1-turn main while the real multi-agent session is demoted to
    worker chains. Without the fix the giant founds a 1-turn root and BOTH real
    agents become fa:* chains; with it the giant re-attaches to the true main
    (agent A) and only the genuine second agent (B) is a worker."""
    requests = [
        _normal(0.0, [900, 901, 902, 903], out=500, api_time=0.5),  # disjoint giant
        _normal(1.0, [1, 2, 3], api_time=0.5),  # agent A (real main)
        _normal(2.0, [1, 2, 3, 4], api_time=0.5),
        _normal(3.0, [1, 2, 3, 4, 5], api_time=0.5),
        _normal(4.0, [50, 51, 52], api_time=0.5, model=_HAIKU),  # agent B (worker)
        _normal(5.0, [50, 51, 52, 53], api_time=0.5, model=_HAIKU),
    ]
    loader = _build_loader()
    convs = _convert(loader, _trace("flt_big_pre", requests))
    root = convs["flt_big_pre"]
    fa_chains = [sid for sid in convs if "::fa:" in sid]
    assert len(root.turns) == 4  # giant (re-attached preamble) + agent A's 3 turns
    assert len(fa_chains) == 1  # only agent B remains a worker
