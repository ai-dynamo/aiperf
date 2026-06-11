# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the Weka trace loaders.

These tests hunt for accounting anomalies, timeline inconsistencies, and
edge-case math errors that the existing suites do not cover. Each test
probes ONE thing:

Confirmed bugs (xfail, strict):
  * ``ConversationBranchInfo.start_timestamp_ms`` is emitted in *raw* trace
    seconds while every other timestamp on the conversation is rewritten by
    the per-trace idle-gap warp, so the branch's recorded spawn time can land
    long after the parent conversation's final turn and after the child's own
    first request.
  * The parallel-path subagent payload declares an ``effective_t`` field but
    never populates it, so ``_process_task`` falls back to the raw ``t`` for
    ``start_timestamp`` even when an idle-gap warp is active.
  * ``_sa_end_seconds`` trusts ``duration_ms`` blindly: a negative duration
    yields a subagent end strictly *before* its own spawn timestamp.
  * ``_sa_end_seconds`` propagates a NaN inner ``api_time`` straight into the
    recorded end time, poisoning the join-selection comparison.
  * ``--use-think-time-only`` emits a recorded negative ``think_time`` as a
    negative ``Turn.delay`` (a delay that points into the past).

Passing characterizations (surprising-but-intended):
  * idle-gap cap uses a strict ``>`` so a gap exactly equal to the cap is left
    uncompressed.
  * ``_pack_into_streams`` treats a NaN ``api_time`` interval as non-reusable,
    forcing every later request into a fresh stream.
  * equal-``t`` inner requests pack deterministically in recorded order.
  * duplicate hash-ids within a single request inflate the theoretical
    prefix-cache hit count to a (still <= total) 100%.
  * an empty-``requests`` trace reconstructs to an empty conversation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import (
    WekaTraceLoader,
    _IdleGapTimeWarp,
    _pack_into_streams,
    _sa_end_seconds,
)
from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaSubagentEntry,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


# ---------------------------------------------------------------------------
# Shared harness (mirrors test_weka_trace.py / *_filters_adversarial.py)
# ---------------------------------------------------------------------------


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
    uc.endpoint.model_names = [
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
    ]
    return uc


def _stub_pg(loader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = (
        lambda toks: f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _make_loader(path, uc):
    loader = WekaTraceLoader(filename=str(path), user_config=uc)
    _stub_pg(loader)
    return loader


def _normal(
    t: float,
    hash_ids: list[int],
    *,
    in_tokens: int = 64,
    out_tokens: int = 10,
    api_time: float = 1.0,
    think_time: float = 0.0,
    model: str = "claude-opus-4-5-20251101",
) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": model,
        "in": in_tokens,
        "out": out_tokens,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": api_time,
        "think_time": think_time,
    }


def _subagent(
    t: float,
    agent_id: str,
    *,
    duration_ms: int | None = 1000,
    inner_hash_ids: list[int] | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    inner_hash_ids = inner_hash_ids if inner_hash_ids is not None else [8]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "duration_ms": duration_ms,
        "total_tokens": 10,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [_normal(t, inner_hash_ids, model=model)],
        "models": [model],
        "tool_tokens": 0,
        "system_tokens": 0,
    }


def _base_trace(requests: list[dict], trace_id: str = "trace") -> dict:
    return {
        "id": trace_id,
        "models": ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _make_subagent_entry(**overrides) -> WekaSubagentEntry:
    base = {
        "t": 10.0,
        "type": "subagent",
        "agent_id": "a",
        "subagent_type": "Explore",
        "duration_ms": None,
        "total_tokens": None,
        "tool_use_count": None,
        "status": "completed",
        "requests": [],
        "models": ["m"],
        "tool_tokens": 0,
        "system_tokens": 0,
    }
    base.update(overrides)
    return WekaSubagentEntry.model_validate(base)


def _inner_request(**overrides) -> WekaNormalRequest:
    base = {
        "t": 0.0,
        "type": "n",
        "model": "m",
        "in": 10,
        "out": 5,
        "hash_ids": [1],
        "api_time": 1.0,
    }
    base.update(overrides)
    return WekaNormalRequest.model_validate(base)


# ===========================================================================
# Regression: idle-gap-mapped subagent spawn time (fixed)
# ===========================================================================


def test_idle_gap_branch_start_timestamp_uses_mapped_time_not_raw(tmp_path):
    """SPAWN ``start_timestamp_ms`` must live on the same (mapped) timeline as
    every other turn timestamp on the conversation.

    With ``--trace-idle-gap-cap-seconds`` active, parent turns and child turns
    are rewritten onto the compressed timeline, but the branch's recorded
    ``start_timestamp_ms`` keeps the raw spawn seconds, so it can land far past
    the parent conversation's last turn (and past the child's own first
    request). The branch's start time should never exceed the maximum mapped
    turn timestamp in the trace.
    """
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _normal(1000.0, [1, 2]),  # 1000s start-gap -> compressed
            _subagent(1005.0, "a", inner_hash_ids=[8]),
            _normal(1006.0, [1, 2, 3]),
        ],
        trace_id="idle_branch",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config()
    uc.loadgen.trace_idle_gap_cap_seconds = 60.0
    loader = _make_loader(path, uc)

    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "idle_branch")
    child = next(c for c in convs if c.session_id == "idle_branch::sa:a")

    max_mapped_ms = max(t.timestamp for t in root.turns)
    branch = root.branches[0]
    # The branch entered the timeline when the subagent spawned; on the mapped
    # timeline that is the child's first-request timestamp, never ~940s later.
    assert branch.start_timestamp_ms <= max_mapped_ms
    assert branch.start_timestamp_ms == child.turns[0].timestamp


def test_parallel_subagent_payload_carries_mapped_spawn_time(tmp_path):
    """The parallel marker payload must carry the mapped spawn time.

    ``_WekaSubagentMarkerPayload`` declares an ``effective_t`` field and
    ``_process_task`` reads ``e.get("effective_t", e["t"])`` for
    ``start_timestamp``. But ``_parallel_subagents`` only populates
    ``effective_sa_end_seconds``, never ``effective_t`` -- so the parallel
    branch start time silently reverts to raw seconds whenever an idle-gap
    warp shifts the timeline.
    """
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _normal(1000.0, [1, 2]),
            _subagent(1005.0, "a", inner_hash_ids=[8]),
            _normal(1006.0, [1, 2, 3]),
        ],
        trace_id="idle_parallel",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config()
    uc.loadgen.trace_idle_gap_cap_seconds = 60.0
    loader = _make_loader(path, uc)

    data = loader.load_dataset()
    parent_plans, child_plans = loader._build_reconstruction_plans(data)
    timing = loader._build_trace_idle_timing_by_trace(parent_plans, child_plans)
    tasks = loader._build_parallel_reconstruction_tasks(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        model_map_per_trace={"idle_parallel": {}},
        trace_idle_timing_by_trace=timing,
    )
    _, marker = tasks[0].parent["subagents"][0]
    # The mapped end time is plumbed through; the mapped spawn time must be too.
    assert "effective_t" in marker
    assert marker["effective_t"] != marker["t"]


def test_sa_end_seconds_negative_duration_not_before_spawn():
    """A subagent's recorded end time can never precede its own spawn.

    ``_sa_end_seconds`` returns ``entry.t + duration_ms/1000`` with no floor,
    so a corrupt negative ``duration_ms`` produces an end time strictly before
    the spawn timestamp -- a subagent that finished before it started.
    """
    entry = _make_subagent_entry(t=10.0, duration_ms=-5000)
    assert _sa_end_seconds(entry) >= entry.t


def test_sa_end_seconds_nan_inner_api_time_is_finite():
    """A NaN inner ``api_time`` must not poison the subagent end time.

    When ``duration_ms`` is None the end falls back to
    ``max(inner.t + inner.api_time)``. A NaN ``api_time`` makes that NaN, which
    then flows into the ``parent.t >= sa_end`` join comparison where every
    comparison is False -- silently forcing the subagent to a background branch.
    """
    entry = _make_subagent_entry(
        t=10.0,
        duration_ms=None,
        requests=[_inner_request(t=20.0, api_time=float("nan"))],
    )
    end = _sa_end_seconds(entry)
    assert end == end  # not NaN  # noqa: PLR0124


def test_think_time_only_negative_think_time_not_negative_delay(tmp_path):
    """A recorded negative ``think_time`` must not become a negative delay.

    In ``--use-think-time-only`` mode ``Turn.delay`` is set to
    ``think_time * 1000`` directly with no lower bound, so a corrupt negative
    recorded think_time yields a negative inter-turn delay -- a request the
    load generator would be told to dispatch in the past.
    """
    trace = _base_trace(
        [
            _normal(0.0, [1], think_time=0.0),
            _normal(10.0, [1, 2], think_time=-3.0),
        ],
        trace_id="neg_tt",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    uc = _mk_user_config()
    uc.input.use_think_time_only = True
    loader = _make_loader(path, uc)

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert convs[0].turns[1].delay >= 0.0


# ===========================================================================
# PASSING CHARACTERIZATIONS (surprising but intended / not invariant-breaking)
# ===========================================================================


def test_idle_gap_exactly_equal_to_cap_is_not_compressed():
    """A request-start gap exactly equal to the cap is left untouched.

    ``_IdleGapTimeWarp`` compresses only gaps with ``gap_seconds > cap_seconds``
    (strict ``>``), so a gap of exactly the cap maps through unchanged. This is
    the inclusive-boundary mirror of the clamp's ``> cap_ms`` semantics.
    """
    warp = _IdleGapTimeWarp([0.0, 60.0], cap_seconds=60.0)
    assert warp.map(60.0) == 60.0
    # One microsecond over the cap does get compressed back to the boundary.
    warp_over = _IdleGapTimeWarp([0.0, 60.001], cap_seconds=60.0)
    assert warp_over.map(60.001) == pytest.approx(60.0)


def test_idle_gap_collapsed_tail_event_maps_to_cap_boundary():
    """A non-request event inside a collapsed gap tail maps to the boundary.

    A subagent end marker that falls in the collapsed region of a long
    request-start gap is intentionally pinned to ``raw_start + cap`` so a join
    cannot wait past the next shifted request. Probe the documented behavior
    for a gap [20, 220] with cap 60.
    """
    warp = _IdleGapTimeWarp([0.0, 20.0, 220.0], cap_seconds=60.0)
    assert warp.map(80.0) == pytest.approx(80.0)  # at the boundary
    assert warp.map(150.0) == pytest.approx(80.0)  # deep in the collapsed tail
    assert warp.map(220.0) == pytest.approx(80.0)  # the gap end
    assert warp.map(300.0) == pytest.approx(160.0)  # after: shifted left by excess


def test_pack_into_streams_nan_api_time_forces_extra_stream():
    """A NaN ``api_time`` interval is never reusable, so it forces a new stream.

    ``r_end = r.t + (r.api_time or 0.0)`` becomes NaN, and ``end <= r.t`` is
    False for every later request against a NaN end, so the second request
    cannot reuse the first stream and a redundant parallel stream is opened.
    """
    reqs = [
        _inner_request(t=0.0, api_time=float("nan"), hash_ids=[1]),
        _inner_request(t=100.0, api_time=1.0, hash_ids=[2]),
    ]
    streams = _pack_into_streams(reqs)
    assert len(streams) == 2


def test_pack_into_streams_equal_t_zero_duration_is_deterministic_order():
    """Equal-``t`` zero-duration inner requests pack in recorded order.

    ``api_time=None`` is treated as a zero-width interval that never overlaps,
    so requests sharing an instant collapse into one stream; Python's stable
    sort preserves their recorded order, keeping reconstruction reproducible.
    """
    reqs = [
        _inner_request(t=5.0, api_time=None, hash_ids=[3]),
        _inner_request(t=5.0, api_time=None, hash_ids=[1]),
        _inner_request(t=5.0, api_time=None, hash_ids=[2]),
    ]
    streams = _pack_into_streams(reqs)
    assert len(streams) == 1
    assert [r.hash_ids[0] for r in streams[0]] == [3, 1, 2]


def test_duplicate_hash_ids_in_request_inflate_theoretical_hit_to_full(tmp_path):
    """Duplicate hash-ids within one request drive theoretical hits to 100%.

    ``_count_seen_prefix_blocks`` counts every leading hash-id present in the
    seen set, with no per-request de-duplication. A request that repeats its
    own already-seen blocks reports a hit count equal to its (also-inflated)
    total -- a theoretical 100% prefix-cache hit rate for what is physically
    only two distinct blocks. The hit<=total invariant still holds.
    """
    trace = _base_trace(
        [
            _normal(0.0, [1, 2], in_tokens=128),
            _normal(1.0, [1, 2, 1, 2], in_tokens=256),
        ],
        trace_id="dup_hash",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    turn1 = convs[0].turns[1]
    assert turn1.theoretical_prefix_cache_hit_blocks == 4
    assert turn1.theoretical_prefix_cache_total_blocks == 4
    assert (
        turn1.theoretical_prefix_cache_hit_blocks
        <= turn1.theoretical_prefix_cache_total_blocks
    )


def test_empty_requests_trace_reconstructs_empty_conversation(tmp_path):
    """A trace with zero requests yields a single empty Conversation, no crash."""
    trace = _base_trace([], trace_id="empty_trace")
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs) == 1
    assert convs[0].session_id == "empty_trace"
    assert convs[0].turns == []
    assert convs[0].branches == []


def test_zero_duration_subagent_joins_first_following_turn(tmp_path):
    """A duration_ms=0 subagent (end == spawn) joins the first later parent turn.

    With ``_sa_end_seconds`` equal to the spawn time, the join condition
    ``parent.t + epsilon >= sa_end`` is satisfied by the very next parent turn,
    producing a SPAWN_JOIN prereq rather than a background branch.
    """
    trace = _base_trace(
        [
            _normal(0.0, [1]),
            _subagent(1.0, "a", duration_ms=0),
            _normal(2.0, [1, 2]),
        ],
        trace_id="zero_dur",
    )
    path = tmp_path / "t.json"
    path.write_text(json.dumps(trace))
    loader = _make_loader(path, _mk_user_config())

    convs = loader.convert_to_conversations(loader.load_dataset())
    root = next(c for c in convs if c.session_id == "zero_dur")
    assert root.branches[0].is_background is False
    assert len(root.turns[1].prerequisites) == 1
