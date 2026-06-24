# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in tool-shaped raw_messages for weka trace replay.

Default replay renders every turn's new input as a plain user-role text
segment whose token structure matches the recording -- byte-exact ISL by
construction. With ``AIPERF_DATASET_WEKA_TOOL_SHAPED_MESSAGES`` enabled, a
turn classified TOOL_RESULT instead emits the OpenAI tool-call wire shape:
the same-delta assistant segment gains a synthetic ``tool_calls`` entry and
the new-input segment becomes ``{"role": "tool", "tool_call_id": ...}`` with
identical content text. This exercises the server's tool-message template
path at the cost of exact ISL fidelity (tool messages tokenize differently
under chat templates), which is why it is opt-in and excluded from nothing:
the mmap dataset cache key must include the flag.

Shaping is per-current-turn only and guarded: it requires an assistant
segment immediately before the final user segment in the same delta, so
turn 0 (no prior assistant) and live-assistant mode (assistant segments not
emitted) fall back to the plain user shape automatically.
"""

from __future__ import annotations

import json
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aiperf.common import environment as env_mod
from aiperf.dataset.loader import weka_parallel_convert as wpc
from aiperf.dataset.loader.weka_tool_shape import tool_shape_segment_messages
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

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


def _stub_loader(loader: WekaTraceLoader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    pg = MagicMock()
    pg._cache = {}
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg._bpe_stable_terminator_tokens = []
    stub_hash_id_corpus_rng(pg)
    pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
    pg._hash_id_corpus_rng.seed = 12345
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _req(
    *,
    t: float = 0.0,
    hash_ids: list[int],
    input_types: list[str] | None = None,
    stop: str | None = None,
    model: str = _MODEL,
) -> dict:
    d = {
        "t": t,
        "type": "n",
        "model": model,
        "in": len(hash_ids) * 64,
        "out": 10,
        "hash_ids": hash_ids,
        "api_time": 1.0,
        "think_time": 0.0,
    }
    if input_types is not None:
        d["input_types"] = input_types
    if stop is not None:
        d["stop"] = stop
    return d


def _scope_trace(trace_id: str) -> dict:
    sub = {
        "t": 2.0,
        "type": "subagent",
        "agent_id": "agent_001",
        "subagent_type": "Explore",
        "duration_ms": 2000,
        "total_tokens": 100,
        "tool_use_count": 1,
        "status": "completed",
        "requests": [
            _req(
                t=0.0,
                hash_ids=[50, 51],
                input_types=["text"],
                stop="tool_use",
                model=_HAIKU,
            ),
            _req(t=1.0, hash_ids=[50, 51, 52, 53], model=_HAIKU),
        ],
        "models": [_HAIKU],
        "tool_tokens": 0,
        "system_tokens": 0,
    }
    return {
        "id": trace_id,
        "models": [_MODEL, _HAIKU],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            _req(t=0.0, hash_ids=[1, 2], input_types=["text"], stop="tool_use"),
            _req(
                t=1.0,
                hash_ids=[1, 2, 3, 4],
                input_types=["tool_result"],
                stop="tool_use",
            ),
            sub,
            _req(t=5.0, hash_ids=[1, 2, 3, 4, 5, 6], stop="end_turn"),
            _req(t=6.0, hash_ids=[1, 2, 3, 4, 5, 6, 7, 8]),
        ],
    }


def _make_loader(tmp_path: Path, trace: dict) -> WekaTraceLoader:
    p = tmp_path / f"{trace['id']}.json"
    p.write_text(json.dumps(trace))
    loader = WekaTraceLoader(filename=str(p), user_config=_mk_user_config())
    _stub_loader(loader)
    return loader


@pytest.fixture
def tool_shaped_env(monkeypatch):
    monkeypatch.setattr(
        env_mod.Environment.DATASET,
        "WEKA_TOOL_SHAPED_MESSAGES",
        True,
        raising=False,
    )


# ---------------------------------------------------------------------------
# tool_shape_segment_messages helper
# ---------------------------------------------------------------------------


def _seg(role: str, tool_result_turn: int | None = None):
    return SimpleNamespace(role=role, tool_result_turn=tool_result_turn)


def test_tool_shape_segment_messages_shapes_marked_pair():
    msgs = [
        {"role": "assistant", "content": "calling a tool"},
        {"role": "user", "content": "tool output"},
    ]
    segs = [_seg("assistant"), _seg("user", tool_result_turn=3)]
    asst, tool = tool_shape_segment_messages(msgs, segs)
    call_id = asst["tool_calls"][0]["id"]
    assert call_id == "call_turn_3"
    assert asst["tool_calls"][0]["type"] == "function"
    assert tool == {"role": "tool", "tool_call_id": call_id, "content": "tool output"}


def test_tool_shape_segment_messages_noop_for_unmarked_segments():
    msgs = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    segs = [_seg("assistant"), _seg("user")]
    assert tool_shape_segment_messages(msgs, segs) == msgs


def test_tool_shape_segment_messages_noop_without_preceding_assistant():
    turn0_msgs = [{"role": "user", "content": "first prompt"}]
    assert tool_shape_segment_messages(turn0_msgs, [_seg("user", 0)]) == turn0_msgs
    sys_user_msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    segs = [_seg("system"), _seg("user", 1)]
    assert tool_shape_segment_messages(sys_user_msgs, segs) == sys_user_msgs


def test_tool_shape_segment_messages_shapes_every_marked_pair_in_window():
    """A reset re-emit window carries the whole history: every marked pair
    must shape, each with its own recorded turn's call id."""
    msgs = [
        {"role": "user", "content": "t0"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "tool out 1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "real input"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "tool out 3"},
    ]
    segs = [
        _seg("user"),
        _seg("assistant"),
        _seg("user", 1),
        _seg("assistant"),
        _seg("user"),
        _seg("assistant"),
        _seg("user", 3),
    ]
    shaped = tool_shape_segment_messages(msgs, segs)
    assert [m["role"] for m in shaped] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert shaped[2]["tool_call_id"] == "call_turn_1"
    assert shaped[6]["tool_call_id"] == "call_turn_3"
    assert shaped[1]["tool_calls"][0]["id"] == "call_turn_1"
    assert shaped[5]["tool_calls"][0]["id"] == "call_turn_3"
    assert "tool_calls" not in shaped[3]


def test_tool_shape_segment_messages_does_not_mutate_input():
    msgs = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    segs = [_seg("assistant"), _seg("user", 1)]
    tool_shape_segment_messages(msgs, segs)
    assert msgs[0] == {"role": "assistant", "content": "a"}
    assert msgs[1] == {"role": "user", "content": "b"}


# ---------------------------------------------------------------------------
# Loader, serial path
# ---------------------------------------------------------------------------


def _roles(turn) -> list[str]:
    return [m["role"] for m in turn.raw_messages]


def test_serial_tool_shaping_disabled_by_default(tmp_path):
    loader = _make_loader(tmp_path, _scope_trace("trace_off"))
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert all(m["role"] != "tool" for m in t.raw_messages)
            assert all("tool_calls" not in m for m in t.raw_messages)


def test_serial_tool_shaping_emits_tool_messages(tool_shaped_env, tmp_path):
    loader = _make_loader(tmp_path, _scope_trace("trace_on"))
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    main = convs["trace_on"].turns
    # turn 0: user_input, no shaping
    assert _roles(main[0]) == ["user"]
    # turn 1 (tool_result): [assistant+tool_calls, tool]
    assert _roles(main[1]) == ["assistant", "tool"]
    asst, tool = main[1].raw_messages
    assert asst["tool_calls"][0]["id"] == "call_turn_1"
    assert tool["tool_call_id"] == "call_turn_1"
    # turn 2 (tool_result via prev stop fallback): shaped with its own id
    assert _roles(main[2]) == ["assistant", "tool"]
    assert main[2].raw_messages[1]["tool_call_id"] == "call_turn_2"
    # turn 3 (user_input via prev end_turn): unshaped
    assert _roles(main[3]) == ["assistant", "user"]

    child = convs["trace_on::sa:agent_001"].turns
    assert _roles(child[0]) == ["user"]
    assert _roles(child[1]) == ["assistant", "tool"]


def test_serial_tool_shaping_preserves_content_text(tool_shaped_env, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    baseline = _make_loader(tmp_path / "a", _scope_trace("trace_same"))
    base_convs = {
        c.session_id: c
        for c in baseline.convert_to_conversations(baseline.load_dataset())
    }
    # baseline loaded with env ON -- compare against a stripped-signal copy,
    # which cannot shape (input_kind None) and so keeps plain user roles.
    legacy = _scope_trace("trace_same")
    for r in legacy["requests"]:
        for rr in [r] + (r.get("requests") or []):
            rr.pop("input_types", None)
            rr.pop("stop", None)
    plain_loader = _make_loader(tmp_path / "b", legacy)
    plain_convs = {
        c.session_id: c
        for c in plain_loader.convert_to_conversations(plain_loader.load_dataset())
    }
    for sid, conv in base_convs.items():
        for shaped_turn, plain_turn in zip(
            conv.turns, plain_convs[sid].turns, strict=True
        ):
            shaped_contents = [m["content"] for m in shaped_turn.raw_messages]
            plain_contents = [m["content"] for m in plain_turn.raw_messages]
            assert shaped_contents == plain_contents


def test_serial_legacy_trace_never_shaped(tool_shaped_env, tmp_path):
    legacy = _scope_trace("trace_legacy")
    for r in legacy["requests"]:
        for rr in [r] + (r.get("requests") or []):
            rr.pop("input_types", None)
            rr.pop("stop", None)
    loader = _make_loader(tmp_path, legacy)
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert all(m["role"] != "tool" for m in t.raw_messages)


# ---------------------------------------------------------------------------
# Parallel parity
# ---------------------------------------------------------------------------


def _run_pool_inproc(tasks, *, corpus, base_seed, block_size, **_kwargs):
    corpus_arr = np.array(corpus, dtype=np.int32)
    shm = shared_memory.SharedMemory(create=True, size=corpus_arr.nbytes)
    np.ndarray(corpus_arr.shape, dtype=np.int32, buffer=shm.buf)[:] = corpus_arr
    saved = wpc._worker_state
    try:
        wpc._init_worker(
            wpc._WekaWorkerInitArgs(
                shm_name=shm.name,
                corpus_len=len(corpus_arr),
                tokenizer_name="test-tok",
                base_seed=base_seed,
                block_size=block_size,
                bpe_stable_terminator_tokens=[],
            )
        )
        return [wpc._process_task(t) for t in tasks]
    finally:
        wpc._worker_state = saved
        shm.close()
        shm.unlink()


def test_parallel_tool_shaping_matches_serial(tool_shaped_env, tmp_path):
    loader = _make_loader(tmp_path, _scope_trace("trace_par"))
    data = loader.load_dataset()
    plans = loader._build_reconstruction_plans(data)
    parent_plans, child_plans = plans.parent_plans, plans.child_plans
    model_maps = {tid: loader._build_model_map(w[0]) for tid, w in data.items()}
    common = dict(
        parent_plans=parent_plans,
        child_plans=child_plans,
        data=data,
        ignore_delays=False,
        think_time_only=False,
        cap_seconds=None,
        t_start=0.0,
        model_map_per_trace=model_maps,
        trace_idle_timing_by_trace={},
        metric_values_by_trace=loader._build_shared_metric_values(
            parent_plans, child_plans, plans.flat_plans
        ),
        flat_plans=plans.flat_plans,
    )
    serial = loader._reconstruct_serial(dropped_per_trace={}, **common)
    with (
        patch(
            "aiperf.dataset.loader.weka_parallel_convert.run_parallel_weka_reconstruction",
            side_effect=lambda tasks, **kw: _run_pool_inproc(tasks, **kw),
        ),
        patch(
            "aiperf.dataset.loader.weka_parallel_convert.Tokenizer.from_pretrained",
            return_value=loader.prompt_generator.tokenizer,
        ),
    ):
        parallel = loader._reconstruct_parallel(configured_workers=1, **common)

    s = {c.session_id: c for c in serial}
    p = {c.session_id: c for c in parallel}
    assert set(s) == set(p)
    shaped_seen = False
    for sid in s:
        for st, pt in zip(s[sid].turns, p[sid].turns, strict=True):
            assert st.raw_messages == pt.raw_messages, f"{sid}: shaping drift"
            if any(m["role"] == "tool" for m in st.raw_messages):
                shaped_seen = True
    assert shaped_seen, "expected at least one tool-shaped turn in this trace"


# ---------------------------------------------------------------------------
# Shaping must survive reset_context re-emits: a reset REPLACES the wire
# context, so if the full re-emission renders previously-shaped tool turns as
# plain user, the reset retroactively unshapes everything already sent.
# ---------------------------------------------------------------------------


def _reset_trace(trace_id: str) -> dict:
    """Turn 1 is a shaped tool-result. Turn 3's hash chain diverges INSIDE
    turn-2's already-emitted region (LCP cut past turn-1's pair -> the pair
    SURVIVES truncation and must re-emit shaped in the reset full state)."""
    return {
        "id": trace_id,
        "models": [_MODEL, _HAIKU],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            _req(t=0.0, hash_ids=[1, 2], input_types=["text"], stop="tool_use"),
            _req(
                t=1.0,
                hash_ids=[1, 2, 3, 4],
                input_types=["tool_result"],
                stop="tool_use",
            ),
            _req(
                t=2.0,
                hash_ids=[1, 2, 3, 4, 5, 6],
                input_types=["text"],
                stop="tool_use",
            ),
            # diverges at block index 5 (inside turn-2's emitted content,
            # after turn-1's marked pair at blocks 2-3) -> reset re-emit.
            _req(t=3.0, hash_ids=[1, 2, 3, 4, 5, 9, 10], input_types=["tool_result"]),
        ],
    }


def _mk_recon(tool_shaped: bool = True):
    from aiperf.dataset.loader.weka_synth_buf import ConversationReconstructor

    bs = 16

    def decode_block_tokens(hash_ids):
        out = []
        for h in hash_ids:
            out.extend(range(h * 1000, h * 1000 + bs))
        return out

    def sample_partial_tail_tokens(n, seed):
        return list(range(900_000, 900_000 + n))

    return ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=lambda toks: f"<dec:{len(toks)}>",
        tool_shaped_messages=tool_shaped,
    )


def test_unpaired_tool_turn_stays_plain_across_reset_reemit():
    """A tool-result turn whose pairing assistant was NOT in its first
    emission window is sent plain. A later reset re-emission must keep it
    plain: re-sending it as ``role: tool`` (and retroactively injecting
    ``tool_calls`` into the already-sent assistant) changes the wire shape
    of previously-sent context across the reset."""
    bs = 16
    r = _mk_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=2 * bs, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.turn_delta()
    # Turn 1: new region exactly covers prev_out -> appends assistant only.
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * bs,
        prev_out_tokens=bs,
        curr_hash_ids=[1, 2, 3],
        curr_in_tokens=3 * bs,
        seed="s1",
    )
    r.turn_delta()
    # Turn 2: tool result after a zero-output response -> the whole 2-block
    # new region becomes a marked user segment appended ALONE, directly
    # after turn 1's assistant segment.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3 * bs,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3, 4, 5],
        curr_in_tokens=5 * bs,
        seed="s2",
        is_tool_result=True,
    )
    d2 = r.turn_delta()
    # First emission: assistant not in the window -> plain user on the wire.
    assert [m["role"] for m in d2.delta_messages] == ["user"]
    # Turn 3: diverge inside the marked segment -> it survives the cut with
    # one of its two blocks, and the disturbance forces a reset re-emit.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4, 5],
        prev_in_tokens=5 * bs,
        prev_out_tokens=8,
        curr_hash_ids=[1, 2, 3, 4, 99, 100],
        curr_in_tokens=6 * bs,
        seed="s3",
    )
    d3 = r.turn_delta()
    assert d3.reset_context is True
    # Stability across the reset: the turn was SENT plain, so the re-emitted
    # history must stay plain — no tool role, no retroactive tool_calls.
    assert all(m["role"] != "tool" for m in d3.delta_messages), d3.delta_messages
    assert all("tool_calls" not in m for m in d3.delta_messages), d3.delta_messages


def test_paired_tool_turn_first_emitted_in_reset_window_stays_shaped():
    """The mirror case: a tool-result turn whose FIRST emission is a reset
    full window pairs against the surviving assistant directly before it,
    ships shaped, and must re-emit shaped (same call id) on later resets."""
    bs = 16
    r = _mk_recon()
    r.init_turn_0(
        hash_ids=[1, 2], in_tokens=2 * bs, tool_tokens=0, system_tokens=0, seed="s0"
    )
    r.turn_delta()
    # Turn 1: appends [assistant, user] (prev_out=1 block, +1 user block).
    r.advance_turn(
        prev_hash_ids=[1, 2],
        prev_in_tokens=2 * bs,
        prev_out_tokens=bs,
        curr_hash_ids=[1, 2, 3, 4],
        curr_in_tokens=4 * bs,
        seed="s1",
    )
    r.turn_delta()
    # Turn 2: tool result whose hash chain replaces turn 1's user block
    # (diverges at block 3) -> truncation disturbs emitted context, so the
    # FIRST emission of the marked segment is the reset full window, where
    # it pairs with turn 1's surviving assistant.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 4],
        prev_in_tokens=4 * bs,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3, 50, 51],
        curr_in_tokens=5 * bs,
        seed="s2",
        is_tool_result=True,
    )
    d2 = r.turn_delta()
    assert d2.reset_context is True
    roles2 = [m["role"] for m in d2.delta_messages]
    assert "tool" in roles2, roles2
    call_id = next(m for m in d2.delta_messages if m["role"] == "tool")["tool_call_id"]
    # Turn 3: diverge inside the marked segment (it keeps 1 of 2 blocks, the
    # pair survives) -> reset re-emit must reproduce the identical shape and
    # call id.
    r.advance_turn(
        prev_hash_ids=[1, 2, 3, 50, 51],
        prev_in_tokens=5 * bs,
        prev_out_tokens=0,
        curr_hash_ids=[1, 2, 3, 50, 61, 62],
        curr_in_tokens=6 * bs,
        seed="s3",
    )
    d3 = r.turn_delta()
    assert d3.reset_context is True
    tools3 = [m for m in d3.delta_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools3] == [call_id]


def test_reset_reemit_preserves_tool_shape(tool_shaped_env, tmp_path):
    loader = _make_loader(tmp_path, _reset_trace("trace_reset"))
    convs = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    turns = convs["trace_reset"].turns
    # turn 1: shaped append delta
    assert [m["role"] for m in turns[1].raw_messages] == ["assistant", "tool"]
    t1_id = turns[1].raw_messages[1]["tool_call_id"]
    # turn 2 (user_input): plain append, no shaping
    assert all(m["role"] != "tool" for m in turns[2].raw_messages)
    # turn 3: reset full re-emit -- turn-1's surviving tool pair must STILL be
    # shaped, with the SAME deterministic call id it was first sent with.
    assert turns[3].reset_context is True
    roles = [m["role"] for m in turns[3].raw_messages]
    assert "tool" in roles, f"reset re-emit unshaped the history: {roles}"
    reemit_tools = [m for m in turns[3].raw_messages if m["role"] == "tool"]
    assert any(m["tool_call_id"] == t1_id for m in reemit_tools)
    reemit_calls = [
        c["id"] for m in turns[3].raw_messages for c in m.get("tool_calls") or []
    ]
    assert t1_id in reemit_calls
