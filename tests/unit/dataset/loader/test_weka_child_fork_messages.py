# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child conversation's turn 0 follows its parent's message layout for the shared prefix."""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

import aiperf.common.environment as env_mod
from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
from aiperf.common.models import Conversation
from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    LayoutSegment,
    ParentPrefix,
    compute_asst_block_caps,
)
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from tests.unit.dataset.loader.conftest import make_weka_run
from tests.unit.dataset.loader.test_weka_flat_split_parallel_adv import _run_both

_BS = 4
_TRACE_BS = 64


def _decode_blocks(hash_ids: list[int]) -> list[int]:
    return [h * 100 + i for h in hash_ids for i in range(_BS)]


def _decode_text(tokens: list[int]) -> str:
    return " ".join(str(t) for t in tokens)


def _recon(tool_shaped: bool = False) -> ConversationReconstructor:
    return ConversationReconstructor(
        block_size=_BS,
        decode_block_tokens=_decode_blocks,
        sample_partial_tail_tokens=lambda n, seed: [7] * n,
        decode_tokens_to_text=_decode_text,
        tool_shaped_messages=tool_shaped,
    )


def _render(messages: list[dict[str, Any]]) -> str:
    """Stand-in chat template: role delimiters around every message."""
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _tokens(messages: list[dict[str, Any]]) -> list[str]:
    """Content token ids (the test decoder writes one number per token)."""
    return [tok for m in messages for tok in (m.get("content") or "").split()]


def _parent(
    t0: tuple[list[int], int] = ([1, 2], 2 * _BS),
    t1: tuple[list[int], int] = ([1, 2, 3, 4, 5], 5 * _BS),
    *,
    out: int = _BS,
    tool_shaped: bool = False,
    system_tokens: int = 0,
) -> ConversationReconstructor:
    """A parent after two turns; by default user[1,2] + assistant[3] + user[4,5]."""
    parent = _recon(tool_shaped)
    parent.init_turn_0(
        hash_ids=t0[0],
        in_tokens=t0[1],
        tool_tokens=0,
        system_tokens=system_tokens,
        seed="p0",
    )
    parent.turn_delta()
    parent.advance_turn(
        prev_hash_ids=t0[0],
        prev_in_tokens=t0[1],
        prev_out_tokens=out,
        curr_hash_ids=t1[0],
        curr_in_tokens=t1[1],
        seed="p1",
        is_tool_result=tool_shaped,
    )
    parent.turn_delta()
    parent.t1_hash_ids = t1[0]
    return parent


_GAP = dict(t0=([1, 2, 3], 2 * _BS + 2), t1=([1, 2, 3, 4, 5, 6], 6 * _BS))
"""Turn 1 shares turn 0's partial last block: turn 0's segment keeps its tail
and the next segment starts one block later."""
_TOOL_PAIR = dict(t1=([1, 2, 3, 4, 5, 6], 6 * _BS), out=2 * _BS, tool_shaped=True)
"""user[1,2] + assistant[3,4] + tool result[5,6]."""


def _fork(
    parent: ConversationReconstructor,
    hash_ids: list[int],
    in_tokens: int,
    later: list[tuple[list[int], int]] = (),
) -> ConversationReconstructor:
    pullbacks: set[int] = set()
    compute_asst_block_caps([(hash_ids, in_tokens), *later], _BS, pullbacks)
    child = _recon(parent.tool_shaped_messages)
    child.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="c0",
        parent_prefix=parent.parent_prefix(parent.t1_hash_ids),
        turn_0_pullbacks=pullbacks,
    )
    return child


def _flat(hash_ids: list[int], in_tokens: int, tool_shaped: bool = False):
    flat = _recon(tool_shaped)
    flat.init_turn_0(
        hash_ids=hash_ids,
        in_tokens=in_tokens,
        tool_tokens=0,
        system_tokens=0,
        seed="c0",
    )
    return flat


def _advance(child: ConversationReconstructor, prev, curr, is_tool_result=False):
    child.advance_turn(
        prev_hash_ids=prev[0],
        prev_in_tokens=prev[1],
        prev_out_tokens=_BS,
        curr_hash_ids=curr[0],
        curr_in_tokens=curr[1],
        seed="c1",
        is_tool_result=is_tool_result,
    )


def test_init_turn_0_parent_prefix_keeps_parent_messages_then_one_user():
    parent = _parent()
    child = _fork(parent, [1, 2, 3, 4, 5, 90, 91], 7 * _BS + 2)
    delta = child.turn_delta()

    parent_messages = parent.snapshot_messages()
    assert [m["role"] for m in parent_messages] == ["user", "assistant", "user"]
    assert delta.delta_messages[:3] == parent_messages
    assert delta.delta_messages[3] == {
        "role": "user",
        "content": _decode_text(_decode_blocks([90, 91]) + [7, 7]),
    }
    assert delta.reset_context is False
    assert sum(len(s.tokens) for s in child._segments) == 7 * _BS + 2
    assert _render(delta.delta_messages).startswith(_render(parent_messages))


def test_init_turn_0_parent_prefix_keeps_layout_after_a_shared_partial_block():
    """The parent's first message holds 2 blocks and a tail, and the next
    starts at block 3; the child fills block 3 with its own hash content."""
    parent = _parent(**_GAP)
    assert [s.block_start for s in parent._segments] == [0, 3, 4]
    child = _fork(parent, [1, 2, 3, 4, 5, 6, 90], 7 * _BS)

    assert [(s.role, s.block_start, s.block_count) for s in child._segments] == [
        ("user", 0, 3),
        ("assistant", 3, 1),
        ("user", 4, 2),
        ("user", 6, 1),
    ]
    assert child._segments[0].tokens == _decode_blocks([1, 2, 3])
    assert child.turn_delta().delta_messages[1:3] == parent.snapshot_messages()[1:]


def test_init_turn_0_parent_prefix_drops_synthesized_missing_blocks():
    """A truncated parent hash list has synthesized blocks with no hash behind
    them; the child keeps only hashed content, so its own blocks stay in place."""
    parent = _parent(t1=([1, 2, 3, 4], 6 * _BS))
    child = _fork(parent, [1, 2, 3, 4, 90], 8 * _BS)

    segs = child._segments
    assert [(s.role, s.block_start, s.block_count) for s in segs] == [
        ("user", 0, 2),
        ("assistant", 2, 1),
        ("user", 3, 1),
        ("user", 4, 1),
    ]
    assert segs[2].tokens == _decode_blocks([4])
    assert segs[3].tokens[:_BS] == _decode_blocks([90])
    assert sum(len(s.tokens) for s in segs) == 8 * _BS


def test_init_turn_0_parent_prefix_cut_inside_later_message_keeps_roles_to_cut():
    child = _fork(_parent(), [1, 2, 3, 4, 80], 5 * _BS)
    segs = child._segments
    assert [(s.role, s.block_start, s.block_count) for s in segs] == [
        ("user", 0, 2),
        ("assistant", 2, 1),
        ("user", 3, 1),
        ("user", 4, 1),
    ]
    assert segs[2].tokens == _decode_blocks([4])


@pytest.mark.parametrize(
    "parent_kwargs, hash_ids",
    [
        param({}, [1, 2, 60], id="shared-prefix-inside-first-message"),
        param({}, [1, 2, 3], id="nothing-new-after-assistant"),
        param({}, [70, 71], id="no-shared-prefix"),
        param(_TOOL_PAIR, [1, 2, 3, 4, 90], id="tool-pair-cut-at-assistant-end"),
        param(_TOOL_PAIR, [1, 2, 3, 91, 92], id="tool-pair-cut-inside-assistant"),
    ],
)  # fmt: skip
def test_init_turn_0_parent_prefix_falls_back_to_flat_split(parent_kwargs, hash_ids):
    parent = _parent(**parent_kwargs)
    in_tokens = len(hash_ids) * _BS
    forked = _fork(parent, hash_ids, in_tokens)
    flat = _flat(hash_ids, in_tokens, parent.tool_shaped_messages)
    assert forked.turn_delta() == flat.turn_delta()
    assert forked._turn_index == 0


@pytest.mark.parametrize(
    "parent_kwargs, hash_ids, in_tokens",
    [
        param({}, [1, 2, 3, 4, 5, 90], 6 * _BS + 1, id="whole-blocks"),
        param(_GAP, [1, 2, 3, 4, 5, 6, 90], 6 * _BS + 3, id="parent-partial-block"),
        param(dict(t1=([1, 2, 3, 4], 6 * _BS)), [1, 2, 3, 4, 90], 8 * _BS, id="parent-missing-blocks"),
        param({}, [1, 2, 3, 4, 5], 9 * _BS, id="child-missing-blocks"),
    ],
)  # fmt: skip
def test_init_turn_0_parent_prefix_keeps_own_tokens(
    parent_kwargs, hash_ids: list[int], in_tokens: int
):
    """Only roles and message starts come from the parent: the tokens are the
    flat split's, in order, and sum to ``in_tokens``."""
    child = _fork(_parent(**parent_kwargs), hash_ids, in_tokens)
    messages = child.turn_delta().delta_messages
    assert len(messages) > 1
    assert _tokens(messages) == _tokens(_flat(hash_ids, in_tokens).snapshot_messages())
    assert len(_tokens(messages)) == in_tokens


def test_parent_prefix_is_a_layout_copy():
    """The captured layout has no tokens and survives the parent's later cut."""
    parent = _parent()
    layout = parent.parent_prefix(parent.t1_hash_ids)
    expected = (
        LayoutSegment("user", 0),
        LayoutSegment("assistant", 2),
        LayoutSegment("user", 3),
    )
    assert layout.segments == expected
    _advance(parent, (parent.t1_hash_ids, 5 * _BS), ([1, 2, 3], 3 * _BS))
    assert layout.segments == expected


def test_init_turn_0_parent_prefix_with_empty_parent_falls_back():
    empty = ParentPrefix(hash_ids=(), segments=(), next_tool_turn=1)
    assert empty.spans() == []
    child = _recon()
    child.init_turn_0(
        hash_ids=[1, 2],
        in_tokens=2 * _BS,
        tool_tokens=0,
        system_tokens=0,
        seed="c0",
        parent_prefix=empty,
    )
    assert child.turn_delta() == _flat([1, 2], 2 * _BS).turn_delta()


def test_forked_child_later_turn_appends_delta_with_fresh_tool_call_ids():
    parent = _parent(tool_shaped=True)
    first = ([1, 2, 3, 4, 5, 90], 6 * _BS)
    child = _fork(parent, *first)
    turn_0 = child.turn_delta().delta_messages
    _advance(child, first, ([1, 2, 3, 4, 5, 90, 91, 92], 8 * _BS), True)
    turn_1 = child.turn_delta()

    assert [m["role"] for m in turn_0] == ["user", "assistant", "tool", "user"]
    assert turn_0[1]["tool_calls"][0]["id"] == "call_turn_1"
    assert turn_0[2]["tool_call_id"] == "call_turn_1"
    assert [m["role"] for m in turn_1.delta_messages] == ["assistant", "tool"]
    assert turn_1.delta_messages[1]["tool_call_id"] == "call_turn_3"
    assert turn_1.reset_context is False
    assert child._turn_index == 1


@pytest.mark.parametrize(
    "parent_kwargs, first, later_hash_ids, forks",
    [
        param({}, [1, 2, 3, 4, 5, 90], [1, 2, 3], False, id="cut-to-end-of-assistant"),
        param({}, [1, 2, 3, 4, 5, 90], [1, 2, 3, 4], True, id="cut-inside-user"),
        param(_GAP, [1, 2, 3, 4, 5, 6, 90], [1, 2, 3], True, id="gap-cut-into-gap-segment"),
        param(_GAP, [1, 2, 3, 4, 5, 6, 90], [1, 2, 3, 4, 5, 6], True, id="gap-cut-to-end"),
    ],
)  # fmt: skip
def test_forked_child_later_pullback_never_ends_on_assistant(
    parent_kwargs, first: list[int], later_hash_ids: list[int], forks: bool
):
    """A later block-aligned pull-back must cut at the positions the inherited
    segments occupy and must not leave an inherited assistant last."""
    parent = _parent(**parent_kwargs)
    first_turn = (first, len(first) * _BS)
    later = (later_hash_ids, len(later_hash_ids) * _BS)
    child = _fork(parent, *first_turn, later=[later])
    assert (len(child.turn_delta().delta_messages) > 1) is forks
    _advance(child, first_turn, later)

    messages = child.snapshot_messages()
    assert messages[-1]["role"] == "user"
    assert child._trailing_non_user_turns == []
    if forks:
        expected = parent.snapshot_messages()[: len(messages)]
        assert [m["role"] for m in messages] == [m["role"] for m in expected]


def test_init_turn_0_parent_prefix_keeps_declared_system_message():
    parent = _parent(system_tokens=_BS)
    messages = _fork(parent, [1, 2, 3, 4, 5, 90], 6 * _BS).turn_delta().delta_messages
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert messages[:4] == parent.snapshot_messages()


# Loader level: flat chains, subagents, nested subagent chains.


def _req(
    t: float,
    hash_ids: list[int],
    *,
    api: float = 1.0,
    short: int = 0,
    tool_result: bool = False,
) -> dict:
    req = {
        "t": t,
        "type": "n",
        "model": "m",
        "in": len(hash_ids) * _TRACE_BS - short,
        "out": 40,
        "hash_ids": hash_ids,
        "api_time": api,
    }
    if tool_result:
        req["input_types"] = ["tool_result"]
    return req


def _blocks(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def _trace(trace_id: str, requests: list[dict]) -> dict:
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": _TRACE_BS,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
    }


def _subagent(t: float, requests: list[dict]) -> dict:
    return {
        "t": t,
        "type": "subagent",
        "agent_id": "a1",
        "subagent_type": "X",
        "duration_ms": 6000,
        "total_tokens": 0,
        "tool_use_count": 0,
        "status": "completed",
        "requests": requests,
        "models": ["m"],
    }


def _flat_fork_trace() -> dict:
    """Main R0..R3; child C0 extends R1's prefix while R1 runs."""
    return _trace(
        "fork",
        [
            _req(0, _blocks(10)),
            _req(2, _blocks(20), api=5),
            _req(3, _blocks(20) + _blocks(10, 100)),
            _req(8, _blocks(30)),
            _req(10, _blocks(20) + _blocks(20, 100)),
            _req(12, _blocks(40)),
        ],
    )


def _subagent_fork_trace() -> dict:
    """Subagent forks the root's turn 1; an inner chain forks the subagent's turn 1."""
    sa_base = _blocks(20) + _blocks(10, 200)
    return _trace(
        "safork",
        [
            _req(0, _blocks(10)),
            _req(1, _blocks(20)),
            _subagent(
                2.5,
                [
                    _req(3, sa_base),
                    _req(5, sa_base + _blocks(10, 210)),
                    _req(5.5, sa_base + _blocks(10, 210) + _blocks(10, 300)),
                    _req(7, sa_base + _blocks(10, 210) + _blocks(20, 300)),
                ],
            ),
            _req(10, _blocks(30)),
        ],
    )


def _chained_fork_trace() -> dict:
    """Partial last blocks everywhere, tool-result turns, and a flat chain
    (fa:001) that forks from another flat chain (fa:000) while it runs."""
    a0 = _blocks(20) + _blocks(10, 100)
    b0 = a0 + _blocks(10, 200)
    return _trace(
        "chain",
        [
            _req(0, _blocks(10), short=20),
            _req(2, _blocks(20), api=5, short=10, tool_result=True),
            _req(3, a0, api=4, short=5),
            _req(4, b0, short=7),
            _req(8, a0 + _blocks(10, 110), short=3, tool_result=True),
            _req(9, _blocks(30), short=9, tool_result=True),
            _req(9.5, b0 + _blocks(10, 210), short=11),
        ],
    )


def _convert(trace_dict: dict) -> dict[str, Conversation]:
    loader = WekaTraceLoader(filename=None, run=make_weka_run())
    pg = MagicMock()
    pg._cache = {}
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg._hash_id_corpus_rng = HashIdRandomGenerator(12345, _internal=True)
    pg.tokenizer.decode.side_effect = _decode_text
    loader.prompt_generator = pg
    trace = WekaTrace.model_validate(trace_dict)
    convs = loader.convert_to_conversations({trace.id: [trace]})
    return {c.session_id: c for c in convs}


def _messages_at(conv: Conversation, k: int) -> list[dict[str, Any]]:
    """The full message list sent at turn ``k`` (deltas accumulated)."""
    messages: list[dict[str, Any]] = []
    for turn in conv.turns[: k + 1]:
        if turn.reset_context:
            messages = []
        messages.extend(turn.raw_messages)
    return messages


def _set_fork_messages(monkeypatch, enabled: bool, tool_shaped: bool = False) -> None:
    dataset = env_mod.Environment.DATASET
    monkeypatch.setattr(dataset, "WEKA_FORK_PARENT_MESSAGES", enabled)
    monkeypatch.setattr(dataset, "WEKA_TOOL_SHAPED_MESSAGES", tool_shaped)


def _turn_0(conv: Conversation) -> list[dict[str, Any]]:
    return _messages_at(conv, 0)


@pytest.fixture
def flat_children(monkeypatch):
    _set_fork_messages(monkeypatch, False)


@pytest.mark.parametrize(
    "trace_fn, child_sid, parent_sid, parent_turn, whole_blocks",
    [
        param(_flat_fork_trace, "fork::fa:000", "fork", 1, True, id="flat-chain"),
        param(_subagent_fork_trace, "safork::sa:a1", "safork", 1, True, id="subagent"),
        param(
            _subagent_fork_trace, "safork::sa:a1:fa:000", "safork::sa:a1", 1, True,
            id="subagent-inner-chain",
        ),
        param(_chained_fork_trace, "chain::fa:000", "chain", 1, False, id="partial-blocks"),
        param(_chained_fork_trace, "chain::fa:001", "chain::fa:000", 0, False, id="chain-to-chain"),
    ],
)  # fmt: skip
def test_child_turn_0_follows_parent_layout(
    trace_fn, child_sid: str, parent_sid: str, parent_turn: int, whole_blocks: bool
):
    """The child sends the parent's roles; the content matches too when the
    parent's messages hold whole blocks (a partial tail is the parent's own)."""
    convs = _convert(trace_fn())
    parent_messages = _messages_at(convs[parent_sid], parent_turn)
    child_messages = _turn_0(convs[child_sid])

    assert len(parent_messages) > 1
    assert [m["role"] for m in child_messages[:-1]] == [
        m["role"] for m in parent_messages
    ]
    assert child_messages[-1]["role"] == "user"
    if whole_blocks:
        assert child_messages[:-1] == parent_messages
        assert _render(child_messages).startswith(_render(parent_messages))


@pytest.mark.parametrize(
    "trace_fn, child_sid, parent_sid, parent_turn",
    [
        param(_flat_fork_trace, "fork::fa:000", "fork", 1, id="flat-chain"),
        param(_subagent_fork_trace, "safork::sa:a1", "safork", 1, id="subagent"),
    ],
)  # fmt: skip
def test_child_turn_0_flag_off_sends_one_flattened_user_message(
    flat_children, trace_fn, child_sid: str, parent_sid: str, parent_turn: int
):
    convs = _convert(trace_fn())
    parent_messages = _messages_at(convs[parent_sid], parent_turn)
    child_messages = _turn_0(convs[child_sid])

    assert [m["role"] for m in child_messages] == ["user"]
    assert not _render(child_messages).startswith(_render(parent_messages))


@pytest.mark.parametrize(
    "trace_fn", [_flat_fork_trace, _subagent_fork_trace, _chained_fork_trace]
)
def test_non_child_sessions_unchanged_by_fork_messages(monkeypatch, trace_fn):
    forked = _convert(trace_fn())
    _set_fork_messages(monkeypatch, False)
    flat = _convert(trace_fn())

    root = trace_fn()["id"]
    assert forked.keys() == flat.keys()
    assert [t.raw_messages for t in forked[root].turns] == [
        t.raw_messages for t in flat[root].turns
    ]


def _overhang_trace() -> dict:
    """The parent's last request renders more tokens than it recorded; the
    subagent child extends it."""
    cut = _blocks(15) + [900]
    parent = cut + _blocks(10, 910)
    return _trace(
        "overhang",
        [
            _req(0, _blocks(10), short=20),
            _req(2, _blocks(20), short=10),
            _req(8, cut, api=5),
            _req(14, parent, api=5),
            _subagent(15, [_req(16, parent + _blocks(10, 1000))]),
        ],
    )


def _filler_trace() -> dict:
    """The parent's first message holds 9 blocks and a 44-token tail; the
    child shares hash 9 as a full block and ends with a 1-token tail."""
    return _trace(
        "filler",
        [
            _req(0, _blocks(10), short=20),
            _req(2, _blocks(20), api=5, short=10),
            _req(3, _blocks(20) + _blocks(5, 100), short=63),
            _req(8, _blocks(30)),
            _req(10, _blocks(20) + _blocks(10, 100)),
            _req(12, _blocks(40)),
        ],
    )


@pytest.mark.parametrize(
    "trace_fn, child_sid, recorded",
    [
        param(_overhang_trace, "overhang::sa:a1", 36 * _TRACE_BS, id="parent-overhang"),
        param(_filler_trace, "filler::fa:000", 24 * _TRACE_BS + 1, id="parent-tail-filler"),
    ],
)  # fmt: skip
def test_child_turn_0_tokens_unchanged_by_fork_messages(
    monkeypatch, trace_fn, child_sid: str, recorded: int
):
    _set_fork_messages(monkeypatch, False)
    flat = _turn_0(_convert(trace_fn())[child_sid])
    _set_fork_messages(monkeypatch, True)
    forked = _turn_0(_convert(trace_fn())[child_sid])

    assert len(forked) > 1
    assert _tokens(forked) == _tokens(flat)
    assert len(_tokens(forked)) == recorded


def _tool_fork_trace(shared_blocks: int) -> dict:
    """Root turn 2 is user[0,10) + assistant[10,11) + tool[11,20) +
    assistant[20,22) + tool[22,30); a tool-result child shares its first
    ``shared_blocks`` blocks, then sends one more tool result."""
    return _trace(
        "tf",
        [
            _req(0, _blocks(10)),
            {**_req(2, _blocks(20), tool_result=True), "out": 100},
            _req(4, _blocks(30), api=5, tool_result=True),
            _req(5, _blocks(shared_blocks) + _blocks(10, 100), tool_result=True),
            _req(10, _blocks(shared_blocks) + _blocks(20, 100), tool_result=True),
            _req(12, _blocks(40)),
        ],
    )


@pytest.mark.parametrize(
    "shared_blocks, roles",
    [
        param(22, ["user", "assistant", "tool", "assistant", "tool"], id="ends-at-assistant"),
        param(21, ["user", "assistant", "tool", "user"], id="cuts-inside-assistant"),
    ],
)  # fmt: skip
def test_tool_result_child_answers_inherited_assistant(
    monkeypatch, shared_blocks: int, roles: list[str]
):
    trace = _tool_fork_trace(shared_blocks)
    _set_fork_messages(monkeypatch, False, True)
    flat = _turn_0(_convert(trace)["tf::fa:000"])
    _set_fork_messages(monkeypatch, True, True)
    convs = _convert(trace)
    child = _turn_0(convs["tf::fa:000"])
    parent = _messages_at(convs["tf"], 2)

    assert [m["role"] for m in child] == roles
    assert child[: len(roles) - 1] == parent[: len(roles) - 1]
    if roles[-1] == "tool":
        assert child[-1]["tool_call_id"] == child[-2]["tool_calls"][0]["id"]
        assert child[-1]["tool_call_id"] == parent[-1]["tool_call_id"]
    assert _tokens(child) == _tokens(flat)
    assert len(_tokens(child)) == trace["requests"][3]["in"]
    ids = [
        m["tool_call_id"]
        for m in _messages_at(convs["tf::fa:000"], 1)
        if m["role"] == "tool"
    ]
    assert len(ids) == len(set(ids)) == roles.count("tool") + 1


def _random_trace(seed: int) -> dict:
    """Growing root requests, flat branches off them, and maybe a subagent,
    with random partial last blocks, missing blocks and tool results."""
    rng = random.Random(seed)

    def short() -> int:
        # Negative: ``in`` runs past the recorded hash blocks (missing blocks).
        return rng.choice([0, rng.randint(1, 63), -rng.randint(1, 3 * _TRACE_BS)])

    main = _blocks(rng.randint(2, 10))
    reqs = [_req(0, main, short=short())]
    t = 1.0
    for i in range(rng.randint(2, 4)):
        # A branch cut here ends with the next request's 1-block assistant.
        after_assistant = len(main) + 1
        main = main + _blocks(rng.randint(2, 10), 1000 * (i + 1))
        reqs.append(_req(t, main, api=5, short=short(), tool_result=rng.random() < 0.3))
        if rng.random() < 0.7:
            base = main if rng.random() < 0.5 else main[:after_assistant]
            branch = base + _blocks(rng.randint(1, 10), 50000 + 1000 * i)
            reqs.append(
                _req(t + 1, branch, short=short(), tool_result=rng.random() < 0.5)
            )
            more = _blocks(rng.randint(1, 5), 60000 + 1000 * i)
            reqs.append(_req(t + 6, branch + more, short=short()))
        t += 7
    if rng.random() < 0.5:
        sa = main + _blocks(rng.randint(1, 10), 90000)
        reqs.append(
            _subagent(
                t + 0.5,
                [
                    _req(t + 1, sa, short=short()),
                    _req(t + 3, sa + _blocks(5, 91000), short=short()),
                ],
            )
        )
    return _trace(f"rand{seed}", reqs)


def _recorded_in(trace: dict, conv: Conversation) -> int:
    turn = conv.turns[0]
    req = trace["requests"][turn.source_outer_idx]
    if turn.source_inner_idx is not None:
        req = req["requests"][turn.source_inner_idx]
    return req["in"]


@pytest.mark.parametrize("tool_shaped", [False, True])
def test_random_traces_tokens_unchanged_by_fork_messages(monkeypatch, tool_shaped):
    """Every turn of every session sends the same tokens with the setting on
    and off, and every turn 0 sends exactly its recorded ``in``."""
    forks = missing = tool_tails = 0
    for seed in range(40):
        trace = _random_trace(seed)
        missing += any(
            len(r["hash_ids"]) < r["in"] // _TRACE_BS
            for r in trace["requests"]
            if r["type"] == "n"
        )
        _set_fork_messages(monkeypatch, False, tool_shaped)
        flat = _convert(trace)
        _set_fork_messages(monkeypatch, True, tool_shaped)
        forked = _convert(trace)
        assert forked.keys() == flat.keys()
        for sid, conv in forked.items():
            for k in range(len(conv.turns)):
                assert _tokens(_messages_at(conv, k)) == _tokens(
                    _messages_at(flat[sid], k)
                ), (seed, sid, k)
            assert len(_tokens(_turn_0(conv))) == _recorded_in(trace, conv)
            forks += len(_turn_0(conv)) > len(_turn_0(flat[sid]))
            tool_tails += _turn_0(conv)[-1]["role"] == "tool"
    assert forks > 10
    assert missing > 10
    assert (tool_tails > 0) is tool_shaped


@pytest.mark.parametrize("tool_shaped", [False, True])
def test_child_fork_messages_serial_parallel_byte_identical(
    tmp_path, monkeypatch, tool_shaped: bool
):
    monkeypatch.setattr(
        env_mod.Environment.DATASET, "WEKA_TOOL_SHAPED_MESSAGES", tool_shaped
    )
    traces = [
        _flat_fork_trace(),
        _subagent_fork_trace(),
        _chained_fork_trace(),
        _tool_fork_trace(22),
    ]
    serial, _parallel = _run_both(tmp_path, monkeypatch, traces)
    by_sid = {c.session_id: c for c in serial}
    assert len(by_sid["chain::fa:001"].turns[0].raw_messages) > 2
    for conv in serial:
        for k in range(len(conv.turns)):
            assert _messages_at(conv, k)[-1]["role"] in ("user", "tool")


def test_empty_parent_turns_do_not_break_forking(tmp_path, monkeypatch):
    """Zero-input root turns have no segments to inherit; the child keeps the
    flat split on both paths."""
    trace = _trace(
        "empty",
        [
            {**_req(0, []), "in": 0},
            {**_req(1, []), "in": 0},
            _subagent(2, [_req(3, _blocks(5))]),
        ],
    )
    serial, _parallel = _run_both(tmp_path, monkeypatch, [trace])
    child = next(c for c in serial if c.session_id == "empty::sa:a1")
    assert [m["role"] for m in child.turns[0].raw_messages] == ["user"]
