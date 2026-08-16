# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX FIRST_TURN_PREFIX cache-bust marker on the agent-graph dispatch path."""

from __future__ import annotations

import copy

import pytest
from pytest import param

from aiperf.common.enums import CacheBustScope, CacheBustTarget
from aiperf.graph.worker_materialize import stamp_cache_bust_marker
from aiperf.timing.strategies.cache_bust import (
    build_trace_instance_marker,
    inject_marker_at_first_user_message,
    inject_marker_at_system_message,
)

_BENCH = "bench-42"
_FTP = CacheBustTarget.FIRST_TURN_PREFIX


def _payload() -> dict:
    """A representative materialized graph payload (system + two user turns)."""
    return {
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "first user turn"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second user turn"},
        ],
        "model": "m",
        "stream": True,
    }


def _stamp(
    payload: dict, trace_instance_id: str, target: CacheBustTarget = _FTP
) -> None:
    stamp_cache_bust_marker(
        payload,
        benchmark_id=_BENCH,
        trace_instance_id=trace_instance_id,
        target=target,
    )


def _first_user_content(payload: dict) -> str:
    return payload["messages"][1]["content"]


def test_marker_format_is_rid_prefix_with_blank_line() -> None:
    """The marker is ``[rid:<12hex>]\\n\\n`` -- agentx FIRST_TURN_PREFIX shape."""
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    assert marker is not None
    assert marker.startswith("[rid:")
    assert marker.endswith("]\n\n")
    digest = marker[len("[rid:") : -len("]\n\n")]
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


def test_none_target_mints_no_marker() -> None:
    """``NONE`` mints ``None`` so callers can pass it through unconditionally."""
    assert (
        build_trace_instance_marker(_BENCH, "t-1#0", target=CacheBustTarget.NONE)
        is None
    )


def test_stamp_none_is_byte_identical_noop() -> None:
    """With cache-bust NONE the materialized payload is unchanged (today's behavior)."""
    payload = _payload()
    original = copy.deepcopy(payload)
    _stamp(payload, "t-1#0", target=CacheBustTarget.NONE)
    assert payload == original


def test_stamp_first_turn_prefix_marks_only_first_user_turn() -> None:
    """The marker prepends to the FIRST user turn only; all other turns untouched."""
    payload = _payload()
    _stamp(payload, "t-1#0")
    msgs = payload["messages"]
    assert msgs[0] == {"role": "system", "content": "you are a helpful assistant"}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"].startswith("[rid:")
    assert msgs[1]["content"].endswith("first user turn")
    assert msgs[2] == {"role": "assistant", "content": "ok"}
    # SECOND user turn is NOT stamped.
    assert msgs[3] == {"role": "user", "content": "second user turn"}


def test_marker_is_shared_across_all_dispatches_of_one_trace_instance() -> None:
    """Every dispatch of ONE trace instance carries the IDENTICAL marker."""
    p_turn0 = _payload()
    p_turn1 = _payload()
    _stamp(p_turn0, "t-1#0")
    _stamp(p_turn1, "t-1#0")
    m0 = _first_user_content(p_turn0)[: len("[rid:000000000000]\n\n")]
    m1 = _first_user_content(p_turn1)[: len("[rid:000000000000]\n\n")]
    assert m0 == m1


@pytest.mark.parametrize(
    "left,right,same",
    [
        param(
            (_BENCH, "t-1#0"), (_BENCH, "t-2#0"), False,
            id="distinct_instances_bust_each_other",
        ),
        param(
            (_BENCH, "t-1#0"), (_BENCH, "t-1#1"), False,
            id="recycled_template_mints_fresh_marker",
        ),
        param(
            ("bench-A", "t-1#0"), ("bench-B", "t-1#0"), False,
            id="benchmark_id_salts_the_marker",
        ),
        param(
            (_BENCH, "t-9#0"), (_BENCH, "t-9#0"), True,
            id="same_inputs_are_deterministic",
        ),
        param(
            (_BENCH, "t-1#0"), (_BENCH, "t-1#0"), True,
            id="subagent_descendant_shares_root_instance_marker",
        ),
    ],
)  # fmt: skip
def test_marker_identity_is_keyed_on_benchmark_and_instance_id(
    left: tuple[str, str], right: tuple[str, str], same: bool
) -> None:
    """Markers are a deterministic function of (benchmark_id, trace_instance_id) only."""
    # A subagent dispatch keyed on the same ROOT instance id reuses the marker:
    # the adapter pins credit.trace_id to the root instance, and only the runtime
    # parent_trace_id carries the ``::sa:`` / ``::loop#N`` suffixes.
    a = build_trace_instance_marker(*left, target=_FTP)
    b = build_trace_instance_marker(*right, target=_FTP)
    assert (a == b) is same


def test_stamp_is_idempotent() -> None:
    """Re-stamping the same instance's marker does not stack it (agentx idempotency)."""
    payload = _payload()
    _stamp(payload, "t-1#0")
    once = copy.deepcopy(payload)
    _stamp(payload, "t-1#0")
    assert payload == once


def test_inject_into_multimodal_first_user_content() -> None:
    """Multimodal list content gets a leading text marker part (agentx parity)."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    inject_marker_at_first_user_message(messages, marker)
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": marker.strip()}
    assert content[1] == {"type": "text", "text": "hi"}


def test_inject_no_user_turn_is_noop() -> None:
    """No user-role message -> nothing is stamped (no crash)."""
    messages = [{"role": "system", "content": "sys"}]
    original = copy.deepcopy(messages)
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    inject_marker_at_first_user_message(messages, marker)
    assert messages == original


def test_inject_none_marker_is_noop() -> None:
    """A ``None`` marker (NONE target) stamps nothing."""
    messages = [{"role": "user", "content": "hi"}]
    original = copy.deepcopy(messages)
    inject_marker_at_first_user_message(messages, None)
    assert messages == original


@pytest.mark.parametrize(
    "missing_messages",
    [
        param({}, id="no_messages_key"),
        param({"messages": None}, id="messages_is_none"),
        param({"messages": "not-a-list"}, id="messages_is_not_a_list"),
    ],
)  # fmt: skip
def test_stamp_tolerates_payload_without_messages_list(
    missing_messages: dict,
) -> None:
    """A payload lacking a ``messages`` list is left untouched (graceful)."""
    payload = dict(missing_messages)
    original = copy.deepcopy(payload)
    _stamp(payload, "t-1#0")
    assert payload == original


# ---------------------------------------------------------------------------
# SYSTEM_PREFIX / SYSTEM_SUFFIX (per-instance, graph path)
# ---------------------------------------------------------------------------

_SP = CacheBustTarget.SYSTEM_PREFIX
_SS = CacheBustTarget.SYSTEM_SUFFIX


def test_system_prefix_stamps_system_message_not_user() -> None:
    """SYSTEM_PREFIX prepends to the system message; user turns are untouched."""
    payload = _payload()
    stamp_cache_bust_marker(
        payload, benchmark_id=_BENCH, trace_instance_id="t-1#0", target=_SP
    )
    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("[rid:")
    assert msgs[0]["content"].endswith("you are a helpful assistant")
    # user turns unchanged
    assert msgs[1] == {"role": "user", "content": "first user turn"}
    assert msgs[3] == {"role": "user", "content": "second user turn"}


def test_system_suffix_stamps_system_message_suffix() -> None:
    """SYSTEM_SUFFIX appends to the system message."""
    payload = _payload()
    stamp_cache_bust_marker(
        payload, benchmark_id=_BENCH, trace_instance_id="t-1#0", target=_SS
    )
    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("you are a helpful assistant")
    assert "[rid:" in msgs[0]["content"]


def test_system_prefix_is_per_instance() -> None:
    """Two instances in the same run carry different system-prompt markers."""
    p1 = _payload()
    p2 = _payload()
    stamp_cache_bust_marker(
        p1, benchmark_id=_BENCH, trace_instance_id="t-1#0", target=_SP
    )
    stamp_cache_bust_marker(
        p2, benchmark_id=_BENCH, trace_instance_id="t-2#0", target=_SP
    )
    assert p1["messages"][0]["content"] != p2["messages"][0]["content"]


def test_system_prefix_run_scope_is_shared() -> None:
    """Run scope deliberately permits cross-trace system-prompt cache reuse."""
    p1 = _payload()
    p2 = _payload()
    stamp_cache_bust_marker(
        p1,
        benchmark_id=_BENCH,
        trace_instance_id="t-1#0",
        target=_SP,
        scope=CacheBustScope.RUN,
    )
    stamp_cache_bust_marker(
        p2,
        benchmark_id=_BENCH,
        trace_instance_id="t-2#0",
        target=_SP,
        scope=CacheBustScope.RUN,
    )
    assert p1["messages"][0]["content"] == p2["messages"][0]["content"]


def test_system_prefix_reuses_the_trace_instance_marker() -> None:
    """SYSTEM_PREFIX shares the trace marker but applies it to a system message."""
    sp_marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_SP)
    ftp_marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    assert sp_marker == ftp_marker


def test_system_prefix_stamp_is_idempotent() -> None:
    """Re-stamping SYSTEM_PREFIX on the same payload does not stack the marker."""
    payload = _payload()
    stamp_cache_bust_marker(
        payload, benchmark_id=_BENCH, trace_instance_id="t-1#0", target=_SP
    )
    once = copy.deepcopy(payload)
    stamp_cache_bust_marker(
        payload, benchmark_id=_BENCH, trace_instance_id="t-1#0", target=_SP
    )
    assert payload == once


def test_inject_system_message_multimodal() -> None:
    """SYSTEM_PREFIX on a multimodal system message inserts a leading text part."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": "hi"},
    ]
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_SP)
    inject_marker_at_system_message(messages, marker, is_prefix=True)
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": marker.strip()}
    assert content[1] == {"type": "text", "text": "sys"}
    # user untouched
    assert messages[1] == {"role": "user", "content": "hi"}


def test_inject_system_message_no_system_role_is_noop() -> None:
    """No system-role message -> nothing is stamped."""
    messages = [{"role": "user", "content": "hi"}]
    original = copy.deepcopy(messages)
    inject_marker_at_system_message(messages, "[rid:abc]\n\n", is_prefix=True)
    assert messages == original
