# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX FIRST_TURN_PREFIX cache-bust marker on the graph-IR dispatch path.

The marker is PER-TRACE-INSTANCE: minted from ``(benchmark_id, credit.trace_id)``
and stamped onto the first user turn of the wire ``messages`` at worker
materialize time. Every dispatch of one trace instance shares the SAME marker
(its own prefix stays consistent); distinct instances get distinct markers
(cross-instance bust); a recycled template (a fresh instance id, e.g. ``t-1#1``)
mints a fresh marker. The default (``CacheBustTarget.NONE``) stamps nothing, so
the verbatim replay path is byte-identical to today unless the run passes
``--cache-bust``. Mirrors agentx's per-trajectory-tree scoping
(``cache_bust.py::resolve_tree_marker``, reset on recycle).
"""

from __future__ import annotations

import copy

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.graph.worker_materialize import stamp_cache_bust_marker
from aiperf.timing.strategies.cache_bust import (
    build_trace_instance_marker,
    inject_marker_at_first_user_message,
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


def _stamp(payload: dict, trace_instance_id: str, target: CacheBustTarget = _FTP):
    stamp_cache_bust_marker(
        payload,
        benchmark_id=_BENCH,
        trace_instance_id=trace_instance_id,
        target=target,
    )


def _first_user_content(payload: dict) -> str:
    return payload["messages"][1]["content"]


def test_marker_format_is_rid_prefix_with_blank_line():
    """The marker is ``[rid:<12hex>]\\n\\n`` -- agentx FIRST_TURN_PREFIX shape."""
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    assert marker is not None
    assert marker.startswith("[rid:")
    assert marker.endswith("]\n\n")
    digest = marker[len("[rid:") : -len("]\n\n")]
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


def test_none_target_mints_no_marker():
    """``NONE`` mints ``None`` so callers can pass it through unconditionally."""
    assert (
        build_trace_instance_marker(_BENCH, "t-1#0", target=CacheBustTarget.NONE)
        is None
    )


def test_stamp_none_is_byte_identical_noop():
    """With cache-bust NONE the materialized payload is unchanged (today's behavior)."""
    payload = _payload()
    original = copy.deepcopy(payload)
    _stamp(payload, "t-1#0", target=CacheBustTarget.NONE)
    assert payload == original


def test_stamp_first_turn_prefix_marks_only_first_user_turn():
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


def test_marker_is_shared_across_all_dispatches_of_one_trace_instance():
    """Every dispatch of ONE trace instance carries the IDENTICAL marker.

    Two distinct dispatches (different turns/nodes of the same trace instance
    ``t-1#0``) must produce the same first-user marker, so the instance's own
    conversation prefix stays consistent and prefix-caches WITHIN the instance.
    """
    p_turn0 = _payload()
    p_turn1 = _payload()
    _stamp(p_turn0, "t-1#0")
    _stamp(p_turn1, "t-1#0")
    m0 = _first_user_content(p_turn0)[: len("[rid:000000000000]\n\n")]
    m1 = _first_user_content(p_turn1)[: len("[rid:000000000000]\n\n")]
    assert m0 == m1


def test_marker_differs_across_distinct_trace_instances():
    """Distinct trace instances mint distinct markers (cross-instance bust)."""
    a = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    b = build_trace_instance_marker(_BENCH, "t-2#0", target=_FTP)
    assert a != b


def test_marker_resets_on_recycle_of_same_template():
    """A recycled template (fresh instance id ``t-1#1``) mints a FRESH marker.

    Recycling reuses the dataset trace TEMPLATE ``t-1`` in a new session slot,
    minting a new instance id (``#1`` vs ``#0``). The marker must reset so the
    recycled instance does not warm the server's cache on the prior instance's
    prefix -- mirroring agentx's per-recycle ``recycle_pass`` bump.
    """
    instance0 = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    recycled = build_trace_instance_marker(_BENCH, "t-1#1", target=_FTP)
    assert instance0 != recycled


def test_subagent_descendant_shares_root_instance_marker():
    """A nested/subagent dispatch keyed on the SAME root instance id reuses the marker.

    The adapter pins ``credit.trace_id`` to the root instance for nested/subagent
    dispatches too (only the runtime ``parent_trace_id`` carries ``::sa:`` /
    ``::loop#N`` suffixes), so a subagent turn keyed on ``t-1#0`` gets the same
    marker as the main session's turns -- matching agentx, where the whole
    trajectory TREE shares one value.
    """
    main = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    subagent = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    assert main == subagent


def test_marker_is_deterministic_for_same_inputs():
    """Same (benchmark_id, trace_instance_id) -> same marker (reproducible reruns)."""
    a = build_trace_instance_marker(_BENCH, "t-9#0", target=_FTP)
    b = build_trace_instance_marker(_BENCH, "t-9#0", target=_FTP)
    assert a == b


def test_marker_varies_per_benchmark_id():
    """Different run salts mint different markers for the same trace instance."""
    a = build_trace_instance_marker("bench-A", "t-1#0", target=_FTP)
    b = build_trace_instance_marker("bench-B", "t-1#0", target=_FTP)
    assert a != b


def test_stamp_is_idempotent():
    """Re-stamping the same instance's marker does not stack it (agentx idempotency)."""
    payload = _payload()
    _stamp(payload, "t-1#0")
    once = copy.deepcopy(payload)
    _stamp(payload, "t-1#0")
    assert payload == once


def test_inject_into_multimodal_first_user_content():
    """Multimodal list content gets a leading text marker part (agentx parity)."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    inject_marker_at_first_user_message(messages, marker)
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": marker.strip()}
    assert content[1] == {"type": "text", "text": "hi"}


def test_inject_no_user_turn_is_noop():
    """No user-role message -> nothing is stamped (no crash)."""
    messages = [{"role": "system", "content": "sys"}]
    original = copy.deepcopy(messages)
    marker = build_trace_instance_marker(_BENCH, "t-1#0", target=_FTP)
    inject_marker_at_first_user_message(messages, marker)
    assert messages == original


def test_inject_none_marker_is_noop():
    """A ``None`` marker (NONE target) stamps nothing."""
    messages = [{"role": "user", "content": "hi"}]
    original = copy.deepcopy(messages)
    inject_marker_at_first_user_message(messages, None)
    assert messages == original


@pytest.mark.parametrize(
    "missing_messages",
    [{}, {"messages": None}, {"messages": "not-a-list"}],
)
def test_stamp_tolerates_payload_without_messages_list(missing_messages):
    """A payload lacking a ``messages`` list is left untouched (graceful)."""
    payload = dict(missing_messages)
    original = copy.deepcopy(payload)
    _stamp(payload, "t-1#0")
    assert payload == original
