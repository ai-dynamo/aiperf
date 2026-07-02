# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.credit.structs import Credit, TurnToSend


def _make_credit(**overrides) -> Credit:
    defaults = dict(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-1",
        x_correlation_id="x-1",
        turn_index=0,
        num_turns=3,
        issued_at_ns=1000,
    )
    defaults.update(overrides)
    return Credit(**defaults)


def test_credit_default_cache_bust_fields():
    credit = _make_credit()
    assert credit.cache_bust_marker is None
    assert credit.cache_bust_target == CacheBustTarget.NONE
    assert credit.max_tokens_override is None


def test_credit_cache_bust_roundtrip():
    credit = _make_credit(
        cache_bust_marker="\n\n[rid:abc123]",
        cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
    )
    encoded = msgspec.msgpack.encode(credit)
    decoded = msgspec.msgpack.decode(encoded, type=Credit)
    assert decoded.cache_bust_marker == "\n\n[rid:abc123]"
    assert decoded.cache_bust_target == CacheBustTarget.SYSTEM_SUFFIX


def test_credit_omit_defaults_keeps_wire_flat_when_disabled():
    credit_off = _make_credit()
    credit_on = _make_credit(
        cache_bust_marker="[rid:abc123]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
    )
    off_size = len(msgspec.msgpack.encode(credit_off))
    on_size = len(msgspec.msgpack.encode(credit_on))
    assert on_size > off_size
    encoded_off = msgspec.msgpack.encode(credit_off)
    assert b"cache_bust" not in encoded_off
    assert b"max_tokens_override" not in encoded_off


def test_credit_max_tokens_override_roundtrip():
    credit = _make_credit(max_tokens_override=1)
    encoded = msgspec.msgpack.encode(credit)
    decoded = msgspec.msgpack.decode(encoded, type=Credit)
    assert decoded.max_tokens_override == 1


def test_turn_to_send_from_previous_credit_propagates_cache_bust():
    parent = _make_credit(
        cache_bust_marker="[rid:abc123]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
    )
    next_turn = TurnToSend.from_previous_credit(parent)
    assert next_turn.cache_bust_marker == "[rid:abc123]\n\n"
    assert next_turn.cache_bust_target == CacheBustTarget.SYSTEM_PREFIX
    assert next_turn.turn_index == parent.turn_index + 1


def test_turn_to_send_does_not_propagate_max_tokens_override():
    parent = _make_credit(max_tokens_override=1)
    next_turn = TurnToSend.from_previous_credit(parent)
    assert next_turn.max_tokens_override is None
