# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip + plumbing tests for the cache-bust fields on
``Credit`` / ``CreditContext`` / ``TurnToSend``.

Covers Slice 2 of the cache-bust subsystem: the marker text + target enum
ride on the credit wire struct, default to ``None`` / ``CacheBustTarget.NONE``,
and propagate through ``TurnToSend.from_previous_credit`` so subsequent
turns in a multi-turn session re-emit the same marker.
"""

import msgspec

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.credit.structs import Credit, CreditContext, TurnToSend


def _base_credit_kwargs() -> dict:
    return {
        "id": 0,
        "phase": CreditPhase.PROFILING,
        "conversation_id": "c",
        "x_correlation_id": "x",
        "turn_index": 0,
        "num_turns": 1,
        "issued_at_ns": 0,
    }


class TestCreditCacheBustFields:
    def test_credit_defaults(self):
        c = Credit(**_base_credit_kwargs())
        assert c.cache_bust_marker is None
        assert c.cache_bust_target is CacheBustTarget.NONE

    def test_credit_with_target_only(self):
        c = Credit(
            **_base_credit_kwargs(),
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert c.cache_bust_marker is None
        assert c.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX

    def test_credit_with_marker_and_target(self):
        c = Credit(
            **_base_credit_kwargs(),
            cache_bust_marker="\n<!-- cb:abc123 -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_SUFFIX,
        )
        assert c.cache_bust_marker == "\n<!-- cb:abc123 -->\n"
        assert c.cache_bust_target is CacheBustTarget.FIRST_TURN_SUFFIX

    def test_credit_msgspec_roundtrip_defaults_omitted(self):
        """omit_defaults=True must keep wire footprint small when feature is off."""
        c = Credit(**_base_credit_kwargs())
        encoded = msgspec.json.encode(c)
        assert b"cache_bust_marker" not in encoded
        assert b"cache_bust_target" not in encoded
        decoded = msgspec.json.decode(encoded, type=Credit)
        assert decoded.cache_bust_marker is None
        assert decoded.cache_bust_target is CacheBustTarget.NONE

    def test_credit_msgspec_roundtrip_with_values(self):
        c = Credit(
            **_base_credit_kwargs(),
            cache_bust_marker="\n<!-- cb:zzz -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
        )
        encoded = msgspec.json.encode(c)
        decoded = msgspec.json.decode(encoded, type=Credit)
        assert decoded.cache_bust_marker == "\n<!-- cb:zzz -->\n"
        assert decoded.cache_bust_target is CacheBustTarget.SYSTEM_SUFFIX


class TestCreditContextCarriesCacheBust:
    def test_context_holds_credit_with_cache_bust_fields(self):
        credit = Credit(
            **_base_credit_kwargs(),
            cache_bust_marker="\n<!-- cb:ctx -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=12345)
        assert ctx.credit.cache_bust_marker == "\n<!-- cb:ctx -->\n"
        assert ctx.credit.cache_bust_target is CacheBustTarget.FIRST_TURN_PREFIX

    def test_context_msgspec_roundtrip(self):
        credit = Credit(
            **_base_credit_kwargs(),
            cache_bust_marker="\n<!-- cb:rt -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=99)
        encoded = msgspec.json.encode(ctx)
        decoded = msgspec.json.decode(encoded, type=CreditContext)
        assert decoded.credit.cache_bust_marker == "\n<!-- cb:rt -->\n"
        assert decoded.credit.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX


class TestTurnToSendPropagatesCacheBust:
    def test_default_fields_on_direct_construct(self):
        tts = TurnToSend(
            conversation_id="c",
            x_correlation_id="x",
            turn_index=0,
            num_turns=1,
        )
        assert tts.cache_bust_marker is None
        assert tts.cache_bust_target is CacheBustTarget.NONE

    def test_from_previous_credit_propagates_fields(self):
        kwargs = _base_credit_kwargs()
        kwargs["num_turns"] = 2
        credit = Credit(
            **kwargs,
            cache_bust_marker="\n<!-- cb:carry -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        )
        tts = TurnToSend.from_previous_credit(credit)
        assert tts.cache_bust_marker == "\n<!-- cb:carry -->\n"
        assert tts.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX
        assert tts.turn_index == 1
        assert tts.num_turns == 2

    def test_from_previous_credit_no_marker_stays_none(self):
        credit = Credit(**_base_credit_kwargs())
        tts = TurnToSend.from_previous_credit(credit)
        assert tts.cache_bust_marker is None
        assert tts.cache_bust_target is CacheBustTarget.NONE
