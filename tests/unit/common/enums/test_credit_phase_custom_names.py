# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom phase names must decode everywhere CreditPhase is a wire type.

The YAML config allows any identifier for ``phases[].name`` (docs use names
like ``steady_state_profile`` and ``main``), and that name rides the wire as
``Credit.phase``, ``CreditPhaseStats.phase``, and
``MetricRecordMetadata.benchmark_phase``. ``CreditPhase._missing_``
materializes a pseudo-member for unknown names so msgspec/Pydantic decode
accepts them in every service instead of crashing with
``Invalid enum value`` (the pre-fix behavior).
"""

import msgspec
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.message_codecs import get_message_codec
from aiperf.common.models import CreditPhaseStats
from aiperf.credit.messages import CreditPhaseStartMessage, RouterToWorkerMessage
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig

CUSTOM_NAME = "steady_state_profile"


def make_credit(phase: CreditPhase) -> Credit:
    return Credit(
        id=1,
        phase=phase,
        conversation_id="conv-1",
        x_correlation_id="x-1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=123,
    )


class TestCreditPhaseCustomMembers:
    """CreditPhase materializes pseudo-members for user-defined names."""

    def test_custom_name_constructs_pseudo_member(self) -> None:
        phase = CreditPhase(CUSTOM_NAME)
        assert isinstance(phase, CreditPhase)
        assert phase.value == CUSTOM_NAME
        assert phase == CUSTOM_NAME

    def test_custom_member_is_cached(self) -> None:
        assert CreditPhase(CUSTOM_NAME) is CreditPhase(CUSTOM_NAME)

    @pytest.mark.parametrize(
        "value,expected",
        [
            param("warmup", CreditPhase.WARMUP, id="exact"),
            param("WARMUP", CreditPhase.WARMUP, id="upper"),
            param("Profiling", CreditPhase.PROFILING, id="mixed_case"),
            param("cooldown", CreditPhase.COOLDOWN, id="cooldown"),
        ],
    )  # fmt: skip
    def test_declared_members_still_win(self, value, expected) -> None:
        assert CreditPhase(value) is expected

    def test_declared_member_set_unchanged(self) -> None:
        CreditPhase("some_other_custom_phase")
        assert set(CreditPhase.__members__) == {"WARMUP", "PROFILING", "COOLDOWN"}

    @pytest.mark.parametrize(
        "bad_value",
        [param("", id="empty_string"), param(123, id="non_string"), param(None, id="none")],
    )  # fmt: skip
    def test_invalid_values_still_rejected(self, bad_value) -> None:
        with pytest.raises(ValueError):
            CreditPhase(bad_value)

    def test_custom_member_usable_as_dict_key(self) -> None:
        counts = {CreditPhase.WARMUP: 1, CreditPhase(CUSTOM_NAME): 2}
        assert counts[CreditPhase(CUSTOM_NAME)] == 2
        assert counts[CreditPhase.WARMUP] == 1


class TestCustomPhaseWireDecode:
    """Both ZMQ transports must round-trip a custom phase name."""

    def test_credit_channel_roundtrip(self) -> None:
        """The router->worker credit channel uses a plain msgspec Decoder
        (no dec_hook) — the exact path that crashed pre-fix."""
        credit = make_credit(CreditPhase(CUSTOM_NAME))
        blob = msgspec.msgpack.Encoder().encode(credit)
        decoded = msgspec.msgpack.Decoder(RouterToWorkerMessage).decode(blob)
        assert isinstance(decoded, Credit)
        assert isinstance(decoded.phase, CreditPhase)
        assert decoded.phase == CUSTOM_NAME

    def test_message_bus_credit_phase_start_roundtrip(self) -> None:
        """The message-bus codec must round-trip CreditPhaseStats.phase and
        CreditPhaseConfig.phase carrying a custom name."""
        msg = CreditPhaseStartMessage(
            service_id="timing-manager-1",
            stats=CreditPhaseStats(phase=CUSTOM_NAME, start_ns=1),
            config=CreditPhaseConfig(
                phase=CreditPhase(CUSTOM_NAME), timing_mode=TimingMode.REQUEST_RATE
            ),
        )
        codec = get_message_codec()
        decoded = codec.decode(codec.encode(msg))
        assert decoded.stats.phase == CUSTOM_NAME
        assert isinstance(decoded.stats.phase, CreditPhase)
        assert decoded.config.phase == CUSTOM_NAME

    def test_declared_phase_roundtrip_unchanged(self) -> None:
        credit = make_credit(CreditPhase.PROFILING)
        blob = msgspec.msgpack.Encoder().encode(credit)
        decoded = msgspec.msgpack.Decoder(RouterToWorkerMessage).decode(blob)
        assert decoded.phase is CreditPhase.PROFILING
