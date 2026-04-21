# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the six credit-progress envelopes plus the records-
manager envelopes that carry the same msgspec payloads.

Exists because the credit/records stats are ``msgspec.Struct`` but the
envelopes remain Pydantic during Phase 2 of the msgspec ZMQ migration. The
round-trip must hold for every envelope that carries a struct payload.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.messages.progress_messages import (
    AllRecordsReceivedMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.models import CreditPhaseStats, PhaseRecordsStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhasesConfiguredMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    CreditsCompleteMessage,
)
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig


def _stats() -> CreditPhaseStats:
    return CreditPhaseStats(
        phase="profiling",
        start_ns=100,
        total_expected_requests=100,
        requests_sent=50,
        requests_completed=40,
        total_session_turns=8,
    )


def _record_stats() -> PhaseRecordsStats:
    return PhaseRecordsStats(
        phase="profiling",
        start_ns=100,
        success_records=7,
        error_records=1,
    )


def _config() -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase="profiling",
        timing_mode=TimingMode.REQUEST_RATE,
        arrival_pattern=ArrivalPattern.POISSON,
        total_expected_requests=100,
        concurrency=10,
    )


@pytest.mark.parametrize(
    "message_factory",
    [
        param(
            lambda: CreditPhasesConfiguredMessage(service_id="t", configs=[_config()]),
            id="CreditPhasesConfiguredMessage",
        ),
        param(
            lambda: CreditPhaseStartMessage(
                service_id="t", stats=_stats(), config=_config()
            ),
            id="CreditPhaseStartMessage",
        ),
        param(
            lambda: CreditPhaseProgressMessage(service_id="t", stats=_stats()),
            id="CreditPhaseProgressMessage",
        ),
        param(
            lambda: CreditPhaseSendingCompleteMessage(service_id="t", stats=_stats()),
            id="CreditPhaseSendingCompleteMessage",
        ),
        param(
            lambda: CreditPhaseCompleteMessage(service_id="t", stats=_stats()),
            id="CreditPhaseCompleteMessage",
        ),
        param(
            lambda: CreditsCompleteMessage(service_id="t"),
            id="CreditsCompleteMessage",
        ),
        param(
            lambda: RecordsProcessingStatsMessage(
                service_id="t", processing_stats=_record_stats()
            ),
            id="RecordsProcessingStatsMessage",
        ),
        param(
            lambda: AllRecordsReceivedMessage(
                service_id="t",
                final_processing_stats=_record_stats(),
                request_ns=123,
            ),
            id="AllRecordsReceivedMessage",
        ),
    ],
)  # fmt: skip
def test_envelope_roundtrips_via_pydantic_json(message_factory) -> None:
    """Pydantic envelope with msgspec payload must round-trip through JSON."""
    message = message_factory()

    payload = message.model_dump_json()
    decoded = type(message).model_validate_json(payload)

    assert decoded == message
    # The structs are frozen / kw-only, so equality suffices. Sanity-check
    # one field explicitly for the payload-carrying envelopes.
    if hasattr(message, "stats"):
        assert decoded.stats == message.stats


def test_progress_message_decodes_from_dict() -> None:
    """`model_validate` with a dict payload mirrors a re-delivered queue record."""
    payload = {
        "service_id": "timing-manager",
        "request_ns": 1,
        "message_type": "credit_phase_progress",
        "stats": {
            "phase": "profiling",
            "start_ns": 100,
            "total_expected_requests": 100,
            "requests_sent": 50,
            "requests_completed": 40,
            "total_session_turns": 8,
        },
    }

    msg = CreditPhaseProgressMessage.model_validate(payload)

    assert msg.stats.phase == "profiling"
    assert msg.stats.requests_completed == 40
    assert isinstance(msg.stats, CreditPhaseStats)


def test_phases_configured_message_decodes_timing_mode_enum() -> None:
    """CreditPhaseConfig.timing_mode is an ExtensibleStrEnum — dec_hook handles it."""
    payload = {
        "service_id": "timing-manager",
        "request_ns": 1,
        "message_type": "credit_phases_configured",
        "configs": [
            {
                "phase": "profiling",
                "timing_mode": "request_rate",
                "total_expected_requests": 100,
                "concurrency": 10,
            }
        ],
    }

    msg = CreditPhasesConfiguredMessage.model_validate(payload)

    assert len(msg.configs) == 1
    cfg = msg.configs[0]
    assert cfg.timing_mode == TimingMode.REQUEST_RATE
    assert cfg.concurrency == 10
