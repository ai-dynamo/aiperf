# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for no_request plumbing through build_first_turn.

Verifies that a synthesized orchestrator turn (``TurnMetadata.no_request=True``)
propagates into the ``TurnToSend`` produced by ``build_first_turn``.
"""

import uuid

from aiperf.common.models import ConversationMetadata, TurnMetadata
from aiperf.timing.conversation_source import SampledSession


def _session(first_turn: TurnMetadata) -> SampledSession:
    return SampledSession(
        conversation_id="conv",
        metadata=ConversationMetadata(
            conversation_id="conv",
            turns=[first_turn],
            agent_depth=0,
        ),
        x_correlation_id=str(uuid.uuid4()),
    )


def test_build_first_turn_propagates_no_request_true():
    session = _session(TurnMetadata(timestamp_ms=0.0, no_request=True))
    turn = session.build_first_turn()
    assert turn.no_request is True


def test_build_first_turn_propagates_no_request_false():
    session = _session(TurnMetadata(timestamp_ms=0.0))
    turn = session.build_first_turn()
    assert turn.no_request is False
