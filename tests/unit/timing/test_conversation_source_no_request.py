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


def test_sample_ordinal_stable_for_orchestrator_roots_only():
    """next() assigns a monotonic, seed-order-stable ordinal to orchestrator
    roots (used to seed reproducible per-instance think-time); non-orchestrator
    sessions get None. Two instances of the same orchestrator get distinct
    ordinals so their draws are independent."""
    from aiperf.common.models import DatasetMetadata
    from aiperf.plugin.enums import DatasetSamplingStrategy
    from aiperf.timing.conversation_source import ConversationSource

    orch = ConversationMetadata(
        conversation_id="start",
        turns=[TurnMetadata(timestamp_ms=0.0, no_request=True)],
        agent_depth=0,
        is_orchestrator=True,
    )
    plain = ConversationMetadata(
        conversation_id="plain",
        turns=[TurnMetadata(timestamp_ms=0.0)],
        agent_depth=0,
    )
    meta = DatasetMetadata(
        conversations=[orch, plain],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )

    class _Sampler:
        _seq = iter(["start", "start", "plain"])

        def next_conversation_id(self):
            return next(self._seq)

    src = ConversationSource(meta, _Sampler())
    s0, s1, sp = src.next(), src.next(), src.next()
    assert src.sample_ordinal(s0.x_correlation_id) == 0
    assert src.sample_ordinal(s1.x_correlation_id) == 1
    assert src.sample_ordinal(sp.x_correlation_id) is None
