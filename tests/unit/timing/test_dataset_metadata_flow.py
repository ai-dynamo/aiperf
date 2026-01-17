# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.models import ConversationMetadata, TurnMetadata


class TestFloatTimestamps:
    """Verify that float timestamps/delays are preserved through the model layer.

    This ensures Pydantic doesn't coerce floats to ints, which would cause
    precision loss in sub-millisecond timing scenarios.
    """

    def test_conversation_preserves_floats(self):
        turns = [
            TurnMetadata(timestamp_ms=0.0, delay_ms=None),
            TurnMetadata(timestamp_ms=100.5, delay_ms=100.5),
            TurnMetadata(timestamp_ms=200.75, delay_ms=100.25),
        ]
        conv = ConversationMetadata(conversation_id="test", turns=turns)

        assert conv.turns[0].timestamp_ms == 0.0
        assert conv.turns[1].timestamp_ms == 100.5
        assert conv.turns[2].timestamp_ms == 200.75
        assert conv.turns[1].delay_ms == 100.5
        assert conv.turns[2].delay_ms == 100.25
