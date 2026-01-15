# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.models import ConversationMetadata, TurnMetadata
from tests.unit.timing.conftest import make_dataset, make_dataset_with_schedule


class TestDatasetMetadata:
    def test_multi_turn_from_schedule(self):
        sched = [(0, "c1"), (100, "c2"), (150, "c1"), (200, "c1"), (250, "c2")]
        md = make_dataset_with_schedule(sched)
        d = {c.conversation_id: c for c in md.conversations}
        assert len(md.conversations) == 2
        assert len(d["c1"].turns) == 3
        assert d["c1"].turns[0].timestamp_ms == 0
        assert [t.delay_ms for t in d["c1"].turns[1:]] == [150, 50]
        assert len(d["c2"].turns) == 2

    def test_extract_fixed_schedule(self):
        sched = [(0, "c1"), (100, "c2"), (200, "c3")]
        md = make_dataset_with_schedule(sched)
        ext = sorted(
            [
                (c.turns[0].timestamp_ms, c.conversation_id)
                for c in md.conversations
                if c.turns and c.turns[0].timestamp_ms is not None
            ],
            key=lambda x: x[0],
        )
        assert ext == sched

    def test_mixed_turn_counts(self):
        md = make_dataset(conv_ids=["s", "d", "t"], turn_counts=[1, 2, 3])
        d = {c.conversation_id: c for c in md.conversations}
        assert [len(d[k].turns) for k in ["s", "d", "t"]] == [1, 2, 3]


class TestFloatTimestamps:
    def test_conversation_preserves_floats(self):
        turns = [
            TurnMetadata(timestamp_ms=0.0, delay_ms=None),
            TurnMetadata(timestamp_ms=100.5, delay_ms=100.5),
            TurnMetadata(timestamp_ms=200.75, delay_ms=100.25),
        ]
        c = ConversationMetadata(conversation_id="t", turns=turns)
        assert (
            c.turns[0].timestamp_ms,
            c.turns[1].timestamp_ms,
            c.turns[2].timestamp_ms,
        ) == (0.0, 100.5, 200.75)
        assert (c.turns[1].delay_ms, c.turns[2].delay_ms) == (100.5, 100.25)

    def test_dataset_preserves_floats(self):
        md = make_dataset_with_schedule([(0.0, "c1"), (100.5, "c2"), (150.75, "c1")])
        d = {c.conversation_id: c for c in md.conversations}
        assert (
            d["c1"].turns[0].timestamp_ms,
            d["c1"].turns[1].timestamp_ms,
            d["c2"].turns[0].timestamp_ms,
        ) == (0.0, 150.75, 100.5)
