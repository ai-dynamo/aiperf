# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource, SampledSession
from tests.unit.timing.conftest import make_credit


def _mk_source(ds: DatasetMetadata) -> ConversationSource:
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations],
    )
    return ConversationSource(ds, sampler)


@pytest.fixture
def ds():
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="c1",
                turns=[TurnMetadata(timestamp_ms=0.0), TurnMetadata(delay_ms=100.0)],
            ),
            ConversationMetadata(
                conversation_id="c2", turns=[TurnMetadata(timestamp_ms=50.0)]
            ),
            ConversationMetadata(
                conversation_id="c3",
                turns=[
                    TurnMetadata(timestamp_ms=100.0),
                    TurnMetadata(delay_ms=50.0),
                    TurnMetadata(delay_ms=75.0),
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


@pytest.fixture
def src(ds):
    return _mk_source(ds)


class TestConversationSource:
    def test_next_returns_sampled_session(self, src):
        s = src.next()
        assert isinstance(s, SampledSession)
        assert s.conversation_id in ["c1", "c2", "c3"]
        assert s.metadata is not None
        assert len(s.x_correlation_id) == 36

    def test_unique_correlation_ids(self, src):
        assert src.next().x_correlation_id != src.next().x_correlation_id

    def test_sequential_order(self, ds):
        src = _mk_source(ds)
        assert [src.next().conversation_id for _ in range(3)] == ["c1", "c2", "c3"]

    def test_get_metadata_returns_conversation(self, src):
        m = src.get_metadata("c1")
        assert m.conversation_id == "c1"
        assert len(m.turns) == 2

    def test_get_metadata_raises_for_invalid(self, src):
        with pytest.raises(KeyError, match="No metadata for conversation bad"):
            src.get_metadata("bad")


class TestMultiTurn:
    @pytest.fixture
    def mt_src(self):
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="mt",
                    turns=[
                        TurnMetadata(timestamp_ms=1000.0),
                        TurnMetadata(delay_ms=50.0),
                        TurnMetadata(delay_ms=100.0),
                    ],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        return _mk_source(ds)

    @pytest.mark.parametrize("turn,exp_delay", [(0, 50.0), (1, 100.0)])  # fmt: skip
    def test_get_next_turn_metadata(self, mt_src, turn, exp_delay):
        cr = make_credit(conv_id="mt", turn=turn, is_final=False)
        assert mt_src.get_next_turn_metadata(cr).delay_ms == exp_delay

    def test_raises_when_no_next_turn(self, mt_src):
        cr = make_credit(conv_id="mt", turn=2, is_final=True)
        with pytest.raises(ValueError, match="No turn 3"):
            mt_src.get_next_turn_metadata(cr)
