# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ConversationSource DAG child builders.

Covers the FORK and SPAWN entry points used by the DAG BranchOrchestrator:
- ``start_branch_child``: child shares parent's correlation_id (sticky routing)
- ``start_pre_session_child``: child gets a fresh correlation_id (free routing)
"""

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource, SampledSession


def _mk_source(ds: DatasetMetadata) -> ConversationSource:
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations],
    )
    return ConversationSource(ds, sampler)


@pytest.fixture
def src() -> ConversationSource:
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=0,
            ),
            ConversationMetadata(
                conversation_id="child-conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                agent_depth=1,
                parent_conversation_id="root-conv",
                is_root=False,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    return _mk_source(ds)


class TestStartBranchChild:
    def test_returns_sampled_session(self, src: ConversationSource):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        assert isinstance(child, SampledSession)
        assert child.conversation_id == "child-conv"

    def test_inherits_parent_routing_key(self, src: ConversationSource):
        """FORK: child's routing_key equals parent's correlation_id (sticky pin)."""
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        assert child.routing_key == "parent-corr"
        assert child.parent_correlation_id == "parent-corr"
        assert child.x_correlation_id != "parent-corr"

    def test_fresh_x_correlation_id(self, src: ConversationSource):
        a = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        b = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        assert a.x_correlation_id != b.x_correlation_id
        assert len(a.x_correlation_id) == 36

    def test_carries_parent_correlation_id_to_turn_to_send(
        self, src: ConversationSource
    ):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        tts = child.build_first_turn()
        assert tts.parent_correlation_id == "parent-corr"
        assert tts.agent_depth == 1
        assert tts.branch_mode == ConversationBranchMode.FORK

    def test_default_branch_mode_is_fork(self, src: ConversationSource):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
        )
        assert child.branch_mode == ConversationBranchMode.FORK

    def test_explicit_spawn_branch_mode(self, src: ConversationSource):
        child = src.start_branch_child(
            parent_correlation_id="parent-corr",
            child_conversation_id="child-conv",
            agent_depth=1,
            branch_mode=ConversationBranchMode.SPAWN,
        )
        assert child.branch_mode == ConversationBranchMode.SPAWN
        # SPAWN still sticky-pins via parent_correlation_id at this layer;
        # routing freedom is enforced upstream by the orchestrator/router.
        assert child.routing_key == "parent-corr"

    def test_unknown_child_conversation_raises(self, src: ConversationSource):
        with pytest.raises(KeyError):
            src.start_branch_child(
                parent_correlation_id="parent-corr",
                child_conversation_id="missing",
                agent_depth=1,
            )


class TestStartPreSessionChild:
    def test_returns_sampled_session(self, src: ConversationSource):
        child = src.start_pre_session_child(child_conversation_id="child-conv")
        assert isinstance(child, SampledSession)
        assert child.conversation_id == "child-conv"

    def test_fresh_routing_key(self, src: ConversationSource):
        """SPAWN pre-session: routing_key equals child's own x_correlation_id (free routing)."""
        child = src.start_pre_session_child(child_conversation_id="child-conv")
        assert child.parent_correlation_id is None
        assert child.routing_key == child.x_correlation_id
        assert len(child.x_correlation_id) == 36

    def test_no_parent_correlation_on_turn_to_send(self, src: ConversationSource):
        child = src.start_pre_session_child(child_conversation_id="child-conv")
        tts = child.build_first_turn()
        assert tts.parent_correlation_id is None
        assert tts.agent_depth == 1
        assert tts.branch_mode == ConversationBranchMode.SPAWN

    def test_independent_per_call(self, src: ConversationSource):
        a = src.start_pre_session_child(child_conversation_id="child-conv")
        b = src.start_pre_session_child(child_conversation_id="child-conv")
        assert a.x_correlation_id != b.x_correlation_id
        assert a.routing_key != b.routing_key

    def test_unknown_child_conversation_raises(self, src: ConversationSource):
        with pytest.raises(KeyError):
            src.start_pre_session_child(child_conversation_id="missing")
