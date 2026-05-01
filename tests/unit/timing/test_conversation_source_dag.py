# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DAG-related extensions to SampledSession and ConversationSource."""

from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.plugin import plugins
from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType
from aiperf.timing.conversation_source import ConversationSource, SampledSession


def test_routing_key_uses_parent_when_set():
    s = SampledSession(
        conversation_id="c",
        metadata=None,
        x_correlation_id="child",
        parent_correlation_id="root",
    )
    assert s.routing_key == "root"


def test_routing_key_falls_back_to_self():
    s = SampledSession(
        conversation_id="c",
        metadata=None,
        x_correlation_id="self",
    )
    assert s.routing_key == "self"


def test_sampled_session_defaults():
    s = SampledSession(conversation_id="c", metadata=None, x_correlation_id="x")
    assert s.agent_depth == 0
    assert s.parent_correlation_id is None


def _mk_source() -> ConversationSource:
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="child_conv",
                turns=[TurnMetadata(timestamp_ms=0.0)],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    SamplerClass = plugins.get_class(PluginType.DATASET_SAMPLER, ds.sampling_strategy)
    sampler = SamplerClass(
        conversation_ids=[c.conversation_id for c in ds.conversations],
    )
    return ConversationSource(ds, sampler)


def test_start_branch_child_inherits_parent_routing():
    source = _mk_source()
    child = source.start_branch_child(
        parent_correlation_id="parent-xid",
        child_conversation_id="child_conv",
        agent_depth=2,
    )
    assert child.conversation_id == "child_conv"
    assert child.parent_correlation_id == "parent-xid"
    assert child.agent_depth == 2
    assert child.routing_key == "parent-xid"
    assert child.x_correlation_id != "parent-xid"


def test_build_first_turn_propagates_dag_fields():
    """build_first_turn must carry agent_depth / parent_correlation_id into TurnToSend,
    otherwise DAG children lose sticky-routing at first dispatch."""
    source = _mk_source()
    child = source.start_branch_child(
        parent_correlation_id="parent-xid",
        child_conversation_id="child_conv",
        agent_depth=3,
    )
    turn = child.build_first_turn()
    assert turn.conversation_id == "child_conv"
    assert turn.x_correlation_id == child.x_correlation_id
    assert turn.turn_index == 0
    assert turn.agent_depth == 3
    assert turn.parent_correlation_id == "parent-xid"


def test_build_first_turn_defaults_for_root_session():
    source = _mk_source()
    session = source.next()
    turn = session.build_first_turn()
    assert turn.agent_depth == 0
    assert turn.parent_correlation_id is None
