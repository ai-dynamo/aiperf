# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphPhaseChannel: first-class graph-phase state, no sampler for graph runs.

Graph runs used to fabricate stub conversations so PhaseOrchestrator's
unconditional sampler build would not crash, and smuggled the ParsedGraph +
warmup handoff through attributes bolted onto the generic ConversationSource.
These tests pin the native shape: with a parsed graph the orchestrator builds
NO sampler and NO conversation source, and phase state travels on a typed
channel.
"""

from unittest.mock import MagicMock

from aiperf.common.models import DatasetMetadata
from aiperf.common.models.dataset_models import GraphDatasetMetadata
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.graph_channel import GraphPhaseChannel
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from tests.unit.timing.conftest import make_timing_config


def _parsed() -> ParsedGraph:
    return ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-1", tags=["x"])])


def _make_router() -> MagicMock:
    router = MagicMock()
    router.set_return_callback = MagicMock()
    router.set_first_token_callback = MagicMock()
    return router


def test_channel_holds_parsed_graph_and_handoff_slot():
    channel = GraphPhaseChannel(parsed_graph=_parsed())
    assert channel.parsed_graph.traces[0].id == "t-1"
    assert channel.warmup_handoff is None


def test_conversation_source_has_no_graph_attributes():
    """The R6 attribute-injection channel is gone."""
    from aiperf.timing.conversation_source import ConversationSource

    assert "parsed_graph" not in ConversationSource.__init__.__code__.co_names
    src = ConversationSource.__new__(ConversationSource)
    assert not hasattr(src, "parsed_graph")
    assert not hasattr(src, "graph_warmup_handoff")


def test_graph_orchestrator_builds_no_sampler_and_owns_channel():
    """A graph run builds NO sampler and NO conversation source.

    ``DatasetMetadata.conversations`` is empty for graph runs, so the
    conversation-shaped sampler would crash on empty ids; the orchestrator must
    skip both and own a ``GraphPhaseChannel`` instead.
    """
    orchestrator = PhaseOrchestrator(
        config=make_timing_config(TimingMode.GRAPH_IR),
        phase_publisher=MagicMock(),
        credit_router=_make_router(),
        dataset_metadata=DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            graph=GraphDatasetMetadata(trace_ids=["t-1"]),
        ),
        parsed_graph=_parsed(),
    )
    assert orchestrator._dataset_sampler is None
    assert orchestrator._conversation_source is None
    assert orchestrator._graph_channel is not None
    assert orchestrator._graph_channel.parsed_graph.traces[0].id == "t-1"


def test_non_graph_orchestrator_builds_sampler_and_source():
    """A non-graph run is unchanged: sampler + ConversationSource built, no channel."""
    from aiperf.common.models import ConversationMetadata, TurnMetadata

    orchestrator = PhaseOrchestrator(
        config=make_timing_config(
            TimingMode.REQUEST_RATE, request_count=5, request_rate=10.0
        ),
        phase_publisher=MagicMock(),
        credit_router=_make_router(),
        dataset_metadata=DatasetMetadata(
            conversations=[
                ConversationMetadata(conversation_id="conv-0", turns=[TurnMetadata()])
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        ),
    )
    assert orchestrator._dataset_sampler is not None
    assert orchestrator._conversation_source is not None
    assert orchestrator._graph_channel is None
