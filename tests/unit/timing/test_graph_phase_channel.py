# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphPhaseChannel owns graph-phase state so graph runs need no sampler or source."""

from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.common.models.dataset_models import GraphDatasetMetadata
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.graph_channel import GraphPhaseChannel
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from tests.unit.timing.conftest import make_timing_config


def _parsed() -> ParsedGraph:
    """Single-trace parsed graph used as the channel payload."""
    return ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-1", tags=["x"])])


def _conversation_source() -> ConversationSource:
    """A really-constructed ConversationSource, so instance state is inspectable."""
    return ConversationSource(
        DatasetMetadata(
            conversations=[
                ConversationMetadata(conversation_id="conv-0", turns=[TurnMetadata()])
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        ),
        MagicMock(),
    )


def _make_router() -> MagicMock:
    """Credit router stub exposing the callback setters the orchestrator wires."""
    router = MagicMock()
    router.set_return_callback = MagicMock()
    router.set_first_token_callback = MagicMock()
    return router


def _graph_orchestrator() -> PhaseOrchestrator:
    """Orchestrator for a AGENT_GRAPH run carrying a parsed graph."""
    return PhaseOrchestrator(
        config=make_timing_config(TimingMode.AGENT_GRAPH),
        phase_publisher=MagicMock(),
        credit_router=_make_router(),
        dataset_metadata=DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            graph=GraphDatasetMetadata(trace_ids=["t-1"]),
        ),
        parsed_graph=_parsed(),
    )


def _rate_orchestrator() -> PhaseOrchestrator:
    """Orchestrator for an ordinary request-rate run over one conversation."""
    return PhaseOrchestrator(
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


def test_channel_holds_parsed_graph() -> None:
    """A fresh channel exposes its parsed graph."""
    channel = GraphPhaseChannel(parsed_graph=_parsed())
    assert channel.parsed_graph.traces[0].id == "t-1"


def test_conversation_source_carries_no_graph_attributes() -> None:
    """ConversationSource no longer participates in graph state (R6 injection removed).

    Checks a REAL constructed instance, not ``__init__.__code__.co_names``:
    ``co_names`` membership is exact, so a reintroduced ``self._parsed_graph``
    -- the private-attr convention every other field in that ``__init__`` uses
    -- would leave a name-based assertion green. And probing ``__new__`` cannot
    fail either, since no ``__init__`` ran to set an instance attribute.
    """
    source = _conversation_source()
    graph_attrs = [
        name
        for name in vars(source)
        if "parsed_graph" in name or "graph" in name.lstrip("_").split("_")
    ]
    assert graph_attrs == [], (
        f"ConversationSource must carry no graph state; found {graph_attrs}"
    )


@pytest.mark.parametrize(
    "build_orchestrator,expects_graph_channel",
    [
        param(_graph_orchestrator, True, id="graph_run_owns_channel_only"),
        param(_rate_orchestrator, False, id="rate_run_owns_sampler_and_source_only"),
    ],
)  # fmt: skip
def test_orchestrator_builds_channel_xor_sampler_and_source(
    build_orchestrator, expects_graph_channel: bool
) -> None:
    """Graph runs own a channel and skip sampling; non-graph runs do the opposite."""
    orchestrator = build_orchestrator()
    assert (orchestrator._graph_channel is not None) is expects_graph_channel
    assert (orchestrator._dataset_sampler is None) is expects_graph_channel
    assert (orchestrator._conversation_source is None) is expects_graph_channel


def test_graph_orchestrator_channel_exposes_parsed_traces() -> None:
    """The graph run's channel carries the parsed graph through to the phase runner."""
    orchestrator = _graph_orchestrator()
    assert orchestrator._graph_channel.parsed_graph.traces[0].id == "t-1"
