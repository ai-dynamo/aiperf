# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GRAPH_ADAPTER plugin category is registered, resolves its classes, and backs format detection."""

from __future__ import annotations

from pathlib import Path

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

_DYNAMO_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "dataset/graph/adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
)


def test_graph_adapter_category_registered() -> None:
    """``PluginType.GRAPH_ADAPTER`` exists and lists the ``dynamo_trace`` entry."""
    assert hasattr(PluginType, "GRAPH_ADAPTER")
    names = {e.name for e in plugins.iter_entries(PluginType.GRAPH_ADAPTER)}
    assert {"dynamo_trace"} <= names


def test_graph_adapter_classes_resolve() -> None:
    """The ``dynamo_trace`` entry resolves to the real ``DynamoTraceAdapter`` class."""
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    assert (
        plugins.get_class(PluginType.GRAPH_ADAPTER, "dynamo_trace")
        is DynamoTraceAdapter
    )


def test_detect_format_resolves_dynamo_fixture() -> None:
    """Registry-backed detection identifies a real nested dynamo trace fixture."""
    from aiperf.dataset.graph.parser import detect_format

    assert detect_format(_DYNAMO_FIXTURE) == "dynamo_trace"
