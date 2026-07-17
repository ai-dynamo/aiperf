# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def test_graph_adapter_category_registered() -> None:
    assert hasattr(PluginType, "GRAPH_ADAPTER")
    names = {e.name for e in plugins.iter_entries(PluginType.GRAPH_ADAPTER)}
    assert {"native", "weka_trace", "dynamo_trace"} <= names


def test_graph_adapter_classes_resolve() -> None:
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter
    from aiperf.dataset.graph.adapters.weka.trace import WekaTraceAdapter

    assert (
        plugins.get_class(PluginType.GRAPH_ADAPTER, "dynamo_trace")
        is DynamoTraceAdapter
    )
    assert plugins.get_class(PluginType.GRAPH_ADAPTER, "weka_trace") is WekaTraceAdapter


def test_detect_format_resolves_dynamo_fixture() -> None:
    from pathlib import Path

    from aiperf.dataset.graph.parser import detect_format

    fixture = (
        Path(__file__).resolve().parents[1]
        / "dataset/graph/adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
    )
    assert detect_format(fixture) == "dynamo_trace"
