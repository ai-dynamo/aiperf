# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-native dataset broadcast types.

Graph runs broadcast a graph facet on ``DatasetMetadata`` and a graph-typed
``client_metadata`` (store + sidecar locations) instead of fabricating stub
conversations. These tests lock the discriminated-union wire round-trip and
the plugin registration the worker's generic client-store construction
resolves through.
"""

from pathlib import Path

import pytest

from aiperf.common.exceptions import InvalidOperationError
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models.dataset_models import (
    DatasetMetadata,
    GraphDatasetMetadata,
    GraphSegmentClientMetadata,
)
from aiperf.plugin.enums import DatasetClientStoreType, DatasetSamplingStrategy


def _notification(tmp_path: Path) -> DatasetConfiguredNotification:
    return DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
            graph=GraphDatasetMetadata(
                trace_ids=["t-1", "t-2"],
                prefix_cache_by_trace={"t-1": {"n0": [3, 7]}},
            ),
        ),
        client_metadata=GraphSegmentClientMetadata(
            store_base_path=tmp_path,
            benchmark_id="bench-x",
            sidecar_path=tmp_path / "aiperf_graph_meta_bench-x" / "graph_meta.msgpack",
        ),
    )


def test_graph_client_metadata_wire_round_trip(tmp_path):
    msg = _notification(tmp_path)
    decoded = DatasetConfiguredNotification.model_validate(msg.model_dump(mode="json"))
    assert isinstance(decoded.client_metadata, GraphSegmentClientMetadata)
    assert decoded.client_metadata.client_type == DatasetClientStoreType.GRAPH_SEGMENT
    assert decoded.client_metadata.benchmark_id == "bench-x"
    assert decoded.metadata.graph is not None
    assert decoded.metadata.graph.trace_ids == ["t-1", "t-2"]
    assert decoded.metadata.graph.prefix_cache_by_trace["t-1"]["n0"] == [3, 7]


def test_graph_client_store_plugin_resolves(tmp_path):
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    StoreClass = plugins.get_class(
        PluginType.DATASET_CLIENT_STORE, DatasetClientStoreType.GRAPH_SEGMENT
    )
    store = StoreClass(client_metadata=_notification(tmp_path).client_metadata)
    assert store.client_metadata.benchmark_id == "bench-x"


@pytest.mark.asyncio
async def test_graph_client_store_get_conversation_hard_errors(tmp_path):
    from aiperf.dataset.graph_client_store import GraphSegmentDatasetClientStore

    store = GraphSegmentDatasetClientStore(
        client_metadata=_notification(tmp_path).client_metadata
    )
    with pytest.raises(InvalidOperationError, match="no conversation store"):
        await store.get_conversation("t-1")
