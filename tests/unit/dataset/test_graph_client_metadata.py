# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph runs broadcast a graph facet plus graph-typed ``client_metadata`` instead of stub conversations, so the discriminated-union wire round-trip and the client-store plugin registration must both hold."""

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
    """Build a graph-flavored ``DatasetConfiguredNotification``: empty conversations, a populated graph facet, and graph-segment client metadata."""
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


def test_graph_client_metadata_wire_round_trip(tmp_path: Path) -> None:
    """A JSON round-trip preserves the ``GraphSegmentClientMetadata`` union member, its discriminator, and the graph facet's trace ids and prefix-cache map."""
    msg = _notification(tmp_path)
    decoded = DatasetConfiguredNotification.model_validate(msg.model_dump(mode="json"))
    assert isinstance(decoded.client_metadata, GraphSegmentClientMetadata)
    assert decoded.client_metadata.client_type == DatasetClientStoreType.GRAPH_SEGMENT
    assert decoded.client_metadata.benchmark_id == "bench-x"
    assert decoded.metadata.graph is not None
    assert decoded.metadata.graph.trace_ids == ["t-1", "t-2"]
    assert decoded.metadata.graph.prefix_cache_by_trace["t-1"]["n0"] == [3, 7]


def test_graph_client_store_plugin_resolves(tmp_path: Path) -> None:
    """The ``GRAPH_SEGMENT`` client-store type is registered and constructs from graph client metadata."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    StoreClass = plugins.get_class(
        PluginType.DATASET_CLIENT_STORE, DatasetClientStoreType.GRAPH_SEGMENT
    )
    store = StoreClass(client_metadata=_notification(tmp_path).client_metadata)
    assert store.client_metadata.benchmark_id == "bench-x"


@pytest.mark.asyncio
async def test_graph_client_store_get_conversation_hard_errors(tmp_path: Path) -> None:
    """The graph client store holds no conversations, so ``get_conversation`` must fail loud instead of returning an empty stub."""
    from aiperf.dataset.graph_client_store import GraphSegmentDatasetClientStore

    store = GraphSegmentDatasetClientStore(
        client_metadata=_notification(tmp_path).client_metadata
    )
    with pytest.raises(InvalidOperationError, match="no conversation store"):
        await store.get_conversation("t-1")
