# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
)


def _make_raw_conversation(session_id: str, payloads: list[dict]) -> Conversation:
    """Create a conversation where every turn has a raw_payload."""
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user", raw_payload=p) for p in payloads],
    )


@pytest.mark.asyncio
async def test_payload_mmap_round_trip(tmp_path, monkeypatch):
    """Test writing and reading payload bytes through the mmap backing store."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_payload", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()

    payload_1 = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}
    payload_2 = {"messages": [{"role": "user", "content": "World"}], "model": "gpt-4"}

    conv1 = _make_raw_conversation("conv-1", [payload_1, payload_2])

    await store.add_conversation("conv-1", conv1)
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    # Check payload bytes for conv-1
    pb0 = client.get_payload_bytes("conv-1", 0)
    assert pb0 is not None
    assert orjson.loads(pb0) == payload_1

    pb1 = client.get_payload_bytes("conv-1", 1)
    assert pb1 is not None
    assert orjson.loads(pb1) == payload_2

    # Out of range
    assert client.get_payload_bytes("conv-1", 99) is None

    # Non-existent conversation
    assert client.get_payload_bytes("conv-999", 0) is None

    client.close()
    await store.stop()


@pytest.mark.asyncio
async def test_conversation_format_returns_none_for_payload_bytes(
    tmp_path, monkeypatch
):
    """When format is CONVERSATION, get_payload_bytes returns None."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_no_payload")
    await store.initialize()

    conv = Conversation(session_id="conv-1", turns=[Turn(role="user")])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    assert client.get_payload_bytes("conv-1", 0) is None
    # Conversation format still works
    conversation = client.get_conversation("conv-1")
    assert conversation.session_id == "conv-1"

    client.close()
    await store.stop()


@pytest.mark.asyncio
async def test_client_store_get_payload_bytes(tmp_path, monkeypatch):
    """Test MemoryMapDatasetClientStore.get_payload_bytes async wrapper."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_client_payload", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()

    payload = {"messages": [{"role": "user", "content": "test"}]}
    conv = _make_raw_conversation("conv-1", [payload])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    client_store = MemoryMapDatasetClientStore(client_metadata=metadata)
    await client_store.initialize()

    result = await client_store.get_payload_bytes("conv-1", 0)
    assert result is not None
    assert orjson.loads(result) == payload

    result_none = await client_store.get_payload_bytes("conv-1", 99)
    assert result_none is None

    await client_store.stop()
    await store.stop()


@pytest.mark.asyncio
async def test_payload_bytes_format_multi_conversation(tmp_path, monkeypatch):
    """Test multiple conversations in payload_bytes format."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_multi", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()

    p1 = {"messages": [{"role": "user", "content": "a"}]}
    p2 = {"messages": [{"role": "user", "content": "b"}]}
    p3 = {"messages": [{"role": "user", "content": "c"}]}

    conv1 = _make_raw_conversation("conv-1", [p1, p2])
    conv2 = _make_raw_conversation("conv-2", [p3])

    await store.add_conversation("conv-1", conv1)
    await store.add_conversation("conv-2", conv2)
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    assert client.index.format == MemoryMapFormat.PAYLOAD_BYTES
    assert orjson.loads(client.get_payload_bytes("conv-1", 0)) == p1
    assert orjson.loads(client.get_payload_bytes("conv-1", 1)) == p2
    assert orjson.loads(client.get_payload_bytes("conv-2", 0)) == p3

    client.close()
    await store.stop()
