# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor

import orjson
import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
)


def _make_raw_conversation(
    session_id: str,
    payloads: list[dict],
) -> Conversation:
    """Create a conversation where every turn has a raw_payload."""
    turns = [Turn(role="user", raw_payload=p) for p in payloads]
    return Conversation(session_id=session_id, turns=turns)


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
async def test_concurrent_conversation_reads_do_not_share_mmap_position(
    tmp_path, monkeypatch
):
    """Executor reads must use offsets directly instead of shared seek state."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_concurrent_reads")
    await store.initialize()
    for conversation_id, content in (("conv-a", "a" * 4096), ("conv-b", "b" * 4096)):
        await store.add_conversation(
            conversation_id,
            Conversation(
                session_id=conversation_id,
                turns=[Turn(role="user", content=content)],
            ),
        )
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(metadata.data_file_path, metadata.index_file_path)
    base_mmap = client.data_mmap

    class InterleavingMmap:
        def __init__(self):
            self.position = 0
            self.barrier = threading.Barrier(2)

        def __getitem__(self, key):
            return base_mmap[key]

        def seek(self, offset):
            self.position = offset
            self.barrier.wait(timeout=5)

        def read(self, size):
            start = self.position
            self.position += size
            return base_mmap[start : start + size]

        def close(self):
            base_mmap.close()

    client.data_mmap = InterleavingMmap()
    with ThreadPoolExecutor(max_workers=2) as executor:
        conversations = list(
            executor.map(client.get_conversation, ("conv-a", "conv-b"))
        )

    assert [conversation.session_id for conversation in conversations] == [
        "conv-a",
        "conv-b",
    ]
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


@pytest.mark.asyncio
async def test_get_conversation_raises_for_payload_bytes_format(tmp_path, monkeypatch):
    """PAYLOAD_BYTES stores cannot serve full Conversations."""
    from aiperf.common.exceptions import MemoryMapSerializationError

    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_conv_raises", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    conv = _make_raw_conversation("conv-1", [{"messages": []}])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    with pytest.raises(MemoryMapSerializationError, match="payload_bytes"):
        client.get_conversation("conv-1")

    client.close()
    await store.stop()


@pytest.mark.asyncio
async def test_client_store_get_payload_bytes_requires_initialize(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_uninit", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    conv = _make_raw_conversation("conv-1", [{"messages": []}])
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    client_store = MemoryMapDatasetClientStore(
        client_metadata=store.get_client_metadata()
    )
    # initialize() intentionally NOT called.
    with pytest.raises(RuntimeError, match="not initialized"):
        await client_store.get_payload_bytes("conv-1", 0)

    await store.stop()


@pytest.mark.asyncio
async def test_adopt_existing_files_rejects_double_adoption(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_adopt_twice")
    store._data_path.parent.mkdir(parents=True, exist_ok=True)
    store._data_path.write_bytes(b"DATA")
    store._index_path.write_bytes(b"IDX")

    store.adopt_existing_files(session_ids=["s1"], total_size_bytes=4)
    with pytest.raises(RuntimeError, match="already-finalized"):
        store.adopt_existing_files(session_ids=["s1"], total_size_bytes=4)

    await store.stop()


@pytest.mark.asyncio
async def test_adopt_existing_files_requires_files_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(benchmark_id="test_adopt_missing")

    with pytest.raises(FileNotFoundError, match="requires both files"):
        store.adopt_existing_files(session_ids=["s1"], total_size_bytes=4)


@pytest.mark.asyncio
async def test_adopt_existing_files_compress_only_uses_zst_paths(tmp_path, monkeypatch):
    """Cache HIT restore writes .zst only; adopt must not require uncompressed paths."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_adopt_zst", compress_only=True
    )
    store._compressed_data_path.parent.mkdir(parents=True, exist_ok=True)
    store._compressed_data_path.write_bytes(b"ZDATA")
    store._compressed_index_path.write_bytes(b"ZIDX")
    assert not store._data_path.exists()
    assert not store._index_path.exists()

    store.adopt_existing_files(
        session_ids=["s1"], total_size_bytes=100, compressed_size_bytes=5
    )
    assert store._finalized is True
    assert store._compressed_size == 5

    await store.stop()


@pytest.mark.asyncio
async def test_adopt_existing_files_compress_only_missing_zst_raises(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_adopt_zst_missing", compress_only=True
    )
    with pytest.raises(FileNotFoundError, match=r"dataset\.dat\.zst"):
        store.adopt_existing_files(session_ids=["s1"], total_size_bytes=4)


@pytest.mark.asyncio
async def test_payload_mmap_persists_turn_scalars(tmp_path, monkeypatch):
    """PAYLOAD_BYTES index must round-trip max_tokens and timestamp.

    Turn scalars live outside the wire body for some loaders (e.g. mooncake
    ``output_length`` / ``timestamp``). Persisting them on PayloadOffset keeps
    OSL-mismatch and schedule-lag metrics alive on the verbatim path.
    """
    from aiperf.dataset.memory_map_utils import (
        PayloadOffset,
        max_tokens_from_wire_payload,
        turn_from_payload_turn,
    )

    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))

    store = MemoryMapDatasetBackingStore(
        benchmark_id="test_payload_scalars", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()

    # Wire body omits max_tokens; scalar comes only from Turn.max_tokens.
    payload = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    conv = Conversation(
        session_id="conv-1",
        turns=[
            Turn(
                role="user",
                raw_payload=payload,
                max_tokens=128,
                timestamp=42.5,
            )
        ],
    )
    await store.add_conversation("conv-1", conv)
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(
        metadata.data_file_path,
        metadata.index_file_path,
    )

    entry = client.get_payload_turn("conv-1", 0)
    assert entry is not None
    assert entry.max_tokens == 128
    assert entry.timestamp == 42.5
    assert orjson.loads(entry.payload_bytes) == payload

    turn = turn_from_payload_turn(entry)
    assert turn.max_tokens == 128
    assert turn.timestamp == 42.5
    assert turn.raw_payload == payload

    # Wire-JSON fallback recovers max_tokens when index scalars are absent
    # (legacy PayloadOffset with only offset/size).
    legacy = PayloadOffset(offset=0, size=0)
    assert legacy.max_tokens is None
    assert (
        max_tokens_from_wire_payload({"max_completion_tokens": 16, "messages": []})
        == 16
    )
    assert max_tokens_from_wire_payload({"max_output_tokens": 8}) == 8
    assert max_tokens_from_wire_payload({"max_tokens": 0}) is None

    client.close()
    await store.stop()
