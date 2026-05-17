# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for MemoryMapFormat.PAYLOAD_BYTES storage."""

import threading

import orjson
import pytest

from aiperf.common.enums import ConversationBranchMode, MemoryMapFormat
from aiperf.common.models import Conversation, ConversationBranchInfo, Turn
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
)


@pytest.mark.asyncio
async def test_payload_bytes_roundtrip_preserves_per_turn_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payloads = [
        {"messages": [{"role": "user", "content": "turn 0"}], "temperature": 0.1},
        {"messages": [{"role": "user", "content": "turn 1"}], "max_tokens": 5},
    ]
    conv = Conversation(
        session_id="s1",
        turns=[Turn(role="user", raw_payload=p) for p in payloads],
    )

    store = MemoryMapDatasetBackingStore(
        benchmark_id="payload_bytes_t1", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        metadata = store.get_client_metadata()
        assert metadata.format == MemoryMapFormat.PAYLOAD_BYTES

        client = MemoryMapDatasetClient(
            metadata.data_file_path, metadata.index_file_path
        )
        try:
            for i, expected in enumerate(payloads):
                got = client.get_payload_bytes("s1", i)
                assert orjson.loads(got) == expected
            with pytest.raises(IndexError):
                client.get_payload_bytes("s1", 99)
            with pytest.raises(KeyError):
                client.get_payload_bytes("nope", 0)
            with pytest.raises(RuntimeError, match="PAYLOAD_BYTES"):
                client.get_conversation("s1")
        finally:
            client.close()
    finally:
        await store.stop()


class _InterleavingMmap:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.first_seek_done = threading.Event()
        self.second_seek_done = threading.Event()

    def seek(self, offset: int) -> None:
        self._pos = offset
        if offset == 0:
            self.first_seek_done.set()
            assert self.second_seek_done.wait(timeout=5)
        else:
            self.second_seek_done.set()

    def read(self, size: int) -> bytes:
        start = self._pos
        self._pos = start + size
        return self._data[start : start + size]

    def __getitem__(self, key: slice) -> bytes:
        return self._data[key]

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_payload_bytes_concurrent_reads_do_not_share_cursor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    payload_a = {"id": "a"}
    payload_b = {"id": "b"}
    convs = {
        "a": Conversation(
            session_id="a", turns=[Turn(role="user", raw_payload=payload_a)]
        ),
        "b": Conversation(
            session_id="b", turns=[Turn(role="user", raw_payload=payload_b)]
        ),
    }
    store = MemoryMapDatasetBackingStore(
        benchmark_id="payload_bytes_cursor", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        await store.add_conversations(convs)
        await store.finalize()
        metadata = store.get_client_metadata()
        client = MemoryMapDatasetClient(
            metadata.data_file_path, metadata.index_file_path
        )
        try:
            raw_data = client.data_mmap[:]
            client.data_mmap = _InterleavingMmap(raw_data)
            results: dict[str, bytes] = {}

            def read_a() -> None:
                results["a"] = client.get_payload_bytes("a", 0)

            def read_b() -> None:
                results["b"] = client.get_payload_bytes("b", 0)

            thread_a = threading.Thread(target=read_a)
            thread_b = threading.Thread(target=read_b)
            thread_a.start()
            thread_b.start()
            thread_a.join(timeout=5)
            thread_b.join(timeout=5)

            assert orjson.loads(results["a"]) == payload_a
            assert orjson.loads(results["b"]) == payload_b
        finally:
            client.close()
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_payload_bytes_rejects_missing_raw_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(session_id="s1", turns=[Turn(role="user")])
    store = MemoryMapDatasetBackingStore(
        benchmark_id="payload_bytes_t2", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    try:
        with pytest.raises(ValueError, match="raw_payload"):
            await store.add_conversation("s1", conv)
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_conversation_format_unaffected(tmp_path, monkeypatch):
    """The default CONVERSATION format keeps existing semantics."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    conv = Conversation(session_id="s1", turns=[Turn(role="user")])
    store = MemoryMapDatasetBackingStore(benchmark_id="payload_bytes_t3")
    await store.initialize()
    try:
        await store.add_conversation("s1", conv)
        await store.finalize()
        metadata = store.get_client_metadata()
        assert metadata.format == MemoryMapFormat.CONVERSATION

        client = MemoryMapDatasetClient(
            metadata.data_file_path, metadata.index_file_path
        )
        try:
            got = client.get_conversation("s1")
            assert got.session_id == "s1"
            with pytest.raises(RuntimeError, match="PAYLOAD_BYTES"):
                client.get_payload_bytes("s1", 0)
        finally:
            client.close()
    finally:
        await store.stop()


# ---------------------------------------------------------------------------
# DatasetManager._select_mmap_format guards (Task 4)
# ---------------------------------------------------------------------------


def _conv_with_raw_payload(session_id: str, n_turns: int = 1) -> Conversation:
    """Build a Conversation where every turn has raw_payload set."""
    return Conversation(
        session_id=session_id,
        turns=[
            Turn(
                role="user",
                raw_payload={
                    "messages": [{"role": "user", "content": f"{session_id}#{i}"}]
                },
            )
            for i in range(n_turns)
        ],
    )


def test_select_mmap_format_empty_returns_conversation():
    """Empty conversation list falls back to CONVERSATION."""
    assert DatasetManager._select_mmap_format([]) == MemoryMapFormat.CONVERSATION


def test_select_mmap_format_picks_payload_bytes_when_clean():
    """Every turn has raw_payload, no FORK branches -> PAYLOAD_BYTES."""
    conversations = [
        _conv_with_raw_payload("s1"),
        _conv_with_raw_payload("s2", n_turns=2),
    ]
    assert (
        DatasetManager._select_mmap_format(conversations)
        == MemoryMapFormat.PAYLOAD_BYTES
    )


def test_select_mmap_format_can_disable_payload_bytes_for_multipart():
    """Multipart endpoints need CONVERSATION format so raw_payload dicts reach FormData."""
    conversations = [_conv_with_raw_payload("s1")]

    assert (
        DatasetManager._select_mmap_format(conversations, allow_payload_bytes=False)
        == MemoryMapFormat.CONVERSATION
    )


def test_select_mmap_format_falls_back_to_conversation_without_raw_payload():
    """A single turn missing raw_payload -> CONVERSATION, even with FORK.

    The session_manager's existing FORK + raw_payload guard handles the
    CONVERSATION path, so format selection just needs to step out of the way.
    """
    conversations = [
        Conversation(
            session_id="s1",
            turns=[Turn(role="user", raw_payload={"messages": []}), Turn(role="user")],
            branches=[
                ConversationBranchInfo(
                    branch_id="s1:0",
                    child_conversation_ids=["c1"],
                    mode=ConversationBranchMode.FORK,
                )
            ],
        )
    ]
    # Must NOT raise — falls back cleanly because not all turns have raw_payload.
    assert (
        DatasetManager._select_mmap_format(conversations)
        == MemoryMapFormat.CONVERSATION
    )


def test_select_mmap_format_refuses_fork_lists_offending_branches():
    """Error message names the conversations:branches that triggered the refusal."""
    conversations = [
        Conversation(
            session_id="conv_a",
            turns=[Turn(role="user", raw_payload={"messages": []})],
            branches=[
                ConversationBranchInfo(
                    branch_id="conv_a:0",
                    child_conversation_ids=["c1"],
                    mode=ConversationBranchMode.FORK,
                )
            ],
        ),
        Conversation(
            session_id="conv_b",
            turns=[Turn(role="user", raw_payload={"messages": []})],
            branches=[
                ConversationBranchInfo(
                    branch_id="conv_b:0",
                    child_conversation_ids=["c2"],
                    mode=ConversationBranchMode.FORK,
                )
            ],
        ),
    ]
    with pytest.raises(ValueError, match="FORK-mode branches") as exc_info:
        DatasetManager._select_mmap_format(conversations)
    assert "conv_a:conv_a:0" in str(exc_info.value)
    assert "conv_b:conv_b:0" in str(exc_info.value)


def test_select_mmap_format_allows_spawn_branches_with_raw_payload():
    """SPAWN-mode branches (fresh context, not seeded from parent turn_list)
    do NOT conflict with raw_payload — only FORK does."""
    spawn_branch = ConversationBranchInfo(
        branch_id="s1:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    conversations = [
        Conversation(
            session_id="s1",
            turns=[Turn(role="user", raw_payload={"messages": []})],
            branches=[spawn_branch],
        )
    ]
    assert (
        DatasetManager._select_mmap_format(conversations)
        == MemoryMapFormat.PAYLOAD_BYTES
    )
