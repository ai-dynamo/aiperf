# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MemoryMapDatasetBackingStore compress_only mode."""

import pytest
import zstandard

from aiperf.common.models import Conversation, MemoryMapClientMetadata
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetIndex,
)


def _make_conversation(session_id: str) -> Conversation:
    """Create a minimal conversation for testing."""
    return Conversation(session_id=session_id, turns=[])


class TestCompressOnlyInit:
    """Test compress_only backing store initialization."""

    def test_compress_only_flag_stored(self) -> None:
        store = MemoryMapDatasetBackingStore(
            benchmark_id="test-init", compress_only=True
        )
        assert store._compress_only is True

    def test_default_compress_only_is_false(self) -> None:
        store = MemoryMapDatasetBackingStore(benchmark_id="test-default")
        assert store._compress_only is False

    def test_compressed_paths_use_zst_extension(self) -> None:
        store = MemoryMapDatasetBackingStore(
            benchmark_id="test-paths", compress_only=True
        )
        assert store._compressed_data_path.suffix == ".zst"
        assert store._compressed_index_path.suffix == ".zst"


class TestCompressOnlyRoundTrip:
    """Test full write-finalize-read cycle in compress_only mode."""

    @pytest.mark.asyncio
    async def test_add_conversation_single_roundtrip_succeeds(
        self, tmp_path, monkeypatch
    ) -> None:
        """Write one conversation in compress_only mode, decompress, verify."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="rt-single", compress_only=True
        )
        await store.initialize()

        conv = _make_conversation("sess-1")
        await store.add_conversation("sess-1", conv)
        await store.finalize()

        metadata = store.get_client_metadata()
        assert metadata.compressed_data_file_path is not None
        assert metadata.compressed_data_file_path.exists()
        assert metadata.compressed_index_file_path is not None
        assert metadata.compressed_index_file_path.exists()
        assert metadata.compressed_size_bytes > 0

        dctx = zstandard.ZstdDecompressor()
        # Stream-compressed data doesn't include content size; use stream_reader
        with (
            open(metadata.compressed_data_file_path, "rb") as fh,
            dctx.stream_reader(fh) as reader,
        ):
            decompressed_data = reader.read()
        decompressed_index = dctx.decompress(
            metadata.compressed_index_file_path.read_bytes()
        )

        roundtrip_conv = Conversation.model_validate_json(decompressed_data)
        assert roundtrip_conv.session_id == "sess-1"

        index = MemoryMapDatasetIndex.model_validate_json(decompressed_index)
        assert index.conversation_ids == ["sess-1"]
        assert "sess-1" in index.offsets
        assert index.total_size == len(decompressed_data)

        await store.stop()

    @pytest.mark.asyncio
    async def test_add_conversations_multiple_roundtrip_succeeds(
        self, tmp_path, monkeypatch
    ) -> None:
        """Write multiple conversations in compress_only mode and verify index."""
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="rt-multi", compress_only=True
        )
        await store.initialize()

        ids = [f"sess-{i}" for i in range(5)]
        convs = {sid: _make_conversation(sid) for sid in ids}
        await store.add_conversations(convs)
        await store.finalize()

        metadata = store.get_client_metadata()
        assert metadata.conversation_count == 5

        dctx = zstandard.ZstdDecompressor()
        decompressed_index = dctx.decompress(
            metadata.compressed_index_file_path.read_bytes()
        )
        index = MemoryMapDatasetIndex.model_validate_json(decompressed_index)
        assert index.conversation_ids == ids

        await store.stop()


class TestCompressOnlyMetadata:
    """Test metadata produced by compress_only mode."""

    @pytest.mark.asyncio
    async def test_get_client_metadata_compress_only_includes_compressed_fields(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="meta-test", compress_only=True
        )
        await store.initialize()
        await store.add_conversation("s1", _make_conversation("s1"))
        await store.finalize()

        meta = store.get_client_metadata()
        assert isinstance(meta, MemoryMapClientMetadata)
        assert meta.compressed_data_file_path is not None
        assert meta.compressed_index_file_path is not None
        assert meta.compressed_size_bytes > 0
        assert meta.total_size_bytes > 0

        await store.stop()

    @pytest.mark.asyncio
    async def test_get_client_metadata_normal_mode_has_no_compressed_fields(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="meta-normal", compress_only=False
        )
        await store.initialize()
        await store.add_conversation("s1", _make_conversation("s1"))
        await store.finalize()

        meta = store.get_client_metadata()
        assert meta.compressed_data_file_path is None
        assert meta.compressed_index_file_path is None
        assert meta.compressed_size_bytes == 0

        await store.stop()


class TestCompressOnlyErrors:
    """Test error handling in compress_only mode."""

    @pytest.mark.asyncio
    async def test_add_conversation_after_finalize_raises_error(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="err-finalized", compress_only=True
        )
        await store.initialize()
        await store.add_conversation("s1", _make_conversation("s1"))
        await store.finalize()

        with pytest.raises(RuntimeError, match="Cannot add conversations"):
            await store.add_conversation("s2", _make_conversation("s2"))

        await store.stop()

    @pytest.mark.asyncio
    async def test_finalize_called_twice_raises_error(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="err-double", compress_only=True
        )
        await store.initialize()
        await store.add_conversation("s1", _make_conversation("s1"))
        await store.finalize()

        with pytest.raises(RuntimeError, match="finalize called twice"):
            await store.finalize()

        await store.stop()

    @pytest.mark.asyncio
    async def test_get_client_metadata_before_finalize_raises_error(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(
            benchmark_id="err-meta", compress_only=True
        )
        await store.initialize()

        with pytest.raises(
            RuntimeError, match="Cannot get metadata before finalization"
        ):
            store.get_client_metadata()

        await store.stop()


class TestCompressOnlyCleanup:
    """Test cleanup of compressed files."""

    @pytest.mark.asyncio
    async def test_stop_removes_compressed_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="cleanup", compress_only=True)
        await store.initialize()
        await store.add_conversation("s1", _make_conversation("s1"))
        await store.finalize()

        meta = store.get_client_metadata()
        assert meta.compressed_data_file_path.exists()
        assert meta.compressed_index_file_path.exists()

        await store.stop()

        assert not meta.compressed_data_file_path.exists()
        assert not meta.compressed_index_file_path.exists()
