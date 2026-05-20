# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the content-addressed mmap dataset cache.

Covers:
- ``compute_cache_key`` stability + collision sensitivity to inputs/settings/tokenizer
- ``populate`` + ``lookup`` round-trip with manifest version gating
- HIT / MISS file restoration to run dirs
- Corrupt and version-mismatched manifests treated as MISS
"""

from __future__ import annotations

import time
from pathlib import Path

import orjson
import pytest

from aiperf.dataset import mmap_cache


@pytest.fixture(autouse=True)
def _isolated_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the cache to a tmpdir so tests never touch ~/.cache."""
    from aiperf.common.environment import Environment

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", cache_root)
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)
    return cache_root


def _write_input_file(tmp_path: Path, content: bytes) -> Path:
    p = tmp_path / "input.jsonl"
    p.write_bytes(content)
    return p


def _stable_settings() -> dict[str, object]:
    return {"a": 1, "prompt": {"input_tokens": {"mean": 100}}}


def _stable_tokenizer() -> dict[str, object]:
    return {
        "name": "meta-llama/Llama-2-7b-hf",
        "revision": None,
        "trust_remote_code": False,
        "apply_chat_template": False,
    }


class TestComputeCacheKey:
    def test_key_is_deterministic_for_identical_inputs(self, tmp_path: Path) -> None:
        f = _write_input_file(tmp_path, b"hello world")
        k1 = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type="single_turn",
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        k2 = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type="single_turn",
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        assert k1 == k2
        assert len(k1) == 32

    def test_key_changes_when_input_bytes_change(self, tmp_path: Path) -> None:
        f1 = _write_input_file(tmp_path, b"alpha")
        f2 = tmp_path / "input2.jsonl"
        f2.write_bytes(b"beta")
        k1 = mmap_cache.compute_cache_key(
            input_file=f1,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        k2 = mmap_cache.compute_cache_key(
            input_file=f2,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        assert k1 != k2

    def test_key_changes_when_tokenizer_identity_changes(self, tmp_path: Path) -> None:
        f = _write_input_file(tmp_path, b"x")
        base = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        other = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity={**_stable_tokenizer(), "name": "different/model"},
            settings_payload=_stable_settings(),
        )
        chat_tmpl = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity={**_stable_tokenizer(), "apply_chat_template": True},
            settings_payload=_stable_settings(),
        )
        assert base != other
        assert base != chat_tmpl

    def test_key_changes_when_settings_change(self, tmp_path: Path) -> None:
        f = _write_input_file(tmp_path, b"x")
        base = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        bumped = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload={**_stable_settings(), "a": 2},
        )
        assert base != bumped

    def test_key_independent_of_settings_dict_key_order(self, tmp_path: Path) -> None:
        f = _write_input_file(tmp_path, b"x")
        a = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload={"a": 1, "b": 2},
        )
        b = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type=None,
            tokenizer_identity=_stable_tokenizer(),
            settings_payload={"b": 2, "a": 1},
        )
        assert a == b


def _populate_entry(
    cache_root: Path,
    *,
    cache_key: str,
    data_bytes: bytes = b"DATA",
    index_bytes: bytes = b"IDX",
    inputs_json: bytes | None = None,
    compressed: bool = False,
) -> Path:
    """Populate a cache entry through the public API and return the entry dir."""
    src_dir = cache_root.parent / "src"
    src_dir.mkdir(exist_ok=True)
    ext = ".dat.zst" if compressed else ".dat"
    data_p = src_dir / f"dataset{ext}"
    idx_p = src_dir / f"index{ext}"
    data_p.write_bytes(data_bytes)
    idx_p.write_bytes(index_bytes)

    inputs_p: Path | None = None
    if inputs_json is not None:
        inputs_p = src_dir / "inputs.json"
        inputs_p.write_bytes(inputs_json)

    manifest = mmap_cache.CacheManifest(
        cache_key=cache_key,
        created_at=time.time(),
        num_conversations=1,
        total_size_bytes=len(data_bytes),
        compressed=compressed,
        compressed_size_bytes=len(data_bytes) if compressed else 0,
        mmap_format="conversation",
        dataset_metadata_json='{"conversations": [], "sampling_strategy": "random"}',
    )
    out = mmap_cache.populate(
        cache_key=cache_key,
        run_data_path=data_p,
        run_index_path=idx_p,
        manifest=manifest,
        inputs_json_path=inputs_p,
    )
    assert out is not None
    return out


class TestLookupAndPopulate:
    def test_lookup_returns_none_when_no_entry(self) -> None:
        assert mmap_cache.lookup("deadbeef" * 4, compressed=False) is None

    def test_populate_then_lookup_roundtrip(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        entry_dir = _populate_entry(cache_root, cache_key="abc123")

        hit = mmap_cache.lookup("abc123", compressed=False)
        assert hit is not None
        assert hit.entry_dir == entry_dir
        assert hit.data_path.read_bytes() == b"DATA"
        assert hit.index_path.read_bytes() == b"IDX"
        assert hit.inputs_json_path is None
        assert hit.manifest.cache_key == "abc123"
        assert hit.manifest.num_conversations == 1

    def test_populate_includes_inputs_json_when_provided(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="withjson", inputs_json=b'{"data": []}')
        hit = mmap_cache.lookup("withjson", compressed=False)
        assert hit is not None
        assert hit.inputs_json_path is not None
        assert hit.inputs_json_path.read_bytes() == b'{"data": []}'
        assert hit.manifest.has_inputs_json is True

    def test_lookup_corrupt_manifest_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="corrupt")
        # Overwrite the manifest with garbage.
        (cache_root / "corrupt" / mmap_cache.MANIFEST_FILENAME).write_bytes(
            b"not json at all"
        )
        assert mmap_cache.lookup("corrupt", compressed=False) is None

    def test_lookup_missing_manifest_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="partial")
        (cache_root / "partial" / mmap_cache.MANIFEST_FILENAME).unlink()
        assert mmap_cache.lookup("partial", compressed=False) is None

    def test_lookup_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="oldver")
        manifest_path = cache_root / "oldver" / mmap_cache.MANIFEST_FILENAME
        raw = orjson.loads(manifest_path.read_bytes())
        raw["version"] = mmap_cache.MANIFEST_VERSION + 99
        manifest_path.write_bytes(orjson.dumps(raw))
        assert mmap_cache.lookup("oldver", compressed=False) is None

    def test_lookup_compressed_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="uncomp", compressed=False)
        # Same key requested as compressed -> MISS.
        assert mmap_cache.lookup("uncomp", compressed=True) is None

    def test_restore_copies_to_run_dir(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="restore")
        hit = mmap_cache.lookup("restore", compressed=False)
        assert hit is not None
        run_dir = tmp_path / "run_mmap"
        run_data = run_dir / "dataset.dat"
        run_index = run_dir / "index.dat"
        mmap_cache.restore_to_run_dir(hit, run_data, run_index)
        assert run_data.read_bytes() == b"DATA"
        assert run_index.read_bytes() == b"IDX"


class TestCacheToggle:
    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", False)
        assert mmap_cache.cache_enabled() is False


class TestAdoptExistingFiles:
    """Cover the cache HIT path on ``MemoryMapDatasetBackingStore``.

    The cache restores ``dataset.dat`` / ``index.dat`` into the run mmap dir,
    so the backing store must skip the writer + finalize and report the
    on-disk files as if it had produced them itself.
    """

    @pytest.mark.asyncio
    async def test_adopt_existing_files_finalizes_without_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore

        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="adopt-test")
        # Pre-create the on-disk files the cache HIT would have restored.
        store._data_path.parent.mkdir(parents=True, exist_ok=True)
        store._data_path.write_bytes(b"x" * 16)
        store._index_path.write_bytes(b"i" * 8)

        store.adopt_existing_files(
            session_ids=["sess-a", "sess-b"],
            total_size_bytes=16,
        )

        metadata = store.get_client_metadata()
        assert metadata.conversation_count == 2
        assert metadata.total_size_bytes == 16
        assert metadata.data_file_path == store._data_path
        assert metadata.index_file_path == store._index_path

    @pytest.mark.asyncio
    async def test_adopt_rejects_missing_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore

        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="adopt-missing")
        with pytest.raises(FileNotFoundError):
            store.adopt_existing_files(session_ids=[], total_size_bytes=0)

    @pytest.mark.asyncio
    async def test_adopt_rejects_when_already_finalized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore

        monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
        store = MemoryMapDatasetBackingStore(benchmark_id="adopt-twice")
        store._data_path.parent.mkdir(parents=True, exist_ok=True)
        store._data_path.write_bytes(b"x")
        store._index_path.write_bytes(b"i")
        store.adopt_existing_files(session_ids=["s"], total_size_bytes=1)
        with pytest.raises(RuntimeError):
            store.adopt_existing_files(session_ids=["s"], total_size_bytes=1)
