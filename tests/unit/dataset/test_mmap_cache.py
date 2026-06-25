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

import os
import time
from pathlib import Path

import orjson
import pytest

from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.dataset import mmap_cache
from aiperf.plugin.enums import PublicDatasetType


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
    def test_public_dataset_key_distinguishes_loader_metadata(self) -> None:
        qualitative = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(public_dataset=PublicDatasetType.SPEED_BENCH_QUALITATIVE),
        )
        coding = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(public_dataset=PublicDatasetType.SPEED_BENCH_CODING),
        )

        qualitative_key = mmap_cache.compute_cache_key_from_user_config(qualitative)
        coding_key = mmap_cache.compute_cache_key_from_user_config(coding)

        assert qualitative_key is not None
        assert coding_key is not None
        assert qualitative_key != coding_key

    def test_key_changes_when_weka_live_assistant_setting_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        cfg = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
            ),
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_LIVE_ASSISTANT_RESPONSES", False)
        pre_canned_key = mmap_cache.compute_cache_key_from_user_config(cfg)
        monkeypatch.setattr(Environment.DATASET, "WEKA_LIVE_ASSISTANT_RESPONSES", True)
        live_key = mmap_cache.compute_cache_key_from_user_config(cfg)

        assert pre_canned_key is not None
        assert live_key is not None
        assert pre_canned_key != live_key

    def test_key_changes_when_weka_split_flattened_agents_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        cfg = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
            ),
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", True)
        split_key = mmap_cache.compute_cache_key_from_user_config(cfg)
        monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False)
        legacy_key = mmap_cache.compute_cache_key_from_user_config(cfg)

        assert split_key is not None
        assert legacy_key is not None
        # The flag changes loader output (split vs legacy single-stream), so
        # a warm cache from one mode must never serve the other.
        assert split_key != legacy_key

    def test_key_changes_with_load_time_timing_knobs(self) -> None:
        cfg = UserConfig(
            endpoint=EndpointConfig(model_names=["test-model"]),
            input=InputConfig(
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
            ),
        )
        base = mmap_cache.compute_cache_key_from_user_config(cfg)

        cfg_ignore = cfg.model_copy(deep=True)
        cfg_ignore.input.ignore_trace_delays = True
        cfg_think = cfg.model_copy(deep=True)
        cfg_think.input.use_think_time_only = True
        cfg_cap = cfg.model_copy(deep=True)
        cfg_cap.loadgen.inter_turn_delay_cap_seconds = 60.0
        cfg_warp = cfg.model_copy(deep=True)
        cfg_warp.loadgen.trace_idle_gap_cap_seconds = 60.0

        keys = [
            base,
            mmap_cache.compute_cache_key_from_user_config(cfg_ignore),
            mmap_cache.compute_cache_key_from_user_config(cfg_think),
            mmap_cache.compute_cache_key_from_user_config(cfg_cap),
            mmap_cache.compute_cache_key_from_user_config(cfg_warp),
        ]
        assert all(k is not None for k in keys)
        # These knobs are applied at LOAD time (baked into cached Turn
        # timestamps/delays), so each must produce a distinct cache key.
        assert len(set(keys)) == len(keys), keys

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

    def test_populate_ignores_inputs_json_when_provided(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        entry_dir = _populate_entry(
            cache_root, cache_key="withjson", inputs_json=b'{"data": []}'
        )
        hit = mmap_cache.lookup("withjson", compressed=False)
        assert hit is not None
        assert hit.inputs_json_path is None
        assert hit.manifest.has_inputs_json is False
        assert not (entry_dir / mmap_cache.INPUTS_JSON_FILENAME).exists()

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

    def test_lookup_rejects_pre_overlap_frontier_manifest(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="pre-overlap-frontier")
        manifest_path = (
            cache_root / "pre-overlap-frontier" / mmap_cache.MANIFEST_FILENAME
        )
        raw = orjson.loads(manifest_path.read_bytes())
        raw["version"] = 20
        manifest_path.write_bytes(orjson.dumps(raw))

        assert mmap_cache.MANIFEST_VERSION == 21
        assert mmap_cache.lookup("pre-overlap-frontier", compressed=False) is None

    def test_lookup_compressed_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="uncomp", compressed=False)
        # Same key requested as compressed -> MISS.
        assert mmap_cache.lookup("uncomp", compressed=True) is None

    def test_restore_hardlinks_to_run_dir(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="restore")
        hit = mmap_cache.lookup("restore", compressed=False)
        assert hit is not None
        run_dir = tmp_path / "run_mmap"
        run_data = run_dir / "dataset.dat"
        run_index = run_dir / "index.dat"
        with caplog.at_level("INFO", logger="aiperf.dataset.mmap_cache"):
            mmap_cache.restore_to_run_dir(hit, run_data, run_index)
        assert run_data.read_bytes() == b"DATA"
        assert run_index.read_bytes() == b"IDX"
        assert os.stat(run_data).st_ino == os.stat(hit.data_path).st_ino
        assert os.stat(run_index).st_ino == os.stat(hit.index_path).st_ino
        assert "Restored mmap cache file dataset.dat via hardlink" in caplog.text
        assert "Restored mmap cache file index.dat via hardlink" in caplog.text
        assert "Restored mmap cache files in" in caplog.text

    def test_restore_falls_back_to_copy_when_hardlink_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="restore-copy")
        hit = mmap_cache.lookup("restore-copy", compressed=False)
        assert hit is not None
        run_dir = tmp_path / "run_mmap"
        run_data = run_dir / "dataset.dat"
        run_index = run_dir / "index.dat"

        def raise_cross_device(_src: Path, _dst: Path) -> None:
            raise OSError("cross-device link")

        monkeypatch.setattr(mmap_cache.os, "link", raise_cross_device)

        mmap_cache.restore_to_run_dir(hit, run_data, run_index)

        assert run_data.read_bytes() == b"DATA"
        assert run_index.read_bytes() == b"IDX"
        assert os.stat(run_data).st_ino != os.stat(hit.data_path).st_ino
        assert os.stat(run_index).st_ino != os.stat(hit.index_path).st_ino

    @pytest.mark.asyncio
    async def test_cleanup_unlinks_run_hardlinks_without_removing_cache_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment
        from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore

        monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path / "mmap")
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="cleanup")
        hit = mmap_cache.lookup("cleanup", compressed=False)
        assert hit is not None
        store = MemoryMapDatasetBackingStore(benchmark_id="cleanup")
        run_data = tmp_path / "mmap" / "aiperf_mmap_cleanup" / "dataset.dat"
        run_index = tmp_path / "mmap" / "aiperf_mmap_cleanup" / "index.dat"

        mmap_cache.restore_to_run_dir(hit, run_data, run_index)
        assert os.stat(run_data).st_ino == os.stat(hit.data_path).st_ino
        assert os.stat(run_index).st_ino == os.stat(hit.index_path).st_ino

        store.adopt_existing_files(session_ids=["s1"], total_size_bytes=4)
        await store._cleanup()

        assert not run_data.exists()
        assert not run_index.exists()
        assert hit.data_path.read_bytes() == b"DATA"
        assert hit.index_path.read_bytes() == b"IDX"


class TestCacheToggle:
    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", False)
        assert mmap_cache.cache_enabled() is False


class TestAcquireCacheLock:
    """Coverage for :func:`mmap_cache.acquire_cache_lock` populate gate."""

    @pytest.mark.asyncio
    async def test_serializes_concurrent_acquires(self) -> None:
        """Five concurrent contenders on the same key never overlap inside."""
        import asyncio
        import time

        events: list[tuple[str, float]] = []
        t0 = time.monotonic()

        async def hold(name: str, dwell: float) -> None:
            async with mmap_cache.acquire_cache_lock("k", timeout=10.0):
                events.append((f"{name}:enter", time.monotonic() - t0))
                await asyncio.sleep(dwell)
                events.append((f"{name}:exit", time.monotonic() - t0))

        await asyncio.gather(*(hold(n, 0.05) for n in "ABCDE"))

        ordered = sorted(events, key=lambda e: e[1])
        balance = 0
        for tag, _ in ordered:
            balance += 1 if "enter" in tag else -1
            assert balance <= 1, f"overlap at {tag}: {ordered}"

    @pytest.mark.asyncio
    async def test_independent_keys_dont_serialize(self) -> None:
        """Two contenders on different keys MAY run in parallel."""
        import asyncio
        import time

        events: list[str] = []

        async def hold(key: str) -> None:
            async with mmap_cache.acquire_cache_lock(key, timeout=5.0):
                events.append(f"{key}:enter")
                await asyncio.sleep(0.2)
                events.append(f"{key}:exit")

        t0 = time.monotonic()
        await asyncio.gather(hold("alpha"), hold("beta"))
        elapsed = time.monotonic() - t0
        # Sequential would be ~0.4s; parallel is ~0.2s. Allow generous
        # scheduler slop but assert clearly under fully-serialized timing.
        assert elapsed < 0.35, (
            f"distinct-key acquires unexpectedly serialized: "
            f"elapsed={elapsed:.3f}s, events={events}"
        )

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        """Holder beyond timeout causes the waiter to raise filelock.Timeout."""
        import asyncio

        from filelock import Timeout as FileLockTimeout

        holder_acquired = asyncio.Event()
        holder_release = asyncio.Event()

        async def holder() -> None:
            async with mmap_cache.acquire_cache_lock("k", timeout=5.0):
                holder_acquired.set()
                await holder_release.wait()

        async def waiter() -> None:
            await holder_acquired.wait()
            with pytest.raises(FileLockTimeout):
                async with mmap_cache.acquire_cache_lock("k", timeout=0.5):
                    pass

        holder_task = asyncio.create_task(holder())
        try:
            await asyncio.wait_for(waiter(), timeout=5.0)
        finally:
            holder_release.set()
            await holder_task
