# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for DatasetManager mmap cache HIT/MISS pathway.

Verifies that:
- A second run with byte-identical inputs serves from cache (composer + tokenizer skipped).
- A first run populates the cache.
- Tokenizer changes invalidate the cache.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset import mmap_cache
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.conftest import make_run_from_cli


def _write_legacy_cache_entry_with_inputs_json(
    cache_key: str, tmp_path: Path
) -> mmap_cache.CacheHit:
    import orjson

    entry_dir = mmap_cache.cache_dir() / cache_key
    entry_dir.mkdir(parents=True)
    (entry_dir / "dataset.dat").write_bytes(b"DATA")
    (entry_dir / "index.dat").write_bytes(b"IDX")
    (entry_dir / mmap_cache.INPUTS_JSON_FILENAME).write_bytes(b'{"requests": []}')
    manifest = mmap_cache.CacheManifest(
        cache_key=cache_key,
        created_at=time.time(),
        num_conversations=0,
        total_size_bytes=4,
        compressed=False,
        compressed_size_bytes=0,
        mmap_format="conversation",
        dataset_metadata_json='{"conversations": [], "sampling_strategy": "random"}',
        has_inputs_json=True,
    )
    (entry_dir / mmap_cache.MANIFEST_FILENAME).write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"))
    )
    hit = mmap_cache.lookup(cache_key, compressed=False)
    assert hit is not None
    assert hit.inputs_json_path is not None
    return hit


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin cache to tmp + isolate the run mmap dir."""
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_DIR", cache_root)
    monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", True)
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path / "mmap")


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Patch Tokenizer.from_pretrained so we can count tokenizer loads."""
    with patch("aiperf.common.tokenizer.Tokenizer.from_pretrained") as mock:
        mock.return_value = mock_tokenizer_cls.from_pretrained("test-model")
        yield mock


def _write_trace(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    entries = [
        '{"session_id": "s1", "timestamp": 0, "input_length": 8, "output_length": 4}\n',
        '{"session_id": "s2", "timestamp": 100, "input_length": 8, "output_length": 4}\n',
    ]
    p.write_bytes("".join(entries).encode())
    return p


def _make_run(
    *, file_path: Path, benchmark_id: str, tokenizer_name: str = "test-tokenizer"
) -> BenchmarkRun:
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            tokenizer_name=tokenizer_name,
            input_file=str(file_path),
            custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
        )
    )


async def _run_configure(run: BenchmarkRun) -> DatasetManager:
    dataset_manager = DatasetManager(run=run, service_id="dm-test")
    await dataset_manager.initialize()
    dataset_manager.publish = AsyncMock()
    await dataset_manager._profile_configure_command(
        ProfileConfigureCommand(service_id="dm-test")
    )
    return dataset_manager


class TestDatasetManagerCacheRoundtrip:
    @pytest.mark.asyncio
    async def test_first_run_misses_then_populates_cache(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="run-1")

        # Lookup should MISS before run.
        key = mmap_cache.compute_cache_key_from_run(run)
        assert key is not None
        assert mmap_cache.lookup(key, compressed=False) is None

        dm = await _run_configure(run)
        await dm.stop()

        # After run, the cache MUST have the entry.
        hit = mmap_cache.lookup(key, compressed=False)
        assert hit is not None
        assert hit.manifest.cache_key == key
        assert hit.data_path.exists()
        assert hit.index_path.exists()

    @pytest.mark.asyncio
    async def test_second_run_hits_cache_and_skips_tokenizer(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        trace = _write_trace(tmp_path)

        # Run 1: populate the cache.
        run1 = _make_run(file_path=trace, benchmark_id="run-1")
        dm1 = await _run_configure(run1)
        await dm1.stop()
        assert mock_tokenizer.call_count >= 1
        first_call_count = mock_tokenizer.call_count

        # Run 2: identical config should HIT and skip the tokenizer entirely.
        run2 = _make_run(file_path=trace, benchmark_id="run-2")
        dm2 = await _run_configure(run2)

        # Tokenizer.from_pretrained must NOT have been called again.
        assert mock_tokenizer.call_count == first_call_count, (
            "Cache HIT must skip tokenizer load"
        )
        # The HIT path still publishes a DatasetConfiguredNotification.
        from aiperf.common.messages import DatasetConfiguredNotification

        published = [c.args[0] for c in dm2.publish.call_args_list]  # type: ignore[union-attr]
        notifs = [m for m in published if isinstance(m, DatasetConfiguredNotification)]
        assert len(notifs) == 1
        assert dm2._cache_hit_used is True
        await dm2.stop()

    @pytest.mark.asyncio
    async def test_tokenizer_change_invalidates_cache(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        trace = _write_trace(tmp_path)
        run_a = _make_run(file_path=trace, benchmark_id="run-a", tokenizer_name="t1")
        run_b = _make_run(file_path=trace, benchmark_id="run-b", tokenizer_name="t2")
        key_a = mmap_cache.compute_cache_key_from_run(run_a)
        key_b = mmap_cache.compute_cache_key_from_run(run_b)
        assert key_a is not None and key_b is not None
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_cache_disabled_skips_lookup(
        self, tmp_path: Path, mock_tokenizer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Environment.DATASET, "MMAP_CACHE_ENABLED", False)
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="dis-1")

        dm = await _run_configure(run)
        await dm.stop()
        # Even with caching disabled, the run completes successfully.
        # No populate happens, so the cache dir stays empty.
        cache_root = mmap_cache.cache_dir()
        assert not cache_root.exists() or not any(cache_root.iterdir())

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_restore_inputs_json(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="hit-no-inputs")
        key = mmap_cache.compute_cache_key_from_run(run)
        assert key is not None
        _write_legacy_cache_entry_with_inputs_json(key, tmp_path)

        target = run.cfg.artifacts.dir / "inputs.json"
        if target.exists():
            target.unlink()

        with patch.object(
            DatasetManager,
            "_configure_dataset_client_and_free_memory",
            new_callable=AsyncMock,
        ):
            dm = await _run_configure(run)

        assert dm._cache_hit_used is True
        assert not target.exists()
        await dm.stop()


class TestDatasetManagerCacheEdgeCases:
    """Error paths and guards around the cache lookup / populate pipeline."""

    @pytest.mark.asyncio
    async def test_hit_under_lock_after_stale_initial_miss(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        """A waiter whose pre-lock lookup missed must re-check under the lock
        and adopt the entry the winner populated meanwhile."""
        trace = _write_trace(tmp_path)
        run1 = _make_run(file_path=trace, benchmark_id="lock-run-1")
        dm1 = await _run_configure(run1)
        await dm1.stop()
        calls_after_populate = mock_tokenizer.call_count

        run2 = _make_run(file_path=trace, benchmark_id="lock-run-2")
        key = mmap_cache.compute_cache_key_from_run(run2)
        assert key is not None

        def miss_but_keep_key(self: DatasetManager):
            # Simulate the race: the pre-lock lookup ran before the winner
            # committed, but the key is retained for the under-lock re-check.
            self._cache_key_for_run = key
            return None

        with patch.object(DatasetManager, "_try_cache_lookup", miss_but_keep_key):
            dm2 = await _run_configure(run2)

        assert dm2._cache_hit_used is True
        assert mock_tokenizer.call_count == calls_after_populate, (
            "hit-under-lock must skip the tokenizer"
        )
        await dm2.stop()

    @pytest.mark.asyncio
    async def test_lookup_under_lock_failure_returns_none(self, tmp_path: Path) -> None:
        trace = _write_trace(tmp_path)
        dm = DatasetManager(
            run=_make_run(file_path=trace, benchmark_id="ul-fail"),
            service_id="dm-test",
        )
        dm._cache_key_for_run = "somekey"

        with patch(
            "aiperf.dataset.dataset_manager.mmap_cache.lookup",
            side_effect=OSError("disk unplugged"),
        ):
            assert dm._lookup_under_lock() is None

    @pytest.mark.asyncio
    async def test_try_cache_lookup_key_failure_disables_cache(
        self, tmp_path: Path
    ) -> None:
        """A key-computation crash must degrade to MISS, never fail the run."""
        trace = _write_trace(tmp_path)
        dm = DatasetManager(
            run=_make_run(file_path=trace, benchmark_id="key-fail"),
            service_id="dm-test",
        )

        with patch(
            "aiperf.dataset.dataset_manager.mmap_cache.compute_cache_key_from_run",
            side_effect=RuntimeError("unserializable config"),
        ):
            assert dm._try_cache_lookup() is None
        assert dm._cache_key_for_run is None, "no key -> no post-run populate"

    @pytest.mark.asyncio
    async def test_try_cache_lookup_lookup_failure_keeps_key(
        self, tmp_path: Path
    ) -> None:
        """A lookup crash is a MISS, but the key survives so the post-run
        populate still writes the entry."""
        trace = _write_trace(tmp_path)
        dm = DatasetManager(
            run=_make_run(file_path=trace, benchmark_id="lookup-fail"),
            service_id="dm-test",
        )

        with patch(
            "aiperf.dataset.dataset_manager.mmap_cache.lookup",
            side_effect=ValueError("corrupt entry"),
        ):
            assert dm._try_cache_lookup() is None
        assert dm._cache_key_for_run is not None

    @pytest.mark.asyncio
    async def test_cache_hit_with_corrupt_metadata_falls_back_to_full_configure(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        """A HIT whose manifest dataset_metadata_json fails validation must be
        treated as a MISS: restored files removed, poisoned cache entry
        invalidated, and the full pipeline re-run (so populate can heal)."""
        import orjson

        from aiperf.common.models import DatasetMetadata

        trace = _write_trace(tmp_path)
        run1 = _make_run(file_path=trace, benchmark_id="corrupt-1")
        dm1 = await _run_configure(run1)
        await dm1.stop()
        key = mmap_cache.compute_cache_key_from_run(run1)
        assert key is not None

        manifest_path = mmap_cache.cache_dir() / key / mmap_cache.MANIFEST_FILENAME
        raw = orjson.loads(manifest_path.read_bytes())
        raw["dataset_metadata_json"] = "certainly not json"
        manifest_path.write_bytes(orjson.dumps(raw))
        calls_before = mock_tokenizer.call_count

        run2 = _make_run(file_path=trace, benchmark_id="corrupt-2")
        dm2 = await _run_configure(run2)

        assert dm2._cache_hit_used is False
        assert mock_tokenizer.call_count > calls_before, (
            "corrupt HIT must fall back to the full tokenize path"
        )
        # The run still completed: metadata was rebuilt by the full pipeline.
        assert dm2.dataset_metadata is not None
        assert len(dm2.dataset_metadata.conversations) == 2

        # invalidate() dropped the poison; populate rewrote a valid entry.
        healed = mmap_cache.lookup(key, compressed=False)
        assert healed is not None, "post-run populate must heal the invalidated key"
        DatasetMetadata.model_validate_json(healed.manifest.dataset_metadata_json)
        await dm2.stop()

    @pytest.mark.asyncio
    async def test_populate_cache_after_run_guards(self, tmp_path: Path) -> None:
        """Every early-return guard leaves the cache untouched."""
        trace = _write_trace(tmp_path)
        dm = DatasetManager(
            run=_make_run(file_path=trace, benchmark_id="guards"),
            service_id="dm-test",
        )
        cache_root = mmap_cache.cache_dir()

        # Guard 1: HIT used -> nothing to write back.
        dm._cache_hit_used = True
        dm._populate_cache_after_run()

        # Guard 2: no key / no backing store.
        dm._cache_hit_used = False
        dm._cache_key_for_run = None
        dm._populate_cache_after_run()
        dm._cache_key_for_run = "guardkey"
        dm._backing_store = None
        dm._populate_cache_after_run()

        # Guard 3: no dataset metadata.
        dm._backing_store = object()  # never dereferenced past the guard
        dm.dataset_metadata = None
        dm._populate_cache_after_run()

        assert not (cache_root / "guardkey").exists()

    @pytest.mark.asyncio
    async def test_populate_cache_after_run_missing_files_skips(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        """Backing store finalized but run files vanished -> guard, no crash."""
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="vanished")
        dm = await _run_configure(run)
        key = dm._cache_key_for_run
        assert key is not None

        # Stop FIRST so the client store's mmap handles are closed; Windows
        # refuses to unlink files that still back an open memory map
        # (PermissionError WinError 32).
        await dm.stop()

        # Wipe the cache AND the run mmap files, then re-run the populate.
        import shutil

        shutil.rmtree(mmap_cache.cache_dir())
        data_p, index_p = dm._run_mmap_paths()
        data_p.unlink(missing_ok=True)
        index_p.unlink(missing_ok=True)

        dm._cache_hit_used = False
        dm._populate_cache_after_run()

        assert not (mmap_cache.cache_dir() / key).exists()

    @pytest.mark.asyncio
    async def test_populate_failure_does_not_fail_the_run(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        """mmap_cache.populate crashing must be swallowed with a warning."""
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="popfail")

        with patch(
            "aiperf.dataset.dataset_manager.mmap_cache.populate",
            side_effect=OSError("disk full"),
        ):
            dm = await _run_configure(run)

        assert dm.dataset_metadata is not None
        key = dm._cache_key_for_run
        assert key is not None
        assert mmap_cache.lookup(key, compressed=False) is None
        await dm.stop()


class TestDatasetManagerCacheThreadOffload:
    """The multi-GiB cache copyfile paths must run off the event loop.

    Both the post-run populate and the HIT restore wrap their blocking file
    copies in ``asyncio.to_thread``; a revert to direct sync calls would block
    the DatasetManager event loop for the duration of the copy.
    """

    @pytest.fixture
    def to_thread_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> list[Callable[..., Any]]:
        """Record every callable dispatched through ``asyncio.to_thread``.

        The wrapper delegates to the real implementation, so the offloaded
        work still executes and behavior can be asserted alongside dispatch.
        """
        recorded: list[Callable[..., Any]] = []
        real_to_thread = asyncio.to_thread

        async def recording_to_thread(
            func: Callable[..., Any], /, *args: Any, **kwargs: Any
        ) -> Any:
            recorded.append(func)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)
        return recorded

    @pytest.mark.asyncio
    async def test_configure_dataset_locked_populate_offloaded_to_thread(
        self,
        tmp_path: Path,
        mock_tokenizer,
        to_thread_calls: list[Callable[..., Any]],
    ) -> None:
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="offload-populate")
        key = mmap_cache.compute_cache_key_from_run(run)
        assert key is not None

        dm = await _run_configure(run)
        await dm.stop()

        assert any(
            getattr(func, "__func__", None) is DatasetManager._populate_cache_after_run
            for func in to_thread_calls
        ), "_populate_cache_after_run must be dispatched via asyncio.to_thread"
        # The offloaded populate still ran for real: the entry is in the cache.
        assert mmap_cache.lookup(key, compressed=False) is not None

    @pytest.mark.asyncio
    async def test_cache_lookup_offloaded_to_thread(
        self,
        tmp_path: Path,
        mock_tokenizer,
        to_thread_calls: list[Callable[..., Any]],
    ) -> None:
        # Computing the key hashes the entire (multi-GB) input file, so the
        # lookup must run off the event loop or it blocks the DatasetManager's
        # heartbeat/command handlers for the full hash during configure.
        trace = _write_trace(tmp_path)
        run = _make_run(file_path=trace, benchmark_id="offload-lookup")

        dm = await _run_configure(run)
        await dm.stop()

        assert any(
            getattr(func, "__func__", None) is DatasetManager._try_cache_lookup
            for func in to_thread_calls
        ), "_try_cache_lookup must be dispatched via asyncio.to_thread"

    @pytest.mark.asyncio
    async def test_configure_from_cache_hit_restore_offloaded_to_thread(
        self,
        tmp_path: Path,
        mock_tokenizer,
        to_thread_calls: list[Callable[..., Any]],
    ) -> None:
        trace = _write_trace(tmp_path)
        run1 = _make_run(file_path=trace, benchmark_id="offload-restore-1")
        dm1 = await _run_configure(run1)
        await dm1.stop()

        to_thread_calls.clear()
        run2 = _make_run(file_path=trace, benchmark_id="offload-restore-2")
        dm2 = await _run_configure(run2)

        assert any(func is mmap_cache.restore_to_run_dir for func in to_thread_calls), (
            "restore_to_run_dir must be dispatched via asyncio.to_thread"
        )
        # The offloaded restore still ran for real: HIT adopted, files restored.
        assert dm2._cache_hit_used is True
        run_data_path, run_index_path = dm2._run_mmap_paths()
        assert run_data_path.exists()
        assert run_index_path.exists()
        await dm2.stop()
