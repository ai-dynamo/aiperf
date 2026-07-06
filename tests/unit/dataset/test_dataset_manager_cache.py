# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for DatasetManager mmap cache HIT/MISS pathway.

Verifies that:
- A second run with byte-identical inputs serves from cache (composer + tokenizer skipped).
- A first run populates the cache.
- Tokenizer changes invalidate the cache.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.environment import Environment
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset import mmap_cache
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.plugin.enums import CustomDatasetType, PublicDatasetType
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


def _make_weka_run(*, public_dataset: PublicDatasetType) -> BenchmarkRun:
    """Build a run for a public weka dataset (no input file / custom type)."""
    return make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            tokenizer_name="test-tokenizer",
            public_dataset=public_dataset,
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
    async def test_hf_weka_dataset_change_invalidates_cache(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        # The ``--hf-weka-dataset`` flag auto-selects ``weka_hf`` in v2, so
        # passing the repo alone resolves a public weka_hf dataset. Different
        # HF repos must produce different cache keys.
        run_a = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                tokenizer_name="test-tokenizer",
                hf_weka_dataset="semianalysisai/cc-traces-weka-051826",
            )
        )
        run_b = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                tokenizer_name="test-tokenizer",
                hf_weka_dataset="semianalysisai/cc-traces-weka-with-subagents-051826",
            )
        )

        key_a = mmap_cache.compute_cache_key_from_run(run_a)
        key_b = mmap_cache.compute_cache_key_from_run(run_b)

        assert key_a is not None and key_b is not None
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_public_dataset_aliases_with_same_hf_source_share_cache(
        self, tmp_path: Path, mock_tokenizer
    ) -> None:
        # SEMIANALYSIS_CC_TRACES_WEKA and ..._NO_SUBAGENTS resolve to the SAME
        # HF source (cc-traces-weka-no-subagents-051826), so the cache key (which
        # keys on resolved HF source identity, not the alias name) must match.
        run_a = _make_weka_run(
            public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA
        )
        run_b = _make_weka_run(
            public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_NO_SUBAGENTS
        )

        key_a = mmap_cache.compute_cache_key_from_run(run_a)
        key_b = mmap_cache.compute_cache_key_from_run(run_b)

        assert key_a is not None and key_b is not None
        assert key_a == key_b

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
        treated as a MISS: restored files removed, full pipeline re-run."""
        import orjson

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

        # Wipe the cache AND the run mmap files, then re-run the populate.
        import shutil

        shutil.rmtree(mmap_cache.cache_dir())
        data_p, index_p = dm._run_mmap_paths()
        data_p.unlink(missing_ok=True)
        index_p.unlink(missing_ok=True)

        dm._cache_hit_used = False
        dm._populate_cache_after_run()

        assert not (mmap_cache.cache_dir() / key).exists()
        await dm.stop()

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
