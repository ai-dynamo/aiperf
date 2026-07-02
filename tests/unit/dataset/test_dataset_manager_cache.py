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
