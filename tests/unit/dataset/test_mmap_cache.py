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
from unittest.mock import patch

import orjson
import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset import mmap_cache
from tests.unit.conftest import make_run_from_cli


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
    def test_osl_set_produces_a_computable_key(self, tmp_path: Path) -> None:
        # Regression: --osl puts a SamplingDistribution on dataset.osl. The key
        # serializes it to a JSON dict; a raw model would TypeError in
        # orjson.dumps and the dataset manager would silently DISABLE caching for
        # every --osl trace run. The key must compute AND track the OSL value.
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8}\n',
        )

        def _key(osl: int) -> str | None:
            run = make_run_from_cli(
                CLIConfig(
                    model_names=["test-model"],
                    input_file=str(trace),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    prompt_output_tokens_mean=osl,
                )
            )
            payload = mmap_cache._settings_payload_from_run(run)
            assert isinstance(payload["osl_fallback"], dict)  # serialized, not a model
            return mmap_cache.compute_cache_key_from_run(run)

        k64 = _key(64)
        k128 = _key(128)
        assert k64 is not None and len(k64) == 32  # computed, caching not disabled
        assert k64 != k128  # OSL fallback tracked in the key

    def test_key_changes_with_synthesis_multipliers(self, tmp_path: Path) -> None:
        # The full synthesis dump (not just max_isl/max_osl) must enter the key:
        # speedup_ratio + every *_multiplier rewrite the decoded trace bytes, so
        # two runs differing only in a multiplier must NOT share a cache entry.
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )

        def _key(**synthesis_kw: float) -> str | None:
            run = make_run_from_cli(
                CLIConfig(
                    model_names=["test-model"],
                    input_file=str(trace),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    prompt_output_tokens_mean=64,
                    **synthesis_kw,
                )
            )
            return mmap_cache.compute_cache_key_from_run(run)

        base = _key()
        speedup = _key(synthesis_speedup_ratio=2.0)
        out_mult = _key(synthesis_output_len_multiplier=3.0)
        prefix_mult = _key(synthesis_prefix_len_multiplier=1.5)
        assert base is not None
        assert len({base, speedup, out_mult, prefix_mult}) == 4

    def test_key_changes_with_preformat_payloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PREFORMAT_PAYLOADS flips the stored mmap FORMAT (conversation vs
        # payload_bytes); a cache HIT adopts the stored format verbatim, so a
        # warm entry built with the other setting would serve the wrong format.
        from aiperf.common.environment import Environment
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8}\n',
        )
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(trace),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                prompt_output_tokens_mean=64,
            )
        )
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", False)
        off = mmap_cache.compute_cache_key_from_run(run)
        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        on = mmap_cache.compute_cache_key_from_run(run)
        assert off is not None and on is not None and off != on

    def test_preformat_endpoint_knobs_change_key_only_when_preformat_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With preformat on, endpoint.format_payload() bakes the stream flag,
        # the max_tokens-vs-max_completion_tokens field name, and (for streaming
        # OpenAI-compatible endpoints) stream_options.include_usage from
        # use_server_token_count into the stored bytes, so runs differing only
        # in those knobs must NOT share a cache entry (else a HIT serves bytes
        # the run never asked for). With preformat off the knobs don't touch
        # the stored mmap, so the key must be stable.
        from aiperf.common.environment import Environment
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8}\n',
        )

        def key_for(
            *,
            streaming: bool = False,
            legacy: bool = False,
            server_token_count: bool = False,
        ) -> str | None:
            run = make_run_from_cli(
                CLIConfig(
                    model_names=["test-model"],
                    input_file=str(trace),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    prompt_output_tokens_mean=64,
                    streaming=streaming,
                    use_legacy_max_tokens=legacy,
                    use_server_token_count=server_token_count,
                )
            )
            return mmap_cache.compute_cache_key_from_run(run)

        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", True)
        assert key_for(streaming=True) != key_for(streaming=False)
        assert key_for(legacy=True) != key_for(legacy=False)
        assert key_for(server_token_count=True) != key_for(server_token_count=False)

        monkeypatch.setattr(Environment.DATASET, "PREFORMAT_PAYLOADS", False)
        assert key_for(streaming=True, legacy=True, server_token_count=True) == key_for(
            streaming=False, legacy=False, server_token_count=False
        )

    def test_inter_turn_delay_cap_changes_key_on_file_dataset(
        self, tmp_path: Path
    ) -> None:
        # v2 routes ``--inter-turn-delay-cap-seconds`` onto ``FileDataset`` only
        # (``_apply_inter_turn_delay_cap`` returns early for non-FILE datasets),
        # so the cap must distinguish the cache key on a file/trace dataset.
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )

        def _key(**extra: object) -> str | None:
            run = make_run_from_cli(
                CLIConfig(
                    model_names=["test-model"],
                    input_file=str(trace),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    **extra,
                )
            )
            return mmap_cache.compute_cache_key_from_run(run)

        base = _key()
        capped = _key(inter_turn_delay_cap_seconds=60.0)
        assert base is not None and capped is not None
        assert base != capped

    def test_random_seed_changes_key_on_trace_dataset(self, tmp_path: Path) -> None:
        # The base seed feeds per-block hash_id token derivation, so two runs
        # differing only in --random-seed must NOT share a cache entry.
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )

        def _key(seed: int) -> str | None:
            run = make_run_from_cli(
                CLIConfig(
                    model_names=["test-model"],
                    input_file=str(trace),
                    custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                    random_seed=seed,
                )
            )
            return mmap_cache.compute_cache_key_from_run(run)

        k1, k2 = _key(1), _key(2)
        assert k1 is not None and k2 is not None
        assert k1 != k2, "random_seed must distinguish the cache key"

    def test_settings_payload_includes_seed_corpus_osl(self, tmp_path: Path) -> None:
        # Regression guard for the wrong-cache-hit findings: random_seed,
        # prompts.corpus, and the per-record OSL fallback must enter the key.
        from aiperf.common.enums import PromptCorpus
        from aiperf.config.dataset import PromptSelectionConfig
        from aiperf.plugin.enums import CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(trace),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                random_seed=123,
            )
        )
        dataset = run.cfg.get_default_dataset()
        dataset.prompts = PromptSelectionConfig(corpus=PromptCorpus.CODING)
        payload = mmap_cache._settings_payload_from_run(run)
        assert payload["random_seed"] == 123
        assert "corpus" in payload
        assert payload["corpus"] == "coding"
        assert "osl_fallback" in payload

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

    def test_key_computes_with_a_pydantic_model_in_the_payload(
        self, tmp_path: Path
    ) -> None:
        # Defense for the osl_fallback class of bug: a pydantic model leaking
        # into the settings payload must NOT make orjson.dumps raise (which the
        # caller catches and silently disables caching) -- _cache_key_default
        # reduces it to model_dump so the key is always computable.
        from aiperf.config.distributions import NormalDistribution

        f = _write_input_file(tmp_path, b"x")
        settings = _stable_settings()
        settings["leaked_model"] = NormalDistribution(mean=64)
        key = mmap_cache.compute_cache_key(
            input_file=f,
            public_dataset=None,
            custom_dataset_type="single_turn",
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=settings,
        )
        assert len(key) == 32  # computed, not TypeError'd into a disabled cache

    def test_cache_key_default_reduces_models_and_other_objects(self) -> None:
        from aiperf.config.distributions import NormalDistribution

        dumped = mmap_cache._cache_key_default(NormalDistribution(mean=64))
        assert isinstance(dumped, dict) and dumped["mean"] == 64
        # Non-model objects fall back to a deterministic str (coarse but stable).
        assert mmap_cache._cache_key_default(object) == str(object)

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
        assert hit.manifest.has_inputs_json is False

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
        raw["version"] = 22
        manifest_path.write_bytes(orjson.dumps(raw))

        assert mmap_cache.MANIFEST_VERSION == 26
        assert mmap_cache.lookup("pre-overlap-frontier", compressed=False) is None

    def test_lookup_compressed_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="uncomp", compressed=False)
        # Same key requested as compressed -> MISS.
        assert mmap_cache.lookup("uncomp", compressed=True) is None

    def test_invalidate_removes_entry_so_populate_can_heal(
        self, tmp_path: Path
    ) -> None:
        cache_root = mmap_cache.cache_dir()
        entry_dir = _populate_entry(cache_root, cache_key="poison")
        assert entry_dir.exists()
        assert mmap_cache.lookup("poison", compressed=False) is not None

        assert mmap_cache.invalidate("poison") is True
        assert not entry_dir.exists()
        assert mmap_cache.lookup("poison", compressed=False) is None
        assert mmap_cache.invalidate("poison") is False  # already gone

        # populate can rewrite the key after invalidation
        healed = _populate_entry(cache_root, cache_key="poison", data_bytes=b"HEALED")
        hit = mmap_cache.lookup("poison", compressed=False)
        assert hit is not None
        assert hit.entry_dir == healed
        assert hit.data_path.read_bytes() == b"HEALED"

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
        """Five concurrent contenders on the same key never overlap inside.

        Asserts a LIVE occupancy counter never exceeds 1 rather than reasoning
        about wall-clock event timestamps: under the autouse ``no_sleep`` fixture
        (asyncio.sleep -> 0) and ``-n auto`` load, timestamp ties made the
        sorted-balance check flaky. The ``sleep(0)`` inside the lock yields so a
        non-serializing implementation WOULD let a second contender in."""
        import asyncio

        inside = 0
        max_inside = 0

        async def hold() -> None:
            nonlocal inside, max_inside
            async with mmap_cache.acquire_cache_lock("k", timeout=10.0):
                inside += 1
                max_inside = max(max_inside, inside)
                await asyncio.sleep(0)  # yield: a broken lock would overlap here
                inside -= 1

        await asyncio.gather(*(hold() for _ in range(5)))
        assert max_inside == 1, f"same-key lock allowed {max_inside} contenders inside"

    @pytest.mark.asyncio
    async def test_independent_keys_dont_serialize(self) -> None:
        """Two contenders on different keys CAN be held simultaneously.

        Deterministic rendezvous instead of a wall-clock ``elapsed`` bound (which
        lost all signal once ``no_sleep`` zeroed the dwell and flaked under load):
        each holder signals that it is inside, then waits for the OTHER to signal
        it is inside too. Both can only complete if both locks are held at the
        same time. If distinct keys wrongly shared one lock, the first holder
        would block the second forever and ``wait_for`` would time out."""
        import asyncio

        alpha_in = asyncio.Event()
        beta_in = asyncio.Event()

        async def hold(key: str, mine: asyncio.Event, other: asyncio.Event) -> None:
            async with mmap_cache.acquire_cache_lock(key, timeout=5.0):
                mine.set()
                await asyncio.wait_for(other.wait(), timeout=5.0)

        await asyncio.gather(
            hold("alpha", alpha_in, beta_in),
            hold("beta", beta_in, alpha_in),
        )
        assert alpha_in.is_set() and beta_in.is_set()

    @pytest.mark.asyncio
    async def test_timeout_degrades_to_unlocked_populate(self) -> None:
        """Holder beyond timeout lets the waiter proceed unlocked, not raise.

        A populator SIGKILLed before completing leaves the lock held (NFS
        tombstone) with no complete entry, so the cache-complete bypass never
        fires. Rather than fail the whole run, the waiter degrades to an
        unlocked populate (safe: populate is atomic). The waiter therefore
        enters the context body while the holder still holds the lock."""
        import asyncio

        holder_acquired = asyncio.Event()
        holder_release = asyncio.Event()
        waiter_entered = asyncio.Event()

        async def holder() -> None:
            async with mmap_cache.acquire_cache_lock("k", timeout=5.0):
                holder_acquired.set()
                await holder_release.wait()

        async def waiter() -> None:
            await holder_acquired.wait()
            # Does NOT raise filelock.Timeout: proceeds unlocked after timeout.
            async with mmap_cache.acquire_cache_lock("k", timeout=0.5):
                waiter_entered.set()

        holder_task = asyncio.create_task(holder())
        try:
            await asyncio.wait_for(waiter(), timeout=5.0)
            assert waiter_entered.is_set()
        finally:
            holder_release.set()
            await holder_task


class TestTraceVerbatimGate:
    """Only trace / verbatim datasets are cacheable; everything else
    bypasses the cache (and emits inputs.json). ``is_trace_or_verbatim_dataset``
    and the ``compute_cache_key_from_run`` gate keep those two decisions in
    lockstep.
    """

    @pytest.mark.parametrize(
        "custom_type",
        [
            "mooncake_trace",
            "bailian_trace",
            "burst_gpt_trace",
            "sagemaker_data_capture",
            "raw_payload",
            "inputs_json",
        ],
    )
    def test_predicate_true_for_trace_custom_types(self, custom_type: str) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(custom_type, None) is True

    @pytest.mark.parametrize(
        "custom_type",
        ["single_turn", "multi_turn", "random_pool", "dag_jsonl", "speed_bench_coding"],
    )
    def test_predicate_false_for_non_trace_custom_types(self, custom_type: str) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(custom_type, None) is False

    def test_predicate_false_for_synthetic(self) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(None, None) is False

    def test_predicate_false_for_public_datasets(self) -> None:
        assert (
            mmap_cache.is_trace_or_verbatim_dataset(None, "openai_humaneval") is False
        )

    def test_compute_cache_key_none_for_non_trace_file_dataset(
        self, tmp_path: Path
    ) -> None:
        """A non-trace file dataset is not cacheable -> key is None -> miss path
        every run (so inputs.json is always re-emitted)."""
        from aiperf.plugin.enums import CustomDatasetType

        f = _write_input_file(tmp_path, b'{"text": "hello"}\n')
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(f),
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            )
        )
        assert mmap_cache.compute_cache_key_from_run(run) is None

    def test_compute_cache_key_present_for_trace_file_dataset(
        self, tmp_path: Path
    ) -> None:
        """A trace file dataset IS cacheable -> non-None key."""
        from aiperf.plugin.enums import CustomDatasetType

        f = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(f),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
            )
        )
        assert mmap_cache.compute_cache_key_from_run(run) is not None


class TestHashDirContents:
    """Directory inputs are hashed by relative path + bytes so two corpora
    with the same directory name but different contents get distinct keys."""

    def _make_corpus(self, root: Path) -> Path:
        corpus = root / "corpus"
        (corpus / "nested").mkdir(parents=True)
        (corpus / "a.jsonl").write_bytes(b'{"x": 1}\n')
        (corpus / "nested" / "b.jsonl").write_bytes(b'{"y": 2}\n')
        return corpus

    def test_digest_is_deterministic(self, tmp_path: Path) -> None:
        corpus = self._make_corpus(tmp_path)
        assert mmap_cache.hash_dir_contents(corpus) == mmap_cache.hash_dir_contents(
            corpus
        )

    def test_digest_changes_when_file_bytes_change(self, tmp_path: Path) -> None:
        corpus = self._make_corpus(tmp_path)
        before = mmap_cache.hash_dir_contents(corpus)
        (corpus / "a.jsonl").write_bytes(b'{"x": 999}\n')
        assert mmap_cache.hash_dir_contents(corpus) != before

    def test_digest_changes_when_file_renamed(self, tmp_path: Path) -> None:
        corpus = self._make_corpus(tmp_path)
        before = mmap_cache.hash_dir_contents(corpus)
        (corpus / "a.jsonl").rename(corpus / "renamed.jsonl")
        assert mmap_cache.hash_dir_contents(corpus) != before

    def test_empty_subdirectories_do_not_affect_digest(self, tmp_path: Path) -> None:
        corpus = self._make_corpus(tmp_path)
        before = mmap_cache.hash_dir_contents(corpus)
        (corpus / "empty_dir").mkdir()
        assert mmap_cache.hash_dir_contents(corpus) == before

    def test_hash_input_path_routes_dir_and_file(self, tmp_path: Path) -> None:
        corpus = self._make_corpus(tmp_path)
        single = corpus / "a.jsonl"
        assert mmap_cache._hash_input_path(corpus) == mmap_cache.hash_dir_contents(
            corpus
        )
        assert mmap_cache._hash_input_path(single) == mmap_cache.hash_file_bytes(single)

    def test_compute_cache_key_accepts_directory_input(self, tmp_path: Path) -> None:
        corpus_a = self._make_corpus(tmp_path)
        key_a = mmap_cache.compute_cache_key(
            input_file=corpus_a,
            public_dataset=None,
            custom_dataset_type="mooncake_trace",
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        (corpus_a / "a.jsonl").write_bytes(b'{"x": 42}\n')
        key_b = mmap_cache.compute_cache_key(
            input_file=corpus_a,
            public_dataset=None,
            custom_dataset_type="mooncake_trace",
            tokenizer_identity=_stable_tokenizer(),
            settings_payload=_stable_settings(),
        )
        assert key_a != key_b


def _make_manifest(
    cache_key: str, *, compressed: bool = False
) -> mmap_cache.CacheManifest:
    return mmap_cache.CacheManifest(
        cache_key=cache_key,
        created_at=time.time(),
        num_conversations=1,
        total_size_bytes=4,
        compressed=compressed,
        compressed_size_bytes=4 if compressed else 0,
        mmap_format="conversation",
        dataset_metadata_json='{"conversations": [], "sampling_strategy": "random"}',
    )


class TestPopulateEdgeCases:
    """Failure and race paths of :func:`mmap_cache.populate`."""

    def _write_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        data_p = tmp_path / "src_dataset.dat"
        idx_p = tmp_path / "src_index.dat"
        data_p.write_bytes(b"DATA")
        idx_p.write_bytes(b"IDX")
        return data_p, idx_p

    def test_populate_winner_stays_on_existing_entry(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="winner", data_bytes=b"FIRST")
        data_p, idx_p = self._write_sources(tmp_path)

        out = mmap_cache.populate(
            cache_key="winner",
            run_data_path=data_p,
            run_index_path=idx_p,
            manifest=_make_manifest("winner"),
        )

        assert out == cache_root / "winner"
        # First writer's bytes stay; the second populate is a no-op.
        assert (cache_root / "winner" / "dataset.dat").read_bytes() == b"FIRST"

    def test_populate_cleans_leftover_tmp_dir(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)
        leftover = cache_root / f".leftover.tmp.{os.getpid()}"
        leftover.mkdir()
        (leftover / "stale.dat").write_bytes(b"STALE")
        data_p, idx_p = self._write_sources(tmp_path)

        out = mmap_cache.populate(
            cache_key="leftover",
            run_data_path=data_p,
            run_index_path=idx_p,
            manifest=_make_manifest("leftover"),
        )

        assert out == cache_root / "leftover"
        assert not leftover.exists()
        assert mmap_cache.lookup("leftover", compressed=False) is not None

    def test_populate_missing_source_returns_none(self, tmp_path: Path) -> None:
        out = mmap_cache.populate(
            cache_key="nosrc",
            run_data_path=tmp_path / "does_not_exist.dat",
            run_index_path=tmp_path / "also_missing.dat",
            manifest=_make_manifest("nosrc"),
        )
        assert out is None
        # The tmp dir must not be left behind.
        assert not any(mmap_cache.cache_dir().glob(".nosrc.tmp.*"))

    def test_populate_replace_race_returns_existing_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """os.replace loses to a concurrent writer -> their entry is returned."""
        cache_root = mmap_cache.cache_dir()
        data_p, idx_p = self._write_sources(tmp_path)

        final_dir = cache_root / "race"

        def replace_and_lose(src: str | Path, dst: str | Path) -> None:
            # Simulate the concurrent winner committing first, then our
            # rename failing (non-empty destination on POSIX).
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "dataset.dat").write_bytes(b"WINNER")
            raise OSError("Directory not empty")

        monkeypatch.setattr(mmap_cache.os, "replace", replace_and_lose)

        out = mmap_cache.populate(
            cache_key="race",
            run_data_path=data_p,
            run_index_path=idx_p,
            manifest=_make_manifest("race"),
        )

        assert out == final_dir
        assert (final_dir / "dataset.dat").read_bytes() == b"WINNER"
        assert not any(cache_root.glob(".race.tmp.*"))

    def test_populate_replace_failure_without_winner_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_p, idx_p = self._write_sources(tmp_path)

        def always_fail(src: str | Path, dst: str | Path) -> None:
            raise OSError("EXDEV")

        monkeypatch.setattr(mmap_cache.os, "replace", always_fail)

        out = mmap_cache.populate(
            cache_key="orphan",
            run_data_path=data_p,
            run_index_path=idx_p,
            manifest=_make_manifest("orphan"),
        )

        assert out is None
        assert not (mmap_cache.cache_dir() / "orphan").exists()


class TestLookupPartialEntries:
    """A manifest without its sibling data files is a partial entry -> MISS."""

    def test_lookup_missing_dataset_file_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="nodata")
        (cache_root / "nodata" / "dataset.dat").unlink()
        assert mmap_cache.lookup("nodata", compressed=False) is None

    def test_lookup_missing_index_file_returns_none(self, tmp_path: Path) -> None:
        cache_root = mmap_cache.cache_dir()
        _populate_entry(cache_root, cache_key="noindex")
        (cache_root / "noindex" / "index.dat").unlink()
        assert mmap_cache.lookup("noindex", compressed=False) is None

    def test_lookup_legacy_inputs_json_entry_resolves_path(
        self, tmp_path: Path
    ) -> None:
        """A legacy entry with has_inputs_json=True resolves the sibling path."""
        cache_root = mmap_cache.cache_dir()
        entry = cache_root / "legacy"
        entry.mkdir(parents=True)
        (entry / "dataset.dat").write_bytes(b"DATA")
        (entry / "index.dat").write_bytes(b"IDX")
        (entry / mmap_cache.INPUTS_JSON_FILENAME).write_bytes(b'{"data": []}')
        manifest = _make_manifest("legacy")
        manifest.has_inputs_json = True
        (entry / mmap_cache.MANIFEST_FILENAME).write_bytes(
            orjson.dumps(manifest.model_dump(mode="json"))
        )

        hit = mmap_cache.lookup("legacy", compressed=False)
        assert hit is not None
        assert hit.inputs_json_path == entry / mmap_cache.INPUTS_JSON_FILENAME

    def test_lookup_legacy_inputs_json_flag_without_file(self, tmp_path: Path) -> None:
        """has_inputs_json=True but the blob is gone -> HIT with path=None."""
        cache_root = mmap_cache.cache_dir()
        entry = cache_root / "legacy-gone"
        entry.mkdir(parents=True)
        (entry / "dataset.dat").write_bytes(b"DATA")
        (entry / "index.dat").write_bytes(b"IDX")
        manifest = _make_manifest("legacy-gone")
        manifest.has_inputs_json = True
        (entry / mmap_cache.MANIFEST_FILENAME).write_bytes(
            orjson.dumps(manifest.model_dump(mode="json"))
        )

        hit = mmap_cache.lookup("legacy-gone", compressed=False)
        assert hit is not None
        assert hit.inputs_json_path is None


class TestComputeCacheKeyRunGates:
    """Run-level gates that disable caching entirely (key is None)."""

    def test_accuracy_mode_disables_caching(self, tmp_path: Path) -> None:
        """Accuracy mode must gate BEFORE the dataset-source checks: pair it
        with a trace input file that would otherwise be cacheable."""
        from aiperf.plugin.enums import AccuracyBenchmarkType, CustomDatasetType

        trace = _write_input_file(
            tmp_path,
            b'{"session_id": "s1", "timestamp": 0, "input_length": 8, '
            b'"output_length": 4}\n',
        )
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(trace),
                custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
                accuracy_benchmark=AccuracyBenchmarkType.MMLU,
            )
        )
        assert run.cfg.accuracy is not None and run.cfg.accuracy.enabled is True

        assert mmap_cache.compute_cache_key_from_run(run) is None

    def test_synthetic_only_run_disables_caching(self) -> None:
        run = make_run_from_cli(CLIConfig(model_names=["test-model"]))
        assert mmap_cache.compute_cache_key_from_run(run) is None


class TestSettingsPayloadFromRun:
    """Field extraction from the resolved run into the cache-key payload."""

    def test_prompt_dump_excludes_cache_bust(self) -> None:
        run = make_run_from_cli(CLIConfig(model_names=["test-model"]))
        payload = mmap_cache._settings_payload_from_run(run)
        assert isinstance(payload["prompt"], dict)
        assert payload["prompt"], "synthetic run should carry a prompt config"
        assert "cache_bust" not in payload["prompt"]

    def test_public_dataset_source_plugin_only(self) -> None:
        """A public dataset without an HF source reduces to its plugin name."""
        from aiperf.plugin.enums import PublicDatasetType

        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SHAREGPT,
            )
        )
        assert mmap_cache._public_dataset_source_from_run(run) == {"plugin": "sharegpt"}

    def test_public_dataset_source_includes_hf_identity(self) -> None:
        """An HF-backed public dataset keys on the resolved HF source."""
        from aiperf.plugin.enums import PublicDatasetType

        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.MMSTAR,
            )
        )
        source = mmap_cache._public_dataset_source_from_run(run)
        assert source is not None
        assert source["hf_dataset_name"] == "Lin-Chen/MMStar"
        assert "hf_split" in source

    def test_public_dataset_source_none_for_file_dataset(self, tmp_path: Path) -> None:
        from aiperf.plugin.enums import CustomDatasetType

        f = _write_input_file(tmp_path, b'{"text": "hello"}\n')
        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(f),
                custom_dataset_type=CustomDatasetType.SINGLE_TURN,
            )
        )
        assert mmap_cache._public_dataset_source_from_run(run) is None


class TestAcquireCacheLockBypassAndFallback:
    """Stale-lock bypass, SoftFileLock fallback, and release robustness."""

    @pytest.mark.asyncio
    async def test_manifest_presence_skips_lock_acquire(self) -> None:
        """A complete cache entry bypasses the lock entirely (SIGKILLed
        populator tombstone must not wedge waiters)."""
        cache_root = mmap_cache.cache_dir()
        entry = cache_root / "prepopulated"
        entry.mkdir(parents=True)
        (entry / mmap_cache.MANIFEST_FILENAME).write_bytes(b"{}")

        async with mmap_cache.acquire_cache_lock("prepopulated", timeout=1.0):
            pass

        assert not (cache_root / "prepopulated.lock").exists()

    def test_blocking_acquire_bypasses_when_cache_complete(
        self, tmp_path: Path
    ) -> None:
        from filelock import FileLock

        from aiperf.dataset import mmap_cache_lock

        lock_path = tmp_path / "key.lock"
        lock = FileLock(str(lock_path), thread_local=False)

        acquired = mmap_cache_lock._blocking_acquire(
            lock, 5.0, lock_path, cache_complete_check=lambda: True
        )

        assert acquired is False
        assert not lock.is_locked

    def test_blocking_acquire_bypasses_between_retries(self, tmp_path: Path) -> None:
        """The bypass also unwedges a waiter mid-retry once the cache lands."""
        from filelock import FileLock

        from aiperf.dataset import mmap_cache_lock

        lock_path = tmp_path / "key.lock"
        holder = FileLock(str(lock_path), thread_local=False)
        holder.acquire(timeout=1.0)
        try:
            checks = iter([False, True])
            waiter = FileLock(str(lock_path), thread_local=False)

            with patch.object(mmap_cache_lock, "_LOCK_LOG_EVERY_SECONDS", 0.05):
                acquired = mmap_cache_lock._blocking_acquire(
                    waiter,
                    5.0,
                    lock_path,
                    cache_complete_check=lambda: next(checks),
                )

            assert acquired is False
            assert not waiter.is_locked
        finally:
            holder.release()

    @pytest.mark.asyncio
    async def test_soft_file_lock_fallback_on_flock_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os
        import stat

        from filelock import FileLock, SoftFileLock

        from aiperf.dataset import mmap_cache_lock

        # Pin the substring we match against filelock's NFS flock message so a
        # future filelock upgrade that drops/renames it fails this test.
        assert mmap_cache_lock._FLOCK_UNSUPPORTED_HINT == "use SoftFileLock instead"

        attempted: list[type] = []
        real_blocking = mmap_cache_lock._blocking_acquire
        lock_path_holder: list[Path] = []

        def flock_then_soft(lock, timeout, lock_path, cache_complete_check=None):
            attempted.append(type(lock))
            if len(attempted) == 1:
                raise NotImplementedError(
                    f"FileLock is unavailable, {mmap_cache_lock._FLOCK_UNSUPPORTED_HINT}"
                )
            lock_path_holder.append(Path(lock_path))
            return real_blocking(lock, timeout, lock_path, cache_complete_check)

        monkeypatch.setattr(mmap_cache_lock, "_blocking_acquire", flock_then_soft)

        # SoftFileLock's mode= is umask-masked; the post-acquire chmod must still
        # yield 0o664 under a restrictive cluster umask.
        old_umask = os.umask(0o077)
        try:
            async with mmap_cache.acquire_cache_lock("softlock", timeout=5.0):
                assert attempted == [FileLock, SoftFileLock]
                assert lock_path_holder, "SoftFileLock acquire path was not exercised"
                on_disk = stat.S_IMODE(lock_path_holder[0].stat().st_mode)
                assert on_disk == mmap_cache_lock._LOCK_FILE_MODE, (
                    f"SoftFileLock lock file mode {oct(on_disk)} != "
                    f"{oct(mmap_cache_lock._LOCK_FILE_MODE)} under umask 077"
                )
        finally:
            os.umask(old_umask)

    @pytest.mark.asyncio
    async def test_unrelated_not_implemented_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.dataset import mmap_cache_lock

        def explode(lock, timeout, lock_path, cache_complete_check=None):
            raise NotImplementedError("something else entirely")

        monkeypatch.setattr(mmap_cache_lock, "_blocking_acquire", explode)

        with pytest.raises(NotImplementedError, match="something else"):
            async with mmap_cache.acquire_cache_lock("hardfail", timeout=5.0):
                pass

    @pytest.mark.asyncio
    async def test_release_oserror_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A release() failing (e.g. lock file unlinked externally) must not
        propagate out of the context manager."""
        from filelock import FileLock

        real_release = FileLock.release
        raised = False

        def bad_release(self, *args, **kwargs):
            # Raise only on the context-manager release; delegate afterwards so
            # FileLock.__del__ at GC time doesn't emit an unraisable error.
            nonlocal raised
            if not raised:
                raised = True
                raise OSError("lock file vanished")
            return real_release(self, *args, **kwargs)

        monkeypatch.setattr(FileLock, "release", bad_release)

        async with mmap_cache.acquire_cache_lock("badrelease", timeout=5.0):
            pass

        assert raised
