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

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset import mmap_cache
from aiperf.plugin.enums import PublicDatasetType
from tests.unit.conftest import make_run_from_cli


def _cache_key_for_public_dataset(
    public_dataset: PublicDatasetType,
) -> str | None:
    """Resolve a v2 BenchmarkRun for a public dataset and compute its cache key."""
    run = make_run_from_cli(
        CLIConfig(model_names=["test-model"], public_dataset=public_dataset)
    )
    return mmap_cache.compute_cache_key_from_run(run)


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
        # v1 used SPEED_BENCH_{QUALITATIVE,CODING}, which are not ported to v2.
        # Use two weka aliases with DISTINCT HF sources instead -- the loader
        # metadata (hf_dataset_name) differs, so the keys must differ.
        no_subagents_key = _cache_key_for_public_dataset(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_NO_SUBAGENTS
        )
        with_subagents_key = _cache_key_for_public_dataset(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
        )

        assert no_subagents_key is not None
        assert with_subagents_key is not None
        assert no_subagents_key != with_subagents_key

    def test_key_changes_when_weka_live_assistant_setting_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
            )
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_LIVE_ASSISTANT_RESPONSES", False)
        pre_canned_key = mmap_cache.compute_cache_key_from_run(run)
        monkeypatch.setattr(Environment.DATASET, "WEKA_LIVE_ASSISTANT_RESPONSES", True)
        live_key = mmap_cache.compute_cache_key_from_run(run)

        assert pre_canned_key is not None
        assert live_key is not None
        assert pre_canned_key != live_key

    def test_key_changes_when_weka_split_flattened_agents_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.environment import Environment

        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
            )
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", True)
        split_key = mmap_cache.compute_cache_key_from_run(run)
        monkeypatch.setattr(Environment.DATASET, "WEKA_SPLIT_FLATTENED_AGENTS", False)
        legacy_key = mmap_cache.compute_cache_key_from_run(run)

        assert split_key is not None
        assert legacy_key is not None
        # The flag changes loader output (split vs legacy single-stream), so
        # a warm cache from one mode must never serve the other.
        assert split_key != legacy_key

    def test_key_changes_when_weka_seam_knobs_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # WEKA_SEAM_MAX_GAP_SECONDS / WEKA_SEAM_MIN_OVERLAP_RATIO gate join-seam
        # stitching (detect_agent_chains) -> they change session structure baked
        # into the mmap, so two runs differing only in them must NOT share a key.
        from aiperf.common.environment import Environment

        run = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
            )
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_SEAM_MAX_GAP_SECONDS", 3600.0)
        base = mmap_cache.compute_cache_key_from_run(run)
        monkeypatch.setattr(Environment.DATASET, "WEKA_SEAM_MAX_GAP_SECONDS", 60.0)
        gap_key = mmap_cache.compute_cache_key_from_run(run)
        monkeypatch.setattr(Environment.DATASET, "WEKA_SEAM_MAX_GAP_SECONDS", 3600.0)
        monkeypatch.setattr(Environment.DATASET, "WEKA_SEAM_MIN_OVERLAP_RATIO", 0.9)
        overlap_key = mmap_cache.compute_cache_key_from_run(run)

        assert base is not None and gap_key is not None and overlap_key is not None
        assert len({base, gap_key, overlap_key}) == 3, (
            base,
            gap_key,
            overlap_key,
        )

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

    def test_key_changes_with_load_time_timing_knobs(self) -> None:
        # ``ignore_trace_delays``, ``use_think_time_only`` and
        # ``trace_idle_gap_cap_seconds`` route onto BOTH file and public
        # (weka_hf) datasets in v2 (``_apply_weka_trace_fields`` in
        # ``config/flags/_converter_dataset.py``), so a public weka dataset is
        # enough to exercise them. ``inter_turn_delay_cap_seconds`` is verified
        # separately on a file dataset (see
        # ``test_inter_turn_delay_cap_changes_key_on_file_dataset``) because v2
        # scopes that flag to ``FileDataset`` only.
        base = _cache_key_for_public_dataset(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS
        )

        run_ignore = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
                ignore_trace_delays=True,
            )
        )
        run_think = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
                use_think_time_only=True,
            )
        )
        run_warp = make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                public_dataset=PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS,
                trace_idle_gap_cap_seconds=60.0,
            )
        )

        keys = [
            base,
            mmap_cache.compute_cache_key_from_run(run_ignore),
            mmap_cache.compute_cache_key_from_run(run_think),
            mmap_cache.compute_cache_key_from_run(run_warp),
        ]
        assert all(k is not None for k in keys)
        # These knobs are applied at LOAD time (baked into cached Turn
        # timestamps/delays), so each must produce a distinct cache key.
        assert len(set(keys)) == len(keys), keys

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
        # prompt_corpus, and the per-record OSL fallback must enter the key.
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
        payload = mmap_cache._settings_payload_from_run(run)
        assert payload["random_seed"] == 123
        assert "prompt_corpus" in payload
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
        raw["version"] = 22
        manifest_path.write_bytes(orjson.dumps(raw))

        assert mmap_cache.MANIFEST_VERSION == 25
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


class TestTraceVerbatimGate:
    """Only trace / verbatim / weka datasets are cacheable; everything else
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
            "weka_trace",
        ],
    )
    def test_predicate_true_for_trace_custom_types(self, custom_type: str) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(custom_type, None) is True

    @pytest.mark.parametrize(
        "public",
        ["weka_hf", "semianalysis_cc_traces_weka", "semianalysis_cc_traces_weka_v3"],
    )
    def test_predicate_true_for_weka_public_datasets(self, public: str) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(None, public) is True

    @pytest.mark.parametrize(
        "custom_type",
        ["single_turn", "multi_turn", "random_pool", "dag_jsonl", "speed_bench_coding"],
    )
    def test_predicate_false_for_non_trace_custom_types(self, custom_type: str) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(custom_type, None) is False

    def test_predicate_false_for_synthetic(self) -> None:
        assert mmap_cache.is_trace_or_verbatim_dataset(None, None) is False

    def test_predicate_false_for_non_weka_public(self) -> None:
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
