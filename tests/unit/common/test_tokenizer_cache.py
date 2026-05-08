# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace cache detection in the tokenizer module."""

from pathlib import Path

import pytest

from aiperf.common.tokenizer import (
    Tokenizer,
    _get_revision_snapshot_dir,
    _is_hf_cached,
    _offline_config_fallback,
)


@pytest.fixture
def hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HF_HUB_CACHE at a temporary directory."""
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    return tmp_path


def _make_revision_snapshot(model_dir: Path, ref: str, commit_hash: str) -> None:
    """Create a refs/<ref> file and the corresponding snapshots/<hash>/ directory."""
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / ref).write_text(commit_hash)
    (model_dir / "snapshots" / commit_hash).mkdir(parents=True, exist_ok=True)


class TestIsHfCached:
    def test_returns_false_when_cache_dir_missing(self, tmp_path, monkeypatch) -> None:
        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(nonexistent))
        assert _is_hf_cached("some-model") is False

    def test_exact_match(self, hf_cache) -> None:
        (hf_cache / "models--meta-llama--Llama-2-7b-hf").mkdir()
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf") is True

    def test_alias_match_case_insensitive(self, hf_cache) -> None:
        (hf_cache / "models--openai-community--GPT2").mkdir()
        assert _is_hf_cached("gpt2") is True

    def test_no_match(self, hf_cache) -> None:
        (hf_cache / "models--some-org--other-model").mkdir()
        assert _is_hf_cached("nonexistent") is False

    def test_ignores_non_model_directories(self, hf_cache) -> None:
        (hf_cache / "refs").mkdir()
        (hf_cache / "blobs").mkdir()
        assert _is_hf_cached("refs") is False

    def test_empty_cache_dir(self, hf_cache) -> None:
        assert _is_hf_cached("anything") is False

    def test_ambiguous_alias_returns_false(self, hf_cache) -> None:
        (hf_cache / "models--org-a--gpt2").mkdir()
        (hf_cache / "models--org-b--gpt2").mkdir()
        assert _is_hf_cached("gpt2") is False

    # --- revision-aware tests ---

    def test_revision_returns_true_when_named_ref_and_snapshot_exist(
        self, hf_cache
    ) -> None:
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        _make_revision_snapshot(model_dir, "main", "abc123")
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf", revision="main") is True

    def test_revision_returns_false_when_refs_file_missing(self, hf_cache) -> None:
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "snapshots" / "abc123").mkdir(parents=True)
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf", revision="v1.2") is False

    def test_revision_returns_false_when_snapshot_dir_missing(self, hf_cache) -> None:
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "v1.2").write_text("def456")
        # snapshots/def456/ intentionally not created
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf", revision="v1.2") is False

    def test_revision_returns_false_when_different_revision_cached(
        self, hf_cache
    ) -> None:
        # "main" is cached; "v1.2" is not
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        _make_revision_snapshot(model_dir, "main", "abc123")
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf", revision="v1.2") is False

    def test_revision_as_direct_commit_hash_returns_true(self, hf_cache) -> None:
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "snapshots" / "abc123").mkdir(parents=True)
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf", revision="abc123") is True

    def test_no_revision_returns_true_when_only_directory_exists(
        self, hf_cache
    ) -> None:
        # Backward-compat: no revision arg → directory-only check
        (hf_cache / "models--meta-llama--Llama-2-7b-hf").mkdir()
        assert _is_hf_cached("meta-llama/Llama-2-7b-hf") is True


class TestFindCachedModelForAlias:
    def test_finds_cached_alias(self, hf_cache) -> None:
        (hf_cache / "models--openai-community--gpt2").mkdir()
        result = Tokenizer._find_cached_model_for_alias("gpt2")
        assert result == "openai-community/gpt2"

    def test_returns_none_when_no_match(self, hf_cache) -> None:
        (hf_cache / "models--some-org--other-model").mkdir()
        assert Tokenizer._find_cached_model_for_alias("gpt2") is None

    def test_returns_none_when_cache_missing(self, tmp_path, monkeypatch) -> None:
        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(nonexistent))
        assert Tokenizer._find_cached_model_for_alias("gpt2") is None

    def test_case_insensitive_match(self, hf_cache) -> None:
        (hf_cache / "models--OpenAI-Community--GPT2").mkdir()
        result = Tokenizer._find_cached_model_for_alias("gpt2")
        assert result == "OpenAI-Community/GPT2"

    def test_ambiguous_alias_returns_none(self, hf_cache) -> None:
        (hf_cache / "models--org-a--gpt2").mkdir()
        (hf_cache / "models--org-b--gpt2").mkdir()
        assert Tokenizer._find_cached_model_for_alias("gpt2") is None


class TestGetRevisionSnapshotDir:
    """Branch coverage for the _get_revision_snapshot_dir helper."""

    def test_returns_none_when_cache_dir_missing(self, tmp_path, monkeypatch) -> None:
        nonexistent = tmp_path / "does_not_exist"
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(nonexistent))
        assert _get_revision_snapshot_dir("any-model", "main") is None

    def test_returns_snapshot_for_exact_match_with_named_ref(self, hf_cache) -> None:
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        _make_revision_snapshot(model_dir, "main", "abc123")
        result = _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "main")
        assert result == model_dir / "snapshots" / "abc123"

    def test_resolves_via_alias_when_exact_match_missing(self, hf_cache) -> None:
        # Single alias match — exact `models--<name>` dir is not present.
        model_dir = hf_cache / "models--openai-community--gpt2"
        _make_revision_snapshot(model_dir, "main", "abc123")
        result = _get_revision_snapshot_dir("gpt2", "main")
        assert result == model_dir / "snapshots" / "abc123"

    def test_returns_none_when_alias_ambiguous(self, hf_cache) -> None:
        for org in ("org-a", "org-b"):
            model_dir = hf_cache / f"models--{org}--gpt2"
            _make_revision_snapshot(model_dir, "main", "abc123")
        assert _get_revision_snapshot_dir("gpt2", "main") is None

    def test_returns_none_when_no_alias_match(self, hf_cache) -> None:
        # Empty cache — neither exact dir nor any alias matches.
        assert _get_revision_snapshot_dir("nonexistent", "main") is None

    def test_returns_none_when_snapshots_dir_missing(self, hf_cache) -> None:
        # Model dir exists, but `snapshots/` subdir is absent.
        (hf_cache / "models--meta-llama--Llama-2-7b-hf").mkdir()
        assert _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "main") is None

    def test_uses_revision_as_snapshot_name_when_no_refs_file(self, hf_cache) -> None:
        # No refs/<rev> file: the revision is treated as a direct commit hash.
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "snapshots" / "abc123").mkdir(parents=True)
        result = _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "abc123")
        assert result == model_dir / "snapshots" / "abc123"

    def test_returns_none_when_resolved_snapshot_dir_missing(self, hf_cache) -> None:
        # refs/main points to a hash whose snapshots/<hash>/ was never created.
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "main").write_text("def456")
        (model_dir / "snapshots").mkdir(parents=True)
        assert _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "main") is None

    def test_rejects_revision_traversing_out_of_snapshots(self, hf_cache) -> None:
        """A revision like '../sibling' must not return a path outside snapshots/.

        Even when a sibling directory under model_dir exists with the right
        layout, the function must refuse to return it.
        """
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "snapshots").mkdir(parents=True)
        # Plant a sibling directory that '../sibling' would resolve to.
        (model_dir / "sibling").mkdir(parents=True)
        assert (
            _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "../sibling") is None
        )

    def test_rejects_revision_with_traversal_via_refs(self, hf_cache) -> None:
        """A revision that traverses the refs lookup must not return a path.

        Plants a file at model_dir/sibling that would be reached by
        refs/../sibling. The function must reject this without reading it.
        """
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "snapshots" / "abc123").mkdir(parents=True)
        # Sibling file that refs/../sibling would land on; contents look like
        # a valid commit hash so the snapshot lookup would otherwise succeed.
        (model_dir / "sibling").write_text("abc123")
        assert (
            _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", "../sibling") is None
        )

    def test_rejects_absolute_path_revision(self, hf_cache) -> None:
        """An absolute path as revision must be refused even if the dir exists."""
        model_dir = hf_cache / "models--meta-llama--Llama-2-7b-hf"
        (model_dir / "snapshots").mkdir(parents=True)
        outside = hf_cache / "outside"
        outside.mkdir()
        assert (
            _get_revision_snapshot_dir("meta-llama/Llama-2-7b-hf", str(outside)) is None
        )


class TestOfflineConfigFallback:
    """Patch transformers.PreTrainedConfig.from_pretrained so AutoTokenizer's
    offline fallback returns an empty config for tokenizer-only repos.

    Why: tokenizer-only HF repos (e.g. hf-internal-testing/llama-tokenizer)
    ship no config.json. Online, the hub returns 404 and AutoTokenizer falls
    through to tokenizer_config.json. Offline, the missing file raises a
    misleading "couldn't connect" OSError that AutoTokenizer doesn't recover
    from. The patch swallows the OSError only when the on-disk evidence
    matches a tokenizer-only repo (config.json absent, tokenizer_config.json
    present), returning an empty PreTrainedConfig so dispatch falls through.
    """

    def _make_snapshot(
        self,
        hf_cache: Path,
        name: str,
        *,
        with_config: bool,
        with_tokenizer_config: bool,
    ) -> Path:
        """Build a minimal cached snapshot and return its directory."""
        model_dir = hf_cache / f"models--{name.replace('/', '--')}"
        snap = model_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "main").write_text("abc123")
        if with_config:
            (snap / "config.json").write_text('{"model_type": "llama"}')
        if with_tokenizer_config:
            (snap / "tokenizer_config.json").write_text(
                '{"tokenizer_class": "LlamaTokenizer"}'
            )
        return snap

    def test_returns_empty_config_when_only_tokenizer_config_present(
        self, hf_cache
    ) -> None:
        from transformers import PreTrainedConfig

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        # Replace the underlying loader so we exercise the patched wrapper
        # without touching the network.
        with (
            _offline_config_fallback(name, "main"),
            pytest.MonkeyPatch.context() as mp,
        ):
            mp.setattr(
                "transformers.configuration_utils.PreTrainedConfig.get_config_dict",
                raising,
            )
            result = PreTrainedConfig.from_pretrained(name, local_files_only=True)

        assert isinstance(result, PreTrainedConfig)

    def test_reraises_when_config_present(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        name = "meta-llama/Llama-2-7b-hf"
        self._make_snapshot(
            hf_cache, name, with_config=True, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        with (
            _offline_config_fallback(name, "main"),
            pytest.MonkeyPatch.context() as mp,
            pytest.raises(OSError, match="Couldn't connect"),
        ):
            mp.setattr(
                "transformers.configuration_utils.PreTrainedConfig.get_config_dict",
                raising,
            )
            PreTrainedConfig.from_pretrained(name, local_files_only=True)

    def test_reraises_when_tokenizer_config_absent(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        name = "some-broken/repo"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=False
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        with (
            _offline_config_fallback(name, "main"),
            pytest.MonkeyPatch.context() as mp,
            pytest.raises(OSError, match="Couldn't connect"),
        ):
            mp.setattr(
                "transformers.configuration_utils.PreTrainedConfig.get_config_dict",
                raising,
            )
            PreTrainedConfig.from_pretrained(name, local_files_only=True)

    def test_reraises_for_non_oserror(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise ValueError("schema mismatch")

        with (
            _offline_config_fallback(name, "main"),
            pytest.MonkeyPatch.context() as mp,
            pytest.raises(ValueError, match="schema mismatch"),
        ):
            mp.setattr(
                "transformers.configuration_utils.PreTrainedConfig.get_config_dict",
                raising,
            )
            PreTrainedConfig.from_pretrained(name, local_files_only=True)

    def test_reraises_when_snapshot_dir_none(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        # No snapshot is created — _get_revision_snapshot_dir returns None.
        name = "not-cached/anywhere"

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        with (
            _offline_config_fallback(name, "main"),
            pytest.MonkeyPatch.context() as mp,
            pytest.raises(OSError, match="Couldn't connect"),
        ):
            mp.setattr(
                "transformers.configuration_utils.PreTrainedConfig.get_config_dict",
                raising,
            )
            PreTrainedConfig.from_pretrained(name, local_files_only=True)

    def test_restores_original_on_normal_exit(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        original = PreTrainedConfig.__dict__["from_pretrained"]
        with _offline_config_fallback("anything", "main"):
            assert PreTrainedConfig.__dict__["from_pretrained"] is not original
        assert PreTrainedConfig.__dict__["from_pretrained"] is original

    def test_restores_original_on_exception(self, hf_cache) -> None:
        from transformers import PreTrainedConfig

        original = PreTrainedConfig.__dict__["from_pretrained"]
        with (
            pytest.raises(RuntimeError, match="boom"),
            _offline_config_fallback("anything", "main"),
        ):
            raise RuntimeError("boom")
        assert PreTrainedConfig.__dict__["from_pretrained"] is original
