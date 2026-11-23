# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace cache detection in the tokenizer module."""

from pathlib import Path, PureWindowsPath
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.exceptions import TokenizerError
from aiperf.common.tokenizer import Tokenizer, _is_hf_cached, _is_local_path


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


class TestTokenizerOnlyRepoCacheLoad:
    """Regression: tokenizer-only HF repos lack ``config.json``.

    Before the fix, ``Tokenizer.from_pretrained`` switched to
    ``local_files_only=True`` whenever ``_is_hf_cached`` returned True,
    which made ``AutoTokenizer`` fail on its initial ``PreTrainedConfig``
    lookup with a misleading "Cannot connect to HuggingFace Hub and files
    not cached" error — even though the cache was warm and HF was reachable.
    """

    def _materialize_tokenizer_only_snapshot(
        self,
        hf_cache: Path,
        repo_id: str,
        revision: str,
        commit: str,
    ) -> Path:
        """Build a minimal HF cache layout that mirrors a tokenizer-only repo."""
        model_dir = hf_cache / f"models--{repo_id.replace('/', '--')}"
        snapshot_dir = model_dir / "snapshots" / commit
        snapshot_dir.mkdir(parents=True)
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / revision).write_text(commit)
        # Tokenizer files only — no config.json (this is the whole point).
        (snapshot_dir / "tokenizer.json").write_text("{}")
        (snapshot_dir / "tokenizer_config.json").write_text("{}")
        return snapshot_dir

    def test_online_cached_load_does_not_force_local_files_only(
        self, hf_cache, monkeypatch
    ) -> None:
        """When online and cache is warm, ``local_files_only`` must NOT be set.

        Forcing ``local_files_only=True`` was the bug: tokenizer-only repos
        lack ``config.json``, so the offline path mistakes the missing file
        for a missing cache and surfaces the wrong error.
        """
        repo = "hf-internal-testing/llama-tokenizer"
        self._materialize_tokenizer_only_snapshot(hf_cache, repo, "main", "deadbeef")

        # Ensure online (offline env vars cleared).
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        seen_kwargs: dict[str, str] = {}

        def fake_from_pretrained(name_or_path, **kwargs):
            seen_kwargs["name_or_path"] = name_or_path
            seen_kwargs.update(kwargs)
            # Return a sentinel — we only care about the call shape, not
            # an actual tokenizer.
            return _FakeHfTokenizer()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            Tokenizer.from_pretrained(repo, revision="main", resolve_alias=False)

        assert "local_files_only" not in seen_kwargs, (
            "Online cached load must not pass local_files_only=True; "
            "transformers cannot distinguish a missing-on-server file (e.g. "
            "config.json for a tokenizer-only repo) from a missing cache "
            "entry, and surfaces a misleading 'Cannot connect to HF Hub' error."
        )
        assert seen_kwargs["name_or_path"] == repo
        assert seen_kwargs["revision"] == "main"

    def test_offline_cached_load_uses_snapshot_path(
        self, hf_cache, monkeypatch
    ) -> None:
        """Offline + cached: must hand AutoTokenizer the snapshot directory.

        Loading via a path bypasses transformers' ``config.json`` lookup,
        which is the only way a tokenizer-only repo loads cleanly under
        ``HF_HUB_OFFLINE=1``.
        """
        repo = "hf-internal-testing/llama-tokenizer"
        commit = "deadbeefcafebabe"
        snapshot = self._materialize_tokenizer_only_snapshot(
            hf_cache, repo, "main", commit
        )

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        # Stub snapshot_download so it returns our materialized path without
        # touching the network or the real cache (it would otherwise key off
        # huggingface_hub.constants.HF_HUB_CACHE which is monkeypatched but
        # the local copy huggingface_hub already imported elsewhere may not
        # see the patch).
        from aiperf.common import tokenizer as tk_module

        def fake_snapshot_download(name, revision, local_files_only):
            assert local_files_only is True
            assert name == repo
            assert revision == "main"
            return str(snapshot)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

        seen: dict[str, str] = {}

        def fake_from_pretrained(name_or_path, **kwargs):
            seen["name_or_path"] = name_or_path
            seen.update(kwargs)
            return _FakeHfTokenizer()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            tk_module.Tokenizer.from_pretrained(
                repo, revision="main", resolve_alias=False
            )

        assert seen["name_or_path"] == str(snapshot), (
            "Offline cached load must pass the snapshot directory path so "
            "AutoTokenizer skips the config.json round-trip."
        )
        # No revision/local_files_only when loading via path; transformers
        # infers everything from the directory contents.
        assert "revision" not in seen
        assert "local_files_only" not in seen


class TestOfflineLocalPathLoad:
    """Offline workers must load the same local directory the online parent loads.

    Worker pools (``parallel_decode``, ``parallel_convert``) force
    ``HF_HUB_OFFLINE=1`` and reload the tokenizer by name. Online,
    transformers loads any existing directory before consulting the hub, so
    every row below is a local load in the parent. Offline, handing these
    names to ``snapshot_download`` raised ``HFValidationError`` (anchored
    paths) or missed the cache (relative paths) and broke the process pool,
    or silently loaded a different cached tokenizer (shadowed repo id).
    """

    @pytest.fixture
    def offline_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, str]:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        def fail_snapshot_download(*args, **kwargs):
            raise AssertionError(
                f"snapshot_download must not see a local path: {args} {kwargs}"
            )

        monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot_download)
        seen: dict[str, str] = {}

        def fake_from_pretrained(name_or_path, **kwargs):
            seen["name_or_path"] = name_or_path
            return _FakeHfTokenizer()

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", fake_from_pretrained
        )
        return seen

    @pytest.mark.parametrize(
        ("rel_dir", "absolute"),
        [
            param("tok", True, id="absolute-path"),
            param("./tok", False, id="dot-slash-relative"),
            param("models/tok", False, id="relative-no-dot-slash"),
            param("tok", False, id="bare-dir-in-cwd"),
            param("Org/Model", False, id="cwd-dir-shadows-repo-id"),
        ],
    )  # fmt: skip
    def test_from_pretrained_offline_local_dir_loads_that_dir(
        self,
        tmp_path: Path,
        offline_loader: dict[str, str],
        rel_dir: str,
        absolute: bool,
    ) -> None:
        local_dir = tmp_path / rel_dir
        local_dir.mkdir(parents=True)
        name = str(local_dir) if absolute else rel_dir

        Tokenizer.from_pretrained(name, resolve_alias=False)

        assert Path(offline_loader["name_or_path"]).resolve() == local_dir.resolve()

    def test_from_pretrained_offline_missing_local_path_raises_clear_error(
        self, tmp_path: Path, offline_loader: dict[str, str]
    ) -> None:
        with pytest.raises(TokenizerError, match="is not a directory"):
            Tokenizer.from_pretrained(str(tmp_path / "missing"), resolve_alias=False)

        assert "name_or_path" not in offline_loader

    def test_is_local_path_treats_posix_absolute_path_as_local_on_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``WindowsPath("/data/x").is_absolute()`` is False; the anchor is not."""

        class _NonexistentWindowsPath(PureWindowsPath):
            def is_dir(self) -> bool:
                return False

        monkeypatch.setattr("aiperf.common.tokenizer.Path", _NonexistentWindowsPath)

        assert _NonexistentWindowsPath("/data/models/x").is_absolute() is False
        assert _is_local_path("/data/models/x") is True
        assert _is_local_path("Org/Model") is False


class _FakeHfTokenizer:
    """Minimal stub that satisfies ``Tokenizer._apply_kwarg_overrides``."""

    def encode(self, text, **kwargs):
        return []

    def decode(self, ids, **kwargs):
        return ""

    def __call__(self, text, **kwargs):
        return {"input_ids": []}
