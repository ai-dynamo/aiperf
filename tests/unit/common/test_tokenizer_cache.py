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
    """Patch transformers' cached_file resolver so AutoTokenizer's offline
    config.json lookup falls through to tokenizer_config.json for
    tokenizer-only HF repos (e.g. hf-internal-testing/llama-tokenizer)
    cached without a config.json.

    Online, the hub returns 404 for the missing config.json and AutoTokenizer
    dispatches via tokenizer_config.json. Offline, the missing file raises a
    misleading "couldn't connect" OSError. The context manager hands back a
    synthetic empty config.json path only when the on-disk evidence matches
    a tokenizer-only repo, leaving every other lookup untouched.
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

    def test_returns_stub_path_when_only_tokenizer_config_present(
        self, hf_cache, monkeypatch
    ) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        monkeypatch.setattr(hub_module, "cached_file", raising)

        with _offline_config_fallback(name, "main"):
            path = hub_module.cached_file(name, "config.json")
            assert Path(path).is_file()
            assert Path(path).read_text() == "{}"

    def test_passes_through_non_config_filenames(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        monkeypatch.setattr(hub_module, "cached_file", raising)

        # tokenizer.json is not the file we stub — OSError must propagate.
        with (
            _offline_config_fallback(name, "main"),
            pytest.raises(OSError, match="Couldn't connect"),
        ):
            hub_module.cached_file(name, "tokenizer.json")

    def test_passes_through_when_cached_file_succeeds(
        self, hf_cache, monkeypatch
    ) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        snap = self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )
        # Resolver returns a real cached path — wrapper must not intercept.
        cached = snap / "tokenizer_config.json"
        monkeypatch.setattr(hub_module, "cached_file", lambda *a, **kw: str(cached))

        with _offline_config_fallback(name, "main"):
            result = hub_module.cached_file(name, "config.json")

        assert result == str(cached)

    def test_noop_when_config_already_cached(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "meta-llama/Llama-2-7b-hf"
        self._make_snapshot(
            hf_cache, name, with_config=True, with_tokenizer_config=True
        )

        original = hub_module.cached_file
        with _offline_config_fallback(name, "main"):
            # Snapshot already has config.json — context manager is a no-op.
            assert hub_module.cached_file is original

    def test_noop_when_tokenizer_config_absent(self, hf_cache) -> None:
        from transformers.utils import hub as hub_module

        name = "some-broken/repo"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=False
        )

        original = hub_module.cached_file
        with _offline_config_fallback(name, "main"):
            # Not a tokenizer-only repo — context manager is a no-op.
            assert hub_module.cached_file is original

    def test_noop_when_snapshot_dir_none(self) -> None:
        from transformers.utils import hub as hub_module

        # No snapshot exists for this repo — _get_revision_snapshot_dir returns None.
        original = hub_module.cached_file
        with _offline_config_fallback("not-cached/anywhere", "main"):
            assert hub_module.cached_file is original

    def test_reraises_for_non_oserror(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise ValueError("schema mismatch")

        monkeypatch.setattr(hub_module, "cached_file", raising)

        with (
            _offline_config_fallback(name, "main"),
            pytest.raises(ValueError, match="schema mismatch"),
        ):
            hub_module.cached_file(name, "config.json")

    def test_restores_original_on_normal_exit(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )
        sentinel = lambda *a, **kw: "sentinel"  # noqa: E731
        monkeypatch.setattr(hub_module, "cached_file", sentinel)

        with _offline_config_fallback(name, "main"):
            assert hub_module.cached_file is not sentinel
        assert hub_module.cached_file is sentinel

    def test_restores_original_on_exception(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )
        sentinel = lambda *a, **kw: "sentinel"  # noqa: E731
        monkeypatch.setattr(hub_module, "cached_file", sentinel)

        with (
            pytest.raises(RuntimeError, match="boom"),
            _offline_config_fallback(name, "main"),
        ):
            raise RuntimeError("boom")
        assert hub_module.cached_file is sentinel

    def test_patches_local_bindings_in_other_modules(
        self, hf_cache, monkeypatch
    ) -> None:
        """transformers modules typically do ``from .utils import cached_file``
        at import time, creating per-module local bindings. The actual
        config.json lookup happens via configuration_utils' local binding
        (configuration_utils.py:725), not via transformers.utils.hub directly.
        Patching only hub_module.cached_file leaves those bindings untouched,
        which silently bypasses the fallback. The context manager must reach
        every module that imported cached_file by name.
        """
        from transformers import configuration_utils
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )

        def raising(*args, **kwargs):
            raise OSError("Couldn't connect")

        # Replace BOTH bindings with the raising stub, mirroring real usage
        # where transformers internals call their local binding.
        monkeypatch.setattr(hub_module, "cached_file", raising)
        monkeypatch.setattr(configuration_utils, "cached_file", raising)

        with _offline_config_fallback(name, "main"):
            # The bug: callers using configuration_utils' local binding get
            # the unpatched original. The fix must patch that too.
            path = configuration_utils.cached_file(name, "config.json")
            assert Path(path).is_file()
            assert Path(path).read_text() == "{}"

    def test_cleans_up_stub_dir_on_exit(self, hf_cache, monkeypatch) -> None:
        from transformers.utils import hub as hub_module

        name = "hf-internal-testing/llama-tokenizer"
        self._make_snapshot(
            hf_cache, name, with_config=False, with_tokenizer_config=True
        )
        monkeypatch.setattr(
            hub_module,
            "cached_file",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("x")),
        )

        with _offline_config_fallback(name, "main"):
            path = Path(hub_module.cached_file(name, "config.json"))
            assert path.is_file()
            stub_dir = path.parent
            assert stub_dir.is_dir()

        # Temp dir is removed once the context manager exits.
        assert not stub_dir.exists()
