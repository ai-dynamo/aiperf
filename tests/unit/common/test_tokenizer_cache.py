# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HuggingFace cache detection in the tokenizer module."""

from pathlib import Path

import pytest

from aiperf.common.tokenizer import Tokenizer, _is_hf_cached


@pytest.fixture
def hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HF_HUB_CACHE at a temporary directory."""
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    return tmp_path


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
