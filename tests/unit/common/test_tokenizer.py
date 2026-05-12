# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import patch

import pytest

from aiperf.common.exceptions import NotInitializedError, TokenizerError
from aiperf.common.tokenizer import (
    BUILTIN_TOKENIZER_NAME,
    TIKTOKEN_ENCODING_NAMES,
    Tokenizer,
)


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(NotInitializedError):
            tokenizer("test")
        with pytest.raises(NotInitializedError):
            tokenizer.encode("test")
        with pytest.raises(NotInitializedError):
            tokenizer.decode([1])
        with pytest.raises(NotInitializedError):
            _ = tokenizer.bos_token_id


class TestBuiltinTokenizer:
    @pytest.fixture
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_pretrained(BUILTIN_TOKENIZER_NAME)

    def test_from_pretrained_returns_tokenizer(self, tokenizer: Tokenizer) -> None:
        assert tokenizer._tokenizer is not None

    def test_resolved_name(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.resolved_name == "o200k_base"

    def test_encode_returns_token_ids(self, tokenizer: Tokenizer) -> None:
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_decode_returns_string(self, tokenizer: Tokenizer) -> None:
        decoded = tokenizer.decode([15339, 1917])
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_encode_decode_roundtrip(self, tokenizer: Tokenizer) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_bos_token_id_is_none(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.bos_token_id is None

    def test_eos_token_id_is_set(self, tokenizer: Tokenizer) -> None:
        assert isinstance(tokenizer.eos_token_id, int)

    def test_block_separation_token_id_falls_back_to_eos(
        self, tokenizer: Tokenizer
    ) -> None:
        assert tokenizer.block_separation_token_id == tokenizer.eos_token_id

    def test_call_returns_input_ids(self, tokenizer: Tokenizer) -> None:
        result = tokenizer("hello")
        assert "input_ids" in result

    def test_no_hf_imports_required(self) -> None:
        """Builtin tokenizer must not trigger HuggingFace imports."""
        import sys

        hf_modules = {
            k for k in sys.modules if k.startswith(("transformers", "huggingface_hub"))
        }
        tokenizer = Tokenizer.from_pretrained(BUILTIN_TOKENIZER_NAME)
        new_hf_modules = {
            k for k in sys.modules if k.startswith(("transformers", "huggingface_hub"))
        }
        assert new_hf_modules == hf_modules
        assert tokenizer.encode("test") is not None


class TestTiktokenEncodingNames:
    @pytest.mark.parametrize("encoding_name", sorted(TIKTOKEN_ENCODING_NAMES))
    def test_from_pretrained_with_encoding_name(self, encoding_name: str) -> None:
        tokenizer = Tokenizer.from_pretrained(encoding_name)
        assert tokenizer.resolved_name == encoding_name
        assert tokenizer.encode("hello") is not None

    def test_o200k_base_direct(self) -> None:
        tokenizer = Tokenizer.from_pretrained("o200k_base")
        assert tokenizer.resolved_name == "o200k_base"
        assert tokenizer.encode("test") == Tokenizer.from_pretrained("builtin").encode(
            "test"
        )


class TestTiktokenImportError:
    def test_raises_tokenizer_error_when_tiktoken_missing(self) -> None:
        with (
            patch.dict("sys.modules", {"tiktoken": None}),
            pytest.raises(TokenizerError, match="tiktoken is required"),
        ):
            Tokenizer.from_pretrained(BUILTIN_TOKENIZER_NAME)


class TestIsFakeModelName:
    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("mock-model", id="mock-prefix"),
            pytest.param("mock-llama", id="mock-prefix-suffix"),
            pytest.param("test-model", id="test-model-substring"),
            pytest.param("fake-model", id="fake-prefix"),
            pytest.param("fake-llama-3", id="fake-prefix-with-suffix"),
            pytest.param("dummy", id="exact-dummy"),
            pytest.param("placeholder", id="exact-placeholder"),
            pytest.param("example", id="exact-example"),
            pytest.param("sample", id="exact-sample"),
            pytest.param("test", id="exact-test"),
            pytest.param("mock", id="exact-mock"),
            pytest.param("fake", id="exact-fake"),
            pytest.param("MOCK_MODEL", id="upper-with-underscore"),
            pytest.param("Test-Model-v2", id="title-case-suffix"),
            pytest.param("Test_Model_v2", id="underscore-normalize"),
            pytest.param("my-model", id="my-model"),
            pytest.param("your-model", id="your-model"),
            pytest.param("model-name", id="model-name"),
            pytest.param("model-id", id="model-id"),
            pytest.param("llama-test-model", id="test-model-as-suffix"),
            pytest.param("llama-mock", id="-mock-suffix"),
        ],
    )  # fmt: skip
    def test_returns_true_for_placeholder_names(self, name: str) -> None:
        from aiperf.common.tokenizer_fake_names import is_fake_model_name

        assert is_fake_model_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("Qwen/Qwen3-0.6B", id="hf-org-repo"),
            pytest.param("meta-llama/Llama-3-test-finetune", id="hf-with-test-in-suffix"),
            pytest.param("./mock-model", id="relative-dot-path"),
            pytest.param("../mock-model", id="parent-relative-path"),
            pytest.param("~/mock-model", id="home-path"),
            pytest.param("/abs/path/to/model", id="absolute-path"),
            pytest.param("C:\\models\\mock", id="windows-path"),
            pytest.param("gpt2", id="gpt2"),
            pytest.param("bert-base-uncased", id="bert"),
            pytest.param("Llama-3-8B-Instruct", id="real-llama"),
            pytest.param("", id="empty-string"),
            pytest.param("testing-real-models", id="testing-prefix-no-token"),
            pytest.param("contestant", id="contains-test-but-no-marker"),
        ],
    )  # fmt: skip
    def test_returns_false_for_real_or_path_like(self, name: str) -> None:
        from aiperf.common.tokenizer_fake_names import is_fake_model_name

        assert is_fake_model_name(name) is False


class TestIsTiktokenCached:
    """Tests for _is_tiktoken_cached — disk-cache hit check that skips subprocess pre-warm."""

    def _cache_dir_and_key(self, name: str, base_dir) -> tuple[str, str]:
        import hashlib

        from aiperf.common.tokenizer import _BUILTIN_ENCODING, _TIKTOKEN_ENCODING_URLS

        encoding = _BUILTIN_ENCODING if name == BUILTIN_TOKENIZER_NAME else name
        url = _TIKTOKEN_ENCODING_URLS[encoding]
        key = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
        return str(base_dir), key

    def test_returns_true_when_cache_file_exists(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.tokenizer import _is_tiktoken_cached

        cache_dir, key = self._cache_dir_and_key("o200k_base", tmp_path)
        (tmp_path / key).write_bytes(b"fake bpe")
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        assert _is_tiktoken_cached("o200k_base") is True

    def test_returns_false_when_cache_file_missing(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.tokenizer import _is_tiktoken_cached

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
        assert _is_tiktoken_cached("o200k_base") is False

    def test_builtin_maps_to_o200k_base(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Writing the o200k_base file should satisfy a 'builtin' cache check.
        from aiperf.common.tokenizer import _is_tiktoken_cached

        cache_dir, key = self._cache_dir_and_key("o200k_base", tmp_path)
        (tmp_path / key).write_bytes(b"fake bpe")
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        assert _is_tiktoken_cached(BUILTIN_TOKENIZER_NAME) is True

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("p50k_edit", id="p50k_edit-shares-p50k_base"),
            pytest.param("o200k_harmony", id="o200k_harmony-shares-o200k_base"),
        ],
    )  # fmt: skip
    def test_derived_encodings_check_base_url(
        self, name: str, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # p50k_edit and o200k_harmony reuse another encoding's BPE; their URL
        # mapping must point to the shared file.
        from aiperf.common.tokenizer import _is_tiktoken_cached

        cache_dir, key = self._cache_dir_and_key(name, tmp_path)
        (tmp_path / key).write_bytes(b"fake bpe")
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)

        assert _is_tiktoken_cached(name) is True

    def test_unknown_encoding_returns_false(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.tokenizer import _is_tiktoken_cached

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
        # Even with the dir set, an encoding we have no URL for falls through
        # to "not cached" so the caller runs the real pre-warm.
        assert _is_tiktoken_cached("not_a_real_encoding") is False

    def test_empty_cache_dir_disables_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tiktoken treats TIKTOKEN_CACHE_DIR="" as "disable caching" — we must
        # mirror that and report not-cached so the caller re-downloads.
        from aiperf.common.tokenizer import _is_tiktoken_cached

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")
        assert _is_tiktoken_cached("o200k_base") is False

    def test_falls_back_to_data_gym_cache_dir(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.common.tokenizer import _is_tiktoken_cached

        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        cache_dir, key = self._cache_dir_and_key("o200k_base", tmp_path)
        (tmp_path / key).write_bytes(b"fake bpe")
        monkeypatch.setenv("DATA_GYM_CACHE_DIR", cache_dir)

        assert _is_tiktoken_cached("o200k_base") is True


class TestTiktokenUrlsMatchTiktokenSource:
    """Drift detector: our hardcoded URLs must match what tiktoken would download."""

    def test_each_encoding_url_appears_in_tiktoken_source(self) -> None:
        import inspect

        from tiktoken_ext import openai_public

        from aiperf.common.tokenizer import _TIKTOKEN_ENCODING_URLS

        for name, url in _TIKTOKEN_ENCODING_URLS.items():
            # o200k_harmony reuses o200k_base's BPE; the URL must appear in
            # o200k_base's source (which o200k_harmony invokes).
            target = "o200k_base" if name == "o200k_harmony" else name
            # p50k_edit reuses p50k_base
            target = "p50k_base" if name == "p50k_edit" else target
            ctor = getattr(openai_public, target, None)
            assert ctor is not None, f"tiktoken_ext.openai_public missing {target}"
            assert url in inspect.getsource(ctor), (
                f"hardcoded URL for {name!r} ({url!r}) is no longer in "
                f"tiktoken_ext.openai_public.{target} — tiktoken changed the URL; "
                f"update _TIKTOKEN_ENCODING_URLS in src/aiperf/common/tokenizer.py"
            )
