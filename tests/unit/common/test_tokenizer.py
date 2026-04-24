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
