# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tokenizer kwarg override detection and application.

Tokenizers like Kimi use non-standard kwargs (e.g. `allow_special_tokens`
instead of `add_special_tokens`). Passing unsupported kwargs triggers the
slow `PreTrainedTokenizer.super()` fallback. These tests verify that
`_supports_kwarg` and `_apply_kwarg_overrides` correctly detect and adapt.
"""

import pytest

from aiperf.common.tokenizer import Tokenizer, _supports_kwarg

# -- Fake tokenizer backends for testing --


class StandardTokenizerBackend:
    """Mimics a standard HuggingFace tokenizer (e.g. Qwen)."""

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> list[int]:
        return list(range(len(text.split())))

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False, **kwargs
    ) -> str:
        return " ".join(f"t{i}" for i in token_ids)

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs) -> dict:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    bos_token_id = 1
    eos_token_id = 2


class KimiLikeTokenizerBackend:
    """Mimics Kimi's TikTokenTokenizer: uses allow_special_tokens, no skip_special_tokens."""

    def encode(
        self, text: str, allow_special_tokens: bool = True, **kwargs
    ) -> list[int]:
        if kwargs:
            raise TypeError(
                f"Unexpected kwargs would trigger slow super().encode: {kwargs}"
            )
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int] | int, **kwargs) -> str:
        if kwargs:
            raise TypeError(
                f"Unexpected kwargs would trigger slow super().decode: {kwargs}"
            )
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(f"t{i}" for i in token_ids)

    def __call__(self, text: str, allow_special_tokens: bool = True, **kwargs) -> dict:
        return {
            "input_ids": self.encode(text, allow_special_tokens=allow_special_tokens)
        }

    bos_token_id = 1
    eos_token_id = 2


class MinimalDecodeTokenizerBackend:
    """Tokenizer with standard encode but minimal decode (no skip_special_tokens)."""

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], **kwargs) -> str:
        if kwargs:
            raise TypeError(f"Unexpected kwargs: {kwargs}")
        return " ".join(f"t{i}" for i in token_ids)

    bos_token_id = 1
    eos_token_id = 2


class KwargsOnlyTokenizerBackend:
    """Tokenizer that only accepts **kwargs (no named params beyond self/text)."""

    def encode(self, text, **kwargs):
        return [0]

    def decode(self, token_ids, **kwargs):
        return "decoded"

    bos_token_id = 0
    eos_token_id = 0


class MismatchedCallEncodeBackend:
    """Backend where encode uses allow_special_tokens but __call__ uses add_special_tokens."""

    def encode(
        self, text: str, allow_special_tokens: bool = True, **kwargs
    ) -> list[int]:
        if kwargs:
            raise TypeError(f"Unexpected kwargs in encode: {kwargs}")
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"t{i}" for i in token_ids)

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs) -> dict:
        if kwargs:
            raise TypeError(f"Unexpected kwargs in __call__: {kwargs}")
        return {"input_ids": self.encode(text)}

    bos_token_id = 1
    eos_token_id = 2


# -- _supports_kwarg tests --


class TestSupportsKwarg:
    def test_detects_named_param(self):
        backend = StandardTokenizerBackend()
        assert _supports_kwarg(backend, "encode", "add_special_tokens") is True

    def test_rejects_missing_param(self):
        backend = StandardTokenizerBackend()
        assert _supports_kwarg(backend, "encode", "allow_special_tokens") is False

    def test_detects_allow_special_tokens(self):
        backend = KimiLikeTokenizerBackend()
        assert _supports_kwarg(backend, "encode", "allow_special_tokens") is True

    def test_rejects_add_special_tokens_on_kimi(self):
        backend = KimiLikeTokenizerBackend()
        assert _supports_kwarg(backend, "encode", "add_special_tokens") is False

    def test_detects_skip_special_tokens_on_standard(self):
        backend = StandardTokenizerBackend()
        assert _supports_kwarg(backend, "decode", "skip_special_tokens") is True

    def test_rejects_skip_special_tokens_on_kimi(self):
        backend = KimiLikeTokenizerBackend()
        assert _supports_kwarg(backend, "decode", "skip_special_tokens") is False

    def test_missing_method_returns_false(self):
        backend = StandardTokenizerBackend()
        assert _supports_kwarg(backend, "nonexistent_method", "anything") is False

    def test_kwargs_only_method(self):
        backend = KwargsOnlyTokenizerBackend()
        assert _supports_kwarg(backend, "encode", "add_special_tokens") is False
        assert _supports_kwarg(backend, "encode", "allow_special_tokens") is False

    def test_non_introspectable_callable_returns_false(self):
        """C extension methods that can't be introspected should return False."""

        class WithBuiltinMethod:
            encode = object.__init_subclass__  # raises ValueError in inspect.signature

        assert _supports_kwarg(WithBuiltinMethod(), "encode", "anything") is False

    @pytest.mark.parametrize(
        ("method", "kwarg", "expected"),
        [
            ("encode", "text", True),
            ("encode", "add_special_tokens", True),
            ("decode", "token_ids", True),
            ("decode", "skip_special_tokens", True),
            ("encode", "nonexistent", False),
        ],
    )
    def test_standard_backend_parametrized(self, method, kwarg, expected):
        backend = StandardTokenizerBackend()
        assert _supports_kwarg(backend, method, kwarg) is expected


# -- _apply_kwarg_overrides tests --


class TestApplyKwargOverrides:
    @staticmethod
    def _make_tokenizer(backend) -> Tokenizer:
        tok = Tokenizer()
        tok._tokenizer = backend
        tok._apply_kwarg_overrides()
        return tok

    def test_standard_tokenizer_keeps_defaults(self):
        tok = self._make_tokenizer(StandardTokenizerBackend())
        assert tok._encode_args == {"add_special_tokens": False}
        assert tok._call_args == {"add_special_tokens": False}
        assert tok._decode_args == {"skip_special_tokens": True}

    def test_kimi_like_overrides_encode_and_call_args(self):
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        assert tok._encode_args == {"allow_special_tokens": False}
        assert tok._call_args == {"allow_special_tokens": False}

    def test_kimi_like_clears_decode_args(self):
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        assert tok._decode_args == {}

    def test_minimal_decode_clears_decode_args(self):
        tok = self._make_tokenizer(MinimalDecodeTokenizerBackend())
        assert tok._encode_args == {"add_special_tokens": False}
        assert tok._decode_args == {}

    def test_mismatched_call_encode_sets_args_independently(self):
        """When encode uses allow_special_tokens but __call__ uses add_special_tokens."""
        tok = self._make_tokenizer(MismatchedCallEncodeBackend())
        assert tok._encode_args == {"allow_special_tokens": False}
        assert tok._call_args == {"add_special_tokens": False}
        assert tok._decode_args == {"skip_special_tokens": True}

    def test_none_tokenizer_is_noop(self):
        tok = Tokenizer()
        tok._apply_kwarg_overrides()
        assert tok._encode_args == {"add_special_tokens": False}
        assert tok._call_args == {"add_special_tokens": False}
        assert tok._decode_args == {"skip_special_tokens": True}


# -- End-to-end: encode/decode through Tokenizer wrapper --


class TestKwargOverridesEndToEnd:
    @staticmethod
    def _make_tokenizer(backend) -> Tokenizer:
        tok = Tokenizer()
        tok._tokenizer = backend
        tok._apply_kwarg_overrides()
        return tok

    def test_standard_encode_passes_correct_kwargs(self):
        tok = self._make_tokenizer(StandardTokenizerBackend())
        result = tok.encode("hello world")
        assert isinstance(result, list)

    def test_standard_decode_passes_correct_kwargs(self):
        tok = self._make_tokenizer(StandardTokenizerBackend())
        result = tok.decode([0, 1, 2])
        assert isinstance(result, str)

    def test_kimi_encode_does_not_raise(self):
        """Kimi backend raises TypeError if unexpected kwargs are passed."""
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        result = tok.encode("hello world")
        assert isinstance(result, list)

    def test_kimi_decode_does_not_raise(self):
        """Kimi backend raises TypeError if unexpected kwargs are passed."""
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        result = tok.decode([0, 1, 2])
        assert isinstance(result, str)

    def test_kimi_call_does_not_raise(self):
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        result = tok("hello world")
        assert "input_ids" in result

    def test_standard_encode_without_override_would_fail_on_kimi(self):
        """Verify that without overrides, Kimi backend rejects add_special_tokens."""
        tok = Tokenizer()
        tok._tokenizer = KimiLikeTokenizerBackend()
        # Don't call _apply_kwarg_overrides - defaults still have add_special_tokens
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            tok.encode("hello world")

    def test_standard_decode_without_override_would_fail_on_kimi(self):
        """Verify that without overrides, Kimi backend rejects skip_special_tokens."""
        tok = Tokenizer()
        tok._tokenizer = KimiLikeTokenizerBackend()
        with pytest.raises(TypeError, match="Unexpected kwargs"):
            tok.decode([0, 1, 2])

    def test_user_kwargs_override_defaults(self):
        """User-provided kwargs should override the defaults."""
        tok = self._make_tokenizer(StandardTokenizerBackend())
        result = tok.encode("hello", add_special_tokens=True)
        assert isinstance(result, list)

    def test_kimi_user_kwargs_override_defaults(self):
        tok = self._make_tokenizer(KimiLikeTokenizerBackend())
        result = tok.encode("hello", allow_special_tokens=True)
        assert isinstance(result, list)

    def test_mismatched_call_encode_does_not_raise(self):
        """Backend with different kwargs on encode vs __call__ should work."""
        tok = self._make_tokenizer(MismatchedCallEncodeBackend())
        assert isinstance(tok.encode("hello world"), list)
        assert "input_ids" in tok("hello world")
