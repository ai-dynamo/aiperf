# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to import BatchEncoding only during static type checks
if TYPE_CHECKING:
    from transformers import BatchEncoding

from aiperf.common.exceptions import InitializationError, NotInitializedError

# Silence tokenizer warning on import and first use
with (
    contextlib.redirect_stdout(io.StringIO()) as _,
    contextlib.redirect_stderr(io.StringIO()),
):
    from transformers import AutoTokenizer


class Tokenizer:
    """
    This class provides a simplified interface for using Huggingface
    tokenizers, with default arguments for common operations.
    """

    def __init__(self) -> None:
        """
        Initialize the tokenizer with default values for call, encode, and decode.
        """
        self._tokenizer = None
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
    ) -> "Tokenizer":
        """
        Factory to load a tokenizer for the given pretrained model name.

        Args:
            name: The name or path of the pretrained tokenizer model.
            trust_remote_code: Whether to trust remote code when loading the tokenizer.
            revision: The specific model version to use.
        """
        try:
            tokenizer_cls = cls()
            tokenizer_cls._tokenizer = AutoTokenizer.from_pretrained(
                name, trust_remote_code=trust_remote_code, revision=revision
            )
        except Exception as e:
            raise InitializationError(e) from e
        return tokenizer_cls

    def __call__(self, text, **kwargs) -> "BatchEncoding":
        """
        Call the underlying Huggingface tokenizer with default arguments,
        which can be overridden by kwargs.

        Args:
            text: The input text to tokenize.

        Returns:
            A BatchEncoding object containing the tokenized output.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer(text, **{**self._call_args, **kwargs})

    def encode(self, text, **kwargs) -> list[int]:
        """
        Encode the input text into a list of token IDs.

        This method calls the underlying Huggingface tokenizer's encode
        method with default arguments, which can be overridden by kwargs.

        Args:
            text: The input text to encode.

        Returns:
            A list of token IDs.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.encode(text, **{**self._encode_args, **kwargs})

    def decode(self, token_ids, **kwargs) -> str:
        """
        Decode a list of token IDs back into a string.

        This method calls the underlying Huggingface tokenizer's decode
        method with default arguments, which can be overridden by kwargs.

        Args:
            token_ids: A list of token IDs to decode.

        Returns:
            The decoded string.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.decode(token_ids, **{**self._decode_args, **kwargs})

    @property
    def bos_token_id(self) -> int:
        """
        Return the beginning-of-sequence (BOS) token ID.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """
        Return the end-of-sequence (EOS) token ID.
        """
        if self._tokenizer is None:
            raise NotInitializedError("Tokenizer is not initialized.")
        return self._tokenizer.eos_token_id

    def __repr__(self) -> str:
        """
        Return a string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__repr__()

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__str__()
