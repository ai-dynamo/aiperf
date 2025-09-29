# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer service for handling different model tokenizers."""

import contextlib
import io
import logging

# Silence tokenizer warning on import and first use
with (
    contextlib.redirect_stdout(io.StringIO()) as _,
    contextlib.redirect_stderr(io.StringIO()),
):
    from transformers import AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TokenizerService:
    """Service for managing tokenizers for different models."""

    def __init__(self):
        self._tokenizers: dict[str, PreTrainedTokenizer] = {}
        self._fallback_tokenizer: str | None = None

    def load_tokenizers(self, model_names: list[str]) -> None:
        """Pre-load tokenizers for one or more models.

        Args:
            model_names: List of model names to load tokenizers for
        """
        for model_name in model_names:
            try:
                logger.info(f"Pre-loading tokenizer for model: {model_name}")
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception as e:
                logger.exception(f"Failed to load tokenizer for {model_name}: {e}")

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Get or create a tokenizer for the specified model."""
        if model_name not in self._tokenizers:
            if self._fallback_tokenizer not in self._tokenizers:
                raise ValueError(
                    f"No tokenizer loaded for {model_name} or {self._fallback_tokenizer}"
                )
            model_name = self._fallback_tokenizer

        return self._tokenizers[model_name]

    def tokenize(self, text: str, model_name: str) -> list[str]:
        """Tokenize text using the specified model's tokenizer."""
        tokenizer = self.get_tokenizer(model_name)

        # Encode and decode to get actual tokens as they would appear
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = []

        for token_id in token_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            tokens.append(token_text)

        return tokens

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count the number of tokens in the text for the specified model."""
        tokenizer = self.get_tokenizer(model_name)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def set_fallback_tokenizer(self, fallback_tokenizer: str) -> None:
        """Set the fallback tokenizer to use if the requested tokenizer is not found."""
        self._fallback_tokenizer = fallback_tokenizer


# Global tokenizer service instance
tokenizer_service = TokenizerService()
