# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simplified fake tokenizer for testing.

Unlike the other fakes in this package, FakeTokenizer is not factory-registered.
Use monkeypatching or dependency injection to substitute it for production
tokenizers (e.g., HuggingFace AutoTokenizer).

Provides deterministic encode/decode based on character count (~4 chars per token)
rather than actual BPE/WordPiece tokenization.
"""

TOKEN = "tok$"
TOKEN_LEN = len(TOKEN)


class FakeTokenizer:
    """Simplified tokenizer for testing (test double: Fake).

    Provides deterministic encode/decode based on character count rather than
    actual tokenization. Each ~4 characters becomes one token. Not registered
    with a factory - use monkeypatching or dependency injection to substitute.
    """

    def __init__(self):
        # This is used by certain services to print details about the internal tokenizer.
        self._tokenizer = self
        print(
            "*** Using FakeTokenizer to bypass tokenization. This is for testing only. ***"
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        trust_remote_code: bool = False,
        revision: str = "main",
        resolve_alias: bool = True,
    ) -> "FakeTokenizer":
        """Return a FakeTokenizer."""
        return cls()

    def __call__(self, text, **kwargs) -> dict:
        return {"input_ids": self.encode(text)}

    def encode(self, text, **kwargs) -> list[int]:
        """Encode text to token IDs."""
        return [i for i in range(round(len(text) / TOKEN_LEN))]

    def decode(self, token_ids, **kwargs) -> str:
        """Decode token IDs to string."""
        return TOKEN * len(token_ids)

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def block_separation_token_id(self) -> int:
        return 1
