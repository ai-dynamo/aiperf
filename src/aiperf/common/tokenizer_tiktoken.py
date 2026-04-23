# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter that exposes a tiktoken.Encoding through the HuggingFace tokenizer API."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken


class TiktokenAdapter:
    """Adapts tiktoken.Encoding to the interface expected by Tokenizer._tokenizer."""

    def __init__(self, encoding: "tiktoken.Encoding") -> None:
        self._encoding = encoding

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int:
        return self._encoding.eot_token

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._encoding.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self._encoding.decode(token_ids)

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": self.encode(text)}

    def __repr__(self) -> str:
        return f"TiktokenAdapter({self._encoding.name})"

    def __str__(self) -> str:
        return repr(self)
