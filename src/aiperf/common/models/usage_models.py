# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar


class Usage(dict):
    """Usage wraps API-reported token consumption data with a unified interface.

    Inference frameworks like vLLM, TensorRT-LLM, and TGI return token usage
    in varying formats (prompt_tokens vs input_tokens, completion_tokens vs
    output_tokens). Usage normalizes these differences through properties while
    preserving the full underlying dictionary for framework-specific fields.

    Inherits from dict so it serializes as a plain dict and accepts any dict
    structure, allowing framework-specific fields to pass through unchanged.
    """

    PROMPT_DETAILS_KEYS: ClassVar[list[str]] = [
        "prompt_tokens_details",
        "input_tokens_details",
    ]
    COMPLETION_DETAILS_KEYS: ClassVar[list[str]] = [
        "completion_tokens_details",
        "output_tokens_details",
    ]

    @property
    def prompt_tokens(self) -> int | None:
        """Get prompt/input token count from API usage dict."""
        return self.get("prompt_tokens") or self.get("input_tokens")

    @property
    def completion_tokens(self) -> int | None:
        """Get completion/output token count from API usage dict."""
        return self.get("completion_tokens") or self.get("output_tokens")

    @property
    def total_tokens(self) -> int | None:
        """Get total token count from API usage dict."""
        return self.get("total_tokens")

    @property
    def reasoning_tokens(self) -> int | None:
        """Get reasoning tokens from nested details (reasoning models).

        Reasoning tokens are nested in completion_tokens_details.reasoning_tokens
        or output_tokens_details.reasoning_tokens.
        """
        for details_key in self.COMPLETION_DETAILS_KEYS:
            if reasoning_tokens := self.get(details_key, {}).get("reasoning_tokens"):
                return reasoning_tokens
        return None
