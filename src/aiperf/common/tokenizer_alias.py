# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alias-resolution value objects and exceptions for the tokenizer module."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class AliasResolutionResult:
    """Result of tokenizer alias resolution."""

    resolved_name: str
    """The resolved name (canonical ID or original if not resolved)."""

    suggestions: list[tuple[str, int]] = field(default_factory=list)
    """List of (model_id, downloads) suggestions if ambiguous."""

    @property
    def is_ambiguous(self) -> bool:
        """Whether the name was ambiguous (has suggestions but no resolution)."""
        return len(self.suggestions) > 0


class AmbiguousTokenizerNameError(ValueError):
    """Raised when a tokenizer name is ambiguous and has multiple possible matches."""

    def __init__(self, name: str, suggestions: list[tuple[str, int]]) -> None:
        self.name = name
        self.suggestions = suggestions
        super().__init__(
            f"'{name}' is ambiguous. Did you mean: {', '.join(s[0] for s in suggestions[:3])}?"
        )
