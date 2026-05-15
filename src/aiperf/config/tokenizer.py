# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer configuration models.

Split out of ``models.py`` so each config section lives in its own file.
Re-exported via :mod:`aiperf.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig


@dataclass(frozen=True)
class TokenizerDefaults:
    NAME = None
    REVISION = "main"
    TRUST_REMOTE_CODE = False


class TokenizerConfig(BaseConfig):
    """
    Tokenizer configuration for token counting and prompt generation.

    AIPerf uses a HuggingFace tokenizer for accurate token counting,
    which is essential for ISL/OSL enforcement and metrics calculation.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[
        str | None,
        Field(
            default=None,
            description="HuggingFace tokenizer identifier, local filesystem path, or `builtin` "
            "for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). "
            "Should match the model's tokenizer for accurate token counts. "
            "If `--tokenizer` is not set and the model name looks like an obvious placeholder "
            "(e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically "
            "and emits a warning. "
            "Example: 'meta-llama/Llama-3.1-8B-Instruct'",
        ),
    ]

    revision: Annotated[
        str,
        Field(
            default="main",
            description="Model revision to use: branch name, tag, or commit hash. "
            "Use for version pinning to ensure reproducibility.",
        ),
    ]

    trust_remote_code: Annotated[
        bool,
        Field(
            default=False,
            description="Allow execution of custom tokenizer code from the repository. "
            "Required for some models but poses security risk. "
            "Only enable for trusted sources.",
        ),
    ]

    resolved_names: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            exclude=True,
            description="Pre-resolved tokenizer names from alias resolution. "
            "[runtime-only; populated by the CLI or WorkerGroupManager after "
            "tokenizer validation. Excluded from JSON/YAML serialization. Do not "
            "set in a CR spec — any user value is ignored.]",
        ),
    ]

    def get_tokenizer_name_for_model(self, model_name: str) -> str:
        """Get the tokenizer name to use for a given model.

        Resolution order:
        1. Pre-resolved name from `resolved_names` (set by CLI after alias resolution)
        2. Explicitly configured tokenizer name
        3. The model name itself (assumes model repo contains tokenizer)
        """
        if self.resolved_names and model_name in self.resolved_names:
            return self.resolved_names[model_name]
        return self.name or model_name

    @property
    def should_resolve_alias(self) -> bool:
        """Whether alias resolution should be performed when loading tokenizers.

        Returns False if `resolved_names` is set (CLI already resolved aliases),
        True otherwise to enable HuggingFace Hub alias resolution.
        """
        return self.resolved_names is None


__all__ = [
    "TokenizerConfig",
    "TokenizerDefaults",
]
