# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verbatim system-prompt configuration shared by every dataset variant.

``SyntheticDataset``, ``FileDataset``, and ``PublicDataset`` each inherit
``SystemPromptMixin`` so a user-supplied system prompt works identically
regardless of where the conversations came from.

This is the content-valued counterpart to
``PrefixPromptConfig.shared_system_length``: both fill
``Conversation.system_message``, one by naming a token length for synthetic
filler, the other by naming the exact text. They are mutually exclusive
(enforced in ``AIPerfConfig``, which can see both blocks at once).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, PrivateAttr, model_validator

from aiperf.common.path_safety import safe_read_template_path
from aiperf.config.base import BaseConfig


class SystemPromptMixin(BaseConfig):
    """Adds ``system_prompt`` / ``system_prompt_file`` to a dataset variant."""

    system_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Verbatim system prompt text, identical across every conversation. "
            "Emitted as a system-role message ahead of all turns. "
            "When the dataset already carries its own system message, this text is "
            "prepended to it rather than replacing it. "
            "Tokens are additive: --isl continues to size the generated user prompt only. "
            "Mutually exclusive with system_prompt_file and with "
            "prefix_prompts.shared_system_length/pool_size/length.",
        ),
    ]

    system_prompt_file: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to a UTF-8 text file holding the verbatim system prompt. "
            "Preferred over system_prompt for real production prompts, which are long "
            "enough that shell quoting mangles them. Read once at config-validation "
            "time so a missing or unreadable file fails at startup. "
            "Mutually exclusive with system_prompt.",
        ),
    ]

    # Resolved text for whichever source was configured. Populated once during
    # validation so conversation-building never re-reads the file.
    _resolved_system_prompt: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_system_prompt(self) -> Self:
        """Reject conflicting/empty sources and cache the resolved text.

        Reading here rather than at conversation-build time means an unreadable
        path is a startup error instead of a mid-benchmark one, and the file is
        read exactly once instead of once per conversation.
        """
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError(
                "system_prompt (--system-prompt) and system_prompt_file "
                "(--system-prompt-file) are mutually exclusive; set exactly one."
            )

        if self.system_prompt is not None:
            if not self.system_prompt.strip():
                raise ValueError(
                    "system_prompt (--system-prompt) is empty or whitespace-only. "
                    "Omit the option entirely to run without a system prompt."
                )
            self._resolved_system_prompt = self.system_prompt
            return self

        if self.system_prompt_file is not None:
            text = safe_read_template_path(str(self.system_prompt_file))
            if text is None:
                raise ValueError(
                    f"system_prompt_file (--system-prompt-file) could not be read: "
                    f"{self.system_prompt_file}. Expected a readable regular UTF-8 "
                    "text file with no symlinked path component."
                )
            if not text.strip():
                raise ValueError(
                    f"system_prompt_file (--system-prompt-file) is empty or "
                    f"whitespace-only: {self.system_prompt_file}. Omit the option "
                    "entirely to run without a system prompt."
                )
            self._resolved_system_prompt = text
            return self

        return self

    @property
    def resolved_system_prompt(self) -> str | None:
        """The verbatim system prompt text, or ``None`` when unconfigured.

        Resolved from whichever of ``system_prompt`` / ``system_prompt_file``
        was set. Composers read this rather than the raw fields so the file
        vs inline distinction stays confined to validation.
        """
        return self._resolved_system_prompt
