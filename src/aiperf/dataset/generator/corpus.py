# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt-corpus selector for synthetic and trace synthesis."""

from __future__ import annotations

from aiperf.common.enums import PromptCorpus
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator


def resolve_prompt_generator(
    *,
    corpus: PromptCorpus | str | None,
    default_corpus: PromptCorpus | str | None,
    tokenizer: Tokenizer,
    prompts: PromptConfig | None = None,
    prefix_prompts: PrefixPromptConfig | None = None,
) -> PromptGenerator | CodingContentGenerator:
    """Pick sonnet vs coding generator from authored corpus + loader default.

    Resolution: explicit ``corpus`` -> ``default_corpus`` -> ``PromptCorpus.SONNET``.
    """
    resolved = corpus or default_corpus or PromptCorpus.SONNET
    if resolved == PromptCorpus.CODING or resolved == "coding":
        return CodingContentGenerator(
            config=prompts or PromptConfig(),
            tokenizer=tokenizer,
        )
    return PromptGenerator(
        prompts=prompts,
        prefix_prompts=prefix_prompts,
        tokenizer=tokenizer,
    )
