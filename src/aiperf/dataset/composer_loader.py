# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for selecting a dataset composer and loading conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.plugin import plugins
from aiperf.plugin.enums import ComposerType, PluginType

if TYPE_CHECKING:
    from aiperf.common.enums import ConversationContextMode
    from aiperf.common.models import Conversation
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config import BenchmarkRun


ConversationLoadResult = tuple[list["Conversation"], "ConversationContextMode | None"]


def _is_rankings_endpoint(endpoint_type: str) -> bool:
    return "rankings" in endpoint_type.lower()


def _composer_for(
    composer_type: ComposerType,
    run: BenchmarkRun,
    tokenizer: Tokenizer | None,
):
    ComposerClass = plugins.get_class(PluginType.DATASET_COMPOSER, composer_type)
    return ComposerClass(run=run, tokenizer=tokenizer)


async def load_public_dataset(
    run: BenchmarkRun, tokenizer: Tokenizer | None
) -> ConversationLoadResult:
    """Load conversations from a public-dataset composer."""
    composer = _composer_for(ComposerType.PUBLIC, run, tokenizer)
    conversations = await composer.create_dataset_async()
    return conversations, composer.get_default_context_mode()


def load_custom_dataset(
    run: BenchmarkRun, tokenizer: Tokenizer | None
) -> ConversationLoadResult:
    """Load conversations from a user-supplied (file) dataset composer."""
    composer = _composer_for(ComposerType.CUSTOM, run, tokenizer)
    conversations = composer.create_dataset()
    return conversations, composer.get_default_context_mode()


def load_synthetic_dataset(
    run: BenchmarkRun, tokenizer: Tokenizer | None
) -> ConversationLoadResult:
    """Load conversations from the synthetic composer (rankings-aware)."""
    composer_type = (
        ComposerType.SYNTHETIC_RANKINGS
        if _is_rankings_endpoint(run.cfg.endpoint.type)
        else ComposerType.SYNTHETIC
    )
    composer = _composer_for(composer_type, run, tokenizer)
    conversations = composer.create_dataset()
    return conversations, composer.get_default_context_mode()


async def load_conversations_for_run(
    run: BenchmarkRun, tokenizer: Tokenizer | None
) -> ConversationLoadResult:
    """Pick the composer based on the dataset config and return conversations."""
    from aiperf.config.resolved import is_file_dataset, is_public_dataset

    dataset_config = run.cfg.get_default_dataset()
    if is_public_dataset(dataset_config):
        return await load_public_dataset(run, tokenizer)
    if is_file_dataset(dataset_config):
        return load_custom_dataset(run, tokenizer)
    return load_synthetic_dataset(run, tokenizer)
