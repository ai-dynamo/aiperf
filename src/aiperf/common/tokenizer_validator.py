# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early tokenizer validation before spawning services."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.config import UserConfig


def validate_tokenizer_early(
    user_config: UserConfig, logger: AIPerfLogger
) -> dict[str, str] | None:
    """Validate tokenizers before spawning services.

    Resolves aliases using fast API calls. Full tokenizer loading happens later.

    Args:
        user_config: Configuration containing tokenizer settings.
        logger: Logger for output.

    Returns:
        Mapping of model names to resolved tokenizer names, or None if skipped.

    Raises:
        SystemExit: If tokenizer validation fails.
    """
    from rich.console import Console

    from aiperf.common.tokenizer import Tokenizer
    from aiperf.common.tokenizer_display import (
        TokenizerDisplayEntry,
        display_tokenizer_ambiguous_name,
        log_tokenizer_validation_results,
    )
    from aiperf.plugin import plugins

    endpoint_meta = plugins.get_endpoint_metadata(user_config.endpoint.type)

    # Skip if using server token counts with non-synthetic data
    input_cfg = user_config.input
    is_synthetic = (
        input_cfg.public_dataset is None
        and input_cfg.custom_dataset_type is None
        and input_cfg.file is None
    )
    if user_config.endpoint.use_server_token_count and not is_synthetic:
        logger.debug("Using server token counts, skipping tokenizer validation")
        return None

    if not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input:
        logger.debug("Endpoint doesn't require tokenizer, skipping validation")
        return None

    # Determine tokenizers to validate
    tokenizer_cfg = user_config.tokenizer
    model_names = user_config.endpoint.model_names
    names = [tokenizer_cfg.name] if tokenizer_cfg.name else list(model_names)

    # Validate and resolve aliases
    console = Console()
    entries: list[TokenizerDisplayEntry] = []
    resolved: dict[str, str] = {}

    start = time.perf_counter()
    for name in names:
        try:
            result = Tokenizer.resolve_alias(name)
        except Exception as e:
            logger.error(f"Failed to validate tokenizer '{name}': {e}")
            sys.exit(1)

        if result.is_ambiguous:
            display_tokenizer_ambiguous_name(name, result.suggestions, console)
            sys.exit(1)

        resolved[name] = result.resolved_name
        entries.append(
            TokenizerDisplayEntry(
                original_name=name,
                resolved_name=result.resolved_name,
                was_resolved=name != result.resolved_name,
            )
        )

    log_tokenizer_validation_results(entries, logger, time.perf_counter() - start)

    # Build final mapping
    if tokenizer_cfg.name:
        return {model: resolved[tokenizer_cfg.name] for model in model_names}
    return resolved
