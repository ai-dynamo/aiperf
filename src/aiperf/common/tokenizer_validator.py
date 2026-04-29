# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early tokenizer validation and preloading before spawning services."""

from __future__ import annotations

import asyncio
import os
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

    from aiperf.common.tokenizer import (
        BUILTIN_TOKENIZER_NAME,
        TIKTOKEN_ENCODING_NAMES,
        Tokenizer,
    )
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

    # tiktoken-backed tokenizers need no HF resolution
    if (
        tokenizer_cfg.name == BUILTIN_TOKENIZER_NAME
        or tokenizer_cfg.name in TIKTOKEN_ENCODING_NAMES
    ):
        logger.debug("Using tiktoken tokenizer, skipping HF alias resolution")
        return {model: tokenizer_cfg.name for model in model_names}

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


async def preload_tokenizers(
    resolved_names: dict[str, str] | None,
    trust_remote_code: bool = False,
    revision: str = "main",
    logger: AIPerfLogger | None = None,
) -> None:
    """Preload tokenizer files into HF disk cache before spawning child processes.

    Child processes call _is_hf_cached() inside Tokenizer.from_pretrained().
    When True, they use local_files_only=True and make zero HF network calls.

    Args:
        resolved_names: Mapping of model names to resolved tokenizer names.
                        If None or empty (validation was skipped), this is a no-op.
        trust_remote_code: Whether to trust remote code when loading.
        revision: The specific model version to use.
        logger: Optional logger for progress output.
    """
    from pathlib import Path

    from aiperf.common.tokenizer import (
        BUILTIN_TOKENIZER_NAME,
        TIKTOKEN_ENCODING_NAMES,
        Tokenizer,
        _is_hf_cached,
    )

    if not resolved_names:
        if logger:
            logger.debug("Tokenizer preload skipped: validation was not run")
        return

    names_to_load: list[str] = []
    for name in set(resolved_names.values()):
        # tiktoken/builtin: no HF download needed
        if name == BUILTIN_TOKENIZER_NAME or name in TIKTOKEN_ENCODING_NAMES:
            if logger:
                logger.debug(
                    f"Tokenizer preload skipped for '{name}': tiktoken backend"
                )
            continue
        # Local path: files already on disk
        p = Path(name)
        if p.is_absolute() or name.startswith(("./", "../")) or p.is_dir():
            if logger:
                logger.debug(f"Tokenizer preload skipped for '{name}': local path")
            continue
        # Already in HF disk cache
        if _is_hf_cached(name, revision):
            if logger:
                logger.debug(
                    f"Tokenizer preload skipped for '{name}': already in HF cache"
                )
            continue
        names_to_load.append(name)

    if not names_to_load:
        if logger:
            logger.debug(
                "Tokenizer preload: all tokenizers already cached, no download needed"
            )
        _enable_hf_offline_mode(logger)
        return

    if logger:
        logger.info(f"Preloading {len(names_to_load)} tokenizer(s) into local cache...")

    failed: list[str] = []
    for name in names_to_load:
        if logger:
            logger.info(f"  Caching tokenizer: {name}")
        try:
            # Discard result — side effect is populating the HF disk cache so
            # child processes find it cached and skip all network calls.
            await asyncio.to_thread(
                Tokenizer.from_pretrained,
                name,
                trust_remote_code=trust_remote_code,
                revision=revision,
                resolve_alias=False,  # already resolved by validate_tokenizer_early
            )
        except Exception:  # noqa: BLE001
            failed.append(name)

    if failed:
        if logger:
            names_str = ", ".join(f"'{n}'" for n in failed)
            logger.warning(
                f"Failed to preload {len(failed)} tokenizer(s): {names_str}. "
                "Child processes will attempt to load them themselves."
            )
    else:
        _enable_hf_offline_mode(logger)


def _enable_hf_offline_mode(logger: AIPerfLogger | None = None) -> None:
    """Set HF environment variables so spawned processes never make network calls."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if logger:
        logger.debug("Enabled HF offline mode for child processes")
