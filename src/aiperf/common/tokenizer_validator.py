# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early tokenizer validation and HuggingFace cache warming.

This module runs before any service processes are spawned and has two jobs:

1. **Alias resolution** -- fast HF Hub API calls to resolve short names
   (e.g. "qwen3-0.6b") to canonical repo IDs. Runs in the parent process since
   it's lightweight and network-only.

2. **Cache warming** -- full ``Tokenizer.from_pretrained`` calls that
   download model files into the HF disk cache. These run in a
   ``ProcessPoolExecutor`` so the parent process never imports the
   Rust-backed tokenizer internals that create threads and other state
   incompatible with ``fork()``. Once the cache is warm, child service
   processes set ``HF_HUB_OFFLINE=1`` (see ``bootstrap.py``) and load
   from disk with zero network traffic, eliminating the thundering-herd
   problem that occurs when N record processors all hit the Hub concurrently.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry

if TYPE_CHECKING:
    from rich.console import Console

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkConfig


# ---------------------------------------------------------------------------
# Default registry hook
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: TokenizerBundleRegistry | None = None


def set_default_registry(registry: TokenizerBundleRegistry | None) -> None:
    """Module-level hook so the FastAPI app and validator share one registry."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry


def get_default_registry() -> TokenizerBundleRegistry | None:
    """Return the registry installed by ``set_default_registry``, if any."""
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# Subprocess cache warming
# ---------------------------------------------------------------------------


def _init_worker(log_level: str) -> None:
    """ProcessPoolExecutor initializer: bootstrap logging in each worker."""
    from aiperf.common.logging import setup_subprocess_logging

    setup_subprocess_logging(log_level)


def _cache_tokenizer(
    name: str, trust_remote_code: bool, revision: str
) -> tuple[str, float]:
    """Subprocess target: download one tokenizer into the HF disk cache.

    Must be module-level so ``ProcessPoolExecutor`` can pickle it.
    """
    from aiperf.common.tokenizer import Tokenizer

    begin = time.perf_counter()
    Tokenizer.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        resolve_alias=False,
    )
    return name, time.perf_counter() - begin


def _partition_cached_names(
    names: set[str],
    *,
    revision: str,
    logger: AIPerfLogger,
) -> tuple[set[str], set[str]]:
    """Split *names* into (already_cached, to_fetch) using on-disk cache state.

    Cache hits are registered with the bundle registry here so the downstream
    TokenizerRouter/WGM can ship snapshot dirs to sibling pods without the
    subprocess fetch step.
    """
    from pathlib import Path

    from aiperf.common.tokenizer import _is_hf_cached

    already_cached: set[str] = set()
    to_fetch: set[str] = set()
    for name in names:
        if _is_hf_cached(name, revision):
            already_cached.add(name)
        else:
            to_fetch.add(name)

    if already_cached:
        from huggingface_hub import snapshot_download

        logger.info(f"HF cache hit (skipping prefetch): {sorted(already_cached)}")
        registry = _DEFAULT_REGISTRY
        if registry is not None:
            for name in already_cached:
                registry.register_pending(name)
                snapshot_dir = Path(
                    snapshot_download(
                        repo_id=name,
                        revision=revision,
                        repo_type="model",
                        local_files_only=True,
                    )
                )
                registry.mark_ready(name, snapshot_dir)

    return already_cached, to_fetch


def _prefetch_tokenizers(
    names: set[str],
    *,
    trust_remote_code: bool,
    revision: str,
    logger: AIPerfLogger,
    console: Console,
) -> None:
    """Cache unique tokenizers concurrently, one subprocess each.

    On failure, displays a rich diagnostic panel and exits.
    """
    import logging as _logging
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    from aiperf.common.models import ErrorDetails
    from aiperf.common.tokenizer_display import display_tokenizer_validation_error

    _, to_fetch = _partition_cached_names(names, revision=revision, logger=logger)
    if not to_fetch:
        return

    names = to_fetch
    count = len(names)
    log_level = _logging.getLevelName(_logging.getLogger().getEffectiveLevel())
    logger.info(
        f"Prefetching {count} tokenizer{'s' if count > 1 else ''} into HF cache..."
    )
    registry = _DEFAULT_REGISTRY
    if registry is not None:
        for name in names:
            registry.register_pending(name)
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=count,
        initializer=_init_worker,
        initargs=(log_level,),
    ) as pool:
        futures = {
            pool.submit(_cache_tokenizer, n, trust_remote_code, revision): n
            for n in names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, elapsed = future.result()
                logger.info(f"  Cached {name} ({elapsed:.2f}s)")
                if registry is not None:
                    from huggingface_hub import snapshot_download

                    snapshot_dir = Path(
                        snapshot_download(
                            repo_id=name,
                            revision=revision,
                            repo_type="model",
                            local_files_only=True,
                        )
                    )
                    registry.mark_ready(name, snapshot_dir)
            except Exception as e:  # noqa: BLE001 - tokenizer prefetch may raise arbitrary HF/network/subprocess errors; surface via rich panel
                details = ErrorDetails.from_exception(e)
                display_tokenizer_validation_error(
                    getattr(e, "tokenizer_name", None) or name,
                    cause_chain=details.cause_chain,
                    error_message=details.message,
                    cause_message=details.cause,
                    console=console,
                )
                sys.exit(1)
    total = time.perf_counter() - start
    logger.info(f"{count} tokenizer{'s' if count > 1 else ''} cached • {total:.1f}s")


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


def _resolve_aliases(
    names: list[str], logger: AIPerfLogger, console: Console
) -> dict[str, str]:
    """Resolve tokenizer names to canonical HF repo IDs.

    Exits on ambiguous or failed lookups.

    Returns:
        Mapping of ``{original_name: resolved_name}``.
    """
    from aiperf.common.tokenizer import (
        Tokenizer,
    )
    from aiperf.common.tokenizer_display import (
        TokenizerDisplayEntry,
        display_tokenizer_ambiguous_name,
        log_tokenizer_validation_results,
    )

    entries: list[TokenizerDisplayEntry] = []
    resolved: dict[str, str] = {}

    start = time.perf_counter()
    for name in names:
        try:
            result = Tokenizer.resolve_alias(name)
        except Exception as e:  # noqa: BLE001 - validator must surface any HF/network failure to the user as a startup error
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
    return resolved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_tokenizer_early(
    config: BenchmarkConfig, logger: AIPerfLogger
) -> dict[str, str] | None:
    """Resolve aliases and warm the HF cache (see module docstring).

    Returns:
        Mapping of ``{model_name: resolved_tokenizer_name}``, or ``None``
        if tokenizer validation was skipped (e.g. server token counts).

    Raises:
        SystemExit: If alias resolution, ambiguity check, or caching fails.
    """
    from rich.console import Console

    from aiperf.common.enums import DatasetType
    from aiperf.common.tokenizer import (
        BUILTIN_TOKENIZER_NAME,
        TIKTOKEN_ENCODING_NAMES,
    )
    from aiperf.plugin import plugins

    endpoint_meta = plugins.get_endpoint_metadata(config.endpoint.type)

    # Skip if using server token counts with non-synthetic data
    default_dataset = config.get_default_dataset()
    is_synthetic = getattr(default_dataset, "type", None) == DatasetType.SYNTHETIC
    if config.endpoint.use_server_token_count and not is_synthetic:
        logger.debug("Using server token counts, skipping tokenizer validation")
        return None

    if not endpoint_meta.produces_tokens and not endpoint_meta.tokenizes_input:
        logger.debug("Endpoint doesn't require tokenizer, skipping validation")
        return None

    tokenizer_cfg = config.tokenizer
    model_names = config.get_model_names()
    names = (
        [tokenizer_cfg.name]
        if tokenizer_cfg and tokenizer_cfg.name
        else list(model_names)
    )

    if tokenizer_cfg and (
        tokenizer_cfg.name == BUILTIN_TOKENIZER_NAME
        or tokenizer_cfg.name in TIKTOKEN_ENCODING_NAMES
    ):
        logger.debug("Using tiktoken tokenizer, skipping HF alias resolution")
        return {model: tokenizer_cfg.name for model in model_names}

    # Fake-model-name fallback: when --tokenizer is unset, names that look
    # like LLM-hallucinated placeholders default to builtin instead of an HF
    # Hub lookup. Explicit --tokenizer always wins.
    fake_to_builtin: dict[str, str] = {}
    if not (tokenizer_cfg and tokenizer_cfg.name):
        fake_to_builtin, real_models = _partition_fake_models(model_names, logger)
        if not real_models:
            # All models are placeholders. Mutate config.tokenizer so every
            # downstream consumer (forkserver preload env, child processes
            # that read cfg.tokenizer.name directly, the dataset_manager's
            # tokenizer loader) sees `builtin` without depending on
            # run.resolved.tokenizer_names propagation.
            from aiperf.config.v1 import TokenizerConfig

            if tokenizer_cfg is None:
                config.tokenizer = TokenizerConfig(name=BUILTIN_TOKENIZER_NAME)
            else:
                tokenizer_cfg.name = BUILTIN_TOKENIZER_NAME
            return fake_to_builtin
        names = real_models

    console = Console()
    resolved = _resolve_aliases(names, logger, console)

    # Skip if already in offline mode -- the cache is assumed warm.
    if os.environ.get("HF_HUB_OFFLINE") and os.environ.get("TRANSFORMERS_OFFLINE"):
        logger.info("HF offline mode already set, skipping cache warming")
    else:
        _prefetch_tokenizers(
            set(resolved.values()),
            trust_remote_code=tokenizer_cfg.trust_remote_code
            if tokenizer_cfg
            else False,
            revision=tokenizer_cfg.revision if tokenizer_cfg else "main",
            logger=logger,
            console=console,
        )

    if tokenizer_cfg and tokenizer_cfg.name:
        return {model: resolved[tokenizer_cfg.name] for model in model_names}
    return {**fake_to_builtin, **resolved}


def _partition_fake_models(
    model_names: list[str], logger: AIPerfLogger
) -> tuple[dict[str, str], list[str]]:
    """Split ``model_names`` into (fake → builtin map, real names list).

    Emits one ``WARNING`` log line per detected placeholder. Called only
    when ``--tokenizer`` was not explicitly set.
    """
    from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME
    from aiperf.common.tokenizer_fake_names import is_fake_model_name

    fake_to_builtin: dict[str, str] = {}
    real_models: list[str] = []
    for model in model_names:
        if is_fake_model_name(model):
            logger.warning(
                f"Model name '{model}' looks like a placeholder; defaulting "
                f"tokenizer to '{BUILTIN_TOKENIZER_NAME}' (tiktoken o200k_base). "
                f"Pass --tokenizer <name> to override."
            )
            fake_to_builtin[model] = BUILTIN_TOKENIZER_NAME
        else:
            real_models.append(model)
    return fake_to_builtin, real_models
