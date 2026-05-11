# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early tokenizer validation and preloading before spawning services."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing.queues

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

    # Fake-model-name fallback: when --tokenizer is unset, names that look
    # like LLM-hallucinated placeholders default to builtin instead of an HF
    # Hub lookup. Explicit --tokenizer always wins.
    fake_to_builtin: dict[str, str] = {}
    if not tokenizer_cfg.name:
        fake_to_builtin, real_models = _partition_fake_models(model_names, logger)
        if not real_models:
            # All models are placeholders. Mutate tokenizer_cfg.name so every
            # downstream consumer (child processes that read
            # cfg.tokenizer.name directly, the preload step, the dataset
            # manager's tokenizer loader) sees `builtin` without depending
            # on resolved_names propagation.
            tokenizer_cfg.name = BUILTIN_TOKENIZER_NAME
            return fake_to_builtin
        names = real_models

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
        cached = _is_hf_cached(name, revision)
        if logger:
            logger.debug(f"_is_hf_cached('{name}', revision={revision!r}) -> {cached}")
        if cached:
            if logger:
                logger.debug(
                    f"Tokenizer preload skipped for '{name}': already in HF cache"
                )
            continue
        names_to_load.append(name)

    failed: set[str] = set()
    if names_to_load:
        if logger:
            logger.info(
                f"Preloading {len(names_to_load)} tokenizer(s) into local cache..."
            )
        for name in names_to_load:
            if logger:
                logger.info(f"  Caching tokenizer: {name}")
            start = time.perf_counter()
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
            except Exception as e:  # noqa: BLE001
                failed.add(name)
                if logger:
                    logger.debug(
                        f"Tokenizer preload failed for '{name}' after "
                        f"{time.perf_counter() - start:.2f}s: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
            else:
                if logger:
                    logger.debug(
                        f"Tokenizer preload succeeded for '{name}' in "
                        f"{time.perf_counter() - start:.2f}s"
                    )
    elif logger:
        logger.debug(
            "Tokenizer preload: all tokenizers already cached, no download needed"
        )

    if failed and logger:
        names_str = ", ".join(f"'{n}'" for n in failed)
        logger.warning(
            f"Failed to preload {len(failed)} tokenizer(s): {names_str}. "
            "Child processes will attempt to load them themselves."
        )


def preload_tokenizers_in_subprocess(
    resolved_names: dict[str, str] | None,
    trust_remote_code: bool = False,
    revision: str = "main",
    logger: AIPerfLogger | None = None,
) -> None:
    """Run preload_tokenizers in a spawned subprocess.

    Isolates `transformers` / `huggingface_hub` imports and any side-effecting
    global state from the parent interpreter. The HF cache lives on disk, so
    cached files written by the subprocess are visible to children spawned by
    the parent for the rest of the run.

    Failures are non-fatal: a warning is logged and the function returns
    normally so the benchmark can proceed and children retry online.
    """
    if logger:
        logger.debug(
            f"preload_tokenizers_in_subprocess entry: "
            f"resolved_names={resolved_names}, trust_remote_code={trust_remote_code}, "
            f"revision={revision!r}"
        )

    if not resolved_names:
        if logger:
            logger.debug("Tokenizer preload skipped: validation was not run")
        return

    import logging
    import multiprocessing

    # Mirror parent's effective log level into the subprocess so debug logs
    # surface there too when the parent is run with --verbose.
    log_level = logger.get_effective_level() if logger else logging.INFO

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.queues.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_preload_subprocess_main,
        kwargs={
            "resolved_names": resolved_names,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "log_level": log_level,
            "result_queue": result_queue,
        },
        name="aiperf-tokenizer-preload",
    )

    start = time.perf_counter()
    proc.start()
    if logger:
        logger.debug(
            f"Tokenizer preload subprocess started: pid={proc.pid}, "
            f"name={proc.name!r}, start_method=spawn"
        )
    proc.join()
    elapsed = time.perf_counter() - start
    if logger:
        logger.debug(
            f"Tokenizer preload subprocess finished: pid={proc.pid}, "
            f"exitcode={proc.exitcode}, elapsed={elapsed:.2f}s"
        )

    status, error = _drain_preload_result(result_queue, proc.exitcode, logger)
    if status != "ok" and logger:
        logger.warning(
            f"Tokenizer preload subprocess failed: {error}. "
            "Child processes will attempt to load tokenizers themselves."
        )


def _drain_preload_result(
    result_queue,
    exitcode: int | None,
    logger: AIPerfLogger | None,
) -> tuple[str, str | None]:
    """Read (status, error) from the subprocess's result queue.

    Returns a synthetic error tuple if the queue is empty (e.g. the subprocess
    crashed before reporting). Debug-logs the outcome either way.
    """
    try:
        status, error = result_queue.get_nowait()
    except Exception as e:  # noqa: BLE001 - queue.Empty or unpickling failure
        if logger:
            logger.debug(
                f"Tokenizer preload result queue empty/unreadable: "
                f"{type(e).__name__}: {e}"
            )
        status, error = (
            "error",
            f"subprocess exited with code {exitcode} without reporting a result",
        )
    if logger:
        logger.debug(f"Tokenizer preload subprocess result: status={status!r}")
    return status, error


def _preload_subprocess_main(
    *,
    resolved_names: dict[str, str],
    trust_remote_code: bool,
    revision: str,
    log_level: int,
    result_queue,
) -> None:
    """Spawn-safe subprocess entry point for tokenizer preload.

    Imports HF libraries fresh inside this process so the parent stays clean.
    Progress logs go to stderr (inherited from the parent); success/failure is
    reported via `result_queue`.
    """
    import logging
    import os
    import platform
    import traceback

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    from aiperf.common.aiperf_logger import AIPerfLogger

    logger = AIPerfLogger(__name__)

    logger.debug(
        f"_preload_subprocess_main started: pid={os.getpid()}, "
        f"ppid={os.getppid()}, python={platform.python_version()}, "
        f"platform={platform.system()}"
    )
    logger.debug(
        f"HF env (subprocess): HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')!r}, "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')!r}, "
        f"HF_HOME={os.environ.get('HF_HOME')!r}, "
        f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')!r}"
    )
    logger.debug(
        f"_preload_subprocess_main args: {len(resolved_names)} resolved name(s), "
        f"trust_remote_code={trust_remote_code}, revision={revision!r}"
    )

    start = time.perf_counter()
    try:
        asyncio.run(
            preload_tokenizers(
                resolved_names,
                trust_remote_code=trust_remote_code,
                revision=revision,
                logger=logger,
            )
        )
        logger.debug(
            f"_preload_subprocess_main completed successfully in "
            f"{time.perf_counter() - start:.2f}s"
        )
        result_queue.put(("ok", None))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"_preload_subprocess_main raised after "
            f"{time.perf_counter() - start:.2f}s: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        result_queue.put(("error", repr(exc)))
