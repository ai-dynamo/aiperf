# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early tokenizer validation and preloading before spawning services."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


def _prefetch_one(
    name: str,
    trust_remote_code: bool,
    revision: str,
) -> tuple[str, str | None]:
    """Subprocess target: warm one tokenizer's disk cache.

    All heavy imports (``transformers``, ``tokenizers`` Rust ext,
    ``tiktoken``) happen here, inside the subprocess — the CLI parent
    never imports them. Returns ``(name, error_message_or_None)`` so the
    parent can log per-name and continue.
    """
    try:
        from aiperf.common.tokenizer import Tokenizer

        Tokenizer.from_pretrained(
            name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            resolve_alias=False,
        )
        return (name, None)
    except Exception as e:  # noqa: BLE001 - HF/tiktoken/network surface arbitrary errors; serialize and let the parent log per-name
        return (name, f"{type(e).__name__}: {e}")


def _partition_preload_names(
    resolved_names: dict[str, str],
    revision: str,
    logger: AIPerfLogger | None,
) -> tuple[set[str], set[str], set[str]]:
    """Split resolved tokenizer names into:

      (tiktoken_to_warm, hf_to_warm, hf_already_cached)

    Local paths and tiktoken encodings already in the disk cache are
    dropped entirely. ``hf_already_cached`` is reported separately so
    the caller can write an offline-config stub for tokenizer-only HF
    repos that are already present but lack a ``config.json``.
    """
    from pathlib import Path

    from aiperf.common.tokenizer import (
        BUILTIN_TOKENIZER_NAME,
        TIKTOKEN_ENCODING_NAMES,
        _is_hf_cached,
        _is_tiktoken_cached,
    )

    tiktoken_names: set[str] = set()
    hf_names: set[str] = set()
    hf_already_cached: set[str] = set()
    for name in set(resolved_names.values()):
        if name == BUILTIN_TOKENIZER_NAME or name in TIKTOKEN_ENCODING_NAMES:
            if _is_tiktoken_cached(name):
                if logger:
                    logger.debug(
                        f"Tokenizer preload skipped for '{name}': already in tiktoken cache"
                    )
                continue
            tiktoken_names.add(name)
            continue
        p = Path(name)
        if p.is_absolute() or name.startswith(("./", "../")) or p.is_dir():
            if logger:
                logger.debug(f"Tokenizer preload skipped for '{name}': local path")
            continue
        if _is_hf_cached(name, revision):
            if logger:
                logger.debug(
                    f"Tokenizer preload skipped for '{name}': already in HF cache"
                )
            hf_already_cached.add(name)
            continue
        hf_names.add(name)
    return tiktoken_names, hf_names, hf_already_cached


def _ensure_offline_config_stub(
    name: str, revision: str, logger: AIPerfLogger | None = None
) -> None:
    """Write a stub ``config.json`` for tokenizer-only HF repos.

    Tokenizer-only repos (e.g. ``hf-internal-testing/llama-tokenizer``)
    ship no ``config.json``. ``AutoTokenizer.from_pretrained`` with
    ``local_files_only=True`` then raises a misleading "Couldn't connect"
    ``OSError`` because its config lookup can't fall through to
    ``tokenizer_config.json`` like the online path does. Writing an empty
    ``{}`` config.json into the cached snapshot satisfies the offline
    lookup. Idempotent; no-op when the file already exists or when the
    snapshot has no tokenizer indicator file at all.

    Atomic via ``NamedTemporaryFile`` + ``Path.replace`` so a child
    reading the file mid-write can never see a partial / empty config.

    Raises:
        OSError: If the stub cannot be written. Aborting startup here is
            preferable to a downstream worker hanging with a misleading
            "Couldn't connect" error pointing at HF Hub.
    """
    import tempfile
    from pathlib import Path

    from aiperf.common.tokenizer import _get_revision_snapshot_dir

    snapshot = _get_revision_snapshot_dir(name, revision)
    if snapshot is None:
        return
    config_path = snapshot / "config.json"
    if config_path.exists():
        return
    # Need at least one tokenizer indicator file — otherwise the snapshot
    # is broken or unrelated, and fabricating a config won't help. Covers
    # both legacy (tokenizer_config.json) and fast-only (tokenizer.json)
    # repos.
    if not any(
        (snapshot / f).exists() for f in ("tokenizer_config.json", "tokenizer.json")
    ):
        return
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="config.aiperf-",
            dir=str(snapshot),
            delete=False,
        ) as tmp:
            tmp.write("{}")
            tmp_path = Path(tmp.name)
        tmp_path.replace(config_path)
    except OSError as e:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if logger:
            logger.error(
                f"Could not write stub config.json for tokenizer '{name}' "
                f"at {config_path}: {e!r}. Offline tokenizer load would fail "
                "in child services; aborting initialization."
            )
        raise
    if logger:
        logger.debug(f"Wrote stub config.json for tokenizer-only repo '{name}'")


async def _run_prefetch_pool(
    all_names: list[str],
    *,
    trust_remote_code: bool,
    revision: str,
    timeout: float,
    logger: AIPerfLogger | None,
) -> list[tuple[str, str | None]]:
    """Run ``_prefetch_one`` for each name in a spawn ProcessPoolExecutor.

    Bounds total wall-clock by ``timeout``. On timeout, kills running
    subprocesses and returns ``TimeoutError`` results for every name.
    """
    ctx = mp.get_context("spawn")
    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=len(all_names), mp_context=ctx)
    try:
        futures = [
            loop.run_in_executor(pool, _prefetch_one, name, trust_remote_code, revision)
            for name in all_names
        ]
        try:
            return await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
        except asyncio.TimeoutError:
            if logger:
                logger.warning(
                    f"Tokenizer preload exceeded "
                    f"AIPERF_TOKENIZER_PRELOAD_TIMEOUT={timeout}s; "
                    "killing prefetch subprocesses and continuing."
                )
            # ProcessPoolExecutor's cancel_futures only cancels queued work,
            # not running subprocesses. Kill them explicitly to prevent
            # zombies after pool.shutdown returns.
            for proc in list(pool._processes.values()):  # noqa: SLF001
                if proc.is_alive():
                    proc.kill()
            return [(name, "TimeoutError") for name in all_names]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


async def preload_tokenizers(
    resolved_names: dict[str, str] | None,
    trust_remote_code: bool = False,
    revision: str = "main",
    logger: AIPerfLogger | None = None,
) -> None:
    """Preload tokenizer files into disk cache before spawning child services.

    All prefetch work runs in short-lived ``ProcessPoolExecutor`` workers
    using the ``spawn`` mp context. The CLI parent never imports
    ``transformers``, the Rust-backed ``tokenizers`` extension, or
    ``tiktoken`` — important because:

    * macOS spawn re-imports everything per child; a bloated parent
      wastes memory and startup time.
    * Linux fork inherits parent state. ``tokenizers`` spins up Rayon
      thread pools at import; forking after Rayon init is the textbook
      "process forked after parallelism" trap (deadlock-prone).

    Total wall-clock time across all parallel subprocess pre-warms is
    bounded by ``Environment.TOKENIZER.PRELOAD_TIMEOUT``. On timeout,
    subprocesses are killed and AIPerf continues — child services will
    download tokenizers themselves on first use.

    The CLI parent never mutates ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
    in its own ``os.environ``: child services inherit env at spawn, so a
    parent-side mutation poisons legitimate online consumers (e.g.
    ``dataset_manager`` loading a public HF dataset). Workers that
    genuinely need offline mode set it locally in their own
    ``_init_worker`` (see ``aiperf.dataset.generator.parallel_decode``).

    Args:
        resolved_names: Mapping of model names to resolved tokenizer names.
                        If None or empty (validation was skipped), this is a no-op.
        trust_remote_code: Whether to trust remote code when loading.
        revision: The HF revision to fetch.
        logger: Optional logger for progress output.
    """
    from aiperf.common.environment import Environment

    if not resolved_names:
        if logger:
            logger.debug("Tokenizer preload skipped: validation was not run")
        return

    tiktoken_names, hf_names, hf_already_cached = _partition_preload_names(
        resolved_names, revision, logger
    )

    all_names = sorted(tiktoken_names | hf_names)
    if not all_names and not hf_already_cached:
        if logger:
            logger.debug(
                "Tokenizer preload: nothing to warm (all builtin/local/cached)"
            )
        return

    results: list[tuple[str, str | None]] = []
    if all_names:
        if logger:
            parts: list[str] = []
            if hf_names:
                parts.append(f"{len(hf_names)} HF tokenizer(s)")
            if tiktoken_names:
                parts.append(f"{len(tiktoken_names)} tiktoken encoding(s)")
            logger.info(
                f"Preloading {' + '.join(parts)} in {len(all_names)} subprocess(es)..."
            )

        timeout = Environment.TOKENIZER.PRELOAD_TIMEOUT
        start = time.perf_counter()
        results = await _run_prefetch_pool(
            all_names,
            trust_remote_code=trust_remote_code,
            revision=revision,
            timeout=timeout,
            logger=logger,
        )
        elapsed = time.perf_counter() - start

        for name, err in results:
            if err is None:
                if logger:
                    logger.debug(f"Preloaded tokenizer: {name}")
            elif logger:
                kind = "tiktoken cache" if name in tiktoken_names else "tokenizer"
                logger.warning(
                    f"Failed to pre-warm {kind} for '{name}' ({err}). "
                    "Child processes will load it themselves on first use."
                )

        if logger:
            logger.debug(f"Tokenizer preload completed in {elapsed:.2f}s")

    # Write offline-config stub for tokenizer-only HF repos. Covers both
    # repos we warmed this run and repos that were already cached from a
    # prior run. Idempotent and a no-op for repos that have a real
    # config.json or aren't tokenizer-only.
    warmed_hf = {name for name, err in results if err is None and name in hf_names}
    for name in warmed_hf | hf_already_cached:
        _ensure_offline_config_stub(name, revision, logger)
