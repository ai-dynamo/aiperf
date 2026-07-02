# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forkserver preload module: load the trace-loader tokenizer once in the helper.

Listed in :data:`aiperf.dataset._mp_context._LOADER_PRELOAD` so Python's
forkserver helper imports it at startup. Any tokenizer instantiated here
lives in the helper's anonymous heap; every worker child forked from it
CoW-shares those pages instead of re-loading the tokenizer from disk.

For a 700 MiB Qwen tokenizer with 16 workers, this takes the per-spawn
cost from ~700 ms × 16 (sequential disk reads under file-lock contention)
down to a single shared resident copy.

Configuration is via environment variables, populated by
:func:`aiperf.dataset._mp_context.get_loader_mp_context` **before** it
calls :func:`multiprocessing.forkserver.ensure_running`. The env is
inherited into the helper when Python spawns it, and into every worker
forked from the helper:

    AIPERF_LOADER_PRELOAD_TOKENIZER          tokenizer name to preload
    AIPERF_LOADER_PRELOAD_TRUST_REMOTE_CODE  "true" or "false" (default false)
    AIPERF_LOADER_PRELOAD_REVISION           HF revision (default "main")

Fail-soft: any failure is logged to stderr and silently skipped. The
worker's :func:`Tokenizer.from_pretrained` fallback covers misses, so a
preload failure never blocks the run — it just means workers re-load
from disk individually.

Fork-safety: we deliberately **do not** call ``tokenizer.encode`` or
``tokenizer.decode`` here. HF fast tokenizers spawn rayon threads at
first parallel encode; a forkserver that has triggered parallel state
would propagate stale thread references into every forked child. Loading
the tokenizer object alone does not trigger parallel execution. We also
set ``TOKENIZERS_PARALLELISM=false`` so HF does not emit its post-fork
"disabling parallelism to avoid deadlocks" warning in every worker.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_LOADED: dict[tuple[str, bool, str], Any] = {}

_ENV_NAME = "AIPERF_LOADER_PRELOAD_TOKENIZER"
_ENV_TRUST = "AIPERF_LOADER_PRELOAD_TRUST_REMOTE_CODE"
_ENV_REVISION = "AIPERF_LOADER_PRELOAD_REVISION"


def _env_name() -> str:
    return os.environ.get(_ENV_NAME, "").strip()


def _env_trust_remote_code() -> bool:
    return os.environ.get(_ENV_TRUST, "false").strip().lower() in ("1", "true", "yes")


def _env_revision() -> str:
    return os.environ.get(_ENV_REVISION, "main").strip() or "main"


def _preload() -> None:
    name = _env_name()
    if not name:
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from aiperf.common.tokenizer import Tokenizer
    except ImportError as e:
        print(
            f"[aiperf.loader_tokenizer_preload] tokenizer module unavailable; "
            f"skipping preload: {e!r}",
            file=sys.stderr,
            flush=True,
        )
        return

    trust = _env_trust_remote_code()
    revision = _env_revision()

    try:
        tok = Tokenizer.from_pretrained(
            name,
            trust_remote_code=trust,
            revision=revision,
            resolve_alias=False,
        )
        _LOADED[(name, trust, revision)] = tok
        print(
            f"[aiperf.loader_tokenizer_preload] preloaded '{name}' "
            f"(trust_remote_code={trust}, revision={revision}) into forkserver heap",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - preload must never crash the forkserver helper
        print(
            f"[aiperf.loader_tokenizer_preload] failed to preload '{name}': {e!r}; "
            "workers will load on demand",
            file=sys.stderr,
            flush=True,
        )


def get_preloaded(
    name: str,
    *,
    trust_remote_code: bool = False,
    revision: str = "main",
) -> Any | None:
    """Return the preloaded tokenizer for ``(name, trust_remote_code, revision)``.

    Returns ``None`` when nothing was preloaded with this exact triple, so the
    caller can fall through to :meth:`Tokenizer.from_pretrained`.
    """
    return _LOADED.get((name, trust_remote_code, revision))


def clear_preloaded() -> None:
    """Drop every preloaded tokenizer.

    Primarily for test isolation: a test that triggers ``_preload`` (or
    otherwise populates ``_LOADED``) would leak its tokenizer into every later
    ``get_preloaded`` caller in the same process -- including parallel_convert
    workers, whose ``_init_worker`` prefers a preloaded tokenizer over
    ``Tokenizer.from_pretrained``.
    """
    _LOADED.clear()


_preload()
