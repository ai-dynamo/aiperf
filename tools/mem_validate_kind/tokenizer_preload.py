"""Import-side-effect module that loads HF tokenizers into the forkserver heap.

Listed in ``multiprocessing.set_forkserver_preload`` so the forkserver
helper process imports it at startup. Any tokenizer instantiated here
lives in the forkserver's anonymous memory for the lifetime of the
helper; every worker/RP child forked from it CoW-shares those pages
until it writes to them (tokenizer internals are mostly read-only after
load, so sharing holds).

Configuration is via environment variable so the operator controls which
models get preloaded per pod without code changes:

    AIPERF_PRELOAD_TOKENIZERS=Qwen/Qwen3-0.6B,openai/gpt-oss-120b

Fail-soft by design: if a model fails to load the forkserver must not
crash — workers fall back to loading on demand. This is also safe to
import from non-forkserver contexts: with no env var set it is a no-op.
"""

from __future__ import annotations

import os
import sys


_LOADED: dict[str, object] = {}


def _env_models() -> list[str]:
    raw = os.environ.get("AIPERF_PRELOAD_TOKENIZERS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _preload() -> None:
    models = _env_models()
    if not models:
        return
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(
            f"[tokenizer-preload] transformers not available, skipping: {e!r}",
            file=sys.stderr,
        )
        return

    for m in models:
        try:
            print(f"[tokenizer-preload] loading {m} into forkserver heap", file=sys.stderr)
            tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            # Run an encode so any lazy initialization (byte-fallback tables,
            # merge tries, special-tokens maps) is realized here in the
            # forkserver, not per-child.
            tok.encode("warmup " * 32)
            _LOADED[m] = tok
        except Exception as e:  # noqa: BLE001 - preload must not crash forkserver
            print(
                f"[tokenizer-preload] failed to load {m}: {e!r}; "
                "workers will load on demand",
                file=sys.stderr,
            )


def get_preloaded(model_id: str) -> object | None:
    """Return a preloaded tokenizer if one is available for model_id."""
    return _LOADED.get(model_id)


def preloaded_models() -> list[str]:
    """Enumerate the model IDs that succeeded at preload time."""
    return list(_LOADED)


_preload()
