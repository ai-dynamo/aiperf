# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forkserver preload module for sharing HF tokenizers across RecordProcessors.

Listed in :data:`aiperf.common.mp_context._FORKSERVER_PRELOAD` so Python's
forkserver helper imports it at startup. Any tokenizer instantiated in this
module lives in the forkserver helper's anonymous memory for the lifetime
of the helper; every RecordProcessor child forked from it CoW-shares those
pages. In a typical benchmark with two 150 MiB tokenizers and four RPs,
this saves ~900 MiB of per-pod RAM over on-demand loading.

Only effective in ``ServiceRunType.MULTIPROCESSING`` mode — in Kubernetes
every RecordProcessor runs as its own sibling container with its own
address space, so there is no forkserver to share from. This module is
still imported in K8s (harmless because it short-circuits when the env
var is unset), so the operator does not need to conditionally configure
the forkserver preload list per deployment mode.

Configuration is exclusively via environment variables, populated by the
topmost aiperf process (``cli_runner._run_single_benchmark`` /
``cli_commands/service.py``) **before** the first call to
:func:`aiperf.common.mp_context.get_mp_context`. The env is inherited
into the forkserver helper when Python spawns it, and into every RP
child forked from the helper:

    AIPERF_PRELOAD_TOKENIZERS              comma-separated model IDs to preload
    AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE  "true" or "false" (default false)
    AIPERF_PRELOAD_TOKENIZER_REVISION      git revision string (default "main")

Fail-soft semantics: any model that fails to preload is logged to stderr
and silently skipped. The RP's existing on-demand
:meth:`aiperf.common.tokenizer.Tokenizer.from_pretrained` fallback covers
misses, so a preload failure never blocks a benchmark — it just means
that model doesn't get the CoW saving for this run.

Fork-safety: we deliberately **do not** call ``tokenizer.encode`` here.
HF fast tokenizers spawn rayon threads at first parallel encode; a
forkserver that has already triggered parallel state would propagate
stale thread references into every forked child. Loading the tokenizer
object alone does not trigger parallel execution. We also set
``TOKENIZERS_PARALLELISM=false`` explicitly so HF does not emit its
post-fork "disabling parallelism to avoid deadlocks" warning in every
child. See ``tools/mem_validate_kind`` for empirical verification.

Caveats:

* The preloaded tokenizers persist for the forkserver helper's lifetime.
  A long-running driver that switches models between runs keeps the
  previous run's tokenizers resident until the forkserver is torn down.
* Preload applies one ``trust_remote_code`` / ``revision`` to every model
  — the common case in AIPerf where a BenchmarkRun shares tokenizer
  config across all models. Call sites with bespoke overrides skip the
  cache automatically because :func:`get_preloaded` lookup is exact.
"""

from __future__ import annotations

import os
import sys

_LOADED: dict[str, object] = {}
_ENV_MODELS = "AIPERF_PRELOAD_TOKENIZERS"
_ENV_TRUST = "AIPERF_PRELOAD_TOKENIZER_TRUST_REMOTE_CODE"
_ENV_REVISION = "AIPERF_PRELOAD_TOKENIZER_REVISION"


def _env_models() -> list[str]:
    raw = os.environ.get(_ENV_MODELS, "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _env_trust_remote_code() -> bool:
    return os.environ.get(_ENV_TRUST, "false").strip().lower() in ("1", "true", "yes")


def _env_revision() -> str:
    return os.environ.get(_ENV_REVISION, "main").strip() or "main"


def _suppress_post_fork_warning() -> None:
    """Set TOKENIZERS_PARALLELISM=false unless the operator already set it.

    HF fast tokenizers warn once per forked child when they detect that
    parallel state existed in the parent. The warning is benign (HF
    auto-disables parallelism in the child) but noisy in RP logs. Setting
    it to false explicitly makes behavior deterministic. Benchmark shows
    <=10% variation on single-encode workloads (the only pattern RP uses),
    and the setting only affects batch-encode parallelism which RP never
    uses. See tools/mem_validate_kind/README for the measured data.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _preload() -> None:
    models = _env_models()
    if not models:
        return

    _suppress_post_fork_warning()

    try:
        from aiperf.common.tokenizer import Tokenizer
    except ImportError as e:
        print(
            f"[aiperf.tokenizer_preload] aiperf.common.tokenizer unavailable; "
            f"skipping preload: {e!r}",
            file=sys.stderr,
            flush=True,
        )
        return

    trust_remote_code = _env_trust_remote_code()
    revision = _env_revision()

    for name in models:
        try:
            tok = Tokenizer.from_pretrained(
                name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            _LOADED[name] = tok
            print(
                f"[aiperf.tokenizer_preload] preloaded '{name}' into forkserver heap",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - preload must never crash the forkserver
            print(
                f"[aiperf.tokenizer_preload] failed to preload '{name}': {e!r}; "
                "RP will load on demand",
                file=sys.stderr,
                flush=True,
            )


def get_preloaded(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    revision: str = "main",
) -> object | None:
    """Return a preloaded tokenizer whose config matches the caller's needs.

    Returns ``None`` if no tokenizer was preloaded for ``model_id``, or if
    the caller's ``trust_remote_code`` / ``revision`` differ from the
    values that were used at preload time. Callers that get ``None`` fall
    through to ``Tokenizer.from_pretrained`` as before.

    The config match guard is important: the preload module loads every
    tokenizer with one ``trust_remote_code`` / ``revision`` pair read from
    env vars. A caller that wants a different combination (e.g. pinned to
    a specific commit) should skip the cache and load fresh.
    """
    tok = _LOADED.get(model_id)
    if tok is None:
        return None
    if trust_remote_code != _env_trust_remote_code():
        return None
    if revision != _env_revision():
        return None
    return tok


def preloaded_models() -> list[str]:
    """Return the model IDs that succeeded at preload time."""
    return list(_LOADED)


def get_or_load(
    name: str,
    *,
    trust_remote_code: bool = False,
    revision: str = "main",
    resolve_alias: bool = True,
) -> object:
    """Return a preloaded tokenizer if available, else load on demand.

    Drop-in replacement for callers that today call
    :meth:`aiperf.common.tokenizer.Tokenizer.from_pretrained` directly.
    When the forkserver-preloaded cache has a matching entry the call
    returns immediately with the shared instance; otherwise it falls back
    to the normal loader. The return type is
    :class:`aiperf.common.tokenizer.Tokenizer` in both paths.

    The match check in :func:`get_preloaded` enforces identical
    ``trust_remote_code`` and ``revision`` so callers with bespoke
    overrides always get a fresh load.
    """
    tok = get_preloaded(name, trust_remote_code=trust_remote_code, revision=revision)
    if tok is not None:
        return tok
    from aiperf.common.tokenizer import Tokenizer

    return Tokenizer.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        revision=revision,
        resolve_alias=resolve_alias,
    )


_preload()
