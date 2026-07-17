# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-derived parse knobs threaded through the graph adapter protocol.

:class:`GraphParseContext` is the single carrier for every run-config-derived
knob a graph adapter needs to parse byte-identically to the run. The parser
passes it opaquely (``adapter_cls.parse(path, ctx)``); each adapter maps the
fields it consumes onto its own entry function and ignores the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Sentinel for ctx fields where ``None`` is a meaningful user value. Compare
# with ``is UNSET``, never ``==``. msgspec's singleton survives pickle and
# deepcopy with identity intact, so round-trips cannot silently break checks.
from msgspec import UNSET, UnsetType


@dataclass(frozen=True, slots=True)
class GraphParseContext:
    """Run-derived knobs an adapter needs to parse byte-identically to the run.

    Field semantics:
    * ``UNSET`` (where the type admits it) / ``None`` (elsewhere) means
      "adapter default" — a ctx-less parse behaves exactly like today's
      protocol-default entry.
    * A SET value — including an explicit ``None`` on tri-state fields — is
      forwarded to the adapter entry VERBATIM. ``idle_gap_cap_seconds=None``
      means DISABLE WARPING (the user's ``synthesis.idle_gap_cap_seconds:
      null``), which is distinct from unset.
    Adapters read the fields they consume and ignore the rest. Adapters must
    only forward a field when it is set (not UNSET/None per the rules above),
    so entry defaults that are not None (e.g. dynamo ``prompt_corpus="coding"``,
    dag ``run_streaming=True``) are never clobbered by a partial ctx.
    """

    content_root_seed: int | None = None
    """Run ``--random-seed`` for seed-dependent content synthesis (weka, dynamo)."""

    content_tokenizer: str | None = None
    """Tokenizer name content synthesis decodes with (weka, dynamo)."""

    tokenizer_trust_remote_code: bool | None = None
    """Run tokenizer trust flag; content-synthesizing adapters publish it to the
    loader-preload env before building callbacks (weka, dynamo) so a registry
    parse in a fresh process loads the same tokenizer the run does."""

    tokenizer_revision: str | None = None
    """Run tokenizer revision pin; published alongside trust (weka, dynamo)."""

    prompt_corpus: str | None = None
    """Synthesis corpus selector (weka, dynamo)."""

    max_osl: int | None = None
    """``--synthesis-max-osl`` cap on top-level chains (weka)."""

    num_dataset_entries: int | None = None
    """Explicit ``entries`` cap on the run's default dataset
    (``--num-dataset-entries``): the graph-plane ceiling on distinct traces
    selected. ``None`` = unset (all eligible traces after filters)."""

    max_context_length: int | None = None
    """``--max-context-length`` per-trace context ceiling (input+output tokens)
    for graph-plane dataset selection. ``None`` = no context-length filter."""

    idle_gap_cap_seconds: float | None | UnsetType = UNSET
    """Idle-gap warp cap (weka, dynamo). TRI-STATE: UNSET = adapter default
    (60s); a float = that cap; ``None`` = warping DISABLED (user's explicit
    null)."""

    trajectory_start_max_ratio: float = 0.0
    """Resolved t* snapshot-window upper bound
    (``--trajectory-start-max-ratio``, scenario-auto-applied when unset).
    Consumed by the parse-time dynamic-slot gate; ``0.0`` = window OFF."""

    default_model: str | None = None
    """Worker dispatch fallback model stamped into node overrides (dag_jsonl)."""

    run_streaming: bool | None = None
    """Resolved endpoint ``stream`` flag stamped onto nodes (dag_jsonl)."""

    delay_cap_seconds: float | None = None
    """Legacy ``inter_turn_delay_cap_seconds`` clamp on authored delays (dag_jsonl)."""

    endpoint_extra: list[tuple[str, Any]] | None = None
    """Run ``--extra-inputs`` pairs folded into node overrides (dag_jsonl)."""


def publish_ctx_tokenizer_env(ctx: GraphParseContext | None) -> None:
    """Publish ``ctx``'s tokenizer trust/revision to the loader-preload env.

    No-op unless ``ctx.tokenizer_trust_remote_code`` is set (not None), so a
    ctx-less or partial-ctx parse never clobbers values the run path already
    published. ``tokenizer_revision`` passes verbatim —
    :func:`~aiperf.dataset._mp_context.configure_loader_tokenizer_env` maps
    ``None`` to ``"main"`` itself. Idempotent with the run path's publish
    (same run-derived values). Lazy import: this module stays a pure-data
    leaf at import time.
    """
    if ctx is None or ctx.tokenizer_trust_remote_code is None:
        return
    from aiperf.dataset._mp_context import configure_loader_tokenizer_env

    configure_loader_tokenizer_env(
        trust_remote_code=ctx.tokenizer_trust_remote_code,
        revision=ctx.tokenizer_revision,
    )
