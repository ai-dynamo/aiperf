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


@dataclass(frozen=True, slots=True)
class GraphParseContext:
    """Run-derived knobs an adapter needs to parse byte-identically to the run.

    Field semantics: ``None`` means "adapter default", so a ctx-less parse
    behaves exactly like the protocol-default entry. Adapters read the fields
    they consume, ignore the rest, and forward a field ONLY when it is not
    ``None``, so entry defaults that are not None (e.g. dynamo
    ``prompt_corpus="coding"``, dag ``run_streaming=True``) are never clobbered
    by a partial ctx.
    """

    content_root_seed: int | None = None
    """Run ``--random-seed`` for seed-dependent content synthesis (dynamo)."""

    content_tokenizer: str | None = None
    """Tokenizer name content synthesis decodes with (dynamo)."""

    tokenizer_trust_remote_code: bool | None = None
    """Run tokenizer trust flag; content-synthesizing adapters publish it to the
    loader-preload env before building callbacks (dynamo) so a registry
    parse in a fresh process loads the same tokenizer the run does."""

    tokenizer_revision: str | None = None
    """Run tokenizer revision pin; published alongside trust (dynamo)."""

    prompt_corpus: str | None = None
    """Synthesis corpus selector (dynamo)."""

    max_osl: int | None = None
    """``--synthesis-max-osl`` cap requested for the graph adapter."""

    max_isl: int | None = None
    """``--synthesis-max-isl`` cap requested for the graph adapter."""

    num_dataset_entries: int | None = None
    """Explicit ``entries`` cap on the run's default dataset
    (``--num-dataset-entries``): the graph-plane ceiling on distinct traces
    selected. ``None`` = unset (all eligible traces after filters)."""

    max_context_length: int | None = None
    """``--max-context-length`` per-trace context ceiling (input+output tokens)
    for graph-plane dataset selection. ``None`` = no context-length filter."""

    idle_gap_cap_seconds: float | None = None
    """Idle-gap warp cap (dynamo). A float caps idle gaps at that value;
    ``None`` means no per-trace compression."""

    trajectory_start_max_ratio: float = 0.0
    """Resolved t* snapshot-window upper bound
    (``--trajectory-start-max-ratio``, scenario-auto-applied when unset).
    Consumed by the parse-time dynamic-slot gate; ``0.0`` = window OFF."""

    default_model: str | None = None
    """Worker dispatch fallback model stamped into node overrides (dag_jsonl)."""

    run_streaming: bool | None = None
    """Resolved endpoint ``stream`` flag stamped onto nodes (dag_jsonl)."""

    delay_cap_seconds: float | None = None
    """``--inter-turn-delay-cap-seconds`` cap requested for graph replay delays."""

    ignore_trace_delays: bool = False
    """Whether ``--ignore-trace-delays`` was enabled for the trace dataset."""

    use_think_time_only: bool = False
    """Whether ``--use-think-time-only`` was enabled for the trace dataset."""

    endpoint_extra: list[tuple[str, Any]] | None = None
    """Run ``--extra-inputs`` pairs folded into node overrides (dag_jsonl)."""

    open_loop_replay: bool | None = None
    """The run's RESOLVED open-loop replay setting (``--open-loop-replay``).

    ``None`` = no trace-replay dataset on the run, so nothing told the adapter
    how arrival is paced; ``True``/``False`` are the run's effective value.
    Carried as a value rather than through :attr:`replay_only_knobs` because
    that tuple is keyed on value-differs-from-default and open-loop replay
    DEFAULTS TO TRUE -- the default-on case, which is the common one, would
    never be named there, and a guard keyed on it could not fire at all.

    Adapters whose semantics break under timeline pacing (Agent Trace Replay with real
    tool execution: a host faster than the capture host gets held back to the
    recorded schedule) read this and refuse."""

    execute_tools: bool | None = None
    """Whether the run asked for REAL tool execution (``--graph-execute-tools``).

    ``None`` = unset, so a ctx-less parse keeps the adapter's own default (off).
    This is the ONLY channel the flag can travel: the sole production parse
    caller (``GraphStoreBuilder`` -> :func:`parse_graph_workload`) passes no
    adapter kwargs, so an adapter-keyword-only setting is unreachable from
    ``aiperf profile``."""

    use_family_sampling: bool = True
    """Whether to inject Agent Trace Replay's per-family wire sampling into replayed nodes.

    ``True`` (default): apply the family-specific sampling table from
    ``AGENT_TRACE_FAMILY_SAMPLING`` in the mini_swe_agent_trace adapter (e.g. ``temperature=0.0,
    parallel_tool_calls=True`` for swebench).  ``False``: send no family-default
    sampling, leaving nodes with only what the recording itself carries
    (or ``--use-recorded-sampling``).  Controlled via
    ``--[no-]graph-use-family-sampling``."""

    emit_warmup: bool = False
    """Whether to prepend a warmup LlmNode to each Agent Trace Replay recording's graph.

    When ``True``, each recording graph starts with a single-message
    ``"Reply with exactly: ok"`` call (max 8 tokens, same tools as the first
    recorded call, same family sampling) before the first real model turn.
    This mirrors the per-recording warmup Agent Trace Replay's own playback emits.
    Controlled via ``--graph-emit-warmup``."""

    replay_only_knobs: tuple[str, ...] = ()
    """CLI flags the run set that only the linear trace replay loaders consume.

    Resolved as the flags whose value DIFFERS FROM THEIR DEFAULT, so a run that
    never named them carries an empty tuple. Value-vs-default rather than
    ``model_fields_set`` because the latter does not survive
    ``model_dump`` -> ``model_validate``, which the sweep orchestrator crosses
    per cell -- a knob the operator really set would look unset in a swept run
    (the same trap :func:`~aiperf.config.phases.resolve_graph_tstar_window`
    documents).

    An adapter that cannot honor them refuses; carrying the NAMES rather than
    the values keeps the refusal message able to quote the flag the operator
    typed without this bundle growing a field per knob.
    """


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
