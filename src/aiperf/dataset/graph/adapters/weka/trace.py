# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: convert Weka KV-cache-tester agentic-coding traces to ParsedGraph.

A Weka trace is a single JSON object whose ``requests`` list interleaves
normal API calls (``type: "n"``), streaming API calls (``type: "s"``), and
subagent markers (``type: "subagent"``) with their own nested request lists.

This adapter emits a **flat segment-trie IR** via
:func:`~aiperf.dataset.graph.adapters.weka.trie_build.build_trie_graph`:
one ``LlmNode`` per recorded normal/streaming request (including recursive
subagent-inner requests) connected by dependency-only ``StaticEdge``s.
Dependency edges are derived from a global interval-order pass over the
recorded request intervals (finished-before + rank).

Prompt content is real-content and content-addressed: ``build_trie_graph`` walks
the recorded ``hash_ids`` prefix tree, assigns frozen per-block ``(role,
starts_new_message)`` tags, and emits one deduplicated ``SegmentPool`` entry per
message so shared block-aligned prefixes produce identical segment ids (a real
prefix-cache hit). Each node carries its ``prompt_segment_ids`` chain in
``metadata["trie"]``; the recorded ``out`` is plumbed to ``max_tokens`` (dispatch
overrides) and the stream flag tracks the request type. There is no
dispatch-time rewrite.

This module performs the Weka JSON validation, streaming/parallel ingest, and
trace-record conversion; the flat-trie construction itself lives in
``weka/trie_build.py``, the shared trie core in ``segment_ir/``, and native
JSONL reading in ``parser.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import orjson
from pydantic import ValidationError

from aiperf.common.constants import IS_WINDOWS
from aiperf.dataset.graph.adapters.shared.idle_gap import (
    DEFAULT_IDLE_GAP_CAP_SECONDS as _DEFAULT_IDLE_GAP_CAP_SECONDS,
)
from aiperf.dataset.graph.adapters.shared.idle_gap import (
    IDLE_GAP_CAP_USE_DEFAULT as _USE_DEFAULT,
)
from aiperf.dataset.graph.adapters.shared.selection import SelectionStats
from aiperf.dataset.graph.adapters.weka import trace_parallel
from aiperf.dataset.graph.adapters.weka.trace_models import (
    EmptyWekaTraceError,
    WekaHashScopeError,
    WekaSchemaError,
    WekaTraceAdapterError,
)

if TYPE_CHECKING:
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parse_context import (
    UNSET,
    GraphParseContext,
    publish_ctx_tokenizer_env,
)

_DEFAULT_TAG = "from-weka-trace"

# Bounded sniff budget for ``WekaTraceAdapter.can_load`` — Weka traces are
# single-document JSON with the discriminator keys near the top-level object.
_DETECTION_SNIFF_BYTES = 4096

# HuggingFace repo-id pattern: ``org/name`` with no path separators beyond the
# single slash and no file extension. The weka HF corpora live at e.g.
# ``semianalysisai/cc-traces-weka-061526``. A ``--graph`` arg that is NOT an
# existing path AND that matches this shape AND carries the weka corpus marker
# is treated as a weka HF dataset id. The marker requirement is what keeps an
# arbitrary ``org/name`` (``meta-llama/Llama-3``, ``openai/gpt``) from being
# silently routed into ``datasets.load_dataset``.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# A repo id is only treated as a weka HF corpus when this case-insensitive
# marker appears in it. The published weka corpora carry ``weka`` in the repo
# name (e.g. ``semianalysisai/cc-traces-weka-061526``); any non-existent
# ``org/name`` string WITHOUT it stays on the unchanged linear pipeline.
_HF_WEKA_MARKER = "weka"

# File extensions that disqualify a string from being an HF repo id even when it
# happens to match the org/name shape (e.g. ``foo/bar.json``).
_HF_EXCLUDED_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml")

# The complete set of top-level keys a genuine on-disk weka trace object may
# carry: the required + optional fields of :class:`WekaTrace`, plus the ``kind``
# marker the native JSONL writer stamps on serialized rows. Any top-level key
# outside this set marks the object as a FOREIGN format (mooncake / sharegpt /
# synthetic) that merely happens to share the weka discriminator keys, so we
# reject it rather than ignoring the extras (mirrors WekaTrace's extra="forbid").
_WEKA_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "models",
        "block_size",
        "hash_id_scope",
        "tool_tokens",
        "system_tokens",
        "requests",
        "totals",
        "kind",
    }
)


# The hash-id namespace scopes the adapter supports. Kept in sync with
# ``WekaTrace.hash_id_scope``'s Literal; the explicit pre-validation gate in
# ``_parse_trace_dict`` uses this so an unrecognized scope raises
# WekaHashScopeError instead of a generic ValidationError.
_SUPPORTED_HASH_ID_SCOPES: frozenset[str] = frozenset({"local", "global"})


# The idle-gap entry default + ``_USE_DEFAULT`` sentinel are shared with the
# dynamo adapter (imported above from ``adapters.shared.idle_gap``): the run
# path passes the resolved cap explicitly; the default only applies when
# ``idle_gap_cap_seconds`` is left at the sentinel (direct adapter / CLI
# tooling / tests).


def _resolve_parse_kwargs(
    *,
    tag: str,
    idle_gap_cap_seconds: float | None | object,
    content_root_seed: int | None,
    content_tokenizer: str | None,
    prompt_corpus: str | None,
    max_osl: int | None,
) -> dict[str, Any]:
    """Resolve the per-trace parse kwargs shared by EVERY weka route.

    One spelling for the ``_parse_trace_dict`` keyword set: the single-file
    path, the directory pool, the HF pool, and the streaming build plane all
    pass this exact dict, so a knob added here reaches every route instead of
    needing four separately-spelled kwarg sets that could silently drift.
    Resolves the ``_USE_DEFAULT`` idle-gap sentinel and pins the effective content
    seed to a concrete int (:func:`resolve_effective_root_seed`) so serial and
    pool-worker parses synthesize identical bytes at any parallel threshold.
    """
    from aiperf.dataset.graph.adapters.shared.content import (
        resolve_effective_root_seed,
    )

    if idle_gap_cap_seconds is _USE_DEFAULT:
        idle_gap_cap_seconds = _DEFAULT_IDLE_GAP_CAP_SECONDS
    return {
        "tag": tag,
        "idle_gap_cap_seconds": idle_gap_cap_seconds,
        "content_root_seed": resolve_effective_root_seed(content_root_seed),
        "content_tokenizer": content_tokenizer,
        "prompt_corpus": prompt_corpus,
        "max_osl": max_osl,
    }


def from_weka_trace(
    path: str | Path,
    *,
    tag: str = _DEFAULT_TAG,
    idle_gap_cap_seconds: float | None | object = _USE_DEFAULT,
    workers: int | None = None,
    content_root_seed: int | None = None,
    content_tokenizer: str | None = None,
    prompt_corpus: str | None = None,
    max_osl: int | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    selection_out: list[SelectionStats] | None = None,
) -> ParsedGraph:
    """Parse a Weka KV-cache-tester trace into a :class:`ParsedGraph`.

    Args:
        path: Path to a ``.json`` file or a directory of ``.json`` files.
        tag: Base tag added to every emitted ``TraceRecord``.
        idle_gap_cap_seconds: Per-trace idle-gap cap (seconds) on the wall clock
            that stamps each node's ``arrival_offset_us`` — gaps longer than this
            are compressed to the cap (agentx ``_IdleGapTimeWarp`` parity), so the
            warped trace duration and the snapshot-at-t* resume instant match
            agentx. The run path resolves this from the dataset's
            ``synthesis.idle_gap_cap_seconds`` (``--synthesis-idle-gap-cap``);
            left unset here it defaults to 60.0. Pass ``None`` to disable warping
            (raw recorded ``t``).
        workers: Override worker count for parallel parsing of directory
            inputs. ``None`` defers to the
            ``AIPERF_DATASET_WEKA_GRAPH_PARALLEL_WORKERS`` env var (``0``/unset
            auto-scales like AgentX).
        content_root_seed: Run ``--random-seed`` pinned through to the
            real-content synthesizer so the corpus pool + per-hash reseed key
            derive from the run seed even in a ``spawn``-started parse worker
            that did not inherit the parent's seeded global RNG ``_manager``.
            ``None`` keeps the ambient global seed (bootstrap-seeded in-process
            path). See :class:`CorpusContentSynthesizer`.
        content_tokenizer: Tokenizer name the content synthesizer decodes block /
            partial-tail token IDs with. Derived from the RUN config by
            :func:`resolve_graph_content_tokenizer` (no env override) and threaded
            here so the synthesized content matches what the run dispatches and
            counts against. ``None`` (direct adapter / tooling with no run config)
            falls back to ``builtin``. Threaded like ``content_root_seed`` so
            spawn-started parse workers decode identically to the in-process path.
        prompt_corpus: Named corpus the real-content synthesizer draws block /
            partial-tail text from. Threaded through to
            :class:`CorpusContentSynthesizer` like ``content_root_seed``. ``None``
            defaults to ``"coding"``.
        max_osl: ``--synthesis-max-osl`` cap. Threaded through so the content
            synthesizer caps each top-level chain's native ``LlmNode.max_tokens``
            to ``min(recorded out, max_osl)`` (agentx
            ``_cap_output`` parity); subagent-body chains are left uncapped.
            ``None`` (default) leaves the recorded ``out`` uncapped. Threaded
            like ``content_root_seed`` so spawn-started parse workers cap
            identically to the in-process path.
        num_dataset_entries: ``--num-dataset-entries`` cap on the number of
            distinct traces built (the graph-plane fix for ai-dynamo/aiperf#1106,
            where the loader built every trace and dispatch cloned to fill lanes).
            ``None`` (default) builds every eligible trace. Applies to the
            multi-trace directory and HF corpus routes only; a single ``.json``
            file is one trace and is never capped.
        max_context_length: ``--max-context-length`` per-trace ceiling on peak
            context (input+output tokens on the largest single request, computed
            schema-only via :func:`weka_trace_peak_context`). Traces whose peak
            exceeds it are rejected BEFORE the build; the first
            ``num_dataset_entries`` eligible traces are then kept (filter THEN
            cap). ``None`` (default) applies no context filter.
        selection_out: Optional sink; when provided AND a selection knob is set,
            the single :class:`SelectionStats` for the filter-then-cap scan is
            appended for the caller's load-summary report. Untouched when both
            knobs are ``None`` (no selection performed).

    Returns:
        A :class:`ParsedGraph` with one :class:`TraceRecord` per Weka trace
        (one trace per file for directory input).

    Raises:
        WekaTraceAdapterError: any conversion failure (see the
            ``WekaTraceAdapterError`` subclasses defined in ``trace_models`` and
            re-exported here).
    """
    parse_kwargs = _resolve_parse_kwargs(
        tag=tag,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        content_root_seed=content_root_seed,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        max_osl=max_osl,
    )

    # HuggingFace-direct path: ``--graph <org/name>`` that is not an existing
    # filesystem path loads the corpus straight from the `datasets` library and
    # parses each row dict in-memory through the SAME shared core as files. No
    # temporary JSON is materialized.
    if _looks_like_hf_dataset_id(path):
        return _from_hf_dataset(
            _hf_dataset_id_str(path),
            workers=workers,
            parse_kwargs=parse_kwargs,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            max_osl=max_osl,
            selection_out=selection_out,
        )

    p = Path(path)
    if p.is_dir():
        files = _list_directory_json_files(p)
        files = _apply_weka_selection(
            files,
            source=str(p),
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            max_osl=max_osl,
            selection_out=selection_out,
        )
        return trace_parallel.parse_items(
            trace_parallel.file_work_items(files),
            source_label=str(p),
            item_count=len(files),
            workers=workers,
            parse_kwargs=parse_kwargs,
        )
    return _parse_single_file(p, **parse_kwargs)


def _list_directory_json_files(directory: Path) -> list[Path]:
    """Return the directory's ``*.json`` children sorted by name (stable)."""
    files = sorted(
        c for c in directory.iterdir() if c.is_file() and c.suffix.lower() == ".json"
    )
    if not files:
        raise EmptyWekaTraceError(
            f"weka_trace directory {str(directory)!r} contains no .json files"
        )
    return files


def _parse_single_file(
    path: Path,
    *,
    tag: str,
    idle_gap_cap_seconds: float | None = None,
    content_root_seed: int | None = None,
    content_tokenizer: str | None = None,
    prompt_corpus: str | None = None,
    max_osl: int | None = None,
) -> ParsedGraph:
    """Single-file path: validate schema, build topology, build records.

    Split out so :func:`from_weka_trace` can pass the same per-file logic into
    the parallel directory parser without a self-recursive lambda that would
    loop through the directory branch. Defers dict -> trace conversion to
    :func:`_parse_trace_dict`, the shared core used by the HF-source path too.
    """
    try:
        raw = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as e:
        raise WekaSchemaError(f"{path}: invalid JSON: {e}") from e

    return _parse_trace_dict(
        raw,
        source=str(path),
        tag=tag,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        content_root_seed=content_root_seed,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        max_osl=max_osl,
    )


def _from_hf_dataset(
    repo_id: str,
    *,
    workers: int | None = None,
    parse_kwargs: dict[str, Any],
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    max_osl: int | None = None,
    selection_out: list[SelectionStats] | None = None,
) -> ParsedGraph:
    """Load a weka corpus directly from a HuggingFace dataset, no temp files.

    Resolves row dicts via :func:`_load_hf_rows` as a streaming iterator and
    parses each through :func:`parse_items` — the SAME process pool the
    directory path uses, over the SAME shared core
    (:func:`_parse_trace_dict`). The per-row :class:`ParsedGraph` objects are
    merged with the SAME multi-graph merge the directory path uses, so
    HF-sourced and directory-sourced corpora produce identical structure.
    Above ``WEKA_GRAPH_PARALLEL_THRESHOLD`` rows the parse fans out across
    worker processes (so per-row builds are not a serial bottleneck); at or
    below it the parse stays serial in-process.

    ``split`` / ``revision`` are resolved from the
    ``Environment.DATASET.WEKA_HF_*`` knobs. HuggingFace rows are always read in
    streaming mode; bound the ingested rows with a split slice
    (``WEKA_HF_SPLIT=train[:N]``) or the loader cap (``--num-dataset-entries``).

    ``datasets`` is imported lazily here so non-HF runs never pay its import.
    """
    from aiperf.common.environment import Environment

    split = Environment.DATASET.WEKA_HF_SPLIT
    revision = Environment.DATASET.WEKA_HF_REVISION

    rows = _load_hf_rows(repo_id, split=split, revision=revision)
    if num_dataset_entries is not None or max_context_length is not None:
        kept_rows, stats = _select_weka_rows(
            rows,
            repo_id=repo_id,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            max_osl=max_osl,
        )
        if selection_out is not None:
            selection_out.append(stats)
        return trace_parallel.parse_items(
            trace_parallel.row_work_items(kept_rows, repo_id),
            source_label=repo_id,
            item_count=len(kept_rows),
            workers=workers,
            parse_kwargs=parse_kwargs,
        )
    return trace_parallel.parse_items(
        trace_parallel.row_work_items(rows, repo_id),
        source_label=repo_id,
        workers=workers,
        parse_kwargs=parse_kwargs,
    )


def _load_hf_rows(
    repo_id: str,
    *,
    split: str,
    revision: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield Weka trace row dicts from a HuggingFace dataset lazily.

    The published Weka corpora are large JSONL splits. Always using HF streaming
    mode avoids the Arrow/full-list materialization path; bound the streamed
    rows with a split slice (``split="train[:N]"``) instead. Each yielded row is
    shallow-copied to a plain ``dict`` so it is picklable for the forkserver
    parse pool and detached from the dataset row view.

    A load failure is wrapped with BOTH interpretations of the arg: ``repo_id``
    only reaches here through the weka-marker HF heuristic
    (:func:`_looks_like_hf_dataset_id`), which fires on any non-existent
    weka-marked ``org/name`` string -- including a typo'd local path -- so the
    error must not present the arg as an HF id only.
    """
    from datasets import load_dataset

    try:
        streamed = load_dataset(repo_id, split=split, streaming=True, revision=revision)
    except Exception as e:
        raise WekaTraceAdapterError(
            f"no such local file or directory {repo_id!r}, and loading it as a "
            f"HuggingFace weka dataset id failed: {e}. If you meant a local "
            "path, check the spelling; if you meant a HuggingFace dataset id, "
            "verify the repo exists and is accessible."
        ) from e
    for row in streamed:
        yield dict(row)


def stream_weka_trace_segment_payloads(
    source: str,
    *,
    tag: str = _DEFAULT_TAG,
    idle_gap_cap_seconds: float | None | object = _USE_DEFAULT,
    workers: int | None = None,
    content_root_seed: int | None = None,
    content_tokenizer: str | None = None,
    prompt_corpus: str | None = None,
    max_osl: int | None = None,
    num_dataset_entries: int | None = None,
    max_context_length: int | None = None,
    selection_out: list[SelectionStats] | None = None,
) -> Iterator[Any]:
    """Yield worker-built trace segment payloads for ANY weka source.

    The build-plane streaming counterpart of :func:`from_weka_trace`, now for
    HF corpora AND local files/directories: the SAME run-derived content knobs
    must thread into the per-item parse via ``_resolve_parse_kwargs`` so the
    streamed store's content and node ordinals stay consistent with the run's
    parse elsewhere in THIS process. For HF sources, ``split`` / ``revision``
    are resolved from the ``Environment.DATASET.WEKA_HF_*`` knobs.

    ``num_dataset_entries`` / ``max_context_length`` drive the SAME schema-only
    filter-then-cap selection as :func:`from_weka_trace` (ai-dynamo/aiperf#1106),
    applied BEFORE the payload fan-out so only kept traces are built here too --
    this is the entry the DatasetManager build plane actually drains, so the run
    honors the knobs. ``selection_out`` receives the :class:`SelectionStats` when
    a knob is set.
    """
    parse_kwargs = _resolve_parse_kwargs(
        tag=tag,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        content_root_seed=content_root_seed,
        content_tokenizer=content_tokenizer,
        prompt_corpus=prompt_corpus,
        max_osl=max_osl,
    )

    item_count: int | None = None
    if _looks_like_hf_dataset_id(source):
        from aiperf.common.environment import Environment

        split = Environment.DATASET.WEKA_HF_SPLIT
        revision = Environment.DATASET.WEKA_HF_REVISION
        rows = _load_hf_rows(_hf_dataset_id_str(source), split=split, revision=revision)
        if num_dataset_entries is not None or max_context_length is not None:
            kept_rows, stats = _select_weka_rows(
                rows,
                repo_id=source,
                num_dataset_entries=num_dataset_entries,
                max_context_length=max_context_length,
                max_osl=max_osl,
            )
            if selection_out is not None:
                selection_out.append(stats)
            item_count = len(kept_rows)
            items = trace_parallel.row_work_items(kept_rows, source)
        else:
            items = trace_parallel.row_work_items(rows, source)
    else:
        p = Path(source)
        files = _list_directory_json_files(p) if p.is_dir() else [p]
        files = _apply_weka_selection(
            files,
            source=source,
            num_dataset_entries=num_dataset_entries,
            max_context_length=max_context_length,
            max_osl=max_osl,
            selection_out=selection_out,
        )
        item_count = len(files)
        items = trace_parallel.file_work_items(files)

    yield from trace_parallel.iter_item_segment_payloads(
        items,
        source_label=source,
        item_count=item_count,
        workers=workers,
        parse_kwargs=parse_kwargs,
    )


def _validate_weka_trace(raw: dict[str, Any], *, source: str) -> WekaTrace:
    """Schema-validate one weka trace row (NO content synthesis / graph build).

    Shared by the build core (:func:`_parse_trace_dict`) and the pre-build
    selection scan (:func:`_select_weka_rows`) so both surface identical errors
    (hash-scope, schema, empty-requests) for the same row and the selector can
    screen a corpus at schema cost only.
    """
    from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace

    if not isinstance(raw, dict):
        raise WekaSchemaError(
            f"{source}: top-level value must be an object, got {type(raw).__name__}"
        )

    # Surface the explicit hash-scope error before Pydantic's Literal turns it
    # into a generic ValidationError, so users see the precise cause.
    scope = raw.get("hash_id_scope")
    if scope is not None and scope not in _SUPPORTED_HASH_ID_SCOPES:
        raise WekaHashScopeError(
            f"{source}: hash_id_scope={scope!r} is not supported (supported "
            "scopes: 'local' = per-trace hash namespace, 'global' = one hash "
            "namespace shared across all traces in the corpus)"
        )

    try:
        weka_trace = WekaTrace.model_validate(raw)
    except ValidationError as e:
        raise WekaSchemaError(f"{source}: {e}") from e

    if not weka_trace.requests:
        raise EmptyWekaTraceError(f"{source}: trace {weka_trace.id!r} has no requests")
    return weka_trace


def _apply_weka_selection(
    files: list[Path],
    *,
    source: str,
    num_dataset_entries: int | None,
    max_context_length: int | None,
    max_osl: int | None,
    selection_out: list[SelectionStats] | None,
) -> list[Path]:
    """Filter-then-cap a directory's ``.json`` files by schema-only peak context.

    No-op (returns ``files`` unchanged, nothing appended) when both knobs are
    ``None`` -- the ctx-less / knob-less build path stays byte-identical. Files
    arrive already sorted by name (:func:`_list_directory_json_files`), the
    deterministic scan order the cap is applied over.
    """
    if num_dataset_entries is None and max_context_length is None:
        return files

    from aiperf.dataset.graph.adapters.shared.peak_context import (
        weka_trace_peak_context,
    )
    from aiperf.dataset.graph.adapters.shared.selection import (
        filter_then_cap,
        log_selection_summary,
    )

    def _candidates() -> Iterator[tuple[Path, int]]:
        for f in files:
            try:
                raw = orjson.loads(f.read_bytes())
            except orjson.JSONDecodeError as e:
                raise WekaSchemaError(f"{f}: invalid JSON: {e}") from e
            trace = _validate_weka_trace(raw, source=str(f))
            yield f, weka_trace_peak_context(trace, max_osl=max_osl)

    kept, stats = filter_then_cap(
        _candidates(),
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    # Parent-side finalize point for the directory/file path (one caller runs per
    # build); mutually exclusive with the HF-row path's own summary.
    log_selection_summary(
        stats,
        source=source,
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    if selection_out is not None:
        selection_out.append(stats)
    return kept


def _select_weka_rows(
    rows: Iterator[dict[str, Any]],
    *,
    repo_id: str,
    num_dataset_entries: int | None,
    max_context_length: int | None,
    max_osl: int | None,
) -> tuple[list[dict[str, Any]], SelectionStats]:
    """Filter-then-cap streamed HF row dicts by schema-only peak context.

    Rows are consumed LAZILY in stream order (the deterministic scan order); the
    cap short-circuits the stream so a capped load pulls only the scanned prefix.
    The kept row dicts (already shallow-copied by :func:`_load_hf_rows`) are
    returned for the SAME parse fan-out the full corpus would take.
    """
    from aiperf.dataset.graph.adapters.shared.peak_context import (
        weka_trace_peak_context,
    )
    from aiperf.dataset.graph.adapters.shared.selection import (
        filter_then_cap,
        log_selection_summary,
    )

    def _candidates() -> Iterator[tuple[dict[str, Any], int]]:
        for index, row in enumerate(rows):
            trace = _validate_weka_trace(row, source=f"{repo_id}#{index}")
            yield row, weka_trace_peak_context(trace, max_osl=max_osl)

    kept, stats = filter_then_cap(
        _candidates(),
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    # Parent-side finalize point for the HF-row path (one caller runs per build).
    log_selection_summary(
        stats,
        source=repo_id,
        num_dataset_entries=num_dataset_entries,
        max_context_length=max_context_length,
    )
    return kept, stats


def _parse_trace_dict(
    raw: dict[str, Any],
    *,
    source: str,
    tag: str,
    idle_gap_cap_seconds: float | None = None,
    content_root_seed: int | None = None,
    content_tokenizer: str | None = None,
    prompt_corpus: str | None = None,
    max_osl: int | None = None,
) -> ParsedGraph:
    """Shared dict -> :class:`ParsedGraph` core.

    Operates on an in-memory Weka trace row ``dict`` already matching the schema
    (the same dict a ``.json`` file or a HuggingFace dataset row contains), so
    the file path (read file -> dict) and the HF path (``load_dataset`` -> row
    dict) feed the identical topology / aux / flat-chain / content-synthesis
    pipeline. Only the SOURCE of the dict differs.

    ``source`` is a human-readable origin label (a file path or an
    ``org/name#row`` HF locator) used only for error messages and
    ``ParsedGraph.source_path``.
    """
    weka_trace = _validate_weka_trace(raw, source=source)

    from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME
    from aiperf.dataset.graph.adapters.weka.trie_build import (
        build_trie_graph,
    )
    from aiperf.dataset.graph.models import TraceRecord

    parsed, pool = build_trie_graph(
        weka_trace,
        tokenizer_name=content_tokenizer or BUILTIN_TOKENIZER_NAME,
        prompt_corpus=prompt_corpus or "coding",
        root_seed=content_root_seed,
        idle_gap_cap_seconds=idle_gap_cap_seconds,
        max_osl=max_osl,
    )
    # The trivial trie IR is a single top graph with no subgraphs; attach the
    # trace record (with the base tag) so the schedule plane + worker address
    # its nodes by trace id + node ordinal, and surface the
    # SegmentPool so the build plane drains it into a GraphSegmentBackingStore.
    trie_trace = TraceRecord(id=weka_trace.id, tags=[tag])
    return msgspec.structs.replace(
        parsed,
        traces=[trie_trace],
        segment_pool=pool,
    )


class WekaTraceAdapter:
    """Weka KV-cache-tester trace v1 graph adapter.

    See module docstring for topology / prompt-synthesis details. Implements
    :class:`GraphAdapterProtocol` (registered under ``graph_adapter.weka_trace``
    in ``plugins.yaml`` with ``detection_priority: 85``).
    """

    @classmethod
    def can_load(cls, path: Path) -> bool:
        """Return True if ``path`` looks like a Weka trace file or directory.

        HuggingFace id: a non-existent ``org/name`` path (no workload file
        extension) that carries the weka corpus marker is recognized as a weka
        HF dataset id so ``detect_format`` resolves it to ``weka_trace`` and the
        load goes straight through ``datasets.load_dataset`` — no file is read.
        An arbitrary ``org/name`` WITHOUT the marker is NOT treated as weka.
        Single ``.json`` file: read the first ~4 KB, parse as JSON, require the
        discriminator key set ``{id: str, models: list, block_size: int,
        hash_id_scope: "local"|"global", requests: list}``, AND reject objects
        carrying any top-level key outside the WekaTrace field set. Directory: pick the
        lexicographically-first ``*.json`` child and apply the file check.
        Sniffs are bounded — no full-file parse on detection.
        """
        if _looks_like_hf_dataset_id(path):
            return True
        if path.is_dir():
            try:
                candidates = sorted(
                    c
                    for c in path.iterdir()
                    if c.is_file() and c.suffix.lower() == ".json"
                )
            except OSError:
                return False
            if not candidates:
                return False
            return cls._file_matches(candidates[0])
        if path.suffix.lower() == ".json":
            return cls._file_matches(path)
        return False

    @classmethod
    def _file_matches(cls, path: Path) -> bool:
        """Bounded JSON sniff on a single ``.json`` file.

        We attempt a streaming decode of the first ``_DETECTION_SNIFF_BYTES``;
        if that slice contains a complete JSON object we evaluate it directly.
        Otherwise we fall back to a full-file parse only when the head slice
        already shows the expected discriminator-key prefix, so detection
        remains O(1) on non-Weka inputs.
        """
        try:
            with path.open("rb") as f:
                head = f.read(_DETECTION_SNIFF_BYTES)
        except OSError:
            return False
        if not head:
            return False
        try:
            doc = orjson.loads(head)
        except orjson.JSONDecodeError:
            # Head slice was truncated mid-document. Only do a full parse if
            # the head contains all five discriminator-key prefixes; this keeps
            # detection cheap on unrelated large JSON files.
            if not _head_has_signature_keys(head):
                return False
            try:
                with path.open("rb") as f:
                    doc = orjson.loads(f.read())
            except (OSError, orjson.JSONDecodeError):
                return False
        return _is_weka_trace_object(doc)

    @classmethod
    def parse(cls, path: Path, ctx: GraphParseContext | None = None) -> ParsedGraph:
        """Convert ``path`` into a :class:`ParsedGraph` via :func:`from_weka_trace`.

        ``ctx`` carries the run-derived knobs (seed / tokenizer / corpus /
        max_osl / idle-gap cap), each forwarded ONLY when set so a ctx-less
        parse matches the :func:`from_weka_trace` defaults byte-for-byte.
        ``idle_gap_cap_seconds`` is TRI-STATE: ``UNSET`` keeps the entry's
        ``_USE_DEFAULT`` default, while an explicit ``None`` forwards verbatim and
        DISABLES warping (the user's ``synthesis.idle_gap_cap_seconds: null``).
        The run tokenizer trust/revision publish to the loader-preload env
        before any callbacks are built (:func:`publish_ctx_tokenizer_env`).
        ``ctx.num_dataset_entries`` / ``ctx.max_context_length`` forward the
        schema-only filter-then-cap trace selection (ai-dynamo/aiperf#1106) when
        set; the run's build plane drains :func:`stream_weka_trace_segment_payloads`
        (also selection-aware), so both this oracle parse and the run honor them.
        """
        publish_ctx_tokenizer_env(ctx)
        kwargs: dict[str, Any] = {}
        if ctx is not None:
            if ctx.content_root_seed is not None:
                kwargs["content_root_seed"] = ctx.content_root_seed
            if ctx.content_tokenizer is not None:
                kwargs["content_tokenizer"] = ctx.content_tokenizer
            if ctx.prompt_corpus is not None:
                kwargs["prompt_corpus"] = ctx.prompt_corpus
            if ctx.max_osl is not None:
                kwargs["max_osl"] = ctx.max_osl
            if ctx.idle_gap_cap_seconds is not UNSET:
                kwargs["idle_gap_cap_seconds"] = ctx.idle_gap_cap_seconds
            if ctx.num_dataset_entries is not None:
                kwargs["num_dataset_entries"] = ctx.num_dataset_entries
            if ctx.max_context_length is not None:
                kwargs["max_context_length"] = ctx.max_context_length
        return from_weka_trace(path, **kwargs)


# Discriminator-key prefixes we require to appear in the head sniff before
# we are willing to fall back to a full-file parse on a truncated head.
_SIGNATURE_KEY_PREFIXES: tuple[bytes, ...] = (
    b'"id"',
    b'"models"',
    b'"block_size"',
    b'"hash_id_scope"',
    b'"requests"',
)


def _head_has_signature_keys(head: bytes) -> bool:
    return all(prefix in head for prefix in _SIGNATURE_KEY_PREFIXES)


def _hf_dataset_id_str(arg: str | Path) -> str:
    """Forward-slash string form of a candidate HF ``org/name`` repo id.

    An HF id that round-trips through ``Path`` flips its ``/`` to ``\\`` on
    Windows, which breaks the single-slash repo-id shape check and the
    canonical-repo prefix compare, and would 404 against the hub. HF repo ids
    are always forward-slash, so normalize before matching or loading.
    """
    s = str(arg)
    if IS_WINDOWS:
        s = s.replace("\\", "/")
    return s


def _looks_like_hf_dataset_id(arg: str | Path) -> bool:
    """True if ``arg`` is a weka HuggingFace ``org/name`` repo id, not a local path.

    The published weka HF corpora are referenced by ``--graph <org/name>`` and
    carry the weka corpus marker in the repo name (e.g.
    ``semianalysisai/cc-traces-weka-061526``). We treat the arg as a weka HF id
    only when ALL of the following hold:

    * it is NOT an existing filesystem path (an existing path -- even one shaped
      like ``org/name`` -- keeps the file/dir behavior),
    * its leading path component does NOT exist as a local directory (a typo'd
      relative path like ``traces/weka-061526`` under an existing ``traces/``
      dir is a local-path mistake, not an HF repo id),
    * it has no recognized workload file extension,
    * it matches the repo-id shape (single slash, ``org/name`` -- which also
      rejects path-like markers such as a ``./`` prefix or a trailing ``/``),
      and
    * it carries the case-insensitive weka marker (:data:`_HF_WEKA_MARKER`).

    The marker is the tightening: an arbitrary ``org/name`` string with no weka
    marker (``meta-llama/Llama-3``, ``openai/gpt``, ``a/b``) is NOT a weka HF id
    and stays on the unchanged linear pipeline rather than being sent to
    ``datasets.load_dataset``. When the heuristic DOES fire and the HF load then
    fails, :func:`_load_hf_rows` raises with both interpretations (local path vs
    HF id) so a misrouted arg is still diagnosable.
    """
    s = _hf_dataset_id_str(arg)
    p = Path(s)
    if p.exists():
        return False
    lowered = s.lower()
    if any(lowered.endswith(suffix) for suffix in _HF_EXCLUDED_SUFFIXES):
        return False
    if not _HF_REPO_ID_RE.match(s):
        return False
    if _HF_WEKA_MARKER not in lowered:
        return False
    # The regex guarantees exactly one slash, so ``p.parent`` is the leading
    # component; if it exists locally the arg is a typo'd local path.
    return not p.parent.exists()


def _is_weka_trace_object(doc: Any) -> bool:
    """True only for a strict on-disk weka :class:`WekaTrace` discriminator.

    Requires the five discriminator keys with their expected types
    (``id: str``, ``models: list``, ``block_size: int`` (not bool),
    ``hash_id_scope`` in :data:`_SUPPORTED_HASH_ID_SCOPES`, ``requests: list``)
    AND rejects any object carrying a top-level key outside
    :data:`_WEKA_ALLOWED_KEYS`. The foreign-key rejection is what prevents a
    mooncake / sharegpt / synthetic object that merely happens to share the
    five weka keys from being misclassified as a weka graph workload --
    mirroring ``WekaTrace``'s ``extra="forbid"`` config.
    """
    if not isinstance(doc, dict):
        return False
    if not isinstance(doc.get("id"), str):
        return False
    if not isinstance(doc.get("models"), list):
        return False
    if not isinstance(doc.get("block_size"), int) or isinstance(
        doc.get("block_size"), bool
    ):
        return False
    if doc.get("hash_id_scope") not in _SUPPORTED_HASH_ID_SCOPES:
        return False
    if not isinstance(doc.get("requests"), list):
        return False
    return _WEKA_ALLOWED_KEYS.issuperset(doc.keys())


__all__ = [
    "EmptyWekaTraceError",
    "WekaHashScopeError",
    "WekaSchemaError",
    "WekaTraceAdapter",
    "WekaTraceAdapterError",
    "from_weka_trace",
]
