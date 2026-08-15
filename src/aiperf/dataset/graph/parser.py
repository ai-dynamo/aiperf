# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph file parser backed by registered graph adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parse_context import GraphParseContext


class GraphParseError(ValueError):
    """Raised when a graph workload file cannot be parsed."""


def parse_graph(
    path: str | Path,
    *,
    format: WorkloadFormat | None = None,
    ctx: GraphParseContext | None = None,
    **adapter_kwargs: Any,
) -> ParsedGraph:
    """Parse a workload file into a ParsedGraph.

    Delegates to registered graph adapters, using extension and content
    sniffing as defined by the plugin registry. Pass `format=` to override
    detection (CLI: `--graph-format`).

    `ctx` carries the run-derived parse knobs (see
    :class:`~aiperf.dataset.graph.parse_context.GraphParseContext`) and is
    passed opaquely to the selected adapter's ``parse(path, ctx)``; each
    adapter maps the fields it consumes and ignores the rest. ``None`` (CLI
    tooling / direct callers with no run config) keeps every adapter on its
    protocol-default entry.

    ``**adapter_kwargs`` is format-agnostic plumbing for ADAPTER-SPECIFIC knobs
    the build plane must inject as live objects the ``GraphParseContext`` (a
    frozen bundle of run-derived VALUES) cannot carry -- currently only the
    dynamo direct route's ``direct_store`` (``GraphStoreBuilder`` ->
    :func:`~aiperf.dataset.graph.workload_detect.parse_graph_workload`). It
    names no adapter: it is forwarded verbatim to ``adapter_cls.parse(path,
    ctx, **adapter_kwargs)``, so a kwarg reaching an adapter whose ``parse``
    does not accept it fails loud with ``TypeError`` (not silently dropped).
    """
    p = Path(path)
    fmt = format or _detect(p)
    try:
        pb = _parse_via_adapter(p, fmt, ctx, **adapter_kwargs)
    except msgspec.ValidationError as e:
        raise GraphParseError(_format_graph_error(p, e)) from e
    return pb


def _format_graph_error(path: Path, exc: msgspec.ValidationError) -> str:
    """Render a typed-graph-model (de)coding error as a single readable line.

    Without this wrapper, parse-time graph errors (e.g. ``max_iterations=0``,
    out-of-range field values) escape ``parse_graph`` as
    raw tracebacks. The CLI catches ``GraphParseError`` cleanly, so callers see
    a friendly message instead of a stack trace. msgspec carries the field path
    inline in its message (``... - at `$.nodes...```).
    """
    return f"{path}: graph error: {exc}"


def _detect(path: Path) -> WorkloadFormat:
    try:
        return detect_format(path)
    except WorkloadFormatError as e:
        raise GraphParseError(str(e)) from e


def _parse_via_adapter(
    path: Path,
    fmt: WorkloadFormat,
    ctx: GraphParseContext | None = None,
    **adapter_kwargs: Any,
) -> ParsedGraph:
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    try:
        adapter_cls = plugins.get_class(PluginType.GRAPH_ADAPTER, fmt)
    except Exception as e:
        raise GraphParseError(f"{path}: unknown format {fmt!r}") from e
    try:
        return adapter_cls.parse(path, ctx, **adapter_kwargs)
    except ValueError as e:
        # Adapters raise their own ValueError subclasses. Re-wrap so callers
        # only need to catch GraphParseError, prefixing the workload path the
        # adapter's own message does not carry -- adapter messages identify a
        # node/trace/segment id, which does not name the file to open in a
        # directory ingest (dynamo discovers N `trace.*.jsonl.gz` segments).
        # This matches the two sibling wrappers above, which both carry the
        # path. A TypeError from an adapter_kwarg the adapter's parse does not
        # accept is NOT caught here: it propagates as the documented fail-loud
        # (a build-plane wiring bug, not a workload-file error).
        raise GraphParseError(f"{path}: {e}") from e


# ---------------------------------------------------------------------------
# Workload-format detection
#
# Walks the `graph_adapter` plugin registry and asks each adapter's
# `can_load(path)` method whether it recognizes the file. Ties between
# multiple matches are broken by `detection_priority` from each plugin's
# metadata (higher wins). Folded in from the adapters package because `parser`
# is its only importer (no adapter ever consumed it).
# ---------------------------------------------------------------------------

_logger = AIPerfLogger(__name__)

WorkloadFormat: TypeAlias = str
"""String alias for graph adapter names (e.g. "dynamo_trace").

The dynamic `GraphAdapterType` enum is intentionally not referenced here: the
`graph_adapter` plugin category is not registered on this base, and the graph
ingestion detection path only needs the string alias at import time."""


class WorkloadFormatError(ValueError):
    """Raised when a file's workload format cannot be determined."""


def detect_format(path: str | Path) -> WorkloadFormat:
    """Return the workload format of `path`, walking the graph_adapter registry.

    Each registered adapter's `can_load(path)` is queried; ties between
    multiple matches are broken by `detection_priority` from the plugin
    metadata (higher wins). Raises `WorkloadFormatError` if no adapter
    recognizes the file.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType
    from aiperf.plugin.schema.schemas import GraphAdapterMetadata

    p = Path(path)
    matches: list[tuple[int, str]] = []
    for entry, cls in plugins.iter_all(PluginType.GRAPH_ADAPTER):
        try:
            claimed = cls.can_load(p)
        except Exception as e:
            # One adapter's sniff crashing on corrupt candidate bytes (or an
            # adapter bug) must not abort detection for every other adapter;
            # treat it as "does not claim". Real errors in the winning
            # adapter still surface from its parse().
            _logger.debug(
                f"graph adapter {entry.name!r} can_load({p}) raised {e!r}; "
                "treating as does-not-claim"
            )
            continue
        if claimed:
            meta = entry.get_typed_metadata(GraphAdapterMetadata)
            matches.append((meta.detection_priority, entry.name))
    if not matches:
        raise WorkloadFormatError(
            f"{p}: no graph adapter recognizes this file. "
            f"Registered adapters: "
            f"{[e.name for e in plugins.iter_entries(PluginType.GRAPH_ADAPTER)]}"
        )
    # max() picks the highest priority; ties resolved by lexicographic name
    # which is fine for tie-equal-priority cases since priorities should be
    # distinct for adapters whose can_load might overlap.
    return max(matches)[1]
