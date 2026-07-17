# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph file parser backed by native readers and registered graph adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import msgspec
import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.decode import GraphDecodeError, decode_graph
from aiperf.dataset.graph.models import (
    END_NODE_ID,
    START_NODE_ID,
    GraphRecord,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
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

    Auto-detects the native graph format or delegates to registered graph
    adapters, using extension and content sniffing as defined by the plugin
    registry. Pass `format=` to override detection (CLI: `--graph-format`).

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
    The ``native`` path takes NO adapter kwargs (it has no adapter ``parse``),
    so passing any with ``format="native"`` (or a file detected as native)
    raises :class:`GraphParseError` up front rather than a confusing
    downstream signature error.
    """
    p = Path(path)
    fmt = format or _detect(p)
    if fmt == "native" and adapter_kwargs:
        raise GraphParseError(
            f"{p}: the native parse path accepts no adapter-specific kwargs, but "
            f"received {sorted(adapter_kwargs)}; adapter_kwargs (e.g. dynamo's "
            f"direct_store) are only valid for a registered graph adapter"
        )
    try:
        pb = (
            parse_native(p)
            if fmt == "native"
            else _parse_via_adapter(p, fmt, ctx, **adapter_kwargs)
        )
    except (msgspec.ValidationError, GraphDecodeError) as e:
        raise GraphParseError(_format_ir_error(p, e)) from e
    return pb


def _format_ir_error(
    path: Path, exc: msgspec.ValidationError | GraphDecodeError
) -> str:
    """Render an IR (de)coding error as a single readable line.

    Without this wrapper, parse-time IR errors (e.g. ``max_iterations=0``,
    out-of-range field values) escape ``parse_graph`` as
    raw tracebacks. The CLI catches ``GraphParseError`` cleanly, so callers see
    a friendly message instead of a stack trace. msgspec carries the field path
    inline in its message (``... - at `$.nodes...```).
    """
    return f"{path}: IR error: {exc}"


def _detect(path: Path) -> WorkloadFormat:
    try:
        return detect_format(path)
    except WorkloadFormatError as e:
        raise GraphParseError(str(e)) from e


def parse_native(path: Path) -> ParsedGraph:
    """Native graph parse entry point. Used by `NativeGraphAdapter`.

    Native graph parsing is the canonical (non-adapter) path: JSONL or YAML
    that already conforms to the graph schema. Auto-derive runs at the end
    so callers get the same `ParsedGraph` shape regardless of input format.
    """
    ext = path.suffix.lower()
    records = _read_jsonl(path) if ext == ".jsonl" else _read_yaml(path)
    pb = _assemble(records)
    pb = _auto_inject_start_end(pb)
    from aiperf.dataset.graph.auto_derive import auto_derive
    from aiperf.dataset.graph.native_lowering import lower_native_to_unified

    return lower_native_to_unified(auto_derive(pb))


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
        # only need to catch GraphParseError. A TypeError from an
        # adapter_kwarg the adapter's parse does not accept is NOT caught here:
        # it propagates as the documented fail-loud (a build-plane wiring bug,
        # not a workload-file error).
        raise GraphParseError(str(e)) from e


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for lineno, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = orjson.loads(stripped)
            except orjson.JSONDecodeError as e:
                raise GraphParseError(
                    f"{path}: line {lineno}: invalid JSON: {e}"
                ) from e
            if not isinstance(rec, dict):
                raise GraphParseError(
                    f"{path}: line {lineno}: record must be a JSON object"
                )
            out.append(rec)
    return out


def _read_yaml(path: Path) -> list[dict[str, Any]]:
    import yaml

    try:
        docs = list(yaml.safe_load_all(path.read_text()))
    except yaml.YAMLError as e:
        raise GraphParseError(f"{path}: invalid YAML: {e}") from e
    docs = [d for d in docs if d is not None]
    if len(docs) > 1 or (len(docs) == 1 and _looks_like_multi_doc(docs[0])):
        return [_normalize_yaml_record(d, path) for d in docs]
    if not docs:
        return []
    return _expand_single_doc(docs[0], path)


def _looks_like_multi_doc(doc: Any) -> bool:
    return isinstance(doc, dict) and "kind" in doc


def _normalize_yaml_record(doc: Any, path: Path) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise GraphParseError(f"{path}: YAML document must be a mapping")
    return doc


_SINGLE_DOC_KEYS = ("graph", "traces")


def _expand_single_doc(doc: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(doc, dict):
        raise GraphParseError(f"{path}: top-level YAML must be a mapping")
    unknown = [k for k in doc if k not in _SINGLE_DOC_KEYS]
    if unknown:
        import difflib

        hints: list[str] = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), _SINGLE_DOC_KEYS, n=1)
            hints.append(
                f"{key!r} (did you mean {close[0]!r}?)" if close else f"{key!r}"
            )
        raise GraphParseError(
            f"{path}: unknown top-level key(s) {', '.join(hints)}; a "
            f"single-document graph workload may only contain "
            f"{list(_SINGLE_DOC_KEYS)} (node/edge topology goes under 'graph:')"
        )
    if not doc:
        raise GraphParseError(
            f"{path}: top-level mapping contains none of {list(_SINGLE_DOC_KEYS)}; "
            f"author the workload as a 'graph:' block plus a 'traces:' list"
        )
    out: list[dict[str, Any]] = []
    if "graph" in doc:
        graph_body = doc["graph"] or {}
        if not isinstance(graph_body, dict):
            raise GraphParseError(f"{path}: 'graph' must be a mapping")
        out.append({"kind": "graph", **graph_body})
    if "traces" in doc:
        traces = doc["traces"] or []
        if not isinstance(traces, list):
            raise GraphParseError(f"{path}: 'traces' must be a list")
        for t in traces:
            if not isinstance(t, dict):
                raise GraphParseError(f"{path}: each trace must be a mapping")
            out.append({"kind": "trace", **t})
    return out


def _auto_inject_start_end(pb: ParsedGraph) -> ParsedGraph:
    """Inject ``START -> <root>`` and ``<leaf> -> END`` edges when missing.

    Mirrors the foreign-adapter read paths' sentinel-edge auto-injection so a
    native workload that declares only inter-node edges produces an equivalent
    :class:`ParsedGraph` shape. Behavior-preserving: existing explicit
    START/END edges are left untouched.
    """
    if not pb.graph.nodes:
        return pb
    existing_sources: set[str] = set()
    existing_targets: set[str] = set()
    for e in pb.graph.edges:
        if isinstance(e, StaticEdge):
            existing_sources.add(e.source)
            existing_targets.add(e.target)
    has_start = any(
        isinstance(e, StaticEdge) and e.source == START_NODE_ID for e in pb.graph.edges
    )
    has_end = any(
        isinstance(e, StaticEdge) and e.target == END_NODE_ID for e in pb.graph.edges
    )
    if has_start and has_end:
        return pb
    new_edges: list[Any] = list(pb.graph.edges)
    if not has_start:
        roots = [nid for nid in pb.graph.nodes if nid not in existing_targets]
        for r in roots:
            new_edges.insert(0, StaticEdge(source=START_NODE_ID, target=r))
    if not has_end:
        leaves = [nid for nid in pb.graph.nodes if nid not in existing_sources]
        for leaf in leaves:
            new_edges.append(StaticEdge(source=leaf, target=END_NODE_ID))
    if len(new_edges) == len(pb.graph.edges):
        return pb
    return msgspec.structs.replace(
        pb, graph=msgspec.structs.replace(pb.graph, edges=new_edges)
    )


def _assemble(records: list[dict[str, Any]]) -> ParsedGraph:
    graph: GraphRecord | None = None
    traces: list[TraceRecord] = []
    any_trace_seen = False
    for idx, rec in enumerate(records):
        kind = rec.get("kind", "trace")
        body = {k: v for k, v in rec.items() if k != "kind"}
        if kind == "graph":
            if any_trace_seen:
                raise GraphParseError(
                    f"record {idx}: graph record must precede trace records (rule-19)"
                )
            if graph is not None:
                raise GraphParseError(
                    f"record {idx}: more than one 'kind: graph' record (rule-20)"
                )
            graph = decode_graph(body)
        elif kind == "trace":
            any_trace_seen = True
            try:
                traces.append(msgspec.convert(body, type=TraceRecord))
            except msgspec.ValidationError as e:
                raise GraphParseError(f"record {idx}: trace decode failed: {e}") from e
        else:
            raise GraphParseError(f"record {idx}: unknown kind {kind!r}")
    return ParsedGraph(
        graph=graph if graph is not None else GraphRecord(),
        traces=traces,
    )


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
"""String alias for graph adapter names (e.g. "weka_trace", "native").

The dynamic `GraphAdapterType` enum is intentionally not referenced here: the
`graph_adapter` plugin category is not registered on this base, and the IR
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
