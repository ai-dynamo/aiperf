# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph validator framework: shared helpers, rule implementations, and entry points.

See `docs/reference/graph-ir-validation.md` for the full rule list. Rules 19
and 20 are enforced at parse time (see `parser.py`).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterator
from enum import Enum

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.dataset.graph.models import (
    END_NODE_ID,
    START_NODE_ID,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    StaticEdge,
)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(AIPerfBaseModel):
    """A single validator finding."""

    rule_id: str = Field(description="Rule identifier, e.g. 'rule-1'.")
    severity: ValidationSeverity = Field(
        description="ERROR (blocks execution) or WARNING (informational)."
    )
    location: str = Field(
        description="Human-readable pointer (e.g. 'graph.nodes.plan')."
    )
    message: str = Field(description="Plain-English description of the problem.")
    suggested_fix: str | None = Field(
        default=None, description="Optional remediation hint."
    )


_RESERVED_NAMES = {START_NODE_ID, END_NODE_ID}
_KNOWN_VERSIONS = {"2.0"}


def validate(parsed: ParsedGraph) -> list[ValidationIssue]:
    """Run every implemented rule, return all issues. Caller decides whether warnings block.

    Rules run over the main ``parsed.graph`` AND over every per-trace graph in
    ``parsed.graphs`` (multi-graph workloads: weka heterogeneous directories,
    per-trace native lowering). Issues found in a ``parsed.graphs`` entry are
    re-located under ``graphs[<name>]`` so the offending graph is identifiable.
    """
    issues = _validate_graph_record(parsed.graph)
    for name, graph in parsed.graphs.items():
        if graph is parsed.graph:
            # The native lowering aliases parsed.graph to the first graphs
            # entry; skip the duplicate pass.
            continue
        for issue in _validate_graph_record(graph):
            issues.append(
                issue.model_copy(
                    update={"location": _graphs_location(name, issue.location)}
                )
            )
    return issues


def _validate_graph_record(graph: GraphRecord) -> list[ValidationIssue]:
    """Run every implemented rule over one :class:`GraphRecord`."""
    issues: list[ValidationIssue] = []
    issues.extend(_rule_01_cycles(graph))
    issues.extend(_rule_09_overwrite_conflict(graph))
    issues.extend(_rule_11_reserved_names(graph))
    issues.extend(_rule_12_version(graph))
    issues.extend(_rule_13_provenance_tool(graph))
    issues.extend(_rule_15_expected_cached_tokens(graph))
    issues.extend(_rule_21_unreachable(graph))
    issues.extend(_rule_54_edge_delay_exclusivity(graph))
    issues.extend(_rule_55_first_token_anchor_shape(graph))
    issues.extend(_rule_56_edge_endpoints(graph))
    issues.extend(_rule_57_finite_delays(graph))
    return issues


def _graphs_location(graph_name: str, location: str) -> str:
    """Re-root a ``graph.*`` issue location under ``graphs[<name>]``."""
    prefix = "graph."
    if location.startswith(prefix):
        return f"graphs[{graph_name}].{location[len(prefix) :]}"
    return f"graphs[{graph_name}]: {location}"


# ---- helpers ---------------------------------------------------------------


def _build_adjacency(graph: GraphRecord) -> dict[str, set[str]]:
    """Map src -> set of declared static successors."""
    adj: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        if isinstance(e, StaticEdge):
            adj[e.source].add(e.target)
    return adj


def _node_writers(graph: GraphRecord) -> dict[str, list[str]]:
    """Map channel name to the list of node ids that write to it."""
    writers: dict[str, list[str]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        if isinstance(node, LlmNode):
            writers[node.output].append(nid)
    return writers


# ---- rules -----------------------------------------------------------------


def _rule_01_cycles(graph: GraphRecord) -> list[ValidationIssue]:
    # Iterative DFS with an explicit stack: recorded corpora reach 100k+-node
    # chains, far past Python's recursion limit.
    adj = _build_adjacency(graph)
    color: dict[str, int] = {}
    GREY, BLACK = 1, 2

    for root in [*graph.nodes.keys(), START_NODE_ID]:
        if color.get(root, 0) != 0:
            continue
        color[root] = GREY
        stack: list[tuple[str, Iterator[str]]] = [(root, iter(adj.get(root, ())))]
        while stack:
            node, successors = stack[-1]
            for succ in successors:
                c = color.get(succ, 0)
                if c == GREY:
                    return [
                        ValidationIssue(
                            rule_id="rule-1",
                            severity=ValidationSeverity.ERROR,
                            location="graph.edges",
                            message="Graph contains a cycle in the static edges.",
                            suggested_fix="Pre-unroll cycles in the trace topology (cycles are a future graph feature).",
                        )
                    ]
                if c == 0:
                    color[succ] = GREY
                    stack.append((succ, iter(adj.get(succ, ()))))
                    break
            else:
                color[node] = BLACK
                stack.pop()
    return []


def _rule_09_overwrite_conflict(graph: GraphRecord) -> list[ValidationIssue]:
    """Two or more nodes write the same overwrite-reducer channel."""
    writers = _node_writers(graph)
    issues: list[ValidationIssue] = []
    for ch, nodes in writers.items():
        if len(nodes) <= 1:
            continue
        spec = graph.state.get(ch)
        if spec is not None and spec.reducer is not ReducerName.OVERWRITE:
            continue
        issues.append(
            ValidationIssue(
                rule_id="rule-9",
                severity=ValidationSeverity.ERROR,
                location=f"graph.state.{ch}",
                message=(
                    f"Channel '{ch}' has overwrite reducer but is written by "
                    f"multiple nodes: {sorted(nodes)}."
                ),
                suggested_fix=(
                    f"Either give '{ch}' an add_messages reducer (only valid for "
                    f"type=messages) or rename one of the writer outputs."
                ),
            )
        )
    return issues


def _rule_11_reserved_names(graph: GraphRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for nid in graph.nodes:
        if nid in _RESERVED_NAMES or nid.startswith("_aiperf"):
            issues.append(
                ValidationIssue(
                    rule_id="rule-11",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.nodes.{nid}",
                    message=(
                        f"Node id '{nid}' is reserved (matches START/END or "
                        f"begins with '_aiperf')."
                    ),
                    suggested_fix="Rename the node.",
                )
            )
    return issues


def _rule_12_version(graph: GraphRecord) -> list[ValidationIssue]:
    if graph.version in _KNOWN_VERSIONS:
        return []
    return [
        ValidationIssue(
            rule_id="rule-12",
            severity=ValidationSeverity.ERROR,
            location="graph.version",
            message=(
                f"Unknown major version {graph.version!r}; known: "
                f"{sorted(_KNOWN_VERSIONS)}."
            ),
            suggested_fix=f"Set version to one of {sorted(_KNOWN_VERSIONS)}.",
        )
    ]


def _rule_13_provenance_tool(graph: GraphRecord) -> list[ValidationIssue]:
    p = graph.provenance
    if p.source == "hand-authored":
        return []
    tool = (p.tool or "").strip()
    if not tool:
        return [
            ValidationIssue(
                rule_id="rule-13",
                severity=ValidationSeverity.ERROR,
                location="graph.provenance.tool",
                message=(
                    f"provenance.source={p.source!r} requires a non-empty "
                    f"provenance.tool (origin tool plus version)."
                ),
                suggested_fix=(
                    "Set provenance.tool to '<tool-name>/<version>', or change "
                    "source to 'hand-authored'."
                ),
            )
        ]
    if tool == "manual":
        # "manual" is the field default; an adapter-emitted graph still
        # carrying it never stamped its generating tool.
        return [
            ValidationIssue(
                rule_id="rule-13",
                severity=ValidationSeverity.WARNING,
                location="graph.provenance.tool",
                message=(
                    f"provenance.source={p.source!r} carries the default "
                    f"provenance.tool 'manual'; an adapter-emitted graph "
                    f"should stamp its generating tool plus version."
                ),
                suggested_fix=(
                    "Have the generating adapter set provenance.tool to "
                    "'<tool-name>/<version>', or change source to "
                    "'hand-authored'."
                ),
            )
        ]
    return []


def _rule_15_expected_cached_tokens(graph: GraphRecord) -> list[ValidationIssue]:
    """WARNING: emit when an LLM node sets expected.cache_read_tokens or
    expected.cache_creation_tokens. The engine may not report cache fields; we
    cannot know at file-load time whether the configured engine supports them.
    Surface as a warning so users see it."""
    issues: list[ValidationIssue] = []
    cache_fields = ("cache_read_tokens", "cache_creation_tokens")
    for nid, node in graph.nodes.items():
        if not isinstance(node, LlmNode) or node.expected is None:
            continue
        for field in cache_fields:
            if getattr(node.expected, field) is None:
                continue
            issues.append(
                ValidationIssue(
                    rule_id="rule-15",
                    severity=ValidationSeverity.WARNING,
                    location=f"graph.nodes.{nid}.expected.{field}",
                    message=(
                        f"Node '{nid}' sets expected.{field} but not all engines "
                        f"report {field}; this assertion will be skipped if the "
                        f"engine doesn't surface that field."
                    ),
                )
            )
    return issues


def _rule_21_unreachable(graph: GraphRecord) -> list[ValidationIssue]:
    adj = _build_adjacency(graph)
    seen: set[str] = set()
    q: deque[str] = deque([START_NODE_ID])
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        q.extend(adj.get(u, ()))
    seen.discard(START_NODE_ID)
    seen.discard(END_NODE_ID)
    declared = set(graph.nodes.keys())
    unreachable = declared - seen
    issues: list[ValidationIssue] = []
    for nid in sorted(unreachable):
        issues.append(
            ValidationIssue(
                rule_id="rule-21",
                severity=ValidationSeverity.ERROR,
                location=f"graph.nodes.{nid}",
                message=f"Node '{nid}' is unreachable from START.",
                suggested_fix=(
                    f"Add an edge `{{from: <ancestor>, to: {nid}}}` or remove the node."
                ),
            )
        )
    return issues


def _rule_54_edge_delay_exclusivity(graph: GraphRecord) -> list[ValidationIssue]:
    """Start-anchored edge legality (two checks, one rule).

    1. An edge must not set both ``delay_after_predecessor_us`` and
       ``delay_after_predecessor_start_us``: the first anchors the successor to
       the predecessor's COMPLETION, the second to its DISPATCH; honoring both
       on one edge has no recorded-trace meaning.
    2. A ``START``-sourced edge must not be start-anchored. The ``START``
       pseudo-node never dispatches, so the runtime routes such a target to
       ``Scheduler._start_anchored_succ`` where ``entry_nodes()`` never sees it
       and ``start_anchored_successors("START")`` is never consulted -- the
       target would be silently orphaned. Fail loudly instead of scheduling a
       node that can never fire.
    """
    issues: list[ValidationIssue] = []
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        if (
            edge.delay_after_predecessor_us is not None
            and edge.delay_after_predecessor_start_us is not None
        ):
            issues.append(
                ValidationIssue(
                    rule_id="rule-54",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}]",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} sets both "
                        "delay_after_predecessor_us and "
                        "delay_after_predecessor_start_us; they are mutually "
                        "exclusive."
                    ),
                    suggested_fix=(
                        "Keep exactly one anchor: end-anchored "
                        "(delay_after_predecessor_us) or start-anchored "
                        "(delay_after_predecessor_start_us)."
                    ),
                )
            )
        if (
            edge.source == START_NODE_ID
            and edge.delay_after_predecessor_start_us is not None
        ):
            issues.append(
                ValidationIssue(
                    rule_id="rule-54",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}]",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} is "
                        "start-anchored (delay_after_predecessor_start_us), but "
                        "START-sourced edges cannot be start-anchored; the START "
                        "pseudo-node never dispatches, so the target would never "
                        "be scheduled."
                    ),
                    suggested_fix=(
                        "Use min_start_delay_us for an absolute offset from trace "
                        "start instead of delay_after_predecessor_start_us."
                    ),
                )
            )
    return issues


def _rule_55_first_token_anchor_shape(graph: GraphRecord) -> list[ValidationIssue]:
    """A first-token-anchored edge must carry its dispatch fallback and a real
    source: ``delay_after_predecessor_first_token_us`` requires
    ``delay_after_predecessor_start_us`` on the same edge (the runtime falls
    back to dispatch + start delay when the predecessor never streams a first
    token), must not combine with the completion anchor
    ``delay_after_predecessor_us``, and cannot source at START (the START
    pseudo-node never dispatches or streams)."""
    issues: list[ValidationIssue] = []
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        if edge.delay_after_predecessor_first_token_us is None:
            continue
        if edge.delay_after_predecessor_start_us is None:
            issues.append(
                ValidationIssue(
                    rule_id="rule-55",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}]",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} is "
                        "first-token-anchored (delay_after_predecessor_first_token_us) "
                        "but sets no delay_after_predecessor_start_us; the runtime "
                        "needs the start delay as the dispatch fallback when the "
                        "predecessor terminates without a first token."
                    ),
                    suggested_fix=(
                        "Add delay_after_predecessor_start_us to this edge as the "
                        "dispatch-anchored fallback for delay_after_predecessor_first_token_us."
                    ),
                )
            )
        if edge.delay_after_predecessor_us is not None:
            issues.append(
                ValidationIssue(
                    rule_id="rule-55",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}]",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} sets both "
                        "delay_after_predecessor_us and "
                        "delay_after_predecessor_first_token_us; the completion "
                        "anchor and the first-token anchor are mutually exclusive."
                    ),
                    suggested_fix=(
                        "Drop delay_after_predecessor_us and keep "
                        "delay_after_predecessor_first_token_us with its "
                        "delay_after_predecessor_start_us fallback."
                    ),
                )
            )
        if edge.source == START_NODE_ID:
            issues.append(
                ValidationIssue(
                    rule_id="rule-55",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}]",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} is "
                        "first-token-anchored (delay_after_predecessor_first_token_us), "
                        "but START-sourced edges cannot be first-token-anchored; the "
                        "START pseudo-node never dispatches or streams a first token."
                    ),
                    suggested_fix=(
                        "Use min_start_delay_us for an absolute offset from trace "
                        "start instead of delay_after_predecessor_first_token_us."
                    ),
                )
            )
    return issues


def _rule_56_edge_endpoints(graph: GraphRecord) -> list[ValidationIssue]:
    """Every edge endpoint must be a declared node or the matching sentinel.

    A dangling endpoint (typo'd node id, `START` as a target, `END` as a
    source) otherwise validates clean and produces a node that never fires or
    an edge the scheduler silently ignores.
    """
    declared = set(graph.nodes)
    issues: list[ValidationIssue] = []
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        loc = f"graph.edges[{edge.source}->{edge.target}]"
        if edge.source not in declared and edge.source != START_NODE_ID:
            issues.append(
                ValidationIssue(
                    rule_id="rule-56",
                    severity=ValidationSeverity.ERROR,
                    location=loc,
                    message=(
                        f"edge source '{edge.source}' is neither a declared node "
                        f"nor the START sentinel."
                    ),
                    suggested_fix=(
                        "Declare the node in graph.nodes or fix the edge "
                        "source (only 'START' is a valid non-node source)."
                    ),
                )
            )
        if edge.target not in declared and edge.target != END_NODE_ID:
            issues.append(
                ValidationIssue(
                    rule_id="rule-56",
                    severity=ValidationSeverity.ERROR,
                    location=loc,
                    message=(
                        f"edge target '{edge.target}' is neither a declared node "
                        f"nor the END sentinel."
                    ),
                    suggested_fix=(
                        "Declare the node in graph.nodes or fix the edge "
                        "target (only 'END' is a valid non-node target)."
                    ),
                )
            )
    return issues


_EDGE_DELAY_FIELDS = (
    "delay_after_predecessor_us",
    "min_start_delay_us",
    "delay_after_predecessor_start_us",
    "delay_after_predecessor_first_token_us",
)


def _rule_57_finite_delays(graph: GraphRecord) -> list[ValidationIssue]:
    """Every delay value must be finite.

    ``msgspec.Meta(ge=0)`` admits ``+inf``, and the executor gates a successor
    at ``finish + delay`` — an infinite delay hangs the trace for the whole
    benchmark. The loose decoder rejects non-finite values at decode time
    (see ``decode.py``); this rule catches graphs constructed directly from
    typed structs (adapter output, programmatic builders).
    """
    issues: list[ValidationIssue] = []
    for edge in graph.edges:
        if not isinstance(edge, StaticEdge):
            continue
        for field in _EDGE_DELAY_FIELDS:
            value = getattr(edge, field)
            if value is None or math.isfinite(value):
                continue
            issues.append(
                ValidationIssue(
                    rule_id="rule-57",
                    severity=ValidationSeverity.ERROR,
                    location=f"graph.edges[{edge.source}->{edge.target}].{field}",
                    message=(
                        f"edge {edge.source!r} -> {edge.target!r} sets "
                        f"{field}={value!r}; delay values must be finite (a "
                        f"non-finite gate never clears and hangs the trace)."
                    ),
                    suggested_fix=(
                        f"Set {field} to a finite microsecond value or drop it."
                    ),
                )
            )
    for nid, node in graph.nodes.items():
        value = node.min_start_delay_us
        if value is None or math.isfinite(value):
            continue
        issues.append(
            ValidationIssue(
                rule_id="rule-57",
                severity=ValidationSeverity.ERROR,
                location=f"graph.nodes.{nid}.min_start_delay_us",
                message=(
                    f"node '{nid}' sets min_start_delay_us={value!r}; delay "
                    f"values must be finite (a non-finite gate never clears "
                    f"and hangs the trace)."
                ),
                suggested_fix=(
                    "Set min_start_delay_us to a finite microsecond value or drop it."
                ),
            )
        )
    return issues


__all__ = [
    "ValidationIssue",
    "ValidationSeverity",
    "validate",
]
