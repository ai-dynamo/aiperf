# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-registry protocol for graph workload-format adapters.

Adapters are registered under the `graph_adapter` plugin category and
selected by name (`--graph-format`) or by priority-ordered `can_load()`
auto-detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parse_context import GraphParseContext


class GraphAdapterProtocol(Protocol):
    """Convert a third-party trace/log file into a ParsedGraph.

    Detection is two-stage:
      * `can_load(path)` is a cheap predicate over file extension + content
        sniff. Implementations should read only enough bytes to disambiguate.
      * `parse(path, ctx)` does the full conversion. `ctx` carries the
        run-derived knobs (see :class:`GraphParseContext`); adapters map the
        fields they consume onto their entry function, forwarding a field
        ONLY when it is set so `parse(path)` stays byte-equal to the
        protocol-default entry. Adapters may raise an adapter-specific
        ValueError subclass on failure; callers wrap into `GraphParseError`
        at the parser layer.

    Multiple adapters may return True for the same file; the registry's
    `detection_priority` metadata breaks ties (higher wins).
    """

    @classmethod
    def can_load(cls, path: Path) -> bool:
        """Cheap predicate: does this adapter recognize ``path``?

        Read only enough bytes to disambiguate -- this runs once per registered
        adapter during auto-detection, for every candidate file. Implementations
        should not raise: an unreadable, truncated, or malformed file is simply
        ``False``. A raising ``can_load`` is downgraded to "does not claim" by
        :func:`~aiperf.dataset.graph.parser.detect_format`, so a real error here
        is silently invisible rather than fatal.
        """
        ...

    @classmethod
    def parse(cls, path: Path, ctx: GraphParseContext | None = None) -> ParsedGraph:
        """Convert ``path`` into a :class:`ParsedGraph`.

        ``ctx`` carries run-derived VALUES (see :class:`GraphParseContext`).
        Forward a field onto the adapter's own entry function ONLY when it is
        set, so a partial ctx never clobbers an entry default with ``None`` and
        ``parse(path)`` stays byte-equal to the protocol-default entry.

        Adapter-specific plumbing that a frozen value bundle cannot carry (live
        objects built by the build plane) arrives as EXTRA KEYWORD ARGUMENTS:
        :func:`~aiperf.dataset.graph.parser.parse_graph` forwards its
        ``**adapter_kwargs`` verbatim as ``adapter_cls.parse(path, ctx,
        **adapter_kwargs)``. Currently the only such kwarg is the dynamo direct
        route's ``direct_store``. This signature deliberately omits ``**kwargs``
        so that an implementation accepting only the kwargs it consumes stays a
        structural subtype of this Protocol; declare each one keyword-only and
        let an unrecognized kwarg raise ``TypeError``. The parser does NOT catch
        ``TypeError`` -- an unaccepted kwarg is a build-plane wiring bug, not a
        workload-file error, and must fail loud rather than be dropped.

        Raise an adapter-specific ``ValueError`` subclass on failure; the parser
        layer catches it and re-wraps into
        :class:`~aiperf.dataset.graph.parser.GraphParseError` with the workload
        path prefixed.
        """
        ...
