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
    def can_load(cls, path: Path) -> bool: ...

    @classmethod
    def parse(cls, path: Path, ctx: GraphParseContext | None = None) -> ParsedGraph: ...
