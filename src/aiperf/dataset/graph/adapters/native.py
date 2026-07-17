# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-registry facade for the canonical native graph format.

Native graph files (`.yaml` / `.yml` / `.jsonl` conforming to the graph
schema) are not third-party trace formats but are registered as a
plugin here so that detection is uniform: every recognized format
goes through the same `graph_adapter` registry walk. This adapter has
the lowest detection_priority (1) so it acts as the catch-all fallback.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parse_context import GraphParseContext


class NativeGraphAdapter:
    """Canonical AIPerf graph workload format (.yaml / .yml / .jsonl)."""

    @classmethod
    def can_load(cls, path: Path) -> bool:
        return path.suffix.lower() in (".yaml", ".yml", ".jsonl")

    @classmethod
    def parse(cls, path: Path, ctx: GraphParseContext | None = None) -> ParsedGraph:
        # ctx is accepted for protocol uniformity and ignored: native graph
        # files are fully self-describing, with no run-derived parse knobs.
        del ctx
        # `parse_native` lives in `parser`. Import it lazily so a cold
        # `import aiperf.dataset.graph.adapters` (which pulls in this module at
        # `adapters/__init__` time, e.g. via the worker's unified store client)
        # does not eagerly import the full `parser` module. `parser` imports
        # nothing under `adapters` (its format-detection helpers live inside
        # `parser` itself), so this lazy import keeps the two decoupled rather
        # than breaking a hard cycle.
        from aiperf.dataset.graph.parser import parse_native

        return parse_native(path)
