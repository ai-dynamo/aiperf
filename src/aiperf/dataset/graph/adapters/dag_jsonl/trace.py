# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter entry + detection for the ``dag_jsonl`` graph workload.

Thin registration/selection seam over the ``dag_jsonl`` core: :func:`from_dag_jsonl`
composes the three already-tested layers -- the legacy loader
(:func:`~aiperf.dataset.graph.adapters.dag_jsonl.tree.load_dag_conversations`),
per-root tree expansion
(:func:`~aiperf.dataset.graph.adapters.dag_jsonl.tree.expand_trees`), and the
unified-segment-store lowering
(:func:`~aiperf.dataset.graph.adapters.dag_jsonl.lowering.lower_dag_trees`) --
threading its run-derived knobs through. :class:`DagJsonlGraphAdapter` implements
:class:`~aiperf.dataset.graph.adapters.protocols.GraphAdapterProtocol` and is
registered under ``graph_adapter.dag_jsonl`` (``detection_priority: 90``).

Explicit-only: excluded from workload auto-detection (see
``workload_detect._AUTODETECT_EXCLUDED``) so legacy
``--custom-dataset-type dag_jsonl`` runs stay on the linear pipeline. Selected via
``--graph-format dag_jsonl``. ``can_load`` remains a real, tested sniff for a
future autodetect flip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.dataset.graph.adapters.dag_jsonl.lowering import lower_dag_trees
from aiperf.dataset.graph.adapters.dag_jsonl.tree import (
    expand_trees,
    load_dag_conversations,
)
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.parse_context import GraphParseContext

__all__ = ["DagJsonlGraphAdapter", "from_dag_jsonl"]

# Dynamo's ``dynamo.request.trace.v1`` discriminator; a dag sniff rejects any
# line carrying it so the two ``.jsonl`` sniffs stay mutually exclusive.
_DYNAMO_SCHEMA = "dynamo.request.trace.v1"


def from_dag_jsonl(
    path: str,
    *,
    default_model: str | None = None,
    run_streaming: bool = True,
    delay_cap_seconds: float | None = None,
    endpoint_extra: list[tuple[str, Any]] | None = None,
) -> ParsedGraph:
    """Load, expand, and lower a ``dag_jsonl`` file into a :class:`ParsedGraph`.

    Composes the three ``dag_jsonl`` layers with no state of its own: the legacy
    loader (delay-cap clamp) -> per-root instanced trees -> unified interned
    segment store. Every knob is a pure function of its arguments, so build-plane
    and schedule-plane parses called with identical arguments stamp
    byte-identical graphs.

    Args:
        path: Path to a ``dag_jsonl`` file.
        default_model: Fallback stamped onto every node's native ``model`` field
            when a turn has no authored ``model`` (the worker's own dispatch
            fallback -- ``ModelEndpointInfo.from_run(run).primary_model_name``).
        run_streaming: Resolved endpoint ``stream`` flag, stamped onto every
            node's ``streaming`` / body ``stream``.
        delay_cap_seconds: Per-turn inter-turn delay cap (seconds); the same
            value the legacy loader clamps authored delays with.
        endpoint_extra: The run's ``--extra-inputs`` pairs
            (``ModelEndpointInfo.from_run(run).endpoint.extra``), folded into
            every node's ``extra_body`` at the legacy
            precedence (turn ``extra`` wins on overlap).
    """
    conversations = load_dag_conversations(
        Path(path), delay_cap_seconds=delay_cap_seconds
    )
    trees = expand_trees(conversations)
    return lower_dag_trees(
        trees,
        default_model=default_model,
        run_streaming=run_streaming,
        endpoint_extra=endpoint_extra,
    )


class DagJsonlGraphAdapter:
    """DAG-conversation-JSONL graph adapter (``graph_adapter.dag_jsonl``)."""

    @classmethod
    def can_load(cls, path: Path) -> bool:
        """True iff ``path`` is a ``.jsonl`` whose first non-blank line is a dag line.

        A dag line is a JSON object carrying both ``session_id`` and ``turns``
        and WITHOUT the dynamo schema discriminator (dynamo's sink envelope
        unwrapped first) -- keeping this sniff mutually exclusive with
        ``DynamoTraceAdapter.can_load``. Bounded to the first non-blank line;
        never raises (any error -> ``False``).
        """
        try:
            if not path.is_file() or path.suffix.lower() != ".jsonl":
                return False
            with path.open("rb") as f:
                for raw in f:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    rec = orjson.loads(stripped)
                    return (
                        isinstance(rec, dict)
                        and "session_id" in rec
                        and "turns" in rec
                        and not _is_dynamo_record(rec)
                    )
        except (OSError, orjson.JSONDecodeError):
            return False
        return False

    @classmethod
    def parse(cls, path: Path, ctx: GraphParseContext | None = None) -> ParsedGraph:
        """Convert ``path`` into a :class:`ParsedGraph` via :func:`from_dag_jsonl`.

        ``ctx`` carries the four run-derived dispatch knobs (``default_model``
        / ``run_streaming`` / ``delay_cap_seconds`` / ``endpoint_extra``), each
        forwarded ONLY when set so a ctx-less parse (CLI tooling / direct
        callers) keeps the :func:`from_dag_jsonl` defaults — in particular a
        partial ctx never clobbers ``run_streaming=True`` with ``None``.

        Wraps the entry call in :func:`_assert_dag_zero_arrival_offsets`:
        production always dispatches through this seam (the registry-driven
        ``parse_graph``), so the all-zero arrival-offset invariant the
        t*/dynamic-slot gate's carve-out relies on is enforced on every parse.
        """
        kwargs: dict[str, Any] = {}
        if ctx is not None:
            if ctx.default_model is not None:
                kwargs["default_model"] = ctx.default_model
            if ctx.run_streaming is not None:
                kwargs["run_streaming"] = ctx.run_streaming
            if ctx.delay_cap_seconds is not None:
                kwargs["delay_cap_seconds"] = ctx.delay_cap_seconds
            if ctx.endpoint_extra is not None:
                kwargs["endpoint_extra"] = ctx.endpoint_extra
        parsed = from_dag_jsonl(str(path), **kwargs)
        _assert_dag_zero_arrival_offsets(parsed)
        return parsed


def _assert_dag_zero_arrival_offsets(parsed: ParsedGraph) -> None:
    """Hold the dag_jsonl invariant that every node has a zero arrival offset.

    The t*/dynamic-slot gate (``workload_detect._gate_dynamic_slots_vs_tstar``)
    carves out graphs whose EVERY node carries an explicit-zero
    ``arrival_offset_us`` -- the shape dag lowering stamps -- because all-zero
    offsets degenerate the t* snapshot chop to a no-op (see the carve-out
    comment there for the full argument). This guard wraps ``from_dag_jsonl``
    inside :meth:`DagJsonlGraphAdapter.parse` (production always dispatches
    through this seam post-ladder), walking the parsed graph once (O(nodes))
    and raising if the lowering ever stamps a nonzero offset -- that would
    engage t* against a graph the lowering did not time, silently
    mis-partitioning nodes into warmup. If dag ever emits recorded offsets
    intentionally, delete this guard and let ``_gate_dynamic_slots_vs_tstar``
    gate it like any recorded workload (the carve-out stops matching on its
    own).
    """
    for record in (parsed.graph, *parsed.graphs.values()):
        for node_id, node in record.nodes.items():
            if node.arrival_offset_us:
                raise ValueError(
                    f"dag_jsonl node '{node_id}' has arrival_offset_us="
                    f"{node.arrival_offset_us}, but dag_jsonl lowering must "
                    "stamp a zero arrival offset on every node -- the "
                    "t*/dynamic-slot gate's explicit-zero carve-out is "
                    "justified by that invariant."
                )


def _is_dynamo_record(rec: dict) -> bool:
    """True iff ``rec`` (dynamo sink envelope unwrapped) is a dynamo trace record."""
    from aiperf.dataset.graph.adapters.dynamo.trace_reader import unwrap_sink_envelope

    return unwrap_sink_envelope(rec).get("schema") == _DYNAMO_SCHEMA
