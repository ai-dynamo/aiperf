# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin that an orchestrator conversation stays in the sampled root pool.

An ``orchestrator: true`` conversation issues a request-less (virtual)
credit whose sole job is to re-fire its conversation-level ``spawns`` on
every sampled iteration. For that re-firing to happen, the orchestrator
MUST remain a sampleable root: ``PhaseOrchestrator`` builds its sampler
pool from conversations where ``is_root`` is truthy (see
``src/aiperf/timing/phase_orchestrator.py``). If an exclusion for
orchestrators were ever added there, the orchestrator would be sampled
zero times and never fire — this test guards against that regression.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "orchestrator.dag.jsonl"
)


def _root_conv_ids(conversations) -> list[str]:
    """Mirror the exact filter ``PhaseOrchestrator`` uses to build its
    sampler pool (``getattr(c, "is_root", True)``)."""
    return [c.conversation_id for c in conversations if getattr(c, "is_root", True)]


def test_orchestrator_conversation_loads_as_root() -> None:
    convs = DagJsonlLoader(filename=FIXTURE).load()
    start = next(c for c in convs if c.session_id == "start")
    assert start.is_orchestrator is True
    assert start.is_root is True, (
        "orchestrator must load as a sampleable root so it re-fires each "
        "sampled iteration"
    )
    # Its single synthesized turn is request-less (no HTTP wire request).
    assert [t.no_request for t in start.turns] == [True]


def test_orchestrator_included_in_sampled_root_pool() -> None:
    """The orchestrator ``start`` must survive the ``is_root`` filter that
    ``PhaseOrchestrator`` applies; the spawn children must NOT (they are
    dispatched by intercept, not sampled as roots)."""
    metadata = [c.metadata() for c in DagJsonlLoader(filename=FIXTURE).load()]
    root_ids = _root_conv_ids(metadata)
    assert "start" in root_ids
    assert "fan-out-a" not in root_ids
    assert "fan-out-b" not in root_ids
