# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin that an orchestrator conversation stays in the sampled root pool.

An ``orchestrator: true`` conversation issues a request-less (virtual)
credit whose sole job is to re-fire its conversation-level ``spawns`` on
every sampled iteration. For that re-firing to happen, the orchestrator
MUST remain a sampleable root: ``PhaseOrchestrator`` builds its sampler
pool from ``DatasetMetadata.sampleable_conversation_ids`` (conversations
where ``is_root`` is truthy). If an exclusion for
orchestrators were ever added there, the orchestrator would be sampled
zero times and never fire — this test guards against that regression.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.common.models import DatasetMetadata
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "orchestrator.dag.jsonl"
)


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
    ``PhaseOrchestrator`` samples from; the spawn children must NOT (they are
    dispatched by intercept, not sampled as roots)."""
    metadata = DatasetMetadata(
        conversations=[c.metadata() for c in DagJsonlLoader(filename=FIXTURE).load()],
        sampling_strategy=DatasetSamplingStrategy.RANDOM,
    )
    root_ids = metadata.sampleable_conversation_ids
    assert "start" in root_ids
    assert "fan-out-a" not in root_ids
    assert "fan-out-b" not in root_ids
