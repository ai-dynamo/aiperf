# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding helpers for AgenticReplayStrategy strategy tests."""

from __future__ import annotations

from aiperf.common.enums import CacheBustTarget
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


def _make_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    convs = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=None)
            for _ in range(turns_per_trace)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _make_run(*, target: CacheBustTarget, benchmark_id: str = "bench-fixed"):
    """Build a v2 ``BenchmarkRun`` exposing the values the strategy reads.

    V2 PORT NOTE: agentx read ``user_config.input.prompt.cache_bust.target`` and
    ``user_config.benchmark_id``. The v2 ``AgenticReplayStrategy`` reads
    ``run.cfg.get_cache_bust_target()`` and ``run.benchmark_id`` instead; the
    target lives on the synthetic dataset's ``prompts.cache_bust.target``.
    """
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "type": "completions",
                "urls": ["http://localhost:8000/v1"],
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "prompts": {"cache_bust": {"target": target}},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=benchmark_id,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )
