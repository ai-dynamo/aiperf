# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures + helpers for tests/unit/search_recipes/.

Recipes on this branch consume a validated ``BenchmarkConfig`` via
``SearchRecipeContext.benchmark_config`` (not a CLI ``CLIConfig`` view —
that's the ajc/k8s design). The ``ctx_factory`` helper here builds a
minimal valid BenchmarkConfig, lets callers toggle ``streaming`` on the
endpoint, and stuffs ``sla_targets`` / ``sweep_overrides`` per recipe.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.config.config import BenchmarkConfig
from aiperf.search_recipes._base import SearchRecipeContext

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "streaming": True,
    },
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
        }
    ],
}


def make_ctx(
    *,
    streaming: bool = True,
    sla_targets: dict[str, float] | None = None,
    benchmark_overrides: dict[str, Any] | None = None,
    **sweep_overrides: Any,
) -> SearchRecipeContext:
    """Build a SearchRecipeContext for recipe ``expand()`` tests.

    Args:
        streaming: Sets ``endpoint.streaming`` on the BenchmarkConfig.
        sla_targets: Recipe-specific SLA target dict (e.g.
            ``{"ttft_sla_ms": 200}``); becomes ``ctx.sla_targets``.
        benchmark_overrides: Optional deep-merge into the minimal config
            kwargs (e.g. ``{"endpoint": {"type": "embeddings"}}``) for
            recipes that branch on endpoint type.
        **sweep_overrides: Any remaining kwargs land in
            ``ctx.sweep_overrides`` (recipe-specific dimension caps,
            step counts, threshold overrides, etc.).
    """
    kwargs = {
        k: dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v
        for k, v in _MINIMAL_CONFIG_KWARGS.items()
    }
    kwargs["endpoint"] = {**kwargs["endpoint"], "streaming": streaming}
    if benchmark_overrides:
        for k, v in benchmark_overrides.items():
            if isinstance(v, dict) and isinstance(kwargs.get(k), dict):
                kwargs[k] = {**kwargs[k], **v}
            else:
                kwargs[k] = v
    bc = BenchmarkConfig(**kwargs)
    return SearchRecipeContext(
        benchmark_config=bc,
        sla_targets=sla_targets or {},
        sweep_overrides=sweep_overrides,
    )


@pytest.fixture
def ctx_factory():
    """Fixture-wrapped ``make_ctx`` for tests that prefer fixture style."""
    return make_ctx
