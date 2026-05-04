# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatibility regression: representative origin/main CLI invocations
produce the expected AIPerfConfig shape after v1->v2 conversion.

These tests guard against silent drift in ``convert_user_to_aiperf`` as v1 grows
(or AIPerfConfig schema changes). Each parametrize case mirrors a real-world
invocation pattern from ``origin/main``'s tutorials/README — concurrency,
request_rate (Poisson + gamma), user_centric, public_dataset, warmup, fixed
schedule, and synthetic-with-ISL.

We construct ``UserConfig`` via ``model_validate`` rather than driving cyclopts
end-to-end. This focuses the regression on the **converter** (the highest-risk
v1->v2 surface), not on cyclopts CLI parsing, which is exercised separately by
the cli_commands integration tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf


def _resolve(obj: Any, path: str) -> Any:
    """Resolve a dotted path with ``[N]`` indexing into the AIPerfConfig tree."""
    cur = obj
    for part in path.split("."):
        if "[" in part:
            name, idx = part.rstrip("]").split("[")
            cur = getattr(cur, name)[int(idx)]
        else:
            cur = getattr(cur, part)
    return cur


def _matches(actual: Any, expected: Any) -> bool:
    """Lenient equality: accept enum -> str suffix matches and float coercion.

    Phase types are str-based enums; comparing ``PhaseType.CONCURRENCY`` to the
    literal ``"concurrency"`` should pass. Likewise ``DatasetType.PUBLIC`` vs
    ``"public"``. We don't want the regression suite to fail on stringification
    cosmetics — only on actual structural drift.
    """
    if actual == expected:
        return True
    if isinstance(expected, str):
        return str(actual).lower().endswith(expected.lower())
    return False


@pytest.mark.parametrize(
    "user_dict,assertions",
    [
        param(
            {
                "endpoint": {
                    "model_names": ["llama"],
                    "urls": ["http://localhost:8000"],
                },
                "loadgen": {"concurrency": 100, "request_count": 1000},
            },
            {
                "phases[0].type": "concurrency",
                "phases[0].concurrency": 100,
                "phases[0].requests": 1000,
                "endpoint.urls": ["http://localhost:8000"],
                "phases[0].name": "profiling",
            },
            id="concurrency-basic",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {
                    "request_rate": 50.0,
                    "benchmark_duration": 30.0,
                },
            },
            {
                "phases[0].type": "poisson",
                "phases[0].duration": 30.0,
                "phases[0].rate": 50.0,
            },
            id="request-rate-poisson",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {
                    "request_rate": 50.0,
                    "arrival_pattern": "gamma",
                    "arrival_smoothness": 2.5,
                    "request_count": 1000,
                },
            },
            {
                "phases[0].type": "gamma",
                "phases[0].smoothness": 2.5,
                "phases[0].rate": 50.0,
            },
            id="request-rate-gamma",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {
                    "user_centric_rate": 5.0,
                    "num_users": 50,
                    "request_count": 100,
                },
                "input": {"conversation": {"turn": {"mean": 3}}},
            },
            {
                "phases[0].type": "user_centric",
                "phases[0].users": 50,
                "phases[0].rate": 5.0,
            },
            id="user-centric",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {"concurrency": 1, "request_count": 1},
                "input": {"public_dataset": "sharegpt"},
            },
            {
                "datasets[0].name": "main",
                "datasets[0].dataset": "sharegpt",
                "datasets[0].type": "public",
            },
            id="public-dataset-sharegpt",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {
                    "concurrency": 10,
                    "request_count": 100,
                    "warmup_concurrency": 2,
                    "warmup_request_count": 10,
                },
            },
            {
                "phases[0].name": "warmup",
                "phases[0].exclude_from_results": True,
                "phases[1].name": "profiling",
                "phases[1].concurrency": 10,
            },
            id="warmup-then-profiling",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {"concurrency": 10, "request_count": 100},
                "input": {"fixed_schedule": True, "fixed_schedule_auto_offset": True},
            },
            {
                "phases[0].type": "fixed_schedule",
                "phases[0].auto_offset": True,
            },
            id="fixed-schedule",
        ),
        param(
            {
                "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
                "loadgen": {"concurrency": 1, "request_count": 1},
                "input": {"prompt": {"input_tokens": {"mean": 128, "stddev": 16}}},
            },
            {
                "datasets[0].type": "synthetic",
                "datasets[0].name": "main",
            },
            id="synthetic-with-isl",
        ),
    ],
)  # fmt: skip
def test_v1_invocation_produces_expected_aiperf_config(
    user_dict: dict[str, Any], assertions: dict[str, Any]
) -> None:
    user = UserConfig.model_validate(user_dict)
    service = ServiceConfig()
    cfg = convert_user_to_aiperf(user, service)

    for path, expected in assertions.items():
        actual = _resolve(cfg.benchmark, path)
        assert _matches(actual, expected), (
            f"At {path}: expected {expected!r}, got {actual!r}"
        )
