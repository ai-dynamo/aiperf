# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D319 -- KVBM ZMQ high-water-mark overflow under bursty short requests."""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_dynamo.test_chaos_d317_kvbm_zmq_publisher_pause import (
    CompletionResult,
    assert_successful_completion,
    discover_kvbm_prefill_target,
    post_completion,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_HWM_ENV_NAMES = (
    "DYN_KVBM_ZMQ_SNDHWM",
    "DYN_KVBM_ZMQ_RCVHWM",
    "DYN_KVBM_ZMQ_HWM",
    "ZMQ_SNDHWM",
    "ZMQ_RCVHWM",
)
_MAX_MEANINGFUL_HWM = 32
_BURST_REQUESTS = 48
_MIN_SUCCESS_RATE = 0.80


async def test_d319_kvbm_zmq_hwm_overflow_burst_is_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Run only when the topology sets a deliberately low KVBM ZMQ HWM."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D319")
    hwm = _configured_hwm(kvbm_pod.env)
    if hwm is None:
        pytest.skip(
            "D319 requires a topology/test hook that sets a low KVBM ZMQ high-water "
            f"mark via one of {_HWM_ENV_NAMES!r}; observed env keys "
            f"{sorted(kvbm_pod.env)!r}"
        )
    if hwm > _MAX_MEANINGFUL_HWM:
        pytest.skip(
            f"D319 requires HWM <= {_MAX_MEANINGFUL_HWM} to force overflow; "
            f"observed configured HWM={hwm} on "
            f"{kvbm_pod.namespace}/{kvbm_pod.pod}/{kvbm_pod.container}"
        )

    results = await asyncio.gather(
        *(
            post_completion(
                endpoint_url,
                content=f"D319 burst request {idx} with low KVBM ZMQ HWM.",
                max_tokens=4,
            )
            for idx in range(_BURST_REQUESTS)
        ),
        return_exceptions=True,
    )
    successes = sum(1 for result in results if _is_success(result))
    failures = [result for result in results if not _is_success(result)]
    success_rate = successes / _BURST_REQUESTS
    assert success_rate >= _MIN_SUCCESS_RATE, (
        f"D319: low-HWM burst success rate {success_rate:.1%} below "
        f"{_MIN_SUCCESS_RATE:.0%}; successes={successes}, "
        f"failures_sample={failures[:3]!r}"
    )
    logger.info(
        lambda: (
            f"D319: low-HWM burst completed with HWM={hwm}, "
            f"successes={successes}/{_BURST_REQUESTS}"
        )
    )

    recovery = await post_completion(
        endpoint_url,
        content="D319 recovery request after low-HWM burst.",
        max_tokens=4,
    )
    assert_successful_completion("D319 recovery", recovery)


def _configured_hwm(env: dict[str, str]) -> int | None:
    for name in _HWM_ENV_NAMES:
        value = env.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            pytest.skip(f"D319: {name} must be an integer HWM, got {value!r}")
    return None


def _is_success(result: CompletionResult | BaseException) -> bool:
    return isinstance(result, CompletionResult) and result.status == 200
