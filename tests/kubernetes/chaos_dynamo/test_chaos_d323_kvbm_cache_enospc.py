# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D323 -- KVBM disk spill path hits ENOSPC."""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos_dynamo.test_chaos_d317_kvbm_zmq_publisher_pause import (
    assert_successful_completion,
    discover_kvbm_prefill_target,
    post_completion,
)
from tests.kubernetes.chaos_dynamo.test_chaos_d321_kvbm_duplicate_free import (
    discover_kvbm_chaos_hook,
    run_kvbm_chaos_hook,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_SPILL_DIR_ENV_NAMES = (
    "DYN_KVBM_SPILL_DIR",
    "DYN_KVBM_CACHE_DIR",
    "AIPERF_DYNAMO_KVBM_ENOSPC_DIR",
)


async def test_d323_kvbm_disk_spill_enospc_does_not_crash_worker(
    request: pytest.FixtureRequest,
) -> None:
    """Trigger ENOSPC through an explicit KVBM hook and verify recovery."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D323")
    spill_dir = _spill_dir(kvbm_pod.env)
    if spill_dir is None:
        pytest.skip(
            "D323 requires a bounded KVBM disk-spill/cache directory to fill via "
            f"one of {_SPILL_DIR_ENV_NAMES!r}; observed env keys "
            f"{sorted(kvbm_pod.env)!r}. Refusing to fill arbitrary container filesystems."
        )
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D323")

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        f"enospc --dir {spill_dir} --duration-seconds 10",
        "D323",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D323 recovery after KVBM disk-spill ENOSPC.",
        max_tokens=8,
    )
    assert_successful_completion("D323 recovery", recovery)


def _spill_dir(env: dict[str, str]) -> str | None:
    for name in _SPILL_DIR_ENV_NAMES:
        value = env.get(name)
        if value:
            return value
    return None
