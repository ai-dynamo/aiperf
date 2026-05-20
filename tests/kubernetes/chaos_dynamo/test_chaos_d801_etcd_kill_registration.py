# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D801 -- etcd kill during decode-worker registration race.

Known flake risk per plan section 4. Timing-sensitive; retry up to 3 times.

Scenario (Wave-0 #3):
    Kill the etcd pod via ``faults.inject("store.etcd.kill")`` while a fresh
    decode worker pod is mid-boot -- the window where ``register_model`` has
    been issued (worker_factory.py:398, 433) and the etcd lease grant
    (runtime/src/transports/etcd/lease.rs:21) is in flight.

Assertion:
    No half-registered state. Either the worker retries to registration
    success within ~90s (lease TTL + timeout) **or** it fails cleanly into a
    ``CrashLoopBackOff`` with a clear error message. A worker stuck in the
    router roster while not actually serving requests is a FAIL.
"""

from __future__ import annotations

import pytest

from aiperf.common.aiperf_logger import AIPerfLogger

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

MAX_RETRIES = 3
"""Per plan section 4: accept first PASS, or 2-of-3 if any flake."""


async def test_d801_etcd_kill_during_registration_race(
    faults,  # noqa: ANN001 - InjectorRegistry, see conftest.py
    kubectl,  # noqa: ANN001 - KubectlClient, see tests.kubernetes.conftest
    dynamo_endpoint_url,  # noqa: ANN001 - str, see gpu.dynamo.conftest
) -> None:
    """Kill etcd during fresh worker boot; assert no half-registered state.

    Retries up to :py:data:`MAX_RETRIES` times because the "fresh worker
    mid-boot" window is non-deterministic. Accepts 2-of-3 success per plan
    section 4 risk note.

    Implementation outline (pending real-cluster validation + dgd-scale
    fixture):

    1. For each attempt 1..MAX_RETRIES:
        a. Scale decode workers from N to N+1::

            kubectl scale dgd <name> -n <ns> --replicas N+1

           (or ``kubectl patch components/[name=decode]/replicas``)
        b. Watch for the new worker pod to be Pending and started
           (``containerStatus`` running but not yet ready)::

            kubectl get pod -l nvidia.com/dynamo-sub-component-type=decode \\
                -n dynamo-server -o json

           Look for the newest pod with ``phase=Running, ready=False``.
        c. Once observed (or 30s elapsed waiting), inject the etcd kill::

            async with faults.inject("store.etcd.kill", grace_period=0):
                pass

        d. Wait 90s (lease TTL + timeout).
        e. Read final state:
            - All decode pods ``Ready``?
            - Or some in ``CrashLoopBackOff`` with clear ``"etcd"`` /
              ``"register"`` error?
            - Frontend routes a test request successfully?
        f. Both conditions are valid PASS states. "Worker in Ready state
           but not actually serving" is FAIL.
    2. Accept first PASS, or 2-of-3 if any flake.
    3. Cleanup: scale back to N.
    """
    pytest.skip(
        "scaffold landed; assertion-body pending real-cluster validation + dgd-scale fixture"
    )
