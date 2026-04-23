# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark completion signaling for Kubernetes mode.

When running in K8s, the controller pod patches an annotation on its parent
AIPerfJob CR to signal that the benchmark has finished and results are ready
for collection.  The operator watches for this annotation via a kopf field
handler and reacts immediately instead of waiting for the next poll cycle.

Environment variables (set automatically by JobSet manifest):
    AIPERF_JOB_ID   - AIPerfJob CR name (= job_id)
    AIPERF_NAMESPACE - Namespace containing the CR
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.constants import Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_GROUP,
    AIPERF_PLURAL,
    AIPERF_VERSION,
)

logger = logging.getLogger(__name__)


async def signal_benchmark_complete() -> bool:
    """Patch the AIPerfJob CR annotation to signal benchmark completion.

    Called by the controller pod after the benchmark finishes and results
    are exported.  The operator's ``on_benchmark_complete`` handler picks
    this up within seconds via kopf's watch mechanism.

    Returns:
        True if the annotation was patched successfully.
    """
    job_id = os.environ.get("AIPERF_JOB_ID")
    namespace = os.environ.get("AIPERF_NAMESPACE")
    if not job_id or not namespace:
        logger.debug("Not in K8s mode (AIPERF_JOB_ID/AIPERF_NAMESPACE not set)")
        return False

    try:
        from kubernetes_asyncio import client

        from aiperf.kubernetes.client import k8s_client

        patch_body: dict[str, Any] = {
            "metadata": {
                "annotations": {
                    Annotations.BENCHMARK_COMPLETE: "true",
                }
            }
        }

        async with k8s_client() as api:
            await client.CustomObjectsApi(api).patch_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                plural=AIPERF_PLURAL,
                namespace=namespace,
                name=job_id,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )

        logger.info(f"Signaled benchmark completion on AIPerfJob {namespace}/{job_id}")
        return True

    except (ApiException, aiohttp.ClientError, OSError) as e:
        logger.warning(f"Failed to signal benchmark completion: {e}")
        return False
