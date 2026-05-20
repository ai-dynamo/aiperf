# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D321 -- inject duplicate KVBM free notifications.

D321 requires a product/test hook because the free notification is an internal
KVBM event, not a Kubernetes object or network flow exposed by the generic chaos
injectors. The test probes for either a configured hook command in the KVBM
container environment or a well-known executable on PATH, and skips with that
concrete prerequisite when absent.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

import pytest

from tests.kubernetes.chaos_dynamo.test_chaos_d317_kvbm_zmq_publisher_pause import (
    KVBMPodTarget,
    assert_successful_completion,
    discover_kvbm_prefill_target,
    post_completion,
)
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]

_HOOK_ENV_NAMES = ("DYN_KVBM_CHAOS_HOOK", "AIPERF_DYNAMO_KVBM_CHAOS_HOOK")
_HOOK_BINARIES = ("dynamo-kvbm-chaos", "kvbm-chaos")


@dataclass(frozen=True, slots=True)
class KVBMChaosHook:
    """Executable chaos hook discovered inside a KVBM container."""

    pod_target: KVBMPodTarget
    command: str


async def test_d321_duplicate_kvbm_free_notification_is_idempotent(
    request: pytest.FixtureRequest,
) -> None:
    """Inject duplicate free events through the KVBM chaos hook."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D321")
    hook = await discover_kvbm_chaos_hook(kubectl, kvbm_pod, "D321")

    baseline = await post_completion(
        endpoint_url,
        content="D321 baseline request before duplicate free injection.",
        max_tokens=8,
    )
    assert_successful_completion("D321 baseline", baseline)

    await run_kvbm_chaos_hook(
        kubectl,
        hook,
        "duplicate-free --request-id d321-synthetic-free --count 2",
        "D321",
    )

    recovery = await post_completion(
        endpoint_url,
        content="D321 recovery after duplicate KVBM free notification.",
        max_tokens=8,
    )
    assert_successful_completion("D321 recovery", recovery)


async def discover_kvbm_chaos_hook(
    kubectl: KubectlClient,
    target: KVBMPodTarget,
    scenario_id: str,
) -> KVBMChaosHook:
    """Return an executable KVBM chaos hook or skip with the missing surface."""
    for env_name in _HOOK_ENV_NAMES:
        command = target.env.get(env_name)
        if command:
            return KVBMChaosHook(target, command)

    command_probe = " || ".join(f"command -v {name}" for name in _HOOK_BINARIES)
    result = await kubectl.run(
        "exec",
        target.pod,
        "-c",
        target.container,
        "-n",
        target.namespace,
        "--",
        "sh",
        "-lc",
        command_probe,
        check=False,
    )
    command = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if result.returncode == 0 and command:
        return KVBMChaosHook(target, command)

    pytest.skip(
        f"{scenario_id}: requires a KVBM event-injection test hook inside "
        f"{target.namespace}/{target.pod}/{target.container}: set one of "
        f"{_HOOK_ENV_NAMES!r} or install one of {_HOOK_BINARIES!r} on PATH. "
        "The stock Dynamo KVBM path does not expose duplicate/reordered Add/Free "
        "events through Kubernetes or Toxiproxy."
    )


async def run_kvbm_chaos_hook(
    kubectl: KubectlClient,
    hook: KVBMChaosHook,
    hook_args: str,
    scenario_id: str,
) -> None:
    """Execute a KVBM chaos hook command and fail with stderr on non-zero exit."""
    target = hook.pod_target
    shell_command = f"{shlex.quote(hook.command)} {hook_args}"
    result = await kubectl.run(
        "exec",
        target.pod,
        "-c",
        target.container,
        "-n",
        target.namespace,
        "--",
        "sh",
        "-lc",
        shell_command,
        check=False,
    )
    assert result.returncode == 0, (
        f"{scenario_id}: KVBM chaos hook failed with exit {result.returncode}; "
        f"command={shell_command!r}, stdout={result.stdout[:512]!r}, "
        f"stderr={result.stderr[:512]!r}"
    )
