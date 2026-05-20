# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D805 -- DynamoWorkerMetadata watch RBAC revocation preserves live traffic.

Scenario (D-series catalog, section D8xx):

* Inspect Kubernetes RBAC for an exact Role/ClusterRole rule granting ``watch``
  on ``dynamoworkermetadatas.nvidia.com``.
* If exactly one owning RBAC resource is found, remove only that ``watch`` verb.
* While the Kubernetes-discovery watcher is unable to refresh, assert the
  already-started frontend continues to serve requests from its cached worker
  snapshot.
* Restore the removed verb in ``finally`` via the unified chaos RBAC injector.

The test self-skips when the exact RBAC owner cannot be identified safely. It
never revokes wildcard rules or guesses among multiple candidates because a bad
RBAC patch can break unrelated tests in the shared cluster.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_API_GROUP = "nvidia.com"
_RESOURCE = "dynamoworkermetadatas"
_VERB = "watch"
_REQUEST_TIMEOUT_S = 30.0
_WATCH_FAILURE_WINDOW_S = 10.0


@dataclass(frozen=True, slots=True)
class _RbacWatchOwner:
    """Exact RBAC resource that grants DynamoWorkerMetadata watch permission."""

    scope: str
    """``role`` or ``clusterrole`` for kubectl/fault-injector patching."""

    name: str
    """Role or ClusterRole name."""

    namespace: str | None
    """Role namespace, or ``None`` for ClusterRole."""

    @property
    def label(self) -> str:
        """Human-readable identifier for skip/failure diagnostics."""
        if self.namespace is None:
            return f"clusterrole/{self.name}"
        return f"role/{self.namespace}/{self.name}"


async def test_d805_discovery_rbac_watch_revocation_preserves_existing_traffic(
    request: pytest.FixtureRequest,
    kubectl: KubectlClient,
    dynamo_deployment_namespace: str,
    dynamo_endpoint_url: str,
) -> None:
    """Revoke DWM watch RBAC only when the exact owner is uniquely discoverable.

    Kubernetes discovery should tolerate a short watch outage by serving from the
    already-populated cache. The test proves that contract with one successful
    request before revocation and one while the ``watch`` verb is absent.
    """
    owner, inspected_names = await _find_unique_dwm_watch_owner(
        kubectl, dynamo_deployment_namespace
    )
    if owner is None:
        pytest.skip(
            "D805 requires exactly one exact RBAC rule granting watch on "
            f"{_RESOURCE}.{_API_GROUP}; inspected RBAC resources: "
            f"{', '.join(inspected_names) or '<none>'}"
        )

    await _assert_frontend_serves(dynamo_endpoint_url, phase="before RBAC revocation")

    faults = request.getfixturevalue("faults")
    target: dict[str, str] = {"scope": owner.scope, "name": owner.name}
    if owner.namespace is not None:
        target["ns"] = owner.namespace

    try:
        async with faults.inject(
            "cluster.rbac.revoke",
            target=target,
            api_group=_API_GROUP,
            resource=_RESOURCE,
            verb=_VERB,
        ) as applied:
            assert applied.metadata["name"] == owner.name
            assert applied.metadata["resource"] == _RESOURCE
            assert applied.metadata["verb"] == _VERB
            logger.info(
                f"D805: revoked {_VERB!r} on {_RESOURCE}.{_API_GROUP} from "
                f"{owner.label}; asserting live traffic during watch failure"
            )
            await asyncio.sleep(_WATCH_FAILURE_WINDOW_S)
            await _assert_frontend_serves(
                dynamo_endpoint_url,
                phase=f"while {owner.label} lacks {_VERB!r}",
            )
    finally:
        restored = await _role_currently_grants_watch(kubectl, owner)
        assert restored, (
            f"D805: RBAC restore did not put {_VERB!r} back on {owner.label} "
            f"for {_RESOURCE}.{_API_GROUP}; manual cluster repair required"
        )


async def _find_unique_dwm_watch_owner(
    kubectl: KubectlClient,
    namespace: str,
) -> tuple[_RbacWatchOwner | None, list[str]]:
    """Return the unique exact RBAC owner, or ``None`` with inspected names.

    Wildcard resources / verbs are intentionally ignored. D805 is only safe to
    run when the RBAC rule explicitly names ``dynamoworkermetadatas`` and
    ``watch`` so the injected patch has a narrow blast radius.
    """
    roles = await _load_rbac_collection(kubectl, "roles", namespace=namespace)
    clusterroles = await _load_rbac_collection(kubectl, "clusterroles")

    inspected: list[str] = []
    candidates: list[_RbacWatchOwner] = []
    for item in roles:
        metadata = item.get("metadata", {})
        owner = _RbacWatchOwner(
            scope="role",
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
        )
        inspected.append(owner.label)
        if _has_exact_dwm_watch_rule(item.get("rules") or []):
            candidates.append(owner)

    for item in clusterroles:
        metadata = item.get("metadata", {})
        owner = _RbacWatchOwner(
            scope="clusterrole",
            name=str(metadata.get("name", "")),
            namespace=None,
        )
        inspected.append(owner.label)
        if _has_exact_dwm_watch_rule(item.get("rules") or []):
            candidates.append(owner)

    if len(candidates) != 1:
        candidate_names = [candidate.label for candidate in candidates]
        return None, candidate_names or inspected
    return candidates[0], inspected


async def _load_rbac_collection(
    kubectl: KubectlClient,
    resource: str,
    *,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Load Roles or ClusterRoles as JSON; skip if the caller lacks list RBAC."""
    args = ["get", resource, "-o", "json"]
    if namespace is not None:
        args.extend(["-n", namespace])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        pytest.skip(
            f"D805 could not inspect {resource} before RBAC mutation: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    data = orjson.loads(result.stdout or b"{}")
    return list(data.get("items", []))


def _has_exact_dwm_watch_rule(rules: list[dict[str, Any]]) -> bool:
    """Return true for explicit ``watch`` on ``dynamoworkermetadatas`` only."""
    for rule in rules:
        groups = rule.get("apiGroups") or []
        resources = rule.get("resources") or []
        verbs = rule.get("verbs") or []
        if _API_GROUP in groups and _RESOURCE in resources and _VERB in verbs:
            return True
    return False


async def _role_currently_grants_watch(
    kubectl: KubectlClient,
    owner: _RbacWatchOwner,
) -> bool:
    """Verify cleanup restored the exact watch permission that D805 removed."""
    args = ["get", owner.scope, owner.name]
    if owner.namespace is not None:
        args.extend(["-n", owner.namespace])
    args.extend(["-o", "json"])
    result = await kubectl.run(*args, check=False)
    if result.returncode != 0:
        return False
    body = orjson.loads(result.stdout or b"{}")
    return _has_exact_dwm_watch_rule(body.get("rules") or [])


async def _assert_frontend_serves(endpoint_url: str, *, phase: str) -> None:
    """Send one OpenAI-compatible streaming request and require HTTP success."""
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 10,
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(f"{endpoint_url}/chat/completions", json=payload) as resp,
    ):
        body_prefix = b""
        async for chunk in resp.content.iter_chunked(1024):
            body_prefix += chunk
            if body_prefix:
                break
        assert resp.status == 200, (
            f"D805: frontend returned HTTP {resp.status} {phase}; "
            f"body_prefix={body_prefix[:256].decode(errors='replace')!r}"
        )
        assert body_prefix, f"D805: frontend returned an empty stream {phase}"
