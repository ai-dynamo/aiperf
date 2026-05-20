# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D317 -- pause a KVBM ZMQ publisher while workers keep generating.

This scenario is only meaningful for a topology that actually enables KVBM and
exposes the KVBM ZMQ publisher as an addressable in-pod process. The stock
Dynamo v1.1.0 fixture can enable ``--connector kvbm`` for prefill workers, but
publisher/subscriber/consolidator internals are normally threads inside the
worker process rather than separate PIDs. In that topology this test skips with
the precise missing prerequisite instead of sending SIGSTOP to the whole worker
and pretending the KVBM publisher was isolated.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)

_PREFILL_SELECTOR = "nvidia.com/dynamo-sub-component-type=prefill"
_KVBM_ENV = "DYN_KVBM_CPU_CACHE_GB"
_REQUEST_TIMEOUT_S = 45.0


@dataclass(frozen=True, slots=True)
class KVBMPodTarget:
    """Address of a KVBM-enabled container in a Dynamo pod."""

    namespace: str
    pod: str
    container: str
    env: dict[str, str]


@dataclass(frozen=True, slots=True)
class KVBMProcessTarget:
    """Address of a KVBM helper process inside a Dynamo pod."""

    pod_target: KVBMPodTarget
    pid: int
    command: str


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """HTTP response from one Dynamo chat completion request."""

    status: int
    body: str
    json_body: dict[str, Any] | None


async def test_d317_kvbm_zmq_publisher_pause_keeps_generation_bounded(
    request: pytest.FixtureRequest,
) -> None:
    """Pause an isolated KVBM ZMQ publisher and verify frontend requests finish."""
    kubectl: KubectlClient = request.getfixturevalue("kubectl")
    namespace: str = request.getfixturevalue("dynamo_deployment_namespace")
    endpoint_url: str = request.getfixturevalue("dynamo_endpoint_url")
    faults: InjectorRegistry = request.getfixturevalue("faults")

    kvbm_pod = await discover_kvbm_prefill_target(kubectl, namespace, "D317")
    publisher = await discover_isolated_kvbm_process(
        kubectl,
        kvbm_pod,
        role="publisher",
        role_patterns=("publisher", "pub", "zmq.*pub"),
        scenario_id="D317",
    )

    baseline = await post_completion(
        endpoint_url, content="D317 baseline.", max_tokens=4
    )
    assert_successful_completion("D317 baseline", baseline)

    async with faults.inject(
        "process.signal",
        target={
            "kind": "pod",
            "ns": publisher.pod_target.namespace,
            "pod": publisher.pod_target.pod,
            "container": publisher.pod_target.container,
            "pid": publisher.pid,
        },
        signal="SIGSTOP",
    ) as applied:
        assert applied.metadata.get("pid") == publisher.pid
        logger.info(
            lambda: (
                "D317: paused KVBM ZMQ publisher "
                f"pid={publisher.pid} command={publisher.command!r}"
            )
        )
        paused = await post_completion(
            endpoint_url,
            content="D317 completion while KVBM ZMQ publisher is paused.",
            max_tokens=8,
        )
        assert_successful_completion("D317 paused publisher", paused)


async def discover_kvbm_prefill_target(
    kubectl: KubectlClient,
    namespace: str,
    scenario_id: str,
) -> KVBMPodTarget:
    """Return one ready KVBM-enabled prefill container or skip precisely."""
    pods = await _list_pods_json(kubectl, namespace, _PREFILL_SELECTOR)
    if not pods:
        pytest.skip(
            f"{scenario_id}: requires a disaggregated prefill pod labelled "
            f"{_PREFILL_SELECTOR!r} in namespace {namespace!r}; none found"
        )

    observed_prefill: list[str] = []
    for pod in pods:
        pod_name = pod.get("metadata", {}).get("name", "<unknown>")
        if not _pod_ready(pod):
            observed_prefill.append(f"{pod_name}:not-ready")
            continue
        for container in pod.get("spec", {}).get("containers", []):
            env = _container_env(container)
            args = " ".join(str(arg) for arg in container.get("args", []))
            if _KVBM_ENV in env or "--connector kvbm" in args:
                name = container.get("name")
                if isinstance(name, str):
                    return KVBMPodTarget(namespace, pod_name, name, env)
        observed_prefill.append(f"{pod_name}:no-{_KVBM_ENV}")

    pytest.skip(
        f"{scenario_id}: requires a ready prefill container with KVBM enabled "
        f"({_KVBM_ENV} env or '--connector kvbm' arg); observed {observed_prefill!r}"
    )


async def discover_isolated_kvbm_process(
    kubectl: KubectlClient,
    target: KVBMPodTarget,
    *,
    role: str,
    role_patterns: tuple[str, ...],
    scenario_id: str,
) -> KVBMProcessTarget:
    """Return an isolated KVBM helper process matching ``role_patterns``."""
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
        "ps -eo pid=,args=",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"{scenario_id}: cannot inspect processes in "
            f"{target.namespace}/{target.pod}/{target.container}; "
            f"kubectl exec returned {result.returncode}: {result.stderr.strip()!r}"
        )

    candidates = _parse_processes(result.stdout)
    role_matches = [
        proc
        for proc in candidates
        if "kvbm" in proc.command.lower()
        and any(
            _contains_pattern(proc.command.lower(), pattern)
            for pattern in role_patterns
        )
    ]
    isolated = [
        proc for proc in role_matches if _looks_like_helper_process(proc.command)
    ]
    if isolated:
        proc = isolated[0]
        return KVBMProcessTarget(target, proc.pid, proc.command)

    kvbm_seen = [proc.command for proc in candidates if "kvbm" in proc.command.lower()]
    pytest.skip(
        f"{scenario_id}: requires an isolated KVBM ZMQ {role} process inside "
        f"{target.namespace}/{target.pod}/{target.container}; found KVBM entries "
        f"{kvbm_seen[:5]!r}. If {role} is a thread inside the vLLM worker, add a "
        "test hook/sidecar exposing it as a separate PID before running this fault."
    )


async def post_completion(
    dynamo_endpoint_url: str,
    *,
    content: str,
    max_tokens: int,
) -> CompletionResult:
    """POST one non-streaming chat completion to the Dynamo frontend."""
    payload: dict[str, object] = {
        "model": "default",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(_chat_completion_url(dynamo_endpoint_url), json=payload) as resp,
    ):
        body = await resp.text()
        parsed_json: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = await resp.json(content_type=None)
            if isinstance(parsed, dict):
                parsed_json = parsed
        return CompletionResult(resp.status, body, parsed_json)


def assert_successful_completion(label: str, result: CompletionResult) -> None:
    """Assert the Dynamo frontend returned a successful completion."""
    assert result.status == 200, (
        f"{label}: expected HTTP 200 from Dynamo frontend, got {result.status}; "
        f"body={result.body[:512]!r}"
    )
    if result.json_body is not None:
        choices = result.json_body.get("choices")
        assert choices, (
            f"{label}: response JSON contains no choices: {result.json_body!r}"
        )


async def wait_for_pod_ready(
    kubectl: KubectlClient,
    namespace: str,
    pod_name: str,
    *,
    timeout_s: float = 120.0,
) -> None:
    """Wait until ``pod_name`` reports Ready=True."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        result = await kubectl.run(
            "get",
            "pod",
            pod_name,
            "-n",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode == 0:
            pod = orjson.loads(result.stdout)
            if _pod_ready(pod):
                return
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"pod {namespace}/{pod_name} did not become Ready within {timeout_s}s"
            )
        await asyncio.sleep(2.0)


@dataclass(frozen=True, slots=True)
class _ProcessRow:
    pid: int
    command: str


async def _list_pods_json(
    kubectl: KubectlClient,
    namespace: str,
    selector: str,
) -> list[dict[str, Any]]:
    result = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        selector,
        "-o",
        "json",
        check=False,
    )
    if result.returncode != 0:
        return []
    payload = orjson.loads(result.stdout)
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)]


def _container_env(container: dict[str, Any]) -> dict[str, str]:
    envs: dict[str, str] = {}
    for env in container.get("env", []):
        name = env.get("name")
        value = env.get("value")
        if isinstance(name, str) and isinstance(value, str):
            envs[name] = value
    return envs


def _pod_ready(pod: dict[str, Any]) -> bool:
    conditions = pod.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )


def _parse_processes(stdout: str) -> list[_ProcessRow]:
    rows: list[_ProcessRow] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        with contextlib.suppress(ValueError):
            rows.append(_ProcessRow(pid=int(pid_text), command=command.strip()))
    return rows


def _contains_pattern(value: str, pattern: str) -> bool:
    parts = [part for part in pattern.split(".*") if part]
    cursor = 0
    for part in parts:
        found = value.find(part, cursor)
        if found == -1:
            return False
        cursor = found + len(part)
    return True


def _looks_like_helper_process(command: str) -> bool:
    lowered = command.lower()
    worker_markers = ("vllm", "sglang", "trtllm", "python", "api_server")
    return not any(marker in lowered for marker in worker_markers)


def _chat_completion_url(dynamo_endpoint_url: str) -> str:
    return dynamo_endpoint_url.rstrip("/") + "/chat/completions"
