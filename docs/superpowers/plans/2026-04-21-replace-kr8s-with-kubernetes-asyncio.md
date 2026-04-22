# Replace `kr8s` with `kubernetes_asyncio` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every `kr8s` call site across AIPerf with `kubernetes_asyncio`, dissolve the `AIPerfKubeClient` wrapper class into free functions that call `CoreV1Api` / `CustomObjectsApi` / `AppsV1Api` / `VersionApi` inline, and remove `kr8s` from the project. Primary driver: LLM-native readability.

**Architecture:** Single branch (`worktree-ajc+remove-kr8s`, based on `ajc/k8s`), eight incremental commits. Each commit keeps the full unit-test suite green. The `AIPerfKubeClient` class is retained as a thin delegating facade during Tasks 3–7 so external callers migrate at their own pace; it is deleted in Task 8. Each file, once migrated, is in its final LLM-native state — direct `async with k8s_client() as api: core = CoreV1Api(api); ...` usage, no wrapper classes, `ApiException` + `e.status` error handling, raw-dict + Pydantic for CRDs.

**Tech Stack:** Python 3.10+ (typed async), `kubernetes_asyncio>=32.0,<33.0`, `kopf>=1.42.0` (unchanged), `aiohttp` (transitive via `kubernetes_asyncio`), `pydantic>=2.10.0` (unchanged), `pytest-asyncio` (unchanged), `uv` for deps.

**Reference spec:** `docs/superpowers/specs/2026-04-21-replace-kr8s-with-kubernetes-asyncio-design.md`

---

## Prerequisites

Before starting, confirm:
- Working directory is the worktree at `/home/anthony/nvidia/projects/aiperf/aiperf.git/.claude/worktrees/ajc+remove-kr8s/`.
- On branch `worktree-ajc+remove-kr8s`, which was based on `ajc/k8s` HEAD.
- `make first-time-setup` has run successfully (`.venv/` exists, `uv run --active python -c "import kopf; import kr8s"` succeeds).
- Baseline unit tests all pass:

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
# expected: 1445 passed
```

If any of these are not true, fix before starting Task 1.

---

## Patterns reference (used throughout the plan)

Every migrated file adopts these shapes. They appear once here; later tasks just say "apply the standard patterns."

### P1 — Config load + ApiClient context manager

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient

@asynccontextmanager
async def k8s_client(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> AsyncIterator[ApiClient]:
    """Load k8s config (in-cluster first, kubeconfig fallback) and yield ApiClient."""
    try:
        config.load_incluster_config()          # SYNC — no await
    except config.ConfigException:
        await config.load_kube_config(          # ASYNC
            config_file=kubeconfig, context=context,
        )
    api = ApiClient()
    try:
        yield api
    finally:
        await api.close()
```

### P2 — List built-in resources (CoreV1Api, AppsV1Api)

```python
from kubernetes_asyncio import client
async with k8s_client() as api:
    core = client.CoreV1Api(api)
    pods = (await core.list_namespaced_pod(
        namespace, label_selector=sel,
    )).items
    for pod in pods:
        phase = pod.status.phase
        statuses = pod.status.container_statuses or []
        ready = bool(statuses) and all(cs.ready for cs in statuses)
```

### P3 — List all namespaces

```python
# kr8s.ALL  →  list_pod_for_all_namespaces
pods = (await core.list_pod_for_all_namespaces(label_selector=sel)).items
```

### P4 — Custom resource (CRD) get / list / patch

```python
from kubernetes_asyncio.client.exceptions import ApiException
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP, AIPERF_JOB_VERSION, AIPERF_JOB_PLURAL,
)

custom = client.CustomObjectsApi(api)

# Get by name — returns dict, caller validates with Pydantic
try:
    raw = await custom.get_namespaced_custom_object(
        group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
        plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    )
    job = AIPerfJobCR.model_validate(raw)
except ApiException as e:
    if e.status == 404:
        return None
    raise

# List, optional field_selector
result = await custom.list_namespaced_custom_object(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns,
    label_selector=label_selector,
    field_selector=field_selector,
)
raws = result.get("items", [])

# Merge patch (default merge; set body as dict)
await custom.patch_namespaced_custom_object(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    body={"spec": {"cancel": True}},
)

# JSON patch — list of ops, explicit content type
await custom.patch_namespaced_custom_object(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    body=patch_ops,
    _content_type="application/json-patch+json",
)

# Status subresource patch
await custom.patch_namespaced_custom_object_status(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    body={"status": {"phase": "Running"}},
)
```

### P5 — Delete resource

```python
await custom.delete_namespaced_custom_object(
    group=JOBSET_GROUP, version=JOBSET_VERSION,
    plural=JOBSET_PLURAL, namespace=ns, name=name,
)
# built-in
await core.delete_namespaced_config_map(name, ns)
await core.delete_namespace(name)
```

### P6 — Error mapping table

| kr8s | kubernetes_asyncio |
|---|---|
| `kr8s.NotFoundError` | `ApiException` with `e.status == 404` |
| `kr8s.ServerError` | `ApiException` |
| `e.response.status_code` (kr8s) | `e.status` (kubernetes_asyncio) |
| `e.response` | `e.body` (string), `e.reason`, `e.headers` |

### P7 — Log streaming (follow=True)

```python
# Always release the aiohttp response in a finally — otherwise connections leak.
raw = await core.read_namespaced_pod_log(
    name=pod, namespace=ns, container=cont,
    follow=True, _preload_content=False,
    since_seconds=since, tail_lines=tail,
)
try:
    async for line in raw.content:
        yield line.decode("utf-8", errors="replace").rstrip("\n")
finally:
    await raw.release()
```

### P8 — One-shot log read

```python
log_text: str = await core.read_namespaced_pod_log(
    name=pod, namespace=ns, container=cont, tail_lines=tail,
)
```

### P9 — Cluster version

```python
version = await client.VersionApi(api).get_code()
# Use: version.git_version, version.major, version.minor, version.platform
```

### P10 — Test mocks

Replace `kr8s.Api`/`async_get` mocks with:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from kubernetes_asyncio.client import ApiClient, CoreV1Api, CustomObjectsApi
from kubernetes_asyncio.client.exceptions import ApiException
from kubernetes_asyncio.client.models import V1Pod, V1PodList, V1PodStatus, V1ObjectMeta

# Mocking typed list responses:
pod = V1Pod(
    metadata=V1ObjectMeta(name="p1", namespace="ns"),
    status=V1PodStatus(phase="Running"),
)
pod_list = V1PodList(items=[pod])
mock_core = MagicMock(spec=CoreV1Api)
mock_core.list_namespaced_pod = AsyncMock(return_value=pod_list)

# Mocking 404:
mock_core.read_namespaced_pod = AsyncMock(side_effect=ApiException(status=404))

# Mocking dict CRD responses (CustomObjectsApi):
mock_custom = MagicMock(spec=CustomObjectsApi)
mock_custom.list_namespaced_custom_object = AsyncMock(return_value={"items": [raw_dict]})

# Patching constructors so `CoreV1Api(api)` returns our mock:
with patch("aiperf.kubernetes.client.client.CoreV1Api", return_value=mock_core), \
     patch("aiperf.kubernetes.client.client.CustomObjectsApi", return_value=mock_custom):
    ...
```

---

## Task 1: Foundation — deps, cr_refs, noisy_loggers

**Files:**
- Modify: `pyproject.toml` (top-level `dependencies` list)
- Create: `src/aiperf/kubernetes/cr_refs.py`
- Modify: `src/aiperf/common/noisy_loggers.py`
- Regenerate: `uv.lock`

**Why:** Introduce the new dep and the new constants so subsequent tasks can import them. `kr8s` stays for now.

- [ ] **Step 1: Add `kubernetes_asyncio` to `pyproject.toml`**

Open `pyproject.toml`. In the `[project].dependencies` list, add one line next to `"kr8s>=0.20.15",` — keep `kr8s` for now; it is removed in Task 8.

```diff
   "python-multipart>=0.0.22",
   "kr8s>=0.20.15",
+  "kubernetes_asyncio>=32.0,<33.0",
   "duckdb>=1.5.0",
```

- [ ] **Step 2: Regenerate lockfile**

```bash
unset VIRTUAL_ENV && uv lock
```

Expected: no conflict errors; `uv.lock` gains `kubernetes_asyncio` entries.

- [ ] **Step 3: Install new dep into the venv**

```bash
unset VIRTUAL_ENV && make install
```

- [ ] **Step 4: Verify install**

```bash
unset VIRTUAL_ENV && uv run --active python -c "
import kubernetes_asyncio
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient, CoreV1Api, CustomObjectsApi, AppsV1Api, VersionApi
from kubernetes_asyncio.client.exceptions import ApiException
print('kubernetes_asyncio', kubernetes_asyncio.__version__)
"
```

Expected: prints a version string. No ImportError.

- [ ] **Step 5: Create `src/aiperf/kubernetes/cr_refs.py`**

Create the file with exactly this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom Resource coordinates for CustomObjectsApi calls.

These constants replace the kr8s ``new_class(...)`` wrappers. Every
``CustomObjectsApi.*_namespaced_custom_object`` call in the codebase
takes the matching (group, version, plural) triple from this module.
"""

# AIPerfJob (the AIPerf-owned CR)
AIPERF_JOB_GROUP = "aiperf.nvidia.com"
AIPERF_JOB_VERSION = "v1alpha1"
AIPERF_JOB_PLURAL = "aiperfjobs"

# JobSet (external — jobset-operator)
JOBSET_GROUP = "jobset.x-k8s.io"
JOBSET_VERSION = "v1alpha2"
JOBSET_PLURAL = "jobsets"
```

- [ ] **Step 6: Update `src/aiperf/common/noisy_loggers.py`**

Read the current file first. It currently suppresses `httpx` (kr8s's HTTP library). Keep the `httpx` suppression but add `aiohttp` / `kubernetes_asyncio` suppression. Rewrite the docstring to note both backends.

Change the docstring and logger list. Example content (full rewrite of module):

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suppress noisy HTTP loggers emitted by Kubernetes client libraries.

kubernetes_asyncio uses aiohttp internally and emits access-log-style
lines for every request. httpx (a legacy kr8s dependency during the
migration) emits similar lines. Both are silenced to WARNING.
"""

import logging

_NOISY_LOGGERS = (
    "aiohttp.access",
    "aiohttp.client",
    "kubernetes_asyncio.client.rest",
    "httpx",  # legacy — removed with kr8s
)


def suppress_noisy_http_loggers() -> None:
    """Raise noisy HTTP-client loggers to WARNING."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
```

- [ ] **Step 7: Run unit tests — nothing should be affected yet**

Per memory `feedback_pytest_single_subfolder.md`, run subfolders one at a time. Use these three commands:

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: all green; counts at or above baseline.

- [ ] **Step 8: ruff + pre-commit**

```bash
ruff format . && ruff check --fix .
```

Expected: no failures.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/aiperf/kubernetes/cr_refs.py src/aiperf/common/noisy_loggers.py
git commit -s -m "$(cat <<'EOF'
feat(deps): add kubernetes_asyncio, CR coordinate constants, update noisy loggers

Introduce kubernetes_asyncio alongside kr8s (both kept for now; kr8s is
removed in the final migration commit). Add cr_refs.py with the six
CR group/version/plural constants that CustomObjectsApi call sites
will consume. Update noisy_loggers to silence aiohttp/kubernetes_asyncio
access logs in addition to the existing httpx suppression.

Foundation-only commit — no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rewrite `kubernetes/client.py` on `kubernetes_asyncio`

**Files:**
- Rewrite: `src/aiperf/kubernetes/client.py`
- Rewrite: `tests/unit/kubernetes/test_client.py`
- Modify: `src/aiperf/kubernetes/cli_helpers.py` (same-package callers) if any caller still references `client.api` as a `kr8s.Api`.

**Why:** `AIPerfKubeClient` is the central wrapper used by almost every call site. To swap its internals to `kubernetes_asyncio` without breaking callers, we retain the class as a **thin delegating facade** over the new free functions. Future tasks migrate callers from `client.method(...)` to `method(api, ...)` and Task 8 deletes the facade.

At end of Task 2: `client.py` has zero `kr8s` imports; `AIPerfKubeClient` exists but every method body is a one-liner delegating to a free function.

### 2.1 New `client.py` — full contents

- [ ] **Step 1: Replace `src/aiperf/kubernetes/client.py` with the new implementation**

Complete file content (approx 350-450 lines; this is the exact target):

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes client — free functions + AIPerfKubeClient facade.

Free functions (``k8s_client``, ``list_aiperf_jobs``, ``find_jobset``, …)
are the canonical interface. They take an ``ApiClient`` explicitly and
call ``CoreV1Api(api)`` / ``CustomObjectsApi(api)`` inline so the reader
sees the native kubernetes_asyncio API surface.

``AIPerfKubeClient`` remains as a thin facade that delegates to the free
functions so existing callers keep working during the migration. It is
removed in the kr8s cleanup commit when no callers remain.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.console import print_info, print_success, print_warning
from aiperf.kubernetes.constants import JobSetLabels, Labels
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import (
    AIPerfJobCR,
    AIPerfJobInfo,
    JobSetInfo,
    PodSummary,
)

logger = logging.getLogger(__name__)


# ----- Config + ApiClient ----------------------------------------------------


@asynccontextmanager
async def k8s_client(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> AsyncIterator[ApiClient]:
    """Load k8s config and yield an ApiClient.

    In-cluster first, kubeconfig fallback. The ApiClient is closed on exit.
    """
    suppress_noisy_http_loggers()
    try:
        config.load_incluster_config()
    except config.ConfigException:
        await config.load_kube_config(config_file=kubeconfig, context=context)
    api = ApiClient()
    try:
        yield api
    finally:
        await api.close()


# ----- Label selectors (pure strings) ----------------------------------------


def job_selector(job_id: str) -> str:
    """Label selector for all AIPerf resources belonging to a job."""
    return f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id}"


def controller_selector(job_id: str) -> str:
    """Label selector for the controller pod of a job."""
    return (
        f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id},"
        f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
    )


# ----- AIPerfJob CR helpers --------------------------------------------------


async def list_aiperf_jobs(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    status_filter: str | None = None,
) -> list[AIPerfJobInfo]:
    """List AIPerfJob CRs, sorted newest-first."""
    custom = client.CustomObjectsApi(api)
    try:
        if all_namespaces:
            result = await custom.list_cluster_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
            )
        else:
            ns = namespace or "default"
            result = await custom.list_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=ns,
            )
    except ApiException as e:
        if e.status == 404:
            return []
        raise

    infos = [AIPerfJobCR.model_validate(raw).to_info() for raw in result.get("items", [])]
    if status_filter:
        infos = [i for i in infos if i.phase == status_filter]
    infos.sort(key=lambda x: x.created, reverse=True)
    return infos


async def find_aiperf_job(
    api: ApiClient,
    name: str,
    namespace: str | None = None,
) -> AIPerfJobInfo | None:
    """Find an AIPerfJob by resource name, with fallback to jobId match."""
    custom = client.CustomObjectsApi(api)

    # Direct lookup by name — most common path.
    if namespace is not None:
        try:
            raw = await custom.get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                plural=AIPERF_JOB_PLURAL,
                namespace=namespace,
                name=name,
            )
            return AIPerfJobCR.model_validate(raw).to_info()
        except ApiException as e:
            if e.status != 404:
                raise

    # Fallback: scan all namespaces for a status.jobId match.
    try:
        result = await custom.list_cluster_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            field_selector=f"metadata.name={name}" if namespace is None else None,
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise

    for raw in result.get("items", []):
        cr = AIPerfJobCR.model_validate(raw)
        if cr.metadata.name == name or cr.status.job_id == name:
            return cr.to_info()
    return None


async def get_raw_aiperfjob_status(
    api: ApiClient, name: str, namespace: str,
) -> dict[str, Any]:
    """Return the raw ``status`` dict of an AIPerfJob by name (empty on miss)."""
    custom = client.CustomObjectsApi(api)
    try:
        raw = await custom.get_namespaced_custom_object(
            group=AIPERF_JOB_GROUP,
            version=AIPERF_JOB_VERSION,
            plural=AIPERF_JOB_PLURAL,
            namespace=namespace,
            name=name,
        )
    except ApiException:
        return {}
    return raw.get("status", {}) or {}


async def cancel_aiperf_job(api: ApiClient, name: str, namespace: str) -> None:
    """Cancel an AIPerfJob by setting ``spec.cancel=true`` (merge patch)."""
    custom = client.CustomObjectsApi(api)
    await custom.patch_namespaced_custom_object(
        group=AIPERF_JOB_GROUP,
        version=AIPERF_JOB_VERSION,
        plural=AIPERF_JOB_PLURAL,
        namespace=namespace,
        name=name,
        body={"spec": {"cancel": True}},
    )


# ----- JobSet helpers --------------------------------------------------------


async def _list_jobsets_raw(
    api: ApiClient,
    label_selector: str,
    namespace: str | None = None,
    field_selector: str | None = None,
) -> list[dict[str, Any]]:
    """List JobSet raw dicts matching selectors."""
    custom = client.CustomObjectsApi(api)
    kwargs: dict[str, Any] = {"label_selector": label_selector}
    if field_selector:
        kwargs["field_selector"] = field_selector

    if namespace is None:
        result = await custom.list_cluster_custom_object(
            group=JOBSET_GROUP, version=JOBSET_VERSION, plural=JOBSET_PLURAL, **kwargs,
        )
    else:
        result = await custom.list_namespaced_custom_object(
            group=JOBSET_GROUP, version=JOBSET_VERSION, plural=JOBSET_PLURAL,
            namespace=namespace, **kwargs,
        )
    return result.get("items", []) or []


async def list_jobsets(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    job_id: str | None = None,
    status_filter: str | None = None,
) -> list[JobSetInfo]:
    """List AIPerf-owned JobSets, sorted newest-first."""
    label_selector = Labels.SELECTOR
    if job_id:
        label_selector += f",{Labels.JOB_ID}={job_id}"

    ns = None if all_namespaces else (namespace or "default")
    try:
        raws = await _list_jobsets_raw(api, label_selector, ns)
    except ApiException as e:
        if e.status == 404:
            return []
        raise

    infos = [JobSetInfo.from_raw(r) for r in raws]
    if status_filter:
        infos = [i for i in infos if i.status == status_filter]
    infos.sort(key=lambda x: x.created, reverse=True)
    return infos


async def find_jobset(
    api: ApiClient,
    job_id: str,
    namespace: str | None = None,
) -> JobSetInfo | None:
    """Find a JobSet by AIPerf job ID label, falling back to resource name."""
    try:
        raws = await _list_jobsets_raw(api, job_selector(job_id), namespace)
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    if raws:
        return JobSetInfo.from_raw(raws[0])

    try:
        raws = await _list_jobsets_raw(
            api, Labels.SELECTOR, namespace,
            field_selector=f"metadata.name={job_id}",
        )
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    return JobSetInfo.from_raw(raws[0]) if raws else None


async def delete_jobset(api: ApiClient, name: str, namespace: str) -> None:
    """Delete a JobSet and its associated ConfigMap/Role/RoleBinding."""
    custom = client.CustomObjectsApi(api)
    core = client.CoreV1Api(api)
    rbac = client.RbacAuthorizationV1Api(api)

    try:
        await custom.delete_namespaced_custom_object(
            group=JOBSET_GROUP, version=JOBSET_VERSION,
            plural=JOBSET_PLURAL, namespace=namespace, name=name,
        )
        print_success(f"Deleted JobSet/{name}")
    except ApiException as e:
        if e.status == 404:
            print_warning(f"JobSet/{name} not found")
        else:
            raise

    # Associated resources named "<jobset>-<suffix>"
    targets = [
        (core.delete_namespaced_config_map, f"{name}-config", "ConfigMap"),
        (rbac.delete_namespaced_role, f"{name}-role", "Role"),
        (rbac.delete_namespaced_role_binding, f"{name}-binding", "RoleBinding"),
    ]
    for delete_fn, resource_name, kind in targets:
        try:
            await delete_fn(name=resource_name, namespace=namespace)
            print_success(f"Deleted {kind}/{resource_name}")
        except ApiException as e:
            if e.status in (404, 409):
                # 404 already gone; 409 namespace terminating — both benign.
                continue
            print_warning(f"Failed to delete {kind}/{resource_name}: {e}")


async def delete_namespace(api: ApiClient, name: str) -> None:
    """Delete a Kubernetes namespace (404 treated as already gone)."""
    core = client.CoreV1Api(api)
    try:
        await core.delete_namespace(name=name)
        print_success(f"Deleted Namespace/{name}")
    except ApiException as e:
        if e.status == 404:
            print_info(f"Namespace {name} not found (may already be deleted)")
        else:
            print_warning(f"Failed to delete namespace: {e}")


# ----- Pod helpers -----------------------------------------------------------


async def get_pod_summary(
    api: ApiClient, jobset_name: str, namespace: str,
) -> PodSummary:
    """Pod readiness summary for a JobSet."""
    core = client.CoreV1Api(api)
    try:
        pod_list = await core.list_namespaced_pod(
            namespace, label_selector=f"{JobSetLabels.JOBSET_NAME}={jobset_name}",
        )
    except ApiException:
        return PodSummary(ready=0, total=0, restarts=0)

    pods = pod_list.items
    total = len(pods)
    ready = 0
    restarts = 0
    for pod in pods:
        statuses = (pod.status.container_statuses or []) if pod.status else []
        pod_ready = bool(statuses) and all(cs.ready for cs in statuses)
        phase = pod.status.phase if pod.status else None
        if pod_ready and phase == PodPhase.RUNNING:
            ready += 1
        restarts += sum(cs.restart_count or 0 for cs in statuses)
    return PodSummary(ready=ready, total=total, restarts=restarts)


async def find_operator_pod(
    api: ApiClient,
    namespace: str = "aiperf-system",
    label_selector: str = "app.kubernetes.io/name=aiperf-operator",
) -> tuple[str, PodPhase] | None:
    """Find the operator pod; returns (name, phase) or None."""
    core = client.CoreV1Api(api)
    pod_list = await core.list_namespaced_pod(namespace, label_selector=label_selector)
    if not pod_list.items:
        return None
    pod = pod_list.items[0]
    raw_phase = pod.status.phase if pod.status and pod.status.phase else "Unknown"
    return (pod.metadata.name, PodPhase(raw_phase))


async def find_controller_pod(
    api: ApiClient,
    namespace: str,
    job_id: str,
) -> tuple[str, PodPhase] | None:
    """Find the controller pod for a job; returns (name, phase) or None."""
    core = client.CoreV1Api(api)
    pod_list = await core.list_namespaced_pod(
        namespace, label_selector=controller_selector(job_id),
    )
    if not pod_list.items:
        return None
    pod = pod_list.items[0]
    raw_phase = pod.status.phase if pod.status and pod.status.phase else "Unknown"
    return (pod.metadata.name, PodPhase(raw_phase))


async def find_retrievable_pod(
    api: ApiClient,
    namespace: str,
    job_id: str,
    *,
    require_running: bool = False,
) -> tuple[str, PodPhase] | None:
    """Find the controller pod only if it is in a retrievable phase."""
    pod_info = await find_controller_pod(api, namespace, job_id)
    if not pod_info:
        return None
    pod_name, pod_phase = pod_info
    if require_running:
        if pod_phase != PodPhase.RUNNING:
            return None
    elif not pod_phase.is_retrievable:
        return None
    return pod_name, pod_phase


async def wait_for_controller_pod_ready(
    api: ApiClient,
    namespace: str,
    job_id: str,
    timeout: int = 300,
) -> str:
    """Poll until the controller pod is Running; returns its name."""
    start = asyncio.get_running_loop().time()
    last_log = 0.0
    while True:
        result = await find_controller_pod(api, namespace, job_id)
        elapsed = asyncio.get_running_loop().time() - start
        if result:
            pod_name, phase = result
            if phase == PodPhase.RUNNING:
                return pod_name
            if elapsed - last_log >= 10:
                logger.info("Controller pod %s: %s (%.0fs)", pod_name, phase, elapsed)
                last_log = elapsed
        elif elapsed - last_log >= 10:
            logger.info("No controller pod found yet (%.0fs)", elapsed)
            last_log = elapsed
        if elapsed > timeout:
            raise TimeoutError(
                f"Controller pod not ready after {timeout}s. "
                f"Check with: kubectl get pods -n {namespace}"
            )
        await asyncio.sleep(2)


async def get_pods(
    api: ApiClient, namespace: str, label_selector: str,
) -> list[Any]:
    """Return list of V1Pod matching label selector (typed access)."""
    core = client.CoreV1Api(api)
    return (await core.list_namespaced_pod(namespace, label_selector=label_selector)).items


async def cluster_version(api: ApiClient) -> dict[str, Any]:
    """Return Kubernetes cluster version info as a dict."""
    vinfo = await client.VersionApi(api).get_code()
    return {
        "major": vinfo.major,
        "minor": vinfo.minor,
        "gitVersion": vinfo.git_version,
        "gitCommit": vinfo.git_commit,
        "platform": vinfo.platform,
    }


# ----- AIPerfKubeClient facade (removed in kr8s cleanup commit) -------------


class AIPerfKubeClient:
    """Backwards-compat facade over the free functions in this module.

    **Deprecated** — new code should call the free functions directly using
    ``async with k8s_client() as api:`` to make the underlying
    ``kubernetes_asyncio`` API surface visible.
    """

    def __init__(self, api: ApiClient) -> None:
        self._api = api

    @classmethod
    async def create(
        cls,
        *,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
    ) -> AIPerfKubeClient:
        """Create a client; CALLER is responsible for eventual cleanup.

        Prefer ``async with k8s_client() as api:`` + free functions. This
        factory opens an ApiClient without a matching close — acceptable
        only for process-long services where the client lives until exit.
        """
        suppress_noisy_http_loggers()
        try:
            config.load_incluster_config()
        except config.ConfigException:
            await config.load_kube_config(config_file=kubeconfig, context=kube_context)
        return cls(ApiClient())

    @property
    def api(self) -> ApiClient:
        return self._api

    async def close(self) -> None:
        await self._api.close()

    # -- label helpers stay as static methods for kr8s-era call sites ---------
    job_selector = staticmethod(job_selector)
    controller_selector = staticmethod(controller_selector)

    # -- delegations ---------------------------------------------------------

    async def list_jobs(self, *args, **kwargs):
        return await list_aiperf_jobs(self._api, *args, **kwargs)

    async def find_job(self, *args, **kwargs):
        return await find_aiperf_job(self._api, *args, **kwargs)

    async def get_raw_status(self, name: str, namespace: str) -> dict[str, Any]:
        return await get_raw_aiperfjob_status(self._api, name, namespace)

    async def cancel_job(self, name: str, namespace: str) -> None:
        await cancel_aiperf_job(self._api, name, namespace)

    async def list_jobsets(self, *args, **kwargs):
        return await list_jobsets(self._api, *args, **kwargs)

    async def find_jobset(self, *args, **kwargs):
        return await find_jobset(self._api, *args, **kwargs)

    async def delete_jobset(self, name: str, namespace: str) -> None:
        await delete_jobset(self._api, name, namespace)

    async def delete_namespace(self, namespace: str) -> None:
        await delete_namespace(self._api, namespace)

    async def get_pod_summary(self, jobset_name: str, namespace: str) -> PodSummary:
        return await get_pod_summary(self._api, jobset_name, namespace)

    async def find_operator_pod(self, *args, **kwargs):
        return await find_operator_pod(self._api, *args, **kwargs)

    async def find_controller_pod(self, namespace: str, job_id: str):
        return await find_controller_pod(self._api, namespace, job_id)

    async def find_retrievable_pod(self, *args, **kwargs):
        return await find_retrievable_pod(self._api, *args, **kwargs)

    async def wait_for_controller_pod_ready(self, *args, **kwargs):
        return await wait_for_controller_pod_ready(self._api, *args, **kwargs)

    async def get_pods(self, namespace: str, label_selector: str):
        return await get_pods(self._api, namespace, label_selector)

    async def version(self) -> dict[str, Any]:
        return await cluster_version(self._api)
```

- [ ] **Step 2: Fix `src/aiperf/kubernetes/cli_helpers.py`**

Read the file. Any code that did `await client.api.async_version()` or expected `client.api` to be a `kr8s.Api` must be updated — `client.api` is now an `ApiClient`. Update docstrings that still say "kr8s" to "kubernetes_asyncio". Keep the existing public API.

Search for `async_` / `.raw` patterns and replace (use typed access). Example:

```python
# before
version_info = await client.api.async_version()
# after
version_info = await client.version()
```

- [ ] **Step 3: Rewrite `tests/unit/kubernetes/test_client.py`**

Replace the file contents to cover the new free functions using mock `ApiClient` + mock `CustomObjectsApi`/`CoreV1Api` per pattern P10.

Structure one test per free function:
- `test_k8s_client_uses_incluster_first` (patch `config.load_incluster_config` to succeed; assert no kubeconfig call)
- `test_k8s_client_falls_back_to_kubeconfig` (patch `load_incluster_config` to raise `ConfigException`; assert kubeconfig loaded)
- `test_list_aiperf_jobs_returns_sorted_infos`
- `test_list_aiperf_jobs_404_returns_empty`
- `test_list_aiperf_jobs_filters_by_phase`
- `test_find_aiperf_job_by_name_namespaced`
- `test_find_aiperf_job_falls_back_to_job_id`
- `test_find_aiperf_job_not_found_returns_none`
- `test_cancel_aiperf_job_applies_spec_cancel_true`
- `test_find_jobset_by_label`
- `test_find_jobset_by_name_fallback`
- `test_delete_jobset_also_removes_aux_resources`
- `test_get_pod_summary_counts_ready_and_restarts`
- `test_find_controller_pod_returns_name_and_phase`
- `test_find_controller_pod_empty_returns_none`
- `test_wait_for_controller_pod_ready_succeeds`
- `test_wait_for_controller_pod_ready_times_out`

Also add tests for the facade:
- `test_facade_list_jobs_delegates_to_free_function`
- `test_facade_close_closes_api_client`

Use `@pytest.mark.asyncio` + `pytest.param` with `# fmt: skip` on the closing `)` per CLAUDE.md.

- [ ] **Step 4: Run the full unit suite**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: green (facade preserves behavior — operator + other callers that still use `AIPerfKubeClient.*` keep passing).

- [ ] **Step 5: ruff**

```bash
ruff format . && ruff check --fix .
```

- [ ] **Step 8: Commit**

```bash
git add src/aiperf/kubernetes/client.py src/aiperf/kubernetes/cli_helpers.py tests/unit/kubernetes/test_client.py
git commit -s -m "$(cat <<'EOF'
refactor(kubernetes): rewrite client.py on kubernetes_asyncio

Replace the kr8s-backed AIPerfKubeClient with native kubernetes_asyncio
free functions (k8s_client ctx manager, list_aiperf_jobs, find_jobset,
get_pod_summary, find_controller_pod, wait_for_controller_pod_ready,
delete_jobset, delete_namespace, cluster_version, …). AIPerfKubeClient
is retained as a thin delegating facade for backwards compatibility;
external callers migrate incrementally in subsequent commits and the
facade is removed in the final cleanup commit.

No external behavior change. Same-package caller (cli_helpers.py)
updated to match the new ApiClient-based accessors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Migrate `kubernetes/preflight.py` + `preflight_utils.py`

**Files:**
- Modify: `src/aiperf/kubernetes/preflight.py` (1073 lines — many `import kr8s` inside helpers)
- Modify: `src/aiperf/kubernetes/preflight_utils.py`
- Modify: `tests/unit/kubernetes/test_preflight.py`
- Modify: `tests/unit/cli_commands/test_kube_preflight.py`

**Why:** `preflight.py` is a self-contained surface (cluster readiness checks: namespace exists, deployment ready, CRD installed, nodes allocatable, resource quotas, secrets present). Each helper does one `kr8s.asyncio.objects.*` lookup. Translate each to `kubernetes_asyncio` typed access.

### 3.1 Translation table (apply to every helper in `preflight.py`)

| kr8s (before) | kubernetes_asyncio (after) |
|---|---|
| `from kr8s.asyncio.objects import Namespace` then `Namespace.get(name, api=api)` | `await client.CoreV1Api(api).read_namespace(name)` |
| `from kr8s.asyncio.objects import Deployment` then `Deployment.get(name, namespace=ns, api=api)` | `await client.AppsV1Api(api).read_namespaced_deployment(name, ns)` |
| `from kr8s.asyncio.objects import ServiceAccount` then `.get(...)` | `await client.CoreV1Api(api).read_namespaced_service_account(name, ns)` |
| `from kr8s.asyncio.objects import Node` then `async for n in api.async_get(Node)` | `(await client.CoreV1Api(api).list_node()).items` |
| `from kr8s.asyncio.objects import ResourceQuota` then `async for q in api.async_get(ResourceQuota, namespace=ns)` | `(await client.CoreV1Api(api).list_namespaced_resource_quota(ns)).items` |
| `from kr8s.asyncio.objects import Secret` then `.get(...)` | `await client.CoreV1Api(api).read_namespaced_secret(name, ns)` |
| `from kr8s.asyncio.objects import NetworkPolicy` then `async for n in api.async_get(NetworkPolicy, namespace=ns)` | `(await client.NetworkingV1Api(api).list_namespaced_network_policy(ns)).items` |
| `from kr8s.asyncio.objects import Service` then `.get(...)` | `await client.CoreV1Api(api).read_namespaced_service(name, ns)` |
| `from kr8s.asyncio.objects import CustomResourceDefinition` then `.get(...)` | `await client.ApiextensionsV1Api(api).read_custom_resource_definition(name)` |
| `obj.raw["spec"]["foo"]` (any kr8s object) | `obj.spec.foo` (typed V1* attribute) |
| `except kr8s.NotFoundError:` | `except ApiException as e: if e.status == 404: ... else: raise` |
| `except kr8s.ServerError as e:` with `e.response.status_code` | `except ApiException as e:` with `e.status` |
| `api: kr8s.Api` in signatures | `api: ApiClient` |
| `import kr8s` (at callsite) | delete |

### 3.2 Steps

- [ ] **Step 1: Translate `src/aiperf/kubernetes/preflight.py`**

Read the file, then for every helper function:
- Replace the inline `import kr8s` / `from kr8s.asyncio.objects import X` with top-of-module `from kubernetes_asyncio import client` and `from kubernetes_asyncio.client.exceptions import ApiException`.
- Swap the kr8s lookup for the `kubernetes_asyncio` call per the translation table above.
- Change type annotations: `api: kr8s.Api` → `api: ApiClient`.
- Change `obj.raw[...]` dict-walking to typed attribute access where possible (e.g., `ns.raw.get("status", {}).get("phase")` → `ns.status.phase if ns.status else None`).
- Error handling: `except kr8s.NotFoundError:` → `except ApiException as e: if e.status == 404: ... else: raise`.

Keep function signatures and return types the same — only internals change. External callers (operator and CLI preflight code) still import these helpers by name.

- [ ] **Step 2: Translate `src/aiperf/kubernetes/preflight_utils.py`**

Read the file. Apply the same translation table. `preflight_utils.py` is smaller (~100 lines) — typically a handful of helpers that build a shared `api` argument or format messages.

- [ ] **Step 3: Rewrite `tests/unit/kubernetes/test_preflight.py`**

Read the existing test file. For every test that mocks `kr8s` (e.g., `MagicMock(spec=kr8s.Api)` or `async_get(...)`):
- Replace with `MagicMock(spec=ApiClient)` and `AsyncMock` returns for the specific typed API method (`read_namespace`, `list_node`, etc.).
- Replace `kr8s.NotFoundError` raises with `ApiException(status=404)`.
- Replace `kr8s.ServerError(response=...)` raises with `ApiException(status=500, reason="...")`.
- Patch the API constructor at use site: `patch("aiperf.kubernetes.preflight.client.CoreV1Api", return_value=mock_core)`.

Use pattern P10 as the mocking template.

- [ ] **Step 4: Rewrite `tests/unit/cli_commands/test_kube_preflight.py`**

Apply the same mock translation.

- [ ] **Step 5: Run preflight-related tests**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: green.

- [ ] **Step 7: Verify no kr8s imports in migrated files**

```bash
grep -n "kr8s" src/aiperf/kubernetes/preflight.py src/aiperf/kubernetes/preflight_utils.py
# expected: empty output (no matches)
```

- [ ] **Step 8: ruff + pre-commit + commit**

```bash
ruff format . && ruff check --fix .
git add src/aiperf/kubernetes/preflight.py src/aiperf/kubernetes/preflight_utils.py tests/unit/kubernetes/test_preflight.py tests/unit/cli_commands/test_kube_preflight.py
git commit -s -m "$(cat <<'EOF'
refactor(kubernetes): port preflight.py and preflight_utils.py to kubernetes_asyncio

Every preflight helper now uses CoreV1Api / AppsV1Api /
ApiextensionsV1Api / NetworkingV1Api directly with typed V1* attribute
access. ApiException-based error handling replaces kr8s NotFoundError
and ServerError. No caller-visible API change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrate `kubernetes/watchdog.py` — `Kr8sWatchdogSource` → `K8sWatchdogSource`

**Files:**
- Modify: `src/aiperf/kubernetes/watchdog.py` (rename class + translate its 5 methods)
- Modify: `tests/unit/kubernetes/test_watchdog.py`
- Grep-check: callers of the old name.

**Why:** `Kr8sWatchdogSource` is an injectable abstraction class that `BenchmarkWatchdog` consumes. Its 5 methods (`list_pods`, `list_events`, `list_nodes`, `list_namespaces`, `pod_logs`) wrap kr8s lookups. Rename and translate internals — no change to the injection contract.

- [ ] **Step 1: Find all call sites of `Kr8sWatchdogSource`**

```bash
grep -rn "Kr8sWatchdogSource" src tests --include="*.py"
```

Expected sites (≤3): the class definition, one or two instantiation sites (`kubernetes/watchdog.py:21`, `kubernetes/watchdog.py:646`, `kubernetes/attach.py`).

- [ ] **Step 2: Translate the class**

In `src/aiperf/kubernetes/watchdog.py`, rename `Kr8sWatchdogSource` → `K8sWatchdogSource` (class keyword, classmethod factories, any type hints). Replace internals:

```python
# K8sWatchdogSource.list_pods
async def list_pods(self, namespace: str) -> list[Any]:
    core = client.CoreV1Api(self._api)
    return (await core.list_namespaced_pod(namespace)).items

# list_events
async def list_events(self, namespace: str) -> list[Any]:
    core = client.CoreV1Api(self._api)
    return (await core.list_namespaced_event(namespace)).items

# list_nodes
async def list_nodes(self) -> list[Any]:
    core = client.CoreV1Api(self._api)
    return (await core.list_node()).items

# list_namespaces
async def list_namespaces(self) -> list[Any]:
    core = client.CoreV1Api(self._api)
    return (await core.list_namespace()).items

# pod_logs (one-shot, tail)
async def pod_logs(self, namespace: str, name: str, tail: int = 100) -> str:
    core = client.CoreV1Api(self._api)
    try:
        return await core.read_namespaced_pod_log(
            name=name, namespace=namespace, tail_lines=tail,
        )
    except ApiException:
        return ""
```

Change the stored `self._api` type annotation from `kr8s.Api` to `ApiClient`.

- [ ] **Step 3: Update the consumer-style call sites inside watchdog.py**

The watchdog may access `.raw` fields on V1Pod/V1Event/V1Node. Translate to typed access:

```python
# Before
pod.raw["status"]["phase"]
# After
pod.status.phase if pod.status else None

# Before
event.raw["message"]
# After
event.message
```

Audit the module for any `.raw[...]` patterns and translate.

- [ ] **Step 4: Update the main entry point of the module**

Replace:
```python
api = await kr8s.asyncio.api()
source = Kr8sWatchdogSource(api)
```
with:
```python
async with k8s_client() as api:
    source = K8sWatchdogSource(api)
    ...
```

If the calling shape doesn't fit a context manager (e.g., the watchdog lives longer than a function scope), manually manage the ApiClient:
```python
try:
    config.load_incluster_config()
except config.ConfigException:
    await config.load_kube_config()
api = ApiClient()
source = K8sWatchdogSource(api)
...
finally:
    await api.close()
```

- [ ] **Step 5: Update `kubernetes/attach.py` reference**

`src/aiperf/kubernetes/attach.py:134` imports `Kr8sWatchdogSource`. Change that single import to `K8sWatchdogSource`. (The rest of `attach.py` is migrated in Task 5.)

- [ ] **Step 6: Rewrite `tests/unit/kubernetes/test_watchdog.py`**

Substitute all kr8s mocks with kubernetes_asyncio mocks per pattern P10. Use `MagicMock(spec=ApiClient)` for the source constructor arg; each test injects an `AsyncMock(return_value=...)` for the specific `CoreV1Api.list_*` method under test.

- [ ] **Step 7: Run the full unit suite + lint**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
ruff format . && ruff check --fix .
```

Expected: green.

- [ ] **Step 8: Verify no kr8s left in watchdog**

```bash
grep -n "kr8s\|Kr8sWatchdogSource" src/aiperf/kubernetes/watchdog.py
# expected: empty
```

- [ ] **Step 10: Commit**

```bash
git add src/aiperf/kubernetes/watchdog.py src/aiperf/kubernetes/attach.py tests/unit/kubernetes/test_watchdog.py
git commit -s -m "$(cat <<'EOF'
refactor(kubernetes): rename Kr8sWatchdogSource to K8sWatchdogSource and port internals

Watchdog source now uses CoreV1Api typed list calls; consumers unchanged.
Pod/event/node attribute access uses typed V1* fields instead of .raw
dict walking. attach.py import renamed; test_watchdog.py mocks swapped
to kubernetes_asyncio per standard pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Migrate remaining `kubernetes/*` modules

**Files:**
- Modify: `src/aiperf/kubernetes/attach.py`
- Modify: `src/aiperf/kubernetes/results.py` (~751 lines — 5 function signatures accepting `client: AIPerfKubeClient`)
- Modify: `src/aiperf/kubernetes/logs.py`
- Modify: `src/aiperf/kubernetes/watch_orchestrator.py` (creates AIPerfKubeClient)
- Modify: `src/aiperf/kubernetes/watch_pollers.py` (imports AsyncAIPerfJob from client)
- Modify: `tests/unit/kubernetes/test_logs.py`
- Modify: `tests/unit/kubernetes/test_cli_helpers.py`
- Modify: `tests/harness/k8s.py` (the shared fake)

**Why:** These remaining `kubernetes/*` modules either import `kr8s` directly (attach, watch_pollers) or accept `AIPerfKubeClient` as a parameter (results, logs, watch_orchestrator). After this task, `kubernetes/*` is fully migrated to `kubernetes_asyncio`; the `AIPerfKubeClient` facade is no longer used within the package (though still available externally for CLI/operator migration in later tasks).

### 5.1 attach.py

- [ ] **Step 1: Translate `src/aiperf/kubernetes/attach.py`**

- Remove `from kr8s.asyncio.objects import Pod` and the `AsyncAIPerfJob` import from `kr8s_resources`.
- Replace Pod lookups with CoreV1Api + typed access:
  ```python
  # before
  from kr8s.asyncio.objects import Pod
  pod = await Pod.get(name, namespace=ns, api=api)
  log_text: str = await pod.logs(tail_lines=tail)
  # after
  core = client.CoreV1Api(api)
  log_text = await core.read_namespaced_pod_log(name, ns, tail_lines=tail)
  ```
- Replace `AsyncAIPerfJob.get(...)` / `async_get(AsyncAIPerfJob, ...)` with `CustomObjectsApi.get_namespaced_custom_object(group=AIPERF_JOB_GROUP, ...)`.
- Change function signatures: `client: AIPerfKubeClient` → `api: ApiClient`. Callers (in Task 6 / 7) pass `api` directly.
- Update `async with port_forward_with_status(...)` call sites — these are unchanged (port_forward uses kubectl subprocess).

### 5.2 results.py

- [ ] **Step 2: Translate `src/aiperf/kubernetes/results.py`**

- Change every function signature `client: AIPerfKubeClient` → `api: ApiClient`.
- Replace `client.api` with `api` everywhere.
- Replace `client.find_controller_pod(...)` with `from aiperf.kubernetes.client import find_controller_pod; await find_controller_pod(api, ...)`.
- Same for `wait_for_controller_pod_ready`, `find_retrievable_pod`.
- Any direct kr8s use — swap per pattern tables.

### 5.3 logs.py

- [ ] **Step 3: Translate `src/aiperf/kubernetes/logs.py`**

Key log-stream release pattern (P7) applies here. Function signature changes from `client: AIPerfKubeClient` to `api: ApiClient`. Controller-pod discovery uses free functions.

### 5.4 watch_orchestrator.py

- [ ] **Step 4: Translate `src/aiperf/kubernetes/watch_orchestrator.py`**

- Remove `AIPerfKubeClient.create(...)` call at line ~60. Replace with `async with k8s_client(kubeconfig=..., context=...) as api:` pattern.
- Pass `api` to `CRPoller`, `EventPoller`, `PodPoller` (import from `watch_pollers`).

### 5.5 watch_pollers.py

- [ ] **Step 5: Translate `src/aiperf/kubernetes/watch_pollers.py`**

- Change `from aiperf.kubernetes.client import AsyncAIPerfJob` → drop; import `AIPERF_JOB_*` from `cr_refs` instead.
- Change the poller inputs from kr8s API type to `ApiClient`.
- Translate CR queries inside `CRPoller` to `CustomObjectsApi.list_namespaced_custom_object(...)`.
- For `EventPoller` and `PodPoller` — translate kr8s Event/Pod iteration to `CoreV1Api.list_namespaced_event` / `list_namespaced_pod`.

### 5.6 tests/harness/k8s.py

- [ ] **Step 6: Rewrite `tests/harness/k8s.py`**

This is the shared test harness. Replace `build_mock_kube_client` with two helpers:
```python
def build_mock_api(config: MagicMock | None = None) -> MagicMock:
    """Return a MagicMock(spec=ApiClient) with AsyncMock methods."""
    api = MagicMock(spec=ApiClient)
    api.close = AsyncMock()
    return api


def patch_api_accessors(*, core: MagicMock | None = None,
                        custom: MagicMock | None = None,
                        apps: MagicMock | None = None,
                        rbac: MagicMock | None = None):
    """Context manager / pytest fixture that patches
    ``kubernetes_asyncio.client.CoreV1Api`` etc. to return the provided mocks.
    """
    ...
```

Remove `build_mock_kube_client` entirely — tests that constructed `AIPerfKubeClient(mock_api)` will need to be rewritten to call free functions with a mock `ApiClient`. If any test file depends on `build_mock_kube_client`, list it in the task and migrate it as part of this step.

Run: `grep -rn "build_mock_kube_client" tests/ --include="*.py"` before editing to see the callers, then fix each.

### 5.7 tests updates

- [ ] **Step 7: Rewrite `tests/unit/kubernetes/test_logs.py`**

Swap `build_mock_kube_client` + `pod.logs(...)` mocks with `CoreV1Api.read_namespaced_pod_log` mocks. Use pattern P7 tests (release verified) and P8 tests.

- [ ] **Step 8: Rewrite `tests/unit/kubernetes/test_cli_helpers.py`**

Same mock-surface change. Test the free-function call paths now.

### 5.8 Verification

- [ ] **Step 9: Run kubernetes + cli_commands subfolders**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: green.

- [ ] **Step 10: Verify no kr8s references remain in `kubernetes/*`**

```bash
grep -rn "kr8s" src/aiperf/kubernetes/ --include="*.py" | grep -vE "kr8s_resources\.py|client\.py:.*facade"
```

Expected: empty — the only surviving kr8s reference should be in `kr8s_resources.py` (deleted in Task 8).

Note: `kubernetes/client.py` may still reference the `AIPerfKubeClient` facade class; that is intentional until Task 8.

- [ ] **Step 11: ruff + pre-commit + commit**

```bash
ruff format . && ruff check --fix .
git add src/aiperf/kubernetes/attach.py src/aiperf/kubernetes/results.py \
        src/aiperf/kubernetes/logs.py src/aiperf/kubernetes/watch_orchestrator.py \
        src/aiperf/kubernetes/watch_pollers.py tests/harness/k8s.py \
        tests/unit/kubernetes/test_logs.py tests/unit/kubernetes/test_cli_helpers.py
git commit -s -m "$(cat <<'EOF'
refactor(kubernetes): port remaining kubernetes/* modules to kubernetes_asyncio

attach, results, logs, watch_orchestrator, and watch_pollers all now
use kubernetes_asyncio directly; their public function signatures
accept an ApiClient instead of an AIPerfKubeClient. tests/harness/k8s.py
is reshaped from build_mock_kube_client to patch_api_accessors helpers.
Test files rewritten against the new mock surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Migrate CLI commands

**Files:**
- Modify: `src/aiperf/cli_commands/kube/dashboard.py`
- Modify: `src/aiperf/cli_commands/kube/debug.py`
- Modify: `src/aiperf/cli_commands/kube/list_.py`
- Modify: `src/aiperf/cli_commands/kube/logs.py`
- Modify: `src/aiperf/cli_commands/kube/profile.py`
- Modify: `src/aiperf/kubernetes/cli_helpers.py` (shared between CLI and earlier tasks — already partially migrated in Task 2; any remaining AIPerfKubeClient-caller patterns are migrated here)
- Modify: `tests/unit/cli_commands/kube/test_debug.py`
- Modify: `tests/unit/cli_commands/test_kube_helpers.py` (already partially — finish)

**Why:** CLI commands each open their own short-lived ApiClient. Migrate to `async with k8s_client(kubeconfig=..., context=...) as api:` + free-function calls.

### 6.1 Pattern for every CLI command

```python
# before
kube_client = await AIPerfKubeClient.create(kubeconfig=kc, kube_context=ctx)
jobs = await kube_client.list_jobs(namespace=ns)

# after
from aiperf.kubernetes.client import k8s_client, list_aiperf_jobs
async with k8s_client(kubeconfig=kc, context=ctx) as api:
    jobs = await list_aiperf_jobs(api, namespace=ns)
```

### 6.2 Per-file specifics

- [ ] **Step 1: `dashboard.py`**

Replace `AIPerfKubeClient.create(...)` with `async with k8s_client(...) as api:`. Anything that used `client.api` now uses `api` directly. Any explicit delete / patch calls get inlined per pattern tables.

- [ ] **Step 2: `debug.py`**

- Translate direct `import kr8s as kr8s_module` uses. Any `except kr8s_module.ServerError as e:` → `except ApiException as e:`.
- The helper that iterates pods: swap to `CoreV1Api(api).list_namespaced_pod(...).items`.
- The log-follow path uses `pod.logs(follow=True)` — translate to pattern P7 (with `raw.release()`).

- [ ] **Step 3: `list_.py`**

Replace `AIPerfKubeClient.create(...)` with `async with k8s_client(...) as api:` and call `list_aiperf_jobs(api, ...)`.

- [ ] **Step 4: `logs.py`**

- Replace `import kr8s` error handling with `ApiException`.
- `pod.logs(container=cont, follow=True, **kwargs)` → pattern P7.
- `AIPerfKubeClient.create(...)` → `k8s_client(...)`.

- [ ] **Step 5: `profile.py`**

This is the most involved CLI command — it creates Namespace, ConfigMap, Role, RoleBinding, JobSet, AIPerfJob resources.

- `from kr8s.asyncio.objects import CustomResourceDefinition` → `client.ApiextensionsV1Api(api).read_custom_resource_definition(name)`.
- `from kr8s.asyncio.objects import Namespace` then `.get`/`.create`:
  ```python
  # get
  try:
      ns = await core.read_namespace(name)
  except ApiException as e:
      if e.status == 404:
          ns = None
      else:
          raise
  # create
  await core.create_namespace(body=client.V1Namespace(
      metadata=client.V1ObjectMeta(name=name, labels=labels),
  ))
  ```
- `from kr8s.asyncio.objects import ConfigMap, Namespace, Role, RoleBinding` → use `CoreV1Api` and `RbacAuthorizationV1Api`. Bodies constructed via `client.V1ConfigMap`, `client.V1Role`, `client.V1RoleBinding` (or pass dicts — both accepted).
- `AsyncAIPerfJob.get(name, namespace=namespace, api=api)` → `CustomObjectsApi(api).get_namespaced_custom_object(group=AIPERF_JOB_GROUP, ...)`.
- `AsyncAIPerfJob(cr, api=api).create()` → `CustomObjectsApi(api).create_namespaced_custom_object(group=AIPERF_JOB_GROUP, version=..., plural=..., namespace=ns, body=cr)`.
- `resource_mapping = {"JobSet": AsyncJobSet}` → replace with a helper that dispatches on kind-string to `CustomObjectsApi`/`CoreV1Api` calls.

### 6.3 Verification

- [ ] **Step 6: Run cli_commands tests**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: green.

- [ ] **Step 7: Manual smoke test (optional, cluster-dependent)**

If a kind/minikube cluster is available:
```bash
unset VIRTUAL_ENV && uv run --active aiperf kube list
unset VIRTUAL_ENV && uv run --active aiperf kube debug --help
```

If no cluster is available, skip — document in the commit body.

- [ ] **Step 8: Verify no `kr8s` left in cli_commands/kube or cli_helpers**

```bash
grep -rn "kr8s" src/aiperf/cli_commands/ src/aiperf/kubernetes/cli_helpers.py --include="*.py"
```

Expected: empty.

- [ ] **Step 9: ruff + pre-commit + commit**

```bash
ruff format . && ruff check --fix .
git add src/aiperf/cli_commands/kube/ src/aiperf/kubernetes/cli_helpers.py \
        tests/unit/cli_commands/
git commit -s -m "$(cat <<'EOF'
refactor(cli): port kube commands to kubernetes_asyncio

profile, debug, logs, dashboard, and list commands now open their own
ApiClient via k8s_client() and call the free functions directly.
CustomObjectsApi replaces AsyncAIPerfJob/AsyncJobSet; CoreV1Api / RbacV1
replaces ConfigMap/Role/RoleBinding/Namespace helpers; log-follow paths
use read_namespaced_pod_log(follow=True, _preload_content=False) with
an explicit raw.release() in the finally block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migrate operator, controller, server_metrics, api/progress

**Files:**
- Modify: `src/aiperf/operator/handlers/create.py`
- Modify: `src/aiperf/operator/handlers/monitor.py`
- Modify: `src/aiperf/operator/handlers/lifecycle.py`
- Modify: `src/aiperf/operator/handlers/completion.py`
- Modify: `src/aiperf/operator/preflight.py` (~1191 lines — apply preflight.py translation table)
- Modify: `src/aiperf/operator/client_cache.py`
- Modify: `src/aiperf/operator/k8s_helpers.py`
- Modify: `src/aiperf/operator/routers/jobs.py`
- Modify: `src/aiperf/operator/results_server.py`
- Modify: `src/aiperf/controller/kubernetes_service_manager.py`
- Modify: `src/aiperf/server_metrics/discovery/kubernetes.py`
- Modify: `src/aiperf/api/routers/progress.py`
- Rewrite tests: `tests/unit/operator/test_main.py`, `tests/unit/operator/test_preflight.py`, `tests/unit/operator/test_completion_claim.py`, `tests/unit/server_metrics/test_kubernetes_discovery.py`

**Why:** This is the largest task — it migrates every remaining external caller of AIPerfKubeClient and every remaining `import kr8s` outside the `kubernetes/*` package. At end of Task 7, the only surviving `kr8s` references are in `kubernetes/kr8s_resources.py` (unused) and `pyproject.toml` (still a dep). Task 8 cleans both.

### 7.1 Operator handlers — key patterns

- [ ] **Step 1: `operator/handlers/create.py`**

- `from kr8s.asyncio.objects import ConfigMap, Role, RoleBinding` → `from kubernetes_asyncio import client` and use `CoreV1Api` / `RbacAuthorizationV1Api`.
- `create_idempotent(AsyncJobSet, jobset, api)` — the `create_idempotent` helper currently takes a kr8s class; change its signature to take `(api, group, version, plural, body, namespace, logger)` and call `CustomObjectsApi.create_namespaced_custom_object(...)` internally, catching `ApiException` with `e.status == 409` for the idempotency path.
- `kopf` decorator body stays unchanged in form (`@kopf.on.create(...)`) — only the API calls inside change.

- [ ] **Step 2: `operator/handlers/monitor.py`**

- The ~5 `AsyncJobSet.get(jobset_name, namespace=namespace, api=api)` calls → `CustomObjectsApi.get_namespaced_custom_object(group=JOBSET_GROUP, ...)`.
- `except kr8s.NotFoundError:` / `except kr8s.ServerError as e:` → `except ApiException as e: if e.status == 404: ... else: raise`.
- `api: kr8s.Api` → `api: ApiClient`.
- The `from kr8s.asyncio.objects import Pod` at top — delete; use `CoreV1Api` inline.
- The `AsyncAIPerfJob.get(name, namespace=namespace, api=api)` at line ~203 → `CustomObjectsApi.get_namespaced_custom_object(group=AIPERF_JOB_GROUP, ...)`.
- Status subresource patches via `await js.patch({...}, type="merge")` → `custom.patch_namespaced_custom_object_status(...)` (or `patch_namespaced_custom_object` if it's a spec patch).
- Annotation patches similarly.

- [ ] **Step 3: `operator/handlers/lifecycle.py`**

- `from aiperf.kubernetes.kr8s_resources import AsyncJobSet` → import `JOBSET_*` from `cr_refs` and use `CustomObjectsApi`.
- `AsyncJobSet.get(jobset_name, namespace=namespace, api=api)` → `CustomObjectsApi(api).get_namespaced_custom_object(...)`.
- `import kr8s` / error types → `ApiException`.

- [ ] **Step 4: `operator/handlers/completion.py`**

- Same pattern as lifecycle + monitor.

- [ ] **Step 5: `operator/preflight.py` (~1191 lines)**

Apply the **same translation table from Task 3** (namespace/deployment/ServiceAccount/Node/ResourceQuota/Secret/NetworkPolicy/Service lookups). Several helpers need near-identical translation; work file-by-file, function-by-function. Keep function signatures stable.

- [ ] **Step 6: `operator/client_cache.py`**

- `from aiperf.kubernetes.client import get_api` — the old function does not exist on the new client.py. Replace the lazy-import of `get_api` + `AsyncAIPerfJob` with a direct call path:
  ```python
  # before
  from aiperf.kubernetes.client import get_api
  from aiperf.kubernetes.kr8s_resources import AsyncAIPerfJob
  api = await get_api()
  obj = await AsyncAIPerfJob.get(name, namespace=namespace, api=api)
  await obj.patch(patch_ops, type="json")

  # after
  from aiperf.kubernetes.client import k8s_client
  from aiperf.kubernetes.cr_refs import (
      AIPERF_JOB_GROUP, AIPERF_JOB_VERSION, AIPERF_JOB_PLURAL,
  )
  async with k8s_client() as api:
      custom = client.CustomObjectsApi(api)
      await custom.patch_namespaced_custom_object(
          group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
          plural=AIPERF_JOB_PLURAL, namespace=namespace, name=name,
          body=patch_ops,
          _content_type="application/json-patch+json",
      )
  ```
- Update the `except kr8s.ServerError as e: status_code = e.response.status_code if e.response else 0` block to `except ApiException as e: status_code = e.status or 0`.

- [ ] **Step 7: `operator/k8s_helpers.py`**

Translate each kr8s lookup per pattern tables. These helpers likely match preflight idioms.

- [ ] **Step 8: `operator/routers/jobs.py`**

Change function signature `client_holder: list[AIPerfKubeClient | None] | None` to `api_holder: list[ApiClient | None] | None`. Inside the route, replace `client.list_jobs(...)` with `list_aiperf_jobs(api, ...)`.

- [ ] **Step 9: `operator/results_server.py`**

Replace `AIPerfKubeClient.create()` with the explicit lifetime pattern from spec §3.4 Pattern B (manual load + ApiClient + close on shutdown). Example:

```python
try:
    config.load_incluster_config()
except config.ConfigException:
    await config.load_kube_config()
self._api = ApiClient()
# ...on shutdown:
await self._api.close()
```

The `kube_client_holder[0] = await AIPerfKubeClient.create()` pattern must be replaced — store the ApiClient directly in the holder.

- [ ] **Step 10: `controller/kubernetes_service_manager.py`**

- `from aiperf.kubernetes.client import AIPerfKubeClient` → `from aiperf.kubernetes.client import k8s_client` + specific free functions.
- `self._kube_client = await AIPerfKubeClient.create()` → module-scoped `ApiClient` with explicit `__aenter__`/`__aexit__` usage, or `async with k8s_client() as api:` scoped to the operation if short.
- `async def _get_kube_client(self) -> AIPerfKubeClient` → `async def _get_api(self) -> ApiClient`.
- Callers inside this file that did `client.method(...)` → free function calls.

- [ ] **Step 11: `server_metrics/discovery/kubernetes.py`**

- `import kr8s` / `kr8s.asyncio.api()` → `k8s_client()` or direct `ApiClient()` per lifetime.
- Swap kr8s object access to CoreV1Api / AppsV1Api typed calls.

- [ ] **Step 12: `api/routers/progress.py`**

- `from aiperf.kubernetes.kr8s_resources import AsyncJobSet` → `from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_VERSION, JOBSET_PLURAL`.
- `await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)` → `await CustomObjectsApi(api).get_namespaced_custom_object(group=JOBSET_GROUP, version=JOBSET_VERSION, plural=JOBSET_PLURAL, namespace=namespace, name=jobset_name)`.

### 7.2 Tests

- [ ] **Step 13: Rewrite `tests/unit/operator/test_main.py`**

This is the largest operator test. 3 separate `import kr8s` blocks at lines 852, 1430, 3363. Swap all kr8s mocks per pattern P10.

Strategy: grep the file for `kr8s` first, then migrate each test in isolation. Many tests use the handler under test plus mocked API — re-point mocks at `CoreV1Api` / `CustomObjectsApi` as per the handler's new internals.

- [ ] **Step 14: Rewrite `tests/unit/operator/test_preflight.py`**

Apply the same mock migration used in `tests/unit/kubernetes/test_preflight.py` (Task 3).

- [ ] **Step 15: Rewrite `tests/unit/operator/test_completion_claim.py`**

Mocks the JSON-patch path. Replace `kr8s.ServerError(response=mock_response)` with `ApiException(status=409)` (or 422 for the test-op collision case).

- [ ] **Step 16: Rewrite `tests/unit/server_metrics/test_kubernetes_discovery.py`**

Swap kr8s mocks. Two sites (lines 429, 465).

### 7.3 Verification

- [ ] **Step 17: Run the three subfolders separately**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: green.

- [ ] **Step 18: Verify kr8s imports only remain in kr8s_resources.py + kr8s facade methods**

```bash
grep -rn "from kr8s\|import kr8s" src/ --include="*.py" | grep -vE "kr8s_resources\.py"
```

Expected: empty.

```bash
grep -rn "AsyncAIPerfJob\|AsyncJobSet" src/ --include="*.py" | grep -vE "kr8s_resources\.py"
```

Expected: empty (all CR references now use cr_refs constants).

- [ ] **Step 19: ruff + pre-commit + commit**

```bash
ruff format . && ruff check --fix .
git add src/aiperf/operator/ src/aiperf/controller/ src/aiperf/server_metrics/ \
        src/aiperf/api/routers/progress.py \
        tests/unit/operator/ tests/unit/server_metrics/
git commit -s -m "$(cat <<'EOF'
refactor(operator): port operator/controller/server_metrics to kubernetes_asyncio

Operator handlers (create, monitor, lifecycle, completion), preflight,
client_cache, k8s_helpers, routers/jobs, results_server — plus
controller/kubernetes_service_manager, server_metrics/discovery/kubernetes,
and api/routers/progress — all now use kubernetes_asyncio directly.
CR reads/writes go through CustomObjectsApi with cr_refs constants;
built-in resources through CoreV1Api/AppsV1Api/RbacAuthorizationV1Api
typed clients. ApiException replaces kr8s.NotFoundError/ServerError
throughout. kopf decorators are unchanged in form.

Tests rewritten for the new mock surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Delete `AIPerfKubeClient` facade, `kr8s_resources.py`, drop `kr8s` dep

**Files:**
- Modify: `src/aiperf/kubernetes/client.py` (delete `AIPerfKubeClient` class)
- Delete: `src/aiperf/kubernetes/kr8s_resources.py`
- Delete: `tests/unit/kubernetes/test_kr8s_resources.py`
- Modify: `pyproject.toml` (remove `kr8s>=0.20.15`)
- Regenerate: `uv.lock`
- Modify: `src/aiperf/common/noisy_loggers.py` (drop `httpx` from `_NOISY_LOGGERS` since kr8s is gone)
- Final: grep sweep + pre-commit

**Why:** No code in the project uses `AIPerfKubeClient` or kr8s anymore after Task 7. Remove the facade, delete the orphan CR-wrapper module, drop the dependency.

- [ ] **Step 1: Final grep confirmation — no AIPerfKubeClient callers**

```bash
grep -rn "AIPerfKubeClient" src tests --include="*.py"
```

Expected: the only hits should be inside `src/aiperf/kubernetes/client.py` (class definition). If there are external callers, STOP and go back to Task 5/6/7 to finish migration.

- [ ] **Step 2: Delete `AIPerfKubeClient` class from `kubernetes/client.py`**

Edit `src/aiperf/kubernetes/client.py` and remove the entire class block at the bottom (`class AIPerfKubeClient:` through the end of its methods). Leave the free functions intact. Update the module docstring to remove the "facade" paragraph.

- [ ] **Step 3: Delete `src/aiperf/kubernetes/kr8s_resources.py`**

```bash
git rm src/aiperf/kubernetes/kr8s_resources.py
```

- [ ] **Step 4: Delete `tests/unit/kubernetes/test_kr8s_resources.py`**

```bash
git rm tests/unit/kubernetes/test_kr8s_resources.py
```

- [ ] **Step 5: Remove `kr8s` from `pyproject.toml`**

```diff
-  "kr8s>=0.20.15",
```

- [ ] **Step 6: Regenerate lockfile**

```bash
unset VIRTUAL_ENV && uv lock
```

Expected: `kr8s` entries gone; `httpx` may also drop if it had no other reverse dependencies (fine).

- [ ] **Step 7: Simplify `noisy_loggers.py`**

Drop `httpx` from `_NOISY_LOGGERS` (no longer in the project). Update docstring.

```python
_NOISY_LOGGERS = (
    "aiohttp.access",
    "aiohttp.client",
    "kubernetes_asyncio.client.rest",
)
```

- [ ] **Step 8: Reinstall to pick up removed dep**

```bash
unset VIRTUAL_ENV && make install
```

- [ ] **Step 9: Run full test suites, subfolder by subfolder**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: all green.

- [ ] **Step 10: Final grep sweep — zero kr8s**

```bash
grep -rn "kr8s\|AsyncAIPerfJob\|AsyncJobSet\|AIPerfKubeClient" src tests --include="*.py"
```

Expected: empty.

```bash
grep -rn "^kr8s" pyproject.toml uv.lock
# expected: empty
```

- [ ] **Step 11: Verify Python import works cleanly**

```bash
unset VIRTUAL_ENV && uv run --active python -c "
import aiperf
from aiperf.kubernetes.client import k8s_client, list_aiperf_jobs, find_jobset
from aiperf.kubernetes.cr_refs import AIPERF_JOB_GROUP
try:
    import kr8s
except ImportError as e:
    print('kr8s confirmed gone:', e)
"
```

Expected: the imports succeed and `kr8s` import raises `ModuleNotFoundError`.

- [ ] **Step 12: ruff + pre-commit + make validate-plugin-schemas**

```bash
ruff format . && ruff check --fix .
make validate-plugin-schemas
```

Expected: all pass.

- [ ] **Step 13: Integration test smoke (if cluster available)**

If a live kind / minikube cluster exists:
```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/kubernetes/ -q --no-header -x
```

Expected: green. If no cluster is available, skip and rely on CI.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -s -m "$(cat <<'EOF'
chore: remove kr8s dependency and AIPerfKubeClient facade

All callers migrated to kubernetes_asyncio free functions in prior
commits. Delete the AIPerfKubeClient class, the kr8s_resources CR
wrappers, and the corresponding test file. Drop kr8s from
pyproject.toml and regenerate uv.lock. Simplify noisy_loggers to
drop the now-unused httpx entry.

Migration complete: no kr8s imports remain in src/ or tests/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (post-Task 8)

- [ ] **Final Step 1: Full unit test run (all subfolders)**

```bash
unset VIRTUAL_ENV && uv run --active pytest -n auto tests/unit/ -q --no-header
```

Expected: full suite green, count ≈ 1445 or higher.

- [ ] **Final Step 2: Optional component_integration + integration**

```bash
unset VIRTUAL_ENV && uv run --active pytest -m component_integration -n auto
unset VIRTUAL_ENV && uv run --active pytest -m integration -n auto
```

Gate on local cluster availability.

- [ ] **Final Step 3: Make targets**

```bash
make validate-plugin-schemas
make generate-all-docs  # confirm no CLI doc drift
```

- [ ] **Final Step 4: Open PR**

Per CLAUDE.md workflow, open a PR from `worktree-ajc+remove-kr8s` → `main` with the spec link in the description.

```bash
gh pr create --title "Replace kr8s with kubernetes_asyncio" --body "$(cat <<'EOF'
## Summary

Replace every `kr8s` call site with `kubernetes_asyncio`; dissolve the
`AIPerfKubeClient` wrapper into free functions whose bodies call
`CoreV1Api` / `CustomObjectsApi` / `AppsV1Api` directly. Primary driver:
LLM-native readability — code now reads like the upstream kubernetes
client examples.

Design: `docs/superpowers/specs/2026-04-21-replace-kr8s-with-kubernetes-asyncio-design.md`
Plan: `docs/superpowers/plans/2026-04-21-replace-kr8s-with-kubernetes-asyncio.md`

## Test plan

- [ ] `make first-time-setup`
- [ ] `uv run pytest -n auto tests/unit/kubernetes`
- [ ] `uv run pytest -n auto tests/unit/operator`
- [ ] `uv run pytest -n auto tests/unit/cli_commands`
- [ ] `uv run pytest -n auto tests/unit/`
- [ ] `ruff format . && ruff check --fix .`
- [ ] `pre-commit run --all-files`
- [ ] (If cluster) `uv run pytest -m integration -n auto`
- [ ] Deploy operator pod in a dev cluster; create an AIPerfJob; verify status transitions end-to-end

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

After writing every task, skim the spec and confirm every requirement has a matching plan task:

- **Spec §2 In scope → Covered:**
  - `src/aiperf/kubernetes/*` — Tasks 2, 3, 4, 5
  - `src/aiperf/operator/*` — Task 7
  - `src/aiperf/cli_commands/kube/*` — Task 6
  - `src/aiperf/api/routers/progress.py` — Task 7
  - `src/aiperf/server_metrics/discovery/kubernetes.py` — Task 7
  - `src/aiperf/common/noisy_loggers.py` — Task 1
  - Tests (harness, unit/*) — Tasks 2-7
  - `pyproject.toml` / `uv.lock` — Tasks 1, 8

- **Additional (not in spec but discovered during planning):**
  - `src/aiperf/controller/kubernetes_service_manager.py` — Task 7
  - `src/aiperf/kubernetes/results.py`, `logs.py`, `watch_orchestrator.py`, `watch_pollers.py` — Task 5
  - `src/aiperf/cli_commands/kube/dashboard.py`, `list_.py` — Task 6

- **Spec §3.2 patterns → Covered in Patterns reference + all tasks refer to it.**

- **Spec §6 risks → All relevant:**
  - kopf / aiohttp cohabitation — Task 1 (lock verification)
  - Config loading — Task 2 design
  - Integration cluster — Final Step 2
  - Log-stream release — Patterns P7, used in Tasks 5, 6
  - `_kr8s_kwargs` workaround dropped — naturally absent after Task 2

- **Spec §7 non-goals → All honored:**
  - No DynamicClient — plan uses CustomObjectsApi only
  - No is_not_found helper — inline `if e.status == 404:` per pattern P4/P6
  - No watch-stream translation — watch pollers stay list-based in Task 5
  - No kopf / aiohttp changes — Task 7 keeps kopf decorators as-is
