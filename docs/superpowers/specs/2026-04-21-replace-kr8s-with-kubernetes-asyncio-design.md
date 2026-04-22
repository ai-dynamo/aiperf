# Replace `kr8s` with `kubernetes_asyncio`

Date: 2026-04-21
Branch: `ajc/k8s` (implementation will happen in a worktree branched from here)

## 1. Goal and motivation

Remove the `kr8s` Python Kubernetes library from AIPerf and rewrite every call
site on top of `kubernetes_asyncio` (the async port of the official
`kubernetes-client/python` SDK).

Primary driver: **LLM-native readability.** LLMs have been trained extensively
on the official Kubernetes Python client's OpenAPI-generated API surface
(`CoreV1Api.list_namespaced_pod`, `CustomObjectsApi.patch_namespaced_custom_object_status`,
`V1Pod.status.phase`, etc.). `kr8s` invents its own abstraction layer (custom
resource classes via `new_class`, `api.async_get(...)` iteration, `kr8s.ALL`
sentinel, bespoke `ServerError` / `NotFoundError` types) that LLMs have to learn
on the fly. Consolidating to `kubernetes_asyncio` — whose method names and
model types match the widely-trained sync client byte-for-byte — makes the
code maximally recognizable.

Secondary driver: dependency consolidation. `kopf>=1.42.0` (operator framework,
staying) does not share its internal client with handler code, so there is no
operational coupling between kopf and our k8s client choice. But we already
pull `kubernetes_asyncio` in transitively via other parts of the Python
ecosystem — pinning it explicitly and removing `kr8s` means one library instead
of two.

This is **not** driven by a specific `kr8s` bug. `kr8s` works; we just want
less code and more LLM-native code.

## 2. Scope

### In scope

Every direct `kr8s` usage across production and tests:
- `src/aiperf/kubernetes/` — `client.py`, `kr8s_resources.py`, `preflight.py`,
  `preflight_utils.py`, `watchdog.py`, `attach.py`, `cli_helpers.py`
- `src/aiperf/operator/` — `handlers/{create,monitor,lifecycle,completion}.py`,
  `preflight.py`, `k8s_helpers.py`, `client_cache.py`, `routers/jobs.py`,
  `results_server.py`
- `src/aiperf/cli_commands/kube/` — `logs.py`, `profile.py`, `debug.py`
- `src/aiperf/api/routers/progress.py`
- `src/aiperf/server_metrics/discovery/kubernetes.py`
- `src/aiperf/common/noisy_loggers.py`
- Tests: `tests/harness/k8s.py`, `tests/unit/kubernetes/*`,
  `tests/unit/operator/*`, `tests/kubernetes/*`
- `pyproject.toml` — drop `kr8s>=0.20.15`, add explicit `kubernetes_asyncio`

Approximate surface area: ~66 `kr8s` import sites, ~109 references to
`kr8s.NotFoundError` / `kr8s.ServerError` / `kr8s.ALL`.

### Not in scope (stay as-is)

- **`kopf` decorators and operator wiring.** Every `@kopf.on.*` decorator,
  finalizer flow, status/annotation patch through kopf, and event loop stays
  unchanged. Only the *API calls inside handlers* change.
- **`kubernetes/port_forward.py`** — already a `kubectl` subprocess wrapper,
  no `kr8s` involvement.
- **`kubernetes/watch_*.py`** (`watch_orchestrator`, `watch_pollers`,
  `watch_diagnosis`, `watch_models`, `watch_render_*`) — list-polling based,
  never used `kr8s.watch()`; the only impact is the pollers calling
  `kubernetes_asyncio` instead of `kr8s` when they list resources.
- **`kubernetes/models.py`**, **`k8s_models.py`**, **`constants.py`**,
  **`enums.py`** — already pure Pydantic / string constants; no library coupling.

## 3. Design

### 3.1 Module layout

**Delete:**

- `src/aiperf/kubernetes/kr8s_resources.py` — the `new_class()` CRD wrappers
  (`AsyncAIPerfJob`, `AsyncJobSet`) go away. CRDs are accessed via
  `CustomObjectsApi` using the group/version/plural constants defined below.
- The `AIPerfKubeClient` class inside `kubernetes/client.py` — dissolved into
  free module functions (see §3.3).

**Add:**

- `src/aiperf/kubernetes/cr_refs.py` — six string constants identifying the
  two CRDs AIPerf works with:

  ```python
  # AIPerfJob CR coordinates for CustomObjectsApi
  AIPERF_JOB_GROUP = "aiperf.nvidia.com"
  AIPERF_JOB_VERSION = "v1alpha1"
  AIPERF_JOB_PLURAL = "aiperfjobs"

  # JobSet CR coordinates
  JOBSET_GROUP = "jobset.x-k8s.io"
  JOBSET_VERSION = "v1alpha2"
  JOBSET_PLURAL = "jobsets"
  ```

  These appear as explicit keyword arguments at every `CustomObjectsApi` call
  site so the reader always sees which CR is being touched.

**Rewrite (public shape preserved where used across packages):**

- `src/aiperf/kubernetes/client.py` — replaces `AIPerfKubeClient` with a
  functional module (see §3.3).
- `src/aiperf/kubernetes/watchdog.py` — `Kr8sWatchdogSource` renames to
  `K8sWatchdogSource`; internals translated. Consumer (`BenchmarkWatchdog`,
  the source-injection abstraction) is unchanged.
- Every call site listed in §2 gets its imports and call shape translated.

**Kept unchanged:**

- `src/aiperf/kubernetes/models.py` — `AIPerfJobCR.model_validate(raw)`,
  `JobSetInfo.from_raw(raw)`, `AIPerfJobInfo`, `PodSummary` — all pure Pydantic.
  The only change is that the raw dict now comes from
  `CustomObjectsApi.list_namespaced_custom_object(...)` instead of
  `kr8s_obj.raw`.
- `src/aiperf/kubernetes/k8s_models.py`, `constants.py`, `enums.py`,
  `port_forward.py`, all `watch_*` modules, `console.py`.

### 3.2 Canonical patterns (LLM-native idioms)

These are the shapes every new call site adopts. They match the upstream
`kubernetes_asyncio` examples and the well-known sync `kubernetes` client
patterns, which is what makes them LLM-native.

**Config loading** — note the sync/async asymmetry:

```python
from kubernetes_asyncio import config

try:
    config.load_incluster_config()              # sync — no await
except config.ConfigException:
    await config.load_kube_config(              # async
        config_file=kubeconfig, context=context,
    )
```

**ApiClient as an async context manager:**

```python
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient

async with ApiClient() as api:
    core = client.CoreV1Api(api)
    pods = (await core.list_namespaced_pod(ns, label_selector=sel)).items
```

**Custom resource access via `CustomObjectsApi` + existing Pydantic:**

```python
custom = client.CustomObjectsApi(api)
raw = await custom.get_namespaced_custom_object(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
)
job = AIPerfJobCR.model_validate(raw)  # existing, unchanged

# Status subresource merge patch
await custom.patch_namespaced_custom_object_status(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    body={"status": {"phase": "Running"}},
)

# JSON patch (for try_claim_completion's test-op claim)
await custom.patch_namespaced_custom_object(
    group=AIPERF_JOB_GROUP, version=AIPERF_JOB_VERSION,
    plural=AIPERF_JOB_PLURAL, namespace=ns, name=name,
    body=patch_ops,                        # list of json-patch ops
    _content_type="application/json-patch+json",
)
```

**Error handling** — idiomatic `ApiException` + `e.status` branch (no helper
functions, no custom hierarchy):

```python
from kubernetes_asyncio.client.exceptions import ApiException

try:
    raw = await custom.get_namespaced_custom_object(...)
except ApiException as e:
    if e.status == 404:
        return None
    raise
```

For the JSON-patch optimistic-concurrency case in `try_claim_completion`,
`e.status in (409, 422)` maps the current kr8s error-response shape 1:1.

**Log streaming** — must explicitly release the underlying aiohttp response
(this is the single biggest connection-leak footgun in `kubernetes_asyncio`):

```python
raw = await core.read_namespaced_pod_log(
    name=pod, namespace=ns, container=cont,
    follow=True, _preload_content=False,
)
try:
    async for line in raw.content:
        yield line.decode()
finally:
    await raw.release()
```

One-shot (non-follow) log reads stay simple: `await core.read_namespaced_pod_log(name, ns, tail_lines=N)`.

**Built-in resource access** — always typed attribute access, no more raw
dict walking:

```python
for pod in pods:
    phase = pod.status.phase
    statuses = pod.status.container_statuses or []
    ready = bool(statuses) and all(cs.ready for cs in statuses)
```

### 3.3 `kubernetes/client.py` final shape

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes.constants import JobSetLabels, Labels


@asynccontextmanager
async def k8s_client(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> AsyncIterator[ApiClient]:
    """Load k8s config and yield an ApiClient.

    In-cluster first, kubeconfig fallback. ApiClient is closed on exit.
    """
    try:
        config.load_incluster_config()
    except config.ConfigException:
        await config.load_kube_config(config_file=kubeconfig, context=context)
    api = ApiClient()
    try:
        yield api
    finally:
        await api.close()


# -- Label selectors (pure strings, reusable) --------------------------------

def job_selector(job_id: str) -> str:
    return f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id}"


def controller_selector(job_id: str) -> str:
    return (
        f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id},"
        f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
    )


# -- AIPerfJob CR helpers -----------------------------------------------------
#
# These encode AIPerf-specific multi-step semantics (fallback lookups, status
# filtering, sort order) that aren't part of the K8s API surface. They all
# take an ApiClient explicitly and call CustomObjectsApi(api) inline so the
# reader can see the canonical method names.

# Bodies call CustomObjectsApi(api)/CoreV1Api(api) inline. Signatures mirror
# the current AIPerfKubeClient methods; callers pass `api` where they used to
# pass the client instance.

async def list_aiperf_jobs(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    status_filter: str | None = None,
) -> list[AIPerfJobInfo]: ...

async def find_aiperf_job(
    api: ApiClient,
    name: str,
    namespace: str | None = None,
) -> AIPerfJobInfo | None: ...

async def list_jobsets(
    api: ApiClient,
    namespace: str | None = None,
    all_namespaces: bool = False,
    job_id: str | None = None,
    status_filter: str | None = None,
) -> list[JobSetInfo]: ...

async def find_jobset(
    api: ApiClient,
    job_id: str,
    namespace: str | None = None,
) -> JobSetInfo | None: ...

async def delete_jobset(api: ApiClient, name: str, namespace: str) -> None: ...

async def get_pod_summary(
    api: ApiClient, jobset_name: str, namespace: str,
) -> PodSummary: ...

async def find_controller_pod(
    api: ApiClient, namespace: str, job_id: str,
) -> tuple[str, PodPhase] | None: ...

async def wait_for_controller_pod_ready(
    api: ApiClient, namespace: str, job_id: str, timeout: int = 300,
) -> str: ...
```

**Dissolved at call sites** (were thin single-method wrappers — inline the
kubernetes_asyncio call instead):

- `AIPerfKubeClient.cancel_job` → `CustomObjectsApi(api).patch_namespaced_custom_object(...)` (2 call sites)
- `AIPerfKubeClient.get_raw_status` → `custom.get_namespaced_custom_object(...)` + dict index
- `AIPerfKubeClient.get_pods` → `core.list_namespaced_pod(...)`
- `AIPerfKubeClient.delete_namespace` → `core.delete_namespace(...)`
- `AIPerfKubeClient.version` → `client.VersionApi(api).get_code()`
- `AIPerfKubeClient.find_operator_pod` — `core.list_namespaced_pod` + first-item inline
- `AIPerfKubeClient.find_retrievable_pod` — composes `find_controller_pod` + a phase check, can stay as a free function if >1 caller, else inline

### 3.4 ApiClient lifetime by caller

- **CLI commands** — `async with k8s_client(...) as api:` per command. Short-lived.
- **Kopf operator handlers** — `async with k8s_client() as api:` inside each
  handler. (Research note: kopf's internal client is aiohttp-based, not
  `kubernetes_asyncio`, and kopf declined to share it with handlers
  [nolar/kopf#366]. Per-handler open is the idiomatic pattern.) Config loading
  is cheap (it's cached at the module level by the underlying
  `kubernetes_asyncio.configuration.Configuration` singleton once set).
- **Long-running services** — `results_server.py`, `watchdog.py`,
  `server_metrics/discovery/kubernetes.py`: hold an open ApiClient for the
  service lifetime. Two equivalent patterns, pick per call site:

  ```python
  # Pattern A — enter the context manager manually in start(), exit in stop():
  self._api_cm = k8s_client(kubeconfig=..., context=...)
  self._api = await self._api_cm.__aenter__()
  ...
  await self._api_cm.__aexit__(None, None, None)

  # Pattern B — construct without the helper:
  try:
      config.load_incluster_config()
  except config.ConfigException:
      await config.load_kube_config(config_file=..., context=...)
  self._api = ApiClient()
  ...
  await self._api.close()
  ```

  Pattern B is slightly more LLM-native (explicit); Pattern A is less code.
  Prefer whichever matches the surrounding service's existing lifecycle shape.

### 3.5 Watchdog source translation

`Kr8sWatchdogSource` → `K8sWatchdogSource` — rename and swap internals. The
class is already the right abstraction (pod/event/node/namespace/logs
source), so no consumer changes.

| Method | New implementation |
|---|---|
| `list_pods(ns)` | `core.list_namespaced_pod(ns).items` |
| `list_events(ns)` | `core.list_namespaced_event(ns).items` |
| `list_nodes()` | `core.list_node().items` |
| `list_namespaces()` | `core.list_namespace().items` |
| `pod_logs(ns, name, tail)` | `core.read_namespaced_pod_log(name, ns, tail_lines=tail)` |

Consumers (`BenchmarkWatchdog`, diagnostic reasoning) continue to see
`V1Pod` / `V1Event` / `V1Node` attribute access. This is a **behavior-preserving
refactor**: the shape of what the watchdog sees is equivalent; `.raw[...]` dict
walks become attribute accesses.

### 3.6 Kopf interop

Kopf handlers open their own ApiClient. Example — a status handler that
needs to read a ConfigMap:

```python
@kopf.on.update(AIPERF_JOB_GROUP, AIPERF_JOB_VERSION, AIPERF_JOB_PLURAL)
async def handle_update(spec, meta, status, **kwargs):
    async with k8s_client() as api:
        core = client.CoreV1Api(api)
        cm = await core.read_namespaced_config_map(f"{meta['name']}-config", meta["namespace"])
        ...
```

Kopf's own CR reads/writes (status subresource patches it does on our
behalf) remain its responsibility and are unaffected.

### 3.7 Test strategy

`tests/unit/kubernetes/` and `tests/unit/operator/` currently mock
`kr8s.Api.async_get(...)` returns and per-object `.raw` / `.patch` /
`.delete`. The mock surface becomes `CoreV1Api` and `CustomObjectsApi`:

- Replace async generators with `AsyncMock(return_value=V1PodList(items=[...]))`
  (typed) or `AsyncMock(return_value={"items": [raw_dict, ...]})` for CRD lists.
- Raise `ApiException(status=404)` instead of `kr8s.NotFoundError`.
- `tests/harness/k8s.py` is the shared fake. Refactor it into a minimal fake
  `ApiClient` whose constructed typed API objects (`CoreV1Api`,
  `CustomObjectsApi`, ...) are `AsyncMock`s with recorded calls.

Integration tests under `tests/kubernetes/` (real cluster — kind/minikube)
keep the same shape; only library imports change.

### 3.8 `noisy_loggers.py`

Current file suppresses `httpx` noise (kr8s's HTTP library). Update to
suppress `kubernetes_asyncio` / `aiohttp.access` / `aiohttp.client` noise
instead. Docstring updated accordingly.

## 4. Dependencies

**`pyproject.toml`:**

```diff
-  "kr8s>=0.20.15",
+  "kubernetes_asyncio>=X.Y,<Z",   # pin at implementation time
```

Pin rationale:

- Read `kopf>=1.42.0`'s installed `kubernetes_asyncio` range (e.g.
  `pip show kubernetes_asyncio` or inspect `uv.lock` before editing).
- Pin to the latest minor series inside that range (e.g. if kopf allows
  `>=24,<33`, pin `kubernetes_asyncio>=32.0,<33.0`).
- Run `uv lock` and confirm a single resolved version exists in `uv.lock`.

`uv.lock` is regenerated in the same commit that removes `kr8s`.

## 5. Commit plan (incremental within one branch)

Branch: `ajc/k8s-remove-kr8s` (or similar), branched from `ajc/k8s` HEAD.
Each commit passes `uv run pytest tests/unit/ -n auto`, `ruff format . && ruff check --fix .`,
`pre-commit run --all-files`, `make validate-plugin-schemas`.

| # | Commit | Scope |
|---|---|---|
| 1 | `feat(deps): add kubernetes_asyncio, add cr_refs, update noisy_loggers` | Introduce new constants + dep; no behavior change yet |
| 2 | `refactor(kubernetes): rewrite client.py on kubernetes_asyncio` | Dissolve `AIPerfKubeClient`; port its domain helpers to free functions; update `tests/unit/kubernetes/test_client.py`; migrate `kubernetes/cli_helpers.py` callers |
| 3 | `refactor(kubernetes): port preflight.py and preflight_utils.py` | Self-contained surface; rewrite `tests/unit/kubernetes/test_preflight.py` and `tests/unit/cli_commands/test_kube_preflight.py` |
| 4 | `refactor(kubernetes): rename Kr8sWatchdogSource → K8sWatchdogSource` | Translate watchdog source internals; rewrite `tests/unit/kubernetes/test_watchdog.py` |
| 5 | `refactor(cli): port attach, profile, logs, debug to kubernetes_asyncio` | CLI surface; rewrite `tests/unit/cli_commands/kube/test_debug.py`, `tests/unit/kubernetes/test_logs.py`, `tests/unit/cli_commands/test_kube_helpers.py` |
| 6 | `refactor(operator): port handlers, preflight, client_cache, routers` | Operator surface; rewrite `tests/unit/operator/*`; also `server_metrics/discovery/kubernetes.py` and `api/routers/progress.py` |
| 7 | `chore: remove kr8s dependency` | Delete `kr8s_resources.py`, drop `kr8s` from `pyproject.toml`, regenerate `uv.lock`; `grep -r 'kr8s' src tests` returns empty; final `pre-commit run --all-files` |

Integration tests under `tests/kubernetes/` run at commits 4 and 7 at minimum.

## 6. Risks and open items

- **kopf compat.** Confirm at lock time that `kopf>=1.42.0` co-installs
  cleanly with the pinned `kubernetes_asyncio` version. If kopf pins a
  different range, widen our constraint to the intersection.
- **Config loading interaction.** kopf runs its own config loading at
  process start; our `k8s_client()` loads config again on first call. This
  is redundant but harmless: the underlying `Configuration` singleton is
  cached. If we observe duplicate load warnings, add a module-level flag to
  load once.
- **Integration test cluster availability.** `tests/kubernetes/test_operator.py`
  and `tests/kubernetes/test_kueue_integration.py` require a live cluster.
  Gated at commit 7 (treated as blocker before merge). If no cluster is
  available locally during implementation, validate via unit tests +
  `pre-commit run --all-files` and defer integration validation to CI.
- **Log-stream release discipline.** Every `follow=True, _preload_content=False`
  site must `await raw.release()` in a `finally`. Encode this in
  `kubernetes/logs.py` (and any copy in `watchdog.py`); reviewers check
  during commit 4/5.
- **Mock rewrite churn.** `tests/harness/k8s.py` and ~8 unit-test modules
  need fresh mocks. Done commit-by-commit so no single commit is
  unreviewably large.
- **`_kr8s_kwargs` workaround is dropped.** kr8s issue #737 (`KeyError` on
  kubeconfig with no `current-context`) is a kr8s-only bug; `kubernetes_asyncio`
  handles this cleanly, so the workaround goes away.

## 7. Non-goals

- No `DynamicClient` usage. Research-confirmed less common than
  `CustomObjectsApi`; `CustomObjectsApi` matches the sync-client pattern
  LLMs are trained on.
- No new exception hierarchy. Direct `ApiException` + `e.status` checks
  everywhere.
- No `is_not_found(e)` helper. Inline `if e.status == 404:` — research
  showed the helper is not an idiomatic shape in the ecosystem.
- No informer/watch stream translation. The existing watch orchestrator
  uses list-polling; it stays list-polling.
- No dependency changes for `kopf`, `aiohttp`, `httpx` (if transitive),
  or any other package.
