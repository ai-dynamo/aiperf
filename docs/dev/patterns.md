---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Code Patterns
---
# AIPerf Code Patterns

Code examples for common development tasks. Referenced from CLAUDE.md.

## CLI Command Pattern

Commands live in `src/aiperf/cli_commands/`, one file per command. They are
lazily loaded via import strings in `aiperf.cli` — modules are only imported
when their command is invoked:

```python
# aiperf/cli.py — register with lazy import strings
app.command("aiperf.cli_commands.profile:app", name="profile")
```

```python
# aiperf/cli_commands/profile.py — thin command definition
from cyclopts import App

from aiperf.config.cli_model import CLIModel

app = App(name="profile")

@app.default
def profile(*, cli_model: CLIModel) -> None:
    """Run the Profile subcommand."""
    from aiperf.cli_runner import run_system_controller  # heavy import deferred

    run_system_controller(cli_model)
```

**Conventions:**
- Export a single `App` named `app`.
- Hyphenate multi-word commands: `App(name="analyze-trace")`.
- Keep module-level imports minimal; heavy deps go inside the function body.
- Import modules, not individual functions. Call functions on the module alias:
  `from aiperf.kubernetes import cli_helpers` then `cli_helpers.resolve_jobset(...)`.
  Use aliases when the module name conflicts with the current scope:
  `from aiperf.kubernetes import console as kube_console`.
- Use `Parameter(name=..., help="...")` for both aliasing and user-visible help
  text. `help=` surfaces in `--help` output and should be concise and imperative.
  Example from `cli_commands/kube/attach.py`:

  ```python
  port: Annotated[
      int,
      Parameter(
          name=["-p", "--port"],
          help="Local port for port-forward (default: 0 = ephemeral).",
      ),
  ] = 0,
  ```

  Longer prose (workflow, examples, side effects) belongs in the command's
  docstring — cyclopts renders both.
- For commands that share cross-cutting k8s flags (`namespace`, `kubeconfig`,
  `kube-context`), accept a `KubeManageOptions` composite kwarg instead of
  redeclaring each `Parameter`. Reference: `cli_commands/kube/attach.py`
  (signature takes `manage_options: KubeManageOptions | None = None`).
- Wrap all heavy work inside `with cli_utils.exit_on_error(...):`; place deferred
  imports inside that block so import errors are caught.
- Heavy implementation logic lives in a `cli.py` inside the owning domain
  package (e.g. `aiperf/plugin/cli.py`), lazily imported at call time.

### Subcommand Groups

For commands with multiple subcommands, use a directory with `_app.py` and
`__init__.py`. The `_app.py` file defines the group `App` and lazily registers
subcommands. Each subcommand lives in its own file within the directory:

```
cli_commands/
  kube/
    __init__.py       # re-exports app from _app.py
    _app.py           # group App + lazy subcommand registration
    attach.py         # aiperf kube attach
    list_.py          # aiperf kube list
    profile.py        # aiperf kube profile
    ...
```

```python
# cli_commands/kube/__init__.py
from aiperf.cli_commands.kube._app import app

__all__ = ["app"]
```

```python
# cli_commands/kube/_app.py
from cyclopts import App

app = App(name="kube", help="Kubernetes deployment and management commands")

app.command(
    "aiperf.cli_commands.kube.attach:app",
    name="attach",
    help="Attach to a running benchmark and stream progress",
)
app.command(
    "aiperf.cli_commands.kube.list_:app",
    name="list",
    help="List benchmark jobs and their status",
)
```

```python
# cli_commands/kube/attach.py — canonical subcommand shape
@app.default
async def attach(
    job_id: Annotated[
        str | None,
        Parameter(help="The AIPerf job ID to attach to (default: last deployed job)."),
    ] = None,
    *,
    manage_options: KubeManageOptions | None = None,
    port: Annotated[
        int,
        Parameter(
            name=["-p", "--port"],
            help="Local port for port-forward (default: 0 = ephemeral).",
        ),
    ] = 0,
) -> None:
    """Attach to a running AIPerf benchmark and stream progress."""
    from aiperf import cli_utils
    from aiperf.kubernetes import attach as kube_attach
    from aiperf.kubernetes import cli_helpers

    manage_options = manage_options or KubeManageOptions()

    with cli_utils.exit_on_error(title="Error Attaching to Benchmark"):
        resolved = await cli_helpers.resolve_job(
            job_id,
            manage_options.namespace,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
        if not resolved:
            return

        await kube_attach.attach_to_benchmark(
            resolved.job_id,
            resolved.namespace,
            port,
            resolved.api,
            phase=resolved.job_info.phase,
            kubeconfig=manage_options.kubeconfig,
            kube_context=manage_options.kube_context,
        )
```

Every kube subcommand follows the same shape: `exit_on_error` wrapper,
`cli_helpers.resolve_job(...)` (or `resolve_jobset(...)`) to look up the target
CR, then `await` the action function in `aiperf.kubernetes.*`.

The group is registered in `cli.py` exactly like a flat command:

```python
app.command("aiperf.cli_commands.kube:app", name="kube")
```

## Group readiness pattern

For churn-safe worker groups in local and Kubernetes mode, prefer queryable current state over rebroadcast-only startup notifications:

- `WorkerGroupManager` is the controller-facing authority for each worker group in both local and Kubernetes mode
- `WorkerGroupManager` is the universal readiness and declared-capacity unit; workers and record processors are group-local children, not controller-facing services
- workers may connect to the credit router early, but must not become dispatchable until group-local startup convergence completes
- group-local ROUTER/DEALER request-reply is the preferred pattern for late joiners querying current dataset state
- controller startup gating should use aggregate worker-group snapshots rather than per-worker registration counts

## Service Pattern

Services run in separate processes via `bootstrap.py`:

```python
class MyService(BaseComponentService):
    @on_message(MessageType.MY_MSG)
    async def _handle(self, msg: MyMsg) -> None:
        await self.publish(ResponseMsg(data=msg.data))
```

Register in `plugins.yaml`:

```yaml
service:
  my_service:
    class: aiperf.my_module.my_service:MyService
    description: My custom service
    metadata:
      required: true
      auto_start: true
```

**Config types:**
- `CLIModel` (`aiperf.config.cli_model`): raw CLI input parsed from argv.
- `BenchmarkConfig` (`aiperf.config`): the resolved, validated benchmark spec
  (endpoint, datasets, phases, runtime, artifacts) — v3 YAML-first schema.
- `BenchmarkRun` (`aiperf.config`): a single run wrapping `BenchmarkConfig` with
  benchmark-id, artifact-dir, and resolver state; what services receive.

## Model Pattern

Use `AIPerfBaseModel` for data, `BaseConfig` for configuration:

```python
from pydantic import Field
from aiperf.common.models import AIPerfBaseModel

class Record(AIPerfBaseModel):
    ts_ns: int = Field(description="Timestamp in nanoseconds")
    value: float = Field(description="Measured value")
```

## Message Pattern

Messages require `message_type` field and handler decorator:

```python
from aiperf.common.messages import Message
from aiperf.common.hooks import on_message

class MyMsg(Message):
    message_type: MessageType = MessageType.MY_MSG
    data: list[Record] = Field(description="Records to process")

# In service class:
@on_message(MessageType.MY_MSG)
async def _handle(self, msg: MyMsg) -> None:
    await self.publish(OtherMsg(data=msg.data))
```

Auto-subscription happens during `@on_init` phase.

For hot-path transport paths, prefer tagged `msgspec.Struct` payloads over routed Pydantic messages.

- Use streaming DEALER/ROUTER + tagged `msgspec.Struct` unions for group-local lifecycle traffic (for example `WorkerGroupManager` talking to sibling workers/record processors in Kubernetes mode, or the same contract in local worker groups).
- Use dedicated msgspec wire structs plus channel-specific codecs for PUSH/PULL hot paths such as the record-processor -> records-manager metric-record channel.
- Keep event-bus `Message` subclasses for general Pub/Sub traffic where routed Pydantic models remain sufficient.
- Keep credit-router traffic on the existing credit channel.

## Plugin System Pattern

YAML-based registry with lazy-loading:

```yaml
# plugins.yaml
endpoint:
  chat:
    class: aiperf.endpoints.openai_chat:ChatEndpoint
    description: OpenAI Chat Completions endpoint
    metadata:
      endpoint_path: /v1/chat/completions
      supports_streaming: true
      produces_tokens: true
      tokenizes_input: true
      supports_audio: true
      supports_images: true
      supports_videos: true
      metrics_title: LLM Metrics
```

```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'chat')
```

## Kubernetes Operator Handler Pattern

All `@kopf.on.*` decorators live in `src/aiperf/operator/main.py`; handler
modules under `src/aiperf/operator/handlers/` are decorator-free and are
invoked by thin delegators in `main.py`. This keeps handler logic
independently testable — tests import the handler function directly and pass
in the kwargs kopf would inject.

```python
# src/aiperf/operator/main.py — decorator + delegation
@kopf.on.create(AIPERF_GROUP, AIPERF_VERSION, AIPERF_PLURAL)
async def on_create(
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    uid: str,
    patch: kopf.Patch,
    **_: Any,
) -> dict[str, Any]:
    """Create ConfigMap and JobSet for the benchmark job."""
    return await create.on_create(
        body=body, spec=spec, name=name, namespace=namespace, uid=uid, patch=patch
    )
```

**Conventions:**
- kopf calls handlers with a fixed kwarg set: `body, spec, name, namespace,
  patch, uid, logger, **_: Any`. Keep the same order and always accept
  `**_: Any` so future kopf additions don't break the call. These signatures
  are baselined against the keyword-only-args check because kopf owns the
  shape.
- Raise `kopf.PermanentError(msg)` for non-recoverable failures (bad spec,
  unsupported config). kopf will stop retrying and surface the error on the CR.
- Raise `kopf.TemporaryError(msg, delay=N)` for transient failures (network
  blip, API server 503). kopf will retry after `N` seconds.
- A generic `Exception` leaking out of a handler causes kopf to retry
  **forever** with exponential backoff — almost never what you want.
- Include `namespace`/`name` in every kopf error message; the text appears in
  `kubectl describe aiperfjob` without the CR identifier otherwise.

Reference files: `src/aiperf/operator/main.py`,
`src/aiperf/operator/handlers/create.py`.

### Kopf Handler Module Split

Keep kopf decorators confined to `operator/main.py` so handler modules stay
independently testable. Handler functions take explicit kwargs matching what
`main.py` forwards — no `**kwargs` pass-through, no reliance on kopf's magic
injection outside the decorator site. Tests can then call
`await create.on_create(body=..., spec=..., name=..., ...)` with fixtures and
without a live kopf process.

## Kubernetes API Access Pattern

Every kubernetes_asyncio access point goes through the `k8s_client()`
async-context-manager in `src/aiperf/kubernetes/client.py`. It handles
in-cluster-vs-kubeconfig fallback and guarantees the `ApiClient` is closed.
Never instantiate `kubernetes_asyncio.client.ApiClient()` directly — that
leaks connections and skips the in-cluster config load.

```python
# src/aiperf/kubernetes/client.py — the canonical entry point
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
```

Call sites use the context manager plus the free-function helpers defined in
the same module:

```python
from aiperf.kubernetes.client import k8s_client, list_aiperf_jobs

async with k8s_client() as api:
    jobs = await list_aiperf_jobs(api, namespace="aiperf-bench")
```

**Conventions:**
- Free functions in `kubernetes/client.py` take `api: ApiClient` as their
  first positional arg and build the typed sub-API (`CustomObjectsApi`,
  `CoreV1Api`, `RbacAuthorizationV1Api`) internally.
- Keep the `async with` scope tight — a single `k8s_client()` per logical
  operation, not per long-lived service.
- Handlers and CLI commands both use this same entry point.

Reference file: `src/aiperf/kubernetes/client.py`.

## FastAPI Router Factory Pattern

Two router shapes coexist:

1. **Module-level router** — for stateless endpoints. Used in
   `src/aiperf/api/routers/*.py`:

   ```python
   from fastapi import APIRouter

   router = APIRouter(prefix="/results", tags=["results"])

   @router.get("/")
   async def list_results() -> ResultsListResponse:
       ...
   ```

2. **Factory returning `APIRouter`** — for endpoints that must close over
   live state (a cache, a shared `ApiClient` holder, a lifespan-scoped
   resource). Used in `src/aiperf/operator/routers/jobs.py`:

   ```python
   def create_jobs_router(progress_cache: ProgressCache) -> APIRouter:
       router = APIRouter(prefix="/jobs", tags=["jobs"])

       @router.get("/")
       async def list_jobs() -> ActiveJobListResponse:
           return await progress_cache.list()

       return router
   ```

**When to use which:**
- Module-level: handlers depend only on per-request dependencies (query
  params, request body, `Depends(...)` on stateless factories).
- Factory: handlers must share an object created by the application's
  lifespan — e.g. a `ProgressCache`, an `ApiClient` holder, a task queue.
  Registering at app-startup time with `app.include_router(create_xxx_router(cache))`
  threads the dependency through without a global.

Reference file: `src/aiperf/operator/routers/jobs.py`.

## Error Handling Pattern

Log errors and publish `ErrorDetails` in messages:

```python
try:
    await risky_operation()
except Exception as e:
    self.error(f"Operation failed: {e!r}")
    await self.publish(ResultMsg(error=ErrorDetails.from_exception(e)))
```

## Logging Pattern

Use lambda for expensive log messages:

```python
# Expensive - lambda defers evaluation
self.debug(lambda: f"Processing {len(self._items())} items")

# Cheap - direct string is fine
self.info("Starting service")
```

## Testing Pattern

```python
import pytest
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.harness import mock_plugin

@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_func()
    assert result.status == "ok"

@pytest.mark.parametrize("input,expected",
    [
        ("a", 1),
        ("b", 2),
    ]
)  # fmt: skip
def test_with_params(input, expected):
    assert process(input) == expected

def test_with_mock_plugin():
    with mock_plugin(PluginType.ENDPOINT, "test", MockClass):
        assert plugins.get_class(PluginType.ENDPOINT, "test") == MockClass
```

**Auto-fixtures** (always active): asyncio.sleep runs instantly, RNG=42, singletons reset.
