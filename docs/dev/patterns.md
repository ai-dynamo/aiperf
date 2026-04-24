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

## Async Subprocess Pattern

Every `kubectl`, `helm`, or other shellout goes through `aiperf.kubernetes.subproc` — never call `asyncio.create_subprocess_exec` directly. The helpers enforce a default 60 s timeout (so an unreachable apiserver cannot hang the CLI), return a structured `CommandResult`, and escalate `terminate()` -> `kill()` on cleanup so no orphan processes survive.

```python
# src/aiperf/kubernetes/subproc.py
async def run_command(cmd: list[str], *, timeout: float | None = 60.0) -> CommandResult:
    """Run a command asynchronously and capture output."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        await terminate_process(proc)   # graceful then kill
        raise
    return CommandResult(
        returncode=proc.returncode,
        stdout=raw_stdout.decode(errors="replace"),
        stderr=raw_stderr.decode(errors="replace"),
    )
```

**Conventions:**
- One-shot commands: `run_command(cmd)` -> `CommandResult`; use `result.ok` to test success. `check_command(cmd) -> bool` when you only need the exit status.
- Streaming (logs, port-forward): `start_streaming_process(cmd)` and always terminate with `terminate_process(proc, timeout=5.0)` in a `finally:` block.
- Never pass the full `shell=True` form; always pass a list of argv. Timeout defaults are deliberate — override only if you know why.

Reference file: `src/aiperf/kubernetes/subproc.py`. Call sites: `src/aiperf/kubernetes/results.py`, `logs.py`, `port_forward.py`.

## Kube Console Facade Pattern

All kube-CLI user output goes through `aiperf.kubernetes.console` imported as `kube_console` — never bare `print`, `rich.print`, or a locally constructed `Console()`. The module owns the single Rich `Console` instance, the uniform step/success/info/warning/error/header printers, and domain-specific printers for recurring surfaces (CR submission summary, detach info, results summary, AIPerfJob tables).

```python
# src/aiperf/cli_commands/kube/profile_deploy.py — canonical usage
from aiperf.kubernetes import console as kube_console

kube_console.print_info("AIPerfJob CRD detected, using operator mode")
kube_console.print_step(1, 4, "Rendering JobSet manifest")
...
kube_console.print_detach_info(name, namespace, name=kube_options.name)
kube_console.save_last_benchmark(name, namespace, name=kube_options.name)
```

**Conventions:**
- Routing every output through `kube_console` is what lets the dual-output pattern (next section) downshift text output cleanly when `--output json` is active. A direct `rich.print` leaks Rich markup into piped JSON.
- `kube_console` also owns last-benchmark persistence: `save_last_benchmark(...)` writes `~/.aiperf/last_kube_benchmark.json`; `cli_helpers.resolve_job(None, ...)` reads it when the user omits `job_id`. A new command that rolls its own `last_X.json` breaks the auto-resolve convention.
- Import as `kube_console`, not `console`, to avoid colliding with other `console` modules.

Reference file: `src/aiperf/kubernetes/console.py`. Call sites: `src/aiperf/cli_commands/kube/profile.py`, `profile_deploy.py`, `results.py`, `debug.py`.

## Kube CLI Dual-Output Pattern (`--output text|json`)

Read-only cluster/spec checks (`aiperf kube preflight`, `aiperf kube validate`) expose a `Literal["text", "json"]` `--output` flag defaulting to `"text"`. The check produces a structured result dataclass with a `to_dict()` method returning a `TypedDict` schema; the CLI branches on the flag and either lets `kube_console` render text or prints clean JSON via `orjson`. In JSON mode the `aiperf.kube` logger is temporarily downshifted to WARNING so stdout contains only the JSON document.

```python
# src/aiperf/cli_commands/kube/preflight.py
async def preflight(
    *,
    output: Annotated[
        Literal["text", "json"],
        Parameter(name=["-o", "--output"], help="Output format."),
    ] = "text",
    manage_options: KubeManageOptions | None = None,
) -> None:
    kube_logger = logging.getLogger("aiperf.kube")
    original_level = kube_logger.level
    if output == "json":
        kube_logger.setLevel(logging.WARNING)
    try:
        results = await checker.run_all_checks()   # -> PreflightResults
    finally:
        kube_logger.setLevel(original_level)

    if output == "json":
        json_output = orjson.dumps(
            results.to_dict(), option=orjson.OPT_INDENT_2
        ).decode()
        kube_console.console.print(json_output, highlight=False)
    else:
        results.render()   # rich text via kube_console

    if not results.passed:
        raise SystemExit(1)
```

**Conventions:**
- Result dataclass is a frozen `CheckResult`/`PreflightResults`/`ValidationResult` with fields `name`, `status`, `message`, `details`, `hints`, `duration_ms`; `to_dict()` produces the machine-readable form — the TypedDict schema is the public contract.
- JSON mode **must** suppress the default logger (otherwise INFO lines leak onto stdout and break CI pipes). Use a `try/finally` so the level is restored even if the check raises.
- Non-zero exit uses `raise SystemExit(1)` when `results.passed is False`. Do not call `sys.exit()` from handler bodies.

Reference files: `src/aiperf/cli_commands/kube/preflight.py`, `validate.py`; core: `src/aiperf/kubernetes/preflight.py`, `validate.py`.

## Watch Orchestrator + Renderer Protocol Pattern

Long-running kube monitoring (`aiperf kube watch`) is structured as three decoupled layers:

1. **Pollers** — one class per resource, each owning one API call and caching the latest result. Slower resources (events, pods) can be polled every Nth tick of the fast loop.
2. **Orchestrator** — opens *one* `k8s_client()` for the entire watch session, installs `SIGINT`/`SIGTERM` on the running event loop, and drives `asyncio.gather(*pollers)` every `interval`, assembling a frozen `WatchSnapshot` dataclass.
3. **Renderer** — a `WatchRenderer` Protocol with three methods (`start`/`render(snapshot)`/`stop`). Rich, text, and JSON implementations live alongside each other; the CLI selects one via a `_build_renderer(output)` factory.

```python
# src/aiperf/kubernetes/watch_orchestrator.py
class WatchRenderer(Protocol):
    """Structural type — any object with these three methods is a renderer."""
    def start(self) -> None: ...
    def render(self, snapshot: WatchSnapshot) -> None: ...
    def stop(self) -> None: ...


class WatchOrchestrator:
    async def run(self) -> None:
        async with k8s_client(kubeconfig=..., context=...) as api:
            cr_poller = CRPoller(api, self.job_id, self.namespace)
            pod_poller = PodPoller(api, self.job_id, self.namespace)
            event_poller = EventPoller(api, self.job_id, self.namespace)
            self._install_signal_handlers()
            self._renderer.start()
            try:
                await self._poll_loop(cr_poller, pod_poller, event_poller)
            finally:
                self._renderer.stop()
```

**Conventions:**
- One `async with k8s_client()` per watch session. A second watch-like feature must hang off the same orchestrator, not open its own client.
- New renderer = new class implementing the Protocol + one line in `_build_renderer`. Do not couple rendering into the poll loop.
- Snapshots are frozen dataclasses (`watch_models.py`); renderers never mutate them.

Reference files: `src/aiperf/kubernetes/watch_orchestrator.py`, `watch_pollers.py`, `watch_render_rich.py`, `watch_render_text.py`, `watch_render_json.py`, `watch_models.py`; CLI wiring: `src/aiperf/cli_commands/kube/watch.py`.

## Durable Completion-Claim Pattern

Operator handlers that do exactly-once work on a CR (final results export, terminal-phase notification) must claim ownership durably via an annotation — not via an in-process `set`. Two reconcile ticks, or a handler that resumes after an operator pod restart, will otherwise both fire. The claim is a JSON-patch with a `test` op: concurrent patches fail atomically with 409/422, so only one caller sees `True`.

```python
# src/aiperf/operator/client_cache.py
async def try_claim_completion(
    namespace: str, name: str, body: dict[str, Any],
) -> bool:
    """Claim the completion branch; return True iff this call newly won the race."""
    key = job_key(namespace, name)
    if key in _shutdown_sent:             # in-process fast path
        return False
    if is_completion_claimed(body):       # annotation already present (prior run)
        _shutdown_sent.add(key)
        return False
    patch_ops = _build_claim_patch_ops(body)   # JSON-patch with `test` op
    claimed = await _submit_claim_patch(namespace, name, patch_ops)
    if claimed is True:
        _shutdown_sent.add(key)
        return True
    if claimed is False:                  # lost race on 409/422
        _shutdown_sent.add(key)
    return False
```

**Conventions:**
- Guard every completion-side-effect path: `if await try_claim_completion(...): await handle_completion(...)`.
- The CR annotation is authoritative; `_shutdown_sent` is only a fast path that survives within one operator pod lifetime. The annotation survives operator restarts.
- Never clear the annotation from the completion side — a re-created CR is what resets the claim (see `on_create` -> `clear_cancellation(key)` in the next pattern).

Reference file: `src/aiperf/operator/client_cache.py`. Call sites: `src/aiperf/operator/handlers/lifecycle.py`, `handlers/monitor.py`.

## Cooperative Cancellation Pattern

Long-running operator handlers (fetch-with-backoff, result export, JobSet teardown) must abort promptly when the CR is deleted. `on_delete` sets a per-job `asyncio.Event` via `request_cancellation(key)`; handlers check `is_cancellation_requested(key)` at every `await` boundary and exit early instead of finishing remaining retries. Inject the check as a callable into helpers so the dependency is explicit — don't import the flag deep in the call stack.

```python
# src/aiperf/operator/handlers/lifecycle.py — signal on delete
@on_delete
async def on_delete(namespace: str, name: str, **_: Any) -> None:
    request_cancellation(job_key(namespace, name))

# src/aiperf/operator/handlers/create.py — clear on re-create
async def on_create(namespace: str, name: str, ...) -> None:
    clear_cancellation(job_key(namespace, name))
    ...

# src/aiperf/operator/handlers/_completion_fetch.py — poll in helpers
key = job_key(namespace, name)
await fetch_with_backoff(
    ...,
    is_cancelled=lambda: is_cancellation_requested(key),
)
if is_cancellation_requested(key):
    logger.info(f"Completion cancelled for {namespace}/{name}")
    return
```

**Conventions:**
- `on_delete` is the only caller of `request_cancellation`; the flag is sticky until the *next* `on_create` clears it. A completion handler that observes cancellation must not clear it.
- Helpers take an `is_cancelled: Callable[[], bool]` kwarg — do not reach into `client_cache` from deep call sites.
- Check cancellation before every patch to the CR: patching a CR that's being deleted wastes the apiserver round-trip and muddies the status.

Reference file: `src/aiperf/operator/client_cache.py`. Call sites: `handlers/lifecycle.py`, `handlers/create.py`, `handlers/completion.py`, `handlers/monitor.py`, `handlers/_completion_fetch.py`.

## Results-Ready Marker Pattern

The controller pod and its results sidecar share a PVC at `/results`. The sidecar refuses to serve top-level artifacts until the controller writes `.aiperf_results_ready.json` via `write_ready_marker(base_dir, was_cancelled=...)`. Checkpoints under `checkpoints/` bypass the gate (partial progress is safe to stream). The marker itself is a reserved filename and is excluded from listings and rejected by `_resolve_result_file`.

```python
# src/aiperf/kubernetes/results_sidecar.py
READY_MARKER_NAME = ".aiperf_results_ready.json"
CHECKPOINTS_DIR_NAME = "checkpoints"


def write_ready_marker(base_dir: Path, *, was_cancelled: bool = False) -> Path:
    """Write the readiness marker after exports complete."""
    marker = ready_marker_path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(
        orjson.dumps({"ready": True, "was_cancelled": was_cancelled})
    )
    return marker


async def _resolve_result_file(base_dir: Path, filename: str) -> Path:
    file_path = _safe_resolve(base_dir, filename)
    if file_path is None or file_path.name == READY_MARKER_NAME:
        raise HTTPException(400, f"Invalid filename {filename!r}")
    if not _is_ready(base_dir) and not _is_checkpoint_path(base_dir.resolve(), file_path):
        raise HTTPException(
            404,
            f"Results not ready for {base_dir.name}; "
            f"marker {READY_MARKER_NAME} not present — retry after completion",
        )
    return file_path
```

**Conventions:**
- Writer side: finish exporting all artifacts, *then* call `write_ready_marker(...)`. Never write the marker first.
- Reader side: check `_is_ready(base_dir)` before serving any top-level file; `checkpoints/` is explicitly allowed to bypass because partial progress is safe.
- New artifact types inherit the gate automatically — as long as they are written under `base_dir` before `write_ready_marker` is called, the sidecar exposes them without special-casing.

Reference file: `src/aiperf/kubernetes/results_sidecar.py`. Writer: `src/aiperf/controller/system_controller.py`. Reader: `src/aiperf/api/routers/results.py`.

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

## LLM-Ergonomics Exemplars

Concrete good/bad examples for the semantic axes CLAUDE.md's "LLM-Ergonomics" section refers to. The mechanical checks (`make check-ergonomics`, `make check-ruff-baselined`) enforce the floor — presence of docstrings, non-empty exception messages, return-type annotations, no bare `except`. These examples show the **ceiling**: when the code passes the mechanical checks and is still ambiguous to an agent.

### Exception messages — include operation, input, and next step

```python
# BAD — passes the >=3-word rule but tells the agent nothing it could not already see
raise ValueError("bad input")
raise ConfigurationError("missing field")
raise DatasetLoadError(f"error: {type(e).__name__}")   # threw away str(e)

# GOOD — names the operation, the specific input, and what the user / agent should do
raise DatasetLoadError(
    f"dataset '{name}' failed to parse at row {row_idx}: "
    f"column 'answer_key' is required but missing; "
    f"add it to the config or set skip_validation=true"
)
raise ConfigurationError(
    f"endpoint '{endpoint_name}' declared in plugins.yaml but class "
    f"'{class_path}' could not be imported: {import_err!s}"
) from import_err
```

When wrapping a stdlib exception, preserve context with `raise ... from e` — the traceback chain is how agents diagnose upstream failures.

### Type hints — encode domain meaning, not just shape

```python
# BAD — passes ANN201 but tells the agent nothing about the contract
def build(spec: dict, phase: str) -> Any: ...
async def dispatch(handler: Callable, msg: dict) -> None: ...

# GOOD — Literal for enum-like strings, Protocol/TypedDict for structural contracts,
# parameterized containers, explicit Callable arg and return types
def build(spec: BenchmarkSpec, phase: Literal["warmup", "measure"]) -> BuildResult: ...
async def dispatch(
    handler: Callable[[AIPerfJobCR], Awaitable[None]],
    msg: JobEventMessage,
) -> None: ...
```

Use `X | None`, never `Optional[X]` or `Union[X, None]` — the bar-form matches the project convention and reads as one token to an agent.

### Docstrings — runnable example with realistic identifiers

```python
# BAD — restates the name, placeholder values, no side-effects noted
def resolve_job(job_id: str | None, namespace: str) -> ResolvedJob | None:
    """Resolves a job.

    Args:
        job_id: the job id
        namespace: the namespace
    """

# GOOD — realistic example, side-effects and raises named
def resolve_job(job_id: str | None, namespace: str) -> ResolvedJob | None:
    """Look up an AIPerfJob CR by id or fall back to the most recently deployed one.

    Reads the cluster (no writes). Logs a one-line "resolved <id> -> <ns>/<name>"
    info message; returns None if no match after fallback.

    Example:
        >>> resolved = await resolve_job("aiperf-bench-7f2a", namespace="aiperf")
        >>> resolved.job_id, resolved.namespace
        ('aiperf-bench-7f2a', 'aiperf')

    Raises:
        KubeAccessError: if the cluster is unreachable or kubeconfig is invalid.
    """
```

### Naming — mention synonyms in the docstring

If two domain concepts share a word (e.g. `Credit` as the internal name vs. "work unit" in user-facing docs, or `Metrics` / `Records` / `Results`), the authoritative class names the synonyms so `grep` from any direction lands here:

```python
class Credit(AIPerfBaseModel):
    """A unit of dispatchable work issued by the timing manager to a worker.

    Also called a "work unit" in user-facing docs and "request-slot" in the
    Kubernetes CR spec. Do not confuse with `Request` (the HTTP payload sent
    to the inference server) or `Session` (a multi-turn conversation).
    """
```

### Comments — WHY, not WHAT

```python
# BAD — restates the code in English
counter += 1  # increment counter
self._cache = {}  # initialize cache

# BAD — dead-code comment; git log is authoritative
# result = old_implementation(x)
result = new_implementation(x)

# GOOD — documents a non-local constraint or past bug the reader cannot see
# records-manager CPU starves above ~500k concurrency with default 1000m
# limits; keep this batch size <= 4096 or raise AIPERF_K8S_RECORDS_MANAGER_CPU.
batch_size = min(len(pending), 4096)

# GOOD — WHY a non-obvious API choice
# orjson is used instead of stdlib json to keep p99 serialization < 50us
# on 10kB records; see dev/benchmarks/json_roundtrip.py.
payload = orjson.dumps(record)
```

A TODO without an issue link, author, or date is almost always noise — delete it or file the issue.

### Reference files — stay exemplary

Files cited in this document (via leading-comment paths such as `# aiperf/kubernetes/client.py`) are the gold standard for the pattern they teach. Before shipping a change that touches one:

- Re-read the snippet shown here; confirm it still matches the edited file.
- Check that the file has **no** new entries in `tools/ergonomics_baseline.json` or `tools/ruff_baseline.json` — a reference file grandfathering a violation of the rule it teaches is the worst possible signal for an agent.
- If you add a `# noqa: <RULE>`, accompany it with a short comment explaining why (e.g. `# noqa: BLE001 - fault-tolerant telemetry, must not raise`).

Conversely, if this branch introduces the first clean implementation of a new pattern, extend this file with a snippet that references it — that is how the pattern stops being tacit.

