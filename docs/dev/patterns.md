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
from aiperf.common.config import ServiceConfig, UserConfig

app = App(name="profile")

@app.default
def profile(user_config: UserConfig, service_config: ServiceConfig | None = None) -> None:
    """Run the Profile subcommand."""
    from aiperf.cli_runner import run_system_controller  # heavy import deferred

    run_system_controller(user_config, service_config)
```

**Conventions:**
- Export a single `App` named `app`.
- Hyphenate multi-word commands: `App(name="analyze-trace")`.
- Keep module-level imports minimal; heavy deps go inside the function body.
- Import modules, not individual functions. Call functions on the module alias:
  `from aiperf.kubernetes import cli_helpers` then `cli_helpers.resolve_jobset(...)`.
  Use aliases when the module name conflicts with the current scope:
  `from aiperf.kubernetes import console as kube_console`.
- `Parameter(name=...)` for naming/aliasing only. Document parameters in the
  docstring `Args:` section — no `help=` on `Parameter()`.
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
    cancel.py         # aiperf kube cancel
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

app.command("aiperf.cli_commands.kube.attach:app", name="attach")
app.command("aiperf.cli_commands.kube.cancel:app", name="cancel")
```

```python
# cli_commands/kube/cancel.py
from cyclopts import App

app = App(name="cancel")

@app.default
async def cancel(job_id: str | None = None) -> None:
    """Cancel a running AIPerf benchmark."""
    from aiperf import cli_utils
    from aiperf.kubernetes import cli_helpers, console as kube_console

    with cli_utils.exit_on_error(title="Error Cancelling Job"):
        result = await cli_helpers.resolve_jobset(job_id, ...)
        kube_console.print_success(f"Job {job_id} cancelled.")
```

The group is registered in `cli.py` exactly like a flat command:

```python
app.command("aiperf.cli_commands.kube:app", name="kube")
```

## Kubernetes readiness pattern

For churn-safe Kubernetes components, prefer queryable current state over rebroadcast-only startup notifications:

- `WorkerPodManager` is the only controller-facing authority for a worker pod
- workers may connect to the credit router early, but must not become dispatchable until pod-local startup convergence completes
- pod-local ROUTER/DEALER request-reply is the preferred pattern for late joiners querying current dataset state
- controller startup gating should use aggregate pod snapshots rather than per-worker registration counts

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
- `ServiceConfig`: infrastructure (ZMQ ports, logging level)
- `UserConfig`: benchmark params (endpoints, loadgen settings)

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

For hot-path pod-local lifecycle traffic (for example `WorkerPodManager` talking to sibling workers/record processors in Kubernetes mode), prefer tagged `msgspec.Struct` unions over event-bus messages and send them over the streaming DEALER/ROUTER clients. Keep credit-router traffic on the existing credit channel.

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
