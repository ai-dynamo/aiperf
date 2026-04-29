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
- Heavy implementation logic lives in a `cli.py` inside the owning domain
  package (e.g. `aiperf/plugin/cli.py`), lazily imported at call time.

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

## DAG Branch Orchestrator Pattern

`BranchOrchestrator` (`src/aiperf/timing/branch_orchestrator.py`) is the
reference implementation for any future intercept-credit-return → fan-out
→ join control flow (e.g. SPAWN_JOIN, BARRIER, TIMER). Built on top of the
existing credit/strategy machinery; no new service, no new plugin category.

New orchestrator features should:

1. **Hook the credit-return path** via `intercept(credit) -> bool`. Return
   `True` to suppress the strategy's default next-turn dispatch when the
   completed turn declared branches; return `False` for non-branching turns.
2. **Coordinate with the three pre-existing guards** the credit-callback
   handler, the credit issuer, and the strategy already honor — see the
   "Stop-condition interaction" section of the module docstring of
   `branch_orchestrator.py`. Specifically: callback-handler child bypass
   (children with `agent_depth > 0` reach `handle_credit_return` even after
   `can_send_any_turn` flips False), completion-event deferral (the
   all-credits-returned event holds while `has_pending_branch_work()`), and
   session-slot bypass for children (`agent_depth > 0` never acquires a
   session slot).
3. **Track per-parent state in dict-of-dataclass form** (see
   `PendingBranchJoin`) and tear it down via a single `cleanup()` entry
   point that `PhaseRunner` calls on every phase-exit path. `cleanup()`
   must be idempotent and short-circuit late credit returns via a
   `_cleaning_up` flag.
4. **Mirror sticky-router refcount mode-gating**: only FORK-mode children
   call `StickyCreditRouter.register_child_routing` /
   `release_child_routing` (parent-pinned for prefix-cache locality);
   SPAWN-mode children do not.
5. **Expose stats via a `BranchStats` extension** on
   `ProfileResults.branch_stats` (mode-agnostic counters: `children_spawned`,
   `children_completed`, `children_errored`, `parents_suspended`,
   `parents_resumed`, `parents_failed_due_to_child_error`).

For load-time rejection of constructs the runtime does not yet honor, see
`src/aiperf/common/validators/orchestrator_v1.py`. New runtime features
**relax** that validator (remove the corresponding `NotImplementedError`
clause); the validator function name and contract stay — there is no v2.

Tunable knobs go on `_DagSettings` in `src/aiperf/common/environment.py`
(env_prefix `AIPERF_DAG_`), accessed as `Environment.DAG.X`. Never read DAG
env vars via `os.getenv` directly — the structured-settings pattern is what
makes them appear in `docs/environment-variables.md` automatically.
