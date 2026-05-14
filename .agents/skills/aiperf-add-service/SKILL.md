---
name: aiperf-add-service
description: Use BEFORE adding a new service, new message type, or new @on_message handler in aiperf — "add a new service", "create a worker that listens for X", "wire a new message bus event", "add an @on_message handler", "register a new MessageType". Five-touch dance (enum, Message class, BaseComponentService class, plugins.yaml registration, optional service_metadata) that produces silent dead-letter symptoms on miss.
---

# AIPerf Add Service / Message

Adding a new service or message type to aiperf is a 5-touch dance. Missing any of them produces dead-letter symptoms — messages get published, but no handler subscribes, and the failure is silent unless you read the message-bus diagnostics.

## The 5 touches (for a new message type with handler)

1. **Add the enum value** to `MessageType` in `src/aiperf/common/enums/enums.py`. Enum is string-based — use `MessageType.X` not `MessageType.X.value`.
2. **Create the Message class** in `src/aiperf/common/messages/<your_module>.py` inheriting from `Message`, with `message_type` set explicitly.
3. **Add the `@on_message(MessageType.X)` handler** in the receiving service. Auto-subscription happens during the service's `@on_init` phase.
4. **Register the service** (if new) in `src/aiperf/plugin/plugins.yaml` under the `service` category. Plugin registrations are keyed by plugin name (a dict), not a list.
5. **Optional**: set `service_metadata.disable_gc: true` for hot-path services (Worker, TimingManager) — the GC pauses dominate tail latency.

## When adding ONLY a message (no new service)

Touches 1-3 only. Step 4 and 5 are skipped.

## When adding ONLY a service (existing messages)

Touches 3-5 only. Steps 1-2 are skipped.

## Steps

### 1. Enum value

```python
# src/aiperf/common/enums/enums.py
class MessageType(CaseInsensitiveStrEnum):
    ...
    YOUR_NEW_EVENT = "your_new_event"
```

`MessageType` is a `CaseInsensitiveStrEnum` (string-based). Use `MessageType.X` directly; never `.value`.

### 2. Message class

```python
# src/aiperf/common/messages/your_new_event.py
from aiperf.common.enums import MessageType
from aiperf.common.messages.base_messages import Message

class YourNewEvent(Message):
    message_type: MessageType = MessageType.YOUR_NEW_EVENT
    # ... your fields, with Field(description="...") on every one
```

### 3. Handler on the receiving service

```python
# Services live in their own package neighborhood (api/, workers/, timing/, etc.) — not under a flat services/ dir.
from aiperf.common.hooks import on_message
from aiperf.common.base_component_service import BaseComponentService

class YourService(BaseComponentService):
    @on_message(MessageType.YOUR_NEW_EVENT)
    async def handle_your_event(self, msg: YourNewEvent) -> None:
        # ... handler body
        ...
```

All decorators live in `src/aiperf/common/hooks.py` (NOT `decorators.py`).

`BaseComponentService` (`src/aiperf/common/base_component_service.py`) for normal services — `BaseService` (`src/aiperf/common/base_service.py`) is reserved for `SystemController`. Use the right base, or you'll bypass component-lifecycle hooks.

### 4. Plugin registration (new service only)

`plugins.yaml` uses a **dict keyed by plugin name** under each category, NOT a list of entries:

```yaml
# src/aiperf/plugin/plugins.yaml
service:
  your_service_name:
    class: aiperf.your_package.your_service:YourService
    description: One-line description of what this service does.
    priority: 100           # top-level field; higher wins, external beats built-in at equal priority
    metadata:
      # ServiceMetadata fields: required (bool), auto_start (bool), disable_gc (bool), replicable (bool)
      # disable_gc: true    # only for hot-path services like Worker / TimingManager
```

Place the service module in the appropriate package (`api/`, `workers/`, `timing/`, `dataset/`, `records/`, `server_metrics/`, `gpu_telemetry/`, `controller/`) — there is no top-level `src/aiperf/services/`.

Then `make generate-all-plugin-files` and `make validate-plugin-schemas`.

### 5. Smoke test

```bash
aiperf plugins --validate     # confirms registry + Protocol conformance
```

Run `aiperf-correctness-testing` if the new service is in the runtime path.

## Communication patterns

| Pattern | When |
|---|---|
| `await self.publish(YourEvent(...))` | Fire-and-forget broadcast; multiple handlers may subscribe. |
| `@on_message(MessageType.X)` | Subscribe to a broadcast. |
| `await self.send_command_and_wait_for_response(...)` | Synchronous request/response across services. |
| `@on_pull_message(MessageType.X)` | Pull from a queue (single-consumer). |
| `@on_request` | RPC-style handler. |
| `@background_task` | Long-running task spawned at `@on_start`. |

Core lifecycle/messaging decorators: `@on_init`, `@on_start`, `@on_stop`, `@on_message`, `@on_command`, `@background_task`, `@on_pull_message`, `@on_request`. Progress/state decorators (see `src/aiperf/common/hooks.py`): `@on_state_change`, `@on_realtime_metrics`, `@on_realtime_telemetry_metrics`, `@on_profiling_progress`, `@on_records_progress`, `@on_warmup_progress`, `@on_worker_status_summary`, `@on_worker_update`.

## Service lifecycle

`AIPerfLifecycleMixin` for standalone components: `CREATED` → `INITIALIZING` → `INITIALIZED` → `STARTING` → `RUNNING` → `STOPPING` → `STOPPED`. `FAILED` is terminal.

## Docs

Per CLAUDE.md's Documentation table: new services and message types update `docs/architecture.md` AND the relevant page under `docs/dev/patterns/` (`service.md` for service patterns, `message.md` for message patterns). Don't skip.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll subscribe to the new message in two services" | Fine, but verify both subscribe during `@on_init`. Late subscription means missed messages. |
| "I'll use `BaseService` instead of `BaseComponentService`" | `BaseService` is for `SystemController` only. Component services need the lifecycle hooks `BaseComponentService` provides. |
| "I'll add `disable_gc: true` to be safe" | GC disabled = no auto-collection. Use only for proven hot paths (Worker, TimingManager); otherwise you leak. |
| "I'll skip `aiperf plugins --validate`, the test will catch it" | The component-integration test catches it eventually. The validate is one second; the test cycle is minutes. |
| "I'll use `MessageType.X.value` for the string" | The enum IS the string. Use `MessageType.X` directly. `.value` is forbidden by project convention. |

## Common mistakes

- **Forgetting to subscribe in `@on_init`.** Messages published before subscription are lost.
- **Sharing message types across services that don't agree on the schema.** The Message class is the contract; both sides import the same class.
- **Putting the Message class in the service file** instead of `src/aiperf/common/messages/`. Breaks the message-bus discovery convention.
- **Re-using a `MessageType` value for a different shape.** Enum value uniqueness is enforced; reusing it produces decode failures.

## Composition

- `aiperf-add-plugin` for the `plugins.yaml` step.
- `aiperf-correctness-testing` for end-to-end validation.
- `aiperf-integration-test` for a test that exercises the new message handler in a multi-service flow.
