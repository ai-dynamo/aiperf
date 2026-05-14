---
name: aiperf-message-trace
description: Use when debugging the aiperf ZMQ message bus or service-to-service communication — "messages aren't reaching the handler", "credit isn't flowing", "@on_message didn't fire", "subscription drops", "service heartbeat missed", "dead-letter symptom", "publish but no consume", "credit-pipeline deadlock", "message bus backpressure". Codifies how to inspect subscriptions registered during @on_init, trace a message from publish to handler, identify dead-letter paths, and verify @on_pull_message vs @on_message semantics.
---

# AIPerf Message Bus Tracing

aiperf has 10 services (1 `BaseService` orchestrator — SystemController — plus 9 `BaseComponentService` components) communicating via a ZMQ-backed message bus. The most subtle production bugs live in this layer because failures are silent — messages publish successfully but never get consumed, and the only symptom is "the thing that should have happened didn't."

This skill encodes how to reason about the bus when you see dead-letter symptoms.

## Subscription model

Services declare subscriptions via decorator. All decorators live in `src/aiperf/common/hooks.py` (NOT `decorators.py`):

| Decorator | Semantics |
|---|---|
| `@on_message(MessageType.X)` | Broadcast subscribe. Every service with this handler receives the message. |
| `@on_pull_message(MessageType.X)` | Pull from a queue. **Single-consumer** — only one handler ever receives a given message. Re-opening a pull on the same queue orphans prior handles silently. |
| `@on_command(CommandType.X)` | Command-style handler with response. Caller uses `send_command_and_wait_for_response`. |
| `@on_request` | RPC-style. |

Auto-subscription happens during `@on_init`. **Messages published before `@on_init` completes are lost.** This is the single most common dead-letter cause.

## When dead-letter is suspected

### Step 1 — Confirm the message is publishing

```bash
# Look for the publish call site in the producer
grep -rn 'publish(.*YourMessageType' src/aiperf/
```

If absent, the producer never called `publish`. Done.

If present, add a temporary debug log at the publish site (or inspect a log already there):

```python
self.debug(lambda: f"publishing {MessageType.YOUR} with payload={payload}")
```

### Step 2 — Confirm at least one service subscribes

```bash
# All @on_message handlers for this type across the codebase. Services don't live
# in a flat services/ dir — they're distributed across api/, workers/, timing/,
# dataset/, records/, server_metrics/, gpu_telemetry/, controller/.
grep -rn '@on_message(MessageType.YOUR' src/aiperf/
grep -rn '@on_pull_message(MessageType.YOUR' src/aiperf/
```

If absent, no service is subscribed. Add the handler (see `aiperf-add-service`).

If present, confirm:
- The handler is on a `BaseComponentService` (not a helper class).
- The service is registered in `src/aiperf/plugin/plugins.yaml`.
- The service is included in the run's service set (some services are conditional).

### Step 3 — Confirm `@on_init` completed before the publish

`aiperf` starts services in dependency order, but a producer can fire `publish` from `@on_start` BEFORE a consumer's `@on_init` finishes registering its handlers. Symptoms:

- First N messages lost; later ones delivered.
- Test passes when concurrency is low (slow init has time); fails at high concurrency.

Fix: producers should hold their first `publish` until either (a) the lifecycle has reached `RUNNING` for the relevant consumer, or (b) the message is queued via `@on_pull_message` (which buffers in ZMQ until the puller catches up).

### Step 4 — For `@on_pull_message`: confirm single-consumer invariant

A pull handle is single-consumer. If two services subscribe to the same queue via `@on_pull_message`, only one wins; the other silently never receives. Auditing:

```bash
grep -rn '@on_pull_message' src/aiperf/ | sort
```

Count occurrences of each `MessageType.X`. If two services declare pull on the same type, you've found the bug.

A related trap: re-creating a pull handle on the same queue (e.g., re-instantiating the service) orphans the prior handle. Restart-style bugs hit this.

### Step 5 — Inspect the bus directly

The bus layer lives in `src/aiperf/common/mixins/` — primarily `message_bus_mixin.py` (publish/subscribe), `command_handler_mixin.py` (request/reply). Transport is in `src/aiperf/common/base_comms.py`. ZMQ proxy registrations are in `plugins.yaml` under the `zmq_proxy` category.

For ad-hoc tracing, the bus exposes a debug-tap pattern (look at how unit tests under `tests/component_integration/` instrument it). For production-runtime tracing, increase log verbosity:

```bash
aiperf profile --log-level DEBUG ...   # or `--verbose` / `-v`
... | grep -i your_message_type
```

### Step 6 — `aiperf service` single-service launch

If the suspect is one specific service's handler, isolate it:

```bash
aiperf service --type <ServiceType.YOUR> --service-id debug-<n>
```

This boots ONE service in the foreground with its config dumped from a previous run. Attach a debugger (`pdb.set_trace()` in the handler), publish a message at it from a sibling shell, watch the handler fire (or not).

## Anatomy of the credit pipeline (recurring source of bugs)

The credit pipeline crosses many services:

1. **PhaseRunner** computes phase config and creates `Credit` instances.
2. **CreditIssuer** publishes credits onto the bus.
3. **Worker** receives credits, dispatches requests against the endpoint.
4. **RecordsManager** receives per-request records, computes aggregates.
5. **TimingManager** controls inter-arrival timing.

Each step has its own message types and queues. The full pipeline involves >10 message types. The earlier-flagged "three-touch per-session field" pattern (struct + CreditIssuer copy + PhaseRunner user_config plumb) is the canonical example: missing any one of the three touches lets the change look correct at unit-test level while silently dropping the field at one of the inter-service handoffs.

When debugging credit-pipeline issues:

```bash
# Credits flow router→worker as a tagged msgspec Struct, NOT a Message subclass.
# The on-wire struct is `Credit` (tag="c") in src/aiperf/credit/structs.py.
# Dispatch input is `TurnToSend`, also in structs.py.
# Phase-lifecycle messages (CreditPhaseStart, CreditReturn, CreditPhaseProgress, ...) live in src/aiperf/credit/messages.py.
grep -rn 'class Credit\b\|class TurnToSend\b' src/aiperf/credit/structs.py
grep -rn 'CreditReturn\|CreditPhase'         src/aiperf/credit/messages.py
grep -rn 'issue_credit'                       src/aiperf/credit/issuer.py
```

Cross-check that every field on the `Credit` struct (`src/aiperf/credit/structs.py`) is plumbed through `CreditIssuer.issue_credit` and the worker's credit consumer. Field drop here is silent — unit tests won't catch it; component-integration will.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "The producer logs say 'published', the consumer must be broken" | Publish succeeds whether or not anyone is subscribed. Verify the subscription side. |
| "I'll add `await asyncio.sleep(0.1)` after publish, then it'll arrive" | The bus IS async. Sleeping doesn't help — if the subscription doesn't exist or the handler doesn't fire, sleep is just a delay. |
| "Two services subscribe to the same `@on_pull_message`, the framework load-balances" | It doesn't. Pull is single-consumer. One wins; the other silently loses. |
| "I'll re-create the pull handle to recover from a transient error" | Re-creating orphans the prior handle. The recovery path must reset the FULL subscription, not just re-pull. |
| "I'll publish from `@on_init` so it's ready early" | Other services' `@on_init` may not have completed; their handlers aren't registered yet. Publish from `@on_start` at earliest, ideally after lifecycle reaches `RUNNING`. |
| "I'll use `time.sleep` to wait for the message to arrive" | This is async code. `time.sleep` blocks the event loop. Use `asyncio.Event` or `asyncio.wait_for`. |
| "I'll add a new MessageType to fix this race" | Adding messages doesn't fix subscription races. The race is between init-order and first-publish. Fix the ordering. |

## Common mistakes

- **Hand-waving `@on_init` ordering** — assume Murphy's law for cross-service init order; don't publish until the consumer is `RUNNING`.
- **Reusing a `MessageType` value for a different payload shape** — decode failures silently drop messages. Use a new enum value.
- **Forgetting that `disable_gc: true` services don't have GC** — long-lived references in their handlers leak forever.
- **Tracing with `print()` instead of `self.debug(lambda: ...)`** — `print` bypasses the structured logger and doesn't get tagged with service ID.
- **Adding a handler in a helper class** — only `BaseComponentService` instances participate in auto-subscription.

## Composition

- `aiperf-debug` for the symptom-catalog scan before deep bus tracing.
- `aiperf-add-service` for the contract when adding new message types or handlers.
- `aiperf-perf-profile` if the issue is throughput-related (bus backpressure, slow handler).
- `superpowers:systematic-debugging` for the broader root-cause methodology.
