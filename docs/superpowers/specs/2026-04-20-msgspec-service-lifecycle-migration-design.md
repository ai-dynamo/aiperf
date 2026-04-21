# Msgspec Conversion: Service Lifecycle

**Status:** Proposed
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20
**Part of:** [msgspec-zmq-migration-overview.md](./2026-04-20-msgspec-zmq-migration-overview.md)

## Goal

Convert the service control-plane messages — status, heartbeat, registration,
health reports, worker lifecycle events — to a fully msgspec payload form.
Envelopes stay Pydantic in this spec and flip in the primitives spec P2,
same pattern as the other domains.

## Motivation

This is the lowest-volume domain, but it covers the broadest fan-out: every
service emits `StatusMessage` / `HeartbeatMessage`, and every service
subscribes to lifecycle events. Converting last (fourth of the four domain
specs) lets patterns settle in the higher-traffic domains first and minimizes
churn — by the time this spec runs, the `MsgspecField` shim, the testing
pattern, and the envelope wiring are all established.

There is no performance argument here (these messages are infrequent); the
argument is invariant consistency. Once P3 asserts "no Pydantic on the wire,"
leaving service-lifecycle as a Pydantic island would break the assertion and
keep `JsonMessageCodec` alive.

## Scope

### Converted to `msgspec.Struct`

**`src/aiperf/common/models/service_models.py`:**

- `ServiceRunInfo` — small struct carrying service identity on registration.

**`src/aiperf/common/models/worker_models.py`:**

- `WorkerTaskStats` — worker task accounting.

**`src/aiperf/common/models/health_models.py`:**

- `NumericAggregate`
- `ProcessHealthAggregates`
- `ProcessHealth`

**Envelopes — stay Pydantic in this spec, convert in primitives spec P2:**

- `StatusMessage`
- `HeartbeatMessage`
- `RegistrationMessage`
- `MemoryReportMessage`
- `ConnectionProbeMessage`
- `WorkerHealthMessage`
- `WorkerStatusSummaryMessage`
- `WorkerPodStateMessage`
- `WorkerStartupStateMessage`
- `BenchmarkCompleteMessage`
- `BaseServiceMessage` — the common parent carrying `service_id` / `service_type`.

Envelopes use `MsgspecField` for their msgspec-typed payload fields (established
in records R5).

### Stays Pydantic

- Nothing. Every service-lifecycle type is either a payload (converted) or an
  envelope (flipped in P2).

### Out of scope

- **Service base class** (`BaseComponentService`, `BaseService`) — behavior
  unchanged; these consume msgspec messages instead of Pydantic.
- **`@on_init` / `@on_start` / `@on_stop` / `@on_message` decorators** —
  behavior unchanged; the message-dispatch path takes msgspec structs
  unchanged.
- **`AIPerfLifecycleMixin`** states — pure enum, not Pydantic.
- **Plugin system** — plugin models (`plugins.yaml` loader, plugin registry)
  are config, not on the wire. Stay Pydantic.
- **CLI command models** — config surface, not on the wire. Stay Pydantic.

## Architecture

### Enum-valued fields

Several messages carry enum values directly (`WorkerStatus`,
`WorkerStartupState`, `ServiceState`). msgspec handles string enums natively
with no special handling. No change beyond the struct-declaration syntax.

### `ProcessHealth` depth

`ProcessHealth` contains `ProcessHealthAggregates`, which contains
`NumericAggregate`. All three convert together in one milestone — splitting
them across milestones would force one of them to be both a msgspec field and
a Pydantic parent simultaneously, which is exactly the shim pattern we're
trying to minimize.

### `BaseServiceMessage` envelope pattern

`BaseServiceMessage` today is a Pydantic parent of most lifecycle messages.
During Phase 2 it stays Pydantic. At P2 it flips to a msgspec.Struct with
`tag_field="message_type"` inherited from the `Message` tagged-union root,
adding `service_id` and `service_type` fields that every lifecycle envelope
needs.

## Execution milestones

Each milestone ends with a green repo and a commit.

**S1 — Health models.** Convert `NumericAggregate`, `ProcessHealthAggregates`,
`ProcessHealth` to msgspec. Update health-reporter code that constructs them
(`src/aiperf/common/health/` or the system-controller health loop). Verify
`tests/unit/server/` (health-related) green and integration health-report
test passes.

**S2 — Worker + service payload structs.** Convert `WorkerTaskStats`,
`ServiceRunInfo` to msgspec. Update worker code that populates
`WorkerHealthMessage.worker_task_stats` and the registration payload. Verify
`tests/unit/worker*/` green and end-to-end worker registration test passes.

**S3 — Envelope shim retrofit.** Retrofit all lifecycle envelopes
(`StatusMessage`, `HeartbeatMessage`, `RegistrationMessage`,
`MemoryReportMessage`, `ConnectionProbeMessage`, `WorkerHealthMessage`,
`WorkerStatusSummaryMessage`, `WorkerPodStateMessage`,
`WorkerStartupStateMessage`, `BenchmarkCompleteMessage`) to use `MsgspecField`
on every msgspec-typed field. Full `tests/unit/ -n auto` green.

## Testing

- Per milestone: relevant unit tests + health/worker integration tests green.
- End of spec: full `tests/unit/ -n auto` + `-m component_integration -n auto`
  + `-m integration -n auto` green (integration included because lifecycle
  touches every service).
- System-controller smoke: 60s benchmark run, verify health-report cadence
  and content match a pre-migration fixture (timestamps vary, structural
  equality asserted).

## Risks

- **Enum stringification in logs.** Pydantic renders enum values via
  `__repr__` which shows the enum member name; msgspec renders via the
  string value directly. Logging output that embeds a status message may
  shift subtly. Only a log-format concern, not a functional one. S1 adjusts
  any log-assertion tests that pinned the Pydantic format.
- **`BenchmarkCompleteMessage` cancellation field.** Current payload carries
  `was_cancelled: bool`. Add no new fields during conversion; this is purely
  a type-system change. If the SystemController adds fields later (not part
  of this spec), they'll land as msgspec natively.
- **Health-report decoder startup ordering.** Health messages start flowing
  very early in service startup, before the typed codec is necessarily
  registered. S1 confirms the control-channel codec is available before the
  first health tick fires.

## Non-goals

- No change to service startup/shutdown ordering or the lifecycle state
  machine.
- No change to `@on_message` / `@on_command` / `@on_request` decorator
  semantics.
- No change to ZMQ socket topology.
- No new lifecycle message types. If SystemController needs a new signal
  later, it's a separate change.
- No rename of any type in this scope.
