# WorkerGroupManager model boundary design

## Goal

Align the new WorkerGroupManager architecture with the repo's model-boundary rules:
- models that go over high-frequency runtime wire paths should use `msgspec.Struct`
- local-only models should use `@dataclass(slots=True)` when they do not need Pydantic behavior

This work is scoped mostly to models introduced as part of the WorkerGroupManager replacement.

## Scope

In scope:
- new WorkerGroupManager wire contracts on group-local lifecycle/ZMQ-style paths
- new WorkerGroupManager runtime coordination models used only in process memory
- adjacent new models introduced specifically for the WorkerGroupManager work where the boundary is clear

Out of scope:
- broad repo-wide migration of existing API/export/Kubernetes models
- unrelated `AIPerfBaseModel` conversions outside the WorkerGroupManager change set
- reworking existing long-lived Pydantic config models that are not part of this replacement

## Boundary rules

### Wire / runtime messaging models

Use `msgspec.Struct` for models that cross high-frequency runtime boundaries, especially:
- group-local lifecycle transport
- pod/group-local query/response messages
- ZMQ/internal runtime message payloads introduced by this spec work

These models should be optimized for:
- low allocation overhead
- fast encode/decode
- explicit tagged/typed payloads where needed

### Local-only models

Use `@dataclass(slots=True)` for models that remain in local memory only, especially:
- group-manager child state
- group capacity snapshots used only in process memory
- adapter-local bookkeeping models

These models should not inherit from `AIPerfBaseModel` unless they genuinely need Pydantic validation/serialization behavior.

## Proposed conversions

### Group-local wire contracts

Models in the group lifecycle contract should be moved toward `msgspec.Struct` rather than `AIPerfBaseModel` or plain dataclass forms if they are sent over runtime transport.

Examples include renamed group-local equivalents of:
- dataset state query/snapshot messages
- child command / command ack messages
- child startup/health update messages if they travel over the group-local wire

### Group-manager local state

Models owned purely by `WorkerGroupManager` internals should be `@dataclass(slots=True)`.

Examples include:
- child state tracking
- declared capacity tracking
- local adapter bookkeeping

## Design intent

This keeps the new WorkerGroupManager boundary clean:
- transport models are explicit wire structs
- in-memory orchestration state is lightweight local data
- Pydantic remains for config/API/export surfaces that benefit from schema behavior

## Non-goals

- converting all existing worker/controller messages in one pass
- changing public API/export schemas just to satisfy the model-boundary cleanup
- introducing compatibility wrappers between old and new model types

## Testing

Add focused tests that verify:
- group-local wire models still round-trip through the runtime messaging path expected by the new code
- local-only models retain the same behavior after dataclass conversion
- no consumer still relies on Pydantic-only APIs for converted local-only models

## Expected outcome

The newly introduced WorkerGroupManager model surface follows a clean rule:
- runtime wire path -> `msgspec.Struct`
- local-only orchestration state -> `@dataclass(slots=True)`

without expanding into a repo-wide migration.