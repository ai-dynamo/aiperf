# Msgspec Conversion: Dataset Path

**Status:** Complete (2026-04-21)
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20
**Commits:** `e1cdd1233` (D1), `a2e8bb599` (D2), `0c767fee4` (D3+D4)
**Part of:** [msgspec-zmq-migration-overview.md](./2026-04-20-msgspec-zmq-migration-overview.md)

## Goal

Finish the dataset-path conversion to `msgspec.Struct` across every model that
travels on the dataset request/response and dataset-notification channels.
Starts from the k8s-rs port-forward baseline (M0 in the overview) and
completes the remaining Pydantic metadata models and envelopes.

## Motivation

`ajc/aiperf-rs-k8s` already converted the hot-path dataset types (`Conversation`,
`Turn`, `Media`, `Text`/`Image`/`Audio`/`Video`) to msgspec — 3-4x speedup on
the per-request instantiation path. The remainder of the dataset surface
stayed Pydantic because it was out of scope for that narrow perf fix:
metadata wrappers, client-metadata discriminated unions, and the dataset
message envelopes.

This spec finishes the job. By the end:

- Every model under `common/models/dataset_models.py` is msgspec, not just
  the hot-path leaves.
- The `ConversationShim` Pydantic↔msgspec adapter in k8s-rs is retired
  (the envelope side moves to the generalized `MsgspecField` shim from the
  records spec during Phase 2, and disappears entirely in P2 of the
  primitives spec).

## Scope

### Already converted (by M0 port-forward from k8s-rs)

These land on this branch via M0 before this spec begins work:

- `Conversation`
- `Turn`
- `Media`, `Text`, `Image`, `Audio`, `Video`
- `ConversationShim` (transitional Pydantic-compat adapter)

### Converted by this spec

**`src/aiperf/common/models/dataset_models.py`:**

- `TurnMetadata`
- `ConversationMetadata`
- `DatasetMetadata`
- `DatasetClientMetadata` (parent of the discriminated union; becomes a
  msgspec tagged union with `tag_field="client_type"`)
- `MemoryMapClientMetadata` (the only subclass today; becomes a tagged
  union member)
- `SessionPayloads`
- `InputsFile`

**Envelopes — stay Pydantic in this spec, convert in primitives spec P2:**

- `ConversationRequestMessage`
- `ConversationResponseMessage`
- `ConversationTurnRequestMessage`
- `ConversationTurnResponseMessage`
- `DatasetConfiguredNotification`
- `DatasetDownloadedNotification`

Envelopes carry msgspec payloads through the `MsgspecField` shim formalized
in the records spec (R5).

### Stays Pydantic

- No models in this domain stay Pydantic long-term. Every dataset type is
  either hot-path (already msgspec via M0) or metadata (converted by this
  spec). Dataset consumers on the API-output side build their responses from
  msgspec structs through the export seam, same pattern as records.

### Out of scope

- Dataset client implementations
  (`src/aiperf/dataset/clients/memory_map/` etc.) — behavior unchanged;
  they produce and consume msgspec structs instead of Pydantic.
- Dataset synthesis / generation code — internal logic unchanged.
- Memory-map file format — the on-disk format is its own contract and is not
  changing. Only the in-memory types that wrap its records change.

## Architecture

### `DatasetClientMetadata` tagged union

Today:

```python
class DatasetClientMetadata(AIPerfBaseModel):
    discriminator_field: ClassVar[str] = "client_type"
    client_type: DatasetClientStoreType = Field(...)

class MemoryMapClientMetadata(DatasetClientMetadata):
    client_type: DatasetClientStoreType = DatasetClientStoreType.MEMORY_MAP
    ...
```

Becomes:

```python
class DatasetClientMetadata(msgspec.Struct, tag_field="client_type", kw_only=True):
    ...

class MemoryMapClientMetadata(DatasetClientMetadata, tag=DatasetClientStoreType.MEMORY_MAP):
    ...
```

Consumers that pattern-match on `client_type` continue to work; msgspec
routes the union on decode.

### Metadata vs. hot-path split

The hot-path types (`Conversation`, `Turn`, media leaves) are `frozen=True`
because they flow on every request. Metadata types (`DatasetMetadata`,
`ConversationMetadata`, `TurnMetadata`) are `frozen=True` too — they're
constructed once per dataset load and never mutated. No accumulator-state
types in this domain.

### Retirement of `ConversationShim`

k8s-rs's `ConversationShim` (dataset_models.py:75 on that branch) exists to
let Pydantic v2 parents carry a `msgspec.Struct` field. After M0 port-forward,
it lives in this branch — alongside the generalized `MsgspecField`
annotation that M0 also lands. During this spec's D1-D2, dataset envelopes
continue to use the specific `ConversationShim` on their
`Conversation` / `Turn` fields. D3 replaces those call sites with
`MsgspecField` and deletes `ConversationShim`. P2 of the primitives spec
deletes `MsgspecField` itself when every envelope flips to msgspec.

## Execution milestones

Each milestone ends with a green repo and a commit.

**D1 — Metadata-leaf conversion.** Convert `TurnMetadata`,
`ConversationMetadata`, `DatasetMetadata` to msgspec. Update constructors
in `src/aiperf/dataset/` and `src/aiperf/dataset_manager/`. Verify
`tests/unit/dataset/` green.

**D2 — `DatasetClientMetadata` tagged union.** Convert the
`DatasetClientMetadata` base and `MemoryMapClientMetadata` subclass to a
msgspec tagged union. Update dataset-manager code that constructs or
routes on `client_type`. Verify client-metadata serialization tests green
and the end-to-end memory-map integration test passes.

**D3 — `InputsFile` / `SessionPayloads` + shim consolidation.** Convert
`InputsFile` and `SessionPayloads`. Remove the k8s-rs-origin
`ConversationShim` definition; every prior use site adopts the generalized
`MsgspecField` annotation (landed in M0). Retrofit dataset envelopes
(`ConversationRequestMessage`, `ConversationResponseMessage`,
`ConversationTurnRequestMessage`, `ConversationTurnResponseMessage`,
`DatasetConfiguredNotification`, `DatasetDownloadedNotification`) to use
`MsgspecField` on every msgspec-typed field.

**D4 — Dataset synthesis audit.** Scan `src/aiperf/dataset/synthesis/` and
any dataset-generation code for construction sites that still hand back
Pydantic types, or that use `model_dump` / `model_validate` on dataset
models. Migrate to msgspec equivalents. This is cleanup-oriented; expected
to be small once D1-D3 are in.

## Testing

- Per milestone: domain unit tests (`tests/unit/dataset/`,
  `tests/unit/dataset_manager/`) green + the relevant
  `tests/integration/test_dataset*/` suites.
- End of spec: full `tests/unit/ -n auto` + `-m component_integration -n
  auto` green.
- JSONL export: any dataset-describing JSON that lands in artifacts must be
  byte-identical to a pre-migration fixture. Field-order parity enforced by
  struct declaration order.
- Large-dataset smoke: run a 100k-conversation dataset load through
  `AIPerfDatasetManager`, verify construction throughput does not regress
  vs. the M0-baseline measurement (k8s-rs showed 3-4x speedup; this spec
  must not give it back).

## Risks

- **`DatasetClientMetadata` single-subclass regression.** The union is
  currently single-member. msgspec tagged unions with one variant are valid
  but unusual. Future subclasses must remember to set the `tag=` class
  argument. D2 adds an `__init_subclass__` check (or a lint test) that fails
  CI if a subclass omits the tag.
- **`model_dump` consumers.** The dataset HTTP API surface (if any) that
  serializes `DatasetMetadata` / `Conversation` through `model_dump_json`
  will break. D1 audits API-service code for dataset-type serialization and
  migrates to `msgspec.json.encode`.
- **Port-forward conflict.** M0 may conflict with recent work on this branch's
  `dataset_models.py`. M0 resolution is in the overview's guidance (favor
  k8s-rs where overlap exists); this spec depends on a clean M0 landing.

## Non-goals

- No change to the memory-map on-disk format.
- No change to the dataset synthesis API (`docs/api/synthesis.md` surface).
- No change to the `DatasetClientStoreType` enum membership.
- No rename of any dataset type.
