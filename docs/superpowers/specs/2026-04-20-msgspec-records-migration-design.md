# Msgspec Conversion: Records Path

**Status:** Proposed
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20
**Part of:** [msgspec-zmq-migration-overview.md](./2026-04-20-msgspec-zmq-migration-overview.md)

## Goal

Convert the inference-results path — every model that travels on the records
PUSH/PULL channel or on the records-manager control messages — to
`msgspec.Struct`. Envelopes stay Pydantic until the primitives spec; payloads
and nested records become msgspec during this spec. `ErrorDetails` embedded
inside converted payloads continues to rehydrate through the `dec_hook`
fallback until retired in P1.

## Motivation

The records channel is the hottest ZMQ surface in AIPerf. At 15k credits/s the
records-manager pull loop deserializes one `InferenceResultsMessage` per
completed request, each carrying a `RequestRecord` with nested responses and
timing. Today every record pays:

- Pydantic `model_validate` on decode (per record).
- Double allocation when a response streams as SSE — raw `SSEMessage` dataclass
  is parsed into a Pydantic `RequestRecord.responses` list.
- A latent `dec_hook` fallback on errored records (`ErrorDetails` rehydration).

Converting the records path collapses these costs without changing the wire
format perceived by downstream exporters: the JSONL output keeps its current
schema because the export seam moves to the `ProcessRecordsResult` → JSON
exporter boundary rather than the wire-decode boundary.

## Scope

### Converted to `msgspec.Struct`

**`src/aiperf/common/models/record_models.py`:**

- `MetricResult` (extends `JsonMetricResult` today — see note below)
- `ProfileResults`
- `ProcessRecordsResult`
- `RequestInfo` (already structurally compatible — no validators, small leaf)
- `RequestRecord` and every `responses` union member:
  - `TextResponse`, `BinaryResponse`, `ReasoningResponse` (currently
    `@dataclass(slots=True)` with `__pydantic_config__` — become msgspec
    structs so the discriminated union is pure msgspec)
  - `SSEField`, `SSEMessage` (currently `@dataclass(slots=True)` — same
    treatment)

**`src/aiperf/common/models/progress_models.py`:**

- `WorkerProcessingStats`
- `WorkerStats`

**Envelopes — stay Pydantic in this spec, convert in primitives spec P2:**

- `InferenceResultsMessage`
- `RealtimeMetricsMessage`
- `ProfileResultsMessage`
- `ProcessRecordsResultMessage`
- `RecordsProcessingStatsMessage`
- `AllRecordsReceivedMessage`

Envelopes carry the msgspec payloads through the field-shim pattern
established in k8s-rs dataset work: the Pydantic envelope declares the field
as `Annotated[RequestRecord, MsgspecField()]` and the shim serializes the
struct through `msgspec.to_builtins` / `msgspec.convert`.

### Stays Pydantic (final-export shape)

- `JsonMetricResult` (parent of `MetricResult`) — part of the
  `profile_export_aiperf.json` schema contract.
- `ErrorDetailsCount` — embedded in `ProfileResults.error_summary`. Converted
  in P1, not here. During this spec, `ProfileResults.error_summary` carries
  Pydantic `ErrorDetailsCount` through the field shim.
- `MetricValue` — already `@dataclass(frozen=True, slots=True)`; unchanged.

### Out of scope

- `ErrorDetails` — converted in primitives spec P1. Embedded in
  `ProcessRecordsResult.errors` and transitively elsewhere on the records
  path; stays Pydantic here, rehydrated via `dec_hook`.
- `InferenceServerResponse` protocol — structural typing only, not a concrete
  type on the wire.
- `MetricRecordInfo`, `RawRecordInfo` — already msgspec.
- Record-processor-side accumulators (`src/aiperf/records/records_manager.py`
  and related) — these are not on-wire types, and their internals can stay
  plain Python / pandas-backed where they already are.

## Architecture

### Response-union collapse

`RequestRecord.responses` is a discriminated union keyed on the `perf_ns`-
bearing leaves' type. Today it's a mix of Pydantic union discrimination and
`@dataclass` types with `__pydantic_config__ = ConfigDict(extra="forbid")` to
keep the Pydantic union matcher honest.

Post-refactor, every union member is a `msgspec.Struct, frozen=True,
tag_field="response_type", tag=<name>`. `RequestRecord` declares:

```python
class RequestRecord(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    responses: list[TextResponse | BinaryResponse | ReasoningResponse | SSEMessage]
    ...
```

msgspec resolves the union through tagged-union decoding — zero ambiguity,
zero `__pydantic_config__` workaround.

### ProfileResults / ProcessRecordsResult

Both are wrapper types that carry records lists + summary scalars to
consumers outside the hot ZMQ path (exporters, the API service). They become
msgspec structs; the JSON-export layer (`JsonExportData` in `export_models.py`)
continues to consume `ProfileResults` and produce the Pydantic export shape.
No change to the exported JSON.

### Records-channel codec

The records pull channel's codec is already `MsgspecStructCodec`. During this
spec it continues to carry the `dec_hook` that rehydrates embedded
`ErrorDetails` fields. Once every payload in this spec is msgspec, the hook
only fires on the `ProcessRecordsResult.errors: list[ErrorDetails]` path.

### Envelope field-shim

During Phase 2, Pydantic envelopes need to carry msgspec payloads. M0 lands
a reusable annotation (`MsgspecField`) in
`src/aiperf/common/models/base_models.py` for this purpose — every envelope
in every Phase 2 spec uses it on fields that hold msgspec structs. It
serializes through `msgspec.to_builtins` on `model_dump` and accepts
msgspec-Struct instances as Pydantic field values. P2 of the primitives
spec deletes the annotation entirely when every envelope flips to msgspec.

## Execution milestones

Each milestone ends with a green repo and a commit.

**R1 — Record-leaf conversion.** Convert `TextResponse`, `BinaryResponse`,
`ReasoningResponse`, `SSEField`, `SSEMessage`, `RequestInfo` to msgspec
structs. Update the inference client code that constructs them
(`src/aiperf/clients/`). Unit tests in `tests/unit/clients/` and
`tests/unit/records/` green.

**R2 — `RequestRecord` conversion.** Convert `RequestRecord` itself to
`msgspec.Struct` with the tagged-union `responses` list. Update the worker
code that emits `RequestRecord` on `InferenceResultsMessage.record`. Update
the records-manager pull loop's decode path. Verify
`tests/unit/records/test_records_manager.py` and
`tests/integration/test_worker/` green.

**R3 — `MetricResult` + result wrappers.** Convert `MetricResult`,
`ProfileResults`, `ProcessRecordsResult`. Migrate the record-processor's
output path. Verify `tests/unit/post_processors/` and
`tests/unit/exporters/` green, plus the JSONL byte-equality test for
`profile_export_aiperf.json`.

**R4 — Processing-stats conversion.** Convert `WorkerProcessingStats`,
`WorkerStats`. Update `RecordsProcessingStatsMessage` envelope to carry them
through the field shim. Verify progress-reporting integration tests green.

**R5 — Envelope shim retrofit.** Retrofit records envelopes
(`InferenceResultsMessage`, `RealtimeMetricsMessage`, `ProfileResultsMessage`,
`ProcessRecordsResultMessage`, `RecordsProcessingStatsMessage`,
`AllRecordsReceivedMessage`) to use the `MsgspecField` annotation (already
available in `common/models/base_models.py` from M0) on every msgspec-typed
field. Full spec-scope test suite green.

## Testing

- Per milestone: domain unit tests (`tests/unit/records/`,
  `tests/unit/post_processors/`, `tests/unit/exporters/`) green + the
  relevant `tests/integration/test_records*/` and
  `tests/integration/test_worker/` suites.
- End of spec: full `tests/unit/ -n auto` + `-m component_integration -n
  auto` green.
- JSONL byte-equality: new test in `tests/unit/exporters/` that asserts
  `profile_export_aiperf.json` is byte-identical (field order + value
  formatting) to a pre-migration fixture. Field-order mismatches are fixed
  by reordering the msgspec struct declaration, not by post-processing.
- Stress smoke: 60s c=1000 run against the mock server, verify zero decode
  errors in logs and no `dec_hook` firings beyond the `ErrorDetails` path.

## Risks

- **`responses` tagged-union ordering.** msgspec tagged unions match on the
  `tag` field, which must appear in the encoded payload. Existing record
  payloads do not have an explicit `response_type` tag. R1 adds the tag to
  the emitter side in the inference clients and to the consumer side
  simultaneously — since the worker-to-records-manager path is in-process for
  CI and in-deployment for production, no backward-compat window is needed.
  If a future change requires wire-format compatibility across a rolling
  upgrade, that compatibility is handled by the deployment cutover strategy
  in the primitives spec, not by a records-side dec_hook.
- **SSE hot path memory.** `SSEField` / `SSEMessage` were moved from Pydantic
  to dataclass explicitly for memory efficiency under streaming load. msgspec
  structs with `__slots__`-equivalent layout are expected to be as memory-
  efficient or better, but this milestone's commit must include a memory-use
  comparison (peak RSS on a 60s streaming run) to confirm no regression.
- **`model_validate_json` consumers.** Any code that feeds raw bytes into a
  record type via `RequestRecord.model_validate_json(...)` breaks. R2 audits
  these via `grep -rn model_validate_json src/aiperf/records src/aiperf/
  workers src/aiperf/clients` and migrates to `msgspec.json.decode`.

## Non-goals

- No change to `profile_export_aiperf.json` schema.
- No change to the JSONL records export schema.
- No change to the inference client protocol surface
  (`InferenceServerResponse`).
- No rename of any type in this scope — `RequestRecord`, `MetricResult`, etc.
  keep their names.
