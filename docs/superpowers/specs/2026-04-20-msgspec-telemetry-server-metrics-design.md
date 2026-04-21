# Msgspec Conversion: GPU Telemetry + Server Metrics

**Status:** Approved, in progress
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20

## Goal

Replace Pydantic with msgspec.Struct for all in-flight and accumulator-state
types in the GPU telemetry and server metrics paths. Final-export JSON models
(the shape written to `profile_export_aiperf.json`) stay Pydantic, since those
are the stable artifact contract.

## Motivation

The RECORDS push/pull channel is decoded via a typed msgspec decoder
(`RECORDS_CODEC`). Pydantic models cannot be msgspec-encoded, which has already
forced two workaround commits (`fix(server-metrics): route records through
msgspec wire envelope`, `fix(telemetry): route GPU telemetry through msgspec
wire envelope`) that carry data through intermediate dict/struct envelopes and
rehydrate Pydantic at the boundary. This is double-work and double-maintenance.
Making the records themselves msgspec removes the seam and the rehydration cost
on the hottest control-plane path.

## Scope

### Converted to `msgspec.Struct`

**GPU telemetry (`src/aiperf/common/models/telemetry_models.py`):**
- `TelemetryMetrics` — the DCGM-field container
- `GpuMetadata` — gpu identity + location
- `TelemetryRecord` — on-wire record (subclass of `GpuMetadata` today)
- `GpuTelemetrySnapshot` — snapshot helper
- `GpuTelemetryData` — accumulator state (metadata + time series)
- `TelemetryHierarchy` — accumulator root

**Server metrics (`src/aiperf/common/models/server_metrics_models.py`):**
- `MetricSample` — single prometheus sample (with histogram buckets)
- `MetricFamily` — group of samples for one metric
- `ServerMetricsRecord` — on-wire record
- `SlimRecord` — JSONL-export form, also emitted by the record processor

Note: the actual accumulator state (`ServerMetricsHierarchy`,
`ServerMetricsTimeSeries`, `ScalarTimeSeries`, `HistogramTimeSeries`) is
already plain Python classes backed by numpy — not Pydantic — and does not
need conversion. The `*Series`, `*Stats`, `*Timeslice` classes originally
listed in the "accumulator state" bucket are actually building blocks of the
final Pydantic export models (`GaugeMetricData.series: list[GaugeSeries]`,
etc.), so keeping them Pydantic preserves the JSON-schema seam cleanly;
msgspec-ifying them would force the enclosing export models to become
msgspec as well, which is out of scope per the middle-ground decision.

### Stays Pydantic (final-export shape)

**GPU telemetry export models:**
- `GpuSummary`, `EndpointData`, `TelemetryExportData`, `TelemetrySummary`
- `ProcessTelemetryResult` (result-pipeline wrapper)

**Server metrics export models:**
- `BaseServerMetricData`, `GaugeMetricData`, `CounterMetricData`,
  `HistogramMetricData`
- `ServerMetricsExportData`, `ServerMetricsEndpointSummary`,
  `ServerMetricsResults`
- `ProcessServerMetricsResult`

### Out of scope

- `ErrorDetails` — shared across the whole codebase; wire-side continues to
  use `WireErrorDetails` + `_error_to_wire`/`_wire_to_error`.
- `AIPerfBaseModel` itself.
- Inference-metric records (`metric_records_wire.py`) — already msgspec.
- Any trace-data conversion — already done in `f45629e88`.

## Architecture

### Record wire collapse

The `*_wire.py` files currently define parallel msgspec mirrors of the Pydantic
records. After this refactor:

- `TelemetryRecord` **is** the msgspec.Struct pushed on the wire. There is no
  `TelemetryRecordWireData`. The batch envelope (`TelemetryRecordsWireMessage`)
  carries `records: tuple[TelemetryRecord, ...]` directly.
- `ServerMetricsRecord` **is** the msgspec.Struct pushed on the wire. There is
  no `dict[str, Any]` fallback on the envelope.

Wire envelopes continue to exist as the tagged union members registered with
`RECORDS_CODEC.decode_type` — they just no longer do any type conversion, and
`build_*` helpers become trivial constructors (kept for backward-compat call
sites but slated for removal).

### Accumulator storage

- GPU: `TelemetryHierarchy.dcgm_endpoints: dict[str, dict[str, GpuTelemetryData]]`
  stays as-is, but the struct values are msgspec. `GpuMetricTimeSeries` is
  already a plain class with numpy buffers — unchanged, held as a regular
  attribute on the msgspec struct.
- Server metrics: the `storage.py` time-series store continues to hold series
  objects; the series objects become msgspec structs. `storage.py`'s own
  classes (dataclass/plain-python) may not need to change.

### Pydantic ↔ msgspec seam

Lives in the exporters (`gpu_telemetry/accumulator.py::export_results`,
`server_metrics/accumulator.py::export_results`) and in the realtime publishing
mixin. These are the only places that build the Pydantic export models from
msgspec accumulator state. All transformations kept in one place; easy to test.

### Mutability

Server-metrics accumulator mutates series state in place (appending samples,
incrementing counters). Those structs use `frozen=False` + `kw_only=True`.
Records and metadata stay `frozen=True`. This is the standard msgspec
performance-ergonomics tradeoff.

### Removed helpers

- `TelemetryRecordWireData`, `_record_to_wire`, `_wire_to_record`,
  `build_telemetry_records_wire_message` (consumers pass `TelemetryRecord`
  tuples directly).
- `build_server_metrics_record_wire_message`, `server_metrics_record_from_wire`
  (consumers pass `ServerMetricsRecord` directly).

Retained under the same names for transitional back-compat only where they
appear in many unit tests.

## Execution milestones

Each milestone ends with a green repo and a commit.

**M1 — Server-metrics records on-wire.** Convert `ServerMetricsRecord`,
`MetricFamily`, `MetricSample` to msgspec. Rewrite
`server_metrics_records_wire.py` to carry the native struct. Update the
server-metrics manager push path and records-manager pull path. Fix unit tests
(`tests/unit/server_metrics/`) and component-integration (`tests/integration/
test_server_metrics/`).

**M2 — Server-metrics accumulator internals.** Convert series/stats/timeslice
types to msgspec. Update `server_metrics/accumulator.py` and `storage.py`. The
Pydantic seam lives at `export_results()` → `ServerMetricsExportData`.

**M3 — GPU telemetry records on-wire.** Convert `TelemetryMetrics`,
`GpuMetadata`, `TelemetryRecord`, `GpuTelemetrySnapshot` to msgspec. Collapse
`TelemetryRecordWireData`. Update DCGM/pynvml collectors, manager push,
records-manager pull, JSONL writer. Fix unit and integration telemetry tests.

**M4 — GPU telemetry accumulator internals.** Convert `GpuTelemetryData`,
`TelemetryHierarchy` to msgspec. Update `gpu_telemetry/accumulator.py`.
Pydantic seam at `export_results()` → `TelemetryExportData`.

**M5 — Message envelopes + realtime publishing cleanup.** Convert
`TelemetryRecordsMessage`, `ServerMetricsRecordMessage` and any remaining
service-to-service wrappers in these paths to msgspec. Update realtime
telemetry mixin. Final test-suite green.

## Testing

- Per-milestone: `uv run pytest tests/unit/{server_metrics,gpu_telemetry}/ -n auto`
  and the relevant `tests/integration/test_{server_metrics,gpu_telemetry}/`
  suites must be green before the commit.
- End: full `tests/unit/ -n auto` + `-m component_integration -n auto` +
  `-m integration -n auto` passes.
- JSONL exporters: new test confirming the on-disk schema is byte-identical
  (field order/type) to the pre-refactor schema so downstream tooling is not
  broken.

## Risks

- **Test churn.** Roughly 30-50 test-file touches across the two accumulators.
  Most are mechanical: `Model(...)` becomes `Struct(...)`; `model_validate`
  becomes `msgspec.convert`; `model_dump` becomes `msgspec.structs.asdict`.
- **`extra="forbid"` loss.** Pydantic's extra-field rejection doesn't carry
  over to msgspec (which rejects unknown fields by default, so this is
  equivalent — verified).
- **Field descriptions.** `Field(description=...)` metadata disappears.
  Docstrings moved to class body / inline comments where the description
  explained domain semantics (not just restating the field name).
- **JSON schema drift.** The export models remain Pydantic, so the
  `profile_export_aiperf.json` schema is unchanged. JSONL records shift from
  Pydantic-emitted to msgspec-emitted JSON — asserted byte-identical in tests.

## Non-goals

- No change to `AIPerfBaseModel` or any config model.
- No change to the CLI option surface.
- No change to plugin registration.
- No performance-optimization work beyond what falls out of the conversion.
