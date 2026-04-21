# Msgspec Conversion: Credit Path

**Status:** Proposed
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20
**Part of:** [msgspec-zmq-migration-overview.md](./2026-04-20-msgspec-zmq-migration-overview.md)

## Goal

Convert the credit-phase status/config models and their envelopes to
`msgspec.Struct`. These are the progress-reporting types that ride on the
system-controller ↔ timing-manager ↔ records-manager control channels, not
the high-volume worker↔router credit plane (which is already msgspec).

## Motivation

`CreditPhaseStats` is published on every phase transition and every progress
tick — low volume but broadcast fan-out is large (TUI, API service, records
manager, system controller). Five distinct envelope messages carry phase
stats today (`CreditPhaseStart`, `CreditPhaseProgress`,
`CreditPhaseSendingComplete`, `CreditPhaseComplete`, `CreditsComplete`), each
a Pydantic wrapper around one or two Pydantic payloads.

Converting this domain:

- Eliminates the Pydantic→JSON→Pydantic roundtrip on every progress tick.
- Lets the TUI and API's progress consumers decode with a typed msgspec
  decoder instead of a generic `Message.from_json`.
- Aligns with the worker↔router credit plane's existing msgspec discipline —
  one domain, one serialization stance.

## Scope

### Converted to `msgspec.Struct`

**`src/aiperf/common/models/credit_models.py`:**

- `BasePhaseStats` — shared fields for all phase-stat types.
- `CreditPhaseStats` — extends `BasePhaseStats` with credit progress fields.
- `PhaseRecordsStats` — extends `BasePhaseStats` with records-processing
  fields.
- `ProcessingStats` — small summary struct (`processed` + `errors`).

**`src/aiperf/timing/config.py`:**

- `CreditPhaseConfig` — config for a single phase: expected counts,
  duration, sampling strategy. Currently `AIPerfBaseModel` at
  `src/aiperf/timing/config.py:95`.

**Envelopes — stay Pydantic in this spec, convert in primitives spec P2:**

- `CreditPhasesConfiguredMessage`
- `CreditPhaseStartMessage`
- `CreditPhaseProgressMessage`
- `CreditPhaseSendingCompleteMessage`
- `CreditPhaseCompleteMessage`
- `CreditsCompleteMessage`

Envelopes carry msgspec payloads via `MsgspecField` (established in records
spec R5).

### Stays Pydantic

- Nothing in this domain stays Pydantic long-term. Every type here is either
  status/config (converted) or an envelope (flipped in P2).

### Out of scope

- **Worker↔router credit-plane structs** in `src/aiperf/credit/messages.py`
  (`WorkerConnected`, `WorkerDispatchable`, `CreditReturn`, `FirstToken`,
  `TimePing/Pong`, `CancelCredits`, `InFlightReconciliation`, `InFlightReport`,
  etc.). Already msgspec, ride on a separate channel, do not inherit from
  `Message`. The credit path in this spec is specifically the progress-
  reporting path on the main message bus.
- Credit issuer / callback handler / sticky router internals
  (`src/aiperf/credit/issuer.py`, `callback_handler.py`, `sticky_router.py`)
  — unchanged; they already operate on the msgspec credit-plane structs.
- CreditPhase enum — already a string enum.

## Architecture

### `BasePhaseStats` → abstract msgspec struct

`BasePhaseStats` today uses Pydantic inheritance: `CreditPhaseStats` and
`PhaseRecordsStats` both extend it. msgspec supports struct inheritance
directly:

```python
class BasePhaseStats(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    phase: CreditPhase
    exclude_from_results: bool = False
    start_ns: int | None = None
    # ... shared timestamp + expectation + final-count fields

class CreditPhaseStats(BasePhaseStats, frozen=True, kw_only=True, omit_defaults=True):
    requests_sent: int = 0
    requests_completed: int = 0
    # ... credit-specific progress fields
```

All current `@property` methods (`is_started`, `is_sending_complete`,
`in_flight_sessions`, `requests_progress_percent`, etc.) transfer directly —
msgspec supports `@property` on structs with no change.

### Validation constraints

Current Pydantic uses `Field(ge=0, gt=0)` on most numeric fields. These are
redundant with the internal logic that produces them (phase stats are
computed by timing-manager and records-manager, not user input). Drop the
constraints at conversion.

The one exception: `phase: CreditPhase` is load-bearing for discriminator
routing elsewhere. Keep the enum type.

### Envelope field-shim

Same pattern as records and dataset: Pydantic envelope declares fields as
`Annotated[CreditPhaseStats, MsgspecField()]` and serializes through the
shim. All five phase-progress envelopes and the configured-message envelope
carry this during Phase 2. P2 flips them.

## Execution milestones

Each milestone ends with a green repo and a commit.

**C1 — `BasePhaseStats` + subclasses.** Convert `BasePhaseStats`,
`CreditPhaseStats`, `PhaseRecordsStats` to msgspec. Update construction
sites in `src/aiperf/timing/` and `src/aiperf/records/`. Verify
`tests/unit/credit/`, `tests/unit/timing/`, and
`tests/unit/records/test_records_manager.py` green.

**C2 — `ProcessingStats` + `CreditPhaseConfig`.** Convert these two
smaller types. Update timing-manager's phase-configuration logic. Verify
`tests/integration/test_timing/` and `tests/integration/test_credit_phases/`
green.

**C3 — Envelope shim retrofit.** Retrofit all six credit-progress envelopes
to use `MsgspecField` for their msgspec-typed payload fields. Verify
end-to-end progress-message flow: run a 30s benchmark, confirm every
`CreditPhaseProgress` → TUI path still works and messages decode cleanly.

## Testing

- Per milestone: `tests/unit/credit/ tests/unit/timing/
  tests/unit/records/test_records_manager.py` green + the relevant
  `tests/integration/test_credit_phases/` and `tests/integration/test_timing/`
  suites.
- End of spec: full `tests/unit/ -n auto` + `-m component_integration -n auto`
  green.
- Progress-tick parity test: capture a sequence of `CreditPhaseProgress`
  messages from a pre-migration 10s run as a fixture; replay through the
  post-migration decoder and assert structural equality (all fields match).
- Message-bus smoke: full integration run confirming no
  `Expected <Stats>, got dict` errors in logs.

## Risks

- **`@property` performance on msgspec structs.** `CreditPhaseStats` has
  several computed properties called on every progress tick (e.g.,
  `requests_progress_percent` iterates multiple percentages). msgspec
  structs do not cache property values; neither did Pydantic. No regression
  expected, but C1's commit includes a before/after micro-benchmark on
  `requests_progress_percent` with realistic field values.
- **Frozen semantics.** Current Pydantic uses `ConfigDict(frozen=True)` and
  construction by copy-with-updates. msgspec `frozen=True` forbids field
  assignment at runtime; any code path that does `stats.requests_sent += 1`
  on a `CreditPhaseStats` instance (vs. constructing a new one) breaks. C1
  audits timing-manager for this pattern — expected to be zero occurrences
  given the current `frozen=True` is already enforced by Pydantic, but grep
  explicitly.
## Non-goals

- No change to the worker↔router credit-plane messages.
- No change to the `CreditPhase` enum or its values.
- No change to phase-transition semantics or timing logic.
- No rename of any type in this scope.
