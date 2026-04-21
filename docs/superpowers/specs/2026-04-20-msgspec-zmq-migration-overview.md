# Msgspec Conversion: ZMQ Migration Overview

**Status:** Proposed
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20

## Goal

Convert every model that crosses a ZMQ boundary in AIPerf from Pydantic
(`AIPerfBaseModel` / `BaseModel`) to `msgspec.Struct`, and retire the Pydantic
`Message` base class along with all Pydantic-aware codec paths. End state: the
only serialization path for inter-service traffic is a typed msgspec codec with
no `dec_hook` / `enc_hook` fallbacks and no Pydantic imports in
`src/aiperf/common/messages/` or the codec layer.

## Motivation

Three seams have accumulated in the codebase that all point at the same thing:

1. **`dec_hook` / `enc_hook` workarounds in the records codec** rehydrate
   embedded Pydantic fields (notably `MetricRecordsData.error: ErrorDetails`)
   into and out of otherwise-typed msgspec structs. Latent until a sustained
   error rate triggers them; fails closed when it does.
2. **`*_wire.py` twin types** in telemetry and server-metrics mirror each
   Pydantic record with a msgspec equivalent, plus paired `_to_wire` /
   `_from_wire` helpers. Double-maintenance; every schema change happens twice.
3. **`PydanticMsgpackCodec`** wraps Pydantic models in msgpack, but still
   `model_dump`s on encode and `model_validate`s on decode — we pay msgpack
   framing without the speedup.

The 2026-04-20 telemetry + server-metrics spec removed the twin types and the
`_to_wire` glue for its scope. This overview extends the same decision to the
rest of the message surface: records, dataset, credit, service lifecycle, and
the cross-cutting primitives (`ErrorDetails`, `Message` base).

Per-hop throughput, Rust-port parity (aiperf-rs consumes these types 1:1 over
the same ZMQ channels), and elimination of the Pydantic↔msgspec seam are all
solved by the same conversion; they are not separable work.

## Document set

Six specs under `docs/superpowers/specs/`:

| Spec | Scope |
|---|---|
| `msgspec-zmq-migration-overview.md` (this doc) | Sequencing, shared conventions, M0 port-forward, status matrix |
| `msgspec-records-migration-design.md` | Inference results path: `RequestRecord`, `ProfileResults`, `ProcessRecordsResult`, `MetricResult`, records envelopes |
| `msgspec-dataset-migration-design.md` | Dataset path: k8s-rs port-forward + remaining metadata types + dataset envelopes |
| `msgspec-credit-migration-design.md` | Credit phase stats (`CreditPhaseStats`, `PhaseRecordsStats`, `CreditPhaseConfig`) + five `CreditPhase*Message` envelopes |
| `msgspec-service-lifecycle-migration-design.md` | Service control messages: `StatusMessage`, `HeartbeatMessage`, `WorkerHealthMessage`, etc. |
| `msgspec-primitives-migration-design.md` | Terminal work: `ErrorDetails`, `Message` base-class flip, codec retirement, `dec_hook` deletion |

## Execution phases

```
Phase 1 (serial)          Phase 2 (parallelizable)            Phase 3 (serial, terminal)
─────────────────────     ─────────────────────────────       ──────────────────────────────
M0 port-forward      ──▶  Records spec    ───────────┐
from ajc/aiperf-rs-k8s    Dataset spec    ───────────┤
                          Credit spec     ───────────┼──▶    Primitives spec
                          Service spec    ───────────┘        (ErrorDetails → Message flip → codec cleanup)
```

### Phase 1 — M0 port-forward (prerequisite)

Before any domain spec starts, two commits land from `ajc/aiperf-rs-k8s` onto
this branch, followed by one small generalization step:

- `073cc3011 perf(dataset): Conversation/Turn/media → msgspec.Struct` — the
  hot-path dataset conversion. Dataset spec picks up from this baseline.
- `59e0a900f fix(codecs): dec_hook rehydrates Pydantic fields inside msgspec
  structs` — the fallback that lets the records codec decode errored records
  carrying Pydantic `ErrorDetails`. Correct-and-necessary until the primitives
  spec retires it.
- **`MsgspecField` generalization.** k8s-rs's `ConversationShim` is a
  one-off Pydantic-field adapter for a msgspec-Struct payload. M0 generalizes
  it into a reusable `MsgspecField` annotation in
  `src/aiperf/common/models/base_models.py` (~30 lines of code). This
  unblocks every Phase 2 spec to retrofit its own envelopes without
  inter-spec ordering constraints. `ConversationShim` stays in place for its
  specific `Conversation` field during Phase 2; dataset D3 removes it in
  favor of the generalized annotation.

Conflicts during cherry-pick are resolved in favor of the k8s-rs
implementation where they overlap, since that code has stress-test
coverage. Single commit (or small stacked sequence), full test suite green
before any Phase 2 work starts.

### Phase 2 — Domain specs (parallelizable after M0)

Records, Dataset, Credit, and Service-lifecycle are internally independent and
can be executed in any order or in parallel. Each spec is self-contained:
converts payloads, keeps envelopes Pydantic using the msgspec-field-shim
pattern (see Conventions), ends with a green repo and a merged PR.

Recommended internal order (not enforced):

1. **Records** — highest volume, most complex payloads; flushes out edge cases
   the other three can reuse.
2. **Credit** — tightly coupled with records-manager; benefits from records
   patterns.
3. **Dataset** — M0 already handles the hot path; remaining work is metadata
   and envelopes.
4. **Service-lifecycle** — simplest payloads; good cooldown before Phase 3.

### Phase 3 — Primitives spec (terminal)

Starts only after all four domain specs are merged. Three sub-milestones:

- **P1:** Convert `ErrorDetails`, `ErrorDetailsCount`, `ExitErrorInfo` to
  msgspec in place — keep class names, keep construction call sites textually
  unchanged. Retire the `dec_hook` entry path that rehydrates
  `ErrorDetails` from dicts.
- **P2:** Flip `Message` base class to `msgspec.Struct`, replacing
  `AutoRoutedModel`'s custom discriminator machinery with msgspec tagged
  unions (`tag_field="message_type"`). Every `@on_message` handler, every
  codec call site, every service test touches here.
- **P3:** Delete `JsonMessageCodec`, `PydanticMsgpackCodec`, `_enc_hook`,
  `_dec_hook`. Collapse `message_codecs.py` to a single `MsgspecStructCodec`
  parameterized by the tagged-union type. Assert zero Pydantic imports under
  `src/aiperf/common/messages/` and in the codec layer.

## Shared conventions

These apply to every domain spec. They are not re-stated in the per-domain
docs — call them out there only when a domain deviates.

### Struct configuration

- **Frozen, on-wire records:** `msgspec.Struct, frozen=True, kw_only=True,
  omit_defaults=True`. Matches the pattern established in
  `MetricRecordInfo`, `GpuMetadata`, `TelemetryRecord`.
- **Mutable accumulator state:** `msgspec.Struct, kw_only=True` (drop
  `frozen`). Used where the struct is updated in place (e.g., phase stats that
  accumulate across a run). Matches `GpuTelemetryData`.
- **`omit_defaults=True`** on every struct that crosses the wire — keeps
  payload size close to the current `exclude_none=True` JSON behavior.
- **`kw_only=True` always** — call sites become unambiguous and match the
  Pydantic ergonomics they're replacing.

### Field annotation migration

- `Field(description="...")` → docstring under the class, or inline comment
  only when the description conveys a non-obvious domain constraint. Field
  descriptions that just restate the field name are dropped.
- `Field(ge=0, gt=0, default=...)` → default stays (`: int = 0`); validation
  constraints (`ge`, `gt`, `le`, `lt`) are dropped unless the validation is
  load-bearing at a trust boundary, in which case use
  `Annotated[int, msgspec.Meta(ge=0)]`. Most of these are redundant with
  upstream logic and can go.
- `Optional[X]` stays `X | None` (already our convention).
- `ClassVar[...]` is honored by msgspec — no change needed.
- `@computed_field` properties: msgspec supports `@property` and
  `@cached_property` directly. No conversion needed; they just work.

### Discriminated unions

Messages currently route via `AutoRoutedModel`'s custom `__init_subclass__`
registry keyed on `discriminator_field = "message_type"`. The msgspec
equivalent is:

```python
class Message(msgspec.Struct, tag_field="message_type", tag=str.lower):
    ...
```

Domain specs that introduce new message types do so under this pattern once
the primitives spec lands. Until then, each domain keeps its Pydantic envelope
and converts only the payload; the envelope flip happens centrally in P2 so
the whole tagged union is registered in one place.

### Codec strategy during migration

During Phase 2, the records codec continues to use `MsgspecStructCodec` with
the `dec_hook` fallback (carried forward from M0). As payloads convert, the
`dec_hook` fires less often. After P1 retires `ErrorDetails`, the hook is
unreachable; P3 deletes it.

Other channels continue using `JsonMessageCodec` with Pydantic messages until
P2 flips the base class, at which point every channel moves to a single
`MsgspecStructCodec` parameterized by the `Message` tagged union.

### Testing rigor (per domain spec)

- Per-milestone: domain unit tests + relevant component-integration tests
  green before the milestone commit.
- Per spec: full `tests/unit/ -n auto` + `-m component_integration -n auto`
  green before merging to `main`.
- At P3 completion: full integration suite (`-m integration -n auto`) must
  pass on the combined branch.
- JSONL export outputs (records, telemetry, server-metrics) must be
  byte-identical to pre-migration output. Each domain spec owns a
  byte-equality test for any exporter it touches.

## Status matrix

Filled as specs land. Each cell = spec status + PR link.

| Domain | Status | Spec | PRs |
|---|---|---|---|
| M0 port-forward | Not started | (this doc) | — |
| Records | Not started | `msgspec-records-migration-design.md` | — |
| Dataset | Not started | `msgspec-dataset-migration-design.md` | — |
| Credit | Not started | `msgspec-credit-migration-design.md` | — |
| Service lifecycle | Not started | `msgspec-service-lifecycle-migration-design.md` | — |
| Primitives (terminal) | Not started | `msgspec-primitives-migration-design.md` | — |

## Non-goals

- No change to `AIPerfBaseModel`. It remains the Pydantic base for config,
  final-export JSON models, and CLI surfaces.
- No change to `profile_export_aiperf.json` schema. Export models
  (`JsonExportData`, `TelemetryExportData`, `ServerMetricsExportData`, etc.)
  stay Pydantic; the Pydantic↔msgspec seam lives at the exporter boundary.
- No change to CLI option surface or plugin registration.
- No Rust-side (aiperf-rs) changes. The migration makes Python-side types
  Rust-compatible; the Rust consumer work is separate.
- No performance-optimization beyond what falls out of the conversion. If a
  milestone's benchmark shows regression, the fix is scope for that
  milestone; speculative optimizations are not.

## Risks

- **P2 size.** The `Message` base-class flip is the largest single PR in this
  plan. Every `@on_message` handler, every service test, every codec touches.
  Mitigation: P2 lands in a dedicated integration branch, rebased clean onto
  `main` after each domain spec merges to minimize conflict surface.
- **JSONL byte-equality.** msgspec JSON emits fields in declaration order with
  slightly different defaults handling (`omit_defaults` vs Pydantic's
  `exclude_none`). Domain specs own per-file byte-equality tests; expect to
  adjust field order on the msgspec struct to match the Pydantic output where
  downstream tooling is schema-sensitive.
- **`ErrorDetails` blast radius.** Hundreds of construction sites. Mitigation:
  P1 is purely type-conversion-in-place — no rename, no signature change, so
  the diff is dominated by `class ErrorDetails(AIPerfBaseModel)` →
  `class ErrorDetails(msgspec.Struct, ...)` and `Field(...)` removal at the
  definition site. Call sites stay textually identical.
- **`AutoRoutedModel` feature parity.** Its hierarchical discriminator pattern
  (parent defines discriminator, nested children re-route on the same field)
  maps cleanly to msgspec tagged unions with inheritance. The one feature that
  doesn't translate is `from_json(cls, data)` accepting either bytes or a
  pre-parsed dict; msgspec always wants bytes. P2 audits call sites that pass
  dicts and routes them through `msgspec.convert` instead.

## Out of scope

- Worker↔router credit-plane structs in `src/aiperf/credit/messages.py` —
  already msgspec, not routed through the main `Message` bus.
- Inference-metric records (`metric_records_wire.py`) — already msgspec.
- Trace data — already converted in `f45629e88`.
