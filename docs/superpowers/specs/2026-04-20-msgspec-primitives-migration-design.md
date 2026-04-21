# Msgspec Conversion: Primitives (Terminal)

**Status:** Complete (P1+P2+P3 landed)
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-20
**Commits so far:** `41e53697e` (P1), `33c149e6b`+`f05c0abcb`+`8899602cf`+`409538d27`+`26fadd851` (P2), `d2fb06a60` (P3)
**Part of:** [msgspec-zmq-migration-overview.md](./2026-04-20-msgspec-zmq-migration-overview.md)

## Completed work

- **P2 (landed)** — `Message` base class is now `msgspec.Struct(tag_field="message_type")`;
  every envelope across `base_messages`, `service_messages`, `worker_messages`,
  `progress_messages`, `inference_messages`, `dataset_messages`,
  `telemetry_messages`, `server_metrics_messages`, and `credit/messages.py` is a
  tagged-union member. `AutoRoutedModel` deleted; `AIPerfBaseModel` detached to
  plain `BaseModel`. `PydanticStructMixin` retired from every struct except
  seven still referenced by Pydantic parents (`ErrorDetailsCount`,
  `ProcessRecordsResult`, `WorkerStats`, `BasePhaseStats`, `CreditPhaseConfig`,
  `Media` hierarchy, `Turn`/`Conversation`). `JsonMessageCodec` and
  `PydanticMsgpackCodec` rewritten on top of msgspec but kept API-compatible;
  P3 collapses them.
- **P3 (landed)** — Codec collapse + invariant assertion. Deleted
  `JsonMessageCodec`, `PydanticMsgpackCodec`, the codec `_enc_hook` /
  `_dec_hook`. `MsgspecStructCodec` is the only codec; `get_message_codec()`
  (a lazy singleton parameterized by `Message._union_type()`) replaces
  `JSON_MESSAGE_CODEC` as the default for push/pull transport. Wire format
  changed from JSON to msgpack on the Message bus. Added
  `tests/unit/test_no_pydantic_on_wire.py` that greps
  `src/aiperf/common/messages/` for `pydantic` imports and
  `model_dump`/`model_validate` usage to lock in the invariant.

> **P3 landed:** every ZMQ channel now uses msgpack-over-msgspec. Live
> deployments must roll all services together — mixed-version clusters will
> not interoperate during the cutover.

## Goal

Finish the ZMQ migration. Convert the remaining cross-cutting Pydantic
primitives (`ErrorDetails`, `ErrorDetailsCount`, `ExitErrorInfo`), flip the
`Message` base class from Pydantic to `msgspec.Struct` with tagged-union
routing, and retire every codec path that touched Pydantic. End state: the
only serialization path for ZMQ traffic is `MsgspecStructCodec`, no
`dec_hook` / `enc_hook` fallbacks exist, and no file under
`src/aiperf/common/messages/` or the codec layer imports from `pydantic`.

## Motivation

The four Phase 2 domain specs convert every payload that crosses the wire.
What remains is:

1. **`ErrorDetails`** — embedded in ~every error-carrying message and record.
   Phase 2 leaves it Pydantic; converting it unblocks deletion of the
   `dec_hook` rehydration fallback.
2. **`Message` base class** — still Pydantic / `AutoRoutedModel`. Every
   envelope in every Phase 2 domain spec uses the `MsgspecField` shim to
   carry msgspec payloads inside a Pydantic envelope. Flipping `Message`
   itself retires the shim, retires `AutoRoutedModel`, and collapses the
   codec layer.
3. **Codec layer** — `JsonMessageCodec`, `PydanticMsgpackCodec`, `_enc_hook`,
   `_dec_hook`. Each exists for a Pydantic-era compatibility reason; with
   every payload and every envelope msgspec, none have a purpose.

This is the terminal cleanup that makes the invariant true: no Pydantic on
the wire.

## Scope

### Converted to `msgspec.Struct`

**`src/aiperf/common/models/error_models.py`:**

- `ErrorDetails` — in place, same class name, same field names, same
  construction ergonomics. Every call site stays textually identical.
- `ErrorDetailsCount`
- `ExitErrorInfo`

**`src/aiperf/common/messages/base_messages.py`:**

- `Message` — from `AIPerfBaseModel` to `msgspec.Struct` with
  `tag_field="message_type"`.
- `RequiresRequestNSMixin` — becomes a msgspec.Struct with the
  `request_ns: int` required field, usable as an additional parent via
  msgspec's struct inheritance.
- `ErrorMessage` — envelope carrying `ErrorDetails`; becomes a tagged-union
  member.

**All other envelopes** — every message in the four Phase 2 domain specs
flips from Pydantic (with `MsgspecField` shims) to `msgspec.Struct`
(tagged-union members). This is the biggest change surface in this spec by
file count; mechanically uniform.

**`src/aiperf/common/message_codecs.py`:**

- `MsgspecStructCodec` — retained, simplified. Becomes the only codec.
- `JsonMessageCodec` — deleted.
- `PydanticMsgpackCodec` — deleted.
- `_enc_hook` — deleted (no Pydantic fields remain).
- `_dec_hook` — deleted (no Pydantic fields remain).

**`src/aiperf/common/models/auto_routed_model.py`:**

- `AutoRoutedModel` — deleted. Its responsibilities migrate to msgspec's
  built-in tagged-union decoding. `AIPerfBaseModel`'s inheritance from
  `AutoRoutedModel` is removed; `AIPerfBaseModel` becomes a plain Pydantic
  `BaseModel` subclass (still used for config and export).

### Stays Pydantic

- `AIPerfBaseModel` — no longer extends `AutoRoutedModel`; becomes a thin
  `BaseModel` subclass with the project's standard `model_config`. Config
  and final-export models continue to inherit from it.
- Every export model: `JsonExportData`, `TelemetryExportData`,
  `ServerMetricsExportData`, `GpuSummary`, `EndpointData`, `TimesliceData`,
  etc.
- Every config model (`BaseConfig` and its subclasses).

### Out of scope

- Export-model changes. The Pydantic↔msgspec seam at the exporter boundary
  (established by the telemetry and server-metrics spec) stays exactly where
  it is.
- Config surface. No CLI or config-file change.
- `AIPerfBaseModel`'s `model_config` defaults. Kept as-is for remaining
  Pydantic consumers.

## Architecture

### `ErrorDetails` in-place conversion

Current (simplified):

```python
class ErrorDetails(AIPerfBaseModel):
    code: int = Field(...)
    type: str = Field(...)
    message: str = Field(...)
    ...
```

Becomes:

```python
class ErrorDetails(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    code: int
    type: str
    message: str
    ...
```

Construction call sites (`ErrorDetails(code=..., type=..., message=...)`)
remain identical. `model_dump()` / `model_validate()` call sites — if any
exist outside the codec layer — migrate to `msgspec.to_builtins` /
`msgspec.convert`. Pre-spec audit: grep all usages and enumerate in P1's
opening task list.

### `Message` tagged-union root

Current:

```python
class Message(AIPerfBaseModel):
    discriminator_field: ClassVar[str] = "message_type"
    message_type: MessageTypeT = Field(...)
    request_ns: int | None = Field(default=None)
    request_id: str | None = Field(default=None)
```

Becomes:

```python
class Message(msgspec.Struct, tag_field="message_type", kw_only=True, omit_defaults=True):
    request_ns: int | None = None
    request_id: str | None = None
```

The `message_type` field is now implicit (supplied by the `tag=` argument on
each subclass). Subclasses:

```python
class HeartbeatMessage(Message, tag=MessageType.HEARTBEAT):
    ...
```

Every Phase 2 envelope migrates to this form. msgspec's `Decoder` resolves
the tagged union on decode — no custom registry, no `from_json` routing, no
`__init_subclass__` machinery.

### `AutoRoutedModel` removal

`AutoRoutedModel`'s hierarchical discriminators (parent defines
`discriminator_field`, child overrides the value, nested routing on the
same field) map to msgspec's nested tagged unions. The only current
hierarchical usage is `Message` → direct subclasses — no deeper nesting.
Removal is clean.

One subtlety: `AutoRoutedModel.from_json` accepted either bytes/str or a
pre-parsed dict. msgspec decoders always want bytes. Audit for call sites
that pass dicts; route through `msgspec.convert(data, Message)` instead.

### Codec collapse

`message_codecs.py` becomes:

```python
import msgspec
from aiperf.common.messages import Message

class MsgspecStructCodec:
    cache_key: str
    def __init__(self, *, decode_type: type = Message, cache_key: str = "msgspec-message") -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(type=decode_type)
    def encode(self, message: Any) -> bytes: return self._encoder.encode(message)
    def decode(self, data: bytes) -> Any: return self._decoder.decode(data)

MESSAGE_CODEC = MsgspecStructCodec()
```

No hooks, no alternates, no cache-key variants. Every transport client in
`src/aiperf/zmq/` (or equivalent) uses `MESSAGE_CODEC`.

### Wire format change

This spec changes the wire format from JSON to msgpack. Every live
deployment using multiple services must roll all services together —
mixed-version clusters will not interoperate during the cutover. Non-issue
for CI and single-binary releases; flagged explicitly because the prior
codec infrastructure was designed to support mixed per-channel formats.

### `dec_hook` / `enc_hook` deletion

After P1 converts `ErrorDetails`, the `dec_hook` path that rehydrates
Pydantic BaseModels is unreachable — every msgspec struct holds only msgspec
or primitive fields. P3 deletes both hooks after verifying via runtime
assertion during a full integration run that neither hook is invoked.

## Execution milestones

Each milestone ends with a green repo and a commit. Phase 3 runs serially
after Phase 2 is fully merged.

**P1 — `ErrorDetails` conversion.** Convert `ErrorDetails`,
`ErrorDetailsCount`, `ExitErrorInfo` to msgspec in place. Audit all usages:
`rg "ErrorDetails\(" src/ tests/` — enumerate construction, `model_dump`,
and `model_validate` sites. Migrate serialization call sites. Retire any
`WireErrorDetails` / `_error_to_wire` / `_wire_to_error` carried forward
from prior specs. Verify full `tests/unit/` + `tests/integration/` green,
including a 10k-concurrency error-path stress run that previously exercised
the `dec_hook`.

**P2 — `Message` base flip.** Convert `Message` to `msgspec.Struct` with
`tag_field="message_type"`. Convert every envelope in the four Phase 2
domain specs (records: 6, dataset: 6, credit: 6, service: 10) to
tagged-union members. Delete `AutoRoutedModel`; update `AIPerfBaseModel`
inheritance. Audit `from_json` call sites; migrate dict-accepting call sites
to `msgspec.convert`. Delete every `MsgspecField` shim annotation and its
supporting code in `common/models/base_models.py`. Full test suite green.

**P3 — Codec collapse + invariant assertion (landed).** Deleted `JsonMessageCodec`,
`PydanticMsgpackCodec`, `_enc_hook`, `_dec_hook`. Collapsed `message_codecs.py`
to `MsgspecStructCodec` only, with a lazy `get_message_codec()` singleton
parameterized by `Message._union_type()`. Every transport client (push/pull/fake)
uses `get_message_codec()` as the default. CI checks added in
`tests/unit/test_no_pydantic_on_wire.py` assert:

- No file under `src/aiperf/common/messages/` imports from `pydantic`.
- `message_codecs.py` imports neither `pydantic` nor `BaseModel`.
- No `model_dump`/`model_validate` calls in message modules.
- `JsonMessageCodec`, `JSON_MESSAGE_CODEC`, `PydanticMsgpackCodec` are absent
  from `message_codecs.py`.

## Testing

- Per milestone: full unit + component-integration suites green.
- P3 completion: full integration suite (`-m integration -n auto`) green,
  plus a 10k-concurrency stress run with forced error injection to confirm
  the error path that previously required `dec_hook` still works end to end.
- Wire-format smoke: before/after byte-size comparison on a 60s benchmark
  run's captured ZMQ traffic. Expect ~15-30% reduction (msgpack vs JSON with
  `exclude_none`).
- Decode-throughput benchmark: records-manager pull loop decode rate
  before/after. Expect ≥2x improvement.
- Assertion tests: grep-based CI checks documented in P3 above, encoded as
  a `tests/unit/test_no_pydantic_on_wire.py` module that enforces them.

## Risks

- **P2 merge conflict surface.** Every envelope in every Phase 2 domain
  spec changes. If P2 is delayed, rebasing it onto a moving `main` becomes
  expensive. Mitigation: P2 lands on a dedicated integration branch
  immediately after the fourth Phase 2 spec merges. Target: P2 PR open
  within 48 hours of the last Phase 2 spec merging.
- **Wire format cutover.** JSON → msgpack is a breaking wire change. Any
  external consumer that connects to aiperf's ZMQ sockets directly (none
  known in-tree, but possible for debugging tooling) must update. P3 adds
  a `docs/` note calling this out explicitly.
- **`ErrorDetails` consumer sprawl.** Call site count is large. P1's audit
  is the bulk of the work; the actual type change is mechanical. Mitigation:
  P1 starts with a full-repo grep capture committed to the PR as a
  checklist, worked through systematically.
- **`AutoRoutedModel` hidden consumers.** If any code path outside
  `Message` subclasses uses `AutoRoutedModel`'s discriminator machinery
  (e.g., for non-message discriminated unions), removal breaks it. P2 greps
  `class .*\(AutoRoutedModel\)` across the tree before deletion; any
  non-message consumer gets its own migration or is excluded with a
  comment.
- **`msgspec.convert` for pre-parsed dicts.** Anywhere code builds a message
  from a dict instead of bytes (e.g., a test helper that constructs a
  message from a literal `{"message_type": "...", ...}`) breaks if it
  called `Message.from_json(dict_obj)`. P2 audits via
  `grep -rn 'from_json(' tests/ src/` and migrates.
- **Log-line / error-format drift.** Error messages from msgspec validation
  read differently from Pydantic's. Any test that asserts on exact error-
  message strings breaks. P2 updates assertions to structural checks where
  needed.

## Non-goals

- No change to `AIPerfBaseModel` beyond removing its `AutoRoutedModel`
  inheritance. Config and export usage unchanged.
- No change to export-model schemas.
- No change to `profile_export_aiperf.json` output.
- No renaming of `Message`, `ErrorDetails`, or any envelope type.
- No performance-tuning beyond what falls out of the conversion and the
  codec collapse.
- No new message types.

## Post-spec verification

After P3 commits, the invariant "no Pydantic on the wire" is enforced by
`tests/unit/test_no_pydantic_on_wire.py`. The overall migration is complete
when:

- Status matrix in the overview doc is all green.
- The `tests/unit/test_no_pydantic_on_wire.py` assertions pass in CI.
- A 60s benchmark run shows ≥2x records-channel decode throughput vs.
  pre-M0 baseline.
- Zero `dec_hook` / `enc_hook` references remain in the codebase.
