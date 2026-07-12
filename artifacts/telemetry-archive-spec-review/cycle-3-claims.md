# Telemetry archive/watch spec — adversarial review cycle 3 claims

Target: `25183e4bc`

This convergence pass found only second-order integration/schema details. Claims
remain unaccepted until cross-agent default-refute adjudication.

## Architecture/integration

### C3A1 — P1 — archive projection is still placed on the request LocalSet

The topology obtains an archive permit on the LocalSet and routes directly to an
`ArchiveFrameDraft`, but family/point construction, sanitization, enrichment,
and hashing are archive-only CPU work.

Proposed fix: after native delivery and permit acquisition, enqueue archive
projection/draft construction on a bounded CPU stage; finalization tracks those
jobs/permits.

### C3A2 — P1 — one observation cannot deliver native before containing later archive admission

Native delivery must precede archive reservation, while the one factual observer
contains the reservation outcome and lists native delivery as a consumer.

Proposed fix: direct prepared-driver native callback first; the observer is a
post-native factual/report/test hook, or split explicit pre/post events.

### C3A3 — P1 — active membership omits start/end boundary subscribers

A start sample may precede phase STARTED and an end sample may follow COMPLETE,
so active membership alone can omit the phase whose boundary it serves.

Proposed fix: carry continuous active membership separately from explicit
`{phase_id,boundary_role}` subscribers.

### C3A4 — P1 — exact old/new summary parity is impossible across intentional cadence change

Replacing completion-paced phase loops with fixed-deadline run-owned scheduling
changes sample instants and therefore time-varying gauge/histogram values.

Proposed fix: exact archive-off/on parity after both use the new driver; retain
formula/boundary goldens and explicitly test expected old-cadence differences.

### C3A5 — P1 — attached scheduled archive has no authorable wire DTO

The standalone envelope is complete, but no strict attached field connects
existing sidecar source configs to archive sink/admission/failure policy.

Proposed fix: reusable `TelemetryArchiveSpecV2` in a named scheduled resource;
select existing stable source IDs without duplicating their config, include a
complete attached envelope, and reject deferred Graph/offline pairs.

### C3A6 — P2 — receipt timestamps lack a cross-thread Clock handshake

Fsync/CAS completes on the worker while only the LocalSet owns `Clock`; no
receipt-draft round trip is defined.

Proposed fix: worker completion -> LocalSet Clock stamp -> receipt draft back to
writer. Crash before observation leaves durable target with unknown observed
time.

## Durability/recovery

### C3D1 — P1 — zero-row projections have no authoritative index coverage

Empty expositions/metadata-only families can require zero sample/family rows,
but the index stores only partition/raw descriptors. Generation history cannot
be the unbounded coverage authority.

Proposed fix: persistent zero-projection entries keyed by frame/table, or an
exact rule making attempt counts authoritative zero evidence; define fragment
aggregation/indivisibility.

### C3D2 — P1 — plaintext raw IDs conflict with randomized AEAD ciphertext

Identical encoded bodies share a plaintext-derived key but independent random
nonces produce different ciphertext/digests at that key.

Proposed fix: separate one physical encrypted object from per-frame references;
create/reuse one persisted envelope for a raw ID, or include ciphertext identity
in the physical key. Never derive an ordinary AEAD nonce unsafely.

### C3D3 — P1 — receipt identity conflates WAL and publication targets and uncertain time

WAL receipts precede generations, and an uncertain successful CAS cannot
reconstruct the original Clock time/sequence bytes.

Proposed fix: discriminated WAL-range versus publication receipt variants;
identity independent of observation time (or durable pre-CAS intent), with
response-observed and recovery-verified times distinct.

### C3D4 — P2 — `LOCAL-RECEIPTS` has no bounded discovery authority

A flat list rewrites history, latest-only loses it, and a chain is unbounded.

Proposed fix: coalesced receipt batches plus canonical persistent receipt
index/head, exact retention and query semantics.

### C3D5 — P2 — writer-claim release is not atomic with terminal head CAS

A second CAS can leave the final head published but claim active, inconsistent
with the receipt.

Proposed fix: terminal CAS installs sealed state and null claim atomically;
receipt attests that object version and uncertain comparison includes absence.

## Schema/security/query

### C3S1 — P1 — raw physical object and per-frame reference are conflated

Duplicate of C3D2, additionally requiring explicit per-frame
`RawObjectReference(frame_id, raw_object_id)` logical coverage.

### C3S2 — P1 — receipt discovery and joins are incomplete

Duplicate of C3D4; marker rows also lack global `record_seq`, so frame-range
receipts cannot join every relation.

### C3S3 — P1 — coalesced archive rejection loses issued-attempt identity

Boundary attempts have source/request IDs but no cadence ticks; the current gap
shape cannot encode their range when rejected/coalesced.

Proposed fix: a distinct loss-range payload with source-record/request-attempt
ranges, optional tick/deadline/boundary identities, and missed-vs-rejected kind.

### C3S4 — P1 — classic histogram components may have unequal timestamps

Prometheus gives each sample line an optional timestamp. Combining unequal
component timestamps invents one MetricPoint time.

Proposed fix: format-specific assembly that either creates explicit partial
timestamp groups or marks point time mixed/component-only and prohibits snapshot
interpretation.

### C3S5 — P1 — canonical logical-row digest encoding is undefined

Field layout does not freeze nested/null/map/float/dictionary byte encoding, so
compatible writers may compute different compaction digests.

Proposed fix: checked-in canonical logical-row encoding with float-bit/null/
length/map/dictionary rules and independent Rust/Python goldens.

### C3S6 — P2 — OpenMetrics Created is modeled as an ordinary number

Created is a timestamp and lacks exact/sub-ns/out-of-range timestamp semantics.

Proposed fix: `SourceTimestamp`/dedicated created timestamp with absent status.

### C3S7 — P2 — sample sort is not total across same-instant scrapes

`metric_point_seq` resets per attempt; same Clock instant can collide.

Proposed fix: include source/global record sequence before point sequence.

### C3S8 — P2 — exact raw retention contradicts absolute known-credential absence

An endpoint can echo a credential into an exact body outside sanitization.

Proposed fix: scope the absolute guarantee to structured metadata/log/report/
error surfaces; classify/encrypt/access-control raw bodies explicitly.

### C3S9 — P2 — Float64 invariant contradicts unavailable analytical f64

A finite lexeme outside f64 range can be retained with `finite_value=null`.

Proposed fix: promise exact lexeme always and analytical f64 only when
representable, or explicitly reject an out-of-f64 numeric subset.

## Areas reported converged

- complete prepared-driver and per-profile HTTP construction;
- strict archive/native fallback split and deadline math;
- Graph/virtual product deferral, coalescing identity, and stop fence;
- startup lock/session/writer claim, WAL corruption/lagged retirement, raw
  transactional coverage, uncertain CAS ancestry, and bounded compaction;
- metadata-only/multi-point/lexeme/timestamp/role schemas, attribute epochs,
  index page wire format, additive native-v2, and required-failure authority.

