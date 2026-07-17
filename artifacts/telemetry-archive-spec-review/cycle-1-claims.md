# Telemetry archive/watch spec — adversarial review cycle 1 claims

Target: `specs/telemetry.md`

Reviewed commit: `89c505900`

This file freezes the claims from three independent, read-only reviews before
refutation or edits. A claim is not accepted merely because it appears here.
The adjudication pass must begin from a default-refute posture, check the whole
spec and current code, and assign `confirmed`, `partially confirmed`, or
`refuted` with evidence.

## Architecture and integration review

### A1 — P1 — watch request is not representable by the authoritative v2 DTO

The watch example in §§3.2 and 14.1 intentionally has no inference model,
endpoint profile, dataset, metrics, artifacts, sidecars, or phase list. The
authoritative `AuthoredRunSpecV2` described by the runner-only spec requires
several of those blocks. The illustrated request therefore cannot deserialize,
while adding dummy values violates the watch rule forbidding inert benchmark
fields.

Proposed correction: make the outer v2 envelope workload-neutral, make resource
blocks requirement-scoped, and let each `RunnerWorkloadFactory` declare and
validate required/optional/forbidden blocks. Show a complete deserializable
watch request.

### A2 — P1 — control-plane HTTP is an asserted capability, not a typed seam

The spec requires `online_http` to expose the injected `Clock` and reusable
control-plane HTTP, but current prepared backend/workload requirements expose
neither an object-safe control-plane handle nor per-request deadline overrides.
The compatibility pair would require a special case or a source-created
concrete transport. Current transport timeout policy is client-wide while watch
sources may have distinct deadlines.

Proposed correction: define a typed `ControlPlaneHttp` prepared-backend
capability and workload requirement with exact response bytes, timing/status
facts, and Clock-deadline overrides.

### A3 — P1 — the attached observer is downstream of irreversible projection

`ServerMetricsRecord` and `GpuScrape` have already dropped or normalized
OpenMetrics facts before the proposed `TelemetryBatchObserver<Record>` sees
them. The observer also sees only successful projected records, not HTTP,
transport, parse, empty, duplicate, disabled, or shutdown attempts. Attached
mode consequently cannot preserve the archive schema or the one-attempt-row
invariant.

Proposed correction: sources emit one typed all-outcome attempt envelope with
raw bytes/hash, timing/status, parse outcome, lossless exposition, and optional
domain projection. Tee that envelope into native and archive projections via an
all-outcome observer.

### A4 — P1 — the sidecar hook cannot emit exact authoritative phase markers

`ScheduledPhaseSidecar` receives only an `i64` sampled outside the lifecycle
transition. It receives no phase ID/config, transition state, or completion
reason. Start and end callbacks are sampled before/after the authoritative
`PhaseStats` transitions, respectively, so archive markers can disagree by
construction.

Proposed correction: tee exact `PhaseObserver` transition facts, or replace the
timestamp callbacks with a typed boundary context captured once and shared by
all consumers.

### A5 — P1 — attached mode retains a scheduler that violates the new scheduler invariant

Current server and GPU sidecars run one completion-paced task that loops through
all sources serially. Seamless phases may overlap their sidecars. Merely adding
an archive observer does not supply one fixed-deadline driver per physical
source, one in-flight request, or cross-source failure isolation.

Proposed correction: one run-owned driver per physical source in both modes;
phase machinery sends boundary/ownership commands rather than owning scrape
loops. Define sample attribution during seamless overlap.

### A6 — P1 — attached boundary scrapes can hang outside phase deadlines

Phase setup/finalization awaits sidecars, while the current metrics transport
has no request/total timeout after connect. The spec says boundary work has a
configured Clock deadline but defines no concrete timeout/cancellation
contract.

Proposed correction: define boundary and continuous request deadlines enforced
by `Clock`, cancellation-safe cleanup, and a timeout attempt outcome.

### A7 — P1 — worker-owned policy objects are neither sendable nor selectable

`ArchiveSink` is `Send`, but several policy traits it must contain lack `Send`
bounds. The spec simultaneously requires stable wire-selected policy IDs while
defining no descriptor/validate/prepare factory registries for those families.

Proposed correction: mark worker-owned traits `Debug + Send`, explicitly retain
LocalSet-only `?Send` traits, and define frozen registries for every
wire-selectable policy family.

### A8 — P1 — local fallback generations do not exist

The sole local authority is an overwritten `archive-manifest.json`, yet recovery
promises fallback to a preceding generation. A generation number inside one
overwritten file is not durable history.

Proposed correction: immutable checksummed manifest-generation files plus an
atomically and durably advanced small local head pointer.

### A9 — P1 — deleting a prefix of a monolithic WAL is not crash-safe

The transaction never defines how a committed prefix is removed without
risking the uncommitted suffix on a crash.

Proposed correction: immutable/sealed WAL segments or an append-only commit
offset journal; retire only complete, fully referenced segments.

### A10 — P1 — the sample schema does not encode the promised OpenMetrics model

The parser contract promises standards-correct metadata, exemplars, and source
timestamps. The schema has only counter/gauge/histogram/summary/untyped and no
source-sample timestamp, omitting OpenMetrics info, stateset, and gauge
histogram semantics.

Proposed correction: explicitly support and model a named standard/version and
all accepted semantic variants, or narrow the promise and store unsupported
valid input as a typed outcome.

### A11 — P2 — Clock-driven archive ticks have no owner

The IO worker cannot own the non-`Send` `Rc<dyn Clock>`. If age rotation,
checkpoint, or sync occurs only on append, idle sources never fire scheduled
maintenance; using worker wall time breaks the time invariant.

Proposed correction: a LocalSet-resident control driver sleeps on `Clock` and
sends timestamped tick/checkpoint/sync commands to the worker.

### A12 — P2 — synchronous decode work can delay unrelated sources and benchmark dispatch

Parsing, canonicalization, hashing, enrichment, redaction, and batch allocation
occur before the worker handoff on one LocalSet. A bounded but large exposition
can monopolize that loop despite network-task isolation.

Proposed correction: hand bounded exact bytes to a bounded decode worker/pool,
preserve source ordering, and enforce admission before large allocation; or
weaken the isolation promise with a much smaller explicit CPU/body budget.

### A13 — P2 — enrichment cannot be additive by construction

`TelemetryEnricher::enrich(&mut ArchiveSample, ...)` can mutate metric identity,
labels, semantic type, and value despite the claimed invariant.

Proposed correction: return an `AttributePatch` through a restricted builder;
use similarly explicit transformations for redaction and keep canonical metric
fields private.

### A14 — P2 — required attached-archive failure has no coherent report outcome

The spec permits a non-authoritative failure report after required archive
finalization fails, but runner authority says a failed run never emits a
successful/partial authoritative report. It is undefined whether completed
benchmark metrics exist or may be consumed.

Proposed correction: define a reporting-stage failed terminal plus a separately
named diagnostic artifact that outer-loop metric consumers must ignore, or add
an explicit incomplete/failed native report status and update the authority
contract.

### A15 — P3 — one epoch anchor overstates long-session wall-time accuracy

A later wall-clock step or oscillator drift can invalidate the original
uncertainty bound although monotonic in-session ordering remains sound.

Proposed correction: bracket anchor capture, define uncertainty growth, and
optionally record non-authoritative later wall anchors; describe Unix time as
approximate placement unless continuously bounded.

## Durability, recovery, and object-store review

### D1 — P1 — caller-observed acknowledgment equality is impossible

A frame may be fsynced and then the process can die before its receipt reaches
the caller. Recovery rightly includes it even though the caller did not observe
the receipt, contradicting exact equality between acknowledged and recovered
sets.

Proposed correction: distinguish accepted, durable, and receipt-observed.
Promise that returned durable receipts are a subset of exactly-once recovered
batches, permit uncertain operations to recover, and require retry with the same
batch ID.

### D2 — P1 — there is no durable genesis before the first acknowledged frame

The manifest owns identity/config/source/anchor facts needed for fail-closed
resume, but source activation and WAL acknowledgment can precede any durable
manifest.

Proposed correction: durably commit generation-zero genesis before source
activation; put its hash in the WAL header and admit no frames before genesis.

### D3 — P1 — local head advancement omits history and directory durability

The overwrite protocol lacks parent-directory fsync and retained generations.
A power failure after retiring WAL can reveal the old manifest and make an
acknowledged partition only an untrusted orphan.

Proposed correction: immutable hash-named generations, fsynced directory, and
an atomically replaced and directory-fsynced `LOCAL-LATEST` containing a
hash-linked head. Retire WAL only after pointer durability.

### D4 — P1 — WAL prefix mutation has no safe physical algorithm

Same core issue as A9, with the additional requirement that a partially covered
segment remain until fully covered or an atomic replacement suffix is durable.

### D5 — P1 — a batch spans tables but commit/dedup is described per partition

A batch may create scrape and sample rows. If one table rotates and references
the batch first, global batch-ID dedup on recovery can skip the still-uncommitted
other table.

Proposed correction: atomically publish a partition bundle covering one complete
WAL prefix across every required table, with per-table counts and exact
watermarks; deduplicate/retire only complete projections.

### D6 — P1 — finalize can overtake accepted data on the reserved control channel

A finalization command may be consumed before an earlier accepted data item,
allowing `finalized=true` while that batch remains unread.

Proposed correction: close admission atomically, capture an accepted sequence
watermark, and forbid finalize commit until every sequence through the watermark
is durably processed or explicitly accounted as loss.

### D7 — P1 — pointer CAS alone does not make generation objects immutable

Two writers can upload different bytes to the same `manifest-N+1` key; the
loser can overwrite the object even if its pointer CAS later fails.

Proposed correction: content-addressed, create-only generation objects. The head
contains generation, object key/hash, parent hash, archive ID, and fencing epoch.

### D8 — P1 — generic object storage does not guarantee the required CAS semantics

Client-emulated GET/PUT CAS on an eventually consistent backend permits two
successful conflicting heads.

Proposed correction: a narrow archive-store trait requiring linearizable
versioned get/CAS and atomic create-if-absent; reject authoritative remote resume
where capabilities are absent.

### D9 — P1 — local/remote reconciliation lacks ancestry and distinct sequences

One generation number cannot distinguish local commit 20 versus remote publish
8 as ordinary lag, rollback, or divergence when one remote publication may
cover several local commits.

Proposed correction: distinct `local_commit_seq` and `remote_publish_seq`,
hash-linked ancestry, and a fail-closed reconciliation matrix.

### D10 — P2 — a visible remote head may reference not-yet-visible objects

Eventually consistent reads can return the new head before its immutable
manifest or partitions are readable, causing a false corruption diagnosis.

Proposed correction: bounded retry for referenced immutable objects over a
declared consistency horizon and separate visibility-lag health.

### D11 — P1 — spool quota does not reserve transaction working space

WAL plus retained partitions can fill the quota so rotation has no space for a
temporary Parquet object, yet no WAL can be retired until rotation succeeds.

Proposed correction: reserve worst-case WAL, temp/open partition, manifest/head,
raw CAS, file-count, and emergency-finalization overhead; also check real
filesystem free space.

### D12 — P2 — remote upload can stall the only local durability owner

If the single writer awaits a slow remote PUT, it stops WAL progress, fills
ingress, and causes drops despite healthy local disk.

Proposed correction: preserve one local mutable owner but upload immutable
objects on a bounded timed uploader executor; return verified receipts to the
owner, which alone advances remote heads.

### D13 — P2 — lifecycle markers lack WAL identity and durable receipts

`append_marker` returns only `Result<()>` and the WAL protocol covers batches.
An exact authoritative phase marker can therefore be accepted and lost.

Proposed correction: markers are identified WAL frames with sequence/CRC and
the same durability receipt/coverage rules as batches.

### D14 — P2 — flat full manifest generations are quadratic metadata IO

Writing a full partition list after every rotation serializes O(P²) total
entries over P partitions and needs O(P) live state, contradicting long-soak
stability/no full-history rewrite.

Proposed correction: immutable deltas with hash-linked parents plus periodic
bounded snapshots; the head points to the delta tip.

### D15 — P2 — local-final/remote-incomplete has no resumable lifecycle

After local `finalized=true`, a crash can strand remote publication because the
spec defines neither reopening nor finalization-only resume.

Proposed correction: explicit open/stop-requested/locally-finalized/
remotely-finalized states and fenced finalization-only resume that cannot start
sources or append telemetry.

### D16 — P2 — compaction has no parent-CAS or replacement-set transaction

A compactor can publish from stale generation G and omit newer partitions; if
old and replacement partitions coexist in one logical generation, query rows
duplicate.

Proposed correction: compact only a locally finalized, locked archive; declare
exact parent hash, validate complete `(batch_id, table)` replacement coverage,
replace old entries atomically, and CAS from that exact parent.

## Schema, parser, security, and capacity review

### S1 — P1 — attached raw fidelity contradiction

Same core defect as A3. A shared pre-projection `DecodedScrape`/attempt envelope
must expose raw entity, lossless exposition, and benchmark projection; archival
must observe that envelope rather than only the lossy domain DTO.

### S2 — P1 — tagged non-finite representation is incomplete

Only scalar and summary quantile values use the tagged representation;
histogram sums/counts/buckets and summary sums/counts remain bare numeric
leaves. Non-finite input therefore leaks through or becomes unrepresentable.

Proposed correction: one `ArchiveNumber { kind, finite_value }` representation at
every numeric leaf, or atomically reject an invalid family with an explicit
outcome.

### S3 — P1 — accepted OpenMetrics surface is incomplete

Same core issue as A10, plus emitted sample role/name and normative exemplar
shape are unclear. Define the exact standard/version, family/sample roles,
source timestamps, exemplar model, and what exact bytes feed content hashes.

### S4 — P1 — `duplicate` is incorrectly modeled as a scrape outcome

An HTTP-successful unchanged body is still a successful observation. If it
contains no samples, charts have artificial holes; if samples are silently
implied, queries cannot resolve them.

Proposed correction: success/empty/failure is the outcome axis;
`body_unchanged` is orthogonal. Either archive each successful observation or
store an explicit `sample_ref_batch_id`.

### S5 — P1 — redaction does not cover the whole archive surface

The contract mentions samples and diagnostics, but source descriptors, markers,
manifest attributes, exemplars, report fields, and optional raw bodies can carry
sensitive data.

Proposed correction: one archive-wide recursive sanitizer/classification policy;
raw artifacts require explicit classified/encrypted handling rather than a
sample-only redactor.

### S6 — P1 — deleting identity labels can silently merge distinct series

Removing a tenant/pod label before identity derivation can map distinct source
series to the same key.

Proposed correction: keyed pseudonymization or a protected
`source_series_key`, plus collision detection and explicit rejection/coalescing
semantics.

### S7 — P1 — parse/encode bounds occur too late for LocalSet isolation

Same core issue as A12. Enforce compressed and decompressed receive/parser
limits during decoding, reserve admission before archive-only encoding, and move
continuous archive parsing/encoding to bounded control-plane CPU workers.

### S8 — P1 — the Arrow/Parquet schema is not normative enough for compatibility

Field encodings are ambiguous (binary versus hex, decimal versus string,
dictionary widths/nullability) and there are no table fingerprints or canonical
union projection rules.

Proposed correction: exact Arrow schemas with nullability/logical types,
fingerprints, and deterministic projections for every semantic variant.

### S9 — P1 — local preceding-generation fallback is impossible

Same core issue as A8/D3: immutable generations plus a durable head are needed.

### S10 — P1 — loss accounting has no guaranteed lane when the data queue is full

If the same full ingress must carry a gap row, the gap row is also lost. A final
aggregate count cannot reconstruct source, deadlines, reason, or interval.

Proposed correction: reserved fixed-memory control/loss journal that coalesces
source, attempt/deadline range, reason, and count without using data capacity.

### S11 — P1 — remote integrity and discovery are under-specified

Object size and provider ETag are not portable cryptographic verification, the
report URI/pointer contract is ambiguous, and required conditional capabilities
are assumed.

Proposed correction: strong checksum/readback, exact head/CAS schema, explicit
pointer plus immutable manifest URI in reports, and backend capability gates.

### S12 — P2 — full manifest rewrite remains quadratic

Same core issue as D14: use chunked/delta manifests from v1 or prove a strict
supported partition bound that makes flat snapshots safe.

### S13 — P2 — digest canonicalization and topology identity are ambiguous

Raw concatenation permits boundary ambiguity and the relationship between
attributes, series identity, and topology epochs is not defined.

Proposed correction: domain-separated, length-prefixed canonical encodings and
an explicit immutable identity/topology-epoch model.

### S14 — P2 — HTTP source hardening/auth is incomplete

The source contract omits credential-provider references, TLS policy, redirect
and proxy behavior, accepted media/content negotiation, and distinct compressed
and decompressed limits/deadlines.

Proposed correction: strict source configuration for those controls, never raw
secrets in durable descriptors.

### S15 — P2 — epoch uncertainty acquisition is unspecified

Proposed correction: monotonic-before/wall/monotonic-after bracket, midpoint
anchor, half-span uncertainty, drift policy, and deterministic tests.

### S16 — P1 — native-v2 evolution is contradictory

Adding a typed `telemetry_archive` block to the established `2.0` report without
a version bump or an already-reserved capability changes the schema contract.

Proposed correction: define the typed DTO and bump to 2.1 (or demonstrate an
explicit 2.0 extension reservation) with old/new consumer goldens.

### S17 — P2 — performance acceptance criteria are not pass/fail

The spec names desired behavior but gives no supported scale profile, numeric
RSS/CPU/lag/shutdown limits, statistical method, or pinned dependency/version
matrix.

Proposed correction: define at least small and long-soak profiles with numeric
budgets, regression thresholds, repetitions/percentiles, and dependency pins.

## Reviewer concerns already investigated and rejected

These are not claims to carry into the correction set unless a refuter finds new
evidence:

- Rust `f64` exactly represents the stated 100,000,001 counter test value.
- The async traits are not inherently non-object-safe; the issue is ownership
  and missing `Send` bounds on contained worker policies.
- Existing code validly uses `Rc<dyn Clock>`; it simply cannot cross to the
  worker thread.
- Runner-owned archive IO does not itself violate Python/Rust ownership; live
  durable acknowledgment justifies the narrowly scoped exception.
- The spec does not make the archive authoritative for native metrics.
- Attached archival does not inherently require a second scrape; the proposed
  observation point is merely too late.
- Exact response-byte hashing is possible because transport retains `Bytes`.
- A correctly bounded enqueue after native ingestion need not perturb the token
  hot path; synchronous decode remains disputed.
- Immutable content-addressed partitions plus a manifest do not repeat
  Tachometer's mutable-checkpoint/final-concat duplication bug.
- Manifest-authoritative discovery correctly makes directory listing and
  unreferenced objects non-authoritative.
- Monotonic plus epoch anchoring does not permit NTP to reorder a session; only
  long-session wall-time accuracy is disputed.
- Standalone watch need not fabricate request distributions.
- Labels are not placed in object keys, and the histogram schema separates
  family labels from bucket bounds.
- There is no requirement for a mutable `final.parquet`.
- One mutable local owner is a sound baseline; the disputes concern ordering and
  blocking around it.
- Compaction can be safe in principle; its exact transaction is missing.

## Cross-review duplicate groups

Adjudicators should decide each underlying issue once while preserving every
affected requirement:

- Pre-projection all-outcome observation: A3, S1.
- Immutable local manifests and durable head: A8, D3, S9.
- Immutable WAL segments: A9, D4.
- OpenMetrics completeness: A10, S3.
- LocalSet CPU isolation/admission: A12, S7.
- Flat manifest scalability: D14, S12.
- Epoch anchor acquisition/accuracy: A15, S15.
- Object-store capability/integrity: D7, D8, S11.

