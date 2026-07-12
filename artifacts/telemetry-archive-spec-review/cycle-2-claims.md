# Telemetry archive/watch spec — adversarial review cycle 2 claims

Target commit: `9e74a4816`

The cycle-2 reviewers reread the entire revised specification and current
authority/code. This file freezes only new or still-actionable claims. None is
accepted until a different agent performs a default-refute adjudication.

## Architecture and integration

### C2A1 — P1 — factory erasure loses the typed pipeline

`ArchiveSourceFactory::prepare` returns only `Box<dyn ArchiveSource>`, which can
fetch bytes but exposes none of the separately declared generic decoder,
native-projection, or archive-projection objects. After registry erasure the
runner cannot recover `Entity`/`Record` without downcasts or string matching.

Proposed fix: return an object-safe `PreparedTelemetryDriver` that internally
owns and drives the complete monomorphized fetch/decode/projection/delivery
pipeline.

### C2A2 — P1 — HTTP capability cannot prepare per-source policy or isolate inference capacity

Sources may require different TLS roots, mTLS/auth providers, proxies, and
reuse policy, but the backend exposes only one executing handle. Sharing an
inference connection pool can also perturb benchmark capacity.

Proposed fix: a backend-owned `ControlPlaneHttpProvider` validates a secret-free
per-source transport profile and returns a dedicated control handle/pool; the
handle takes request data and an absolute call deadline.

### C2A3 — P1 — strict archive parsing contradicts required native compatibility fallback

The archive forbids declared-format fallback, while current native server
semantics intentionally retry OpenMetrics-labeled bodies as classic text for
vLLM compatibility. One strict `Exposition` cannot satisfy both promises.

Proposed fix: one byte fetch/decode job yields a strict archive parse outcome
and, when configured, a separately named native-compatibility fallback view;
record the mismatch and never call the fallback a valid strict archive parse.

### C2A4 — P1 — admission occurs after the object it must reserve

`ArchiveAdmissionPolicy` accepts an already built `ArchiveBatch`, yet the text
requires reservation before archive-only decode/allocation. Denying a shared
decode permit in attached mode would also suppress authoritative native data.

Proposed fix: separate bounded shared fetch/native-decode reservation from a
nonblocking archive projection/frame/WAL reservation represented by an owned
permit. Archive denial cannot block native decode/delivery.

### C2A5 — P1 — accepted sequence has two owners and terminal frames fall beyond its fence

The topology depicts the CPU pool sending `accepted_seq`, while §10 assigns it
to the archive owner. Stop captures a watermark before final marker/loss frames
are submitted.

Proposed fix: one ingress state machine: close new reservations, resolve all
permits, enqueue final frames, close frame admission, then let the sole owner
assign/capture the inclusive final frame sequence.

### C2A6 — P1 — offloaded values and policies lack thread bounds

Decoder traits are sendable, but `Entity`, returned native record, enrichers,
and sanitizers are not required to be `Send + 'static` even though the promised
CPU pool owns them.

Proposed fix: explicit offloaded-pipeline bounds; retain `?Send` only for source,
admission, local observer graph, and a separately qualified inline pipeline.

### C2A7 — P1 — raw-body retention has no byte path

Decode consumes `FetchedAttempt`; `DecodedAttempt` and generic observers do not
retain exact bytes, so the policy-gated raw writer cannot receive them.

Proposed fix: an opaque exact-entity lease accessible only to prepared archive/
raw-retention projection, with explicit compressed-wire versus decoded-entity
semantics.

### C2A8 — P1 — worker completion is absent from SimClock quiescence

The virtual driver advances or deadlocks when the LocalSet is quiescent, but
external decode/IO workers may still have same-instant work pending.

Proposed fix: compose an external-progress/quiescence source that holds virtual
time until ordered worker results arrive, or use a deterministic inline/memory
path and persist after simulation. Otherwise defer offline attachment.

### C2A9 — P1 — one shared scrape cannot use the current single-phase native record during overlap

Current server records contain one `benchmark_phase`, while seamless phases can
overlap and the new run-owned driver emits one physical attempt. Assigning one
phase loses the other; cloning can double-count run-level metadata.

Proposed fix: attempt-identified native records with an explicit phase-membership
set and attempt-ID dedup for run-level facts, or rigorously specified phase-local
clones with parity proof.

### C2A10 — P2 — adjacent boundary coalescing lacks an orchestrator identity

Sequential end/start callbacks cannot be recognized as one logical boundary by
timestamp proximity.

Proposed fix: the phase orchestrator assigns a typed transition epoch to a
non-seamless adjacent pair and distributes one snapshot; genuinely separate
seamless transitions do not coalesce.

### C2A11 — P1 — Graph attachment is promised without built or designed Graph lifecycle integration

The design relies on scheduled `PhaseObserver`/boundary commands even though
the current Graph phase consumer is unbuilt.

Proposed fix: defer Graph attachment explicitly, or fully define a neutral run
lifecycle observer plus Graph boundaries and report joins.

### C2A12 — P2 — best-effort writer death has conflicting success/failure routing

One clause sends dead-writer loss ranges through a failed diagnostic, while
another requires attached best-effort to retain a successful native report.

Proposed fix: keep the loss ledger outside the dead writer and put it into the
successful archive-health block for best effort; diagnostic artifacts are for
primary/required failures.

### C2A13 — P2 — request deadline arithmetic and timeout ownership are ambiguous

The cadence deadline is described as a request deadline, which could expire at
launch; timeout controls also appear at source and transport levels.

Proposed fix: `min(request_start + validated_request_timeout,
boundary_deadline?, run_or_shutdown_deadline?)`; cadence time is lateness data,
and the prepared control profile owns connect/TLS ceilings.

## Durability and recovery

### C2D1 — P1 — exact-resume lacks a durable new-session transaction

Create and resume are both said to commit generation zero, but a resumed
session/anchor can reach WAL and source activation before any new authoritative
generation records it. Recovery validation also occurs before the lock,
creating a TOCTOU.

Proposed fix: create commits genesis once; resume locks, rereads/reconciles,
recovers old WAL, and commits `session_started` with the new anchor before its
WAL/source activation; sync-only creates no session.

### C2D2 — P1 — CRC failure is incorrectly treated as proof of a non-durable tail

A complete fsynced/receipt-observed final frame can later suffer corruption.
Discarding a “corrupt” final frame violates recovery; only an incomplete
physical tail proves it could not be a complete frame.

Proposed fix: restore complete-length CRC failures from exact remote bytes or
fail closed; silently discard only an incomplete tail.

### C2D3 — P1 — immediate WAL deletion makes preceding-head fallback incomplete

After G commits F and its WAL is removed, corruption of G causes fallback to
G-1, which has neither F nor its WAL copy.

Proposed fix: a preceding head is only last-known-good unless WAL or an
independently verified remote copy covers the gap. Retain WAL until redundant
recoverability when local fallback is promised, otherwise fail closed.

### C2D4 — P1 — durability timestamps/publication markers are self-referential

An immutable attempt frame cannot contain the time at which that frame later
became locally durable or remotely referenced. A sealed archive cannot append
a marker after remote publication.

Proposed fix: remove those fields from attempt rows and use a non-self-
referential receipt/event relation attesting earlier frame/ranges; final state
lives in generation/head, not a marker claiming its own commit.

### C2D5 — P1 — final control/loss frames are beyond the captured watermark

Duplicate of C2A5's stop-order defect.

### C2D6 — P1 — remote exact-resume has no pre-activation writer claim

Two hosts can validate the same remote head, activate sources, and acknowledge
divergent local histories before one eventually loses publication CAS.

Proposed fix: acquire a conditional remote writer claim/fencing token before
source activation; restrict crash takeover to canonical-spool ownership or an
explicit operator-mediated fence without unsafe wall-clock leases.

### C2D7 — P2 — uncertain successful CAS is treated as a conflict

CAS can succeed remotely while its response is lost; retry then conflicts with
the exact desired head and is wrongly diagnosed as another writer.

Proposed fix: distinguish uncertain transport from conflict, reread through the
visibility horizon, and treat exact desired head/hash as idempotent success.

### C2D8 — P1 — retained raw objects are outside WAL/index/publication coverage

No required projection or authoritative descriptor prevents frame retirement
before a configured raw object is durable/discoverable.

Proposed fix: optional required `raw_object` projection with an opaque encrypted
descriptor, index coverage, local/remote finalization ordering, and no WAL
retirement before coverage.

### C2D9 — P1 — compaction frame/table coverage cannot prove row equality

A replacement can drop or duplicate rows while preserving `(frame_id, table)`
and partition-wide count.

Proposed fix: freeze per-projection row count plus canonical logical multiset
digest (or stable row IDs/digest) and require equality across compaction.

### C2D10 — P2 — compaction transaction can be unbounded and its remote parent ambiguous

Whole-archive removal lists can be O(P), and remote may lag the exact local
parent named by compaction.

Proposed fix: numerically bounded subset/subtree swaps and remote reconciliation
to the exact compaction parent before mutation/publication.

### C2D11 — P2 — local filesystem/lock capabilities are not qualified

The proof assumes same-filesystem atomic rename, file/directory fsync, stable
inode behavior, and a crash-released exclusive lock, none of which is validated.

Proposed fix: an explicit local-spool capability contract and probe/allowlist;
hold an open-descriptor lock for recovery/run/sync/compaction, rerun recovery
under it, and reject unqualified network/FUSE filesystems.

## Schema, security, and queryability

### C2S1 — P1 — valid multi-point OpenMetrics metrics do not fit one singular payload

OpenMetrics permits repeated points for one metric/label set, but the sample row
has one timestamp and semantic payload. `wire_samples` can retain lines without
defining which point the structured payload represents.

Proposed fix: make a row one metric point with `metric_point_seq`, or store an
ordered point list and associate every emitted sample with its point.

### C2S2 — P1 — metadata-only metric families disappear

A valid HELP/TYPE/UNIT family can contain zero metrics. No sample row exists and
raw retention defaults off.

Proposed fix: a family-metadata table/projection keyed by attempt/family,
including zero-point families and explicit empty-set distinction, with WAL/index
coverage.

### C2S3 — P1 — numeric lexemes and exact integers are not preserved

`ArchiveNumber` has no lexeme, and its exact-u64 equality rule fails above 2^53
when analytical f64 rounds.

Proposed fix: retain every numeric token lexeme; make exact u64 independent of
the f64 analytical projection/status and add boundary goldens through u64 max.

### C2S4 — P1 — source timestamps lack format and conversion semantics

Prometheus 0.0.4 timestamps use milliseconds while OpenMetrics uses seconds;
fractional/sub-ns/range normalization is unspecified and the attempt does not
store the selected parser format.

Proposed fix: persist declared and actual media/grammar version, exact lexeme,
conversion/rounding/range status, and normalized ns only when representable.

### C2S5 — P1 — parser tests require accepting semantically invalid OpenMetrics roles

The gate asks tagged nonfinites at all numeric leaves, but OpenMetrics imposes
integer/nonnegative/non-NaN and boolean role constraints and atomic invalid-
document rejection.

Proposed fix: separate syntax and semantic validation; reject role-invalid
OpenMetrics atomically and test nonfinites only where each advertised format
allows them.

### C2S6 — P1 — gap rows cannot satisfy non-null attempt identity

Missed ticks issue no source request and therefore no request attempt sequence,
but every attempt/gap row requires non-null attempt/batch identity. Gap integer
types remain ambiguous.

Proposed fix: add non-null archive `record_seq`, make request `attempt_seq`
nullable for non-request records, define gap tick/range identity, and freeze
UInt64 sequences/counts versus Int64 deadlines.

### C2S7 — P1 — final watermark precedes terminal frames

Duplicate of C2A5/C2D5.

### C2S8 — P1 — attempt durability timestamps cannot exist in the immutable attempt

Duplicate of C2D4; use independent append-only receipt/publication records.

### C2S9 — P1 — raw objects are outside the archive transaction

Duplicate of C2D8.

### C2S10 — P1 — body identity/encryption domain is contradictory

No mandatory body-digest derivation exists when raw retention is off; “exact”
alternates between encoded and decoded bytes, and plaintext integrity metadata
is both placed in authenticated metadata and promised absent.

Proposed fix: domain-separated subkeys, separate encoded-entity and decoded-
exposition digests with one named unchanged rule, and an exact AEAD envelope
whose plaintext integrity metadata is encrypted with defined nonce/retry/resume.

### C2S11 — P1 — attached admission contradicts shared decode/native parity

Duplicate of C2A4: guarantee shared/native decode independently; only archive
projection/ingress may be rejected.

### C2S12 — P1 — absolute secret invariant is impossible with arbitrary headers and Noop sanitizer

Attempts retain response headers, while endpoints may emit secrets in headers
or labels and the example selects no sanitizers.

Proposed fix: archive only typed allowlisted response metadata, and either use a
safe default sanitizer/classification policy with explicit unsafe-Noop
acknowledgment or narrow the invariant to AIPerf-configured/provider secrets.

### C2S13 — P1 — manifest index is not an independent-reader wire contract

The B-tree lacks page schema, encoding, comparator, split/separator rules,
fingerprint, and source/time zone maps even though readers must walk/prune it.

Proposed fix: freeze canonical head/generation/index page descriptors, key and
ordering rules, hashes/checksums, pruning summaries, and independent-reader/
bounded-page-read goldens.

### C2S14 — P1 — strict parsing conflicts with native fallback

Duplicate of C2A3.

### C2S15 — P1 — remote-finalization receipt has no durable local transition

Remote CAS may succeed before a crash, but a sealed archive admits no frame or
generation to record the local publication receipt.

Proposed fix: a checksummed atomic directory-fsynced receipt object keyed by
generation hash/CAS version with reconciliation rules, or a new immutable
metadata generation with a well-founded publication protocol.

### C2S16 — P2 — source connect ceilings cannot pass through the HTTP capability

Source config owns a connect ceiling while per-call overrides explicitly cover
only request/total and backend policy owns connect/TLS.

Proposed fix: put the connect ceiling in the per-source prepared control profile
or define exact validation/min semantics.

### C2S17 — P2 — attribute epoch ID has no canonical algorithm

The required digest has no domain, input, source/session binding, change
sequence, or marker/sample ordering.

Proposed fix: freeze its domain-separated canonical input and transition rules.

## Cycle-1 areas explicitly found closed

- resource-scoped runner DTO and exact phase timestamp authority;
- required-failure report authority and additive native-v2 compatibility;
- enrichment mutation and full OpenMetrics semantic branches;
- Clock maintenance ownership for real-clock runs;
- monolithic WAL truncation, ordinary multi-table coverage, and receipt states;
- transaction reserve, immutable generations, persistent manifest scaling, and
  remote visibility/ancestry baseline;
- one mutable archive owner and nonblocking bounded immutable uploads.
