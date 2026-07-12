# Telemetry archive/watch spec — adversarial review cycle 4 claims

Target: `bfa45bd47`

All three reviewers read the complete 2,532-line target. This file freezes their
claims before cross-refutation or edits. Duplicate claims remain separate until
adjudication. The default-refute pass must assign `confirmed`, `partially
confirmed`, or `refuted` and may not accept a claim merely because it appears
here.

## Architecture/runtime/protocol

### C4A1 — P1 — final logical hashes precede owner-assigned identity

The projection worker performs logical hashing and returns an unsequenced draft;
only afterward does the owner assign `record_seq` and `frame_id`, even though
those fields enter canonical row bytes. Recovery would recompute a different
coverage digest. Fix by reserving/stamping owner identity before authoritative
hashing or by adding a sequenced finalization/hash stage. A failed reserved
projection must consume the same sequence as a loss frame.

### C4A2 — P1 — parallel jobs have no attribute-epoch order

Attribute epochs are source-local hash chains, but the second pool can have
multiple outstanding jobs and no per-source FIFO authority. Topology A then B
can complete B then A and reverse or fork the chain. Add a bounded at-most-one-
active FIFO projection strand per source, parallel across sources, or move epoch
assignment/marker insertion/final hashing to a source-sequenced owner stage.

### C4A3 — P1 — forced attempts cannot join exact lifecycle markers

Commands carry boundary ID/role/group, attempts persist only boundary IDs, and
markers persist none of those join keys. Coalesced phase-A-end/phase-B-start
cannot be joined without forbidden timestamp inference. Persist structured
boundary references on attempts and matching identity/group on markers (or an
exact join relation), and extend the authoritative lifecycle context.

### C4A4 — P1 — full report byte parity is impossible

Archive-on reports necessarily add `ReportTelemetryArchive`, while archive-off
reports omit it. Define a canonical native-measurement parity projection that
excludes the archive block and every archive-dependent provenance/config/
artifact field; compare that projection, and test the archive block separately.

### C4A5 — P2 — sync-only receipt times lack a durable Clock domain

A fresh `RealClock` has a fresh origin, while sync-only can emit
`recovery_verified_ns` without creating a session/anchor. Persist a receipt-
observer epoch with ID/time-domain/`EpochAnchor` for every execution, including
sync-only, and bind each event to it.

### C4A6 — P1 — fixed-memory loss ledger has no overflow form

Alternating loss kinds/reasons can prevent range coalescing indefinitely after
writer failure. Freeze capacity plus a typed overflow summary with source/kind/
reason totals, first/last identities, dropped-range count, and rolling digest;
surface `complete_ranges=false` and test adversarial alternation.

## Durability/recovery/security

### C4D1 — P1 — authoritative row hashes precede owner identity

Duplicate of C4A1 from an independent crash-recovery review. A crash between
WAL durability and index commit exposes mismatched pre-stamp versus recovered
row digests. Identity must be fixed before authoritative logical hashing, and a
failed reserved sequence must become a loss frame at that sequence.

### C4D2 — P1 — raw equality omits stored Content-Encoding semantics

The shared raw ID hashes only bytes while the shared encrypted envelope stores
response-specific content encoding. Identical bytes observed once as identity
and once as gzip reuse metadata from the first response. Include every envelope
semantic in equality, or make the physical object bytes-only and move content
encoding to each `raw_references` row.

### C4D3 — P1 — fixed-memory loss ledger has no saturation record

Duplicate of C4A6 from an independent failure trace. Once noncontiguous ranges
fill the fixed ledger, the implementation must grow, block, or lose information.
Add a bounded per-source/kind saturation summary with exact totals, first/last
facts, and a domain-separated rolling digest that remains reportable and can be
persisted after recovery.

### C4D4 — P1 — terminal first publication conflicts with claim rules

Create-new may finish locally before its first remote `LATEST`. The spec says
first publication installs an active claim but terminal publication must clear
the claim in the same CAS, forcing a forbidden second update. Split absent-
remote behavior: open collection creates an active-claim head; an already sealed
archive creates the complete terminal absent-claim head directly and reconciles
uncertain create against those exact bytes/version.

## Schema/query/canonicalization

### C4S1 — P1 — logical-row encoding does not select one byte sequence

The literal magic, table-ID registry, header widths, schema-fingerprint bytes,
and map-count width are not frozen. Conforming encoders can choose different
bytes. Freeze every byte/mapping/width/endianness and pin those bytes in cross-
language fixtures.

### C4S2 — P1 — canonical JSON admits multiple encodings

Literal UTF-8 versus `\u` escaping, escaped versus unescaped slash, hex case,
normalization, and duplicate keys remain open. Freeze a complete JSON
canonicalization grammar (or a reconciled named standard), including arrays and
duplicate-key rejection.

### C4S3 — P1 — primary index keys are undefined for several object kinds

Shared raw objects, global markers/losses, zero-row coverage, plural-source
partitions, restarted sessions, and signed Clock values do not map uniquely to
the singular `(kind, table, source, time, digest)` key. Freeze a per-object-kind
key matrix, numeric IDs, session/domain component, inapplicable sentinels,
sortable-i64 encoding, logical ID derivation, authoritative frame time, and
single/plural-source partition rule.

### C4S4 — P1 — receipts lack exact bytes, query schema, and Clock domain

Target/event field types/order/nullability, `ObjectVersion` encoding,
discriminants, batch bytes, and query columns are not frozen. A sync-only event
also lacks an observing Clock domain. Add fingerprinted canonical target/event/
batch/relation descriptors plus a persisted observation epoch/anchor.

### C4S5 — P1 — sample payload is neither an exact nested schema nor a canonical projection

The scalar layout, exemplar fields/nullability, and payload-to-wire role rules
are incomplete. For an OpenMetrics histogram with `+Inf` but no emitted
`_count`, writers can derive or omit payload count. Freeze the full nested Arrow
descriptor and per-format/role projection matrix, including derived versus
emitted count, label removal, exemplar ownership, and child order.

### C4S6 — P2 — analytical f64 projection is nondeterministic

Rounding mode, underflow/overflow, signed zero, exactness, and legal child/status
combinations are open. Require exact-token conversion using IEEE-754 binary64
round-to-nearest ties-to-even and freeze a complete validity matrix.

### C4S7 — P2 — `uniform_explicit` timestamp equality is ambiguous

Classic timestamp lexemes `1000` and `01000` name the same millisecond but can
be classified uniform or mixed. Define equality (parsed value/status or exact
struct) and the deterministic representative lexeme, including sub-ns/out-of-
range cases.

### C4S8 — P2 — enum/loss validity and null collation remain open

Many v1 enum registries, the loss-kind validity/coalescing/count matrix, loss
sequence scope, dictionary-value versus field nullability, and nullable source
sort order are not frozen. Publish those registries/matrices and a `NULLS FIRST`
or `NULLS LAST` rule.

### C4S9 — P2 — B-tree deletion violates stated page minima

Deleting the last leaf entry requires an unspecified borrow/merge/root collapse
or leaves an invalid underfull page, despite delete goldens and compaction
removals. Freeze deterministic rebalancing/tie-breaking or explicitly permit
and encode underfull/tombstone pages.
