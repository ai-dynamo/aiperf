# Telemetry archive/watch spec — adversarial review cycle 5 claims

Target: `78c4ce8a6`

All reviewers read the complete 2,782-line target. Claims are frozen before
cross-refutation or edits; duplicates remain separate until adjudication.

## Architecture/runtime/protocol

### C5A1 — P1 — boundary groups are not atomically registered

One command carries one subscriber, but the driver must launch before a later
same-group phase command may exist. It cannot retroactively attach a completed
start/end subscriber. Submit one atomic group command with all references, or
pre-register and seal exact membership before launch.

### C5A2 — P1 — real-run byte parity is impossible

Two real executions naturally differ in authoritative nanosecond measurements,
even with archive disabled. Byte parity must use one captured event stream or a
deterministic replay; real subprocess comparison must use formula/order
invariants and the existing statistical performance gates.

### C5A3 — P1 — graceful stop does not redeadline an active fetch

A long request launched before SIGINT retains its original deadline and is not
in the projection drain tracker. Define a cancellation latch selected by every
fetch, enforce the minimum shutdown deadline dynamically, reclaim transport
state, emit one terminal attempt, and join it before finalization.

### C5A4 — P1 — post-finalization compaction has no remote fence

Remote finalization clears the active writer claim, but later compaction still
updates the remote head. Prohibit it or define a CAS-acquired maintenance claim
bound to canonical spool/lock/base head, used and cleared by compaction CAS.

### C5A5 — P1 — sync-only unnecessarily requires source credentials

`finalize_remote` activates no sources yet generic preparation still requires
control HTTP, source factories, and endpoint credentials. Make requirements
config-specific, prepare only spool/key/store/receipt resources, verify frozen
source identity from genesis, and provide a complete strict sync-only envelope.

### C5A6 — P2 — local generation completion has no receipt target

Checkpoint/finalize completions use the receipt handshake, but the closed union
has only WAL range and remote CAS publication. Add `local_generation` with local
generation/root/head/state or narrow the handshake claim.

## Durability/recovery/security

### C5D1 — P1 — observer epochs cannot bootstrap the event-keyed receipt index

Duplicate root of C5S1. An epoch must be durable before any event, but the index
key/head only represent events. Make epoch (and target) first-class keyed records
with head counts and an epoch-only batch transaction.

### C5D2 — P1 — loss saturation snapshots lack exact checkpoint/retirement semantics

Related to C5S4. Mutable cumulative snapshots in an immutable relation double-
count or lose a crash window. Define disjoint immutable snapshot epochs with
slot/snapshot/ordinal/previous-digest fields; double-buffer until local receipt
and reduce the chain exactly once.

### C5D3 — P1 — projection failure has no terminal frame-ID rule

The owner declares a final frame ID containing payload kind before projection,
then may replace success with a loss kind. Reserve sequence plus outcome-neutral
reservation ID; derive success/loss frame ID only after terminal kind, or make
frame ID outcome-neutral. Pin crash/retry behavior.

### C5D4 — P1 — complete open-WAL frames lack cryptographic payload integrity

CRC and a pre-projection frame ID do not authenticate final payload; an
unobserved open-segment frame may have no receipt prefix hash. Store a
domain-separated BLAKE3 digest over exact final header/payload and bind ordered
frame digests into segment/prefix hashes; always verify on recovery.

### C5D5 — P2 — tamper-detection gate lacks an authenticated root

An attacker controlling the namespace can recompute all unkeyed hashes and
replace `LATEST`. Either identify authenticated store CAS/ACL plus a trusted
head version as the threat-model root and narrow the claim to corruption/
substitution relative to it, or add signed/MACed anti-rollback roots.

## Schema/query/canonicalization

### C5S1 — P1 — observer epochs are unreachable before the first event

Duplicate root of C5D1. Define receipt-index record/key variants and head counts
for epoch, target, and event records, or directly reference epoch-only batches.

### C5S2 — P1 — receipt target coalescing can backdate observation

Combining R1 observed at t1 with R2 observed at t2 can falsely claim R2 at t1.
Coalesce only before one aggregate completion crosses the Clock bridge; event
draft targets are immutable. Freeze ordered coverage-digest composition.

### C5S3 — P1 — coverage fragment references dangle after compaction

Coverage points at an old partition but compaction replaces only partition
descriptors. Define fragment ID and two-way coverage/partition integrity;
atomically replace affected coverage entries when physical partitions change.

### C5S4 — P2 — saturation snapshots have no query reduction semantics

Related to C5D2. Choose cumulative snapshots with stable slot/snapshot IDs and
latest-wins semantics, or disjoint interval snapshots reset only after durable
receipt. Queries must not sum cumulative snapshots.

### C5S5 — P2 — timestamp precision/range failures overlap ambiguously

A timestamp can be both sub-nanosecond and outside Decimal128 range. Add a
combined state or precedence plus complete child/status matrix for source and
Created timestamps.

### C5S6 — P2 — authoritative frame Clock remains outcome-dependent

Pre-I/O timeout and other failures have no unique chosen Clock fact. Freeze a
frame-kind/outcome matrix, persist a non-null attempt observation Clock, and
derive single-frame coverage min/max from the exact WAL header value.
