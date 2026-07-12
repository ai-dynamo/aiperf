# Telemetry archive/watch spec — adversarial review cycle 6 claims

Target: `50b595405`

All three reviewers read the complete 3,011-line target. Claims are frozen
before cross-refutation or edits.

## Architecture/runtime/protocol

### C6A1 — P1 — boundary correlation is not source-cardinal or loss-aware

Multiple physical sources produce distinct attempts/losses, but a transition
has one source-less boundary reference/marker. Add per-source transition capture
mapping to attempt or loss; scope coalescing groups per physical driver.

### C6A2 — P1 — projection-failure loss lacks a LocalSet Clock bridge

The owner learns worker failure but cannot read Clock while `loss_observed_ns`
requires a LocalSet sealing time. Add a tracked failure terminalization round
trip or select a previously captured Clock fact explicitly.

### C6A3 — P1 — sync sink identity and invocation access are not partitioned

Genesis hashes normalized sink config while sync re-authors a generic sink with
new credentials. Split immutable writer/sink identity from invocation-only store
access, or require factory-owned projections.

### C6A4 — P1 — absent-claim compaction CAS contradicts general active-claim rule

Define explicit collection/publication CAS authorization under active claim and
logical-equivalence finalized compaction under exact absent-claim head/version.

### C6A5 — P2 — sync-only report has no truthful session field

Sync creates no telemetry session but report requires one scalar `session_id`.
Use nullable execution collection session plus observer epoch/execution ID and
an optional latest historical collection session.

### C6A6 — P2 — ownership diagram still assigns frame ID before projection

Update it to owner record sequence/reservation, worker success ID, and owner
failure/loss ID.

## Durability/recovery/security

### C6D1 — P1 — raw terminalization can advance an uncommitted attribute epoch

The next epoch/source job may start before owner-side raw envelope creation can
fail. Resolve all raw candidates before committing epoch state/releasing the
strand; failure becomes loss on prior epoch or fail-stop.

### C6D2 — P1 — raw nonce profile lacks uniqueness/misuse resistance

Freeze AEAD/CSPRNG/per-key bounds/collision handling or a misuse-resistant
profile, and test forced nonce collisions.

### C6D3 — P1 — uncertain first creation of remote `LATEST` is undefined

Add typed head-create conflict/uncertain outcomes and reread through visibility
horizon; exact desired bootstrap head/claim is idempotent success with observed
version, any other head conflicts.

### C6D4 — P2 — WAL CRC32C preimage is self-ambiguous

Freeze polynomial, byte order, seed/final XOR, and exact bytes preceding the
CRC; explicitly exclude the CRC field.

## Schema/query/canonicalization

### C6S1 — P1 — OpenMetrics text cannot reveal Info identity/value label partition

Text flattens metric and Info value labels. Use the complete merged wire label
set for identity, retain it explicitly, and mark value partition unavailable
unless a persisted family policy exists.

### C6S2 — P1 — singular frame Clock is unsafe for multi-time marker/loss rows

Require one timed row or equal Clock values per frame, or store actual row min/
max. Pin topology marker clock equality when sharing a scrape frame.

### C6S3 — P1 — marker/loss terminal identity preimages are undefined

Add a closed frame-identity matrix for scrape, marker, exact/global/saturation/
projection-failure loss, and boundary-capture kinds, including batch,
reservation, and terminal frame preimages.

### C6S4 — P2 — WAL receipt coalescing may cross segment boundaries

Restrict every coalesced range to a contiguous subrange of one named segment
ending at its named prefix.

### C6S5 — P2 — batched B-tree mutation order is not canonical

Freeze removals then additions in ascending composite-key order (or an exact
operation log) and add permutation goldens.
