# Telemetry archive/watch spec — cycle 6 default-refute adjudication

Target: `50b595405`

Claims: `1948a5577`

Every claim family was adjudicated by a reviewer who did not author it. The
default was refutation. Findings here concern the proposed AIPerf archive/watch
contract; they are not additional Tachometer defect claims.

## Verdicts

| Claim | Verdict | Severity | Required disposition |
|---|---|---:|---|
| C6A1 | confirmed | P1 | Model boundary captures per physical source and correlate each expected capture to its attempt or loss outcome. |
| C6A2 | confirmed | P1 | Round-trip projection failure to its tracked LocalSet for the terminal Clock stamp before constructing the loss frame. |
| C6A3 | partially confirmed | P1 | Give each sink factory separate canonical persistent-identity and invocation-access projections. |
| C6A4 | partially confirmed | P2 | Scope active-claim updates to collection/publication and preserve exact absent-claim finalized-compaction CAS. |
| C6A5 | confirmed | P2 | Make the current collection session nullable and report execution/observer-epoch plus optional latest historical session. |
| C6A6 | partially confirmed | P2 | Correct the ownership diagram; the detailed runtime rule is already sound. |
| C6D1 | confirmed | P1 | Commit the next attribute epoch and release its source strand only after owner-side raw terminalization succeeds. |
| C6D2 | partially confirmed | P1 | Freeze an AEAD/CSPRNG/nonce-width/per-key-limit/collision-or-rekey profile in the raw-retention descriptor. |
| C6D3 | refuted | — | Existing idempotent head reread/reconciliation covers uncertain first creation; no required change. |
| C6D4 | partially confirmed | P2 | Freeze standard CRC-32C storage byte order and its exact preimage excluding the CRC field. |
| C6S1 | confirmed | P1 | Treat text Info labels as one merged wire identity unless a genesis-persisted family policy defines a split. |
| C6S2 | confirmed | P1 | Require one authoritative Clock across every timed row in a frame or persist actual frame time bounds. |
| C6S3 | partially confirmed | P1 | Close the identity matrix for marker-only and all loss/control-frame variants. |
| C6S4 | confirmed | P2 | Coalesce WAL receipt targets only within one named segment and durable prefix. |
| C6S5 | confirmed | P2 | Canonicalize transaction removals/additions and reject duplicate or conflicting B-tree operations. |

## Decisive refutations and narrowing

- **C6D3:** `put_if_absent`, `read_head`, a visibility horizon, and the §11.2
  exact-desired-head reconciliation rule already make a lost first-create
  response recoverable. A generic error type name does not negate that
  protocol. An editorial cross-reference may help, but no new state machine is
  required.
- **C6A4:** the finalized-compaction protocol is already safely fenced by the
  canonical-spool lock, exact parent, exact object version, and absent claim.
  The defect is only the earlier absolute phrase “every later head update”; no
  maintenance claim should be added.
- **C6A6:** terminal identity sequencing is correct in the normative prose. The
  stale diagram is dangerous implementation guidance, but not a second runtime
  flaw.
- **C6D2:** same-object nonce reuse is already prevented by the single-owner raw
  registry. The surviving gap concerns distinct objects under one key; a
  misuse-resistant AEAD is one valid profile, not an additional universal
  requirement.
- **C6D4:** BLAKE3 and the prefix chain remain integrity authority. The CRC gap
  is wire interoperability, not an unprotected-integrity failure.
- **C6S3:** success and projection-failure sequencing already exists. Only
  marker-only, ordinary/global loss, and saturation control-frame identity
  inputs remain unspecified.

## Unified sixth-cycle correction set

1. Add a source-cardinal boundary-capture relation that closes every expected
   physical source as attempt or loss, with coalescing groups local to one
   physical driver.
2. Add a tracked owner-to-origin-LocalSet terminalization bridge for projection
   failures and include it in drain accounting.
3. Split sink persistent identity from invocation-only object-store access and
   make sync-only reporting truthful about its lack of a collection session.
4. Separate open collection/publication claim authorization from finalized
   absent-claim compaction authorization, and update the stale ownership
   diagram.
5. Delay attribute-epoch commit/strand release until raw envelope
   terminalization, and freeze a concrete fail-closed raw encryption/nonce
   profile.
6. Define text-native Info identity, timed-row/frame-Clock invariants, and a
   closed control-frame identity matrix.
7. Restrict receipt coalescing to one WAL segment, freeze canonical B-tree
   mutation order, and make CRC-32C bytes interoperable.

## Tachometer boundary

The confirmed Tachometer findings remain the five source/runtime-reproduced
defects in `artifacts/code-review.md`: histogram cross-label contamination,
stale-checkpoint duplication, Float32 precision loss, quoted-label truncation,
and acceptance of non-2xx metric bodies. This cycle does not enlarge that list.
