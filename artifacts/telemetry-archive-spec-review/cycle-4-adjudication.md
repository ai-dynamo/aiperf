# Telemetry archive/watch spec — cycle 4 default-refute adjudication

Target: `bfa45bd47`

Claims: `55ff1c950`

Every claim family was adjudicated by a reviewer who did not author it. The
default was refutation; only concrete correctness/interoperability gaps survive.

## Verdicts

| Claim | Verdict | Severity | Required disposition |
|---|---|---:|---|
| C4A1 | confirmed | P1 | Fix owner identity before authoritative row/projection hashing. Duplicate C4D1. |
| C4A2 | confirmed | P1 | Serialize source-local epoch transition/marker/final hashing by source record sequence. |
| C4A3 | confirmed | P1 | Add one exact structured boundary-attempt-marker join authority. |
| C4A4 | confirmed | P1 | Compare a canonical archive-independent native-measurement projection, not full report bytes. |
| C4A5 | confirmed | P2 | Persist a receipt-observer epoch/anchor for every execution, including sync-only. |
| C4A6 | confirmed | P1 | Add bounded loss-ledger saturation summaries. Duplicate C4D3. |
| C4D1 | partially confirmed | P1 | Duplicate C4A1; pre-reserving a failed sequence is optional if stamped finalization occurs before WAL. |
| C4D2 | confirmed | P1 | Make shared raw objects bytes-only and move normalized content encoding to each reference, or include it in equality. |
| C4D3 | partially confirmed | P1 | Duplicate C4A6; crash recovery after writer death needs a separate durable emergency journal if promised. |
| C4D4 | refuted | — | Active bootstrap create followed by terminal absent-claim CAS is already legal; “no second release” forbids only a later claim-only CAS. |
| C4S1 | refuted | — | Exact logical-row byte minutiae are intentionally authority of mandatory checked-in descriptors/goldens. |
| C4S2 | confirmed | P1 | Freeze complete `canonical-json.v1` string/key/escape/duplicate/array rules. |
| C4S3 | partially confirmed | P1 | Freeze per-object-kind index key derivation; a new session component is optional because logical digest can disambiguate. |
| C4S4 | confirmed | P1 | Add canonical receipt descriptors/query schema/stable object version plus observer epoch. |
| C4S5 | partially confirmed | P1 | Physical children remain descriptor-owned; freeze semantic payload projection from emitted wire roles. |
| C4S6 | confirmed | P2 | Freeze decimal-to-binary64 rounding and child/status validity. |
| C4S7 | confirmed | P2 | Freeze timestamp equality and representative lexeme. |
| C4S8 | partially confirmed | P2 | Descriptor-owned enum registries stand; define null collation and loss-sequence scope. |
| C4S9 | confirmed | P2 | Freeze persistent-index deletion/rebalancing semantics. |

## Unified fourth-cycle correction set

1. Move owner sequence/frame assignment before authoritative projection hashing
   (or add an ordered stamped finalization stage), and serialize chained
   attribute-epoch state by source-record sequence.
2. Persist structured boundary references that join forced attempts to exact
   lifecycle markers, including coalescing group and role.
3. Define `NativeMeasurementParityV1` as the byte-parity authority, excluding
   archive-only report/provenance/artifact fields.
4. Give every receipt event a durable observer epoch/anchor and freeze target,
   event, batch, head/pointer, object-version, and query-relation descriptors.
5. Give the fixed-memory loss ledger a bounded saturation summary with exact
   totals/range endpoints, omitted count, rolling digest, and completeness bit.
6. Make shared raw-object equality semantically sound by putting response-
   specific content encoding on each raw reference.
7. Freeze complete canonical JSON plus per-object-kind manifest-index keys and
   deterministic copy-on-write deletion/rebalancing.
8. Complete semantic normalization: histogram payload versus emitted wire
   roles, binary64 conversion/status matrix, timestamp equality/representative,
   nullable-source ordering, and loss-sequence scope.

## Refuted/limited fixes

- Do not duplicate literal logical-row header/table-ID minutiae inline; the
  mandatory checked-in logical-row descriptor and cross-language bytes remain
  their authority.
- A reserved sequence consumed by a loss frame is one valid design, not the
  only correction; stamped authoritative finalization before WAL is sufficient.
- No special direct-to-terminal first remote create is required. An active-
  claim bootstrap create followed by the terminal absent-claim CAS is legal.
- Do not duplicate full nested Arrow field layouts or every enum registry in
  prose when canonical descriptors own them; only the missing semantic
  projection/ordering rules must be explicit.
- An in-memory saturation digest cannot survive a simultaneous process crash;
  do not promise that without a separately designed durable emergency journal.
