# Telemetry archive/watch spec — cycle 3 default-refute adjudication

Target: `25183e4bc`

Claims: `951f88f80`

Every family was adjudicated by a different agent under a default-refute rule.

## Verdicts

| Claim | Verdict | Severity | Disposition |
|---|---|---:|---|
| C3A1 | confirmed | P1 | Archive projection returns to a bounded CPU stage after the LocalSet grants a permit. |
| C3A2 | confirmed | P1 | Prepared driver delivers native directly first; one post-native observer records admission outcome. |
| C3A3 | refuted | — | Boundary command subscribers already exist separately; rename continuous membership only for clarity. |
| C3A4 | confirmed | P1 | Exact parity is archive-off/on on the new cadence; old/new gate covers formulas and intentional schedule differences. |
| C3A5 | confirmed | P1 | Define reusable attached archive DTO/resource and complete scheduled envelope. |
| C3A6 | confirmed | P2 | Worker completion returns to LocalSet for Clock stamp, then a receipt draft returns to the owner. |
| C3D1 | confirmed | P1 | Persist zero-row projection coverage and define v1 projection indivisibility/fragment rule. |
| C3D2 | confirmed | P1 | One physical encrypted envelope per raw equality ID plus per-frame references, or unique ciphertext key. |
| C3D3 | partially confirmed | P1 | Discriminated receipt targets; semantic target ID excludes time, observed/recovery times differ. |
| C3D4 | confirmed | P2 | Canonical bounded receipt batches and persistent receipt index/head. |
| C3D5 | refuted | — | Terminal CAS can already clear claim atomically; clarify receipt binds absent-claim state. |
| C3S1 | partially confirmed | P1 | Duplicate of D2; per-frame reference is required only for shared physical-object choice. |
| C3S2 | confirmed | P2 | Receipt index plus global record sequence on every range-joinable relation. |
| C3S3 | confirmed | P1 | Distinct loss-range schema preserves issued and missed identities. |
| C3S4 | confirmed | P1 | Classic component timestamp assembly has explicit absent/uniform/mixed/partial states. |
| C3S5 | confirmed | P1 | Freeze schema-bound canonical logical-row bytes and cross-language goldens. |
| C3S6 | confirmed | P2 | Structured OpenMetrics Created uses timestamp semantics; wire token remains numeric. |
| C3S7 | partially confirmed | P2 | Add record sequence to deterministic physical sort. |
| C3S8 | partially confirmed | P2 | Exact raw artifacts are an acknowledged encrypted exception or reject on known-secret scan. |
| C3S9 | confirmed | P2 | Exact lexeme always; analytical f64 only when representable. |

## Unified correction set

1. Complete the two-stage worker pipeline and event ordering: native callback,
   archive permit, offloaded projection, post-native observation, outstanding-job
   finalization fence, and receipt Clock round trip.
2. Distinguish continuous phase membership from explicit boundary subscribers,
   and define parity relative to archive enablement/new cadence rather than an
   impossible old live schedule equality.
3. Make attachment authorable through a reusable strict archive spec that
   references existing prepared telemetry source IDs and rejects deferred pairs.
4. Add persistent zero-row coverage and a distinct loss-range relation; give
   every table the global record sequence needed for receipt joins.
5. Deduplicate raw retention safely: one randomized physical envelope per keyed
   equality ID and per-frame reference evidence, with no nonce re-encryption.
6. Turn receipts into discriminated, coalesced immutable batches behind a
   canonical persistent index/head. Target identity is independent of optional
   response/recovery observation times.
7. Freeze classic mixed-component timestamp semantics, Created timestamps,
   canonical logical-row bytes, total sample sort, and exact-lexeme/f64 scope.
8. Clarify that acknowledged encrypted exact raw content is outside the
   structured known-credential absence guarantee.

## Refuted/limited fixes

- No new phase-boundary architecture is required; command subscriber identity
  already handles pre-STARTED/post-COMPLETE samples.
- No second terminal CAS is allowed or needed: the final replacement clears the
  active claim in the same operation.
- Do not require per-frame physical raw objects; shared objects plus explicit
  references are the chosen efficient model.
- Receipt observation time is optional and cannot be reconstructed after a
  crash; recovery verification time is a different fact.

