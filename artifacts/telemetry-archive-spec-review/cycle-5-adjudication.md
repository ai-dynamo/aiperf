# Telemetry archive/watch spec — cycle 5 default-refute adjudication

Target: `78c4ce8a6`

Claims: `f0e1b4b55`

Every family was adjudicated by a reviewer who did not author it. The default
was refutation.

## Verdicts

| Claim | Verdict | Severity | Required disposition |
|---|---|---:|---|
| C5A1 | confirmed | P1 | Atomically submit or seal all boundary-group subscribers before fetch. |
| C5A2 | partially confirmed | P1 | Byte parity uses one captured/deterministic event stream; real pairs use invariants/statistics. |
| C5A3 | confirmed | P1 | Dynamically cancel/redeadline and join an active fetch during stop. |
| C5A4 | refuted | — | Exact parent + exclusive local lock + versioned CAS already fence compaction; no maintenance claim. |
| C5A5 | confirmed | P1 | Make sync-only requirements/preparation source-free and provide a strict envelope. |
| C5A6 | partially confirmed | P2 | Narrow receipt handshake to WAL and remote publication; local head completion may remain direct. |
| C5D1 | confirmed | P1 | Add first-class receipt-index epoch records/head counts/epoch-only transaction. Duplicate C5S1. |
| C5D2 | partially confirmed | P2 | Choose stable cumulative latest-wins snapshots or exact disjoint intervals. Duplicate C5S4 root. |
| C5D3 | partially confirmed | P2 | Distinguish success-candidate and terminal loss frame IDs under one reserved sequence. |
| C5D4 | partially confirmed | P2 | Freeze exact WAL frame/final header-payload integrity and sealed/open prefix verification. |
| C5D5 | partially confirmed | P2 | State authenticated trust roots and narrow hashes to corruption/substitution relative to them. |
| C5S1 | confirmed | P1 | Duplicate C5D1. |
| C5S2 | confirmed | P1 | Coalesce only before one aggregate completion Clock stamp; event targets are immutable. |
| C5S3 | refuted | — | One COW transaction already permits partition plus affected coverage replacement; descriptor owns fragment IDs. |
| C5S4 | confirmed | P2 | Duplicate C5D2 query facet. |
| C5S5 | confirmed | P2 | Add combined precision/range timestamp state or precedence and validity matrix. |
| C5S6 | confirmed | P2 | Freeze per-outcome frame Clock and persist a non-null attempt observation Clock. |

## Unified fifth-cycle correction set

1. Replace per-subscriber boundary commands with an atomically sealed group
   command, and dynamically cancel/redeadline active fetches during stop.
2. Restrict byte parity to identical captured/deterministic event streams;
   retain real-run formula/order and statistical gates.
3. Make `finalize_remote` resource requirements/preparation source-free and
   give it a complete strict envelope.
4. Make observer epochs first-class receipt-index records and restrict receipt
   coalescing to pre-observation aggregate completions. Local head completion
   does not require a publication receipt.
5. Freeze cumulative saturation slot/snapshot/latest-wins semantics.
6. Derive the terminal success or loss frame ID after terminal kind and add a
   cryptographic final WAL frame/prefix integrity envelope.
7. State the authenticated spool/head/store trust roots and scope unkeyed hashes
   to corruption/substitution relative to them.
8. Add the combined timestamp status and exact per-outcome authoritative frame
   Clock mapping.

## Refuted/limited fixes

- Do not add a remote maintenance claim for compaction: exclusive local lock,
  exact parent, and conditional head version already fence it. Compaction CAS
  must simply preserve terminal absent-claim state and fail on any change.
- Do not require a new local-generation receipt target unless queryable local
  completion observation time becomes a product requirement; direct verified
  local head completion is sufficient.
- Do not redesign fragment identifiers in prose: compaction's bounded COW
  descriptor swap already includes affected coverage, and checked-in
  descriptors/goldens own the representation.
- Saturation crash safety does not force disjoint double buffering; stable
  cumulative snapshots with monotonic sequence and latest-wins reduction are
  sufficient.
