# Telemetry archive/watch spec — cycle 1 default-refute adjudication

Target reviewed: `specs/telemetry.md`

Claims commit: `e31d5da98`

Three agents adjudicated claim families they did not author. Their required
default was refutation, and every accepted claim was rechecked against the full
spec and current code/authority. This record is the root disposition used for
the correction pass.

Verdicts:

- `confirmed`: the written design contradicts or cannot establish a stated
  invariant;
- `partially confirmed`: the broad claim is false, but a narrower normative gap
  remains;
- `refuted`: no correction is required for that claim;
- duplicate claims share one correction but retain their individual verdicts.

## Architecture claims

| Claim | Verdict | Final severity | Disposition |
|---|---|---:|---|
| A1 | confirmed | P1 | Make the v2 outer run resources workload-scoped and show a complete watch envelope. |
| A2 | partially confirmed | P2 | Define the already-required prepared-backend `ControlPlaneHttp` capability and per-call Clock deadline. |
| A3 | confirmed | P1 | Replace the lossy success-only observer with one pre-projection, all-outcome attempt envelope. |
| A4 | confirmed | P1 | Emit typed markers from the authoritative `PhaseObserver`, not independently sampled sidecar timestamps. |
| A5 | confirmed | P1 | Both modes use one run-owned fixed-deadline driver per physical source; phases do not own cadence loops. |
| A6 | confirmed | P1 | Give continuous/boundary calls absolute Clock deadlines, cancellation semantics, and timeout outcomes. |
| A7 | partially confirmed | P2 | Specify policy factory registries and only the `Send` bounds required by the policy's execution owner. |
| A8 | confirmed | P1 | Use immutable local generations and a directory-durable head. |
| A9 | confirmed | P1 | Use sealed immutable WAL segments; never rewrite away a pending suffix. |
| A10 | confirmed | P1 | Pin accepted exposition versions and model every accepted semantic role/timestamp/exemplar. |
| A11 | confirmed | P2 | A LocalSet maintenance driver sends Clock-stamped maintenance commands to the IO owner. |
| A12 | confirmed | P2 | Enforce streaming bounds/admission and use bounded ordered decode/projection offload where the attached profile requires it. |
| A13 | confirmed | P2 | Enrichers receive an immutable view and return an attribute-only patch. |
| A14 | partially confirmed | P2 | Required archive failure has no authoritative report path; optional evidence uses a separately typed diagnostic-artifact field. |
| A15 | refuted | — | One anchor is an approximate placement and does not promise continuous UTC accuracy. Clarify only. |

## Durability claims

| Claim | Verdict | Final severity | Disposition |
|---|---|---:|---|
| D1 | partially confirmed | P2 | Distinguish accepted, internally durable, and receipt-observed operations; uncertain durable frames may recover. |
| D2 | confirmed | P1 | Commit and fsync genesis generation zero before admitting frames or activating sources. |
| D3 | confirmed | P1 | Hash-linked immutable generations plus a file- and directory-durable local head; retire WAL afterward. |
| D4 | confirmed | P1 | Sealed WAL segments and whole-segment retirement. |
| D5 | confirmed | P1 | Commit complete per-table projection coverage for a WAL prefix before global dedup/retirement. |
| D6 | partially confirmed | P2 | Close admission and capture a final sequence watermark before drain/final commit. |
| D7 | partially confirmed | P2 | Create-only, content-addressed generation objects; the head names exact object/hash/parent/archive. |
| D8 | partially confirmed | P2 | A narrow store capability contract; never emulate conditional head update with GET plus PUT. |
| D9 | partially confirmed | P2 | Hash-linked ancestry and explicit reconciliation; one immutable generation sequence is sufficient if remote preserves it. |
| D10 | partially confirmed | P2 | Require named-object read-after-write or bounded transient-visibility retries before corruption. |
| D11 | partially confirmed | P2 | Reserve full transaction working space plus filesystem blocks/inodes before admission. |
| D12 | refuted | — | One mutable owner may drive bounded async uploads without blocking local progress; clarify, do not add another mutable owner. |
| D13 | confirmed | P1 | Attempts and markers share an identified, checksummed WAL frame/receipt/coverage protocol. |
| D14 | refuted as stated | — | Flat metadata was consciously allowed, but S12 independently confirms the always-on scalability contradiction. |
| D15 | confirmed | P2 | Add explicit local/remote finalization states and fenced sync-only resume. |
| D16 | partially confirmed | P2 | Compaction binds an exact parent/head, lock/fence, and complete replacement coverage. |

## Schema/security/capacity claims

| Claim | Verdict | Final severity | Disposition |
|---|---|---:|---|
| S1 | confirmed | P1 | Duplicate of A3. |
| S2 | confirmed | P1 | Use one tagged numeric representation at every numeric leaf. |
| S3 | partially confirmed | P1 | Duplicate of A10; an explicit accepted subset is valid, silent normalization is not. |
| S4 | partially confirmed | P2 | Unchanged body is orthogonal to successful parse; v1 stores full rows and an unchanged reference fact. |
| S5 | partially confirmed | P2 | Define sanitization for every structured durable surface; exact raw retention uses separate classification/encryption policy. |
| S6 | confirmed | P1 | Keep keyed pre-redaction source identity plus post-redaction display identity and reject accidental coalescing. |
| S7 | partially confirmed | P2 | Duplicate of A12 with attached-profile gating. |
| S8 | partially confirmed | P2 | Freeze exact Arrow physical schema, nullability, metadata, and fingerprints before increment 1 completes. |
| S9 | confirmed | P1 | Duplicate of A8/D3. |
| S10 | partially confirmed | P2 | Persist a fixed-memory per-source coalescing loss ledger over the reserved lane/final diagnostic. |
| S11 | partially confirmed | P1 | Strong byte integrity, exact head/manifest URIs, immutable create, and capability-gated CAS. |
| S12 | confirmed | P2 | Always-on v1 cannot rewrite a flat full partition list per rotation; use bounded manifest segments/deltas. |
| S13 | partially confirmed | P2 | Domain-separated length-prefixed digests; attributes stay outside series identity but receive topology epochs. |
| S14 | partially confirmed | P2 | Strict HTTP credential/TLS/redirect/proxy/negotiation/size controls owned by the source factory. |
| S15 | confirmed | P3 | Define bracketed anchor capture and distinguish capture uncertainty from later drift. |
| S16 | refuted | — | Native-v2 expressly permits additive typed mode blocks; retain `2.0` and add compatibility goldens. |
| S17 | confirmed | P2 | Define versioned, numeric, measured acceptance profiles before capability advertisement. |

## Unified correction set

The accepted claims reduce to these coherent changes:

1. **Runner composition:** revise the preimplementation v2 authored envelope so
   factories declare required/optional/forbidden resources; add a typed
   `ControlPlaneHttp` backend capability and full watch wire example.
2. **Observation and scheduling:** make a run-owned driver emit one bounded,
   all-outcome attempt envelope; tee native and archive projections before
   information loss; enforce absolute Clock deadlines and exact `PhaseObserver`
   markers.
3. **Worker ownership:** keep source/lifecycle/Clock control on the LocalSet,
   bounded ordered decode/projection and blocking archive work off it, and one
   mutable archive state owner. Timestamped maintenance commands cross the
   boundary.
4. **Schema:** pin Prometheus text 0.0.4 and OpenMetrics text 1.0.0, model the
   complete accepted feature set, use tagged numeric leaves, freeze exact Arrow
   types/fingerprints, and define unchanged-success behavior.
5. **Identity and security:** use versioned length-prefixed digest domains,
   keyed pre-redaction source identity, restricted enrichment, full structured
   surface sanitization, topology epochs, strict HTTP policy, and separately
   classified exact raw-body retention.
6. **Commit state machine:** durable genesis, sealed WAL segments, attempt and
   marker frames, complete multi-table projection coverage, immutable
   hash-linked manifests, a directory-durable local head, explicit accepted/
   durable/observed states, ordered finalization watermark, and transaction
   space reservation.
7. **Remote publication:** require create-only immutable objects, strong
   integrity and conditional head capabilities, ancestry reconciliation,
   transient visibility rules, sync-only finalization resume, and exact-parent
   compaction.
8. **Long-run/query contract:** use bounded incremental manifest metadata,
   expose exact head/immutable-manifest locations, preserve additive native-v2
   compatibility, define failure diagnostics separately, and add measured
   numeric acceptance profiles.

## Explicitly rejected corrections

- Do not bump native-v2 from 2.0 solely for the optional archive block.
- Do not make later wall anchors authoritative or remap the monotonic session.
- Do not add a second mutable archive owner merely to perform remote IO.
- Do not require topology attributes to enter source-series identity.
- Do not make raw bodies pass through a structured sample redactor; exact raw
  retention is a separately classified and protected feature.
- Do not make every policy `Send`; ownership determines the bound.
- Do not emit an incomplete native report on the authoritative success path.

