# Telemetry archive/watch spec — cycle 2 default-refute adjudication

Target: `9e74a4816`

Claims: `ba2c1d268`

Each claim family was adjudicated by an agent that did not author it. The
default was refutation. This record freezes the root disposition used for the
second correction pass.

## Architecture/integration verdicts

| Claim | Verdict | Severity | Required correction |
|---|---|---:|---|
| C2A1 | confirmed | P1 | Factory returns an erased prepared driver that internally owns the complete typed pipeline. |
| C2A2 | partially confirmed | P1 | Backend prepares a profile-bound control handle/pool isolated from inference capacity; equivalent profiles may share. |
| C2A3 | confirmed | P1 | Strict archive parse and named native-compat fallback are separate outcomes over one body. |
| C2A4 | confirmed | P1 | Guaranteed shared/native decode plus a separate owned nonblocking archive permit. |
| C2A5 | confirmed | P1 | Sole owner assigns sequence; resolve permits/final frames before closing frame admission and capturing fence. |
| C2A6 | confirmed | P1 | Freeze exact `Send + 'static` offload boundaries without global `Send` overreach. |
| C2A7 | confirmed | P1 | Policy-gated exact encoded/decoded byte lease reaches only raw/archive projection. |
| C2A8 | confirmed | P1 | Integrate deterministic external progress with virtual quiescence or use inline capture/post-sim persistence. |
| C2A9 | confirmed | P1 | Physical attempt ID plus phase-membership fanout and dedup preserves seamless native parity. |
| C2A10 | partially confirmed | P2 | Orchestrator issues an explicit coalescing group for exactly the transitions intended; never use proximity. |
| C2A11 | confirmed | P1 | Defer Graph attachment from v1/code-truth scope rather than implying scheduled lifecycle exists. |
| C2A12 | partially confirmed | P2 | Best-effort uses LocalSet loss ledger in successful report; primary/required uses diagnostic artifact. |
| C2A13 | confirmed | P2 | Separate cadence target from one exact call-deadline formula and prepared-profile timeout ownership. |

## Durability verdicts

| Claim | Verdict | Severity | Required correction |
|---|---|---:|---|
| C2D1 | confirmed | P1 | Under the lock/fence, resume recovers and commits `session_started` before new WAL/source activation; genesis is create-only. |
| C2D2 | partially confirmed | P1 | Only incomplete physical EOF tail is discardable; complete checksum failure restores or fails closed. |
| C2D3 | confirmed | P1 | One-generation-lag WAL retirement keeps the current-to-previous delta recoverable and reserves its space. |
| C2D4 | confirmed | P1 | Future durability/publication facts move from attempt/marker rows to non-self-referential receipt state. |
| C2D5 | confirmed | P1 | Duplicate of C2A5. |
| C2D6 | confirmed | P1 | Acquire canonical-spool/remote writer fence before activation; crash takeover is explicit, not time-lease guessing. |
| C2D7 | confirmed | P2 | Uncertain CAS rereads; exact desired head is idempotent success. |
| C2D8 | confirmed | P1 | Raw object is an optional required frame projection with opaque descriptor and local/remote coverage. |
| C2D9 | confirmed | P1 | Per-frame/table logical row count plus canonical multiset digest proves compaction equality. |
| C2D10 | partially confirmed | P2 | Bound compaction subsets/subtree swaps; publish the verified descendant path from remote ancestor. |
| C2D11 | confirmed | P2 | Qualify filesystem/lock capabilities and rerun recovery under an open-descriptor lifetime lock. |

## Schema/security/query verdicts

| Claim | Verdict | Severity | Required correction |
|---|---|---:|---|
| C2S1 | confirmed | P1 | Physical sample unit is one ordered MetricPoint with point-owned payload/wire samples. |
| C2S2 | confirmed | P1 | Add required family-metadata projection/table, including zero-point families. |
| C2S3 | confirmed | P1 | Preserve numeric lexeme and exact u64 independently from analytical f64 exactness. |
| C2S4 | confirmed | P1 | Store declared/actual format and exact checked timestamp normalization status. |
| C2S5 | confirmed | P1 | Freeze per-format/per-role validity and reject invalid OpenMetrics atomically. |
| C2S6 | partially confirmed | P2 | Define unified source-record sequence/gap identity and exact integer types; two namespaces are optional. |
| C2S7 | confirmed | P1 | Duplicate of C2A5/C2D5. |
| C2S8 | confirmed | P1 | Duplicate of C2D4. |
| C2S9 | confirmed | P1 | Duplicate of C2D8. |
| C2S10 | partially confirmed | P1 | Define keyed byte domains and AEAD envelope; both digests need not be stored unless used. |
| C2S11 | confirmed | P1 | Duplicate of C2A4. |
| C2S12 | partially confirmed | P2 | Non-disableable removal of AIPerf-known credentials; arbitrary source content remains classified sensitive data. |
| C2S13 | confirmed | P1 | Freeze independent-reader head/generation/index node encoding/order/hash/pruning contract. |
| C2S14 | confirmed | P1 | Duplicate of C2A3. |
| C2S15 | confirmed | P1 | Durable atomic local publication receipt keyed to sealed generation/CAS; avoid recursive generation publication. |
| C2S16 | confirmed | P2 | Source-specific prepared control profile owns/connects ceiling semantics. |
| C2S17 | confirmed | P2 | Freeze attribute-epoch digest domain/input/transition ordering. |

## Unified second-cycle correction set

1. Replace the incomplete source object with an erased prepared driver around a
   typed, thread-qualified pipeline; prepare dedicated/profile-shareable control
   transports, define two-stage admission, exact byte leases, deadline ownership,
   and virtual external-progress behavior.
2. Preserve current native compatibility and phase semantics explicitly: strict
   archive parse versus named fallback, attempt-ID phase membership, orchestrator
   coalescing groups, best-effort loss reporting, and v1 deferral of Graph
   attachment.
3. Correct the archive data model: one MetricPoint per row, a family-metadata
   table, lexeme/exact integer facts, declared/actual parser format, role-validity
   matrix, unified record/gap identity, raw-object descriptors, and canonical
   attribute epochs.
4. Freeze the non-Parquet wire contract too: head, generation, persistent-index
   page encoding/comparator/pruning, per-projection logical digests, and bounded
   compaction transactions.
5. Complete startup/stop durability: create-only genesis, lock/fence/reread,
   durable resumed-session generation, sole sequence owner, terminal frame fence,
   one-generation-lag WAL retention, and qualified filesystem semantics.
6. Remove self-reference: attempt rows do not predict durability/publication;
   atomic receipt objects attest earlier sealed generations/ranges, and uncertain
   CAS reconciles by rereading exact head state.
7. Make raw retention transactional and cryptographically exact: encoded versus
   decoded byte domains, subkeys, AEAD envelope, opaque indexed descriptor, and
   local/remote coverage before WAL retirement/finalization.

## Guardrails retained from refutation

- Equivalent control-plane profiles may share a dedicated telemetry pool; a
  separate pool per source is not mandatory.
- Do not globally add `Send`; ownership/offload boundaries determine it.
- Do not classify native compatibility fallback as strict archive success.
- Do not use timestamp proximity to coalesce boundaries.
- Do not solve virtual progress by mapping OS worker latency into virtual time.
- Do not introduce an expiring wall-clock writer lease.
- Do not use physical Parquet bytes as compaction equality.
- Do not recursively publish a new generation merely to record publication of
  the preceding generation.
- Do not promise removal of arbitrary unknown secrets while retaining exact
  source content; the absolute guarantee covers AIPerf-known credentials.

