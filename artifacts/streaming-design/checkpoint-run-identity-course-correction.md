# Streaming Checkpoint Run-Identity Course Correction

Date: 2026-08-26

This record corrects the Task 5A/5B contract before the atomic checkpoint
backend is implemented. The prior backend interface could open a head by run
but could not name a run when beginning a generation. That made empty
generations unaddressable and allowed otherwise identical generation content
to collide across logical runs.

## Stable identity ruling

`StreamRunIdentity` is a checked transparent wrapper around exactly one
`LogicalReplayRunId`:

```rust
pub struct StreamRunIdentity(LogicalReplayRunId);
```

Its constructor accepts `LogicalReplayRunId`, and its accessor returns that
same typed identity. It never contains or accepts `RunIncarnationId`.
Checkpoint and result state must survive process replacement, so a process-
local incarnation cannot participate in the durable run namespace.

## Generation binding ruling

Task 5A owns `StreamRunIdentity` and adds it to canonical
`CheckpointGenerationCandidate` construction, serialization,
self-verification, expectation verification, and domain-separated hashing. The
run is hashed as its own length-framed canonical field. Consequently, two
empty generations with identical epochs, predecessors, cuts, participant
descriptors, plan digests, result roots, and terminal state still have distinct
generation identities when their logical runs differ.

The opaque `CommittedCheckpointGeneration` exposes the verified run through a
borrow-only accessor. Publication authority remains unchanged: a candidate is
not authoritative until a backend CAS or leased-current-root read produces the
private proof required for promotion.

The same private run value propagates into each
`CommittedParticipantReceipt`. A receipt remains bound to the exact committed
generation, participant descriptor digest, and represented cut, now also under
the exact logical run. The coordinator and participant both compare run before
epoch/digest idempotency logic. A receipt from another run is rejected even if
it has a greater epoch and byte-identical participant state.

## Backend ruling

Task 5B's `begin_generation` accepts an explicit `StreamRunIdentity` and
`CheckpointGenerationExpectations` contains the same run. A mismatch is
rejected before staging. The transaction freezes that run, every staged result
descriptor must name it, and candidate construction receives it unchanged.

`open_latest(run, expectations)` requires `run == expectations.run` before it
reads or promotes anything. The memory reference stores a map of per-run
heads; expected-head comparison and publication occur only inside the selected
run entry. A stale writer conflicts with another writer for the same run, not
with activity in another run.

Commit validates every result descriptor's run and epoch against the frozen
transaction run and commit metadata. A generation must contain the exact
participant inventory and exactly one staged canonical result epoch. Empty
work still stages an epoch with zero partitions and canonical zero totals; an
omitted epoch or second result staging is invalid.

A leased reader does not treat object-store presence as read authority.
Participant reads require the complete descriptor to be in the leased
generation's participant inventory, and segment reads require the complete
descriptor to be reachable through that generation's verified result-index
root. Objects reachable only from another generation or run are refused.

## Restart discovery ownership

Task V1 owns the Config-v2/protocol product projection for the explicit
fresh-or-resume choice and the resume locator carrying the exact
`StreamRunIdentity`. Fresh execution allocates the logical identity and commits
the bootstrap generation—with the exact participant inventory and one
zero-partition result epoch—before source polling or endpoint issue. Resume
must receive that identity in the explicit locator/product projection, or a
future catalog must resolve it from that locator. Missing identity is a refusal;
resume never silently allocates a new logical run.

Task 5E consumes the already resolved identity and enforces the initial
commit-before-issue ordering. Task 5C subsequently allocates a fresh
`RunIncarnationId` while acquiring durable writer authority; it does not choose
or replace the logical run.

## Durable writer authority boundary

Task 5B provides run-scoped in-memory CAS semantics only. It does not allocate,
serialize, compare, or recover `RunIncarnationId`, and it does not claim a
durable writer lease. Task 5C owns process-incarnation allocation, the local
single-writer lease, fencing, and crash-durable authority. This separation
prevents a transient writer identity from being smuggled into stable generation
or result identity.

## Required RED evidence

- Candidate digests differ across two logical runs with otherwise identical
  content, including an empty result epoch.
- `RunIncarnationId` cannot type-check as input to `StreamRunIdentity::new`.
- `begin_generation` rejects explicit-run versus expectation-run mismatch
  before staging or budget transfer.
- `open_latest` rejects explicit-run versus expectation-run mismatch before
  exposing a reader.
- The memory backend retains independent heads for two runs and same-run stale
  CAS failure does not modify either head.
- A result partition whose descriptor names a different run or commit epoch is
  rejected and publishes nothing.
- Empty commits reject missing/duplicate result staging and incomplete
  participant inventories, while accepting exactly one canonical zero-partition
  epoch.
- A leased reader refuses participant or segment descriptors not reachable from
  its exact generation even if those objects exist in the backend.
- A greater-epoch committed receipt from another run is rejected before any
  participant callback, including when the descriptor digest is identical.
- A fresh invocation commits its bootstrap generation before issue, and a
  resume request without a resolvable explicit logical-run identity refuses.

## Ownership disposition

- Task 5A continues to own `rust/runtime/src/streaming/checkpoint.rs`, now
  explicitly including stable run identity and canonical candidate run binding.
- Task 5B owns only its planned backend, memory backend, result DTO, module, and
  backend test/support files. It consumes the Task 5A run-bound candidate API.
- Task 5E owns coordinator-side cross-run receipt refusal and
  bootstrap-before-issue enforcement after receiving a resolved logical run.
- Product Task V1 owns explicit fresh/resume run discovery and resume-locator
  projection; a future catalog may implement locator resolution without
  changing the identity contract.
- Task 5C remains the sole owner of durable writer incarnation and lease
  allocation.
- Task 1C blocking-owner files and behavior are unaffected.
