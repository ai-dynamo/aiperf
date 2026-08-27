# Checkpoint Backend Budget Contract Correction

Date: 2026-08-26

This record corrects Task 5B before the atomic checkpoint backend is
implemented. The landed checkpoint error vocabulary distinguishes participant
state capacity from storage failure, but Task 5B introduces backend-owned
transaction, index, storage, summary, and read budgets. Those failures cannot be
reported truthfully as participant `StateBudget`, immutable-object
`ObjectVerification`, or external `Storage` failure.

## Stable backend-budget ruling

Task 5B owns two additions to `CheckpointError` and therefore adds
`rust/runtime/src/streaming/checkpoint.rs` to its file set. It does not change
Task 5A-R's run identity, hashing, proof, receipt, barrier, or participant
ordering.

`CheckpointBackendBudgetKind` identifies the exact backend-owned category:
`Transaction`, `PreparedIndex`, `Storage`, `ResultSummary`, or `Read`.
`CheckpointBackendBudgetFailureCode` is the stable reason:
`ItemCapacity`, `ByteCapacity`, `Closed`, or `Unrepresentable`.
`CheckpointError::BackendBudget { budget, code }` carries both values.

`RequestExceedsCapacity` maps to item capacity when the requested item count
exceeds its configured limit, otherwise byte capacity. If both exceed, item
capacity wins deterministically. A closed budget maps to `Closed`;
unrepresentable permit counts or accounting map to `Unrepresentable`.
Temporary contention is not an error and waits cancellation-safely.

`MemoryCheckpointBackend` has one constructor only:
`new(limits: MemoryCheckpointLimits) -> Result<Self, CheckpointError>`. It
validates the five transaction, prepared-index, storage, result-summary, and
read item/byte limits in that order before retaining backend state. Zero item or
byte capacity maps to the matching kind plus `ItemCapacity` or `ByteCapacity`;
each nonzero limit is then passed through the existing
`StreamingResourceBudget::new` validator. Its `u32::MAX` `acquire_many`
conversion boundary is authoritative: exact-boundary limits are accepted and
the first larger representable `usize` maps to `Unrepresentable`. RED covers
both dimensions and both failure classes for all five kinds.

## Move-only DTO ruling

Every lease-bearing result wrapper has private fields.
`BudgetedResultDescriptor`, `ResultPartition`, `PreparedResultEpoch`,
`BudgetedResultDescriptors`, `ResultSegmentReader`, and `ResultIndexPage` expose
checked construction and borrow-only accessors. Where an enclosing wrapper has
a consuming `into_parts`, it returns the allocation and its authority together.
No public field or accessor returns a `BudgetLease` independently.
`BudgetedResultDescriptors` itself exposes no consuming separation method; the
enclosing wrappers move it intact.

`ResultProjectionId` stores compact `Box<str>`. A budgeted descriptor slice
charges its boxed inline allocation plus every nested projection byte, using
checked arithmetic. The descriptor slice and exact lease remain inseparable
until the wrapper is consumed. Custom deserialization routes through the
checked constructor, so an empty projection cannot bypass the public invariant.

Every input `ResultPartition` owns one private singular
`BudgetedResultDescriptor`; its exact one-item charge includes the inline
descriptor and compact projection allocation. Public partition consumption
returns that wrapper intact with the separately budgeted payload. Only a
crate-private backend transfer may extract the descriptor and input lease.
`stage_results` validates and totals inputs by borrow, acquires the complete
backend prepared-index and returned-summary reservations first, then moves
charged descriptor copies into backend-owned exact-capacity storage and the
separately leased returned summary while the input vector remains intact. Only
after every checked construction succeeds does one infallible synchronous phase
drain the inputs, move their payloads, and drop their original descriptor
authorities. Cancelling either wait or failing checked construction leaves the
transaction and inputs unchanged and retryable. No fallible operation or await
is permitted after the vector or transaction begins to mutate, nor after commit
publication begins.

Task 6A owns the distinct producer-side singular-descriptor budget and maps its
refusal to `ResultPlaneError::PartitionDescriptorCapacityExceeded`; it neither
borrows Task 5B's private budgets nor mislabels the charge as provisional
capacity.

`ResultSegmentReader` retains only its separately budgeted payload. It borrows
the caller's descriptor while verifying length and digest but does not clone or
return that descriptor. Result-index pages do own descriptor clones, and their
read charge therefore includes every compact projection allocation; RED varies
only projection length and observes the exact charge delta.

One aggregate immutable-storage acquisition cannot be divided among objects by
the existing move-only `BudgetLease`. The memory backend therefore stores one
private `Rc<StorageCommitBundle>` owning that aggregate lease, and every object
newly introduced by the commit retains a clone of that bundle handle. The full
charge remains until the last object from the bundle is reclaimed; it may
over-retain but cannot undercharge. Sequential per-object acquisition while
earlier leases are held is forbidden.

## Non-looping result-index ruling

`ResultIndexReadBudget.max_bytes` bounds the actual retained allocation of a
returned descriptor page. When the next reachable valid descriptor cannot fit
by itself, `scan_result_index` returns
`CheckpointError::ResultIndexReadBudgetTooSmall { required_bytes, max_bytes }`.
It never returns an empty page with the same continuation cursor.

The reader validates generation root, block reachability, and cursor offset
before any budget operation. It then computes the compact allocation needed for
the next descriptor. Caller-page refusal precedes backend read-budget
acquisition. If the caller limit is sufficient but the configured backend read
budget cannot represent or admit that single page, the method returns
`BackendBudget { budget: Read, ... }`. Every refusal leaves the cursor, reader,
backend budget snapshots, and authoritative generation unchanged.

## Required RED evidence

- A commit whose one aggregate immutable storage reservation exceeds only the
  storage byte limit returns `BackendBudget { Storage, ByteCapacity }`, publishes
  no head or object, leaves the exact typed immutable-object inventory unchanged,
  preserves its exact object count, and releases every transaction/prepared
  charge.
- Zero or unrepresentable capacity in each `MemoryCheckpointLimits` field is
  rejected by the sole fallible constructor with the exact budget kind/code;
  exact `u32::MAX` capacities remain accepted and `u32::MAX + 1` is refused.
- Moving or borrowing a lease independently from any lease-bearing result
  wrapper does not type-check; compact nested projection bytes participate in
  exact descriptor-summary charging.
- Varying only an input partition's projection length changes its singular
  descriptor charge by the exact byte delta; public consuming access cannot
  separate the descriptor allocation from that charge, and staging acquires the
  aggregate prepared-index and returned-summary authorities before releasing it.
- Cancelling while returned-summary capacity is blocked releases the already
  acquired prepared-index lease, leaves the transaction and caller vector
  unchanged, and permits the same vector to succeed on retry.
- Empty `ResultProjectionId` text is rejected both by direct construction and
  deserialization, and projection length contributes exactly to index-page read
  charging.
- A valid next descriptor one byte larger than the caller page limit returns
  `ResultIndexReadBudgetTooSmall` with exact required and maximum values, does
  not acquire backend read capacity, and succeeds when retried with the exact
  required value.
- With that caller limit made sufficient but backend read bytes configured one
  byte smaller, the same read returns
  `BackendBudget { Read, ByteCapacity }` without advancing the cursor.
- Foreign-root, unreachable-block, and out-of-range cursors are rejected as
  object verification before either page-limit or backend-budget errors.
- Result objects proven present under a superseded generation and another
  logical run remain unreadable from the current generation, with no read-budget
  mutation; mere content-addressed presence is not authority.

The privacy regressions live as `compile_fail` rustdoc directly on the public
DTOs in `results.rs`; an integration-test comment is not executable coverage.
Task 5B GREEN therefore includes `cargo test -p aiperf-runtime --features
streaming --doc`.

## Ownership disposition

- Task 5B owns these enums, `CheckpointError` variants, `Display` branches,
  backend mappings, and integration regressions.
- Task 5A-R remains exclusively responsible for logical-run authority. Its five
  implementation files and approved behavior are unchanged by this correction.
- Task 5C and later backends consume the same stable vocabulary; they do not
  collapse capacity refusal into storage failure.
