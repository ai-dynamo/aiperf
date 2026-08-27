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
  no head or object, and releases every transaction/prepared charge.
- A valid next descriptor one byte larger than the caller page limit returns
  `ResultIndexReadBudgetTooSmall` with exact required and maximum values, does
  not acquire backend read capacity, and succeeds when retried with the exact
  required value.
- With that caller limit made sufficient but backend read bytes configured one
  byte smaller, the same read returns
  `BackendBudget { Read, ByteCapacity }` without advancing the cursor.
- Foreign-root, unreachable-block, and out-of-range cursors are rejected as
  object verification before either page-limit or backend-budget errors.

## Ownership disposition

- Task 5B owns these enums, `CheckpointError` variants, `Display` branches,
  backend mappings, and integration regressions.
- Task 5A-R remains exclusively responsible for logical-run authority. Its five
  implementation files and approved behavior are unchanged by this correction.
- Task 5C and later backends consume the same stable vocabulary; they do not
  collapse capacity refusal into storage failure.
