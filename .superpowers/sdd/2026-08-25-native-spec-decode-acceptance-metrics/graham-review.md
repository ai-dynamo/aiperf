# Full-range Graham review — first pass

## Range and verdict

- Base: `8b5194bcfc26475c5e06030d8701c82b66eb7b6a`
- Head: `0c57560d39`
- Verdict: **NOT APPROVED**

The reviewer inspected every changed hunk twice, including endpoint parsing,
typed SSE reduction, dispatch/observer propagation, exact and sketch metric
folds, cellular compatibility, default metadata, exporters, the mock fixture,
and the real-profile E2E.

## Important findings

1. `endpoints/spec_decode.rs` borrowed the terminal stats `Value` and then
   deep-cloned its histogram and per-step vectors for `serde_json::from_value`,
   even though the endpoint-dispatch caller owned the retained terminal value.
2. `metrics.rs` deep-cloned the canonical DTO for every `ObserverTee` delegate.
   Standard topology includes a consumer using the default no-op callback, so
   one complete vectors/histogram allocation was immediately discarded while
   another was retained and the original was dropped.

Required fix: consume the owned terminal `Value` during normalization and make
the observer event borrowed so fan-out passes one address and only an interested
retaining observer clones once.

## Minor finding

Remove the new dated, commit-specific history comment above the catalog
fingerprint. The fingerprint invariant already has a timeless re-audit comment;
source control owns the history.

## Reviewer verification

Both passes verified rate validation, finish-only ordering, malformed-payload
degradation, exact `u128` pools and folds, accepted-per-verified arithmetic,
MessagePack compatibility, embedded metadata, artifacts, zero-token tool-call
handling, and default-off behavior. `git diff --check` was clean. No other
correctness, overflow, concurrency, tracing, or artifact defect was validated.
The review was source-based and did not claim test execution.

## Fix-round evidence

Structural tests changed the parser call to transfer an owned `Value` and the
tee recorders to require the same borrowed DTO address. Before production edits,
the engine-enabled focused suite failed compilation with the expected E0308 and
E0053 signature errors. After the minimal ownership changes, the identical
command passed 31 tests with no failures. The native CLI rebuilt successfully,
and the real-profile present/absent E2E passed 14 tests with no failures.
