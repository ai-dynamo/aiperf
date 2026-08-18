# Runtime WebSocket authored-metadata lifecycle repair report

## Basis

This repair starts from exact commit
`9fac0b9e11b9f45caa4e6c7d99904ae183d31a77` and addresses the one critical
finding in `runtime-driver-fifth-final-graham.md`. Verification used the clean
detached worktree `/tmp/aiperf-ws-authored-lifecycle-verify-20260818`, created
from that exact commit with only the staged runtime and product-test patch
applied. Before this report was added, the detached and shared staged patches
had the same SHA-256 digest,
`3eb82274683d4e5a0d4ef594d54c76fb12cd7f6a61fbe78bd3e52c50bfb51174`.

Every Rust command preserved the inherited
`RUSTC_WRAPPER=/usr/bin/sccache`.

## Repair

`OperationCorrelation.operation_id = None` means AIPerf did not inject an
internal response-metadata marker. The fresh-socket `response.created` branch
now checks that local fact and the fresh-socket fact directly. It no longer
requires the server's echoed metadata map to omit a marker-looking authored
key.

Consequently, an authored `_aiperf_ws_operation` pair is preserved and its
created/delta/terminal lifecycle is attributed on the fresh single-operation
socket. The authored value never verifies internal correlation: the state
retains `has_verified_correlation = false`, so the completed socket is not
pooled. Reused sockets still require an actually injected marker whose echoed
value matches the current operation.

## TDD evidence

Before the production branch changed:

- The direct lifecycle regression classified the first echoed
  `response.created` as `Unattributed` instead of
  `Attributed { is_terminal: false }`.
- The product regression ran two bounded requests against the in-repo mock.
  Both timed out and the native run reported that all two inference requests
  failed, matching the reviewed failure mode.

After the one-branch repair:

- The direct regression attributes official created, identity-free delta, and
  terminal events in order while proving reusable correlation remains false.
- The product regression proves two requests both return `mock`, their raw
  request artifacts retain the exact compact metadata fragment
  `"metadata":{"_aiperf_ws_operation":"authored-value"}`, and each raw
  response contains created, delta, and completed events.
- The mock capture endpoint reports two completed WebSocket connections for
  the two requests, proving neither fresh markerless socket was retained.

## Verification

All final commands ran from the clean detached verifier:

- Focused authored-metadata lifecycle regression: 1 passed.
- Focused authored-metadata product regression: 1 passed.
- Runtime complete `transport::ws::` suite: 41 passed.
- Runtime WebSocket execution suite: 19 passed.
- Mock-server WebSocket suite: 26 passed.
- Full product WebSocket target: 15 passed.
- Scoped `rustfmt --check`: pass.
- Scoped staged `git diff --check`: pass.

The commands emitted only warnings from evaluation and metrics code outside
the scoped files. The first isolated RED link stopped before test execution
because `/tmp` was full. Only the prior completed verifier's 18 GB rebuildable
Cargo `target/` directory was removed; its source worktree and repository data
were preserved. The RED was then rerun successfully using the existing Cargo
target on `/home`.

## Boundary

Only the WebSocket dialect state machine, the product WebSocket regression,
and this report belong to the repair. Unrelated dirty evaluation, HTTP, mock
tokenizer, design-record, progress, and generated files remain unstaged and
excluded.

This report records implementation and verification evidence. It does not
claim an independent review verdict.
