# Runtime WebSocket correlation and affinity repair report

## Basis

This repair addresses the two remaining findings recorded against exact commit
`2f0de16a7ff66718f3873be6f5b5c4b3c5e21fbc`. The three scoped source files are
unchanged between that commit and the working base
`59a3da36dd239642e93dfd1eeb2f1dfdcac7fd1e`.

Rust verification preserved the configured
`RUSTC_WRAPPER=/usr/bin/sccache`. Final runtime verification used a detached
worktree containing the exact three-file scoped diff. Final mock verification
used a separately recreated, uniquely named detached worktree after an
external cleanup removed the first verifier during its mock link.

## Repairs

- Every Responses operation carries a reserved per-attempt marker in request
  metadata. Every Realtime client event carries a per-message `event_id`, and
  `response.create.response.metadata` carries the per-attempt marker.
- `response.created` must echo the current marker before it can arm response
  identity on a reused socket. Stale created, delta, usage, audio, and terminal
  sequences are quarantined before receive accounting, raw capture, endpoint
  parsing, token observation, usage observation, or terminal attribution.
- Explicit error event IDs are attributed only when they match a client event
  from the current operation. A markerless error before correlated creation on
  a reused Responses socket triggers the existing replay-safe fresh-connection
  path. The same ambiguous condition fails the request safely when replay is
  unavailable.
- A fresh socket may accept a markerless created event because it has no prior
  operation. Such a socket is never pooled, so a non-correlating wire behavior
  cannot become an unsafe reused connection.
- Realtime continuation affinity loss after rotation, keepalive expiry,
  eviction or absence, and route change now closes the checked-out socket and
  returns a typed per-request protocol failure. It finalizes the request
  record, retains the raw failed request when enabled, and emits exactly one
  failed observer terminal.
- The mock WebSocket routes parse the reserved marker and echo it in
  `response.created` and terminal response metadata for both Responses and
  Realtime.

## TDD evidence

Before production repair:

- `reused_socket_ignores_stale_response_before_correlated_response` failed
  because stale `response.created` was attributed.
- `rotated_realtime_affinity_is_a_typed_failed_dispatch` failed because
  dispatch returned a run-level error instead of a failed request result.

After repair:

- Reused-socket state tests cover stale response sequences, metadata and
  Realtime event-ID injection, and mismatched errors both before and after the
  current response is created.
- A driver-level real-socket test sends a complete stale response before the
  correlated response and proves that output and raw response capture contain
  only the current operation.
- Real-socket dispatch tests cover rotation, keepalive expiry, eviction or
  absence, and route change. Each asserts a typed protocol error, terminal raw
  record, retained raw request, and exactly one failed observer terminal.
- The mock test asserts that `response.created` echoes the client operation
  marker.

## Verification

Detached runtime verifier `/tmp/aiperf-ws-fourth-verify` before its external
removal:

- Runtime `transport::ws` suite: 35 passed.
- Runtime WebSocket execution suite: 18 passed.

Recreated detached mock verifier `/tmp/ws4-correlation-7b2f`:

- Mock correlation-echo regression: 1 passed.
- Mock WebSocket suite: 25 passed.

Product verification with the locally built native binary:

- Responses and Realtime WebSocket profile tests: 2 passed.

Additional scoped verification:

- Scoped `rustfmt --check`: pass.
- Scoped `git diff --check`: pass.

The Rust commands emitted only warnings from unrelated evaluation code outside
the scoped files. The first product invocation stopped at the documented
`AIPERF_E2E_BIN` harness precondition; the explicit-binary rerun passed both
product cases. The first detached mock link was interrupted when its entire
worktree was removed externally; the uniquely named detached rerun passed.

## Boundary

Only the WebSocket dialect, runtime execution path, mock WebSocket route, their
focused tests, and this report belong to the repair. Unrelated dirty evaluation,
proxy, mock tokenizer, progress, and design-record work is preserved and
excluded from the commit.

This report records implementation and verification evidence. It does not
claim an independent review verdict.
