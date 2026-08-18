# Runtime WebSocket driver post-review repair report

## Basis

This repair starts from exact commit
`176f0e86885ff41f1eb4db7e1090cecd29d596d7` and addresses every critical and
important finding in `runtime-driver-second-final-graham.md`. Verification runs
in an isolated worktree created from that exact commit, with only the scoped
repair copied into it. The configured `RUSTC_WRAPPER=/usr/bin/sccache` remains
unchanged for every Rust command.

## Repairs

- Responses continuation state is affinity-bound. A non-final response ID is
  cached only with the matching logical session, and dispatch defensively
  ignores continuation state from a generic socket.
- Realtime sends only authored client input after the last assistant turn,
  retains per-turn audio append/commit order, never lowers assistant content as
  input, and requires the affinity-owned socket for continuation turns. Mixed
  text/audio ordering that cannot be represented faithfully fails closed.
- One parsed response may remain pending under observer backpressure while the
  socket driver continues polling deadlines, cancellation, reads, and control
  frames. A second parsed response fails closed instead of creating an
  unbounded queue.
- Local close sends Close, reads the reciprocal Close, and flushes the reply
  under one absolute `Clock` deadline. Peer-initiated Close waits for the split
  writer to flush Tungstenite's automatic reply under its own `Clock` bound.
- Local `cancel_after` and server `Canceled` terminals both report `Canceled`;
  raw request records receive `cancellation_ns` and therefore report
  `was_cancelled()` consistently.
- Prepared WebSocket operations retain the endpoint request projection and its
  extracted inputs. Token counting consumes that projection rather than
  serializing and parsing the artifact envelope on every dispatch; artifact
  capture remains the complete ordered operation.
- `response.created` binds one non-empty response identity. Content, reasoning,
  audio, usage, and terminal events must carry the same identity before raw or
  observer attribution. Mock Realtime responses now use unique
  connection-and-turn response IDs.
- The unnecessary BLAKE3 hasher clone was removed.

## Verification

Completed in `/tmp/aiperf-ws-final-repair`:

- Runtime engine/WebSocket compile: pass.
- WebSocket dialect tests: 17 passed.
- WebSocket execution tests: 13 passed.
- WebSocket transport tests: 32 passed.
- Runtime `websocket_` materialization/capability filter: 17 passed.
- Runtime `realtime_` lowering/event filter: 10 passed.
- Mock-server WebSocket filter: 33 passed.
- Product WebSocket target with an explicitly built isolated `aiperf` binary:
  14 passed, including the Responses and Realtime profile cases.
- Scoped `rustfmt --check`: pass.
- Scoped `git diff --check`: pass.

The compile emits three unrelated pre-existing evaluation warnings: one
`unused_mut` in `engine/record_lane.rs` and dead-code warnings for a struct and
method in `eval/execution/task_environment.rs`. No scoped warning is emitted.

The whole-workspace format check still reports three formatting differences
already present in the exact base (`mock-server/src/lib.rs`,
`mock-server/src/tokens.rs`, and `runtime/src/transport/http/client/proxy.rs`).
Those paths are outside this repair and the scoped format check is clean.

The first product-target invocation stopped at the harness precondition because
`AIPERF_E2E_BIN` was not set. The final invocation used the
freshly built isolated debug binary and passed all tests. The temporary
filesystem filled while building several Rust feature variants; only
rebuildable binaries and rlibs created in the isolated target were removed,
then the required CLI and E2E target were rebuilt successfully.

## Boundary

Only the WebSocket runtime, endpoint lowering, request projection, token-count
call site, mock WebSocket scenario, focused tests, and this report belong to the
repair. Unrelated dirty evaluation, graph, proxy, mock tokenizer, progress, and
design-record work in the shared worktree is preserved and excluded from the
commit.

This report records implementation and verification evidence. It does not claim
an independent review verdict.
