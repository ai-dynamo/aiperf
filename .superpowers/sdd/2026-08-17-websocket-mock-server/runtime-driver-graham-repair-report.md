# Runtime WebSocket Graham repair report

## Scope

This repair responds to every critical and important finding in
`runtime-driver-final-graham.md`. It does not claim a new Graham verdict.

## Repaired contracts

- Realtime input lowering emits valid text conversation items, validates and
  appends decoded audio chunks, commits only a non-empty audio buffer, and then
  creates the response.
- Realtime output uses the official text/audio event shapes and derives the
  terminal classification from `response.status`.
- Realtime affinity reuse selects only input after the last assistant item on
  the exact state-owning socket.
- The direct one-worker executor exposes the same live typed-response seam as
  worker-thread execution.
- Cancellation and stream-idle timing arm only after the final measured input
  flush.
- Writer capacity reserves RFC 6455 data/control overhead, fragments share the
  original `Bytes`, and a real TCP-pressure test proves Pong progress before
  the remaining maximum-sized application frames.
- Retry capture follows one explicit winning-attempt policy: abandoned receive
  facts are cleared, the actual replay operation is captured, and a failed
  fresh retry connect re-evaluates declared fallback under the original
  deadline.
- Pool eviction is capacity-aware, timed-out affinity waiters participate in
  cleanup, worker shutdown closes idle sockets, and active sockets have a
  separate sixty-minute service boundary.
- Terminal snapshots must exactly match the digest of the streamed prefix.
- Inputs and raw artifacts serialize the complete ordered WebSocket operation.
- WebSocket raw records retain upgrade status and typed logical/transport
  errors, including failed exchanges.
- Endpoint registration now supplies a closed WebSocket capability contract;
  runtime routing no longer switches on endpoint identifier strings.
- The in-repository Realtime target and product test use the same official
  event vocabulary and valid text/audio commit behavior.

## Test-first evidence

Observed RED failures before the corresponding source changes included:

- text-only Realtime lowering emitted an extra empty audio commit;
- official Realtime output events classified as ignored;
- a divergent terminal snapshot was accepted by byte count;
- incremental Realtime operation selection was absent;
- the direct executor reported that response streaming was unsupported;
- cancellation fired before the final measured input flushed;
- the WebSocket-feature build exposed missing writer-capacity and retry typing
  in the repair as it was assembled.

Focused GREEN evidence obtained from the live tree and then repeated from the
isolated repair commit:

```text
cargo test -p aiperf-runtime --features engine,websocket --lib \
  real_tcp_backpressure_allows_pong_before_remaining_max_sized_frames -- --nocapture

test transport::ws::driver::tests::real_tcp_backpressure_allows_pong_before_remaining_max_sized_frames ... ok
test result: ok. 1 passed; 0 failed
```

`RUSTC_WRAPPER` printed `/usr/bin/sccache` for the build and was not changed.

The exact repair source also passed these suites:

```text
runtime transport::ws                  29 passed
runtime engine::ws_execution           10 passed
runtime realtime_ filter                9 passed
runtime websocket_ filter              17 passed
mock-server websocket unit filter      33 passed
mock-server websocket_wire             15 passed
product WebSocket end-to-end            14 passed
aiperf-cli build                        passed
```

The registry inventory tests, metric-catalog invariant, and metric-ID snapshot
also pass with the WebSocket transport and two round-trip metrics present.

## Verification boundary

The shared live worktree contained concurrent, unrelated native-eval edits.
Verification therefore used an isolated worktree at the exact repair commit.
The full runtime library run completed with 2,033 passed, seven failed, and
seven ignored tests. Six failures were outside this change: two missing
recorded-agent fixtures and four native-eval timing/artifact/Compose
expectations. The remaining rate characterization missed its threshold only
under full-suite load and passed alone at 199.9 requests/second against a 200
requests/second target. No WebSocket test failed. No fresh Graham verdict is
claimed by this report.
