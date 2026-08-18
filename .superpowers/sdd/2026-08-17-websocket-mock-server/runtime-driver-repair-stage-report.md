# Runtime WebSocket driver repair stage report

## Scope

- Realtime operations commit materialized input before requesting a response.
- Responses terminal snapshots contribute only content not already emitted by deltas.
- `PreparedTurn` retains the runtime session identity used by sticky worker placement and worker-local socket affinity.
- Continuation identity is owned by the cached socket and is injected only with post-assistant incremental input; ambiguous full-history continuation fails closed.
- Affinity serialization spans checkout, fallback, terminal handling, socket retention, and close.
- Application messages are bounded and fragmented; the writer services control frames while a fragment flush is backpressured.
- Raw capture is opt-in and serializes the complete ordered WebSocket operation plus received application messages only when enabled.
- Realtime text and audio events pass through the prepared endpoint parser to live response observers; the sink reports response-streaming support.
- Idle timeout and authored cancellation arm after measured input flush. Idle socket age rotation no longer aborts an active operation.
- The unsupported Responses terminal acknowledgement was removed.

## TDD evidence

RED, isolated detached source and target:

```text
realtime_websocket_lowering_commits_items_before_response_create
assertion failed: left 2, right 3
```

GREEN, outside the sandbox with `RUSTC_WRAPPER=/usr/bin/sccache` unchanged:

```text
cargo check -p aiperf-runtime --features 'engine websocket'
Finished dev profile

cargo test -p aiperf-runtime --features 'engine websocket' --lib transport::ws::
23 passed; 0 failed

cargo test -p aiperf-runtime --features 'engine websocket' --lib realtime_websocket_lowering_commits_items_before_response_create
1 passed; 0 failed

cargo test -p aiperf-runtime --features 'engine websocket' --lib raw_operation_capture_preserves_every_message_and_role
1 passed; 0 failed
```

Additional focused runs passed the Realtime audio decoder and sticky worker-selection tests.

The product test target compiled. Its first invocation lacked `AIPERF_E2E_BIN`; a second invocation used a pre-existing debug binary and therefore exercised stale runtime code. Rebuilding that binary was interrupted after the agreed no-output cap, so no product-test pass is claimed in this stage.

## Review handoff

This report records implementation and focused evidence only. A fresh strict runtime review remains the parent stage gate.
