# S04 HTTP configuration remediation receipt

## Result

`ExecutionTransportPolicy` now owns the cross-transport execution fields:
timeout, TLS verification, connection retry count and backoff, connection
reuse, session header, and raw capture. HTTP applies those values to its local
`TransportSinkConfig` while retaining the resolved HTTP `ClientConfig` through
`ConfiguredHttpExecutionFactory`. gRPC maps the same neutral policy into its
own local config; WebSocket and dry-run accept the neutral graph-dispatch
inputs without receiving HTTP configuration types.

## Test evidence

Red (temporary mutation `capture_raw: false` in the gRPC binding):

```text
cargo test -p aiperf-runtime --features engine --lib \
  engine::grpc_turn_execution::tests::grpc_binds_all_neutral_execution_policy_fields -- --exact
FAILED: assertion failed: config.capture_raw
```

Green after restoring the policy binding:

```text
engine::grpc_turn_execution::tests::grpc_binds_all_neutral_execution_policy_fields ... ok
test result: ok. 1 passed; 0 failed

engine::turn_execution::raw_retention_tests::execution_transport_policy_carries_raw_capture_explicitly ... ok
test result: ok. 1 passed; 0 failed
```

`cargo fmt --all -- --check` reports pre-existing formatting differences in
unrelated `cli/src/yaml.rs`, graph, and phase-runtime files. The S04-modified
runtime files were formatted individually and `git diff --check` passes.
