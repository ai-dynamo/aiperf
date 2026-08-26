# Rust Core Smell Remediation Specification

## Scope

This specification covers S01, S03-S05, S07-S08, S10-S15 from
`docs/rust-code-smell-remediation-tracker.md`. It preserves public benchmark
configuration, wire formats, defaults, and measurement behavior unless a
requirement below explicitly changes a defect.

## S01: Make required TCP socket setup fail the connection

`apply_socket_opts` failures for `TCP_NODELAY` or keepalive shall become
contextual `ErrorKind::Connect` errors before TLS or Hyper handshake begins.
Best-effort reuse-address and buffer tuning remains non-fatal. Tests inject a
required-option failure and prove connect classification plus no handshake, and
inject optional-tuning failure and prove connection setup continues.

## S03: Restore observer polymorphism at `WorkerSink`

`WorkerSink` shall dispatch through `&dyn RequestObserver`; native record
registration/finalization stays in the engine wrapper. A non-native test
observer must receive normal callbacks and return the sink result. Existing
native paths must still produce one complete record for successful and failed
dispatch. No pluggable metrics finalizer or synchronization is added.

## S04: Remove HTTP types from neutral execution configuration

`ExecutionBackendConfig` and `NativeTransportExecution` shall expose only
shared run inputs. HTTP and gRPC bind their own sink configuration, while raw
capture is an explicit run input instead of a conflicting default. Tests prove
one resolved endpoint profile retains timeout, TLS, retry, reuse, session
header, and raw-capture behavior in both transports; dry-run needs no HTTP
configuration. Config-v2 and wire defaults must not change.

## S05: Avoid artifact-only HTTP hot-path work

Prepared HTTP request state shall retain a parsed `Url` and typed wire headers.
Per-dispatch URL stringification/reparse and static-header string round trips
are removed. Artifact header maps are materialized only when raw capture is
enabled; capture-off records retain no request-header copy while the server
receives identical dynamic and endpoint headers. Dynamic paths and query
parameters remain supported; no global URL cache is introduced.

## S07: Bound graph and media event queues

Both queues shall have explicit bounded capacity. Graph saturation returns a
typed terminal `TraceError`, never drops/reorders accepted events. Media
submission uses nonblocking enqueue with an overflow latch; finalization fails
instead of reporting a successful partial artifact. Tests cover at-capacity
success, capacity-plus-one failure, ordered draining, media overflow failure,
and normal complete media drain. Request threads must never block on queue
capacity.

## S08: Require the cellular log artifact

`test_graph_cellular`'s log helper shall fail with a path-bearing diagnostic if
the log is absent or unreadable. It shall return an empty parsed-event list only
for an existing readable log with no matching event. The focused cellular test
must verify artifact absence cannot satisfy the negative assertion.

## S10: Remove unreachable disaggregated collector surface

Production-only `TraceCollector` methods with no production producer shall be
removed rather than hidden behind `cfg(test)`. Backing fields are retained only
where report compatibility requires their default/absent serialization, which
must be documented. Live observer tests cover supported fields. The change must
not invent disaggregated behavior or alter artifact fields without an explicit
compatibility decision.

## S11: Decompose `Inputs` without changing its wire form

The flat internal `Inputs` DTO shall be decomposed into cohesive endpoint,
telemetry, dataset, workload/phase, runtime, artifact, and replay/scenario
groups. The existing flattened JSON execute wire remains byte-compatible via
`serde(flatten)` or an explicit compatibility DTO. Group-local validation moves
next to each group; cross-group validation remains top-level. Tests prove CLI
and YAML authoring wire equivalence, unknown/missing-field behavior, existing
authoring-wire equivalence, and representative rejection messages.

## S12: Delete redundant internal cellular feature gates

Remove internal `cfg(feature = "cellular")` gates from
`cellular_controller.rs`, retaining unrelated platform/test gates. The existing
module-level feature gate stays authoritative. Feature-off compilation excludes
the module; feature-on tests are unchanged.

## S13: Isolate raw graph transport benchmark code

Move the executable raw-HTTP transport benchmark out of the unconditional graph
library surface into a benchmark/example or explicit non-default feature.
`GraphRpsReport` moves to a neutral report location consumed by dynosim.
Default-library and dynosim builds remain valid. Do not force a microbenchmark
through the product sink, because that changes what it measures.

## S14: Test actual DAG validation rejection

Rename the valid-fixture test to acceptance. Add concise negative fixtures for
lineage mismatch and an unattached/unknown branch, asserting the relevant
`DatasetError::Validation` diagnostic. These tests must fail if `validate_dag`
is bypassed. Validation policy itself is unchanged.

## S15: Give phase-policy booleans semantic names

Encapsulate the independent `ScheduledPhasePlan` policy switches in a documented
phase-policy structure, storing each boolean under an `is_`, `has_`, or
`needs_` name. Preserve fluent builders, defaults, and runtime effects. Tests
cover defaults, builder transitions, stop enforcement, metric retention, credit
dispatch, and local-measurement discard.

## Verification

Each implementation task uses RED-to-GREEN tests first, runs its focused crate
suite with `CARGO_TARGET_DIR` below `/mnt/4tb`, then receives a Graham review.
