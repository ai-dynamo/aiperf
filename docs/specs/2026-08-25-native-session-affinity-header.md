# Native default session-affinity header

## Purpose

Define the native HTTP transport contract for the default additive
`X-Session-Affinity` routing header introduced by origin/main commit #55.

## Built

Every HTTP request with `RequestConfig.correlation_id` emits the existing
correlation header and a canonical `X-Session-Affinity` header whose value is
that correlation ID. The rule is default-on, carries no environment toggle, and
applies to direct transport plus native endpoint preparation for streaming,
ordinary, and bounded streaming transport paths without changing their
endpoint/URL/body behavior.

Derived affinity headers are applied after endpoint and authored request
headers. Before insertion the shared policy removes every existing
`X-Session-Affinity` spelling case-insensitively, preventing a stale or
conflicting value from reaching the wire.  A request without a correlation ID
does not invent an affinity identity.

`--session-header` continues to rename only the correlation header.  It does
not suppress the additive affinity header.  The legacy
`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` environment setting remains an
independent opt-in and controls only `X-Session-ID`; it is not a compatibility
alias for the default affinity header.  SGLang and Dynamo derived headers are
unchanged.

The native binary E2E uses the Rust mock server's raw-record capture to prove
the actual outbound request headers, including default, opt-in, and custom
session-header cases.  The mock itself needs no protocol change because it
already records incoming request headers verbatim.

## Source anchors

- `rust/runtime/src/transport/http/transport/headers.rs`
- `rust/runtime/src/transport/http/transport/endpoint_binding.rs`
- `rust/runtime/src/transport/http/transport/http_transport.rs`
- `rust/e2e-tests/tests/test_port_raw_parity.rs`
- `docs/benchmark-modes/trace-replay.md`
- `docs/environment-variables.md`
