# Origin commit 59cd43e43a: typed Responses mock request model

## Finding and audit

The upstream commit replaces the Python mock server's untyped `/v1/responses`
payload with a `ResponsesRequest` model, centralizes input flattening, and
threads the type through token accounting and request recording. It captures
`max_output_tokens`, `min_tokens`, `ignore_eos`, and `reasoning_effort` in the
recording fingerprint, and adds unit coverage for model parsing, token
extraction, API compliance, and recorder output. The changed files contain no
upstream integration or E2E test files; the response behavior is exercised by
the mock-server unit suites.

The native Rust implementation already provides the equivalent typed
`ResponsesRequest`, `/v1/responses` route, recursive prompt flattening,
shared token/latency dispatch, response usage, and raw request recording. The
Rust mock-server integration tests cover non-streaming and streaming response
shapes. The native `test_new_routes` E2E covers streamed output deltas,
terminal usage, and raw-record retention. No production Rust change or new
test is required for direct parity.

## Closure and review

The exact non-fast-forward merge is retained. Focused native verification is
Green: the two Responses mock-server route tests passed and the native E2E raw
recording test passed. Python upstream test collection was attempted but the
shared environment lacks `zmq.asyncio`; this is an environment dependency
failure, not a source failure. Graham review found no native findings: the
merge adds no Rust code, hot-path allocation, async, synchronization, or error
handling changes.
