# Commit 055 — `dd3f09b0c3`

## Upstream intent

Upstream makes `X-Session-Affinity` a default, additive HTTP header.  When a
request has a stable `X-Correlation-ID` value, the transport sends that same
value in `X-Session-Affinity`.  It does not change the separate
`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` opt-in: `X-Session-ID` remains
absent unless explicitly requested.  The derived affinity value wins over a
caller-supplied case variant, exactly as the existing derived session headers
do.

The upstream Python change has five files: the environment default and table,
the base transport, one transport unit assertion, and trace-replay guidance.

## Native comparison

The native HTTP transport already has the correct ownership seam:
`transport/http/transport/headers.rs` composes the correlation header and
derived additive affinity headers after caller headers, while
`HttpTransport` freezes environment policy once at construction.  It currently
has only three opt-in derived headers: `X-Session-ID`, `X-SMG-Routing-Key`, and
the Dynamo session pair.  Thus a default request has no `X-Session-Affinity`;
the Rust raw-parity control for #26 proves that absence and must change.

Native profiles do not send through `HttpTransport::build_headers`: their
worker-local endpoint path materializes caller headers and lowers
`HttpEndpointRequest` through `prepare_request`. The default policy therefore
must be shared at that preparation boundary too; changing only the facade would
leave profile traffic with a caller-provided stale affinity value on the wire.

No mock-server behavior is required: this is an outbound-client header and the
existing mock's captured raw request records are the product observation seam.

## Required native port

1. Add a default-on `X-Session-Affinity` policy to the pure header composer and
   native endpoint preparation, sending it after caller headers whenever a
   correlation ID exists.
2. Strip caller-provided affinity-header variants case-insensitively before
   inserting the canonical name/value.  Preserve custom `--session-header`
   behavior: it may rename the correlation header, but does not suppress the
   additive affinity header.
3. Keep `X-Session-ID` strictly opt-in and leave the Dynamo and SGLang toggles
   unchanged.
4. Port the upstream default assertion, add no-correlation and stale-casing
   edge cases, and exercise the native binary against the mock with raw records
   for default, opt-in-session-ID, and custom-session-header behavior.
5. Update trace-replay and environment documentation to state the default
   affinity contract.

## Ancestry constraint

The closure provenance merge must have the reviewed native tree as first
parent and exact upstream `dd3f09b0c34710470444bad17c9e7050c1cd694a` as second
parent.  An `ours` tree merge is required because the upstream Python commit
is already reachable in the campaign history; never cherry-pick it or import
its Python files.

## Test translation map

| Upstream behavior | Native evidence |
| --- | --- |
| Default correlation and affinity headers | Pure composer test plus native-binary raw capture. |
| `X-Session-ID` stays absent by default | Pure composer test plus native-binary raw capture. |
| Explicit session-ID remains additive | Existing opt-in composer behavior plus native-binary raw capture. |
| Case-insensitive caller override | Composer test with a stale lower-case affinity header. |
| Renamed correlation header | Composer and native-binary raw capture retain the additive affinity header. |
| Trace replay routing statement | Updated public documentation. |

The source-grounded native contract is in
[`../specs/2026-08-25-native-session-affinity-header.md`](../specs/2026-08-25-native-session-affinity-header.md).

## Closure evidence

The target-only provenance merge is
`1a11e00297e105eb4ad64d0bf46606d6c0eebd0d`: its first parent is the reviewed
native tree and its second parent is exact upstream
`dd3f09b0c34710470444bad17c9e7050c1cd694a`. Its tree equals its first-parent
tree; no upstream Python files were imported or cherry-picked.

Native implementation landed in `e6d03a92f170914872301863b914f2f8299745cd`.
The independent review found and the follow-up
`821412a095c91773b428e81b1e90ea19a01d8ff5` fixed a no-correlation corner:
the normalizer now strips every case-insensitive authored
`X-Session-Affinity` variant when no correlation identity exists. Direct,
prepared/profile, TransportSink, and raw native-request coverage prove both
lower-case and canonical stale forms are absent; with a correlation ID the
same coverage proves a single canonical derived header replaces them.

Source-fresh verification used sccache, an isolated target directory, and the
native binary. The 21-case raw suite passed, including an exact-dd3 Python
oracle with distinct artifact roots, recorded-request-count assertions, and
cleanup verification. It covers default affinity, a conflicting authored
header, no correlation, custom `--session-header`, and independent explicit
`X-Session-ID` opt-in (#26 control). The focused runtime composer/prepared
tests and raw no-correlation regression passed; formatting, documentation, and
diff checks passed.

`cargo clippy -p aiperf-runtime --lib -- -D warnings` remains blocked only by
pre-existing unused imports outside this port (`metrics.rs`, `endpoints/mod.rs`,
and `eval/native_graph*`); no diagnostic names a #55 path. This is inherited
workspace debt, not a port failure.

Root's independent Graham review of
`1a11e00297e105eb4ad64d0bf46606d6c0eebd0d..821412a095c91773b428e81b1e90ea19a01d8ff5`
approved the final normalizer placement and found no blocking, important, or
style findings.

Disposition: **complete**.

GRAHAM APPROVED
