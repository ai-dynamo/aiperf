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

No mock-server behavior is required: this is an outbound-client header and the
existing mock's captured raw request records are the product observation seam.

## Required native port

1. Add a default-on `X-Session-Affinity` argument to the pure header composer
   and send it after caller headers whenever a correlation ID exists.
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
