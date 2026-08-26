# S04 HTTP-Owned Shared Transport Configuration Implementation Plan

**Goal:** Remove HTTP-owned configuration from neutral execution inputs while preserving resolved HTTP and gRPC behavior.

**Spec:** `docs/specs/2026-08-25-rust-smell-core-remediation.md` (S04, lines 26–33).

## Ruling

The referenced Sol plan was unavailable at execution start. This plan is the
minimal implementation record derived directly from the binding S04 spec.

The follow-up ruling fixes the neutral DTO contents to timeout, TLS
verification, connect retries, connection reuse, session header, and explicit
raw-capture. HTTP protocol, proxy, UDS, body bounds, and remaining client
details remain HTTP-local. gRPC and WebSocket receive only the neutral policy.

## Steps

1. Add a focused regression proving each transport binding receives the same
   resolved shared run inputs and explicit raw-capture flag without requiring
   an HTTP sink configuration in `ExecutionBackendConfig`.
2. Run that regression against the current implementation and record the Red
   compile/behavior failure caused by `ExecutionBackendConfig::transport`.
3. Move transport-specific sink configuration into the HTTP, gRPC, and
   WebSocket factories/bindings; leave `ExecutionBackendConfig` with shared
   run inputs only. Preserve the existing Config-v2 resolved values and wire
   defaults.
4. Re-run the focused regression, the transport execution tests, and rustfmt;
   record terminal Green evidence in the remediation receipt.
