# Rust Active Matrix Remediation Specification

## Scope

This specification covers C01-C11, C14a-C14b, C18-C20, and C24-C25 from the
active Rust smell-remediation matrix. All tests are RED-to-GREEN and use
`CARGO_TARGET_DIR` under `/mnt/4tb`.

## Benchmark trust dispositions

C01's raw cellular push authentication/replay gap is accepted under the local
trusted controller/cell model. Add a concise adjacent rationale that states the
trust assumption and the absence of per-push authenticity guarantees; preserve
legitimate replay/live delivery tests. C02 shares that trust assumption only
for its raw-push authenticity portion. Its decompression resource bound remains
required.

## C02-C03: Bound input and receipt identity

C02 shall stream-decompress with a protocol-derived maximum output size and
fail before exceeding it; normal maximum-size chunks remain valid. C03 shall
record terminal cellular receipts by `cell_id`, reject out-of-range/duplicate
IDs and kind changes, and complete only after every expected cell contributes
once. Tests cover expansion beyond the cap, normal chunks, duplicate same-kind,
and mixed-kind messages.

## C04-C07: Make transport boundaries explicit

C04 shall validate eventstream prelude fields as soon as the full prelude is
available, impose a bounded maximum frame/buffer size before retaining body
bytes, and fail invalid/minimum/oversized lengths without retaining payload.
C05 shall return one terminal decode error for trailing bytes at source EOF and
succeed at a clean frame boundary. C06 shall author, strictly validate, and
project a positive HTTP response-body cap from endpoint profile to client
config, with declared-length and chunked enforcement tests. C07 shall retain
authored H2 prior knowledge for UDS and pass it to UDS handshake, preserving
Auto/H1 behavior; h2c-over-UDS and H1 tests prove both paths.

## C08-C11: Preserve graph execution semantics and accounting

C08 schedules first-token anchored successors once at first-token observation,
while completion anchors remain terminal-triggered and fan-in/dedup remain
correct. C09 races first-token/delay waits with abort and rechecks before
dispatch. C10 applies equivalent abort gating to ready tool successors. C11
uses an RAII in-flight guard so normal return, cancellation, and panic all
settle idle accounting. Each change gets a focused behavioral test.

## C14a-C14b: Enforce YAML contract

C14a shall accept `artifacts.records: false` and format lists, rejecting `true`.
Reapply the divergent branch’s narrowly scoped test-first fix onto current
HEAD, then execute fresh RED/GREEN verification. C14b accepts omitted or
`"2.0"` schemaVersion and rejects all others with a diagnostic naming the
supported value. Neither changes the Config-v2 default.

## C18-C20: Preserve exporter results and write errors

C18 routes MLflow and W&B through the shared summary-series selection rule:
export a sole series or unique unlabeled aggregate; consistently and
diagnosably skip multiple series lacking one or having ambiguous aggregates.
C19 maintains the deterministic sorted union of histogram bucket boundaries
and backfills stored rows when new boundaries appear. C20 reapplies the
divergent explicit-`flush()` fix and its writer-failure RED test, proving normal
JSON output remains valid and flush error propagates.

## C24-C25: Close scheduling races

C24 makes decrease publication, release, and acquisition observe coherent
capacity/debt state without hot-path lock contention; a deterministic paused
decrease/release/acquire test proves no over-admission. C25 supplies a
cross-thread release notification for the actual blocking prefill/global pool
and prevents missed wakeups; a released remote prefill slot wakes the cached
issuer before terminal/session release.

## Non-goals

Do not add a per-push authentication system, unbounded channels, shared
hot-path locks, Config-v2 behavior changes beyond the named validation, or
unrelated refactors.
