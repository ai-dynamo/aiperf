# Runtime WebSocket official-event repair report

## Basis

This repair starts from exact commit
`620dfec6f0dca72abcf3b13fcc8c89c78dd017e0` and addresses all five findings in
`runtime-driver-fourth-final-graham.md`. Final verification used the detached
worktree `/tmp/aiperf-ws-fifth-verify-20260818`, created from that exact commit
with only the three-file staged repair applied. The detached and shared staged
patches had the same SHA-256 digest,
`ff292def96fcc20cda1d24da201b8a56eaa0a4e0f0650ba94ed2b97d6e47f326`, before
this report was added.

Every Rust command preserved the configured
`RUSTC_WRAPPER=/usr/bin/sccache`.

## Repairs

- Responses output-text and reasoning deltas accept the official streaming
  event shape without a `response_id`. A correlated `response.created` binds
  the response first; identity-bearing terminal and usage events must still
  match that binding.
- Realtime errors correlate only through the client reference at
  `error.event_id`. The top-level server event identity is never treated as a
  client reference.
- A markerless error on a reused socket is always unsafe, including after the
  current response was created. Replay-capable Responses operations retry on a
  fresh socket before observer-visible facts; other operations fail safely.
- Responses errors deserialize their official top-level `code` and `message`.
  The typed diagnostic retains the server message and includes a non-empty
  code when present.
- Valid authored response metadata is preserved verbatim. Up to 15 free pairs
  permit an internal correlation marker. A full 16-pair map, or an authored
  pair using the marker's key, is not changed or rejected; that operation is
  classified as unsafe for socket reuse before checkout and runs on a fresh
  socket. More than 16 authored pairs remains a typed protocol error because
  it exceeds the public limit. A Realtime continuation that requires its
  affinity socket but cannot carry safe success correlation fails typed before
  an ambiguous send.
- The mock Responses route now emits the official output-delta fields:
  `item_id`, `output_index`, `content_index`, `delta`, and `sequence_number`.
  It no longer invents `response_id`, so mock coverage cannot mask the client
  bug.

## TDD evidence

The initial focused runtime command ran 24 dialect tests: the 19 existing
tests passed and the five new regressions failed for the expected reasons:

- the conforming Responses delta was rejected for missing `response_id`;
- the nested Realtime client event reference was ignored;
- a markerless post-created error on a reused socket was attributed;
- the official top-level Responses error produced the generic fallback;
- a valid 16-pair metadata map was expanded to 17 pairs.

The mock regression failed while the emitted delta still contained invented
`response_id`, then passed after the emitter adopted the official shape. A
separate diagnostic assertion failed while the Responses error code was
dropped, then passed after the typed top-level envelope retained it.

The additional valid-authored-key regression ran RED in the clean verifier:
it failed because the authored `_aiperf_ws_operation` pair was rejected as
reserved. It then passed after the implementation preserved the pair and
selected the non-reuse path. The driver-level full-capacity regression proves
that no application bytes are sent on an available generic cached socket and
that the unchanged 16-pair request completes on a fresh socket.

## Verification

All final commands ran in the clean detached verifier:

- Runtime complete `transport::ws::` suite: 40 passed.
- Runtime WebSocket execution suite: 19 passed.
- Mock-server WebSocket suite: 26 passed.
- Product WebSocket target with the explicitly built isolated `aiperf` binary:
  14 passed, including the Responses and Realtime profile cases and their raw
  record/RTT assertions.
- Scoped `rustfmt`: pass.
- Scoped staged `git diff --check`: pass.

The commands emitted only warnings from evaluation and metrics code outside
the scoped files.

## Boundary

Only the WebSocket dialect, WebSocket execution path, mock WebSocket scenario,
their focused tests, and this report belong to the repair. Unrelated dirty
evaluation, HTTP, mock tokenizer, design-record, progress, and generated files
in the shared worktree remain unstaged and excluded.

This report records implementation and verification evidence. It does not
claim an independent review verdict.
