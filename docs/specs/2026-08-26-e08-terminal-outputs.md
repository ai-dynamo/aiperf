# E08 NativeGraph Terminal Outputs Specification

## Problem

`lower_native_graph` accepts a `terminal_outputs` list only when every name is
both a declared channel and produced by the lowered graph. The live driver
retains that validated list and can turn a `TraceStageResult.output_handles`
map into `TraceTerminalSupplement.terminal_outputs`. The execution path makes
that contract unreachable in two places:

1. `NativeGraphLiveTraceProgramDriver::open` rejects every non-empty declared
   list with `"requires frozen terminal handles before stage execution"`.
2. `GraphWorkerBackend::execute_staged_driver` serializes the completed channel
   snapshot for driver control, but always supplies an empty `output_handles`
   map to `observe_stage`.

Consequently a valid authored terminal declaration fails before graph dispatch.
Removing only the open-time refusal would instead fail after the stage because
the driver correctly treats a missing declared handle as an observation error.

## Scope and non-goals

This change makes validated, static NativeGraph terminal-output declarations
execute. It preserves lowering validation, the existing opaque `Handle` wire
contract, stage bounds, cancellation, graph scheduling, and dynamic-control
receipts.

It does not add arbitrary output selection, expose terminal bytes in artifacts,
change the public terminal-supplement JSON schema, or compose multiple stages'
channels into a new user-visible data API. Dynamic NativeGraph terminal-output
materialization remains covered by the same declared-channel rule; if a dynamic
stage finishes without a value for a declared output, that is the same typed
missing-output observation failure as a static stage.

## Required behavior

1. A lowered NativeGraph with a non-empty, already validated
   `terminal_outputs` list opens and executes normally.
2. Once an executed stage is complete, the engine freezes every concrete value
   currently present for a driver-declared terminal channel as canonical JSON bytes in a
   trace-execution-owned content-addressed segment store. The resulting
   `Handle`s are passed in `TraceStageResult.output_handles`.
3. The output store is a terminal/cold-path seam: it is not shared mutable
   request state, is never used to alter prompt materialization, and does not
   add locking or allocation to per-token work. It is initialized from the
   worker's immutable segment catalog so its handles resolve against the same
   content-addressed representation for the remainder of that trace result.
4. The live driver selects only the validated declared names. Its completed
   supplement contains one opaque handle for each declared name, sorted by the
   existing `BTreeMap` representation; it must contain no undeclared channels
   and no raw `serde_json::Value` terminal payload.
5. A declared channel missing from a static completion, or still missing when a
   dynamic progression becomes terminal, fails `observe_stage` with the existing
   typed `TraceDriverError` diagnostic naming that declared channel. Intermediate
   dynamic stages may omit a declaration produced by a later stage. A
   non-declared channel never causes output freezing or appears in the
   supplement.
6. Empty `terminal_outputs` retains the existing legacy behavior: no terminal
   segment is added and the supplement omits `terminal_outputs` on the wire.

## Design

Introduce a private execution-owned terminal-output freezer alongside the
staged execution loop in `engine/graph_execution.rs`. It accepts the selected
terminal channel names, the executor's completed `BTreeMap<String, ChanVal>`,
and the trace-local current frozen store (initially the worker catalog). It
thaws the catalog into a `SegmentPool`, canonical-serializes each selected
concrete `ChanVal` once, interns the bytes as `Payload::Raw`, and freezes the
resulting pool before the driver is observed. Absent and `Unset` selections are
omitted so the driver remains the typed missing-output authority. Each dynamic
stage extends the prior trace-local frozen store, preserving every issued
handle. The returned map is keyed only by selected names; a private staged
terminal result owns the final frozen store through terminal-supplement
emission, so no worker map, lock, or transient untracked handle is introduced.

`execute_staged_driver` obtains the selected names from the driver contract
through a narrow driver-facing output-selection method, invokes the freezer
after successful static execution, and supplies the resulting map to
`TraceStageResult`. The method is intentionally declaration-oriented rather
than exposing the live driver's internal vector to the engine. The static
driver selects its required handles immediately. The dynamic driver retains
the latest handle for each declaration across observations and applies the same
required-name check when progression becomes terminal.

`NativeGraphLiveTraceProgramDriver::open` removes only its contradictory
non-empty-list refusal. Its existing `observe_stage` map construction remains
the policy boundary for exact selection and missing-output errors.

## Acceptance tests

- A driver-level regression lowers a graph declaring `output`, opens it,
  observes a completed stage with a resolvable `output` handle plus an
  undeclared extra handle, and proves the completion supplement contains only
  `output`.
- The same fixture omits `output` from the result map and proves the typed
  declared-channel diagnostic; it does not produce a completion supplement.
- An engine staged-execution regression uses a terminal declaration and proves
  the engine freezes the completed channel and supplies its handle to the live
  driver, rather than failing at `open` or passing an empty map. It resolves
  that handle through the owned frozen segment store and verifies the canonical
  raw JSON bytes equal the channel's completed value.
- The empty-declaration regression continues to prove the legacy supplement
  wire omits `terminal_outputs`.

## Verification gates

Run the new focused runtime tests and the relevant NativeGraph driver and graph
execution suites with a unique `CARGO_TARGET_DIR` under `/mnt/4tb`. Capture a
real pre-fix RED witness, then GREEN after the minimum implementation. Run
formatting/diff checks, obtain an independent full Graham review with no open
blocker, commit the implementation, and only then update the campaign tracker.
