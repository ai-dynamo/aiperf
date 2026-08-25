# Native Baseten Recorded-Outcome Fidelity

## Status

Design for origin/main tracker #40, exact upstream commit
`215be05b6a534fb19b84bf83f711db2d20f5bea1`.

## Problem

Native Baseten replay parses enough source data to construct requests and
derive replay timing. Its private `BasetenRow` retains recorded end-to-end
duration because closed-loop timing consumes it, but drops recorded TTFT and
cached-token reference values. Once dropped, a future fidelity consumer cannot
compare replay observations with the source outcomes.

The upstream fix makes recorded outcomes independent of projection choices.
The native loader does not yet have upstream #39's column projection mechanism,
so the equivalent boundary is the native parse → `RawRow` → compose pipeline.

## Design

Add `RecordedOutcome` to the dataset model with the Baseten source contract's
three optional values:

- `duration_e2e_ms: Option<f64>`
- `duration_ttft_ms: Option<f64>`
- `cached_tokens_reference: Option<u64>`

`Turn::recorded_outcome` is optional, serde-defaulted, and dispatch-neutral.
The Baseten loader parses the two missing values into `BasetenRow`, serializes
all three into its private `RawRow` value, reconstructs them in the composer,
and attaches `Some(RecordedOutcome)` only when at least one recorded value is
present. Missing outcome columns therefore remain absent and deserialize
compatibly.

Recorded outcomes do not enter `extra_body`, endpoint parameters, request
metadata, scheduling metadata, or token accounting. `duration_e2e_ms` keeps its
existing closed-loop use. Replay speedup changes timestamps and derived delays,
not the stored ground truth.

## Validation and errors

This port follows the native loader's existing optional-field policy. JSON and
Parquet numeric values are accepted only when serde exposes them through the
expected unsigned or floating-point accessor; a missing, null, or incompatible
optional value becomes absent. Required-field validation and existing timing
validation are unchanged.

## Tests

Unit coverage uses the real Parquet reader and the full registry/composer path:

1. exact values survive default replay;
2. exact values survive `omit_kv_hints` and never leak into the request body;
3. exact values survive closed-loop replay while E2E duration still derives the
   expected continuation delay;
4. null TTFT and cached-token values remain absent while recorded E2E remains
   intact.

A Rust integration test outside the module loads a real Parquet fixture through
the public built-in loader registry and inspects the frozen public `Dataset`.
This is stronger than the upstream commit's private projection test and is the
applicable native integration boundary. The upstream commit adds no Python
integration or E2E test to duplicate.

## Ancestry and scope

The final ancestry commit is a real two-parent merge whose second parent is
exactly `215be05b6a534fb19b84bf83f711db2d20f5bea1`. It uses the completed native
tree as the merge result so upstream #39's pending Python performance work is
not imported. No cherry-pick is permitted.

The mandated base contains a duplicate `capture_endpoint_policy` re-export that
prevents every runtime test from compiling. Removing only the duplicate line is
an explicit prerequisite repair, recorded separately from the Baseten semantic
diff.
