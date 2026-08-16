# Harbor semantic execution slice

## Delivered

- `lower_semantic_graph` now produces an ordered `LoweredSemanticGraph` of
  distinct executable-node types rather than a nominal `Exact` marker.
- Lowering preserves source order for LLM and tool nodes and returns an indexed
  `UnsupportedNode` refusal before returning any partial executable program.
- `PairedMeasurements` validates finite quality/cost/latency/critical-path
  values and non-negative system-resource values. `PairedComparisonReport`
  exposes independently attributable quality, cost, latency, critical-path,
  token, and tool-call candidate-minus-baseline deltas only after all fixed
  baseline dimensions match.

## Test evidence

1. Red: `RUSTC_WRAPPER= cargo test -p aiperf-runtime --test harbor_semantic`
   failed as expected because the executable plan, typed report, measurement
   validation, and indexed refusal APIs did not exist.
2. Green: `RUSTC_WRAPPER= cargo fmt -p aiperf-runtime --check &&
   RUSTC_WRAPPER= cargo test -p aiperf-runtime --test harbor_semantic --
   --nocapture` passed all five focused tests.
3. Compatibility: `RUSTC_WRAPPER= cargo test -p aiperf-runtime --test
   eval_semantic --test harbor_semantic` passed all seven tests.

`cargo clippy -p aiperf-runtime --test harbor_semantic -- -D warnings` remains
blocked by 73 pre-existing diagnostics outside this slice (for example
`scheduled.rs`, `agentic_replay.rs`, and `agentx/*`); none are in the touched
semantic files.

## Remaining gaps

- This slice defines and validates the owned executable semantic contract; it
  is not yet projected into `GraphTraceProgram` or driven by the engine.
- Semantic source variants remain the current P0 vocabulary (`Llm`, `Tool`,
  and refused `Barrier`). P1/P2 source-specific operations require fixtures and
  a profile/capability lowerer before they can be admitted.

## Commit

`cfcb423816 feat(eval): execute Harbor semantic comparisons`
