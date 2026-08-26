# Commit 054 — `bfe33151de`

## Upstream intent

When filter-then-cap rejects every candidate for `--max-context-length`, report
the smallest observed peak context and the concrete limit that admits it. The
diagnostic must describe the completed selection pass, not invent a candidate
or alter selection order.

## Native analysis

The target is not Baseten. Native selection seams are AgentX/HuggingFace WEKA,
Graph-IR WEKA (including TraceLab, which converts then delegates to WEKA), and
Graph-IR Dynamo. Their selection order, parse boundary, and peak calculations
remain unchanged.

## Required native behavior

The all-rejected context-cap result carries the source, scanned count, authored
limits, and the smallest observed peak with a concrete replacement cap. Empty,
malformed, and non-context-caused failures preserve their established errors.

## Closure evidence

- Target-only merge: `352ca1b032e44059f7923ac73eba364ea571863d` has first
  parent `2753be631fc41f3af743bdafe615ba3407ce83c3`, exact second parent
  `bfe33151de75426710e51ca054823aa91342cebc`, and an unchanged first-parent
  tree.
- Native implementation and tests: `4022b433c9`.
- Focused runtime verification (sccache, `/mnt/4tb/aiperf-target-port054`,
  clang/lld, preserved `--cfg tokio_unstable`): AgentX 106 passed / 1 ignored;
  recorded graph 78 passed / 1 ignored, plus its 1-test acquisition harness;
  TraceLab rejection control 1 passed.
- Native `--execute` E2E: 1 passed. It uses `http://127.0.0.1:1`, preserves
  the domain-specific terminal diagnostic and 44-token observed minimum, and
  proves selection fails before connection I/O.
- `cargo fmt --all -- --check`, runtime-lib Clippy, and the changed CLI E2E
  target's Clippy exit cleanly.
- Broad runtime library suite: 1,812 passed, 7 ignored, 1 inherited failure:
  `metrics_core::report::tests::v2_uses_type_specific_series_and_null_for_non_finite_tail`
  in `runtime/src/metrics_core/report.rs:1486` expects version `0.0.0` but the
  workspace reports `0.12.0`. This port does not change metrics, reporting, or
  version code.
- Independent Graham review of `4022b433c9`: approved, no blocking,
  important, or style findings.

## Ancestry constraint

The final target-only merge records exact upstream
`bfe33151de75426710e51ca054823aa91342cebc` as its second parent. Its first
parent is the reviewed native Rust tree; upstream Python files are not imported
and no cherry-pick is used.
