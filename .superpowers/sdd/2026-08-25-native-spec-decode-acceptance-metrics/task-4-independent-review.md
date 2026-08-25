# Task 4 independent review — first pass

## Range

- Base: `75a26a56516fd7c0f20749c5241004b17e36eb27`
- Head: `f657dd91cbb7ce8298f5aaa8b76f7c5045c6792e`
- Verdict: **NOT APPROVED**

## Important finding

The eleven new identities existed in the embedded resource's
`console_metrics`, but not in its parallel `header_map`; the five
derived/aggregate identities were also absent from `scalar_tags`.
`Export::build` copies all three sections into the real product configuration,
and the GenAI-Perf exporter falls back to raw tags when a header is absent.

The reviewer reproduced this with the rebuilt native CLI against the opt-in
Rust mock. Public CSV rows used names such as
`spec_decode_acceptance_length (ratio)`,
`spec_decode_overall_draft_acceptance_rate (%)`, and
`total_spec_decode_steps`, rather than the canonical upstream headers.

Required fix: add all eleven canonical v1 headers and the five scalar
classifications, extend the default-profile regression across all three
metadata sections, and assert representative real CSV names in the E2E.

## Minor findings

- Build the absent E2E harness from `MockServerConfig::default()` without
  assigning the flag false so the false-default contract is covered.
- Lock the distinct zero-output/tool-call streaming branch. The reviewer
  manually observed it behave correctly: two unfinished tool deltas, then one
  finish-only `tool_calls` stats frame, then usage-only.

## Evidence reported by reviewer

- Mock fixture: 2/2.
- Embedded console metadata: 1/1.
- Real-profile E2E: 14/14.
- `git diff --check`: passed.

The review found no Critical issues. Its one Important public artifact defect
blocks approval; both Minor test gaps are included in the scoped fix round.
