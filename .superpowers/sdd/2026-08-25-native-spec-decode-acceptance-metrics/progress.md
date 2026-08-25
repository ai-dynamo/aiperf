# SDD ledger — plan: .superpowers/sdd/2026-08-25-native-spec-decode-acceptance-metrics.md

## Integration base

- Exact non-fast-forward merge: `e93d959c62af971cf867ef54c98c608452ade195`
- Parents: `8b5194bcfc26475c5e06030d8701c82b66eb7b6a d32f4bb98edbeac1374ec816aee32d7e4517c5ae`
- Merge hook constraint: this isolated worktree has no `.venv`; the exact merge used `--no-verify`, and final validation must use the shared project environment.
- Build isolation: `RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013`.

## Preflight plan scan

| Tasks | Producer / consumer or internal check | Finding |
| --- | --- | --- |
| 1 | Tests require the DTO/callback/record field that Step 3 defines; the RED command filters the new test names. | Consistent. |
| 2 | Tests require eleven catalog tags, indexed phase selection, exact/sketch pool retention, and report projection that Step 3 defines. | Consistent. |
| 3 | Tests require the console/v1 JSON/JSONL projections and cap policy that Step 3 defines. | Consistent. |
| 4 | Tests require the opt-in mock config and response fixture that Step 3 defines, while exercising Tasks 1-3 through the real binary. | Consistent. |
| 1 → 2 | Task 1 produces `RecordIngest::spec_decode_acceptance`; Task 2 consumes it for columns and pools. | Type and ownership agree; append the positional serde field at the tail. |
| 1 → 3 | Task 1 produces the serializable canonical DTO; Task 3 clones it only at retained JSONL export. | Field names agree with the spec. |
| 1 → 4 | Task 4's finish-only stats chunk must enter Task 1's typed SSE fast path. | The fixture intentionally has no content/usage, so it catches the known bypass risk. |
| 2 → 3 | Task 2 produces grouped scalar report entries and the full pool; Task 3 consumes both without recomputation. | Report boundary agrees. |
| 2 → 4 | Task 4 asserts Task 2's exact formulas and per-request/request-weighted distinction. | Literal values are independently hand-derived. |
| 3 → 4 | Task 4 asserts console, flat v1 JSON, and processed JSONL shapes produced by Task 3. | Artifact names and absence contract agree. |

Ruling: concrete phase-index selection applies to exact row masks, sketch keys, and histogram pools together — upstream pools by phase instance and a histogram-only selector would make scalar totals disagree with the same export context — if wrong, this broadens Task 2's store/sketch diff beyond the smallest histogram-only port.

## Task state

- Task 1: completed, GREEN, and independently approved
- Task 2: completed, GREEN, and independently approved
- Task 3: completed, GREEN, and independently approved
- Task 4: implementation GREEN; independent review pending

Task 1: dispatch fallback — `spawn_agent` was denied because the root thread's other port agents occupied the team limit. Per the SDD skill's harness-denial rule, implementation proceeds inline; an independent task review remains required before Task 2.

Task 1 receipt: see `task-1-report.md`. Initial `cargo test -p aiperf-runtime spec_decode --lib` was RED on the intended absent DTO/callback/record/typed-choice surface. The controller's recovered authoritative GREEN command was `RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 cargo test -p aiperf-runtime --features engine --lib spec_decode -- --nocapture`: 6 passed, 0 failed, with four unchanged baseline warnings. Task 2 remains gated on independent review.

Task 1 first review: rejected. Important: enforce `draft_acceptance_rate` in `0.0..=1.0`. Minor: prove the finish-only typed SSE stats traverse the real endpoint-dispatch loop into a terminal record. Review-fix RED ran eight focused engine tests: seven passed and the range test failed on the intended missing invariant. The minimal range check and dispatch integration regression are implemented. The identical GREEN command passed 8/8 with the same four baseline warnings. Scoped re-review is pending; Task 2 remains gated.

Task 1 re-review: APPROVED at `579b2e7ce1`; see `task-1-rereview.md`. The reviewer reran the authoritative engine-enabled command: 8 passed, 0 failed, 2300 filtered, with four unchanged warnings. Both rejected findings are resolved and the remaining Task 1 scan found no blockers. Task 2 may begin.

Task 2 receipt: see `task-2-report.md`. The strict RED failed to compile on
the intended absent eleven tags, `MetricConsoleGroup::SpecDecode`, indexed
phase context, pooled histogram accessor, and native-report field. The final
focused GREEN command passed 12 tests with no failures. Catalog fingerprint,
derived coverage, and definition-ID snapshot sentinels also pass. The adjacent
metrics-core sweep has one unrelated pre-existing golden mismatch caused by
shared-head version drift (`0.12.0` actual versus `0.0.0` expected); no Task 2
code or fixture owns that version string. Task 3 remains gated on independent
Task 2 review.

Task 2 first review: NOT APPROVED at `a222b4ae7d`. The reviewer confirmed five
important failures: unchecked exact histogram overflow during ingest, append,
and context pooling; suppression of a defined accepted-per-verified value;
positional MessagePack incompatibility; missing exact-row canonical retention;
and inconsistent index-only public contexts. The strict behavioral RED passed
12 tests and failed the five intended regressions. The fix-round focused GREEN
passed 20/20 with 1742 filtered tests, and formatting passed. The adjacent
metrics-core sweep passed 130/131; its sole failure remains the unrelated
shared-head `0.12.0` versus `0.0.0` report-golden drift. See `task-2-report.md`
and `task-2-independent-review.md`. Task 3 remains gated on scoped re-review.

Task 2 re-review: APPROVED at
`d7b460325599057012ee67a1c7f1fdffb5d533b1`; see `task-2-rereview.md`. The
reviewer reran the authoritative focused suite (20 passed, 0 failed, 1742
filtered), formatting, and commit-range diff check. All five rejected findings
are resolved and the remaining fix scan found no blockers. An external probe
also confirmed exact `u64::MAX + 1` native-report JSON serialization while the
compatible cellular MessagePack boundary refuses non-narrowable counts. Task 3
may begin.

Task 3 receipt: see `task-3-report.md`. The strict engine-enabled RED ran 29
focused tests: 24 passed and five failed on the exact missing console, v1 JSON,
and processed JSONL projections. The minimal implementation then passed 29/29.
A self-review micro RED/GREEN added the authored-zero high-bucket cap contract;
the final authoritative suite passed 30/30 with 2300 filtered tests. The full
exporter neighborhood passed 120/120 and the processed-record neighborhood
passed 10/10. Task 4 remains gated on independent Task 3 review.

Task 3 review: APPROVED at
`89dbbb91acc76fa786f342e662af779b40a2add5`; see
`task-3-independent-review.md`. The reviewer reran the authoritative suite
(30/30), full exporter neighborhood (121/121), processed-record neighborhood
(10/10), formatting, and commit-range diff check. No findings remain, and Task
4 may begin.

Task 4 receipt: see `task-4-report.md`. The opt-in mock handler tests passed
2/2 and the default-disabled byte-serialization compatibility regression passed
1/1. The first real-profile run carried the exact summary and per-record values
but caught that the embedded default metric catalog did not group the new
metrics, so the console histogram was absent. A default-profile catalog
regression reproduced that defect (0/1, missing default console metadata); the
catalog-only fix passed 1/1, and the rebuilt native binary then passed the full
present/absent E2E file 14/14. JSON validation, formatting, and diff hygiene
also pass. Task 4 and the adjacent Task 3 default-catalog wiring remain gated
on scoped independent review.
