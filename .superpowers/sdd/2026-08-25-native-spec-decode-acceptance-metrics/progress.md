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

- Task 1: implemented and GREEN; independent review pending
- Task 2: pending
- Task 3: pending
- Task 4: pending

Task 1: dispatch fallback — `spawn_agent` was denied because the root thread's other port agents occupied the team limit. Per the SDD skill's harness-denial rule, implementation proceeds inline; an independent task review remains required before Task 2.

Task 1 receipt: see `task-1-report.md`. Initial `cargo test -p aiperf-runtime spec_decode --lib` was RED on the intended absent DTO/callback/record/typed-choice surface. The controller's recovered authoritative GREEN command was `RUSTC_WRAPPER=sccache CARGO_TARGET_DIR=/mnt/4tb/aiperf-target-port-013 cargo test -p aiperf-runtime --features engine --lib spec_decode -- --nocapture`: 6 passed, 0 failed, with four unchanged baseline warnings. Task 2 remains gated on independent review.
