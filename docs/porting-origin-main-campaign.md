# origin/main Rust Porting Campaign Ledger

## Scope

This ledger tracks every commit reachable from `origin/main` and not reachable
from `ajc/rust-merge-main` at campaign start (`0b67e08284ef`). The baseline
comparison was `origin/main` at `c2889280a66f`; `git rev-list
HEAD..origin/main` reported 60 commits.

For each commit, the campaign records the upstream intent, the Rust parity
assessment, an actual non-fast-forward merge commit, any Rust-port design and
Sol plan, TDD evidence, verification, and the required Graham review.

## Status vocabulary

- `pending`: not yet inspected.
- `analysing`: upstream and Rust code comparison in progress.
- `merged`: an actual merge commit has incorporated the upstream commit.
- `rust-porting`: a Rust delta was identified and is being specified/planned.
- `complete`: merge, applicable Rust port, tests, and Graham review are all
  evidenced.
- `not-applicable`: no Rust behavior corresponds to the upstream change;
  evidence is recorded.

## Campaign inventory

| Order | Upstream commit | Subject | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1 | `817a8d84ddb9` | fix(accuracy): grade LCB codegen in an out-of-process worker (#1145) (#1175) | rust-porting | Merge commit `1c03271dac3e` has upstream as its second parent. Focused upstream tests: 35 passed, 1 skipped. The native evaluator's LCB batch path still forks via `asyncio.to_thread`; the port spec is `docs/specs/2026-08-25-native-lcb-codegen-worker.md`. |
| 2-60 | `git rev-list --reverse HEAD..origin/main` | Remaining ancestry-ordered upstream commits | pending | Exact inventory is reproducible from the baseline above; rows are expanded before each commit is processed. |

## Per-commit record: 817a8d84ddb9

### Upstream intent

Move LiveCodeBench code-generation grading to a supervised child process so
potentially wedged grading does not hold the parent benchmark process.

### Initial Rust comparison

Rust has a versioned evaluator-worker protocol with `load`, `next_problems`,
`grade_batch`, and `shutdown`, plus subprocess supervision in
`rust/runtime/src/accuracy_core/worker.rs`. The remaining inspection will
compare lifecycle, failure propagation, reaping, and LiveCodeBench-specific
selection before deciding whether any Rust delta remains.

### Merge evidence

`1c03271dac3eb6465538dabf6950fd255baeac7d` is a two-parent merge with
`817a8d84ddb90d1e12c2a03327e16d853bb4e6e0` as the second parent. The focused
upstream suite produced `35 passed, 1 skipped`; pytest emitted cleanup warnings
for pre-existing Docker-owned temporary paths. Project-wide commit hooks then
reported pre-existing unrelated import and ergonomics/ruff-baseline failures;
the staged merge touched none of their listed source paths.

### Port decision

Applicable. Rust's `PythonEvaluator` starts `aiperf.accuracy.worker`, whose
`AccuracyWorker._grade_lcb_batch` still invokes `_run_codegen_metrics` through
`asyncio.to_thread`. That retains the upstream fork-from-thread hazard on the
native Rust evaluation route. The selected correction reuses the merged
`CodegenGradingWorker` inside that evaluator, recorded in
`docs/specs/2026-08-25-native-lcb-codegen-worker.md`.

### Required evidence before close

- Upstream semantic diff and current Rust counterpart analysis.
- A true merge commit containing the upstream ancestry.
- A feature spec and Sol-produced plan if any Rust delta is applicable.
- TDD red/green evidence and all applicable tests.
- A full Graham review with every finding resolved.
