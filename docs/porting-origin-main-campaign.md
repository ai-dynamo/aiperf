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
| 1 | `817a8d84ddb9` | fix(accuracy): grade LCB codegen in an out-of-process worker (#1145) (#1175) | analysing | Upstream uses an isolated Python grading worker. Rust already has the analogous evaluator-worker protocol in `rust/runtime/src/accuracy_core/worker.rs`; detailed semantic comparison in progress. |
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

### Required evidence before close

- Upstream semantic diff and current Rust counterpart analysis.
- A true merge commit containing the upstream ancestry.
- A feature spec and Sol-produced plan if any Rust delta is applicable.
- TDD red/green evidence and all applicable tests.
- A full Graham review with every finding resolved.
