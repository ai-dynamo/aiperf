# Session Task 8 — Graham round 4

Target: `bf89ffe3d0` plus this bounded null-graph correction.

## Finding and correction

The controller treated a present JSON `null` at `datasets[0].graph` as a
`RecordedAgentGraphConfig`, so serde rejected a valid optional-field form before
Auto JSONL session dispatch. `null` now follows the same path as an omitted
graph configuration.

The cell rewrite previously created `graph` only when the member was absent.
It now replaces a present `null` with an object before setting the landed
`replay_root`, while preserving rejection of non-object, non-null values.

## Regression coverage

- An `agent_recording` Auto JSONL envelope with `graph: null` builds the
  imported exact-session manifest and serves its one JSONL source.
- A landed `agent_session_set` envelope with `graph: null` rewrites the local
  path and materializes `graph.replay_root`.

## Verification receipt

- `cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- RED and post-fix focused command:
  `cargo test -p aiperf-runtime --features engine agent_session_exact_set -- --nocapture`.
  `RUSTC_WRAPPER=/usr/bin/sccache` was preserved. Both attempts stopped before
  compilation because
  `/home/anthony/nvidia/projects/aiperf/ajc/.worktrees/dag-v3-session-import/rust/target/debug/.cargo-build-lock`
  is on a read-only filesystem: `Read-only file system (os error 30)`. No
  target directory, compiler wrapper, or sccache configuration was overridden.
- The worktree pre-commit hook cannot start its configured
  `.venv/bin/pre-commit` because this isolated worktree has no `.venv`.

## Graham pass

The production diff adds no hot-path allocation beyond the existing serde
configuration decode, no locks, tasks, dependencies, logging, or production
`unwrap`/`expect`. The null normalization is scoped to the two cellular
configuration boundaries; non-null malformed graph values retain their prior
typed rejection.
