# Session Task 8 — Graham round 3

Target: `de8d9820dc` plus this bounded follow-up.

## Findings and correction

`reconstruct_shipped_dataset` validated a manifest entry immediately before
fetching it. A duplicate in a later entry therefore caused the cell to fetch
the first source before it rejected the malformed manifest.

The reconstruction path now validates the complete manifest before its first
network request: every relative file path, duplicate identity, applicable base
name, and kind are checked up front. The duplicate diagnostic remains specific:
`duplicate dataset manifest path "main.jsonl"`.

The session-set reconstruction fixture now uses the layout the controller
produces for a selected directory beneath an explicit replay root:

- `base_name: "selected"`;
- files rooted at `selected/...`, including a direct Claude subagent;
- reconstruction returns the landed selected directory; and
- discovery from that landed root has the exact same root-relative path-to-bytes
  set as discovery from the controller source.

## Regression coverage

- An unreachable authority with a duplicate manifest must return the duplicate
  error rather than a connection error, proving validation completes before a
  fetch starts.
- The selected-directory Claude fixture verifies the complete rediscovered set
  and byte identity for both main and subagent files.

## Verification receipt

- `cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- Focused RED/GREEN command: `cargo test -p aiperf-runtime --features engine
  agent_session_exact_set -- --nocapture`.
- The required `RUSTC_WRAPPER=/usr/bin/sccache` was preserved. The worktree's
  `rust/target/debug/.cargo-build-lock` sits on the read-only `/mnt/4tb` mount,
  so Cargo fails before compilation with `Read-only file system (os error 30)`.
  The shared target, compiler wrapper, and sccache configuration were not
  overridden or bypassed.
- The worktree pre-commit hook also cannot start its configured `.venv/bin/pre-commit`
  because this isolated worktree has no `.venv`. Invoking the available project
  pre-commit binary directly passed all applicable checks before its `add-license`
  hook reached the same missing-worktree-venv failure.

## Graham pass

The diff is limited to manifest prevalidation, the two focused regression cases,
and this receipt. It adds no request-path work, asynchronous tasks, locks,
dependencies, tracing, or production `unwrap`/`expect` calls.
