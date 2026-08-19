# Session Task 8 — Graham round 5

Target: `6c811fcfeb` plus the final cellular regression coverage.

## Coverage added

- The controller exact-session serve-plan path now has a Unix symlink fixture.
  An imported Codex directory with a symlinked JSONL source is rejected while
  preparing its allowlisted manifest. `run_cellular` performs that preparation
  before creating its artifact server, so the untrusted path cannot become an
  exposed dataset route.
- A product E2E uses the in-repo mock server, three imported Codex sessions,
  and three real cellular child processes with
  `AIPERF_CELL_ARTIFACT_HTTP_FORCE=1`. It verifies every cell downloads every
  exact-set source over zstd, compares the deterministic raw request set with
  a one-cell baseline, and requires each cellular raw record to have an HTTP
  200 response with generated content.

## Verification receipt

- `cargo fmt --all -- --check` — passed.
- `git diff --check` — passed.
- Focused runtime regression attempt:
  `cargo test -p aiperf-runtime --features engine
  agent_session_exact_set_rejects_symlink_before_artifact_server_binding --
  --nocapture`.
- Product E2E attempt:
  `cargo test -p aiperf-e2e-tests
  test_cellular_imported_session_exact_set_shipping_matches_single_cell_raw_records
  -- --nocapture`.

Both cargo test commands stopped before compilation because Cargo could not
open `rust/target/debug/.cargo-build-lock` on the read-only target filesystem
(`Read-only file system (os error 30)`). The worktree target, configured
`sccache` wrapper, and environment were not overridden or bypassed.

## Graham pass

The production diff is test-only. It adds no runtime allocations, tasks,
locks, dependencies, logging, or production `unwrap`/`expect` paths. The E2E
uses a three-session exact set so the source-allowlist contract is checked
without an exhaustive fixture matrix.
