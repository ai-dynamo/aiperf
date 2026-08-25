# Origin #38 closure: AgentX FAQ

Upstream commit `c26fe88bd8` removes the standalone AgentX FAQ page,
drops its navigation entry from `docs/index.yml`, and rewrites the remaining
`docs/tutorials/agentx-mvp.md` references so the tutorial stands on its own.

The diff is documentation-only. It does not touch `rust/`, the native CLI
surface, runtime behavior, or any Rust-launched integration seam. Native
behavior is therefore unchanged, so there is no Rust implementation to port and
no Rust TDD target to add.

Disposition: not-applicable; exact merge performed for campaign ancestry.

Verification inventory:

- `git diff --check HEAD^1 HEAD` on merge commit `233b71c98bb49866d883cc03773e6a81cb6b6451`: clean.
- `git diff --stat HEAD^1 HEAD` on merge commit `233b71c98bb49866d883cc03773e6a81cb6b6451`: 3 files changed, all in `docs/`.
- No Rust test candidate exists because the upstream change only removes docs and
  tutorial links.

Graham review outcome:

- No findings. The merge changes only documentation files, outside the
  Rust/runtime review hot paths governed by the Graham rubric.
