---
name: aiperf-worktree
description: Use when you need an isolated working copy of the aiperf repo for review, reproduction, or experimentation that must not disturb the user's active checkout. Creates a temp clone (default) or a git worktree (when explicitly requested), runs make first-time-setup so the editable install + mock server + plugin generation are ready, and returns the workspace path. Other aiperf-* skills (aiperf-pr-checkout, aiperf-re-review, aiperf-code-review) compose with this when isolation is required.
---

# AIPerf Isolated Workspace

Produce an aiperf working copy where review/repro/experimentation can run without touching the user's in-flight changes.

## When to use

- A skill needs to check out a PR branch or another commit and must NOT clobber the user's current branch state.
- A reproduction needs to run setup commands (`make first-time-setup`, `make install-mock-server`) that would pollute the user's environment.
- The user explicitly asked for an isolated copy.

## When NOT to use

- The user is in their normal working branch and the task is purely local (read-only inspection, lint, unit-test of unchanged files). Use the current checkout directly.
- The task is read-only and `git show`/`git diff` is sufficient.

## Choose: temp clone or git worktree

Both isolate the work from the active checkout; pick whichever fits the situation. If the caller didn't specify, ask once, or pick based on which trade-off matters more for the task.

| Strategy | Trade-off | How |
|---|---|---|
| **Temp clone** | Heavier (full clone, network) but completely independent; cleanup is `rm -rf`. Good when the workspace is short-lived or the caller wants zero risk of touching the source checkout. | `gh repo clone <owner>/<repo> "$WORKDIR"` into `mktemp -d`. |
| **Git worktree** | Lighter (shares the object store) but lives next to the source checkout and counts as a real worktree git knows about. Good for longer sessions, repeated rebases, or when network/disk for a full clone is excessive. | `git worktree add "$WORKDIR" <ref>` from the existing checkout. Cleanup needs `git worktree remove <path>` in addition to `rm -rf`. |

If invoked via `superpowers:using-git-worktrees`, defer to that skill — it handles the worktree-branches-from-main gotcha (sub-agent worktree isolation starts at `main`, not the current branch; reset and copy untracked new files manually).

## Steps

```bash
# 1. Pick a workspace path the user can find later.
WORKDIR="$(mktemp -d -t aiperf-iso-XXXXXX)"   # or -t aiperf-pr-<n>-XXXXXX when known

# 2. Get the code there.
gh repo clone <owner>/<repo> "$WORKDIR" -- --quiet
#   (git worktree variant: git worktree add "$WORKDIR" <ref>)

# 3. Switch to the workspace for the remaining setup steps.
cd "$WORKDIR"

# 4. ALWAYS run make first-time-setup (see warning below — never `uv sync`).
make first-time-setup
```

`make first-time-setup` is the only sanctioned setup target for aiperf. It chains the editable install + mock-server install + plugin artifact generation that downstream review/repro skills depend on.

**Never `uv sync` in this repo.** `uv sync` syncs the project venv against the main `pyproject.toml` only — and the mock server lives in a SEPARATE package at `tests/aiperf_mock_server/` that gets installed via `make install-mock-server`. Running `uv sync` after a successful setup will uninstall the mock server (and any other editable installs not listed in the top-level `pyproject.toml`), leaving `aiperf-mock-server` unavailable on `$PATH` and breaking every skill that boots the mock for reproductions or testing.

**Existing workspace that's fallen behind `main`:** if the workspace existed before main added a new dep, you may see `ModuleNotFoundError` from `aiperf`, pytest, or pre-commit hooks. Re-run `make first-time-setup` to re-sync, or for a quick targeted fix install the missing package with `uv pip install <pkg>` (always `uv pip install`, never `pip install`, and NEVER `uv sync`). See `aiperf-debug` "Tooling drift" for the canonical recovery list.

## Handling setup failure

`make first-time-setup` is the only step here that's likely to fail. If it does:

1. Capture the error output verbatim — do not paraphrase.
2. Tell the user the workspace path so they can inspect it (`$WORKDIR`).
3. Offer two options:
   - **(a)** Have them run `! make first-time-setup` themselves (interactive `!` prefix) so the output lands directly in the session and any auth/credential prompts can be handled.
   - **(b)** Continue in read-only mode if the calling skill supports it — surface that the runtime-reproduction steps will be unavailable.
4. Do NOT silently skip setup and let downstream steps appear to succeed.

## Output contract

When the workspace is ready, report to the calling skill (or user):

```
WORKDIR=<absolute path>
SETUP=ok | failed | read-only
GIT_HEAD=<sha of HEAD in the new workspace>
```

Calling skills treat `$WORKDIR` as the cwd for every subsequent `git`/`gh`/`make`/`uv` command.

## Cleanup

Do not auto-delete `$WORKDIR`. The user may want to inspect artifacts (`artifacts/repro-runtime-*/`, `artifacts/code-review-*/`) after the calling skill finishes. Mention the path in the final response so the user can clean it up themselves.

For git worktree paths, remind the user that `git worktree remove <path>` is required in addition to `rm -rf`, otherwise the parent repo's worktree index stays stale.

## Common mistakes

- **Using `uv sync` or `pip install -e .` directly.** Repeating the warning above for emphasis: `uv sync` does not regenerate plugin artifacts and uninstalls the editable `aiperf-mock-server` (separate package, not in the main `pyproject.toml`). Downstream skills break with confusing errors. Always `make first-time-setup` (full setup) or `uv pip install <pkg>` (targeted package install).
- **Running `cd` into the workspace and forgetting to use absolute paths afterwards.** When the workspace is created, prefer to pass `$WORKDIR` explicitly to every subsequent command (`git -C "$WORKDIR" ...`, or `(cd "$WORKDIR" && ...)`).
- **Reusing an existing temp directory across PRs.** `mktemp -d` per invocation; stale state in a shared temp dir produces silent cross-PR contamination.
- **Picking a strategy without thinking about it.** Temp clone and git worktree have different trade-offs (network/disk vs shared object store; independent vs lives-next-to-source). Match the choice to the task; if the caller didn't specify, ask once.
