---
name: aiperf-merge-from-main
description: Use BEFORE merging origin/main into a long-lived aiperf branch — "merge main into this branch", "rebase onto main", "catch up with origin/main", "resolve merge conflicts", "main has moved, pull it in". Codifies the signature-drift trap that bites on aiperf merges: signature renames in main applied to files NOT added to your branch, but callers in files YOU added still use the old signature (auto-merge hides this).
---

# AIPerf Merge from Main

Long-lived branches accumulate a silent merge hazard auto-merge accepts cleanly: **signature drift in added files.** Main renamed `foo(x, y)` → `foo(x, y, z)`. Auto-merge applies the rename to existing files. But files YOU added on your branch (which didn't exist in main) still call `foo(x, y)`. They land green-on-merge but break at runtime.

This skill is the safe path through that trap, plus the routine merge / rebase mechanics.

## Pre-flight

```bash
# 1. Are we on the branch we think we are?
git rev-parse --abbrev-ref HEAD

# 2. Is the working tree clean?
git status

# 3. What's the merge base?
git merge-base origin/main HEAD

# 4. What changed in main since the base?
git log --oneline $(git merge-base origin/main HEAD)..origin/main | head -30

# 5. What did our branch add / change?
git diff --stat $(git merge-base origin/main HEAD)..HEAD
```

If the working tree isn't clean, commit (via `aiperf-commit`) before merging. Never merge with uncommitted changes — they get mixed into the merge commit and become impossible to back out.

## Steps

### 1. Fetch and merge

```bash
git fetch origin main
git merge origin/main         # or: git rebase origin/main, if your branch hasn't been pushed
```

For shared/published branches, prefer `merge` over `rebase` (rebase rewrites history). For solo/unpushed branches, `rebase` keeps history linear.

### 2. Resolve conflicts normally

Resolve conflicts file-by-file on their merits. The pre-commit `check-agent-files-sync` hook will enforce that `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, and `.cursor/rules/python.mdc` end up with identical bodies (only headers/frontmatter differ), so after resolving those four files individually, verify with `make check-agent-files-sync` before committing the merge.

If both sides edited the same skill under `.agents/skills/`, read both diffs and merge the content — there is no auto-prefer-one-side rule for skill files. Verify the resulting skill still passes any sanity check the affected skill documents.

### 3. Audit signature drift on added files

This is the silent failure mode. Auto-merge applied main's renames to existing files; YOUR added files still use the old call sites.

```bash
# Files added on your branch (existed on YOUR side only)
git diff --name-only --diff-filter=A $(git merge-base origin/main HEAD)..HEAD > /tmp/added-on-branch.txt

# Files added on main (existed on MAIN side only)
git diff --name-only --diff-filter=A $(git merge-base origin/main HEAD)..origin/main > /tmp/added-on-main.txt

# Files renamed/heavily-modified on main
git log --diff-filter=R --summary $(git merge-base origin/main HEAD)..origin/main | grep '^ rename' | head
```

For each function/method renamed on main:
- Grep your added files for the old name. Any hit is a latent bug.
- Grep your added files for callers of the renamed module.

```bash
# 1. Find signature-affecting changes on main since the merge base. Look for
#    renames, and also for changed argument lists on functions that kept their
#    names. `-p` shows the patch so you can spot signature changes by eye.
git log -p --diff-filter=RM --pickaxe-regex \
  $(git merge-base origin/main HEAD)..origin/main \
  -- 'src/aiperf/**/*.py' | less

# 2. For each old name you identified, grep your added files for callers.
#    Replace OLD_NAME with the actual symbol; do not skip this step.
OLD_NAME="<paste the old function name from step 1>"
grep -rn "${OLD_NAME}\b" $(cat /tmp/added-on-branch.txt) | head
# Inspect each hit. If signature drifted, fix the call site.
```

Run step 2 once per renamed symbol from step 1. There is no shortcut — the symbols are project-specific to whatever main introduced this merge cycle.

Auto-merge will not flag this. You must look.

### 4. Run the test tier hit by the merge

After resolving conflicts and fixing signature drift, run unit tests at minimum:

```bash
uv run pytest -n auto tests/unit/   # use aiperf-pytest skill for the canonical invocation
```

If the merge touched runtime paths (services, message bus, plugin registry), also run `aiperf-correctness-testing` for an end-to-end smoke.

### 5. Verify four-file sync still passes

```bash
make check-agent-files-sync
```

If you took main's version of the four sync files, they should match by definition. If this fails, one of them didn't fully resolve to main's version — re-do step 2 for the offending file.

### 6. Commit the merge

```bash
git commit -s -m "$(cat <<'EOF'
Merge origin/main into <branch>

<one-paragraph summary of what main introduced and how it affects this branch>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Use `aiperf-commit` for the commit-time discipline (heredoc reflow handling).

For `git rebase` (instead of merge): the commits get re-applied; you'll resolve conflicts per commit. The `--theirs` flag flips meaning during rebase (`--theirs` = your branch's version, `--ours` = main's). Confirm by reading `git status` output before resolving — never act on the flag name alone during a rebase.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "Auto-merge applied cleanly, ship it" | Auto-merge applies renames silently; your added files still use old signatures. Audit step 3 BEFORE committing the merge. |
| "I'll keep my version of CLAUDE.md, my edits are fine" | Merge them. Run `make check-agent-files-sync` after — the four-file sync rule enforces identical bodies across all four agent-facing files. |
| "Tests pass, the merge is good" | Unit tests don't exercise added files' new code paths against main's renamed APIs at runtime. Run `aiperf-correctness-testing` for any runtime-touching merge. |
| "I'll resolve conflicts file-by-file and decide each one on its merits" | Convention-based decisions are faster and less error-prone. Agent docs → main. Generated artifacts → regenerate. Signature drift → fix the call site. |
| "Rebase keeps history clean, I'll always rebase" | For pushed/shared branches, rebase rewrites history that others may have based work on. Use merge when the branch is shared. |
| "I'll just `git checkout --theirs .` for everything" | Loses ALL of your branch's work in conflicts. Resolve conflicts per file on their merits. |

## Common mistakes

- **Skipping the signature-drift audit.** The single most common silent-merge failure.
- **Forgetting to regenerate `.agents/plugin/enums.py`** etc. after the merge — if main added or removed plugins. Run `make generate-all-plugin-files`.
- **Committing the merge while four-file sync is broken.** Pre-commit catches it but you'll hit the heredoc-reflow gotcha. Run `make check-agent-files-sync` first.
- **Merging without rebasing your local `aiperf-mock-server` install** — if main bumped its deps, `which aiperf-mock-server` may now point at a stale binary. Re-run `make first-time-setup` after the merge if mock-server-using tests fail.
- **Confusing `--theirs` and `--ours` during rebase** — they flip meaning vs merge. Always check `git status` instead of trusting the flag name.

## Composition

- `aiperf-worktree` if you want to test the merge in isolation before applying to your active checkout.
- `aiperf-commit` for the merge-commit step (heredoc handling).
- `aiperf-pytest` for the test step.
- `aiperf-correctness-testing` if the merge touched runtime paths.
