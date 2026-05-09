---
name: cherry-pick
description: Cherry-pick a commit from origin/main into a release branch using a temporary git worktree
disable-model-invocation: true
allowed-tools: Bash(git fetch *), Bash(git log *), Bash(git branch *), Bash(git worktree *), Bash(git cherry-pick *), Bash(git -C * status), Bash(git -C * cherry-pick *), Bash(git -C * checkout *), Bash(git -C * push *), Bash(git -C * branch *), Bash(git push *), Bash(git commit *), Bash(git checkout *), Bash(mktemp *), Bash(make *), Bash(gh pr create *), Bash(gh api *), AskUserQuestion
---

# Cherry-Pick to Release Branch

Cherry-pick a commit from `origin/main` into a release branch using a temporary git worktree.

## Resume Check

Before starting, check if there is an existing cherry-pick worktree from a previous run:
```bash
git worktree list | grep cherry-pick/
```
If there is a worktree on a `cherry-pick/*` branch, a previous cherry-pick may have stopped due to conflicts. Ask the user if they want to resume it. If yes:
1. Set `WORKTREE_DIR` to the existing worktree path.
2. Check the cherry-pick state:
   ```bash
   git -C "$WORKTREE_DIR" status
   ```
   - If the cherry-pick is still in progress (conflicts resolved, staged), run `git -C "$WORKTREE_DIR" cherry-pick --continue` to finalize.
   - If the working tree is clean (cherry-pick already completed), proceed to push.
3. Recover the original source commit SHA from the worktree's branch name (it's encoded as `cherry-pick/<source-sha>-to-<release-branch>`):
   ```bash
   BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current)
   SOURCE_SHA=${BRANCH#cherry-pick/}
   SOURCE_SHA=${SOURCE_SHA%-to-*}
   ```
   Do NOT use the worktree's HEAD SHA — that's the new cherry-pick commit, not the original. Then run step 4 with `$SOURCE_SHA` to gather the original PR metadata.
4. Continue from step 10 (push) onward.

If the user does not want to resume, clean up the stale worktree first, then start fresh.

## Steps

1. **Fetch latest from origin:**
   ```bash
   git fetch origin
   ```

2. **Show recent commits on origin/main:**
   Run `git log origin/main --oneline -20` to get the list.

3. **Ask the user which commit to cherry-pick:**
   Use `AskUserQuestion` to present the recent commits as options. Let the user select one.

4. **Look up the original PR for the selected commit:**
   `gh pr view` does not accept a commit SHA, so query the commits→pulls API directly:
   ```bash
   gh api repos/ai-dynamo/aiperf/commits/<source-commit-sha>/pulls \
     --jq '.[0] | "\(.number)\t\(.title)"'
   ```
   Save the PR number and title for use in steps 11 and 13. If the commit message contains a PR number (e.g. `(#669)`), you can also use that directly.

5. **Find all release branches:**
   Run `git branch -r --list 'origin/release/*'` to get all remote release branches matching `release/X.X.X`.

6. **Ask the user which release branch to target:**
   Use `AskUserQuestion` to present the release branches as options. The most recent version (highest semver) should be the first option marked as "(Recommended)".

7. **Create a temporary worktree on a feature branch:**
   Release branches are protected, so work on a `cherry-pick/` feature branch from the start.
   This also makes the worktree identifiable for resume if conflicts occur.
   ```bash
   WORKTREE_DIR=$(mktemp -d)
   git worktree add "$WORKTREE_DIR" origin/<release-branch>
   git -C "$WORKTREE_DIR" checkout -b cherry-pick/<commit-hash>-to-<release-branch>
   ```

8. **Set up the environment in the worktree (required for pre-commit hooks):**
   ```bash
   make -C "$WORKTREE_DIR" first-time-setup
   ```

9. **Cherry-pick the commit:**
   ```bash
   git -C "$WORKTREE_DIR" cherry-pick <commit-hash>
   ```
   If there are conflicts:
   - Do NOT force resolve conflicts automatically.
   - Do NOT clean up the worktree — leave it in place so the user can fix conflicts.
   - Tell the user:
     1. The worktree path: `$WORKTREE_DIR`
     2. To resolve conflicts in that directory, stage the fixes, then run: `git -C "$WORKTREE_DIR" cherry-pick --continue`
     3. If conflict resolution requires a separate commit (not `cherry-pick --continue`), use `git commit -s` with a HEREDOC message.
     4. Once resolved, re-run `/cherry-pick` to resume (it will detect the existing worktree).

10. **Push the feature branch:**
    ```bash
    git -C "$WORKTREE_DIR" push -u origin HEAD
    ```

11. **Create PR (optional — requires `gh` CLI):**
    If `gh` is available and authenticated, create the PR automatically:
    ```bash
    gh pr create \
      --repo ai-dynamo/aiperf \
      --head cherry-pick/<commit-hash>-to-<release-branch> \
      --base <release-branch> \
      --label cherry-pick \
      --title "<original commit title> (cherry-pick to <release-branch>)" \
      --body "$(cat <<'EOF'
    ## Summary
    - Cherry-pick of <commit-hash> from `main` into `<release-branch>`
    - Original PR: #<original-pr-number> — <original commit title>

    ## Conflicts
    <If conflicts were resolved, list the files and briefly describe the resolution. Otherwise write "None — clean cherry-pick.">
    EOF
    )"
    ```
    If `gh` is not set up or the command fails, skip this step and tell the user to create the PR manually at:
    `https://github.com/ai-dynamo/aiperf/compare/<release-branch>...cherry-pick/<commit-hash>-to-<release-branch>`

12. **Clean up the worktree:**
    ```bash
    git worktree remove "$WORKTREE_DIR"
    ```

13. **Print a summary.** Output the following (NOT inside a code block):

    Cherry-pick complete: `<commit-hash-short>` → `<release-branch>`
    <original commit title>

    PR Main: <original-pr-number> :git-merged: https://github.com/ai-dynamo/aiperf/pull/<original-pr-number>
    PR Release: <cherry-pick-pr-number> :pr-opened: https://github.com/ai-dynamo/aiperf/pull/<cherry-pick-pr-number>

    If step 11 was skipped (no PR created), omit the PR Release line and instead print the manual comparison URL from step 11's fallback.

## Rules

- NEVER add a `Co-Authored-By` line
- NEVER manually add a `Signed-off-by` line — `git commit -s` handles this automatically
- If the cherry-pick results in a new commit (e.g. conflict resolution), use `git commit -s` with a HEREDOC message
- NEVER use `--no-verify`
- NEVER force push
- NEVER use `git stash`, `git reset`, `git revert`, `git checkout -- <file>`, `git restore`, or `git clean`
- Always clean up the worktree after a successful push or if the user abandons the cherry-pick
- On cherry-pick conflicts: do NOT clean up the worktree — leave it for the user to resolve, then resume
- If anything else fails, clean up the worktree and report the error to the user
