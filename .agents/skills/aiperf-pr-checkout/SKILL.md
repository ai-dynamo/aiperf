---
name: aiperf-pr-checkout
description: Use when you need an aiperf PR's HEAD checked out in an isolated workspace for review or reproduction work, with baseline review/main SHAs fetched. Composes with aiperf-worktree to create the isolated workspace, then runs gh pr checkout, fetches any additional refs the calling skill needs, and reports the workspace path back. Used by aiperf-re-review (for "since last review" audits) and aiperf-code-review (when reviewing a PR rather than the current branch).
---

# AIPerf PR Checkout

Land a specific aiperf PR's HEAD in an isolated workspace so review/repro/test skills can run against it without touching the user's checkout.

## When to use

- Re-reviewing a PR (`aiperf-re-review`) and you need both PR HEAD and the prior-review SHA reachable.
- Reviewing a PR that's not the current branch (`aiperf-code-review` invoked with a PR number).
- Running reproduction or testing against a PR's code without merging or pulling it locally.

## When NOT to use

- The user is already on the PR branch in their primary checkout and the task tolerates that state.
- You only need to read PR metadata (`gh pr view`, `gh pr diff`) — no checkout required.

## Inputs

- `<pr>` — PR number, URL, or empty (use the current branch's PR via `gh pr view --json number`).
- `<extra-sha>` (optional) — additional commit SHA the caller needs reachable (e.g., `LAST_REVIEW_SHA` for re-review, a baseline SHA the user named).

## Steps

### 1. Resolve PR metadata

```bash
gh pr view <pr> --json number,headRefOid,baseRefName,headRefName,url,author,title \
  | tee /tmp/pr-<pr>-meta.json
```

Note the `headRefOid` — that's the PR HEAD SHA the calling skill will treat as the diff endpoint.

### 2. Create isolated workspace

Delegate to `aiperf-worktree`. Pass a workspace hint like `aiperf-pr-<n>` so the temp dir is identifiable. Wait for its output contract (`WORKDIR`, `SETUP`, `GIT_HEAD`).

If `SETUP=failed`, surface that to the calling skill — do not proceed to step 3 silently. Most review/repro work needs the editable install + mock server.

### 3. Check out the PR HEAD

All commands from here run inside `$WORKDIR`. Use `git -C` or `(cd …)` rather than persistent `cd`.

```bash
cd "$WORKDIR"

# Authenticate gh inside the workspace if needed (usually inherits from user's gh config).
gh pr checkout <pr>

# Verify HEAD matches what gh pr view reported.
git rev-parse HEAD   # expect: headRefOid from step 1
```

`gh pr checkout` handles the fetch+checkout in one shot, including PRs from forks (it creates a tracking branch named after the PR's source branch).

### 4. Fetch any extra refs the caller needs

For re-review, the baseline SHA must be reachable for `git diff LAST_REVIEW_SHA..HEAD`. GitHub does NOT allow fetching by raw SHA on standard repos — `git fetch origin <sha>` fails with `fatal: remote error: upload-pack: not our ref`. Fetch the PR's full ref instead, which makes all commits on the PR reachable (including superseded prior-review SHAs):

```bash
if [ -n "$EXTRA_SHA" ]; then
  # Always fetch via the pull ref — never `git fetch origin <raw-sha>` (GitHub rejects it).
  git fetch origin "+refs/pull/$PR/head:refs/remotes/origin/pr-$PR"
  git cat-file -e "$EXTRA_SHA"^{commit} \
    || { echo "EXTRA_SHA $EXTRA_SHA unreachable on the PR ref (force-pushed away?)"; exit 1; }
fi
```

If the extra SHA is still unreachable after the full PR-ref fetch, it was likely force-pushed away from the PR — tell the caller; do not silently substitute another SHA.

For correctness review (`aiperf-code-review`), ensure `origin/main` is up to date:

```bash
git fetch origin main
```

### 5. Report back

When done, report to the calling skill:

```
WORKDIR=<absolute path>
PR=<n>
HEAD_SHA=<sha>
BASE_REF=<baseRefName from gh pr view, usually 'main'>
EXTRA_SHA=<sha or empty>
SETUP=ok | failed | read-only
```

## Common mistakes

- **Calling `git pull` in the user's primary checkout instead of using a fresh workspace.** That's the failure this skill exists to prevent. Always delegate to `aiperf-worktree`.
- **Using `git fetch <pr-number>` directly instead of `gh pr checkout`.** Works only when the PR is from the same repo, not from forks. `gh pr checkout` handles both cases uniformly.
- **Forgetting to fetch `LAST_REVIEW_SHA` for re-review.** Without it, `git diff LAST_REVIEW_SHA..HEAD` will fail or silently fall back to the wrong range. Verify with `git cat-file -e <sha>^{commit}` before claiming the workspace is ready.
- **Assuming the workspace persists across sessions.** It's a temp dir; tell the user the path so they can inspect it before deleting.
