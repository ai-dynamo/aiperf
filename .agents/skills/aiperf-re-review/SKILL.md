---
name: aiperf-re-review
description: Use when re-reviewing an aiperf GitHub PR that already has prior reviews. Audits commits and inline comments since the last review, classifies each prior thread as addressed-in-code / responded-to / ignored, and flags scope creep — code changes that don't trace back to any review feedback. Triggers on phrases like "re-review this PR", "review the changes since my last review", "did they address my comments", "what changed since last review". For setting up the isolated working copy, see aiperf-worktree and aiperf-pr-checkout. For reproducing fixes, see aiperf-mock-server.
argument-hint: "[pr-number-or-url]"
---

# Re-Review an AIPerf GitHub PR

Audit a PR's progress since the last review: did the author address each prior comment, and did anything else sneak in?

## When to use

- A reviewer asks you to take a second look after the author pushed updates.
- You need to verify that every prior review comment was addressed faithfully or explicitly responded to.
- You want to catch unrelated code changes (scope creep) shipped alongside review fixes.

If the PR has **no prior reviews**, use `aiperf-code-review` instead.

## Inputs

- `$ARGUMENTS` may be a PR number, URL, or empty (then use the current branch's PR).
- Resolve owner/repo from `gh repo view --json nameWithOwner` if not obvious.

## Steps

### 1. Identify the PR and its review history

```bash
gh pr view <pr> --json number,headRefOid,baseRefName,headRefName,author,title,url,commits
gh api repos/{owner}/{repo}/pulls/<pr>/reviews \
  --jq '.[] | {id, user: .user.login, state, submitted_at, commit_id, body_excerpt: (.body[0:80])}'
```

Each review has a `submitted_at` timestamp and a `commit_id` (the SHA the review was anchored to).

### 1a. Pick the baseline review — ASK if ambiguous

A PR usually has reviews from several actors. Group them by `user.login` and classify:

- **Human reviewers** — `acasagrande`, teammates, etc.
- **AI/bot reviewers** — `coderabbitai`, `coderabbitai[bot]`, `copilot-pull-request-reviewer[bot]`, `github-actions[bot]`, `claude[bot]`, anything ending in `[bot]`. CodeRabbit in particular posts a fresh review on every push, so its "last review" is almost always the most recent submission and is rarely what the user means.

**Decision rules for which review's `commit_id` to anchor on:**

| Situation | Action |
|---|---|
| User said "re-review my comments" or invoked from their own account | Use the **invoking user's** most recent non-PENDING review |
| User said "re-review the CodeRabbit feedback" / named a specific reviewer | Use **that reviewer's** most recent review |
| Exactly one human reviewer (besides bots) | Use that human's most recent review; mention the bot reviews exist but don't anchor on them |
| Multiple human reviewers, user didn't specify | **STOP and ask** — list each reviewer with their last review's date and commit SHA, and ask which baseline to use (or whether to do a per-reviewer audit) |
| Only bot reviews exist | Confirm with user — they may want a fresh review (`aiperf-code-review`) instead, since bot feedback is usually addressed inline rather than via a formal re-review |
| User's most recent review was DISMISSED, or its `commit_id` is no longer reachable (force-push) | Ask the user: anchor on the dismissed review anyway, on the prior review, or on a specific SHA they name |

When asking, present a compact table:

```
Reviewer              State        Submitted             commit_id
acasagrande           CHANGES_REQ  2026-04-26 14:02 UTC  abc1234
coderabbitai[bot]     COMMENTED    2026-04-28 09:15 UTC  def5678
teammate-x            APPROVED     2026-04-25 18:30 UTC  abc1234
```

Then proceed once the user picks. Treat the chosen review's `commit_id` as `LAST_REVIEW_SHA` for the rest of the steps, and when auditing threads in step 4, only include threads authored by that reviewer (unless the user asked for an all-reviewer audit).

### 1b. Set up an isolated working copy

**Delegate to `aiperf-pr-checkout`.** It composes with `aiperf-worktree` to create a temp clone (where `make first-time-setup` runs as part of that skill's setup), then runs `gh pr checkout <pr>` and fetches the baseline SHA. Pass `<pr>` and `LAST_REVIEW_SHA` so the baseline is reachable. The sub-skill returns the workspace path; treat it as `$WORKDIR` for every subsequent `git`/`gh`/`make`/`uv` command in this skill. The user's original repo stays untouched.

If `aiperf-pr-checkout` reports setup blocked (sandbox denies network, missing credentials, install errors), it will already have surfaced the failure and asked the user how to proceed. Do not silently continue — steps 2-6 still work for a read-only audit, but step 4's "reproduce the fix" verification degrades to code-reading only.

### 2. Pull every prior review thread

Use the GraphQL API to get resolution state — REST `comments` endpoint omits `isResolved`/`isOutdated`:

```bash
gh api graphql -f query='
  query($owner:String!,$repo:String!,$number:Int!) {
    repository(owner:$owner, name:$repo) {
      pullRequest(number:$number) {
        reviewThreads(first:100) {
          nodes {
            isResolved
            isOutdated
            path
            line
            originalLine
            comments(first:50) {
              nodes { author{login} body createdAt diffHunk url }
            }
          }
        }
      }
    }
  }
' -F owner=<owner> -F repo=<repo> -F number=<pr>
```

Also fetch top-level PR comments (`gh api repos/{owner}/{repo}/issues/<pr>/comments`) — author replies often live there.

### 3. Diff since the last review

All git/gh commands run inside `$WORKDIR` from step 1b.

```bash
cd "$WORKDIR"
LAST_REVIEW_SHA=<commit_id from step 1>
HEAD_SHA=<headRefOid from step 1>

git fetch origin pull/<pr>/head:_pr_<pr>      # if not already local
git log --oneline ${LAST_REVIEW_SHA}..${HEAD_SHA}
git diff ${LAST_REVIEW_SHA}..${HEAD_SHA}      # or per-file
```

If the local checkout doesn't have those SHAs, fetch them. Never fall back to "current diff vs main" — that loses the since-last-review framing.

### 4. Classify every prior review thread

For each thread from step 2, determine:

| Status | How to recognize | Notes |
|---|---|---|
| **Addressed in code** | Diff from step 3 touches `path` near `line`/`originalLine`, *and* the change plausibly resolves the comment | Verify by reading the new code — don't trust line-proximity alone |
| **Responded — agreed/will-fix later** | Author replied with acknowledgement but no code change yet | Quote the reply; flag as deferred |
| **Responded — disagreed** | Author pushed back with rationale, no code change | Quote the rationale; reviewer decides if it's accepted |
| **Ignored** | No code change at that location AND no reply | This is the most important category to surface |
| **Already resolved** | `isResolved: true` on the GraphQL thread | Skip unless the resolution looks premature |
| **Outdated by GitHub** | `isOutdated: true` because surrounding code changed | Re-anchor manually if the concern still applies to the new code |

Read the new code at each touched location. "Addressed" requires the fix to actually resolve the comment, not merely edit nearby.

If a fix claim needs runtime verification, delegate to `aiperf-mock-server` to launch the mock against `127.0.0.1:<random-port>` and run `aiperf` against it. Save outputs under `artifacts/re-review-<pr>-<epoch>/repro/<scenario>/`.

### 5. Find unrelated changes (scope check)

For every file/hunk in the step-3 diff, check whether it maps to a thread from step 2:

- **Mapped** = the hunk's path+line range overlaps a review thread's anchor, OR the commit message references the review.
- **Unmapped** = neither.

List unmapped hunks. They fall into:

- Legitimate follow-up the author mentioned (e.g. "also fixed the typo in X") — fine, note it.
- Drive-by refactors, formatting churn, dependency bumps — **flag explicitly**. The reviewer needs to decide whether to ask for these to be split off.
- New features or behavior changes unrelated to the review — **flag loudly**. These deserve their own review attention.

### 6. Produce the re-review report

Write to `artifacts/re-review-<pr>-<epoch>/REPORT.md` (compute `EPOCH="$(date +%s)"` once at the top of the invocation; create `artifacts/` if missing). Same-day re-invocations for the same PR get distinct directories — do NOT overwrite a prior run. Structure:

```markdown
# Re-review of PR #<n>: <title>
Baseline: <last-review-sha> (reviewed <date> by <reviewer>)
Head: <head-sha> (<N> new commits since baseline)

## Comment-by-comment audit
### Thread 1 — `path/to/file.py:123` — <one-line summary of original comment>
- **Status:** Addressed in code | Responded (agreed/disagreed) | Ignored | Already resolved | Outdated
- **Original comment:** <quote>
- **Author response:** <quote, or "(none)">
- **Code change:** <commit SHA + brief description, or "(none)">
- **Verdict:** <your assessment — was this addressed faithfully?>

[repeat per thread]

## Changes NOT tied to review feedback
- `path/file.py` lines X-Y — <description> — <commit SHA> — <flag: refactor / feature / formatting / acceptable follow-up>

## Summary
- N threads addressed, M deferred, K ignored, J disputed
- L unrelated changes (S worth flagging)
- Overall: ready / needs another pass / scope-split recommended
```

### 7. Present, then post (only with approval)

Show the user the full report and the planned inline GitHub comments before posting. After confirmation, post via:

```bash
gh api repos/{owner}/{repo}/pulls/<pr>/reviews -f event=COMMENT -f body=<summary> -F comments=...
```

For posting inline comments anchored to specific lines, the position-vs-line mechanics are the same as `aiperf-code-review` — see that skill if you need the patch-position math.

## Rules

- **Baseline = last review's `commit_id`**, never `origin/main`. The whole point of a re-review is the delta since last review.
- **When more than one human reviewer exists, ASK** which baseline to use before doing any analysis. Don't guess.
- **Filter out bot reviewers by default** (`coderabbitai`, `*[bot]`). Mention they exist but don't anchor on them unless the user explicitly asks.
- **Always work in an isolated workspace via `aiperf-pr-checkout`**, never in the user's active checkout.
- **Verify "addressed" by reading code**, not line-proximity. A change in the same file isn't a fix.
- **Ignored comments are the headline finding.** Surface them prominently — that's why the user asked for a re-review.
- **Unrelated changes always get listed**, even if benign. The reviewer decides if they're acceptable.
- **Quote, don't paraphrase**, when reporting the original comment and the author's reply. Paraphrasing loses nuance the reviewer needs.
- **Never post comments without explicit user approval.**
- If the last review was dismissed or the SHA was force-pushed away, say so and ask the user how they want to anchor the re-review (e.g., previous-to-last review, or specific SHA).

## Common mistakes

- Diffing against `origin/main` instead of the last-review SHA — turns a re-review into a fresh review.
- Marking a thread "addressed" because the file was edited, without checking that the edit actually fixes the concern.
- Skipping `isOutdated` threads — GitHub marks them outdated when surrounding code shifts, but the underlying issue often still applies.
- Missing top-level PR comments — author rebuttals/explanations often live there, not in the inline thread.
- Lumping scope-creep changes into the review without flagging them separately.
