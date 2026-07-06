---
name: aiperf-code-review
description: Clone the PR under review into a temp directory (or, on explicit request, review uncommitted/unpushed code in-place in the current workspace), review it against its merge-base with origin/main, capture findings in artifacts/pr-<number>/code-review.md as a living document, validate every finding against the actual code, reproduce confirmed issues with the aiperf CLI against the in-repo mock server, and draft inline GitHub PR review comments anchored to specific files and lines. Use when the user asks for a branch or PR code review.
---

 Review a PR (or the current branch's PR) against its merge-base with `origin/main`, then carry the whole task through end-to-end without stopping at analysis.

  Setup — ALWAYS review from a fresh temp-directory clone:
  - Resolve the PR under review: use the PR number the user gave; otherwise run `gh pr view --json number,headRefName,url` on the current branch.
  - Derive the PR identifier `<pr-id>` as `pr-<number>` (e.g. `pr-482`). If no PR exists yet, fall back to `branch-<slug>` where `<slug>` is the branch name with every character outside `[A-Za-z0-9._-]` (including `/`) replaced by `-` (e.g. `ajc/new-feature` -> `branch-ajc-new-feature`). `<pr-id>` is interpolated into filesystem paths and must never contain a path separator.
  - Clone into a temp directory — never review, build, or reproduce in the user's working tree:
    - `tmp=$(mktemp -d "${TMPDIR:-/tmp}/aiperf-review-<pr-id>-XXXXXX")` (respects macOS's per-user `$TMPDIR`; works with GNU and BSD mktemp)
    - Clone the repo into `"$tmp/aiperf"` (a local clone of the invoking repo followed by `git fetch origin` is fine for speed).
    - Check out the PR head inside the clone: `gh pr checkout <number>`, or `git fetch origin "pull/<number>/head" && git checkout FETCH_HEAD`.
    - Set up the clone's environment with `make first-time-setup` before any runtime reproduction.
  - All code reading, diffing, and reproduction happens inside the clone. The user's working tree stays untouched. Never skip the clone on your own judgment — even if the current working tree already has the PR branch checked out. The only exception is the explicit in-place mode below.

  In-place mode — reviewing uncommitted/unpushed code, explicit user request only:
  - Purpose: review work that exists only in the invoking workspace — uncommitted changes and/or local commits not pushed to any remote. A temp clone cannot see this code, so the clone is skipped.
  - Applies only when the user explicitly asks to review the current workspace (wording like "in place", "my working tree", "my uncommitted/unpushed changes", "don't clone"). Convenience, speed, or "the branch is already checked out" are never triggers — if the code under review is pushed, use the clone.
  - Review the invoking working tree exactly as it stands, including uncommitted changes. Do not switch branches, check out refs, or otherwise mutate the tree.
  - Diff basis in this mode: `git diff "$(git merge-base origin/main HEAD)"` — the working tree against the merge-base, so staged, unstaged, and unpushed committed work are all included. Do NOT use `git diff origin/main...` here: the omitted side resolves to HEAD, silently dropping uncommitted changes. Untracked files appear in no diff — check `git status --short` and review new files directly. Runtime reproduction also runs from the working tree.
  - Record in the living document that the review ran in-place: the `git rev-parse HEAD` commit and whether the tree was dirty.
  - `<pr-id>` is `pr-<number>` if the branch has an open PR, otherwise the branch slug. Artifact paths are unchanged — everything still goes under `artifacts/<pr-id>/`.
  - The GitHub deliverable does not apply to code that is not pushed: inline comments cannot anchor to uncommitted/unpushed lines. In this mode the deliverable is the living document; only draft PR comments if the user asks and all reviewed changes are actually on the PR head.

  Artifact locations — everything is scoped to a per-PR subdirectory:
  - The artifact root is `<invoking-repo-root>/artifacts/<pr-id>/` in the repo where the skill was invoked, NOT inside the temp clone, so results survive clone cleanup.
  - Living document: `artifacts/<pr-id>/code-review.md`.
  - Runtime receipts: `artifacts/<pr-id>/repro-runtime-YYYYMMDD/`.
  - Never write review artifacts loose at the top level of `artifacts/`.

  Goals:
  1. Collect the review findings for the PR relative to its merge-base with `origin/main` (`git diff origin/main...HEAD`, matching the diff GitHub shows on the PR).
  2. Write them into `artifacts/<pr-id>/code-review.md` as a living document.
  3. Validate every finding against the actual code in the clone.
  4. Assign practical severity to each issue.
  5. Reproduce the confirmed issues with the real `aiperf` CLI against the in-repo mock server, run from the clone.
  6. Keep runtime receipts under `artifacts/<pr-id>/`.
  7. Update the living document with both source-level and runtime evidence.
  8. Draft inline GitHub PR review comments anchored to the exact file and line of each finding, plus a short top-level summary comment.

  Requirements:
  - Treat `artifacts/<pr-id>/code-review.md` as a living document. Update it in place if it already exists; re-reviews of the same PR reuse the same subdirectory.
  - For each finding, record:
    - status: `Confirmed`, `Partially confirmed`, or `Not confirmed`
    - source-level evidence with exact file paths and line references
    - practical severity and impact
    - runtime reproduction result, if reproduced
    - receipt paths
    - conclusion
  - Use the real codebase, not assumptions.
  - If a finding is not valid, say so explicitly and explain why.
  - If a finding is only partially valid, narrow it precisely.
  - Reproduce with the real `aiperf` binary and the in-repo mock server on a random localhost port, both running from the review checkout (the temp clone, or the working tree in in-place mode).
  - Run outside the sandbox when needed and ask for approval through the normal tool flow.
  - Save all receipts under `artifacts/<pr-id>/repro-runtime-YYYYMMDD/` in the invoking repo, copying them into place if they were produced elsewhere (e.g. inside the temp clone).
  - Keep logs, command outputs, relevant generated files, and small summaries that make the proof easy to inspect.
  - If MLflow reproduction is needed, use a local SQLite MLflow backend so unrelated MLflow filesystem-store issues do not pollute the validation.
  - Do not overwrite unrelated user changes.
  - Do not stop after gathering evidence; finish by updating the document, then present the planned GitHub comments to the user for confirmation before posting.

  GitHub deliverable (clone-mode reviews of pushed code only — in in-place mode the living document is the deliverable, see In-place mode above):
  - Post inline review comments using the GitHub PR review API (`gh api repos/{owner}/{repo}/pulls/{number}/reviews`).
  - Each confirmed finding gets its own inline comment anchored to the relevant file path and diff line number.
  - Include a short top-level summary in the review body covering: fix order, overall assessment, and what is working well.
  - IMPORTANT: Before posting anything to GitHub, show the user the full set of planned comments (inline + summary) and ask for explicit confirmation. Only post after the user approves.
  - To determine the correct diff line position for each inline comment, run `gh api repos/{owner}/{repo}/pulls/{number}/files` to get the patch hunks, then count lines within the patch to find the `position` value.
  - After posting, return the PR review URL.

  Final response to me:
  - Keep it concise.
  - Tell me where the living document is.
  - Tell me where the receipts are.
  - Tell me the temp clone path (it can be deleted once the review is accepted).
  - Tell me the GitHub review URL.
  - Mention any caveats encountered during reproduction.
