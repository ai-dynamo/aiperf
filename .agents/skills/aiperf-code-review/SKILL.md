---
name: aiperf-code-review
description: Review the current branch against origin/main, capture findings in artifacts/code-review.md as a living document, validate every finding against the actual code, reproduce confirmed issues with the aiperf CLI against the in-repo mock server, and draft inline GitHub PR review comments anchored to specific files and lines. Use when the user asks for a branch or PR code review.
---

 Review the current branch against `origin/main`, then carry the whole task through end-to-end without stopping at analysis.

  Goals:
  1. Collect the review findings for the branch relative to `origin/main`.
  2. Write them into `artifacts/code-review.md` as a living document.
  3. Validate every finding against the actual current code.
  4. Assign practical severity to each issue.
  5. Reproduce the confirmed issues with the real `aiperf` CLI against the in-repo mock server.
  6. Keep runtime receipts under `artifacts/`.
  7. Update the living document with both source-level and runtime evidence.
  8. Draft inline GitHub PR review comments anchored to the exact file and line of each finding, plus a short top-level summary comment.

  Requirements:
  - Treat `artifacts/code-review.md` as a living document. Update it in place if it already exists.
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
  - Reproduce with the real `aiperf` binary and the in-repo mock server on a random localhost port.
  - Run outside the sandbox when needed and ask for approval through the normal tool flow.
  - Save all receipts under a dedicated directory such as `artifacts/repro-runtime-YYYYMMDD/`.
  - Keep logs, command outputs, relevant generated files, and small summaries that make the proof easy to inspect.
  - If MLflow reproduction is needed, use a local SQLite MLflow backend so unrelated MLflow filesystem-store issues do not pollute the validation.
  - Do not overwrite unrelated user changes.
  - Do not stop after gathering evidence; finish by updating the document, then present the planned GitHub comments to the user for confirmation before posting.

  GitHub deliverable:
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
  - Tell me the GitHub review URL.
  - Mention any caveats encountered during reproduction.