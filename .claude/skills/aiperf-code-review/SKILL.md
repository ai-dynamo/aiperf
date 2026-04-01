 Review the current branch against `origin/main`, then carry the whole task through end-to-end without stopping at analysis.

  Goals:
  1. Collect the review findings for the branch relative to `origin/main`.
  2. Write them into `artifacts/code-review.md` as a living document.
  3. Validate every finding against the actual current code.
  4. Assign practical severity to each issue.
  5. Reproduce the confirmed issues with the real `aiperf` CLI against the in-repo mock server.
  6. Keep runtime receipts under `artifacts/`.
  7. Update the living document with both source-level and runtime evidence.
  8. Draft the clearest possible top-level GitHub PR comment and post it on the correct PR.

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
  - Do not stop after gathering evidence; finish by updating the document and posting the GitHub comment.

  GitHub deliverable:
  - Post a single top-level PR comment that is clear to the PR authors.
  - The comment should summarize:
    - whether each finding is real
    - severity
    - the key source-level reasoning
    - the key runtime evidence
    - recommended fix order
  - After posting, return the PR comment URL.

  Final response to me:
  - Keep it concise.
  - Tell me where the living document is.
  - Tell me where the receipts are.
  - Tell me the GitHub comment URL.
  - Mention any caveats encountered during reproduction.