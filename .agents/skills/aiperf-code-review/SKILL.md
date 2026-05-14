---
name: aiperf-code-review
description: Use when the user asks for a code review of an aiperf branch or PR with no prior reviews — "review this branch", "review against main", "review the PR", "audit this branch", "find bugs in this diff", "validate my changes", "sanity check this PR", "is this safe to ship". Diffs vs origin/main (NOT vs a prior review SHA — for that, see aiperf-re-review). Captures findings in artifacts/code-review-<epoch>/REPORT.md, validates every finding against the real code, reproduces confirmed issues via aiperf-mock-server, and drafts inline GitHub PR comments anchored to specific lines.
---

# AIPerf Code Review

Review the current branch (or a named PR's HEAD) against `origin/main`. Carry the task end-to-end: gather findings, validate them against the actual code, reproduce confirmed ones with the real `aiperf` CLI, draft inline PR comments, present for approval before posting.

**Precedence:** if the user said "review" without specifying scope (re-review vs ergonomics vs runtime), invoke **`aiperf-review`** first — it owns the routing decision. Invoke this skill directly only when you've already established a fresh-branch / no-prior-reviews code-review is what's wanted, or when `aiperf-review` dispatched here.

## Hard rules

- **REVIEW ONLY — do not modify code.** If you catch yourself editing `src/` during the pass, stop. Findings go to `REPORT.md`; the user decides whether to apply.
- **Baseline is `origin/main`.** If the PR has prior reviews and the user asked for a delta-since-last-review audit, use `aiperf-re-review` instead.
- **Reproduce with the real CLI**, not pseudo-code. Delegate to `aiperf-mock-server` for the backend — do NOT roll a one-off mock launch.
- **Present before posting.** Show the user the full planned set of inline + summary comments and wait for explicit confirmation before any `gh api .../reviews` write.

## Goals

1. Collect review findings for the branch relative to `origin/main`.
2. Write them into `artifacts/code-review-<epoch>/REPORT.md` as a living document for this invocation.
3. Validate every finding against the actual current code.
4. Assign practical severity to each issue.
5. Reproduce confirmed issues with the real `aiperf` CLI against the in-repo mock server (delegate to `aiperf-mock-server`).
6. Keep runtime receipts under the same `artifacts/code-review-<epoch>/` directory (e.g. `repro/<scenario>/`).
7. Update the living document with both source-level and runtime evidence.
8. Draft inline GitHub PR review comments anchored to the exact file and line of each finding, plus a short top-level summary comment.

## Artifact layout

Compute `EPOCH="$(date +%s)"` once at the start of the invocation; reuse for all paths.

```
artifacts/code-review-<epoch>/
  REPORT.md             # the living document; update in place during the run
  meta.json             # branch, head sha, base sha (origin/main), start time
  repro/<scenario>/     # one subdir per reproduction (commands, logs, profile_export.jsonl, mock-server.log)
```

Same-day re-invocations get fresh `<epoch>` subdirectories — do NOT overwrite a previous run's directory.

## Per-finding record

- **status:** `Confirmed`, `Partially confirmed`, or `Not confirmed`
- **source-level evidence:** exact file paths + line references
- **practical severity and impact**
- **runtime reproduction result** (if reproduced)
- **receipt paths**
- **conclusion**

If a finding is not valid, say so explicitly and explain why. If only partially valid, narrow it precisely.

## Reproduction discipline

- Use the real `aiperf` binary and the in-repo mock server (delegate to `aiperf-mock-server` — do NOT hand-roll the launch).
- Run outside the sandbox when needed and ask for approval through the normal tool flow.
- Save receipts under `artifacts/code-review-<epoch>/repro/<scenario>/` (where `<scenario>` is a short slug naming the finding being reproduced).
- Keep logs, command outputs, relevant generated files, and small summaries that make the proof easy to inspect.
  - If the change touches `src/aiperf/exporters/mlflow_*` or `src/aiperf/post_processors/otel_*`, reproduce against the optional MLflow / OTel path: MLflow export runs in a subprocess (`src/aiperf/exporters/mlflow_export_subprocess.py`); MLflow is an optional dep gated by `aiperf.common.optional_dependencies.mlflow_dependency_message`. Without `mlflow` installed, the export silently degrades — confirm both the present (mlflow installed) and absent (graceful no-op) paths in reproductions.
- Do not overwrite unrelated user changes.

## GitHub deliverable

- Post inline review comments via `gh api repos/{owner}/{repo}/pulls/{number}/reviews`.
- Each confirmed finding gets its own inline comment anchored to the relevant file path and diff line number.
- Include a short top-level summary in the review body covering: fix order, overall assessment, what is working well.
- **Before posting**, show the user the full set of planned comments (inline + summary) and ask for explicit confirmation. Only post after approval.
- To determine the correct diff line position for each inline comment, run `gh api repos/{owner}/{repo}/pulls/{number}/files` to get the patch hunks, then count lines within the patch to find the `position` value.
- After posting, return the PR review URL.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll quickly fix this small finding while I'm here" | REVIEW ONLY. Fixes are the user's call. Stop editing. |
| "The mock server is overkill, I'll just read the code" | The reproduction is the evidence. Source reading alone is "I think this breaks"; reproduction is proof. |
| "Most findings look fine, I'll just summarize" | Validate every finding against the actual code. Cite file:line; do not paraphrase. |
| "I'll post the comments and tell the user after" | Never post without explicit pre-approval. |
| "Diff vs origin/main is the same as diff vs the prior review SHA" | It isn't. For prior-review baseline, use `aiperf-re-review`. |

## Final response to the user

- Concise.
- Where the living document is.
- Where the receipts are.
- The GitHub review URL (after posting).
- Any caveats encountered during reproduction.