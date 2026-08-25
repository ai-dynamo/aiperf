# #43 Cache-Bust Help Link Closure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close upstream #43 with exact merge ancestry while preserving the
native disposition that the change does not alter Rust runtime behavior.

**Architecture:** Keep the upstream merge commit limited to the applicable
documentation/help-text and review-skill delta. Record closure evidence
separately so the ancestry remains auditable.

**Spec:** `docs/origin-main-findings/commit-043-6ed4823d12.md`

## Tasks

- [x] Inspect upstream `6ed4823d12` and confirm the change set is limited to the
  cache-bust help text, regenerated CLI docs, and the local review-skill guard.
- [x] Compare the diff against this branch and record that runtime behavior is
  unchanged even though the shared docs/review tree needs the same update.
- [ ] Commit the closure evidence on the target branch before merging upstream.
- [ ] Complete the exact two-parent `--no-ff` merge with upstream as the second
  parent.
- [ ] Run merge verification and perform a Graham-style review; the expected
  outcome is no findings.
