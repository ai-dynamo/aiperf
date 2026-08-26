# #24 BranchStats Test Expectations Closure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve exact upstream merge ancestry and close the Python-only
BranchStats test correction without inventing a native model port.

**Architecture:** Keep the native `PhaseBranchStats` observer/report seam
separate from Python `BranchStats`. No runtime or Rust test changes are needed.

**Spec:** `artifacts/archives/origin-main-findings/commit-024-f8c8e36533.md`

## Tasks

- [x] Audit the complete upstream commit and confirm one unit-test file, with no integration or E2E tests.
- [x] Compare the changed Python fields with native phase/branch statistics contracts.
- [x] Run the closest native phase regression targets: 12 tests passed.
- [x] Complete the exact `--no-ff` merge with upstream as the second parent.
- [x] Perform Graham-style review; no Rust code or hot path changed.
