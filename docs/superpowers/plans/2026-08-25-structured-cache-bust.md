# #22 Structured Cache-Bust Closure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish exact merge ancestry and close upstream #22 as already
covered by native Rust cache-bust seams.

**Architecture:** Keep cache-bust policy in native endpoint capability
registration and runtime target/ledger/wire seams. Do not duplicate Python
plugin metadata or add a speculative implementation when behavior is already
represented natively.

**Spec:** `artifacts/archives/origin-main-findings/commit-022-1d1829540b.md`

## Tasks

- [x] Audit all upstream source and test files; confirm nine unit files, one component-integration file, and no E2E tests.
- [x] Compare endpoint capability validation, structured workload propagation, branch/replay inheritance, and wire placement against native Rust.
- [x] Run focused native agentx cache-bust/replay/Weka regressions.
- [x] Complete the exact `--no-ff` merge with upstream as the second parent.
- [x] Perform Graham-style review; no Rust changes are required because the native seams already cover the behavior.
