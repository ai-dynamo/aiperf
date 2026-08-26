# #23 Authored DAG Branches Closure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish exact merge ancestry and close #23 against the native
Graph-IR/AgentX branch-and-join architecture.

**Architecture:** Treat Python's request-free orchestrator and authored-round
loader as superseded by native graph lowering and execution. Validate behavior
at branch/join/barrier boundaries without adding a parallel Python-shaped
model to Rust.

**Spec:** `artifacts/archives/origin-main-findings/commit-023-fc7bbf3bdd.md`

## Tasks

- [x] Audit all upstream source, unit, integration, component-integration, fixture, and E2E paths.
- [x] Compare request-free stages, branch fan-out, joins, think-time, payload isolation, and accounting with native seams.
- [x] Run native AgentX join-gating parity and chained-diamond barrier/mock regressions.
- [x] Complete the exact `--no-ff` merge with upstream as the second parent.
- [x] Perform Graham-style review; no Rust implementation changes are required.
