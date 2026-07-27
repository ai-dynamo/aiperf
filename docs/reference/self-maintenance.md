---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Self-Maintenance Routines
---

# Self-Maintenance Routines

AIPerf runs a set of scheduled, autonomous maintenance routines. Each routine is a
Claude Code skill that analyzes one narrow aspect of repository health and, when it
finds something it can prove, opens a scoped draft pull request for human review.

This document covers what the routines are, the guarantees they operate under, how to
enable and operate them, and the backlog of routines not yet built.

> [!IMPORTANT]
> Nothing here merges itself. `main` is CODEOWNER-protected; every routine's output is
> a draft PR or an issue. The routines are a source of reviewable proposals, not an
> autonomous committer.

## Why routines rather than ad-hoc cleanup

Repository health decays in ways that are individually too small to prioritize and
collectively expensive: a helper nobody calls anymore, a third copy of the same parsing
logic, a test that has asserted nothing since a refactor two years ago. None of these
justify a ticket. All of them compound.

The routines exist to convert that class of work into a steady trickle of small,
independently reviewable PRs, and to keep each one *cheap to reject*. A maintenance PR
that a reviewer cannot verify in five minutes is a net loss even when it is correct, so
every routine optimizes for reviewability over volume.

## The routines

| Routine | Skill | Schedule | Output |
|---|---|---|---|
| Dead code sweep | `.agents/skills/maint-dead-code/` | Mon 10:00 UTC | `refactor:`/`chore:` deletion PR |
| Duplicate abstractions | `.agents/skills/maint-dup-abstractions/` | Tue 10:00 UTC | `refactor:` unification PR |
| Coverage gaps | `.agents/skills/maint-coverage-gaps/` | Wed 10:00 UTC | `test:` PR adding tests |
| Test pruning | `.agents/skills/maint-test-pruning/` | Thu 10:00 UTC | `test:` deletion PR |
| Experiments | `.agents/skills/maint-experiment/` | Fri 10:00 UTC | Issue; PR only for a proven fix |

The shared contract every routine obeys is `.agents/skills/self-maintenance/SKILL.md`.
Read that file before changing any routine — it holds the guardrails, and the individual
skills only describe what to look for.

### Dead code sweep

Finds unreachable code: unused private helpers, orphaned modules, stale compatibility
shims, dead branches, unregistered plugin classes, and configuration nothing reads.

The hard part in this repository is that roughly 220 classes are resolved at runtime by
dotted-path string from `src/aiperf/plugin/plugins.yaml`, service methods are dispatched
through the message bus by decorator rather than by direct call, and CLI commands are
lazily loaded from import strings. Generic dead-code detection is therefore wrong by
default here, and acting on a false positive breaks the product at runtime with no test
failure to warn you. The routine treats every detector hit as a hypothesis and requires
six independent checks to come back clean before anything is deleted.

### Duplicate abstractions

Finds one concept implemented several times and drifting apart. The signal is not "these
look similar" — it is "a bug fixed in one of these would need fixing in all of them, and
wasn't." Where a duplicate received a fix its siblings never did, the routine says so
explicitly, because that converts a housekeeping PR into a latent-bug fix.

Unification is refused when it would require inventing a new abstraction layer or adding
flags to the merged function for each call site. Both of those are design decisions, and
both get escalated to an issue instead.

### Coverage gaps

Adds tests for behavior that matters: error paths, NaN/Inf boundaries, config
validation, boundary conditions, plugin contracts. Coverage numbers are used only as a
search heuristic — the gate for shipping any test is being able to name, in one
sentence, the bug it catches. Each new test is mutation-checked (introduce the bug,
confirm the test fails, revert) before it ships.

This routine ships test-only changes. If writing a test reveals a product bug, it files
an issue rather than fixing it.

### Test pruning

Removes tests that cost more than they protect: tautologies, mock-only tests, tests for
deleted behavior, permanently-skipped tests, redundant parametrize cases.

This is the most dangerous routine, and it is deliberately the most conservative. Every
other routine's worst case is a rejected PR; this one's worst case is a silently-removed
regression guard. The default answer is *keep*, deletion requires a proven-zero
branch-coverage delta plus evidence from the introducing commit that the test was not a
regression guard, and finding nothing is the expected outcome most weeks.

### Experiments

Runs the real `aiperf` CLI against `tests/aiperf_mock_server` to check that AIPerf
measures what it claims. Unit tests can only verify that a function returns what it
returns; they cannot verify that a reported TTFT of 42 ms corresponds to an actual
42 ms. The mock server's configurable latency, deterministic generation, error
injection, and per-request ISL/OSL recording provide the ground truth to compare
against.

Six rotating families: metric accuracy versus ground truth, run-to-run determinism,
config-space robustness, error-path behavior, load scaling, and cross-endpoint metric
consistency. Findings become issues. A PR is opened only when the root cause is
identified in code, the fix is small and local, it changes no *intended* behavior, and a
regression test that fails before the fix accompanies it.

## Operating the routines

### Enabling

The workflow (`.github/workflows/self-maintenance.yml`) is pinned to
`ai-dynamo/aiperf` and no-ops until an `ANTHROPIC_API_KEY` repository secret exists.
Until then, scheduled runs log a notice and exit cleanly — forks inherit the file
harmlessly.

Two optional pieces of setup materially improve the results:

- **`AIPERF_MAINTENANCE_TOKEN`** (a PAT or GitHub App token). PRs opened with the default
  `GITHUB_TOKEN` do **not** trigger `pull_request` workflows, so they arrive without CI
  signal. Supplying this secret makes the routines' PRs run the normal checks.
- **"Allow GitHub Actions to create and approve pull requests"**, in repository or org
  Actions settings. Without it, `gh pr create` fails and the routine falls back to
  reporting only.

### Running one by hand

Use the `workflow_dispatch` trigger, pick a routine, and leave `dry_run` at its default
of `true` — the run then analyzes, uploads `artifacts/maintenance-report.md` plus the
full log as a workflow artifact, and opens nothing. That is the right way to evaluate a
routine's judgment before letting it file PRs.

Locally, invoke the skill directly in Claude Code:

```
/maint-dead-code
```

The skills carry no CI-specific assumptions, so a local run behaves the same as a
scheduled one.

### Permissions

The workflow requests `contents: write`, `pull-requests: write`, and `issues: write`,
and nothing else. It uses no third-party marketplace actions — the Claude Code CLI is
installed from npm and every action used is one already vetted elsewhere in this
repository — so it remains runnable under an organization action-allowlist policy. It is
never triggered by `pull_request` or `pull_request_target`, so untrusted fork code
cannot reach the API key.

### Guardrails

Summarized from the shared contract:

- **Scope guards.** Generated files, ratchet baselines, `ATTRIBUTIONS*`, `uv.lock`,
  `CODEOWNERS`, workflows, and the four synchronized agent-instruction files are never
  modified by a routine.
- **Baselines burn down, never regenerate.** `tools/ruff_baseline.json`,
  `tools/ergonomics_baseline.json`, and the finite-invariant baselines are ratchets. A
  routine may remove an entry it genuinely fixed; running `--regenerate-baseline` is
  forbidden, because it silently re-grandfathers every violation added since the last
  regeneration.
- **No new dependencies, no public API changes, no behavior changes.** Routines refactor
  and delete. Anything that alters what AIPerf does belongs to a human.
- **Change budget.** At most 400 changed lines, 15 files, and one concern per PR.
  Overflow is deferred to the next run rather than split across simultaneous PRs.
- **Verification gate.** `ruff`, unit tests, property tests, component-integration tests,
  and `pre-commit run --all-files` must all pass, and the PR body must quote the actual
  output rather than claim it passed.
- **Abort conditions.** A red `main`, a failing gate, a scope-guard collision, an
  unexplainable candidate, or zero High/Medium findings all mean the routine opens
  nothing. Silence is a valid outcome, and two consecutive closed-unmerged PRs from the
  same routine escalate to an issue asking a human to retune it rather than producing a
  third.

### Reviewing a routine's PR

Review these the way you would review a contribution from someone competent but
unfamiliar with the codebase's history — because that is exactly what they are.

1. Start at the **Reviewer checklist** in the PR body. Each item is a question the
   routine could not answer for itself, usually about intent or about consumers outside
   this repository.
2. For deletions, confirm nothing external depends on the symbol. That is the one thing
   no routine can verify.
3. For unifications, read the behavior-difference table before the diff. If a difference
   is dispositioned as "deliberate" and you disagree, the whole unification is suspect.
4. Check that the quoted verification output is real, and that no baseline file grew.
5. Close it without ceremony if it is not worth the review. Routines are designed to be
   cheap to reject, and a closed PR is useful signal — the contract escalates after two
   in a row from the same routine.

## Backlog

Routines worth building, roughly in the order the value justifies the effort. Not yet
implemented.

**High value**

- **Doc drift audit.** Verify documentation against the code it describes — CLI options,
  class and function names, env vars, code examples in tutorials. Partially covered
  today by the `markdown-accuracy-auditor` agent, which would become the routine's
  engine. Related: enforcing the four-file sync rule, and catching docs that describe
  behavior that has since changed.
- **Ratchet burndown.** Chip away at `tools/ruff_baseline.json`,
  `tools/ergonomics_baseline.json`, and the finite-invariant baselines a few entries per
  PR until each reaches zero. Highly mechanical, easy to verify, and directly retires
  technical debt the repository has already agreed is debt. Probably the single
  best-value addition to the current set.
- **Flaky test detection.** Mine CI history for reruns (`pytest-rerunfailures` is already
  configured, so the signal exists) and for tests that pass under `-n auto` but fail
  under `-n 0`. Propose a fix, or quarantine with a linked issue. Flakes erode trust in
  the entire suite, which makes every other routine's verification gate less meaningful.
- **Performance regression watch.** Track benchmark timings from the nightly workflow and
  open an issue when a metric regresses beyond a threshold. AIPerf is a performance tool;
  a regression in its own overhead is a product defect.

**Medium value**

- **Dependency hygiene.** Unused declarations in `pyproject.toml`, dependencies pinned
  far behind, and optional-extra groups that no longer match what the code imports.
  Needs care around the license/attribution surface.
- **Error-message ergonomics.** A pass over exception messages and CLI errors asking
  whether each one tells a user what to do next. AIPerf's users hit config errors
  constantly, and the distance between a good and a bad message is the distance between
  a five-minute and a five-hour debugging session.
- **TODO/FIXME triage.** Classify long-lived inline markers into done-already, still-real
  (file an issue and link it), and never-going-to-happen (delete). Prevents the comment
  layer from decaying into noise.
- **Type-hint completeness.** Find public functions missing annotations, per the coding
  standard, and add them where the type is unambiguous from the implementation.
- **Test tier misplacement.** Find unit tests that are really component-integration
  tests, and tests missing the `slow` marker that should have it. Keeps the default
  suite fast, which everything else depends on.

**Lower value / higher risk**

- **Docstring coverage.** Easy to generate, easy to generate badly. Only worth doing for
  public API surface, and only where the docstring says something the signature does not.
- **Config-schema drift.** Check that generated config schemas match the Pydantic models.
  Largely already enforced by `make check-config-schema`.
- **Import hygiene.** Circular-import risk, unused imports beyond what ruff catches,
  imports that should be deferred for startup time. Low yield given ruff's existing
  coverage.
- **Changelog synthesis.** Draft release notes from merged PRs. Useful, but it is a
  release-process routine rather than a maintenance one, and it wants a different
  trigger.

### Deliberately not built

- **Auto-merge on green CI.** The value of this system is that it produces reviewable
  proposals. Removing the reviewer removes the safety property that makes autonomous
  maintenance acceptable at all.
- **Automatic dependency upgrades.** Dependabot and Renovate solve this well, and neither
  needs a language model.
- **Broad automated refactoring for style.** Ruff and the ergonomics checker already
  enforce style mechanically, without judgment calls and without token cost.
- **Anything that changes benchmark semantics.** Metric definitions, timing behavior, and
  wire formats are product decisions. A routine may report a discrepancy; it may not
  resolve one.
