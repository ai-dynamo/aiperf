---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Self-Maintenance Routines
---

# Self-Maintenance Routines

AIPerf's recommended cadence is a monthly whole-repository maintenance analysis. Four
Claude Code routines each examine one aspect of repository health that no existing tool
can see, and record what they find to a single **Maintenance backlog** issue. A human
picks items off that backlog; the routine then does the work interactively, with a
person available to answer the judgment calls.

> [!IMPORTANT]
> Scheduled execution is not wired up in this PR. Ops still needs to choose the right
> runner. When a scheduler is added, it should run in **analysis only** mode and be
> unable to commit, push, or open a pull request. Changes happen only when a person
> invokes a routine directly.

## What this covers that existing tooling does not

This is the load-bearing question for the whole system, and the sharpest filter on what
belongs in it.

| Existing tooling | Sees | Structurally blind to |
|---|---|---|
| PR review agents (CodeRabbit) | one diff | patterns spanning files that no single PR touched |
| Nightly CI | whether `main` builds and passes | whether code should exist at all |
| ruff, `check_ergonomics`, finite-invariant ratchets | mechanical rules | anything requiring a judgment call |

These routines occupy the intersection: **whole-repo scope, historical context, and a
judgment call with no pass/fail answer.**

The gap is real rather than manufactured. A diff-scoped reviewer *cannot* find "this
abstraction now exists in five places," because each of the five PRs that added a copy
looked correct on its own. Dead code is the same: a symbol becomes dead relative to the
entire repository and its history, usually several PRs after the one that orphaned it.
Neither a per-diff reviewer nor a pass/fail signal can see either.

The corollary is a hard scope rule the routines enforce on themselves: **a finding that
could have been made from a single diff does not belong here.** "This PR lacks a test" is
the PR reviewer's job, and duplicating it is noise.

## Why monthly analysis instead of a PR stream

The binding constraint is review capacity, not detection capacity.

AIPerf merges roughly 15 PRs a week past 7 code owners, while `src/` grows tens of
thousands of lines a month. There is no shortage of things a routine could propose. A bot
adding five PRs a week would make maintenance a quarter of the review queue — where it
would starve behind feature work, go stale, and train reviewers to ignore the label.

So the system splits the cheap half from the expensive half. Analysis is cheap and runs
on a schedule. Acting on it is expensive and happens only when a human decides an item is
worth their review.

The backlog issue is also the system's **only memory across runs**. Without it, a
scheduled routine re-proposes findings that were already considered and rejected,
indefinitely. A human moving an item to `Declined` is how the system gets tuned, and the
routines treat that section as read-only.

## The routines

| Routine | Skill | Finds |
|---|---|---|
| Dead code | `.agents/skills/maintain-dead-code/` | Unreachable code: orphaned modules, unused private helpers, stale shims, dead branches, unregistered plugin classes |
| Duplicate abstractions | `.agents/skills/maintain-dup-abstractions/` | One concept implemented several times and drifting apart |
| Subsystem coverage gaps | `.agents/skills/maintain-coverage-gaps/` | A category of behavior untested across an entire subsystem |
| Test pruning | `.agents/skills/maintain-test-pruning/` | Tests that cost more than they protect |

When scheduled automation exists, all four should run monthly and sequentially because
they share one backlog issue. The shared contract every routine obeys is
`.agents/skills/self-maintenance/SKILL.md` — read that before changing any routine,
since the individual skills only describe what to look for.

### Dead code

The hard part here is that AIPerf resolves around 178 distinct classes at runtime by
dotted-path string from `src/aiperf/plugin/plugins.yaml` (220 registry entries),
dispatches service methods through the message bus by decorator rather than direct call,
and lazily loads CLI commands from import strings. Generic dead-code detection is
therefore wrong by default here, and acting on a false positive breaks the product at
runtime with no test failure to warn you.

The routine treats every detector hit as a hypothesis and requires four independent
reference checks to come back clean — plus two further signals read as context — before
recording a candidate. The one question it cannot answer is whether something outside
this repository imports the symbol, so every entry carries that question for a human.

### Duplicate abstractions

The signal is not "these look similar" — it is *"a bug fixed in one of these would need
fixing in all of them, and wasn't."* Where a duplicate received a fix its siblings never
did, the routine leads with that, because it converts the item from housekeeping into a
latent bug.

Unification is refused outright when it would require inventing a new abstraction layer,
or when the merged function would need a flag per call site. Both are design decisions.

### Subsystem coverage gaps

Deliberately *not* per-PR test coverage, which the PR review agents already handle at the
moment it is most useful. This routine clusters uncovered lines and reports only
systematic gaps: *"none of the seventeen error paths in the metrics export layer are
covered, and each PR that added one looked fine."*

One important caveat it must respect: the coverage run does not include
`tests/integration/`, a large suite that does run in nightly CI. Code that looks
uncovered frequently is not, so every finding must state that the integration suite was
checked.

### Test pruning

The most conservative routine by design. Every other one's worst case is a rejected
finding; this one's worst case is a silently-removed regression guard. The default answer
is *keep*, deletion requires a proven-zero branch-coverage delta plus evidence from the
introducing commit that the test was not a regression guard, and recording nothing is the
expected outcome most months.

## The two modes

| | Analysis mode | Apply mode |
|---|---|---|
| Triggered by | Manual or future scheduled analysis run | A person running `/maintain-dead-code` etc. |
| Produces | Backlog entries | A branch, commits, and a draft PR |
| Touches the working tree | Never | Yes |
| Judgment calls | Recorded with the question attached | Asked of the human, then acted on |

Apply mode existing is the point. Roughly half of all findings rest on a question a
person answers in thirty seconds — *was this specialization deliberate?* — that a
scheduled run can only guess at, and a wrong guess becomes a rejected PR.

## Operating the routines

### Scheduling

This PR intentionally does not add a scheduler. The recommended cadence is monthly, but
the runner, credentials, and ownership model should be chosen with Ops before any
automation is enabled.

The scheduler should run the routines sequentially, with only the minimum permission
needed to read the existing **Maintenance backlog** issue and update it. It should not
have credentials that allow commits, pushes, or pull requests. It should also include a
mechanical preflight, such as `uv run pytest tests/unit -n auto -q --tb=no`, before
invoking any model-driven analysis; if `main` is red, the maintenance run should abort.

A `maintenance` label must exist wherever the backlog lives, since the backlog issue is
created and found by that label.

### Running one by hand

Until a scheduler exists, evaluate a routine by running it manually in analysis mode and
having it write the proposed backlog additions plus evidence to
`artifacts/maintenance-report.md`. That is the right way to evaluate a routine's
judgment before letting it write anything.

Locally — and this is the normal path for actually making a change:

```
/maintain-dead-code
```

The skills carry no CI-specific assumptions, so a local run behaves the same, except that
you are present to answer questions.

### Permissions

The analysis runner should be read-only for repository contents and should have no
credential capable of pushing branches or opening pull requests. If it updates the
backlog directly, grant only the issue-writing permission required for that operation.
If Ops prefers a report-only first phase, run without issue write access and publish the
report somewhere humans can inspect.

Two limits worth stating plainly. An untrusted *trigger* is not the same as untrusted
*input*: the routine reads merged repository content and history, which includes text
written by outside contributors, so repo content is a prompt-injection surface even when
fork code never runs. Raw model transcripts should not be published as artifacts, since
secret masking covers rendered logs rather than files written to disk and then published.

### Guardrails

Summarized from the shared contract:

- **Scope guards.** Generated files, ratchet baselines, `ATTRIBUTIONS*`, `uv.lock`,
  `CODEOWNERS`, workflows, and the four synchronized agent-instruction files are never
  modified.
- **Baselines burn down, never regenerate.** `tools/ruff_baseline.json`,
  `tools/ergonomics_baseline.json`, and the finite-invariant baselines are ratchets. A
  routine may remove an entry it genuinely fixed; running `--regenerate-baseline` is
  forbidden, because it silently re-grandfathers every violation added since the last
  regeneration.
- **No new dependencies, no public API changes, no behavior changes.** A routine that
  believes it has found a product bug records it and stops.
- **Change budget** (apply mode): at most 400 changed lines, 15 files, one concern per PR.
- **Verification gate** (apply mode): ruff, unit tests, property tests,
  component-integration tests, and `pre-commit run --all-files` must all pass, and the PR
  body must quote the actual output rather than claim it passed.
- **Backlog discipline.** Read it fully before proposing. Never re-propose a `Declined`
  item. Cap `Open` at 20 per routine. Low-confidence findings are dropped, not recorded.
- **Mechanical preflight.** Any scheduled runner should run a cheap test preflight, such
  as `uv run pytest tests/unit -n auto -q --tb=no`, before model-driven analysis. If
  `main` is red, the analysis jobs never start.
- **Abort conditions.** A red `main`, zero High/Medium findings, a scope-guard collision,
  an unexplainable candidate, a finding that a PR reviewer should have made, a full
  backlog, or a routine whose recent items were mostly declined — all mean the routine
  produces nothing. Silence is a valid and frequent outcome.

### Reviewing the output

For a **backlog entry**, the useful question is whether the finding is worth someone's
review time. If it isn't, move it to `Declined` with a one-line reason — that is the
primary tuning mechanism, and the routines are built to respect it permanently.

For a **PR** from apply mode, start at the `Reviewer checklist` in the body. Each item is
a question the routine could not answer for itself, usually about intent or about
consumers outside this repository. For deletions, confirm nothing external depends on the
symbol. For unifications, read the behavior-difference table before the diff — if a
difference is dispositioned as "deliberate" and you disagree, the whole unification is
suspect. Check that the quoted verification output is real and that no baseline file grew.

## Backlog

Routines worth building, roughly in order of value. Not yet implemented. Each is filtered
by the same question: **can an existing tool already see this?**

**High value**

- **Ratchet burndown.** Chip away at `tools/ruff_baseline.json`,
  `tools/ergonomics_baseline.json`, and the finite-invariant baselines a few entries per
  PR until each reaches zero. Mechanical, easy to verify, and it retires debt the
  repository has already agreed is debt. Probably the best-value addition to the current
  set.
- **Doc drift audit.** Verify documentation against the code it describes — CLI options,
  class and function names, env vars, tutorial examples. Partially covered by the
  `markdown-accuracy-auditor` agent, which would become the engine. It could report
  four-file-sync drift, though not fix it; those files are scope-guarded.
- **Flaky test detection.** Find tests that pass under `-n auto` but fail under `-n 0`, or
  fail intermittently across repeated runs. Note that `pytest-rerunfailures` is only a
  declared dev dependency — no `--reruns` is wired into `addopts`, the Makefile, or CI —
  so there is no rerun signal to mine today; enabling one is a prerequisite. Flakes erode
  trust in the whole suite, which makes every apply-mode verification gate less
  meaningful.

**Medium value**

- **Dependency hygiene.** Unused declarations in `pyproject.toml`, dependencies pinned far
  behind, optional-extra groups that no longer match what the code imports. Needs care
  around the license and attribution surface.
- **Error-message ergonomics.** A pass over exception messages and CLI errors asking
  whether each tells a user what to do next. AIPerf users hit config errors constantly,
  and the distance between a good and a bad message is the distance between a five-minute
  and a five-hour debugging session.
- **TODO/FIXME triage.** Classify long-lived inline markers into done-already, still-real
  (file an issue and link it), and never-going-to-happen (delete).
- **Test tier misplacement.** Unit tests that are really component-integration tests, and
  tests missing the `slow` marker. Keeps the default suite fast, which everything else
  depends on.

**Lower value / higher risk**

- **Type-hint completeness.** Public functions missing annotations. Ruff covers part of
  this already.
- **Docstring coverage.** Easy to generate, easy to generate badly. Only worth it for
  public API surface, and only where the docstring says something the signature does not.
- **Import hygiene.** Circular-import risk, deferred imports for startup time. Low yield
  given ruff's existing coverage.

### Deliberately not built

- **Per-PR test-coverage nagging.** The PR review agents already do this, at the right
  moment, with the author present. A second bot in the same thread is noise.
- **Experiment runner / metric-accuracy validation.** Proposed and dropped: nightly CI
  already runs the full integration suite, and the GitLab pipeline it triggers covers
  performance validation. Revisit only if a specific accuracy gap is identified that
  neither covers.
- **Auto-merge on green CI.** The value of this system is that it produces reviewable
  proposals. Removing the reviewer removes the property that makes it acceptable at all.
- **Automatic dependency upgrades.** Dependabot and Renovate solve this well, and neither
  needs a language model.
- **Broad automated style refactoring.** Ruff and the ergonomics checker already enforce
  style mechanically, without judgment calls and without token cost.
