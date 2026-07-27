---
name: self-maintenance
description: Shared contract for AIPerf autonomous maintenance routines (dead code, duplicate abstractions, subsystem coverage gaps, test pruning). Defines the two run modes, the maintenance backlog, the guardrails, the verification gate, PR conventions, and abort rules that every maint-* skill must follow. Read this before running any maint-* routine; do not invoke it standalone.
disable-model-invocation: true
---

# AIPerf Self-Maintenance Contract

Shared rules for every `maint-*` routine. Each routine skill covers *what* to look
for; this file covers *how to land it safely*. When the two conflict, this file wins.

## What these routines are for — and what they are not

AIPerf already has good automated coverage. Understanding where it stops is the entire
justification for this system, and it is also the sharpest filter on what belongs here.

| Existing tooling | Sees | Structurally blind to |
|---|---|---|
| PR review agents (CodeRabbit) | one diff | patterns spanning files that no single PR touched |
| Nightly CI | whether `main` builds and passes | whether code should exist at all |
| ruff, `check_ergonomics`, finite-invariant ratchets | mechanical rules | anything requiring a judgment call |

These routines occupy the intersection: **whole-repo scope, historical context, and a
judgment call with no pass/fail answer.**

That is a real gap, not a manufactured one. A diff-scoped reviewer cannot find "this
abstraction now exists in five places," because each of the five PRs that added a copy
looked correct on its own. The same is true of dead code: a symbol becomes dead relative
to the entire repository and its history, usually several PRs after the one that orphaned
it. Nothing that looks at one diff, or at a pass/fail signal, can see either.

The corollary is a hard scope rule: **if a finding could have been made by looking at a
single diff, it does not belong to these routines.** "This PR lacks a test" is the PR
reviewer's job. Do not duplicate it. Report only what requires the whole-repo view.

## The prime directive

> Review capacity is the scarce resource, not detection capacity.

AIPerf merges roughly 15 PRs a week past 7 code owners, during a period when `src/` is
growing tens of thousands of lines a month. There is no shortage of things a routine
*could* propose. There is a hard limit on what a human will actually read. A maintenance
PR that a reviewer cannot verify in five minutes is a net loss even when it is correct,
and a stream of them is worse than none — they starve behind feature work, go stale, and
train reviewers to ignore the label.

Every design decision below follows from that: analysis is cheap and runs often, changes
are expensive and are proposed rarely.

## Two modes

Each routine runs in one of two modes. **Which mode you are in determines what you are
allowed to produce.** If the invocation does not make the mode explicit, you are in
analysis mode.

### Analysis mode — scheduled, unattended

- **Produces findings. Never a PR. Never a commit. Never a working-tree change.**
- Appends to the maintenance backlog (see below), or comments on a PR when the routine
  is PR-triggered.
- Runs read-only. If a routine needs to modify a file to evaluate a finding, that
  finding is by definition not verifiable in analysis mode — record it as a candidate
  and stop.
- Nobody is present to answer a question, so any finding resting on a judgment call gets
  *recorded with the question attached*, not resolved by guessing.

### Apply mode — human-invoked, interactive

- Triggered when a person runs the skill directly (`/maint-dead-code`), normally after
  picking an item off the backlog.
- Produces a real branch, real commits, and a PR.
- A human is present. **Ask them the judgment questions.** This is the whole reason this
  mode exists — one 30-second answer resolves ambiguity that would otherwise become a
  rejected PR.
- Do one backlog item at a time.

## The maintenance backlog

A single long-lived GitHub issue titled **"Maintenance backlog"**, labelled
`maintenance`. It is the system's only memory across runs, and it exists because
scheduled runs are otherwise blind: without it, a routine re-proposes findings that were
already considered and rejected, forever.

Structure:

```markdown
## Open — <routine name>
- [ ] `MAINT-<routine>-<NNN>` <one-line finding> — confidence, evidence pointer

## Declined
- `MAINT-dead-code-004` <finding> — declined <date>: <reason>
```

Rules, in priority order:

1. **Read the whole issue before proposing anything.** Every run, without exception.
2. **Never re-propose anything under `Declined`.** If new evidence genuinely overturns a
   declined finding, say so explicitly and reference the original ID — do not quietly
   re-file it under a new one.
3. **Never re-propose anything already open.** Update the existing entry if the evidence
   changed; do not duplicate it.
4. **Cap `Open` at 20 items per routine.** Past that, the backlog is not being worked and
   adding to it is noise. Say so in the run summary and stop proposing.
5. **Prune.** If an open item no longer reproduces (the code changed, someone fixed it),
   remove it and note why.
6. **Never edit `Declined` except to add to it.** That section is a human's decision
   record, not yours.

A human moving an item to `Declined` is how this system gets tuned. Respect it
absolutely.

## Scope guards — files that are never touched

Refuse to modify these, no matter what a routine's analysis suggests. If a finding
requires touching one, downgrade it to an issue and explain why.

| Path | Why |
|---|---|
| `docs/cli-options.md`, `docs/environment-variables.md` | Generated by `make generate-all-docs` |
| `src/aiperf/plugin/enums.py`, the `get_class()` overload block in `src/aiperf/plugin/plugins.py`, `src/aiperf/plugin/schema/*.schema.json` | Generated by `make generate-all-plugin-files` |
| `src/aiperf/config/schema/aiperf-config.schema.json` | Generated by `make generate-config-schema`, checked by `make check-config-schema` |
| `tools/ruff_baseline.json`, `tools/ergonomics_baseline.json`, `tests/unit/property/*baseline*` | Ratchet files — see "Baselines" below |
| `ATTRIBUTIONS*.md`, `uv.lock`, `pyproject.toml` version field | License/release surface, human-owned |
| `.github/CODEOWNERS`, `.github/workflows/**` | Repo governance; the only exception is a routine editing its own workflow, which it may not do autonomously |
| `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc` | Four-file sync rule — coordinated human edits only |

Additional hard limits:

- **No new dependencies.** Not in `pyproject.toml`, not transitively. A routine that
  needs a new tool asks for it in an issue.
- **No public API changes.** Anything importable from `aiperf.*` that a benchmark
  script or plugin author could reasonably import is off-limits without a human.
- **No behavior changes.** Maintenance routines are refactors and deletions. If a
  change alters what AIPerf *does* — metric values, CLI output, wire format,
  timing — it is a feature PR and belongs to a human. No exceptions: a routine that
  believes it has found a product bug records it and stops.

## Baselines: burndown only, never regeneration

`tools/ruff_baseline.json`, `tools/ergonomics_baseline.json`, and the finite-invariant
baselines under `tests/unit/property/` are **ratchets**: they record grandfathered debt
so CI can reject new violations. Running `--regenerate-baseline` to make a check pass
is the single most damaging thing a routine can do, because it silently re-grandfathers
every violation introduced since the last regeneration.

Rules:

- A routine may **shrink** a baseline (remove entries it genuinely fixed).
- A routine may **never** run `--regenerate-baseline`, and may never add entries.
- If a change makes `make check-ruff-baselined` or `make check-ergonomics` fail with a
  *new* violation, fix the code. Do not touch the baseline.
- When a routine shrinks a baseline, the PR body must list the removed keys explicitly
  so the reviewer can confirm each one corresponds to a real fix in the diff.

## Dynamic-reference hazard

AIPerf resolves a large fraction of its own code at runtime, through strings. Static
"is this referenced?" analysis is **wrong by default** here. Before concluding that
any symbol is unused, check every one of these:

- `src/aiperf/plugin/plugins.yaml` — 220 registry entries naming ~178 distinct classes
  by dotted-path string.
- `@on_message`, `@on_command`, `@on_request`, `@on_pull_message`, `@on_init`,
  `@on_start`, `@on_stop`, `@background_task` — handlers are invoked by the message
  bus, never called directly.
- `cli.py` lazy command loading — CLI commands are import strings, not imports.
- Pydantic `@field_validator` / `@model_validator` / `model_config` hooks.
- `getattr`, `importlib`, `__init_subclass__`, registry decorators, entry points.
- Metric classes discovered by tag/registry rather than by import.
- Anything named in a YAML/JSON config file, a test fixture, or a doc example.

The mechanical check, for any candidate symbol `Foo`:

```bash
# dotted-path and bare-name references anywhere in the repo, including non-Python
grep -rn "Foo" --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.json' --include='*.md' src/ tests/ docs/ tools/ | grep -v "def Foo\|class Foo"
```

If that returns anything you cannot explain, the symbol is not dead.

## Verification gate

**Apply mode only** — analysis mode changes nothing, so there is nothing to verify.

No routine opens a PR until all of these pass locally, in this order. Capture the
output — the PR body must quote it.

```bash
uv run ruff format . && uv run ruff check --fix .
uv run pytest tests/unit -n auto
uv run pytest tests/unit/property -n auto
uv run pytest -m component_integration -n auto
uv run pre-commit run --all-files
```

Rules for the gate:

- If `main` itself is red before the routine starts, **abort and open nothing**. Say so
  in the run summary. Maintenance PRs stacked on a broken base are unreviewable.
- Never `-x`, never `-k` to dodge a failure, never add `@pytest.mark.skip` to make the
  gate pass. A failing gate means the change is wrong.
- Test-touching routines (`maint-coverage-gaps`, `maint-test-pruning`) additionally run
  the affected tests in isolation *and* under `-n 0` to catch xdist-order dependence.

## Change budget

Per PR, hard caps:

| Metric | Cap |
|---|---|
| Changed lines (excluding pure deletions of whole files) | 400 |
| Changed files | 15 |
| Distinct concerns | 1 |

If a routine's findings exceed the budget, it ships the highest-confidence subset and
lists the remainder in the PR body under "Deferred to a follow-up run". It does not
split one concern across simultaneous PRs — the next scheduled run picks up the rest.

## Confidence tiers

Every finding gets a tier. The tier decides what happens to it in each mode.

| Tier | Meaning | Analysis mode | Apply mode |
|---|---|---|---|
| **High** | Mechanically provable. No dynamic-reference explanation needed, or one that was checked and cleared. | Backlog, `Open` | Eligible for a PR |
| **Medium** | Very likely correct, but rests on a judgment call (e.g. "these *should* be unified, but the call sites may have diverged intentionally"). | Backlog, `Open`, **with the question stated explicitly** | Ask the human the question, then proceed on their answer |
| **Low** | Suspicious but unproven. | Dropped. Not recorded. | Dropped |

Two rules that follow:

- **Low-tier findings are not backlog items.** A backlog full of maybes is a backlog
  nobody reads. Drop them.
- **A PR containing a single Low-tier change is a failed run.** Downgrade or drop it.

## Git and PR conventions

```bash
git checkout -b claude-maint/<routine>-<YYYYMMDD>   # e.g. claude-maint/maint-dead-code-20260727
git commit -s -m "<type>: <subject>"                # -s is REQUIRED, DCO is enforced
git push -u origin HEAD                             # gh pr create fails without this
```

- **Branch**: `claude-maint/<routine>-<YYYYMMDD>`, where `<routine>` is the full skill
  name (`maint-dead-code`, not `dead-code`). Never commit to `main` — the
  `no-commit-to-branch` pre-commit hook enforces this locally, but it is not installed
  in CI, so do not rely on it.
- **Push before opening the PR.** `gh pr create` fails non-interactively on an unpushed
  branch.
- **Sign-off**: `git commit -s` always. The DCO check requires a `Signed-off-by` trailer
  on every commit from a non-member author, which includes the maintenance bot.
- **PR title**: conventional-commit type from the enforced set
  (`feat|fix|docs|test|ci|refactor|perf|chore|revert|style|build`), validated by
  `.github/workflows/lint-pr-title.yaml`. Maintenance routines almost always use
  `refactor:`, `test:`, or `chore:`.
- **One PR = one concern**, per `CLAUDE.md`.
- **Documentation is not optional.** If a routine changes anything user-facing, the
  documentation table in `CLAUDE.md` applies to it exactly as it applies to a human.
  Any new file under `docs/` must also be added to `docs/index.yml`.

### Backlog entry template — analysis mode

Keep entries to a few lines. The backlog is a queue, not a report; detail belongs in the
PR that eventually acts on the item.

```markdown
- [ ] `MAINT-dead-code-007` `aiperf.foo.bar.unused_helper` appears unreferenced
      — High. No hits in src/tests/docs/plugins.yaml; last touched 2025-03 (#812).
      Question for a human: does any downstream NVIDIA repo import this?
```

Every entry carries: a stable ID, a one-line finding, the tier, a compressed evidence
pointer, and — for Medium — the explicit question a human needs to answer.

### PR body template — apply mode

Every maintenance PR uses this structure. The `Reviewer checklist` section is the
point of the whole exercise — it tells a human exactly what to spot-check.

```markdown
## What this is

Maintenance change from backlog item `MAINT-<routine>-<NNN>`
(see docs/reference/self-maintenance.md). **Not auto-merged** — needs CODEOWNER review.

## Findings shipped

| # | Change | Confidence | Evidence |
|---|--------|-----------|----------|
| 1 | Removed `aiperf.foo.bar.unused_helper` | High | No references in src/tests/docs/plugins.yaml; see command output below |

## Why each is safe

<per-finding: the dynamic-reference checks that were run and what they returned>

## How this was verified

<actual pasted output of the verification gate — not a claim that it passed>

## Reviewer checklist

- [ ] <the specific question the reviewer should answer, per finding>

## Deferred to a follow-up run

<findings that exceeded the change budget, or landed as Low confidence>
```

## Abort conditions

Stop, produce nothing, and report why:

1. `main` is red before the run starts.
2. The routine finds zero High/Medium findings. **Recording a filler item to look busy
   is worse than recording nothing.** Silence is a valid and frequent outcome.
3. The finding would require touching a scope-guarded path.
4. The routine cannot explain *why* a candidate is safe — only that it looks safe.
5. The finding could have been made from a single diff (see the scope rule at the top).
   That belongs to the PR reviewer, not here.
6. The routine's `Open` backlog section is already at 20 items. The backlog is not being
   worked; adding to it is noise.
7. Three or more of a routine's last five backlog items were moved to `Declined`. Its
   judgment is miscalibrated for this repo. Say so in the run summary and propose
   nothing further until a human retunes the skill.

Apply mode adds one more: the verification gate fails and the fix is not obvious within
the routine's scope. Abandon the branch; do not force it.

## Escalation

When a routine finds something real but out of scope — a bug, a security concern, a
design problem — it opens an issue with the `maintenance` label rather than trying to
fix it. Issues are cheap; a wrong autonomous fix to a real bug is not.
