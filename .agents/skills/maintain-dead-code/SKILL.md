---
name: maintain-dead-code
description: Autonomous maintenance routine that finds provably-unreachable code in AIPerf (unused private helpers, orphaned modules, unregistered plugin classes, dead branches, stale compatibility shims). Records findings to the maintenance backlog during analysis runs; opens a scoped deletion PR only when a human invokes it on a backlog item. Accounts for AIPerf's heavy dynamic dispatch via plugins.yaml, message-bus decorators, and lazy CLI imports. Use for manual or future scheduled dead-code analysis, or when asked to clean up dead code.
---

# Dead Code Sweep

Read `.agents/skills/self-maintenance/SKILL.md` first — its scope guards, verification
gate, change budget, and PR conventions all apply. This file only covers what to hunt
and how to prove it.

## The central problem

AIPerf resolves ~178 distinct classes at runtime across 220 registry entries, by
dotted-path string from
`src/aiperf/plugin/plugins.yaml`, dispatches service methods through the ZMQ message bus
via `@on_message`/`@on_command`/`@on_request` decorators, and lazily loads CLI commands
from import strings in `cli.py`. **A generic dead-code tool applied to this repo produces
mostly false positives, and acting on them breaks the product at runtime with no test
failure to warn you.**

So this routine inverts the usual approach: rather than trusting a detector and
spot-checking, it treats every detector hit as a *hypothesis* and requires an
affirmative proof of deadness before anything is deleted.

## Candidate generation

Cheap, high-signal passes. Run them all; each produces hypotheses, not conclusions.
Do not paste raw detector output into the backlog. Summarize it first: group candidates
by mechanism, discard hook/validator/plugin false positives, then qualify only the
strongest few survivors with the deadness proof below.

**1. Unused private symbols.** Leading-underscore functions/methods/classes are not part
of any public or plugin surface, so the reference graph is closed within the repo.

```bash
# for each _private symbol defined in src/aiperf, count references outside its own def
grep -rn "^\s*\(def\|class\) _[a-z]" --include='*.py' src/aiperf/
```

**2. Orphaned modules.** Files under `src/aiperf/` that nothing imports and that
`plugins.yaml` does not name.

```bash
grep -rn "class:" src/aiperf/plugin/plugins.yaml | sed 's/.*class: *//' | sort -u > /tmp/plugin_classes.txt
```

**3. Stale compatibility shims.** Aliases, re-exports, and `# deprecated` markers whose
only remaining references are the shim itself and its own test.

```bash
grep -rn "deprecated\|DeprecationWarning\|backwards compat\|legacy\|BACKCOMPAT" --include='*.py' src/aiperf/
```

**4. Unreachable branches.** `if False`, constant-folded conditions, code after
unconditional `return`/`raise`, `except` clauses for exceptions the `try` body cannot
raise, and platform branches for platforms AIPerf no longer supports. Check platform
branches against `IS_WINDOWS`/`IS_MACOS`/`IS_LINUX` in `aiperf.common.constants` — the
repo now has blocking Windows CI, so Windows-guard removal is almost never correct.

**5. Dead configuration.** Fields on `BaseConfig` subclasses and `AIPERF_*` environment
variables that nothing reads. Cross-check against `docs/environment-variables.md` and
`docs/cli-options.md` (both generated — read them, never edit them).

**6. Unregistered plugin classes.** Classes that implement a plugin base but appear in
no `plugins.yaml` entry. These are usually either a genuine leftover or an unfinished
feature; the git history distinguishes them.

## The deadness proof

Checks 1, 2, 3, and 5 are the gate: a candidate is only deletable when **all four** come
back clean. Checks 4 and 6 are context that informs the judgment but have no pass/fail
state — read the interpretation notes below before weighing them. Record the actual
output for each — the PR body has to quote it.

```bash
SYM="<candidate symbol name>"

# 1. Repo-wide textual references, including non-Python config and docs
grep -rn "\b${SYM}\b" --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.json' --include='*.md' --include='*.toml' src/ tests/ docs/ tools/ notebooks/

# 2. Plugin registry (dotted path or bare class name)
grep -n "${SYM}" src/aiperf/plugin/plugins.yaml

# 3. Dynamic access patterns that defeat grep-by-name
grep -rn "getattr\|importlib\|__subclasses__\|globals()\[\|locals()\[" --include='*.py' src/aiperf/ | grep -i "$(echo $SYM | head -c 8)"

# 4. Provenance — when did it last matter?
git log --oneline -S"${SYM}" -- src/ | head -20

# 5. Is it exported? __all__ and package __init__ re-exports are public surface.
grep -rn "__all__" --include='*.py' src/aiperf/ | grep "${SYM}"

# 6. Coverage — was it executed by any tier of the test suite?
COVERAGE_FILE=.coverage.unit uv run pytest tests/unit -n auto \
  --cov=src/aiperf --cov-report= -m 'not performance and not stress and not slow'
COVERAGE_FILE=.coverage.component uv run pytest tests/component_integration -n auto \
  --cov=src/aiperf --cov-report= -m 'not performance and not stress and not slow'
uv run coverage combine .coverage.unit .coverage.component
uv run coverage report -m
```

Interpreting the results:

- Any hit in (1), (2), (3), or (5) that you cannot affirmatively explain away → **not
  dead**. Drop the candidate. "Probably just a coincidental substring" is not an
  explanation; go read the line.
- (4) is context, not proof. Code added last month that nothing calls is more likely an
  unfinished feature than dead weight — open an issue asking the author, don't delete.
  Code untouched for a year with no references is a strong deletion candidate.
- (6) is *supporting* evidence only. **Uncovered does not mean dead** — much of AIPerf's
  runtime path is exercised only by `-m integration`, which this gate does not run.
  Covered-and-only-by-its-own-test *is* meaningful: a symbol whose sole caller is the
  test written for it is dead code with a dead test, and both go in the same PR.

## What not to delete

- Anything reachable from `plugins.yaml`, even if no test covers it.
- Message handlers, lifecycle hooks, and `@background_task` methods.
- Pydantic validators and `model_config` entries.
- Public exception types — external plugin authors catch them.
- Metric classes. They are discovered by registry and appear in `docs/metrics-reference.md`.
- Windows/macOS platform branches. CI covers all three platforms; a branch that looks
  unreachable on Linux is not.
- Test helpers under `tests/harness/` and `tests/fixtures/` — pruning those is
  `maintain-test-pruning`'s job, and mixing the two violates one-PR-one-concern.

## Output

**Analysis mode** — record each surviving High/Medium candidate as a backlog entry with
its deadness proof compressed to one evidence line, plus the question no static analysis
can answer: *does anything outside this repository import it?* Change nothing.

Group related candidates into a single entry where they share a cause ("the `foo` shim
and its three helpers, orphaned when X landed") — one entry per deletable unit of
review, not one per symbol.

**Apply mode** — take one backlog item and ship it:

- Re-run the deadness proof. The repo has moved since the finding was recorded, and a
  symbol that gained a caller last week is no longer a candidate.
- Order commits by blast radius: whole orphaned files, then module-level symbols, then
  private helpers.
- **One commit per logical deletion group.** This matters more here than in any other
  routine — deletions are exactly where a reviewer is most likely to say "all of these
  but that one," and per-group commits let them drop one without rejecting the PR.
- Delete the symbol's own tests in the same commit as the symbol.
- PR title: `refactor: remove dead <area> code`, or `chore:` if purely deletions.
- `Reviewer checklist` asks, per deletion: *"Confirm nothing outside this repo imports
  `X`."* That is the one thing the routine genuinely cannot check, and it is the only
  question that reliably matters.

If an analysis run yields nothing High-confidence, record nothing. A clean repo is the success
case, not a failure to find work.
