---
name: maint-coverage-gaps
description: Autonomous maintenance routine that finds SYSTEMATIC test-coverage gaps spanning a whole AIPerf subsystem (an entire category of error path, boundary condition, or contract left untested across many files) rather than per-PR omissions. Records findings to the maintenance backlog; writes tests only when a human invokes it on a backlog item. Use for the scheduled coverage sweep or when asked about subsystem-level test gaps.
---

# Subsystem Coverage Gaps

Read `.agents/skills/self-maintenance/SKILL.md` first — the two run modes, the backlog
rules, scope guards, verification gate, and PR conventions all apply.

## The scope line, and why it matters here

"This PR doesn't have a test" is the PR reviewer's job. CodeRabbit already does it, at
the moment it is most useful, with the author present. **Do not duplicate it.** A second
bot saying the same thing later is pure noise.

This routine looks for the thing a diff-scoped reviewer structurally cannot see: a
**category of behavior that is untested across an entire subsystem**. Not "this function
lacks a test" — *"none of the seventeen error paths in the metrics export layer are
covered, and each individual PR that added one looked fine."*

The distinction is concrete. Ask: could this finding have been made by looking at one
PR? If yes, drop it. A finding belongs here only when the pattern is the point and the
individual instances are unremarkable.

## What a real finding looks like

Good — systematic, spans files, the pattern is the insight:

- "Every `except` branch in the response-parsing path is uncovered; 12 handlers across
  6 files, zero tests exercise any of them."
- "No test anywhere passes a `nan` or `inf` through the metric aggregation path, despite
  the NaN/Inf contract in `CLAUDE.md` making that a documented invariant."
- "Config validators: 34 `@field_validator` functions across the config package, 9 have
  any test at all. The untested ones cluster in the endpoint configs."
- "Plugin category X has 14 registered implementations and the base-class contract is
  tested against exactly one of them."
- "Boundary conditions: no test in the repo runs with `--request-count 1`."

Not a finding — belongs to the PR reviewer, or to nobody:

- "`foo()` added last week has no test."
- Coverage percentage moved down.
- Getters, `__repr__`, trivial delegation, log-only branches.
- Anything already covered under `-m integration` (see the caveat below).

## Finding candidates

```bash
uv run pytest tests/unit tests/component_integration -n auto \
  --cov=src/aiperf --cov-branch --cov-report=term-missing --cov-report=json \
  -m 'not performance and not stress and not slow'
```

**Critical caveat:** this does not run `tests/integration/`, which is a large suite
(~55 files) covering endpoints, exporters, tracing, cancellation, telemetry, and more,
and which does run in nightly CI. Code that looks uncovered here is frequently covered
there. **Before recording any gap, grep `tests/integration/` for the subsystem** and say
in the backlog entry that you did. Reporting an integration-covered path as a gap is the
fastest way to get this routine's findings dismissed wholesale.

The analysis that matters is not reading the coverage report line by line — it is
**clustering** the uncovered lines. Group them by subsystem and by *kind* of behavior
(error handling, validation, boundary, contract). A cluster with a shared cause is a
finding. Twenty scattered uncovered lines are not.

Ranked by value once clustered:

1. **Error and failure paths.** Exception bodies, retry exhaustion, timeout handling,
   malformed-response branches. Systematically weakest, because happy paths are easy to
   exercise, and this is where async services actually break.
2. **NaN/Inf boundaries.** `CLAUDE.md` makes this a hard contract and
   `tests/unit/property/test_finite_invariants.py` enforces the mechanical half —
   semantic handling is much thinner.
3. **Config validation.** Cheap to test, high-churn, user-facing: a broken validator is
   the difference between a five-minute and a five-hour debugging session.
4. **Plugin contracts.** Each category's base-class contract, across implementations.
5. **Boundary conditions.** Zero/one request, empty dataset, single-token response,
   concurrency 1, request count below warmup count.

## The gate for recording a gap

A cluster is only worth recording when all four hold:

1. **It spans multiple files or many call sites.** One file is not a subsystem.
2. **It has a shared cause**, not just a shared directory. "These are all untested
   because error paths were never prioritized" is a cause; "these are all in
   `exporters/`" is not.
3. **It is not covered by `tests/integration/`.** Checked, and said so.
4. **You can name the bug class it would catch.** Not one bug — the *class*. "A silent
   swallow in any of these handlers would ship undetected."

## Apply mode — writing the tests

Only when a human pulls a backlog item. Follow `docs/dev/patterns.md` and the testing
conventions in `CLAUDE.md`:

- Name: `test_<function>_<scenario>_<expected>`.
- `@pytest.mark.parametrize` with `from pytest import param`, explicit `id=`, and
  `# fmt: skip` on the closing paren. Systematic gaps usually want one parametrized test
  over many cases, not many near-identical tests.
- Mirror the `src/aiperf/` layout under `tests/unit/`. Never a new top-level test dir.
- Mark ≥3s as `slow`, multi-service as `component_integration`, or the default suite
  slows for everyone.
- The auto-fixtures already give instant `asyncio.sleep`, seeded RNG, and singleton
  reset. Don't reimplement them.

**Mutation-check every test before shipping.** Introduce the bug it should catch —
invert the condition, drop the `await`, return early — confirm the test fails, revert.
A test that still passes asserts nothing; delete it. Then run in isolation and under
`-n 0` to catch order-dependence, and run the suite twice to catch state leakage.

Ship test-only changes. If writing a test reveals a product bug, **do not fix it** —
record it and tell the human. A routine quietly changing behavior under a `test:` title
is the trust violation that gets this whole system switched off.

PR title: `test: cover <the systematic gap>`. The body names the bug class, not just
the files touched.
