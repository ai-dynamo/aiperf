---
name: maint-coverage-gaps
description: Autonomous maintenance routine that finds untested behavior in AIPerf that actually matters (error paths, boundary conditions, NaN/Inf handling, config validation, plugin contracts) and opens one PR adding focused tests. Targets risk, not coverage percentage. Use for the scheduled coverage sweep or when asked to write tests for uncovered code.
---

# Coverage Gap Tests

Read `.agents/skills/self-maintenance/SKILL.md` first — its scope guards, verification
gate, change budget, and PR conventions all apply.

## The failure mode this routine must avoid

Coverage-driven test generation reliably produces tests that assert what the code
currently does rather than what it should do. Those tests are worse than no tests: they
pass forever, they break during legitimate refactors, and they encode bugs as
requirements. AIPerf already has 879 test files — the marginal value of test #880 is
negative unless it would **fail if a specific plausible bug were introduced**.

The gate for every proposed test:

> Name the bug this test catches. If you can't state it in one sentence — as a change
> someone might actually make — don't write the test.

Coverage numbers are a search heuristic for finding candidates. They are not the goal,
and no PR from this routine should be justified by a percentage.

## Finding candidates

```bash
uv run pytest tests/unit tests/component_integration -n auto \
  --cov=src/aiperf --cov-branch --cov-report=term-missing --cov-report=html \
  -m 'not performance and not stress and not slow'
```

Note: `-m integration` is not in this run, so integration-only paths look uncovered here
and are not real gaps. Confirm a candidate is not exercised under
`uv run pytest -m integration` before treating it as untested.

Then filter the uncovered lines hard. Ranked by value:

**Worth testing**

1. **Error and failure paths.** `except` bodies, retry exhaustion, timeout handling,
   malformed-response branches. This is where async services break in production and
   where coverage is systematically weakest, because the happy path is easy to exercise.
2. **NaN/Inf boundaries.** Any numeric path where a value could be `nan`, `inf`, or
   `None` crossing a serialization boundary. `CLAUDE.md` makes this a hard contract and
   `tests/unit/property/test_finite_invariants.py` enforces the mechanical half — but
   *semantic* handling (does a single NaN latency poison the percentile? does it
   propagate to JSON export?) needs real tests.
3. **Config validation.** Every `BaseConfig` field validator and cross-field constraint.
   These are cheap to test, high-churn, and user-facing: a broken validator means a
   confusing CLI error, which is the difference between a five-minute and a five-hour
   debugging session for a user.
4. **Boundary conditions.** Zero requests, one request, empty dataset, single-token
   response, concurrency of 1, request count below warmup count.
5. **Plugin contracts.** Each plugin category's base-class contract, tested once against
   a representative implementation. Use `from tests.harness import mock_plugin`.
6. **Message-handler wiring.** That `@on_message(MessageType.X)` handlers exist for every
   published message type and don't silently drop messages.

**Not worth testing**

- Getters, `__repr__`, trivial delegation, Pydantic field defaults with no logic.
- Anything requiring a real inference server. Use `tests/aiperf_mock_server`.
- Anything already covered under `-m integration` or `-m component_integration`.
- Log-only branches.
- Paths whose test would need to mock so much that the test only asserts the mocks.
  If setting up the test requires patching more than about three things, the finding is
  a *testability* problem — open an issue about the seam, don't force the test.

## Writing the tests

Follow `docs/dev/patterns.md` and the testing conventions in `CLAUDE.md`:

- Name: `test_<function>_<scenario>_<expected>`, e.g.
  `test_parse_config_missing_field_raises_error`.
- `@pytest.mark.asyncio` for async, `@pytest.mark.parametrize` for data-driven, with
  `from pytest import param`, explicit `id=`, and `# fmt: skip` on the closing paren.
- Imports at file top. Fixtures for setup. One focus per test.
- The auto-fixtures apply: `asyncio.sleep` is instant, RNG is seeded to 42, singletons
  reset between tests. Do not re-implement any of that.
- Place tests next to their existing siblings — mirror the `src/aiperf/` layout under
  `tests/unit/`. Do not create a new top-level test directory.
- Mark anything ≥3s as `@pytest.mark.slow` and anything multi-service as
  `component_integration`, or the routine will slow the default suite for everyone.

## Self-check: does the test actually work?

Before shipping, prove each new test has teeth. This step is mandatory and is the main
thing separating this routine from a coverage-number generator.

1. **Mutate and confirm failure.** Introduce the bug the test is supposed to catch —
   invert a condition, drop an `await`, return early — and confirm the new test fails.
   Revert the mutation. If the test still passed, it asserts nothing; delete it.
2. **Run in isolation and without xdist**, to catch order-dependence and shared state:
   ```bash
   uv run pytest <new_test_file> -n 0 -v
   uv run pytest <new_test_file>::<test_name> -n 0 -v
   ```
3. **Run the full unit suite twice.** A test that passes once and fails on repeat is
   leaking state.
4. **Check runtime.** New unit tests should add well under a second each. If one is
   slower, mark it `slow` or reconsider it.

## Shipping

- One PR covers **one area** — one module, or one concern like "config validation error
  paths". Not "tests for everything uncovered this week".
- Title: `test: cover <specific behavior>`.
- Test-only PRs, always. If writing the test reveals a bug in the source, **do not fix
  it here** — that is a behavior change and belongs to a human. Open an issue, link it
  from the PR, and either mark the test `xfail` with the issue link or leave it out.
  A maintenance routine quietly changing product behavior under a `test:` title is
  exactly the trust violation that gets the whole system turned off.
- The PR body's findings table has an extra column: **the bug each test catches**, and
  confirmation that the mutation check failed as expected.
- If no candidate survives the "name the bug" gate, open nothing.
