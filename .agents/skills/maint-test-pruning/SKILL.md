---
name: maint-test-pruning
description: Autonomous maintenance routine that removes tests that cost more than they protect in AIPerf (tautologies, mock-only tests, redundant parametrize cases, tests for deleted behavior, permanently-skipped tests, slow tests with cheap equivalents) and opens one scoped deletion PR. Deliberately conservative — deleting a test that was catching something is the worst outcome this system can produce. Use for the scheduled test-pruning sweep.
---

# Test Pruning

Read `.agents/skills/self-maintenance/SKILL.md` first — its scope guards, verification
gate, change budget, and PR conventions all apply.

## Posture

This is the most dangerous routine in the set, and it should behave like it. Every other
routine's worst case is a rejected PR. This one's worst case is a silently-removed
regression guard that nobody notices until the regression ships.

So the default answer is **keep**. A test earns deletion by being provably worthless, not
by looking redundant. Asymmetry of harm:

- Keeping a useless test costs a few milliseconds of CI and a little reader confusion.
- Deleting a useful test costs a production bug and the credibility of every other
  routine in this system.

When the two are in tension, keep the test. If a run finds nothing that clears the bar,
that is the expected outcome most weeks.

## What qualifies for deletion

**High confidence — deletable**

1. **Tautologies.** Tests that assert their own setup: constructing a Pydantic model and
   asserting the field values just passed in; asserting `mock.method` was called after
   configuring `mock.method` to be called; `assert True`; tests with no assertion at all.
2. **Mock-only tests.** The system under test is entirely mocked, so the test verifies
   the mock's configuration and nothing about AIPerf. A useful signature: the test would
   still pass if the real implementation were replaced with `pass`. Check this by
   actually stubbing the function body and re-running.
3. **Tests for deleted or unreachable behavior.** The code path they exercise no longer
   exists, usually surviving because they test a shim rather than the real path.
4. **Permanently skipped tests.** `@pytest.mark.skip` with no condition, present for
   more than roughly six months of git history, with no linked issue. These are dead
   weight that also mislead the coverage picture. Prefer deleting over unskipping —
   unskipping is a judgment call about whether the behavior is still wanted, which is a
   human's decision.
   ```bash
   grep -rn "pytest.mark.skip\|pytest.mark.xfail" tests/ | grep -v "skipif"
   git log --oneline -S"<test name>" -- tests/ | tail -3   # when was it skipped?
   ```
5. **Redundant parametrize cases.** Cases that exercise an identical code path with
   cosmetically different inputs — five string lengths that all hit the same branch.
   Trim to the boundary values plus one representative. Prove the redundancy by
   confirming branch coverage is unchanged after trimming; do not eyeball it.
6. **Duplicate tests.** The same assertions in two files after a module move, where one
   copy was never removed.

**Never delete**

- Anything you cannot explain the original intent of. Unclear intent means *you* lack
  context, not that the test lacks value. `git log -S` the test and read the PR that
  added it — a test added alongside a bug fix is a regression guard even if it looks
  trivial today.
- Regression tests referencing an issue or PR number. Their whole job is to look
  redundant.
- Tests under `tests/unit/property/` — these enforce the NaN/Inf and field-validator
  invariants that CI ratchets to zero. They are infrastructure.
- The last remaining test of any module.
- `xfail` tests with a linked issue — those are tracked known bugs.
- Tests in `tests/integration/`, `tests/component_integration/`, or
  `tests/server_metrics/`. Cross-service tests are expensive to write, hard to judge
  from a single file, and are where the real coverage lives. Out of scope entirely.
- Slow tests without a proven-equivalent fast replacement. "Slow" is a reason to mark
  `@pytest.mark.slow`, not a reason to delete.

## The deletion proof

Per candidate:

```bash
# 1. Why did it exist? Read the introducing commit, not just the diff.
git log --oneline --follow -- <test_file>
git log -S"<test_function_name>" --oneline -- tests/

# 2. What does it uniquely cover? Compare branch coverage with and without it.
uv run pytest tests/unit -n auto --cov=src/aiperf --cov-branch --cov-report=json \
  -m 'not performance and not stress and not slow'
# then again with the candidate deselected, and diff the two reports
```

A candidate clears the bar only when:

- Removing it changes **no** covered branch (or the branches it uniquely covers are
  themselves dead code — in which case this is a `maint-dead-code` finding and the test
  goes with the code, in that routine's PR, not this one); **and**
- Its introducing commit shows it was not a regression guard; **and**
- It falls into one of the six High-confidence categories above.

## Shipping

- One PR = one category of pruning. Do not mix "deleted tautologies" with "trimmed
  parametrize cases".
- Title: `test: remove <category>` or `test: trim redundant <area> cases`.
- One commit per test file, so a reviewer can drop individual files.
- The PR body must state, per deleted test: **its original intent**, why that intent no
  longer needs coverage, and the branch-coverage delta (which should be zero).
- Report the CI time saved if it is material. If it is negligible, say so — that is
  honest evidence that the change is about readability, not speed, and reviewers should
  weigh it accordingly.
- `Reviewer checklist`: per test, *"Confirm this was not guarding a past regression."*

## Hard stop

If the coverage comparison shows any branch lost, or if the introducing commit is
ambiguous about intent, **keep the test and drop the finding**. Do not compensate by
writing a replacement test in the same PR — that is `maint-coverage-gaps`' job and mixing
them makes both unreviewable.
