---
name: maint-experiment
description: Autonomous maintenance routine that runs controlled AIPerf experiments against the in-repo mock server to validate that AIPerf measures what it claims (metric accuracy vs known ground truth, determinism, config-space robustness, error-path behavior, load scaling) and reports findings as an issue. Opens a PR only for a proven, narrowly-scoped fix. Use for the scheduled experiment run.
---

# Experiment Runner

Read `.agents/skills/self-maintenance/SKILL.md` first — its scope guards, verification
gate, and PR conventions apply. This routine differs from the others in one key way:
**its default output is an issue, not a PR.** Experiments produce knowledge. Turning
knowledge into a code change is usually a human's call.

## Why this routine exists

AIPerf is a measurement tool. Its unit tests verify that its functions return what its
functions return; they cannot verify that a reported TTFT of 42 ms corresponds to an
actual 42 ms. That check requires running the real binary against a server with known
behavior and comparing.

`tests/aiperf_mock_server` makes this possible: configurable TTFT/ITL, deterministic
hash-based token generation, error injection with reproducible rates, and **per-request
ISL/requested-OSL recording to JSONL**. That recording file is the ground truth oracle —
it is what the server actually saw and did, independent of what AIPerf believes.

The whole routine is: *impose a known truth, measure it with AIPerf, compare.*

## Setup

Always a random free localhost port, always `--fast` unless the experiment is
specifically about latency fidelity, always a scratch artifact directory outside the
repo.

```bash
PORT=$(python -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
OUT=$(mktemp -d)

uv run aiperf-mock-server --port "$PORT" <latency/error flags> \
  --record-requests "$OUT/server-requests.jsonl" &
SERVER_PID=$!
# wait for readiness by polling /v1/models — never a fixed sleep

uv run aiperf profile \
  --url "http://localhost:$PORT" \
  --artifact-dir "$OUT/run-01" \
  <experiment flags>

kill $SERVER_PID
```

Discipline that keeps results trustworthy:

- **Never** point an experiment at a real inference endpoint. No network, no credentials.
- Poll for server readiness; a fixed sleep produces flaky first-request timings that get
  misread as AIPerf bugs.
- Pin every source of variance: fixed seed, fixed ISL/OSL, fixed concurrency, fixed
  request count. An experiment that cannot be re-run to the same numbers cannot support
  a conclusion.
- Record the exact command line and the AIPerf commit SHA in the report. A finding
  without a reproducer is a rumor.
- Keep receipts under `$OUT` and attach the relevant excerpts to the issue. Do not
  commit run artifacts to the repo.

## Experiment families

Run one family per scheduled invocation, rotating. Each is a falsifiable question.

**1. Metric accuracy against ground truth.** *Do AIPerf's numbers match what the server
actually did?* Configure known TTFT and ITL, run, then compare AIPerf's reported
percentiles against the server's recorded per-request timings. Compare AIPerf's
input-sequence-length statistics against the server's recorded ISL, and output tokens
against recorded OSL. Set an explicit tolerance before running — a drift you decide is
acceptable after seeing it is not a finding, it is a rationalization.

**2. Determinism.** *Does the same input produce the same output?* Same seed, same
config, three runs. Request counts, ISL distributions, and dataset content must match
exactly. Latency figures will not; assert on structure and counts, not timings. A
determinism break is high-value — it means benchmark results are not comparable across
runs, which is the tool's core promise.

**3. Config-space robustness.** *Does AIPerf fail cleanly at the edges?* Sweep boundary
values: `--request-count 1`, concurrency 1, ISL/OSL of 1, request count below warmup
count, zero-length dataset, conflicting flags. Every case must either succeed or produce
an actionable error naming the offending option. A traceback, a hang, or a silent empty
report is a finding. Confusing-but-technically-correct errors are also findings — file
them, since a bad error message is a real user cost.

**4. Error-path behavior.** *What happens when the server misbehaves?* Use the mock
server's error injection at several rates. Verify AIPerf's error accounting matches the
injected rate, that failed requests are excluded from latency percentiles rather than
recorded as zeros, that exit codes are sensible, and that the run terminates rather than
hanging. Cross-check against the NaN/Inf contract in `CLAUDE.md`: no `nan` or `inf`
should reach the JSON export. Validate the export against
`docs/reference/json-export-schema.md`.

**5. Load scaling.** *Does AIPerf itself distort the measurement under load?* Sweep
concurrency across roughly an order of magnitude against a `--fast` server. AIPerf's own
overhead should stay flat per request; throughput should scale until the mock server
saturates. A superlinear rise in AIPerf-side overhead points at contention in the worker
or timing-manager path.

**6. Cross-endpoint consistency.** *Do shared metrics mean the same thing everywhere?*
Same logical workload through chat completions, text completions, and streaming vs
non-streaming. Metrics defined identically in `docs/metrics-reference.md` should agree.
Divergence is either a real bug or a documentation gap; both are worth filing.

## From result to finding

A result is only a finding when all four hold:

1. **Reproducible.** Re-run the exact command at least three times. Intermittent means
   *flaky*, which is itself a finding — file it as flakiness, not as the underlying
   claim, because the diagnosis differs.
2. **Not environmental.** Rule out the mock server, machine load, and CI-runner noise
   before blaming AIPerf. Where possible, confirm the mock server's own recording agrees
   with what you asked it to do.
3. **Contradicts a documented promise.** Point at the specific line in
   `docs/metrics-reference.md`, `docs/architecture.md`, or `docs/cli-options.md` that the
   behavior violates. If no doc makes a promise, the finding is *"the docs don't say"* —
   still worth filing, but as a documentation issue.
4. **Quantified.** "TTFT p99 reads 12% high at concurrency 64, ground truth 40 ms,
   reported 44.8 ms, n=3 runs" — not "TTFT looks off".

## Output

**Default: an issue.** Title states the discrepancy and its size. Body contains the
hypothesis, the exact reproducer command, the numbers, the ground-truth comparison, the
doc line contradicted, and the suspected area of code. Label `maintenance`.

**PR only when all of these hold**, and it is a separate change from the experiment
itself:

- The root cause is identified in code, not guessed at.
- The fix is small and local — a wrong constant, an off-by-one in a percentile index, a
  missing `scrub_non_finite` call.
- The fix does not change any *intended* behavior. Correcting a metric that was
  documented as correct and is now measurably wrong is a fix; changing what a metric
  means is a product decision and stays an issue no matter how obvious it seems.
- A regression test accompanies it, and that test fails before the fix. Include the
  before/after output.

Title such a PR `fix: <metric/behavior> <specific defect>`, and link the issue.

If the run finds nothing, say so in the run log and open nothing. A clean experiment is
a real result — record which family ran and what tolerance held, so the next run can
widen the search rather than repeating it.
