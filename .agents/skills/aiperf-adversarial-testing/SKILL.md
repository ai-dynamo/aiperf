---
name: aiperf-adversarial-testing
description: Use when you need to prove an aiperf code change handles hostile / degenerate inputs gracefully — server errors, timeouts, slow tokens, malformed SSE, zero-byte responses, oversized prompts, conflicting CLI flags, mid-run server death. Actually runs the aiperf CLI against the in-repo mock server with fault-injection flags, captures crashes/hangs/error-metric anomalies, and reports each scenario as pass (graceful) / fail (crash, hang, silent data loss, misleading error). Companion to aiperf-correctness-testing — that skill proves the happy path; this one proves the unhappy paths.
---

# AIPerf Adversarial Testing

Run the aiperf CLI against the in-repo mock server with **fault-injected, malformed, or boundary inputs** and prove the CLI behaves: it surfaces errors (not crashes), terminates (not hangs), and reports honest metrics (not silent zeros).

This skill **executes code**. The deliverable is a per-scenario verdict + artifacts on disk, not an analysis document.

## When to use

- Verifying an error-handling change in `worker/`, `inference/`, or the response parsers.
- Before shipping a refactor that touches retry, timeout, or backoff logic.
- Validating that a new endpoint formatter degrades gracefully on the mock's error-injection paths.
- As the runtime arm of a `aiperf-code-review` finding that claims an edge case is mishandled.

## When NOT to use

- Smoke-test for happy-path behavior — use `aiperf-correctness-testing`.
- Unit-level fault injection (mocking individual functions) — that's pytest territory.
- Real-server adversarial tests (kill -9 a real vLLM) — out of scope for this skill; this skill uses the in-repo mock.

## What "graceful" means

A scenario PASSES if all of:

1. `aiperf profile` exits within the per-scenario `TIMEOUT_S` wrapper (never hangs past it). A scenario that intends a short run completes well before the wrapper fires.
2. Exit code is informative: 0 if a partial dataset was salvageable, non-zero if not. Never SIGSEGV / SIGABRT / unhandled traceback that loses the partial.
3. `profile_export.jsonl` exists (even if records are mostly errored), so the error pattern is inspectable.
4. The CLI's terminal output names the failure mode in plain English. `"Connection refused: 127.0.0.1:NNNN"` passes. A raw traceback with no top-level message fails.
5. The error metric columns (`error`, `error_type`, `status_code` — whatever exists in the schema) are populated for failed requests, not silently dropped.

A scenario FAILS if any of:
- the wrapper `timeout` fires (rc 124) — that's a hang, never graceful, regardless of whether per-request `--request-timeout-seconds` was set.
- segfault (rc 134/139), unhandled exception with no top-level error message, missing `profile_export.jsonl`, errored requests counted as successes, or metrics that lie (e.g., `errors=0` while `request_count < expected`).

Note the distinction: per-request `--request-timeout-seconds N` is a feature under test in scenario 3 (slow-ttft) — individual requests timing out and surfacing as errors is the GRACEFUL outcome. The wrapper `timeout ${TIMEOUT_S}s` is a separate, outer guard against the overall run hanging. Scenario 3 passes when per-request timeouts surface as errors AND the run completes before the wrapper fires.

## Pre-flight

Same shape as `aiperf-correctness-testing`, but the artifact dir uses an `adversarial-` prefix:

```bash
WORKDIR="${WORKDIR:-$(pwd)}"
cd "$WORKDIR"
which aiperf >/dev/null || { echo "aiperf not installed; run make first-time-setup" >&2; exit 1; }

EPOCH="$(date +%s)"
ART="$WORKDIR/artifacts/adversarial-${EPOCH}"
mkdir -p "$ART"

git rev-parse --abbrev-ref HEAD > "$ART/branch.txt"
git rev-parse HEAD              > "$ART/head-sha.txt"
```

## Scenario matrix

The 10 scenarios below are a **starter set** representing failure modes seen in this codebase — not a canonical list. Extend per change: a retry-logic refactor probably wants more partial-error variants; a parser change wants more malformed-input shapes. The per-scenario `TIMEOUT_S=90` and per-run request counts (5–200) are budgets, not policy.

Each scenario gets its own mock-server launch (with different flags) and its own aiperf invocation. Don't share a server across scenarios — fault state leaks. Use `aiperf-mock-server` for each launch, passing the per-scenario mock flags. The mock-server skill explicitly documents this exemption: adversarial testing is the one caller that legitimately needs per-scenario launches with distinct flag sets.

Set a hard timeout per scenario with `timeout`:

```bash
TIMEOUT_S=90   # any scenario that intends a 30-60s run must complete in 90s
```

Run each scenario with this shape (the `aiperf-mock-server` skill must be invoked via the `Skill` tool per scenario with the per-scenario flags; bind its returned `MOCK_URL` / `MOCK_PID` / `MOCK_LOG` into shell variables inside this helper before running aiperf):

```bash
run_adv() {
  local name=$1; local mock_flags=$2; shift 2     # remaining args are aiperf flags
  local out="$ART/$name"; mkdir -p "$out"

  # 1. Invoke aiperf-mock-server (via Skill tool) with $mock_flags. Capture the
  #    URL/PID/LOG/FLAGS strings it returns and assign here:
  local MOCK_URL="<from skill output>"
  local MOCK_PID="<from skill output>"
  local MOCK_LOG="<from skill output>"
  export NO_PROXY="127.0.0.1,localhost"

  # 2. Run aiperf with the outer timeout wrapper.
  ( cd "$out" && timeout "${TIMEOUT_S}s" aiperf profile \
      --url "$MOCK_URL" --model gpt-4o-mini --random-seed 42 --tokenizer builtin \
      -o . "$@" ) >"$out/aiperf.log" 2>&1
  local rc=$?
  echo "$rc" > "$out/exit-code.txt"

  # 3. Teardown: kill the mock and copy its log next to aiperf.log for attribution.
  kill "$MOCK_PID" 2>/dev/null || true
  wait "$MOCK_PID" 2>/dev/null || true
  cp "$MOCK_LOG" "$out/mock.log" 2>/dev/null || true
  echo "MOCK_FLAGS=$mock_flags" > "$out/mock-flags.txt"
}
```

Do NOT replicate `aiperf-mock-server`'s port-pick / `/health` poll / `NO_PROXY` handling inline — invoke the skill so those gotchas stay centralized. The illustration above shows where its returned values plug in, not a substitute for invoking it.

| # | Scenario | Mock flags | aiperf flags | What "graceful" looks like |
|---|---|---|---|---|
| 1 | **all-errors** | `--fast --error-rate 100 --random-seed 42` | `--endpoint-type chat --streaming --request-count 30 --concurrency 4` | Exit non-zero OR exit 0 with 30 errored records in `profile_export.jsonl`. Top-level message names HTTP error. |
| 2 | **partial-errors** | `--fast --error-rate 50 --random-seed 42` | `--endpoint-type chat --streaming --request-count 40 --concurrency 4` | Exit 0; jsonl has ~20 successes + ~20 errors; error field populated; aggregate JSON reports both counts. |
| 3 | **slow-ttft** | `--ttft 5000 --itl 50 --random-seed 42` | `--endpoint-type chat --streaming --request-count 10 --concurrency 2 --request-timeout-seconds 1` | Per-request timeouts surface as errors; no hang past `TIMEOUT_S`. |
| 4 | **server-dead** | (kill the mock before the run — start it, wait for health, then `kill $MOCK_PID` before launching aiperf) | `--endpoint-type chat --request-count 5 --concurrency 1` | Exit non-zero; top-level message says "connection refused"; no traceback flood. |
| 5 | **server-dies-mid-run** | `--fast --random-seed 42` (kill after 5s in a background job: `(sleep 5; kill $MOCK_PID) &`) | `--endpoint-type chat --streaming --request-count 200 --concurrency 8` | Run terminates; jsonl has the requests that completed before kill; remaining counted as errors; no hang. |
| 6 | **oversized-prompt** | `--fast --random-seed 42` | `--endpoint-type chat --request-count 5 --concurrency 1 --osl 16 --isl 100000` (extreme input length) | Either succeeds, or fails per-request with a clear "context length exceeded" / "request too large" message. No silent truncation. |
| 7 | **zero-concurrency** | `--fast` | `--endpoint-type chat --request-count 10 --concurrency 0` | aiperf rejects the config with a clear error at startup. No crash, no run started. |
| 8 | **conflicting-flags** | `--fast` | `--endpoint-type chat --request-count 10 --request-rate 5.0 --concurrency 4 --benchmark-duration 30` (over-specified) | aiperf either picks one with a warning or rejects with a clear message. Don't accept "silently uses concurrency" — that's a finding. |
| 9 | **malformed-trace-input** | `--fast` | `--input-file /tmp/garbage.jsonl --fixed-schedule` (write 3 lines of invalid JSON beforehand) | Fails fast with line-number + parse error. Don't accept "ran 0 requests and exited 0" — that's silent data loss. |
| 10 | **embedding-on-chat-endpoint** | `--fast` | `--endpoint-type embeddings --url http://127.0.0.1:$PORT --request-count 5 --concurrency 1` (the user wires an embeddings client at the chat endpoint by changing the `--url` suffix). | Per-request error with informative message. |

Scenarios 4, 5, 7, 8, 9 don't even need a healthy mock — they probe aiperf's input validation and connection-error paths. Scenarios 1, 2, 3 lean on the mock's `--error-rate` and latency flags. Scenarios 6, 10 probe boundary behavior.

If a scenario doesn't apply to the change under review (e.g., the branch only touches embeddings code, so scenario 1's chat-stream specifics are noise), call it out in the report — don't drop it.

## Assertion pass

Per scenario, check:

```bash
assert_adv() {
  local name=$1; local out="$ART/$name"
  local rc=$(cat "$out/exit-code.txt")
  local verdict="pass"; local reasons=""

  # rc 124 = timeout(1) timeout — that's a HANG, never graceful
  if [ "$rc" = "124" ]; then
    verdict="fail"; reasons="hang past TIMEOUT_S"
  # rc 134/139 = SIGABRT/SIGSEGV — never graceful
  elif [ "$rc" = "134" ] || [ "$rc" = "139" ]; then
    verdict="fail"; reasons="crashed (signal)"
  elif ! grep -qE 'Error|error|fail|refused|timed?[ -]out' "$out/aiperf.log"; then
    # for scenarios where errors are expected, the log MUST say so
    case "$name" in all-errors|server-dead|server-dies-mid-run|partial-errors|slow-ttft|malformed-trace-input)
      verdict="fail"; reasons="errors occurred but log contains no error message" ;;
    esac
  fi

  echo "$verdict: $name${reasons:+ — $reasons}" >> "$ART/verdicts.txt"
}

for s in all-errors partial-errors slow-ttft server-dead server-dies-mid-run \
         oversized-prompt zero-concurrency conflicting-flags malformed-trace-input \
         embedding-on-chat-endpoint; do
  assert_adv "$s"
done
```

Hand-inspect every `fail` entry before claiming the scenario actually broke. The heuristics above will produce false positives (e.g., a scenario that legitimately succeeds and so logs no error). Adjust the per-scenario asserts to match what "graceful" means for that scenario.

## Report

```bash
cat > "$ART/REPORT.md" <<EOF
# Adversarial Test Run — ${EPOCH}

- Branch: $(cat "$ART/branch.txt")
- HEAD: $(cat "$ART/head-sha.txt")
- Timeout per scenario: ${TIMEOUT_S}s
- Pass criteria: terminates, informative error message, errors visible in profile_export.jsonl, no silent data loss

## Verdicts
$(cat "$ART/verdicts.txt")

## Per-scenario artifacts
$(for d in "$ART"/*/; do echo "- $(basename "$d") (exit $(cat "$d/exit-code.txt")): see \`$d/aiperf.log\`, \`$d/mock.log\`"; done)
EOF
```

## Output contract

```
RESULT=pass | fail
ART_DIR=<absolute path>/artifacts/adversarial-<epoch>
FAILED_SCENARIOS=<comma-separated list>
```

If any scenario fails, the user should read `$ART/REPORT.md` and the failing scenario's `aiperf.log` AND `mock.log` — adversarial failures often need both sides to interpret.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "Exit code != 0 means the scenario failed" | Adversarial scenarios EXPECT non-zero exits (server-dead, malformed input). The fail criterion is *crashes, hangs, silent loss* — not "exit != 0". |
| "I'll share one mock across scenarios to save time" | State leaks. `--error-rate` from scenario 1 still applies in scenario 2. Per-scenario launches. |
| "I'll skip `timeout`, the hang will be obvious" | A hang scenario sits forever and the user has to Ctrl-C. Always wrap aiperf in `timeout ${TIMEOUT_S}s`. |
| "Exit 0 + log says 'completed' = pass" | A `--input-file garbage.jsonl` run that exits 0 with 0 records in `profile_export.jsonl` is silent data loss, not a pass. Check the jsonl line count > 0 for any scenario that should have produced output. |
| "The heuristic flagged a false positive, I'll just override it" | Override is fine, but RECORD it in the report ("override: scenario X is pass; the heuristic mis-classified because Y"). Silent overrides defeat the audit. |
| "I'll hand-roll the mock launch inline — it's just one scenario" | Inline launches re-derive port-pick / health-poll / NO_PROXY. Use `aiperf-mock-server` per scenario with per-scenario flags. |
| "Mid-run kill via `(sleep 5; kill $MOCK_PID) &` is unsafe, I'll skip scenario 5" | Scenario 5 catches real-world failure modes. Keep it. |

## Common mistakes

- **Treating exit-non-zero as automatic failure.** Adversarial scenarios EXPECT some non-zero exits (server-dead, malformed input). The fail criterion is *crashes/hangs/silent loss*, not "exit code != 0."
- **Sharing one mock across scenarios.** State leaks (`--error-rate` from scenario 1 still applies in scenario 2). Per-scenario launches.
- **Skipping the `timeout` wrapper.** A hang scenario will sit forever and the user has to Ctrl-C. Always wrap aiperf in `timeout ${TIMEOUT_S}s`.
- **Not reading the jsonl for "silent success" scenarios.** A `--input-file garbage.jsonl` run that exits 0 with 0 records is a FAIL, not a pass. Check `wc -l profile_export.jsonl > 0` for any scenario that should have produced output.
- **Asserting on text in stdout when the actual evidence is in the log file.** Run with `>` redirection and read the file. stdout/stderr ordering varies.
