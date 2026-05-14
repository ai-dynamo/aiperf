---
name: aiperf-correctness-testing
description: Use when verifying that an aiperf code change preserves correct behavior across the supported endpoint surface — chat (streaming + non-streaming), completions, embeddings, rankings (`nim_rankings`/`cohere_rankings`/`hf_tei_rankings`), multimodal (via `chat` or `template`). Actually runs the aiperf CLI against the in-repo mock server, captures `profile_export.jsonl` + the aggregate `profile_export_aiperf.json`, and asserts metrics fall within expected tolerances. Composes with aiperf-mock-server for the backend and aiperf-worktree when isolation is needed. Use when the user asks "does this still work end-to-end?", "verify the X endpoint", "smoke-test before shipping", or as the runtime arm of aiperf-code-review.
---

# AIPerf Correctness Testing

Run the aiperf CLI against the in-repo mock server with **valid, well-formed inputs** across the supported endpoint matrix and verify the outputs (`profile_export.jsonl` per-request stream, `profile_export_aiperf.json` aggregate, exit code, log) match what a correct run produces.

This skill **executes code**. The deliverable is artifacts on disk + a pass/fail verdict per scenario, not an analysis document.

## When to use

- Smoke-testing a branch before requesting review.
- Runtime verification arm of `aiperf-code-review` — a finding claims behavior X breaks; this skill proves whether it does in practice.
- Validating a refactor that touches the request/response path (dataset loaders, endpoint formatters, tokenizer integration, metric computation).
- Spot-checking after a dependency bump.

## When NOT to use

- Pure documentation or static-analysis changes.
- Adversarial / fault-injection scenarios — use `aiperf-adversarial-testing` instead. This skill is happy-path only.
- Validating numerical precision of a specific metric formula — that belongs in unit tests, not end-to-end.

## Pre-flight

```bash
# 1. Workspace (current checkout OK if user is on the branch under test).
WORKDIR="${WORKDIR:-$(pwd)}"
cd "$WORKDIR"
which aiperf >/dev/null || { echo "aiperf not installed; run make first-time-setup" >&2; exit 1; }

# 2. Artifact dir, unique per invocation (epoch suffix avoids collisions on same-day re-runs).
EPOCH="$(date +%s)"
ART="$WORKDIR/artifacts/correctness-${EPOCH}"
mkdir -p "$ART"

# 3. Capture branch state for the record.
git rev-parse --abbrev-ref HEAD > "$ART/branch.txt"
git rev-parse HEAD                > "$ART/head-sha.txt"
git diff --stat origin/main...HEAD > "$ART/diff-stat.txt"
```

## Steps

### 1. Boot the mock server

Invoke `aiperf-mock-server` via the `Skill` tool with `--fast` (deterministic, latency-free — correctness, not perf). The skill returns the launch's `MOCK_URL`, `MOCK_PID`, `MOCK_LOG`, and `MOCK_FLAGS` strings. Bind those into shell variables in the bash session that will run the scenarios:

```bash
# After aiperf-mock-server returns its output contract, set these from its values:
MOCK_URL="http://127.0.0.1:<port>"   # from the skill's MOCK_URL line
MOCK_PID="<pid>"                     # from MOCK_PID
MOCK_LOG="<log path>"                # from MOCK_LOG
MOCK_FLAGS="--fast"                  # whatever you passed
export NO_PROXY="127.0.0.1,localhost"
```

If a corp `HTTP_PROXY` is set, `NO_PROXY` is mandatory — without it, localhost requests get a 405/502 from the proxy.

Install the EXIT trap immediately so an early failure still tears down the server:

```bash
trap 'kill "$MOCK_PID" 2>/dev/null; wait "$MOCK_PID" 2>/dev/null; cp "$MOCK_LOG" "$ART/mock-server.log" 2>/dev/null' EXIT
```

### 2. Run the endpoint matrix

The matrix below is a **starter set** — six common endpoint shapes that catch most regressions, not a canonical list. Extend or trim per branch: if the change touches only embeddings code, drop the chat scenarios; if the change adds a new endpoint type, add a scenario for it. The defaults (20 requests, concurrency 4, seed 42) keep the matrix under ~2 minutes — not policy, just a sane budget.

For each scenario, run `aiperf profile` against `$MOCK_URL`, write to a per-scenario artifact directory, and assert the listed invariants. Don't skip a scenario silently — if a scenario isn't applicable to the branch under test, call that out in the report rather than dropping the row.

```bash
COMMON=(--model gpt-4o-mini --url "$MOCK_URL" --request-count 20 --concurrency 4
        --random-seed 42 --tokenizer builtin)
```

| Scenario | Command (append `"${COMMON[@]}"` and `-o "$ART/<scenario>/"`) | Pass criteria |
|---|---|---|
| **chat-stream** | `aiperf profile --endpoint-type chat --streaming` | exit 0; `profile_export.jsonl` exists with non-zero lines; aggregate JSON shows `request_count == 20`, zero errors; TTFT and ITL keys populated |
| **chat-nonstream** | `aiperf profile --endpoint-type chat` (no `--streaming`) | exit 0; `request_count == 20`; zero errors; `request_latency` populated; no `inter_token_latency` (correct — non-streaming) |
| **completions** | `aiperf profile --endpoint-type completions` | exit 0; `request_count == 20`; zero errors |
| **embeddings** | `aiperf profile --endpoint-type embeddings --request-count 20 --concurrency 4` (drop streaming-related flags) | exit 0; `request_count == 20`; zero errors |
| **nim-rankings** | `aiperf profile --endpoint-type nim_rankings` (other valid choices: `cohere_rankings`, `hf_tei_rankings`) | exit 0; `request_count == 20`; zero errors |
| **chat-multimodal** | `aiperf profile --endpoint-type chat` with a multimodal-prompt config (chat is the canonical multimodal-capable type; the `/v1/custom-multimodal` mock route is exercised via the `template` endpoint type with a custom endpoint path) | exit 0; `request_count == 20`; zero errors |

Run each scenario, redirect stdout/stderr to `$ART/<scenario>/run.log`, and check `$?` immediately. Don't batch — a failure mid-matrix means the next scenarios are running against potentially stale state, and the logs are easier to read when they sit next to their assertions.

```bash
run_scenario() {
  local name=$1; shift
  local out="$ART/$name"
  mkdir -p "$out"
  ( cd "$out" && aiperf profile "$@" "${COMMON[@]}" -o . ) >"$out/run.log" 2>&1
  echo "$?" > "$out/exit-code.txt"
}

run_scenario chat-stream      --endpoint-type chat --streaming
run_scenario chat-nonstream   --endpoint-type chat
run_scenario completions      --endpoint-type completions
run_scenario embeddings       --endpoint-type embeddings
run_scenario nim-rankings     --endpoint-type nim_rankings
# multimodal: depends on how the user wants to exercise the mock's /v1/custom-multimodal route;
# the simplest case is a chat run with a multimodal-prompt input — leaving as user-chosen.
```

The endpoint-type values above come from the project enum. Run `aiperf profile --help` (or read the EndpointType enum) for the full current list — common values include `chat`, `completions`, `embeddings`, `chat_embeddings`, `responses`, `nim_rankings`, `cohere_rankings`, `hf_tei_rankings`, `nim_embeddings`, `huggingface_generate`, `image_generation`, `image_retrieval`, `video_generation`, `solido_rag`, `template`.

### 3. Assert invariants

For each scenario directory, run the assertions. Use a small Python snippet so the inspection is honest (don't grep the rendered summary — the per-request `profile_export.jsonl` and aggregate `profile_export_aiperf.json` are authoritative):

```bash
python - <<'PY' "$ART"
import sys, json, pathlib

art = pathlib.Path(sys.argv[1])
fail = []
for run in sorted(art.glob("*/exit-code.txt")):
    name = run.parent.name
    exit_code = int(run.read_text().strip())
    if exit_code != 0:
        fail.append(f"{name}: exit {exit_code}")
        continue
    jsonl = run.parent / "profile_export.jsonl"
    if not jsonl.exists():
        fail.append(f"{name}: profile_export.jsonl missing")
        continue
    lines = jsonl.read_text().splitlines()
    if len(lines) == 0:
        fail.append(f"{name}: zero records in profile_export.jsonl")
        continue
    if len(lines) != 20:
        # 20 = --request-count from COMMON; if your scenario overrides it, adjust here
        fail.append(f"{name}: expected 20 records, got {len(lines)}")
        continue
    errs = sum(1 for ln in lines if (rec := json.loads(ln)).get("error"))
    if errs > 0:
        fail.append(f"{name}: {errs} errored requests")
    # cross-check against the aggregate summary
    agg = run.parent / "profile_export_aiperf.json"
    if agg.exists():
        summary = json.loads(agg.read_text())
        # summary schema varies by aiperf version; spot-check a sentinel key exists
        if not summary:
            fail.append(f"{name}: empty aggregate summary")

(art / "assertions.txt").write_text("\n".join(fail) if fail else "all scenarios passed")
print(art / "assertions.txt")
PY
```

For per-request column-level analysis (latency percentiles, token counts), the JSONL is the source of truth — read it as one JSON object per line. The aggregate `profile_export_aiperf.{json,csv}` is convenient for headline numbers but rounds/truncates. See `aiperf-profile-export` for the full output-artifact map.

Replace the assertion block with stricter checks when a scenario calls for it (token counts, latency distribution shape, specific metric presence). Don't paper over a failure by relaxing the assertion — relaxing is a code change for the user to approve, not an in-scenario adjustment.

### 4. Write the report

```bash
cat > "$ART/REPORT.md" <<EOF
# Correctness Test Run — ${EPOCH}

- Branch: $(cat "$ART/branch.txt")
- HEAD: $(cat "$ART/head-sha.txt")
- Mock flags: $MOCK_FLAGS
- Scenarios: chat-stream, chat-nonstream, completions, embeddings, nim-rankings, chat-multimodal

## Result

$(cat "$ART/assertions.txt")

## Per-scenario artifacts
$(for d in "$ART"/*/; do echo "- $(basename "$d"): $(cat "$d/exit-code.txt" 2>/dev/null || echo "?") — see \`$d\`"; done)
EOF
```

### 5. Tear down

The EXIT trap from step 1 handles `kill $MOCK_PID` and `cp $MOCK_LOG $ART/`. Verify the trap fired:

```bash
ls "$ART/mock-server.log"   # should exist
```

## Output contract

Calling skills (or the user) get:

```
RESULT=pass | fail
ART_DIR=<absolute path>/artifacts/correctness-<epoch>
FAILED_SCENARIOS=<comma-separated list, or empty>
```

If `RESULT=fail`, tell the user to read `$ART/REPORT.md` and the specific scenario's `run.log` — don't paraphrase the failure into the chat.

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "The summary says 0 errors, jsonl check is overkill" | The aggregate JSON summary rounds. Read `profile_export.jsonl` line-by-line — it's authoritative for per-request data. There is no parquet output for per-request records. |
| "I'll relax this assertion, it's flaky" | Relaxing is a code change for the user to approve, not an in-scenario adjustment. Report the failure as-is. |
| "Branch only touches embeddings, I'll skip the rest" | Skip in the report explicitly with a one-line reason. Do NOT drop rows silently. |
| "I'll run scenarios in parallel to save time" | Errors interleave in the mock log; attribution becomes impossible. Run sequentially. |
| "It worked once, I don't need `--random-seed`" | Without seed, prompt synthesis varies run-to-run and regressions hide in the noise. |
| "I'll point aiperf at the mock without setting NO_PROXY" | If any `HTTP_PROXY` env is set (corp config), localhost routes through it and returns 405/502. Always set `NO_PROXY=127.0.0.1,localhost`. |
| "Exit 0 means pass" | Exit 0 + zero records in `profile_export.jsonl` = silent data loss, not a pass. Always check the jsonl line count > 0. |

## Common mistakes

- **Skimming `--help` output instead of reading `profile_export.jsonl`.** aiperf's text summary truncates and rounds; the jsonl is authoritative for per-request data.
- **Running scenarios in parallel against one mock server.** Errors interleave in the mock log; assertion failures become impossible to attribute. Run sequentially.
- **Re-using one artifact dir across runs.** Use the epoch suffix from pre-flight. Same-day re-runs collide silently otherwise.
- **Asserting "no errors" by grepping logs instead of reading the jsonl's `error` field.** The log may not include every request-level error; the export does.
- **Forgetting to set `--random-seed 42`.** Without it, run-to-run differences in the synthetic prompts mask real regressions.
- **Pointing aiperf at the mock without `NO_PROXY=127.0.0.1,localhost`** when a corp `HTTP_PROXY` is set. The proxy routes localhost traffic through itself and returns 405/502.
- **Using `--endpoint-type rankings` or `--endpoint-type multimodal`** — neither is a valid enum value. Use `nim_rankings` / `cohere_rankings` / `hf_tei_rankings` for ranker workloads; for multimodal, use `chat` with a multimodal-prompt input or `template` with a custom endpoint path.
