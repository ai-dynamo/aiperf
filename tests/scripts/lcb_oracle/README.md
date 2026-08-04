# LiveCodeBench codegen oracle for the mock server

Drives AIPerf's **real** `lcb_codegeneration` benchmark and `CodeExecutionGrader`
end-to-end against `aiperf-mock-server`, with no GPU and no model.

Most accuracy benchmarks are easy to mock: mmlu just needs the letter `B` back.
LCB is not. `CodeExecutionGrader` hands the response to lighteval, which
**executes** the extracted program against the problem's real test cases, so the
mock has to return code that genuinely passes. LiveCodeBench ships no reference
solutions.

**The trick:** you do not have to solve the problems. For each stdin/stdout
problem, generate a program that maps every test case's exact stdin to its
expected stdout. The grader runs those same cases, so the lookup passes all of
them and scores `pass@1 = 1.0`.

```python
import sys
_T = {'1 2\n': '3\n', '10 20\n': '30\n'}   # every test case for this problem
_d = sys.stdin.read()
if _d in _T:
    sys.stdout.write(_T[_d])
```

Each generated solution is executed through the real grading worker before it is
written, so a row only lands in the oracle if it actually scores 1.0.

## Prerequisites

```bash
uv pip install -e ".[dev,accuracy]"     # lighteval
uv pip install 'datasets<4'             # REQUIRED, see below
uv pip install -e tests/aiperf_mock_server
```

`datasets<4` is not optional. AIPerf's LCB loader is pinned to the HF
*script-based* dataset `livecodebench/code_generation_lite`, and `datasets>=4`
removed loading-script support entirely — the loader raises before a single
prompt is generated. `datasets==3.6.0` works and lighteval 0.13 still imports.

## 1. Generate the oracle

```bash
python tests/scripts/lcb_oracle/gen_lcb_oracle.py --out /tmp/lcb_oracle.jsonl --count 6
```

```
LCB subset (must match AIPerf): 'v4_v5'
  [KEEP]   abc366_f  cases= 27  pass@1=1.0
  [KEEP]   abc366_b  cases= 20  pass@1=1.0
  ...
scanned 6, skipped 0, wrote 6 rows -> /tmp/lcb_oracle.jsonl
```

Each row is:

```json
{"text": "<question_content>", "ground_truth": "```python\n...\n```",
 "format": "passthrough", "task": "abc366_a"}
```

Two things that will silently cost you an afternoon if you change them:

* **`text` must be `question_content`.** That string is a substring of AIPerf's
  wire prompt, which is what `--accuracy-match substring` keys on. Do **not** use
  `question_id` or add an `id` column expecting it to match — ids never appear in
  the prompt, so every request goes unmatched and the mock serves corpus text.
* **The subset must match AIPerf's.** The script defaults to
  `Environment.ACCURACY.LCB_RELEASE_TAG` (`v4_v5`), passed as the positional HF
  config name. Generating against a different release gives different problems in
  a different order and nothing matches.

Only stdin/stdout problems are usable — a `starter_code` or `metadata.func_name`
means the grader calls a function, which a stdin lookup table cannot satisfy.
Those are skipped.

## 2. Run the mock

```bash
aiperf-mock-server --fast --no-tokenizer --port 8000 \
    --accuracy-dataset /tmp/lcb_oracle.jsonl \
    --accuracy-format passthrough \
    --accuracy-match substring \
    --accuracy-correct-rate 0.5 \
    --random-seed 42
```

Keep every oracle row a *correct* solution and let the mock decide which come
back wrong, via `--accuracy-correct-rate` with a pinned `--random-seed`. The
decision is seeded per prompt, so it is order-independent and reproducible — and
the mock's own tally then becomes an **independent oracle** you can check
AIPerf's grades against.

Baking wrongness into the oracle instead works, but decouples the two counters
and you lose that cross-check.

## 3. Run AIPerf

```bash
aiperf profile \
    --model lcb-oracle-model --tokenizer builtin \
    --url http://127.0.0.1:8000 --endpoint-type chat \
    --accuracy-benchmark lcb-codegeneration \
    --request-count 6 --concurrency 2 \
    --artifact-dir /tmp/lcb-run
```

## 4. Cross-check

```bash
curl -s http://127.0.0.1:8000/accuracy | python3 -m json.tool
cat /tmp/lcb-run/accuracy_results.csv
```

```
mock   : {"matched": 6, "correct": 4, "incorrect": 2, "unmatched": 0}
aiperf : OVERALL,4,6,0,0.6667
```

**`mock.correct` should equal `aiperf.correct`.** The mock served 4 correct
programs; AIPerf executed them and graded 4 correct. A mismatch means the grading
path is broken, not the oracle.

Interpreting the mock's numbers:

| field | meaning |
|---|---|
| `matched` | requests whose prompt matched an oracle row. `unmatched > 0` means matching is misconfigured — check `text`/subset. |
| `correct` | how many times the mock chose to serve the gold **unmodified**. It never executes anything; this is a delivery counter, not a verdict. |
| `accuracy` | `correct / matched` — the mock's intent, to be compared against AIPerf's measured accuracy. |

Use `POST /accuracy/reset` to zero the tally between phases (warmup vs profile,
or two arms of an A/B) without restarting the server.

## Caveat: grading currently returns 0.0 through the normal client path

`CodegenGradingWorker` always sets `AIPERF_CODEGEN_DEATH_FD`, which makes the
worker register a second `os.register_at_fork` handler alongside the one
`_install_stdout_guard` installs. Both re-fire at *every* lighteval fork
(`ProcessPoolExecutor` worker → `Manager()` → `Process`) and close fd **numbers**
multiprocessing has since recycled, so the sandbox child dies and every grade
comes back `pass@1 = 0.0` — silently, with `ok: true` and `unparsed = 0`.

If a run reports 0% across the board, check this before suspecting the oracle. A
single known-correct solution through the worker with and without the env var
tells you immediately:

```bash
# with the var unset -> pass@1 1.0 ; with it set -> pass@1 0.0
```

This script clears the variable when validating, which is why generation succeeds
while a full `aiperf profile` run may not.

## Troubleshooting

| symptom | cause |
|---|---|
| loader raises before any request | `datasets>=4` installed; pin `datasets<4` |
| `unmatched` equals your request count | `text` is not `question_content`, or an `id`-style column overrode the match key, or the subset differs from AIPerf's |
| everything scores 0 including known-correct rows | the at-fork caveat above |
| `--accuracy-correct-rate` seems to have no effect | needs the fenced-body corruption fix; older builds appended after the closing fence, which code extractors discard |
| generator writes 0 rows | the chosen subset's first N problems are all function-based; raise `--count` |
