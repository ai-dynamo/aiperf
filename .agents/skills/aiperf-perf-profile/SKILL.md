---
name: aiperf-perf-profile
description: Use when investigating performance issues in aiperf — RSS growth, CPU pegged at 100%, throughput ceiling, tail-latency regressions, slow startup, message-bus backpressure. Triggers like "py-spy", "memray", "scalene", "why is records-manager pegging a core", "RSS is climbing over time", "throughput plateaus around N k/s", "GC pauses", "where is the time going", "what's holding the GIL". Encodes the right profiler for each symptom and the aiperf-specific gotchas (HF tokenizer + native-extension HWM retention, glibc allocator arena explosion).
---

# AIPerf Performance Profiling

aiperf has three recurring perf failure modes, each with a canonical investigation tool:

1. **RSS climbs over time but `gc.collect()` doesn't help.** Glibc + native-extension high-water-mark retention. Memory profiler alone won't show it; you need OS-level RSS sampling. Fork-per-iteration is the only reliable bound.
2. **CPU pegged at 100% in one service, throughput plateaus.** GIL contention or event-loop starvation. py-spy in flamegraph mode.
3. **Tail latency regressed.** Per-request latency distribution shift. Read `profile_export.jsonl`, compare percentiles, then drill with py-spy on the suspect service.

## Tool selection

| Symptom | Tool | Why |
|---|---|---|
| CPU at 100%, throughput plateau | **py-spy** | Sampling profiler; no code changes; reads any running process; emits flamegraphs. |
| RSS climbing across iterations | **OS-level RSS sampling** (`/proc/<pid>/status` VmRSS over time) | Memory profilers (memray, tracemalloc) miss HWM retention by native allocators. RSS sampling is honest. |
| Tail latency regression | `profile_export.jsonl` diff + **py-spy** | Per-request data first; then sample the specific service. |
| Slow startup (`aiperf --help` regressed) | **`-X importtime`** | `python -X importtime` shows what's importing slow. Usually points at an at-module-top heavy import. |
| Specific function's allocation pattern | **memray / tracemalloc** | Python-side allocations only; useful for "why is this dict so big". |
| Multi-process aiperf pipeline bottleneck | **py-spy on each service PID** | Run `py-spy dump --pid <pid>` to snapshot stack; identify which service is the gate. |
| Hot loop allocation churn | **scalene** (CPU + memory unified) | Less common; useful when you want one tool with both signals. |

## Investigation workflow

### Step 1 — Reproduce with a tight loop

Don't profile a flaky one-off. Lock down inputs:

```bash
aiperf profile --model gpt-4o-mini --url $MOCK_URL --request-count 2000 --concurrency 16 \
  --random-seed 42 --tokenizer builtin -o /tmp/perf-repro/
```

Choose the mock latency to match what you're investigating:

- **Client-side cost** (GIL contention, hot loops, allocation churn, startup) → use `--fast` on the mock (`aiperf-mock-server`) so server latency is zero and aiperf-side cost dominates.
- **Backpressure / queue-depth / tail-latency under load** → use realistic mock latency (`--ttft <ms> --itl <ms>`); `--fast` would hide the bug because the queueing dynamics never fire.

In short: `--fast` for "where is aiperf spending its CPU?"; realistic timing for "why does the pipeline back up?".

### Step 2 — Sample the right process

aiperf is multi-process. Identify the suspect:

```bash
pgrep -af aiperf
# Locate by service name. Common offenders: worker, records-manager, timing-manager.
```

Then attach py-spy:

```bash
sudo py-spy record -o /tmp/perf-repro/<service>.svg --pid <pid> --duration 30 --rate 100
# Flamegraph SVG opens in any browser.
```

Or stack-dump:

```bash
sudo py-spy dump --pid <pid>
```

(`sudo` needed because py-spy reads the target process's memory; on macOS, code-sign py-spy or grant the terminal Developer Tools permission.)

### Step 3 — For RSS investigation, sample over time

py-spy doesn't help with RSS. Sample the OS-level RSS yourself:

```bash
PID=<pid>
for i in $(seq 1 60); do
  echo "$(date +%s) $(awk '/VmRSS/{print $2}' /proc/$PID/status)" >> /tmp/perf-repro/rss-${PID}.txt
  sleep 5
done
```

Plot the result. If RSS climbs monotonically and `gc.collect()` doesn't bound it, you've hit glibc / native-extension HWM retention. The only reliable mitigation: fork-per-iteration (subprocess per unit of work).

### Step 4 — For startup regressions

```bash
time aiperf --help                                    # baseline
PYTHONPROFILEIMPORTTIME=1 aiperf --help 2>/tmp/import.txt
sort -t: -k2 -n -r /tmp/import.txt | head -20         # slowest imports first
```

If a new heavy import landed at module top, fix the lazy-import discipline (see `aiperf-add-cli`).

### Step 5 — Per-request distribution analysis

If profiling won't reveal a tail-latency spike, the issue is in a specific request class:

```python
import json
import pandas as pd

df = pd.DataFrame([json.loads(line) for line in open("/tmp/perf-repro/profile_export.jsonl")])
# 99th percentile vs 50th — large ratio suggests tail
# (column name may use a `_ns` suffix; inspect df.columns first)
col = "request_latency_ns" if "request_latency_ns" in df.columns else "request_latency"
df[col].quantile([0.5, 0.9, 0.99])
# Drill: which requests are in the tail?
df[df[col] > df[col].quantile(0.99)].sort_values(col, ascending=False).head(20)
```

## Known perf traps

**Records-manager pegging 1 core at high concurrency.** Records-manager is single-event-loop; under high-throughput runs it can become CPU-bound and starve heartbeat handling. For local repros, scale records-manager concurrency, run it standalone with `aiperf service --type records_manager` and profile with py-spy, or check whether the per-record processor work can move to `record_processor` (see `src/aiperf/records/record_processor_service.py`). Deployment-side CPU sizing is handled outside aiperf core (deployment manifests / values).

**Credit issuance is CPU-bound, not I/O-bound, in tight loops.** Removing `yield_to_event_loop` calls (see `src/aiperf/common/utils.py`) makes throughput WORSE, not better — consumers starve when the issuer doesn't yield. Don't optimize the credit path by removing yields.

**RSS climb in long-running tokenizer loops.** HF tokenizers, native-extension Python packages (`msgspec`, `pyarrow`, `numpy`) leak RSS via allocator HWM retention. The fix is architectural (fork-per-iteration), not allocation hygiene. `gc.collect()` and `malloc_trim` reduce some of it but never bound it cleanly.

**Worker / TimingManager intentionally disable GC.** `service_metadata.disable_gc: true` is set in `plugins.yaml` for hot-path services. Don't re-enable GC to "be safe" — GC pauses dominate tail latency at high concurrency.

**glibc allocator arena explosion under xdist.** Affects integration tests, not the runtime. Set `MALLOC_ARENA_MAX=2` (`aiperf-integration-test` covers this).

## Artifact layout

```
artifacts/perf-<epoch>/
  REPORT.md                       # what symptom, what investigation, what fix
  flamegraphs/<service>.svg       # py-spy output per service
  rss/<service>.txt               # RSS over time samples
  parquet-diff.txt                # before-vs-after metric distributions
  meta.json                       # branch, head sha, concurrency, request count
```

## Red flags — STOP, you're rationalizing

| Thought | Reality |
|---|---|
| "I'll grab memray, that'll show the memory leak" | If it's HWM retention, memray won't see it. Sample OS-level RSS first. |
| "I'll add `gc.collect()` to fix the RSS climb" | Doesn't work for native-extension HWM. Architectural fix (fork) or live with it. |
| "I'll guess at the bottleneck and refactor" | Profile first. Refactoring without evidence wastes the refactor. |
| "py-spy needs sudo, let me skip it and read logs instead" | sudo it. Logs don't show GIL contention or hot loops. |
| "I'll profile in production without `--fast` mock" | If you're hunting client-side cost (CPU, alloc, GIL), real-server latency dominates the trace and the bug disappears in noise — switch to `--fast` mock. If you're hunting backpressure, real latency IS the point — keep it. |
| "Records-manager is slow, let me rewrite it" | First confirm with py-spy that it's actually CPU-bound on this run, not waiting on something else. CPU-bound records-manager usually wants more parallelism or work moved to `record_processor`, not a rewrite. |
| "I'll re-enable GC on the worker, can't be that bad" | GC pauses dominate tail latency at high concurrency. `disable_gc: true` exists for measured reasons. Don't undo without measuring. |

## Common mistakes

- **Profiling without a stable repro** — flamegraph reflects whatever happened during the sample, not the bug.
- **Sampling for too short a duration** — 5-second sample misses periodic pauses. Use `--duration 30` minimum.
- **Comparing flamegraphs across different Python versions / glibc versions** — call-stack shapes shift. Re-run baseline alongside.
- **Confusing total memory with leak** — Python's working set may be large but stable. Leak = monotonic climb. Plot RSS over time.

## Composition

- `aiperf-correctness-testing` first — confirm the issue is perf-only, not correctness.
- `aiperf-mock-server` (with `--fast`) for the repro backend.
- `aiperf-profile-export` for the parquet-side distribution analysis.
- `aiperf-debug` for the symptom-catalog scan before deep profiling.
