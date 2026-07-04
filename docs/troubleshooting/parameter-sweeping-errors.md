# Parameter Sweeping Error Troubleshooting Guide

This guide helps you resolve common errors when using AIPerf's parameter
sweeping feature. Benchmarking runs under the `aiperf profile` subcommand;
a sweep is triggered by passing a comma-separated magic-list value
(`--concurrency 10,20,30`), which is promoted to a top-level `sweep:` block
(`type: grid` by default), or by an explicit `sweep:` block in a YAML config.

## Common Errors and Solutions

### 1. Non-Numeric Concurrency Value or List

**Error Message:**
```text
concurrency
  Value error, invalid literal for int() with base 10: 'abc' [type=value_error, ...]
```

**Cause:** A concurrency value (single or one element of a comma-separated
list) is not an integer.

**Solution:**
```bash
# Wrong
aiperf profile -m my-model --url localhost:8000 --concurrency abc ...
aiperf profile -m my-model --url localhost:8000 --concurrency 10,abc,30 ...

# Correct
aiperf profile -m my-model --url localhost:8000 --concurrency 10 ...
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 ...
```

---

### 2. Zero or Negative Concurrency Values

**Error Message:**
```text
benchmark.phases.0.concurrency.concurrency
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, ...]
```

**Cause:** Concurrency represents the number of concurrent requests, so
every value in the list must be `>= 1`. A single non-numeric or negative
value fails the whole invocation.

**Solution:**
```bash
# Wrong
aiperf profile -m my-model --url localhost:8000 --concurrency 10,-5,30 ...
aiperf profile -m my-model --url localhost:8000 --concurrency 0,10,20 ...

# Correct
aiperf profile -m my-model --url localhost:8000 --concurrency 10,5,30 ...
aiperf profile -m my-model --url localhost:8000 --concurrency 1,10,20 ...
```

---

### 3. Negative Cooldown Duration

**Error Message:**
```text
parameter_sweep_cooldown_seconds
  Input should be greater than or equal to 0 [type=greater_than_equal, input_value='-10', ...]
```

**Cause:** `--parameter-sweep-cooldown-seconds` (and
`--profile-run-cooldown-seconds`) must be non-negative. Use `0` for no
cooldown.

**Solution:**
```bash
# Wrong
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --parameter-sweep-cooldown-seconds -10 ...

# Correct — no cooldown
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --parameter-sweep-cooldown-seconds 0 ...

# Correct — 10-second pause between variations
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --parameter-sweep-cooldown-seconds 10 ...
```

---

### 4. Empty Sweep Parameter List (YAML)

**Error Message:**
```text
Value error, grid sweep parameter 'phases.profiling.concurrency': value list must be non-empty.
```

**Cause:** A `sweep.variables` entry in a YAML config has an empty list.
An empty list would silently produce zero variations, so it is rejected at
config-load time.

**Solution:**
```yaml
# Wrong
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: []

# Correct
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [10, 20, 30]
```

---

### 5. Zip Sweep With Unequal-Length Lists (YAML)

**Error Message:**
```text
Value error, zip sweep variables must all have equal length; got {'phases.profiling.concurrency': 3, 'phases.profiling.request_count': 2}.
```

**Cause:** A `zip` sweep pairs variables element-wise (lockstep), so every
variable list must have the same length. Use `grid` if you want the full
Cartesian product instead.

**Solution:**
```yaml
# Wrong — lengths 3 and 2
sweep:
  type: zip
  variables:
    phases.profiling.concurrency: [10, 20, 30]
    phases.profiling.request_count: [100, 200]

# Correct — equal length
sweep:
  type: zip
  variables:
    phases.profiling.concurrency: [10, 20, 30]
    phases.profiling.request_count: [100, 200, 300]
```

---

### 6. Sweep Path Doesn't Resolve

**Error Message:**
```text
sweep path '<path>': no entry named '<segment>' found (existing: [...]). Add the entry first or fix the typo.
```

**Cause:** A dotted sweep path references a named-list entry that does not
exist (e.g. a typo like `phase.profiling.concurrency` missing the `s`, or
`phases.profilling.concurrency` with an extra `l`). Named segments
(`phases.<name>.*`) match on the entry's `name` field. Resolved by
`_set_nested_value` in `src/aiperf/config/sweep/expand.py`.

**Solution:**
```yaml
# Wrong — 'phase' should be 'phases'
sweep:
  type: grid
  variables:
    phase.profiling.concurrency: [10, 20, 30]

# Correct
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [10, 20, 30]
```

---

### 7. Dashboard UI with Parameter Sweeps

**Error Message:**
```text
Dashboard UI is incompatible with parameter sweeps; sweep results would
overwrite each other in the live console. Use --ui simple or --ui none with
--concurrency <list> / any sweep configuration.
```

**Cause:** The dashboard UI requires exclusive terminal control, which
conflicts with running multiple sequential benchmarks. Raised at
config-validation time (`src/aiperf/config/config.py`) when a `sweep:`
block is present and `--ui dashboard` is explicitly set.

**Solution:**
```bash
# Wrong
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 --ui dashboard ...

# Correct — simple UI (recommended, shows progress bars)
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 --ui simple ...

# Or — no UI
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 --ui none ...
```

---

### 8. Dashboard UI with Multi-Run

**Error Message:**
```text
Dashboard UI is not supported with sweep/multi-run mode. Please use '--ui simple' or '--ui none' instead.
```

**Cause:** Same terminal-control limitation as above, raised from the
multi-run runner (`src/aiperf/cli_runner/_multi_run.py`) when
`--num-profile-runs > 1` is combined with `--ui dashboard`.

**Solution:**
```bash
# Wrong
aiperf profile -m my-model --url localhost:8000 --num-profile-runs 5 --ui dashboard ...

# Correct
aiperf profile -m my-model --url localhost:8000 --num-profile-runs 5 --ui simple ...
```

---

## Quick Reference: Common Patterns

### Single Concurrency (No Sweep)
```bash
# Basic
aiperf profile -m my-model --url localhost:8000 --concurrency 10 ...

# With multi-run confidence reporting
aiperf profile -m my-model --url localhost:8000 --concurrency 10 --num-profile-runs 5 ...
```

### Parameter Sweep (No Confidence)
```bash
# Basic sweep
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 ...

# With cooldown between values
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --parameter-sweep-cooldown-seconds 10 ...

# With same seed across all values
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --parameter-sweep-same-seed ...
```

### Parameter Sweep + Confidence Reporting
```bash
# Repeated mode (default) — full sweep N times
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 --num-profile-runs 5 ...

# Independent mode — N trials at each value
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 \
  --num-profile-runs 5 --parameter-sweep-mode independent ...

# With cooldowns at both levels
aiperf profile -m my-model --url localhost:8000 --concurrency 10,20,30 --num-profile-runs 5 \
  --parameter-sweep-cooldown-seconds 10 \
  --profile-run-cooldown-seconds 5 ...
```

---

## Getting Help

If you encounter an error not covered in this guide:

1. **Check the error message carefully** — the Pydantic validation errors
   name the exact field path (e.g. `benchmark.phases.0.concurrency.concurrency`)
   and the constraint that failed.

2. **Review the documentation**:
   - [Parameter Sweeping Tutorial](../tutorials/parameter-sweeping.md)
   - [Adaptive Search Errors](./adaptive-search-errors.md) — for `--search-*` (Bayesian Optimization) runs
   - [CLI Options Reference](../cli-options.md)

3. **Report a bug** if the error message is unclear, incorrect, or the
   suggested fix doesn't work. Include the full command line, complete
   error message, AIPerf version (`aiperf --version`), and what you
   expected to happen.
