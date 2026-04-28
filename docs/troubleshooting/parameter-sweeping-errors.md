# Parameter Sweeping Error Troubleshooting Guide

This guide helps you resolve common errors when using AIPerf's parameter sweeping feature.

## Common Errors and Solutions

### 1. Invalid Concurrency Value

**Error Message:**
```text
Invalid concurrency value: 'abc'. Must be a positive integer (>= 1).
Examples: --concurrency 10, --concurrency 20
```

**Cause:** You provided a non-numeric value for concurrency.

**Solution:**
```bash
# Wrong
aiperf --concurrency abc ...

# Correct
aiperf --concurrency 10 ...
```

---

### 2. Invalid Concurrency List

**Error Message:**
```text
Invalid concurrency list: '10,abc,30'. Failed to parse value: 'abc'.
All values must be positive integers (>= 1).
Examples: --concurrency 10,20,30 or --concurrency 5,10,15,20
```

**Cause:** One or more values in your concurrency list is not a valid integer.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,abc,30 ...

# Correct
aiperf --concurrency 10,20,30 ...
```

---

### 3. Negative or Zero Concurrency Values

**Error Message:**
```text
Invalid concurrency values at position(s) [2]: [-5].
All concurrency values must be at least 1 (cannot have zero or negative concurrent requests).
Current list: [10, -5, 30].
Example fix: --concurrency 10,1,30
```

**Cause:** You provided zero or negative values in your concurrency list.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,-5,30 ...
aiperf --concurrency 0,10,20 ...

# Correct
aiperf --concurrency 10,5,30 ...
aiperf --concurrency 1,10,20 ...
```

**Why:** Concurrency represents the number of concurrent requests. You cannot have zero or negative concurrent requests.

---

### 4. Using Sweep Parameters Without a Sweep

**Error Message:**
```text
--parameter-sweep-mode only applies when sweeping parameters (e.g., --concurrency 10,20,30).
This parameter controls whether to run the full sweep repeatedly (repeated mode)
or run all trials at each value independently (independent mode).
Either remove --parameter-sweep-mode or provide a comma-separated list: --concurrency 10,20,30
```

**Cause:** You specified sweep-specific parameters but didn't provide a list of values to sweep.

**Solution:**
```bash
# Wrong - sweep parameter without sweep
aiperf --concurrency 10 --parameter-sweep-mode repeated ...

# Correct - remove sweep parameter
aiperf --concurrency 10 ...

# Or - add sweep values
aiperf --concurrency 10,20,30 --parameter-sweep-mode repeated ...
```

**Applies to:**
- `--parameter-sweep-mode`
- `--parameter-sweep-cooldown-seconds`
- `--parameter-sweep-same-seed`

---

### 5. Using Multi-Run Parameters Without Multi-Run

**Error Message:**
```text
--confidence-level only applies when running multiple trials (--num-profile-runs > 1).
Confidence intervals require at least 2 runs to compute.
Either remove --confidence-level or add --num-profile-runs 5 (or higher).
```

**Cause:** You specified multi-run parameters but only running a single trial.

**Solution:**
```bash
# Wrong - multi-run parameter without multi-run
aiperf --concurrency 10 --confidence-level 0.99 ...

# Correct - remove multi-run parameter
aiperf --concurrency 10 ...

# Or - add multiple runs
aiperf --concurrency 10 --num-profile-runs 5 --confidence-level 0.99 ...
```

**Applies to:**
- `--confidence-level`
- `--profile-run-cooldown-seconds`
- `--profile-run-disable-warmup-after-first`
- `--set-consistent-seed`

---

### 6. Dashboard UI with Parameter Sweeps

**Error Message:**
```text
Dashboard UI (--ui dashboard) is not supported with parameter sweeps
due to terminal control limitations when running multiple sequential benchmarks.
Use --ui simple (recommended, shows progress bars) or --ui none (no UI output).
Example: aiperf --concurrency 10,20,30 --ui simple ...
```

**Cause:** Dashboard UI cannot handle multiple sequential benchmark runs.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,20,30 --ui dashboard ...

# Correct - use simple UI (recommended)
aiperf --concurrency 10,20,30 --ui simple ...

# Or - use no UI
aiperf --concurrency 10,20,30 --ui none ...
```

**Why:** The dashboard UI requires exclusive terminal control, which conflicts with running multiple sequential benchmarks.

---

### 7. Dashboard UI with Multi-Run

**Error Message:**
```text
Dashboard UI (--ui dashboard) is not supported with multi-run mode (--num-profile-runs > 1)
due to terminal control limitations when running multiple sequential benchmarks.
Use --ui simple (recommended, shows progress bars) or --ui none (no UI output).
Example: aiperf --num-profile-runs 5 --ui simple ...
```

**Cause:** Same as above - dashboard UI cannot handle multiple sequential runs.

**Solution:**
```bash
# Wrong
aiperf --num-profile-runs 5 --ui dashboard ...

# Correct - use simple UI (recommended)
aiperf --num-profile-runs 5 --ui simple ...

# Or - use no UI
aiperf --num-profile-runs 5 --ui none ...
```

---

### 8. Invalid Cooldown Duration

**Error Message:**
```text
Invalid cooldown duration: -10 seconds.
Cooldown must be non-negative (0 or greater).
Use 0 for no cooldown, or a positive value like 10 for a 10-second pause between runs.
```

**Cause:** You provided a negative cooldown value.

**Solution:**
```bash
# Wrong
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds -10 ...

# Correct - no cooldown
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 0 ...

# Or - positive cooldown
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 10 ...
```

---

### 9. Empty Parameter Values

**Error Message:**
```text
Parameter sweep requires at least one value to test.
Provide a comma-separated list of values: --concurrency 10,20,30.
For a single value, use: --concurrency 10 (no comma).
```

**Cause:** Internal error - this shouldn't normally happen. May indicate a bug.

**Solution:** Report this as a bug with your command line.

---

### 10. Insufficient Successful Runs for Aggregation

**Warning Message:**
```text
Skipping aggregate statistics for concurrency=20:
only 1 successful run(s), need at least 2 for confidence intervals.
Consider increasing --num-profile-runs or investigating why runs failed.
```

**Cause:** Not enough successful runs at a specific concurrency value to compute confidence statistics.

**Solution:**
```bash
# Increase number of runs
aiperf --concurrency 10,20,30 --num-profile-runs 5 ...

# Or investigate why runs are failing
# Check logs for error messages at the failing concurrency value
```

---

## Quick Reference: Common Patterns

### Single Concurrency (No Sweep)
```bash
# Basic
aiperf --concurrency 10 ...

# With multi-run confidence reporting
aiperf --concurrency 10 --num-profile-runs 5 ...
```

### Parameter Sweep (No Confidence)
```bash
# Basic sweep
aiperf --concurrency 10,20,30 ...

# With cooldown between values
aiperf --concurrency 10,20,30 --parameter-sweep-cooldown-seconds 10 ...

# With same seed across all values
aiperf --concurrency 10,20,30 --parameter-sweep-same-seed ...
```

### Parameter Sweep + Confidence Reporting
```bash
# Repeated mode (default) - full sweep N times
aiperf --concurrency 10,20,30 --num-profile-runs 5 ...

# Independent mode - N trials at each value
aiperf --concurrency 10,20,30 --num-profile-runs 5 --parameter-sweep-mode independent ...

# With cooldowns at both levels
aiperf --concurrency 10,20,30 --num-profile-runs 5 \
  --parameter-sweep-cooldown-seconds 10 \
  --profile-run-cooldown-seconds 5 ...
```

---

## Getting Help

If you encounter an error not covered in this guide:

1. **Check the error message carefully** - it should include:
   - What went wrong
   - Why it's a problem
   - How to fix it
   - An example of correct usage

2. **Review the documentation**:
   - [Parameter Sweeping Tutorial](../tutorials/parameter-sweeping.md)
   - [CLI Options Reference](../cli_options.md)

3. **Report a bug** if:
   - The error message is unclear or unhelpful
   - You believe the error is incorrect
   - The suggested fix doesn't work

Include in your bug report:
- Full command line you ran
- Complete error message
- AIPerf version (`aiperf --version`)
- What you expected to happen
