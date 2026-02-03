<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Best Practices

This guide provides recommendations for effective benchmarking with AIPerf to ensure accurate, meaningful, and reproducible results.

## Table of Contents

- [Benchmark Design](#benchmark-design)
- [Load Pattern Selection](#load-pattern-selection)
- [Dataset Configuration](#dataset-configuration)
- [Metric Interpretation](#metric-interpretation)
- [Statistical Significance](#statistical-significance)
- [Common Pitfalls](#common-pitfalls)
- [Production Benchmarking](#production-benchmarking)
- [Result Validation](#result-validation)

## Benchmark Design

### Define Clear Objectives

Before running benchmarks, clearly define what you're measuring:

**✅ Good objectives:**
- "Measure P99 latency under 100 concurrent users"
- "Determine maximum throughput before latency degrades"
- "Compare two models for similar workloads"
- "Validate SLA compliance at expected traffic levels"

**❌ Poor objectives:**
- "See how fast it is"
- "Test performance"
- "Run some benchmarks"

### Choose Appropriate Metrics

Select metrics that align with your objectives:

| Objective | Primary Metrics | Why |
|-----------|----------------|-----|
| **Interactive UX** | TTFT (P50, P95, P99) | User-perceived responsiveness |
| **Streaming quality** | ITL (P50, P95, P99) | Smooth token delivery |
| **Throughput capacity** | Output Token Throughput, Request Throughput | System capacity |
| **SLA compliance** | Goodput | Percentage meeting requirements |
| **Cost efficiency** | Request Throughput / GPU Utilization | Cost per request |

See [Metrics Reference](metrics_reference.md) for complete metric definitions.

### Use Realistic Workloads

Match your benchmark to production conditions:

```bash
# ❌ Unrealistic: Uniform 100-token prompts
aiperf profile --isl 100 --osl 100

# ✅ Realistic: Mixed workload with variance
aiperf profile \
  --seq-dist "256,128:30;512,256:50;1024,512:20" \
  --session-turns-mean 3 \
  --session-turn-delay-mean 2000
```

**Recommendations:**
- Analyze your production logs for ISL/OSL distributions
- Include prompt length variability (use `--isl-stddev`)
- Model user think time (use `--session-turn-delay-mean`)
- Consider multi-turn conversations if applicable

### Always Use Warmup

Cold starts significantly affect metrics. Use warmup to stabilize measurements:

```bash
# ✅ With warmup (recommended)
aiperf profile \
  --warmup-request-count 50 \
  --request-count 500

# Or time-based warmup
aiperf profile \
  --warmup-duration 30.0 \
  --benchmark-duration 300.0
```

**Warmup guidelines:**
- **Minimum**: 20-50 requests
- **Typical**: 10-20% of main benchmark count
- **Long benchmarks**: 30-60 second warmup duration

### Control for Variables

For valid comparisons, change only one variable:

```bash
# ✅ Good: Only changing concurrency
aiperf profile --concurrency 10 --random-seed 42 [options...] > run1.log
aiperf profile --concurrency 20 --random-seed 42 [options...] > run2.log

# ❌ Bad: Multiple variables changing
aiperf profile --concurrency 10 --isl 256 [options...] > run1.log
aiperf profile --concurrency 20 --isl 512 [options...] > run2.log
```

**Use `--random-seed` for consistent datasets:**
```bash
# Same prompts across runs
aiperf profile --random-seed 42 --concurrency 10 ...
aiperf profile --random-seed 42 --concurrency 20 ...
```

## Load Pattern Selection

### Choosing Concurrency vs Request Rate

**Use Concurrency when:**
- Simulating concurrent users (e.g., 100 simultaneous users)
- Server has request queuing
- You want sustained maximum throughput
- Testing with constant in-flight requests

```bash
aiperf profile --concurrency 50 --request-count 1000
```

**Use Request Rate when:**
- Controlling arrival timing (e.g., 10 requests/second)
- Simulating realistic traffic patterns
- Testing server behavior at specific rates
- Measuring capacity under controlled load

```bash
aiperf profile --request-rate 50 --benchmark-duration 60
```

**Combine both for hybrid control:**
```bash
# Rate limit + concurrency ceiling
aiperf profile --request-rate 100 --concurrency 50
```

### Choosing Arrival Patterns

Match arrival patterns to your use case:

```bash
# Constant intervals (uniform spacing)
aiperf profile --arrival-pattern constant --request-rate 10

# Poisson (realistic bursty traffic)
aiperf profile --arrival-pattern poisson --request-rate 10

# Gamma (tunable smoothness)
aiperf profile --arrival-pattern gamma --arrival-smoothness 2.0 --request-rate 10
```

**Pattern selection guide:**
| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| `constant` | Benchmark repeatability | Predictable, uniform spacing |
| `poisson` | Production traffic | Realistic, bursty, vLLM-default |
| `gamma` | Adjustable realism | Tunable between smooth and bursty |

See [Arrival Patterns Tutorial](tutorials/arrival-patterns.md).

### Ramping for Capacity Testing

Use gradual ramping to find capacity limits:

```bash
# Ramp concurrency from 1 to 100 over 60 seconds
aiperf profile \
  --concurrency 100 \
  --concurrency-ramp-duration 60 \
  --benchmark-duration 300

# Ramp request rate
aiperf profile \
  --request-rate 100 \
  --request-rate-ramp-duration 60 \
  --benchmark-duration 300
```

**Ramping guidelines:**
- Ramp duration should be 20-30% of total benchmark duration
- Monitor metrics during ramp to identify inflection points
- Use timeslices to analyze performance at different load levels

See [Ramping Tutorial](tutorials/ramping.md).

## Dataset Configuration

### Sequence Length Selection

Choose sequence lengths based on your use case:

**Short Context (< 512 tokens):**
- Emphasizes decode performance
- Lower memory usage
- Good for chat/assistant applications

**Long Context (> 2048 tokens):**
- Emphasizes prefill performance
- Higher memory usage
- Use `--prefill-concurrency` to prevent OOM

**Mixed Workloads:**
```bash
# Recommended: Use sequence distributions
aiperf profile --seq-dist "256,128:30;512,256:50;2048,512:20"
```

### Prefix Caching Simulation

If your server supports prefix caching:

```bash
# Generate prompts with shared prefixes
aiperf profile \
  --num-prefix-prompts 10 \
  --prefix-prompt-length 1024 \
  --isl 1280  # 1024 prefix + 256 unique
```

This tests cache effectiveness by reusing prefixes.

### Custom Datasets

When using custom datasets:

1. **Validate format before benchmarking:**
   ```bash
   # Check your JSONL is valid
   jq . prompts.jsonl > /dev/null
   ```

2. **Use appropriate sampling:**
   ```bash
   # Sequential for traces (preserves order)
   --dataset-sampling-strategy sequential

   # Shuffle for variety (default for most cases)
   --dataset-sampling-strategy shuffle

   # Random with replacement
   --dataset-sampling-strategy random
   ```

3. **Ensure sufficient dataset size:**
   - Dataset should be ≥ request count
   - Or use sampling strategies that wrap/repeat

See [Custom Prompt Benchmarking](tutorials/custom-prompt-benchmarking.md).

## Metric Interpretation

### Understanding Percentiles

Don't rely solely on averages:

```
# Example output
Time to First Token:
  avg: 50ms  ← Half the story
  p50: 45ms  ← Typical user experience
  p90: 75ms  ← 10% slower than this
  p99: 120ms ← Worst 1% experience
```

**Recommendations:**
- **P50**: Typical user experience
- **P90**: Good for capacity planning (90% satisfied)
- **P99**: Critical for SLAs (catches outliers)
- **Max**: Identifies extreme cases

### TTFT vs TTFO for Reasoning Models

For reasoning-capable models:
- **TTFT**: Includes reasoning tokens
- **TTFO**: First non-reasoning token (equivalent to GenAI-Perf's TTFT)

```bash
# Check both metrics
aiperf profile --streaming --model DeepSeek-R1 ...
# Compare TTFT and TTFO in output
```

See [Migrating from GenAI-Perf](migrating.md#reasoning-tokens-and-differences-in-metrics).

### Throughput Metrics

Understand the difference:

| Metric | Scope | Includes TTFT | Use Case |
|--------|-------|---------------|----------|
| **Output Token Throughput Per User** | Per-request | No | Single-user streaming speed |
| **Output Token Throughput** | Aggregate | Yes | System capacity |

```bash
# Example interpretation:
Output Token Throughput Per User: 100 tokens/sec
  → Each user sees 100 tokens/sec during decode

Output Token Throughput: 10,000 tokens/sec
  → System generates 10K tokens/sec total (100 concurrent users)
```

### Using Goodput

Goodput measures quality-adjusted throughput:

```bash
aiperf profile --goodput "time_to_first_token:50 inter_token_latency:10"
```

**Interpretation:**
```
Request Throughput: 100 req/sec   ← Total throughput
Goodput: 85 req/sec                ← Only requests meeting SLOs
Compliance: 85%                    ← Percentage within SLOs
```

Use goodput for:
- SLA validation
- Model/configuration comparison (quality-adjusted)
- Capacity planning with quality constraints

See [Goodput Tutorial](tutorials/goodput.md).

## Statistical Significance

### Sample Size

More requests → more stable metrics:

| Request Count | Use Case | Metric Stability |
|---------------|----------|------------------|
| 50-100 | Quick test | Low (±15% variance) |
| 200-500 | Standard benchmark | Medium (±5% variance) |
| 1000+ | Statistical analysis | High (±2% variance) |
| 5000+ | Production validation | Very high (±1% variance) |

**For percentile stability:**
- P50: 200+ requests
- P90: 500+ requests
- P99: 1000+ requests
- P99.9: 5000+ requests

### Multiple Runs

Run multiple iterations for confidence:

```bash
# Run 3 times with different seeds
for i in {1..3}; do
  aiperf profile \
    --random-seed $i \
    --concurrency 20 \
    --request-count 1000 \
    --profile-export-prefix run_$i
done

# Compare results
aiperf plot --paths artifacts/run_*
```

**Look for:**
- Consistent medians (P50) within 5-10%
- Consistent P99 within 10-15%
- Similar distributions across runs

### Outlier Analysis

High percentiles can be affected by outliers:

```bash
# Export detailed metrics
aiperf profile --export-level records

# Analyze outliers
python -c "
import pandas as pd
df = pd.read_csv('profile_export.jsonl', lines=True)
outliers = df[df['request_latency'] > df['request_latency'].quantile(0.99)]
print(outliers[['request_latency', 'input_sequence_length', 'output_sequence_length']])
"
```

Investigate if:
- Outliers correlate with long prompts/outputs
- Specific time windows show degradation
- Errors contribute to outliers

## Common Pitfalls

### Pitfall: Not Using Warmup

**Problem:** First requests include cold-start overhead
**Solution:**
```bash
# ❌ No warmup
aiperf profile --request-count 100

# ✅ With warmup
aiperf profile --warmup-request-count 20 --request-count 100
```

### Pitfall: Comparing Apples to Oranges

**Problem:** Comparing runs with different configurations
**Solution:** Use `--random-seed` and keep all parameters except one constant

### Pitfall: Ignoring Percentiles

**Problem:** Only looking at averages
**Solution:** Always check P90, P99 for complete picture

### Pitfall: Overloading the System

**Problem:** Too high concurrency causes OOM or thrashing
**Solution:**
```bash
# Start low and gradually increase
aiperf profile --concurrency 10 ...  # baseline
aiperf profile --concurrency 20 ...  # increase
aiperf profile --concurrency 40 ...  # continue until degradation

# Use prefill concurrency for long contexts
aiperf profile --concurrency 100 --prefill-concurrency 4
```

### Pitfall: Short Benchmarks

**Problem:** Not running long enough for stable metrics
**Solution:**
```bash
# ❌ Too short (unstable metrics)
aiperf profile --benchmark-duration 10

# ✅ Sufficient duration (stable metrics)
aiperf profile --benchmark-duration 300  # 5 minutes
```

### Pitfall: Unrealistic Workloads

**Problem:** Fixed sequence lengths unlike production
**Solution:** Use `--seq-dist` or custom datasets matching production

### Pitfall: No Reproducibility

**Problem:** Can't reproduce results for debugging
**Solution:**
```bash
# Always use --random-seed for reproducible datasets
aiperf profile --random-seed 42 ...
```

## Production Benchmarking

### Pre-Production Testing

Before deploying to production:

1. **Baseline Performance:**
   ```bash
   aiperf profile \
     --concurrency <expected_prod> \
     --warmup-request-count 100 \
     --request-count 1000 \
     --random-seed 42
   ```

2. **Stress Test:**
   ```bash
   aiperf profile \
     --concurrency <2x_expected_prod> \
     --prefill-concurrency 8 \
     --benchmark-duration 600
   ```

3. **SLA Validation:**
   ```bash
   aiperf profile \
     --concurrency <expected_prod> \
     --goodput "time_to_first_token:50 inter_token_latency:10" \
     --benchmark-duration 300
   ```

4. **Long Duration Test:**
   ```bash
   aiperf profile \
     --concurrency <expected_prod> \
     --benchmark-duration 3600  # 1 hour
   ```

### Continuous Benchmarking

Integrate into CI/CD:

```bash
#!/bin/bash
# benchmark.sh

# Run standard benchmark
aiperf profile \
  --model $MODEL \
  --url $SERVER_URL \
  --random-seed 42 \
  --concurrency 20 \
  --request-count 500 \
  --warmup-request-count 50 \
  --profile-export-file ci_benchmark

# Check regression against baseline
python scripts/compare_baseline.py \
  ci_benchmark.json \
  baseline.json \
  --threshold 0.10  # 10% regression tolerance
```

### Monitoring Integration

Collect GPU and server metrics:

```bash
aiperf profile \
  --gpu-telemetry dashboard \
  --server-metrics \
  --export-level raw
```

Correlate AIPerf metrics with server-side observations.

## Result Validation

### Sanity Checks

Always verify results make sense:

```bash
# Check basic math
# total_osl ≈ request_count × avg(osl)
# output_token_throughput ≈ total_osl / benchmark_duration
```

### Compare to Baseline

Establish baselines and track drift:

```bash
# Create baseline
aiperf profile [...] --profile-export-file baseline

# Regular benchmarks
aiperf profile [...] --profile-export-file current

# Compare
aiperf plot --paths baseline/ current/
```

### Cross-Validation

Validate against other tools or approaches:
- Compare request throughput to simple `curl` loops
- Verify token counts match server logs
- Check GPU utilization makes sense

### Document Methodology

Always document:
- AIPerf version
- Server type and version
- Model name
- Hardware specs
- Full command used
- Date and duration

```bash
# Save command to file
aiperf profile [...] 2>&1 | tee benchmark_$(date +%Y%m%d).log
```

## See Also

- **[Getting Started](getting-started.md)** - Installation and first benchmark
- **[Tutorial](tutorial.md)** - Step-by-step examples
- **[Metrics Reference](metrics_reference.md)** - Complete metrics guide
- **[Performance Tuning](performance-tuning.md)** - Optimize AIPerf
- **[Examples Gallery](examples.md)** - Real-world scenarios
- **[Reproducibility](reproducibility.md)** - Deterministic benchmarking
