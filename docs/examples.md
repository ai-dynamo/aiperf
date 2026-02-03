<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Examples Gallery

Real-world benchmark scenarios with complete commands and explanations.

## Table of Contents

- [Basic Benchmarks](#basic-benchmarks)
- [Load Testing](#load-testing)
- [Production Readiness](#production-readiness)
- [A/B Testing](#ab-testing)
- [Stress Testing](#stress-testing)
- [Long-Duration Testing](#long-duration-testing)
- [Multi-Endpoint Testing](#multi-endpoint-testing)
- [Custom Metrics Collection](#custom-metrics-collection)

## Basic Benchmarks

### Simple Chat Completions

Basic streaming benchmark against a chat model:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 10 \
  --request-count 100
```

**Use case:** Quick performance check
**Duration:** ~30 seconds
**Metrics:** TTFT, ITL, throughput

### Non-Streaming Completions

Benchmark without streaming for batch use cases:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --concurrency 20 \
  --request-count 200
```

**Use case:** Batch processing performance
**Duration:** ~1 minute
**Metrics:** Request latency, throughput (no TTFT/ITL without streaming)

### Embeddings Benchmark

Test embedding model performance:

```bash
aiperf profile \
  --model text-embedding-3-small \
  --url http://localhost:8000 \
  --endpoint-type embeddings \
  --prompt-batch-size 10 \
  --concurrency 50 \
  --request-count 500
```

**Use case:** Vector database ingestion capacity
**Duration:** ~2 minutes
**Metrics:** Request latency, throughput, batch processing efficiency

## Load Testing

### Gradual Ramp-Up

Find maximum sustainable throughput:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 100 \
  --concurrency-ramp-duration 60 \
  --benchmark-duration 300 \
  --warmup-duration 30
```

**Explanation:**
- Ramps from 1 to 100 concurrency over 60 seconds
- Maintains 100 concurrency for remaining 4 minutes
- 30-second warmup before ramp starts

**Use case:** Capacity planning, finding limits
**Duration:** 6.5 minutes
**Analysis:** Use `--slice-duration 30` to see performance at different load levels

### Request Rate Control

Test at specific request rates:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --request-rate 50 \
  --arrival-pattern poisson \
  --benchmark-duration 300 \
  --warmup-duration 30
```

**Explanation:**
- Targets 50 requests/second
- Poisson arrival pattern (realistic, bursty)
- 5-minute test after warmup

**Use case:** Validate capacity at target production rate
**Duration:** 5.5 minutes

### Combined Rate and Concurrency

Hybrid control for realistic scenarios:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --request-rate 100 \
  --concurrency 50 \
  --arrival-pattern poisson \
  --benchmark-duration 600
```

**Explanation:**
- Request rate: 100 req/s (Poisson distribution)
- Concurrency ceiling: 50 (never exceed)
- Simulates: 100 req/s traffic with max 50 concurrent connections

**Use case:** Test API with rate and connection limits
**Duration:** 10 minutes

## Production Readiness

### SLA Validation

Measure compliance with service level objectives:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --benchmark-duration 300 \
  --warmup-duration 60 \
  --goodput "time_to_first_token:50 inter_token_latency:10" \
  --random-seed 42
```

**Explanation:**
- SLOs: TTFT < 50ms, ITL < 10ms
- 1-minute warmup for stable measurements
- Deterministic dataset with seed

**Use case:** Pre-production SLA validation
**Duration:** 6 minutes
**Success criteria:** Goodput > 95% of throughput

### Multi-Turn Conversations

Test realistic conversation patterns:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --num-sessions 100 \
  --session-turns-mean 5 \
  --session-turns-stddev 2 \
  --session-turn-delay-mean 2000 \
  --session-turn-delay-stddev 500 \
  --random-seed 42
```

**Explanation:**
- 100 concurrent sessions
- Average 5 turns per session (± 2)
- 2 seconds average think time between turns (± 0.5s)

**Use case:** Chat application simulation
**Duration:** ~2 minutes

### Long-Context Benchmarking

Test with long prompts safely:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --isl 8192 \
  --osl 1024 \
  --concurrency 20 \
  --prefill-concurrency 4 \
  --request-count 100 \
  --warmup-request-count 10
```

**Explanation:**
- 8K token prompts, 1K token outputs
- Max 4 concurrent prefill operations (prevents OOM)
- 20 concurrent requests total

**Use case:** Long-context model capacity testing
**Duration:** ~5 minutes

## A/B Testing

### Model Comparison

Compare two models with identical workload:

```bash
# Baseline Model A
aiperf profile \
  --model model-a \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 20 \
  --request-count 500 \
  --warmup-request-count 50 \
  --random-seed 42 \
  --profile-export-prefix model_a

# Challenger Model B
aiperf profile \
  --model model-b \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 20 \
  --request-count 500 \
  --warmup-request-count 50 \
  --random-seed 42 \
  --profile-export-prefix model_b

# Compare results
aiperf plot --paths artifacts/model_a artifacts/model_b
```

**Key:** Same `--random-seed` ensures identical prompts

### Configuration Comparison

Test different server configurations:

```bash
# Config 1: Default settings
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --benchmark-duration 300 \
  --random-seed 42 \
  --profile-export-prefix config_default

# Restart server with optimized config...

# Config 2: Optimized settings
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --benchmark-duration 300 \
  --random-seed 42 \
  --profile-export-prefix config_optimized

# Statistical comparison
python scripts/compare_runs.py config_default config_optimized
```

## Stress Testing

### Maximum Throughput

Find absolute maximum capacity:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 500 \
  --prefill-concurrency 8 \
  --request-count 5000 \
  --workers-max 64 \
  --gpu-telemetry dashboard
```

**Explanation:**
- Very high concurrency (500)
- Prefill limit prevents OOM
- 64 workers for maximum load generation
- GPU telemetry to monitor server saturation

**Use case:** Find breaking point, maximum capacity
**Duration:** Varies based on throughput

### Burst Traffic

Simulate traffic spikes:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --arrival-pattern gamma \
  --arrival-smoothness 0.5 \
  --request-rate 100 \
  --benchmark-duration 300
```

**Explanation:**
- Gamma distribution with smoothness 0.5 (bursty)
- Average 100 req/s with high variance

**Use case:** Test server behavior under traffic spikes
**Duration:** 5 minutes

### Mixed Workload

Simulate realistic mixed ISL/OSL:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --seq-dist "128,64:20;512,256:50;2048,512:20;8192,1024:10" \
  --concurrency 100 \
  --prefill-concurrency 8 \
  --request-count 1000
```

**Explanation:**
- 20% short prompts (128 tokens)
- 50% medium prompts (512 tokens)
- 20% long prompts (2K tokens)
- 10% very long prompts (8K tokens)

**Use case:** Production-like workload distribution
**Duration:** Varies

## Long-Duration Testing

### Stability Test

Run for extended period to check degradation:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --benchmark-duration 3600 \
  --warmup-duration 60 \
  --slice-duration 300 \
  --gpu-telemetry dashboard \
  --server-metrics
```

**Explanation:**
- 1-hour benchmark
- 5-minute timeslices for trend analysis
- GPU and server metrics for debugging degradation

**Use case:** Detect memory leaks, performance degradation
**Duration:** 61 minutes
**Analysis:** Check for increasing latency over timeslices

### Overnight Soak Test

Comprehensive stability test:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 20 \
  --benchmark-duration 28800 \
  --warmup-duration 300 \
  --slice-duration 600 \
  --random-seed 42 \
  --export-level records
```

**Explanation:**
- 8-hour test (overnight)
- 10-minute timeslices
- Records export for detailed analysis

**Use case:** Production readiness validation
**Duration:** 8 hours

## Multi-Endpoint Testing

### Full API Suite

Test all supported endpoints:

```bash
# Chat
aiperf profile \
  --model your-chat-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 20 \
  --request-count 200 \
  --profile-export-prefix chat_endpoint

# Completions
aiperf profile \
  --model your-completion-model \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --concurrency 20 \
  --request-count 200 \
  --profile-export-prefix completions_endpoint

# Embeddings
aiperf profile \
  --model your-embedding-model \
  --url http://localhost:8000 \
  --endpoint-type embeddings \
  --prompt-batch-size 10 \
  --concurrency 50 \
  --request-count 500 \
  --profile-export-prefix embeddings_endpoint

# Compare all
aiperf plot --paths artifacts/*_endpoint
```

### Multi-Server Load Balancing

Distribute load across multiple servers:

```bash
aiperf profile \
  --model your-model \
  --url http://server1:8000 \
  --url http://server2:8000 \
  --url http://server3:8000 \
  --url-strategy round_robin \
  --endpoint-type chat \
  --streaming \
  --concurrency 150 \
  --request-count 3000 \
  --workers-max 64
```

**Explanation:**
- 3 servers in round-robin
- 150 total concurrency (50 per server average)
- High worker count for distribution

**Use case:** Multi-GPU, multi-node benchmarking
**Duration:** Varies

## Custom Metrics Collection

### Full Metrics Suite

Collect all available metrics:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --request-count 1000 \
  --warmup-request-count 100 \
  --export-level raw \
  --export-http-trace \
  --show-trace-timing \
  --gpu-telemetry dashboard \
  --server-metrics \
  --slice-duration 60 \
  --goodput "time_to_first_token:50 inter_token_latency:10"
```

**Collected data:**
- Per-request metrics (profile_export.jsonl)
- Raw request/response data (profile_export_raw.jsonl)
- HTTP trace timing
- GPU telemetry (profile_export_gpu_telemetry.jsonl)
- Server metrics (profile_export_server_metrics.jsonl)
- Timeslice analysis
- Goodput measurements

**Use case:** Deep performance analysis
**Duration:** Varies

### Custom Dataset Benchmark

Use your production prompts:

```bash
# Create dataset from production logs
cat production_logs.jsonl | jq '{prompt: .user_message}' > prompts.jsonl

# Benchmark with custom data
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file prompts.jsonl \
  --custom-dataset-type single_turn \
  --dataset-sampling-strategy sequential \
  --concurrency 50 \
  --warmup-request-count 50
```

**Use case:** Production workload validation
**Duration:** Varies based on dataset size

### Trace Replay

Replay exact production traffic pattern:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --input-file production_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --fixed-schedule-auto-offset
```

**Explanation:**
- Replays requests at exact timestamps from trace
- Auto-offset normalizes to start at time 0

**Use case:** Regression testing, A/B comparison with production
**Duration:** Same as original trace

## Tips for Success

### Always Use Warmup

```bash
# ❌ No warmup
aiperf profile --request-count 100

# ✅ With warmup
aiperf profile --warmup-request-count 20 --request-count 100
```

### Use Random Seed for Reproducibility

```bash
aiperf profile --random-seed 42 ...
```

### Monitor Resources

```bash
# Terminal 1: Run benchmark
aiperf profile ...

# Terminal 2: Monitor resources
watch -n 1 'ps aux | grep aiperf; echo "---"; nvidia-smi'
```

### Save Complete Command

```bash
# Save for documentation
aiperf profile [...all args...] 2>&1 | tee benchmark_$(date +%Y%m%d).log
```

## See Also

- **[Getting Started](getting-started.md)** - Installation and basics
- **[Tutorial](tutorial.md)** - Step-by-step guide
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
- **[CLI Options](cli_options.md)** - Complete command reference
- **[Tutorials](tutorials/)** - Feature-specific guides
