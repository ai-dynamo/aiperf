<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf: Comprehensive LLM Benchmarking

**Presentation Date**: November 13, 2025<br>
**Updated**: February 2, 2026<br>
**Tool**: [AIPerf v0.5.0](https://github.com/ai-dynamo/aiperf) | [Architecture Overview](architecture.md) | [Full Documentation](index.md)

---

## Table of Contents

1. [Setup: Installing AIPerf 0.5.0](#setup-installing-aiperf-050)
2. [Test Endpoint Details](#test-endpoint-details)
3. [Use Case 1: Simple Profiling with Static ISL/OSL](#use-case-1-simple-profiling-with-static-islosl)
   - [Evolution: Pareto Curve Analysis](#evolution-pareto-curve-analysis---resource-efficiency-vs-user-experience)
4. [Use Case 2: Auditing Raw Results - Custom Percentile Analysis](#use-case-2-auditing-raw-results---custom-percentile-analysis)
5. [Use Case 3: Trace-Based Benchmarking with Mooncake](#use-case-3-trace-based-benchmarking-with-mooncake)
6. [Use Case 4: Goodput Analysis - Measuring SLA Compliance](#use-case-4-goodput-analysis---measuring-sla-compliance)
7. [Use Case 5: Time-Sliced Analysis - Performance Over Time](#use-case-5-time-sliced-analysis---performance-over-time)
8. [Summary](#summary)
9. [Advanced Topics](#advanced-topics)
   - [In-Cluster Benchmarking](#in-cluster-benchmarking)
   - [Request Cancellation Testing](#request-cancellation-testing)
10. [Additional Features (v0.5.0)](#additional-features-v050)
    - [Server-Side Metrics Collection](#server-side-metrics-collection)
    - [Automatic Plot Generation](#automatic-plot-generation)
    - [KV Cache Efficiency Testing](#kv-cache-efficiency-testing)
    - [User-Centric Timing Mode](#user-centric-timing-mode)
11. [Coming Soon](#coming-soon)
    - [Kubernetes-Native Benchmarking](#kubernetes-native-benchmarking)

---

## Setup: Installing AIPerf 0.5.0

```bash
pip install aiperf
```

**Key Features in 0.5.0**:
- ✅ [Server-side metrics collection](server_metrics/server-metrics.md) via Prometheus
- ✅ [Automatic plot generation](tutorials/plot.md) (`aiperf plot` command)
- ✅ [KV cache efficiency testing](tutorials/prefix-synthesis.md) with trace synthesis
- ✅ [User-centric timing mode](tutorials/user-centric-timing.md) for multi-turn KV cache TTL testing
- ✅ [Goodput analysis](tutorials/goodput.md) for SLA compliance measurement
- ✅ [Time-sliced analysis](tutorials/timeslices.md) for performance trends over time

**📚 Documentation**: See the full [CLI Options Reference](cli_options.md) for all available parameters.

---

## Test Endpoint Details

**Note**: This was a demo endpoint used for the November 13, 2025 presentation. The cluster has been taken down.

**Model**: Qwen3-0.6B (Qwen/Qwen3-0.6B)<br>
**Inference Engine**: vLLM v0.11.0<br>
**Architecture**: 8-way data parallelism (8 independent vLLM replicas)<br>
**Hardware**: 8x NVIDIA H200 GPUs (1 GPU per replica)<br>
**Deployment**: Kubernetes on Nebius Cloud

**Why this endpoint was chosen for the demo:**
- Small model (~600M parameters) = high throughput for benchmarking
- 8 replicas = demonstrated horizontal scaling
- Public access = allowed live demonstration

**Follow along locally**: You can run a single vLLM replica to try the commands in this guide (results will differ from the multi-replica setup above):

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000

# Wait for the server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }

export ENDPOINT_URL=localhost:8000
```

---

## Use Case 1: Simple Profiling with Static ISL/OSL

**Goal**: Measure baseline performance under controlled load

### Command

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --concurrency 100 \
  --request-count 1000 \
  --isl 1000 \
  --osl 500 \
  --tokenizer Qwen/Qwen3-0.6B
```

### Parameters Explained

| Arg | Value | Purpose |
|-----|-------|---------|
| `--model` | `qwen3-0.6b` | Model identifier (matches endpoint) |
| `--url` | `$ENDPOINT_URL` | Target inference server |
| `--endpoint-type` | `chat` | OpenAI chat completions API |
| `--streaming` | *(flag)* | Enable token streaming |
| `--concurrency` | `100` | Simultaneous connections |
| `--request-count` | `1000` | Total requests to send |
| `--isl` | `1000` | Input tokens per request |
| `--osl` | `500` | Output tokens per response |
| `--tokenizer` | `Qwen/Qwen3-0.6B` | HuggingFace tokenizer for accuracy |

**Key Insight**: This creates 100 "virtual users" sending 1,000 requests total with large payloads (1000→500 tokens).

### Results

```text
                          NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Metric                  ┃      avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ Time to First Token (ms)│   347.15 │ 204.55 │1,052.66│  815.02│  577.05│  289.49│ 143.57│
│ Request Latency (ms)    │ 2,101.75 │ 693.08 │4,770.98│3,613.75│2,319.79│2,057.50│ 303.17│
│ Inter Token Latency (ms)│     3.57 │   1.99 │   8.55 │   5.78 │   3.93 │   3.49 │   0.54│
│ Output Token Throughput │22,521.42 │    N/A │    N/A │    N/A │    N/A │    N/A │    N/A│
│           (tokens/sec)  │          │        │        │        │        │        │       │
│ Request Throughput      │    45.70 │    N/A │    N/A │    N/A │    N/A │    N/A │    N/A│
│      (requests/sec)     │          │        │        │        │        │        │       │
│ Request Count           │ 1,000.00 │    N/A │    N/A │    N/A │    N/A │    N/A │    N/A│
└─────────────────────────┴──────────┴────────┴────────┴────────┴────────┴────────┴───────┘

Benchmark Duration: 21.88 sec
Success Rate: 100% (0 errors)
```

### Key Takeaways

✅ **TTFT = 347ms**: Fast first token delivery - users see responses quickly<br>
✅ **Request Latency = 2.1s**: Total time to generate 500 tokens per request<br>
✅ **System Throughput = 22.5K tokens/sec**: High capacity with 100 concurrent users<br>
✅ **ITL = 3.57ms**: Smooth, consistent token streaming<br>
✅ **P99 Latency = 3.6s**: Even worst-case requests complete reasonably fast

**What we learned**:
- With 100 concurrent users and large payloads (1000→500 tokens), the system maintained stable performance
- P99 latency (3.6s) vs avg (2.1s) shows good consistency - only ~70% variance at tail
- Zero errors = reliable service under load
- **22.5K tokens/sec** sustained throughput demonstrates 8-replica scaling effectiveness

---

### Evolution: Pareto Curve Analysis - Resource Efficiency vs. User Experience

**Goal**: Understand the trade-off between resource utilization (TPS/GPU) and user experience (TPS/User) at different concurrency levels.

#### The Experiment

We ran the same benchmark at **5 different concurrency levels** (10, 50, 100, 200, 500) to observe how throughput per GPU and throughput per user change:

```bash
# Run the same benchmark at multiple concurrency levels
for c in 10 50 100 200 500; do
  aiperf profile --model qwen3-0.6b --url "$ENDPOINT_URL" \
    --endpoint-type chat --streaming --concurrency $c \
    --request-count 1000 --isl 1000 --osl 500 \
    --tokenizer Qwen/Qwen3-0.6B --artifact-dir "artifacts/pareto-c$c"
done
```

#### Results: The Pareto Curve

| Concurrency | Total TPS | TPS/GPU | TPS/User | TTFT (avg) |
|------------|-----------|---------|----------|------------|
| **10** | 3,045 | **1,522** | **364.69** | ~250 ms |
| **50** | 12,890 | **6,445** | **326.10** | ~270 ms |
| **100** | 22,521 | **11,261** | **285.03** | ~347 ms |
| **200** | 35,999 | **18,000** ⭐ | **238.67** | ~420 ms |
| **500** | 29,836 | **14,918** | **128.85** | ~1,129 ms |

**Hardware**: 8 vLLM replicas on 8 H200 GPUs (so we divide Total TPS by 8 for TPS/GPU)

#### Visualizing the Trade-off

The Pareto frontier shows the inverse relationship between resource efficiency and user experience:

| Point | Concurrency | TPS/GPU | TPS/User | Interpretation |
|-------|-------------|---------|----------|----------------|
| Far Right | **c=10** | 1,522 | 365 | Best user experience, poor GPU utilization |
| Moving Up-Left | **c=50** | 6,445 | 326 | Trading UX for efficiency |
| | **c=100** | 11,261 | 285 | Balanced middle ground |
| Peak ⭐ | **c=200** | 18,000 | 239 | Maximum GPU efficiency |
| Collapse | **c=500** | 14,918 | 129 | Over-saturation degrades both |

**Key Insight**: The Pareto curve demonstrates you cannot optimize both metrics simultaneously. Choose your operating point based on whether you prioritize cost efficiency (c=200) or user experience (c=10-50).

#### Key Insights from the Pareto Curve

✅ **Low Concurrency (10-50)**:
- **Poor resource utilization**: Only 1,500-6,500 TPS/GPU = GPUs are underutilized
- **Best user experience**: 365 tokens/sec per user = very responsive
- Use case: Premium tier, low-latency applications

✅ **Medium Concurrency (100-200)**:
- **Balanced performance**: ~11,000-18,000 TPS/GPU
- **Good user experience**: ~240-285 tokens/sec per user
- **Sweet spot at c=200**: Peak resource utilization (18K TPS/GPU) with acceptable user experience
- Use case: General production workloads

❌ **High Concurrency (500+)**:
- **Degraded resource utilization**: TPS/GPU drops from 18K → 15K
- **Poor user experience**: 129 tokens/sec per user, TTFT = 1.1 seconds
- **Queuing dominates**: Request backlog causes both metrics to degrade
- Use case: Avoid this region unless cost is the only priority

#### The Business Trade-off

**Question**: Should you optimize for **cost efficiency** (max TPS/GPU) or **user satisfaction** (max TPS/User)?

| Priority | Optimal Concurrency | Justification |
|----------|---------------------|---------------|
| **User Experience** | **10-50** | Sub-300ms TTFT, 325+ tokens/sec/user |
| **Balanced** | **100-200** ⭐ | 18K TPS/GPU, 240+ tokens/sec/user |
| **Cost Efficiency** | **200** | Peak TPS/GPU before degradation |

**The c=200 "sweet spot"**:
- 12x better resource utilization vs. c=10 (18K vs. 1.5K TPS/GPU)
- Only 35% reduction in per-user throughput (239 vs. 365 tokens/sec/user)
- TTFT still under 500ms for most requests

#### What We Learned

🔍 **Performance is non-linear**: Doubling concurrency doesn't double throughput<br>
📊 **The U-shaped curve**: TPS/GPU rises, peaks at c=200, then falls due to queuing overhead<br>
⚖️ **No free lunch**: Higher concurrency = better GPU utilization BUT worse user experience<br>
🎯 **Know your SLA**: Choose concurrency based on your latency vs. throughput priorities

**Pro tip**: Run this analysis on YOUR endpoint with YOUR request patterns to find YOUR sweet spot!

---

## Use Case 2: Auditing Raw Results - Custom Percentile Analysis

**Scenario**: Your management defines SLAs using **P75**, not the standard P50/P90/P99 that AIPerf reports by default.

**Goal**: Calculate P75 TTFT from raw benchmark data.

### Understanding the Raw Data: profile_export.jsonl

AIPerf outputs detailed per-request data in `profile_export.jsonl`. Each line is a JSON record. See the [Working with Profile Exports](tutorials/working-with-profile-exports.md) tutorial for more analysis techniques.

```json
{
  "metadata": {
    "session_num": 87,
    "x_request_id": "abd8df1a-7904-4aa0-8107-0d74ba0ac0d7",
    "turn_index": 0,
    "request_start_ns": 1763066701865462000,
    "request_end_ns": 1763066703082535666,
    "worker_id": "worker_b431129c",
    "record_processor_id": "record_processor_a1b2c3d4",
    "benchmark_phase": "profiling"
  },
  "metrics": {
    "time_to_first_token": {
      "value": 582.66,
      "unit": "ms"
    },
    "output_token_count": {
      "value": 194,
      "unit": "tokens"
    },
    "request_latency": {
      "value": 1210.008,
      "unit": "ms"
    },
    "input_sequence_length": {
      "value": 1000,
      "unit": "tokens"
    },
    "output_sequence_length": {
      "value": 194,
      "unit": "tokens"
    },
    "inter_token_latency": {
      "value": 3.25,
      "unit": "ms"
    }
  }
}
```

**Key fields**: Every request has `time_to_first_token`, `request_latency`, ISL, OSL, and more.

**Note**: The metadata section may contain additional optional fields including `was_cancelled`, `cancellation_time_ns`, `conversation_id`, `x_correlation_id`, and timing fields like `credit_issued_ns` and `request_ack_ns`. The `benchmark_phase` field is either `"warmup"` or `"profiling"`.

### Calculating P75 TTFT

```python
import json
import numpy as np
from pathlib import Path

# Read all TTFT values
ttft_values = []
with open("./artifacts/profile_export.jsonl", 'r') as f:
    for line in f:
        record = json.loads(line)
        ttft = record['metrics']['time_to_first_token']['value']
        ttft_values.append(ttft)

# Calculate P75
p75_ttft = np.percentile(ttft_values, 75)
print(f"P75 TTFT: {p75_ttft:.2f} ms")
```

### Results from Our Benchmark

```text
============================================================
TTFT Percentile Analysis
============================================================
Total requests analyzed: 1000

Percentiles (ms):
  P25 (25th percentile): 242.45 ms
  P50 (50th percentile): 289.49 ms
  P75 (75th percentile): 422.87 ms  ⭐ YOUR SLA METRIC
  P90 (90th percentile): 577.05 ms
  P99 (99th percentile): 815.02 ms
============================================================
```

### Key Takeaways

✅ **P75 = 422.87ms**: 75% of requests get first token within this time<br>
✅ **Raw data access**: Calculate ANY custom metric your org needs<br>
✅ **Full transparency**: Every request is logged with complete metrics<br>
✅ **Easy parsing**: Standard JSON format, one record per line

**Why this matters**:
- Different orgs have different SLA definitions
- P75 is a common SLA target (balance between typical and worst-case)
- AIPerf's raw exports let you calculate ANY percentile or custom metric
- No need to re-run benchmarks for different analysis

---

## Use Case 3: Trace-Based Benchmarking with Mooncake

**Goal**: Test your system under realistic production workload patterns using privacy-preserving traces.

**📚 Documentation**: See [Benchmark Datasets](benchmark_datasets.md) for supported dataset formats and [Trace Replay Mode](benchmark_modes/trace_replay.md) for detailed configuration.

### What is Mooncake Trace Data?

[Mooncake](https://github.com/kvcache-ai/Mooncake) is an open-source KV cache sharing system that released **real production traces** from their arXiv Q&A service. These traces capture actual user behavior including:
- Request arrival times
- Input/output token lengths
- **Block hash IDs**: Privacy-preserving identifiers for KV cache reuse patterns

### Understanding Block Hashing

**The Problem**: Sharing production traces risks leaking sensitive user data.

**Mooncake's Solution**: Hash every 512-token block of input. Users asking about the same document get the same hash IDs, enabling cache reuse analysis **without revealing content**.

**Example: Multi-turn conversation**

```text
Turn 1: User uploads paper (7,500 tokens) + question (500 tokens)
├─ Total: 8,000 tokens = 16 blocks
└─ Hash IDs: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

Turn 2: Same paper + different question (8,500 tokens)
├─ Total: 8,500 tokens = 17 blocks
├─ Hash IDs: [46-61] (reused!) + [62] (new)
└─ ✅ Cache hit rate: 94% (16/17 blocks reused)

Turn 3: Same paper + another question (9,000 tokens)
├─ Total: 9,000 tokens = 18 blocks
├─ Hash IDs: [46-61] (reused!) + [62, 63] (new)
└─ ✅ Cache hit rate: 89% (16/18 blocks reused)
```

**Key insight**: Hash IDs reveal cache reuse opportunities while completely protecting user privacy.

### The Mooncake arXiv Trace Dataset

```text
======================================================================
MOONCAKE ARXIV TRACE - DATASET CHARACTERISTICS
======================================================================

📊 OVERALL STATISTICS
  Total Requests: 23,608
  Duration: 60.0 minutes (3,600 seconds)
  Avg Request Rate: 393.5 requests/minute

📏 TOKEN DISTRIBUTION (Input + Output)
  Mean: 8,772 tokens
  Median: 6,402 tokens
  P25: 3,331 tokens  |  P75: 7,562 tokens
  P90: 17,140 tokens |  P99: 61,961 tokens
  Max: 125,878 tokens

📊 REQUEST SIZE DISTRIBUTION
  Token Range          | Count  | Percentage | Visualization
  ──────────────────────────────────────────────────────────
       0 -  5,000   | 7,632 |  32.3%    | ████████████████
   5,000 - 10,000   | 11,626|  49.2%    | ████████████████████████
  10,000 - 20,000   | 2,499 |  10.6%    | █████
  20,000 - 40,000   | 1,325 |   5.6%    | ██
  40,000 - 60,000   |   272 |   1.2%    |
  60,000 - 80,000   |   135 |   0.6%    |
  80,000 - 100,000  |    65 |   0.3%    |
  100,000+          |    54 |   0.2%    |

⏱️  REQUEST ARRIVAL PATTERN (5-minute windows)
  Time Window         | Requests | Load Pattern
  ───────────────────────────────────────────────────────
    0 -   4 min     | 1,765    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    5 -   9 min     | 1,657    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   10 -  14 min     | 1,875    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   15 -  19 min     | 1,860    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   20 -  24 min     | 1,992    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   25 -  29 min     | 2,010    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   30 -  34 min     | 2,012    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   35 -  39 min     | 2,063    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   40 -  44 min     | 2,133    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   45 -  49 min     | 2,026    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   50 -  54 min     | 2,125    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
   55 -  59 min     | 1,680    | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
======================================================================
```

**Key characteristics of real production traffic**:

✅ **Highly Variable Request Sizes**: 49% of requests are 5K-10K tokens, but tail extends to 125K<br>
✅ **Long-Context Dominant**: Median of 6,402 tokens vs. typical benchmarks using 1K-2K<br>
✅ **Consistent Load**: ~393 requests/minute with relatively steady arrival rate<br>
✅ **Heavy Tail Distribution**: 2% of requests exceed 40K tokens (production reality!)

This represents **real-world patterns** you won't get from synthetic benchmarks:
- Multi-turn conversations (shared hash IDs across requests)
- Variable request sizes (not uniform 1K/500 like Use Case 1)
- Realistic timing (actual production arrival patterns)
- Long-context queries that stress-test model limits

### Running a Trace-Based Benchmark

```bash
# Download the Mooncake trace
curl -o mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# Option 1: Replay with original timing (for end-to-end system testing)
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --input-file mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --tokenizer Qwen/Qwen3-0.6B

# Option 2: Replay as fast as possible (for capacity testing)
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --input-file mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --tokenizer Qwen/Qwen3-0.6B
```

### Key Differences from Synthetic Benchmarks

| Aspect | Use Case 1 (Synthetic) | Use Case 3 (Trace-Based) |
|--------|------------------------|--------------------------|
| **Request Pattern** | Uniform (all 1000→500) | Variable (100→2,000K tokens) |
| **Arrival Timing** | Constant concurrency | Bursty, realistic timing |
| **KV Cache** | No reuse patterns | Real cache-sharing patterns |
| **Use Case** | Steady-state capacity | Production validation |

**📚 Documentation**: See the [Timing Modes Reference](benchmark_modes/timing-modes-reference.md) for all supported timing modes.

### Why Trace-Based Benchmarking Matters

✅ **Realistic Load Testing**: Test how your system handles actual production patterns, not idealized synthetic load<br>
✅ **KV Cache Validation**: If you implement cache sharing (like Mooncake), trace data shows real hit rates<br>
✅ **Capacity Planning**: See performance under bursty traffic with variable request sizes<br>
✅ **Privacy-Preserving**: Hash-based traces enable sharing without exposing sensitive data

**Pro tip**: Use `--fixed-schedule` for end-to-end system validation (respects timing), or remove it to stress-test maximum throughput capacity. See the [Fixed Schedule Tutorial](tutorials/fixed-schedule.md) for more details.

### Real Benchmark Results: 5-Minute Mooncake Trace (5x Speed)

We extracted the first 5 minutes of the Mooncake trace (1,765 requests) and sped it up 5x to replay in ~1 minute:

```bash
# Create the subset (first 5 minutes, sped up 5x)
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --input-file mooncake_trace_5min_5x.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --tokenizer Qwen/Qwen3-0.6B
```

**Results**:

```text
                          NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Metric                ┃     avg ┃    min ┃     max ┃     p99 ┃     p90 ┃    p50 ┃     std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ Time to First Token   │  407.42 │ 212.68 │ 1,519.5 │  951.16 │  586.01 │ 370.20 │  150.12 │
│              (ms)     │         │        │         │         │         │        │         │
│ Request Latency (ms)  │ 1,171.0 │ 243.14 │ 6,665.7 │ 4,184.4 │ 2,615.9 │ 648.33 │  978.09 │
│ Inter Token Latency   │    5.97 │   0.00 │   88.31 │   17.88 │   10.72 │   4.54 │    5.46 │
│              (ms)     │         │        │         │         │         │        │         │
│ Output Sequence Length│  175.27 │   1.00 │ 1,165.0 │  761.65 │  510.00 │  28.00 │  220.30 │
│          (tokens)     │         │        │         │         │         │        │         │
│ Input Sequence Length │ 7,243.0 │ 890.00 │32,236.0 │27,260.0 │15,157.0 │6,344.0 │ 5,536.0 │
│          (tokens)     │         │        │         │         │         │        │         │
│ Output Token          │ 4,675.0 │    N/A │     N/A │     N/A │     N/A │    N/A │     N/A │
│ Throughput (tok/sec)  │         │        │         │         │         │        │         │
│ Request Throughput    │   26.68 │    N/A │     N/A │     N/A │     N/A │    N/A │     N/A │
│      (requests/sec)   │         │        │         │         │         │        │         │
│ Request Count         │ 1,690   │    N/A │     N/A │     N/A │     N/A │    N/A │     N/A │
│      (successful)     │         │        │         │         │         │        │         │
└───────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┴────────┴─────────┘

Benchmark Duration: 63.35 sec
Success Rate: 96% (75 requests exceeded 32K context window)
```

### Key Observations from Trace-Based Testing

✅ **Highly Variable Request Sizes**:
- Input: 890→32,236 tokens (36x range!)
- Output: 1→1,165 tokens
- Median input: 6,344 tokens (much larger than our synthetic 1K)

✅ **Performance Under Real Load**:
- TTFT = 407ms average despite 7K+ token median inputs
- System handled 4,675 tokens/sec with bursty, variable traffic
- P99 TTFT = 951ms (some large requests took longer, as expected)

✅ **Realistic Failures**:
- 75 requests (4%) exceeded Qwen3-0.6B's 32K context limit
- This reveals a real operational constraint you'd miss with synthetic tests
- Production insight: Need longer-context model or request filtering

✅ **Production Timing Patterns**:
- Trace shows realistic request bursts and lulls
- Not constant load like `--concurrency 100`
- More representative of actual user traffic patterns

**What we learned from trace-based vs. synthetic testing**:
- **Use Case 1** (synthetic): 100% success, uniform 1K→500 tokens, 22.5K TPS
- **Use Case 3** (trace): 96% success, variable 890→32K input tokens, 4.7K TPS, revealed context window issues

Trace-based testing exposes real-world challenges that synthetic benchmarks hide!

---

## Use Case 4: Goodput Analysis - Measuring SLA Compliance

**Goal**: Measure what percentage of requests meet your defined Service Level Objectives (SLOs), not just average performance.

**📚 Documentation**: See the [Goodput Tutorial](tutorials/goodput.md) for additional examples and SLO configuration options.

### What is Goodput?

**Goodput** = The fraction of requests that meet ALL specified SLA thresholds.

**Why it matters**:
- **Throughput** tells you how many requests/sec your system handles
- **Goodput** tells you how many requests/sec deliver acceptable user experience
- A system can have high throughput but low goodput if most requests miss SLAs!

**Definition** (from [DistServe paper](https://arxiv.org/pdf/2401.09670)):
> "Goodput measures the number of requests per second that meet specified service-level objectives (SLOs), providing a metric that directly reflects user-perceived quality of service."

### Real-World Example: Why Goodput > Throughput

Imagine two systems serving 1000 requests/min:
- **System A**: 950 requests under SLA, 50 requests timeout → **95% goodput**
- **System B**: 500 requests under SLA, 500 requests slow → **50% goodput**

Both have the same throughput, but System A delivers 2x better user experience!

### Running Goodput Analysis

We'll use the same Mooncake trace, but add SLO thresholds:

```bash
# Define SLA thresholds based on your business requirements
# Example: TTFT ≤ 370ms, Request Latency ≤ 648ms

aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --input-file mooncake_trace_5min_5x.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --tokenizer Qwen/Qwen3-0.6B \
  --goodput "time_to_first_token:370 request_latency:648"
```

### Goodput Results

```text
                          NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Metric                ┃     avg ┃    min ┃     max ┃     p99 ┃     p90 ┃    p50 ┃     std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ Time to First Token   │  428.86 │ 209.96 │ 1,651.8 │ 1,109.7 │  649.21 │ 385.29 │  176.32 │
│              (ms)     │         │        │         │         │         │        │         │
│ Request Latency (ms)  │ 1,208.9 │ 229.80 │ 6,280.6 │ 4,350.7 │ 2,726.4 │ 691.07 │ 1,005.5 │
│ Request Throughput    │   26.67 │    N/A │     N/A │     N/A │     N/A │    N/A │     N/A │
│      (requests/sec)   │         │        │         │         │         │        │         │
│ Goodput               │    7.43 │    N/A │     N/A │     N/A │     N/A │    N/A │     N/A │  ⭐
│ (requests/sec)        │         │        │         │         │         │        │         │
└───────────────────────┴─────────┴────────┴─────────┴─────────┴─────────┴────────┴─────────┘

Benchmark Duration: 63.37 sec
Success Rate: 96% (75 requests exceeded 32K context window)
```

### Key Insights from Goodput Analysis

**Goodput vs. Throughput**:
```text
Total Throughput: 26.67 requests/sec (100%)
Goodput:           7.43 requests/sec (28%)  ⚠️
────────────────────────────────────────────
Only 28% of requests met BOTH SLO requirements!
```

**Understanding the results**:
- SLO Thresholds: TTFT ≤ 370ms AND Request Latency ≤ 648ms
- Average TTFT: 428ms (above threshold)
- Median Latency: 691ms (above threshold)
- 72% of requests failed to meet at least one SLO
- Raw throughput doesn't reveal this user experience gap!

**How SLO compliance works**:
- Requests must meet **ALL** SLO criteria to count toward goodput
- A request with TTFT=350ms but latency=700ms **fails** (missed latency SLO)
- A request with TTFT=400ms but latency=600ms **fails** (missed TTFT SLO)
- Only requests with TTFT≤370ms **AND** latency≤648ms count as goodput

### What Goodput Tells You That Metrics Don't

| Metric | What It Measures | What It Misses |
|--------|------------------|----------------|
| **Average TTFT** | Typical first token delay | Tail latency, SLA violations |
| **P99 Latency** | Worst-case performance | Overall SLA compliance rate |
| **Throughput** | System capacity | User experience quality |
| **Goodput** ⭐ | % requests meeting SLAs | *Nothing - it's the complete picture!* |

### Using Goodput for Capacity Planning

**Question**: How many servers do I need to handle 1000 req/sec with 95% goodput?

**Without goodput analysis**:
- Measure throughput: 26.67 req/sec per server
- Calculate: 1000 / 26.67 = 38 servers
- **Problem**: This assumes all requests meet SLAs! ❌

**With goodput analysis**:
- Measure goodput: 7.43 req/sec per server (28% of throughput)
- Calculate: 1000 / 7.43 = 135 servers
- **Reality**: Need 3.5x more capacity to meet SLAs ✅

**The cost of ignoring goodput**: Underprovisioning by 250%!

### Adjusting SLOs for Your Business

Different use cases need different SLOs:

```bash
# Strict SLOs (premium tier)
--goodput "time_to_first_token:250 request_latency:500"

# Balanced SLOs (standard tier)
--goodput "time_to_first_token:370 request_latency:648"

# Relaxed SLOs (batch processing)
--goodput "time_to_first_token:600 request_latency:2500"
```

**Pro tip**: Set SLO thresholds based on your business requirements, then use goodput to measure compliance and plan capacity accordingly.

---

## Use Case 5: Time-Sliced Analysis - Performance Over Time

**Goal**: Understand how performance metrics evolve during a benchmark to detect warm-up effects, degradation patterns, or load-dependent behavior.

**📚 Documentation**: See the [Timeslices Tutorial](tutorials/timeslices.md) for configuration options and the [Warmup Tutorial](tutorials/warmup.md) for managing cold-start effects.

### What is Time-Slicing?

**Time-slicing** divides your benchmark into sequential time windows, computing metrics independently for each window.

**Why it matters**:
- **Detect warm-up effects**: Identify cold-start latency vs. steady-state performance
- **Spot degradation**: Find memory leaks or resource exhaustion over time
- **Understand load patterns**: See how performance changes as traffic evolves
- **Validate SLAs over time**: Ensure consistent performance, not just averages

### Running Time-Sliced Analysis

We'll use the same Mooncake trace with 10-second time slices:

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --input-file mooncake_trace_5min_5x.jsonl \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --tokenizer Qwen/Qwen3-0.6B \
  --slice-duration 10
```

**Output**: AIPerf generates additional files:
- `profile_export_aiperf_timeslices.csv` - Time-series data in tidy format
- `profile_export_aiperf_timeslices.json` - Hierarchical time-series data

### Time-Sliced Results

```text
==========================================================================================
TIME-SLICED PERFORMANCE ANALYSIS (10-second slices)
==========================================================================================

Slice |  Time   | Requests | TTFT (ms) | Latency (ms) | Throughput
  #   | Window  |  Count   |  avg (p90)|  avg  (p90)  | (tokens/s)
------------------------------------------------------------------------------------------
  0   |  0-10s  |      111 |   545 (  900) |  1516 ( 3217) |       3203
  1   | 10-20s  |      223 |   381 (  560) |  1050 ( 2300) |       3027
  2   | 20-30s  |      279 |   376 (  502) |  1266 ( 3008) |       4014
  3   | 30-40s  |      293 |   388 (  655) |  1272 ( 2942) |       3648
  4   | 40-50s  |      302 |   387 (  500) |   976 ( 2173) |       3554
  5   | 50-60s  |      303 |   344 (  444) |   999 ( 2313) |       3470
  6   | 60-70s  |      179 |   374 (  517) |  1427 ( 2803) |       4258
==========================================================================================

TREND ANALYSIS:
  TTFT Range: 344ms - 545ms (variation: 58.6%)
  Throughput Range: 3027 - 4258 tokens/s
  First slice TTFT: 545ms vs. Last slice: 374ms

✅ Warm-up detected: TTFT improved after first slice (cold start effect)
==========================================================================================
```

### Key Insights from Time-Sliced Analysis

**1. Warm-Up Effect Detected**:
```text
Slice 0 (0-10s):   TTFT = 545ms  ⚠️  Cold start
Slice 1 (10-20s):  TTFT = 381ms  ✅  30% improvement after warm-up
Slices 2-6:        TTFT = 344-388ms  ✅  Stable steady-state
```

**Why this matters**:
- First 10 seconds show 545ms TTFT (above target)
- Performance improves 30% after warm-up
- Steady-state performance (344-388ms) is significantly better than cold-start
- **Implication**: Pre-warming servers before production traffic prevents SLA violations

**2. Variable Load Patterns**:
- Request distribution not uniform: 111 requests (slice 0) → 303 requests (slice 5)
- Throughput varies with load: 3.0K - 4.3K tokens/sec
- System handles variable load without significant degradation

**3. No Performance Degradation**:
- TTFT remains stable from slice 1-6 (344-388ms range)
- No upward trend in latency over time
- No signs of memory leaks or resource exhaustion
- System is healthy for sustained operation

### Comparing Overall vs. Time-Sliced Metrics

| Metric | Overall Average | Slice 0 (Cold) | Slice 1-6 (Warm) |
|--------|----------------|----------------|------------------|
| TTFT   | 386ms          | 545ms (+41%)   | 344-388ms (baseline) |
| Latency| 1,172ms        | 1,516ms        | 976-1,427ms      |

**The hidden truth**: Overall averages mask the 41% cold-start penalty!

### Use Cases for Time-Slicing

**Scenario 1: Detecting Warm-Up Effects**
```text
Problem: SLA violations in first minute of operation
Solution: Use time-slicing to quantify warm-up penalty
Action: Pre-warm servers or set longer health check delays
```

**Scenario 2: Finding Memory Leaks**
```text
Problem: Performance degrades after hours of operation
Solution: Run long benchmark with time-slicing (--benchmark-duration 3600 --slice-duration 300)
Look for: Increasing TTFT/latency in later slices
```

**Scenario 3: Load Pattern Validation**
```text
Problem: Trace-based tests with varying load
Solution: Time-slice to see if performance varies with request density
Look for: Correlation between requests/slice and latency
```

### Best Practices

✅ **Choose appropriate slice duration**:
- Too short (<5s): High variance, unstable metrics
- Too long (>60s): Miss fine-grained patterns
- Recommended: 10-30 seconds for most workloads

✅ **Use with trace-based benchmarks**:
- Time-slicing + realistic traces = complete picture
- See both overall AND time-evolving performance

✅ **Compare cold vs. warm state**:
- Exclude slice 0 from steady-state SLA calculations
- Report both cold-start and warm-state performance separately

✅ **Monitor for degradation**:
- Upward trend in latency = resource issue
- Flat or decreasing latency = healthy system

---

## Summary

We've demonstrated 5 powerful AIPerf use cases:

1. **Simple Profiling + Pareto Analysis**: Find the sweet spot between user experience and resource utilization
2. **Custom Percentile Analysis**: Calculate any metric your organization needs from raw data
3. **Trace-Based Benchmarking**: Test with realistic production workload patterns
4. **Goodput Analysis**: Measure actual SLA compliance, not just raw throughput
5. **Time-Sliced Analysis**: Understand performance evolution and detect warm-up/degradation

**Key Takeaway**: Synthetic benchmarks (Use Case 1) provide baseline capacity, but real-world validation requires traces (Use Case 3), goodput (Use Case 4), and time-series analysis (Use Case 5) to ensure production readiness.

---

## Advanced Topics

### In-Cluster Benchmarking

For high-scale testing, consider running AIPerf from within your Kubernetes cluster to:
- **Eliminate network latency** between client and server
- **Avoid ephemeral port exhaustion** on client machines at extreme concurrency
- **Test true server capacity** without client-side bottlenecks

Deploy a load-tester pod in the same cluster as your inference endpoint and use the internal ClusterIP service address for benchmarking.

### Request Cancellation Testing

Simulate real-world user behavior where requests are cancelled mid-flight (e.g., users navigating away, timeouts). See the [Request Cancellation Tutorial](tutorials/request-cancellation.md) for detailed configuration.

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --concurrency 10 \
  --request-count 100 \
  --request-cancellation-rate 20 \
  --request-cancellation-delay 0.5 \
  --isl 800 \
  --osl 400 \
  --tokenizer Qwen/Qwen3-0.6B
```

**Parameters:**
- `--request-cancellation-rate 20`: Cancel 20% of requests
- `--request-cancellation-delay 0.5`: Wait 0.5 seconds before cancelling

**Use Cases:**
- Test server resource cleanup and connection pooling
- Measure impact of cancellations on remaining requests
- Validate graceful degradation under partial failures

---

## Additional Features (v0.5.0)

### Server-Side Metrics Collection

AIPerf can collect server-side metrics from Prometheus endpoints exposed by your inference server (e.g., vLLM, TensorRT-LLM). See the [Server Metrics Guide](server_metrics/server-metrics.md) for detailed configuration and the [Server Metrics Reference](server_metrics/server_metrics_reference.md) for supported metrics.

```bash
# Auto-discovers Prometheus metrics endpoint from your server URL
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --concurrency 100 \
  --request-count 1000

# Or specify additional metrics endpoints
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --server-metrics "http://server1:9090/metrics" "http://server2:9090/metrics"
```

**Capabilities:**
- **Prometheus integration**: Automatically scrape metrics from `/metrics` endpoints
- **Multi-server support**: Collect from multiple inference replicas
- **Resource utilization**: Track GPU memory, request queuing, and batch sizes
- **GPU telemetry**: See the [GPU Telemetry Tutorial](tutorials/gpu-telemetry.md) for DCGM integration

### Automatic Plot Generation

Generate visualizations from your profiling results with the `aiperf plot` command. See the [Plot Tutorial](tutorials/plot.md) for all available plot types and configuration options.

```bash
# Generate plots from default ./artifacts directory
aiperf plot

# Generate plots from specific directories
aiperf plot --paths ./run1 ./run2

# Launch interactive dashboard
aiperf plot --dashboard

# Use dark theme
aiperf plot --theme dark

# Specify output directory
aiperf plot --output ./my_plots
```

**Available Plot Types:**
- **Latency distributions**: Histograms and percentile bands for TTFT, ITL, request latency
- **Throughput curves**: Token throughput and request throughput over time
- **Pareto analysis**: TPS/GPU vs TPS/User trade-off visualization
- **Time-series analysis**: Performance metrics over time slices
- **Scatter plots**: Request-level latency vs. sequence length
- **Comparison views**: Side-by-side analysis of multiple benchmark runs

**Interactive Dashboard:**
```bash
# Launch on localhost:8050
aiperf plot --dashboard

# Custom port (default 8050)
aiperf plot --dashboard --port 8080

# Custom host (coming soon)
# aiperf plot --dashboard --host 0.0.0.0 --port 8080
```

### KV Cache Efficiency Testing

Test prefix caching and KV cache reuse patterns with trace synthesis and user-centric timing. See the [Prefix Synthesis Tutorial](tutorials/prefix-synthesis.md) for detailed configuration.

**Trace Synthesis** - Generate synthetic traces with controlled prefix-sharing patterns:

```bash
# Analyze existing trace for prefix statistics
aiperf analyze-trace mooncake_trace.jsonl --output-file analysis.json

# Synthesize new traces with controlled scaling
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --input-file mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --synthesis-speedup-ratio 2.0 \
  --synthesis-prefix-len-multiplier 1.5 \
  --tokenizer Qwen/Qwen3-0.6B
```

**Synthesis Options:**
- `--synthesis-speedup-ratio`: Scale trace timing (2.0 = 2x faster replay)
- `--synthesis-prefix-len-multiplier`: Scale shared prefix lengths
- `--synthesis-prefix-root-multiplier`: Distribute traces across N independent prefix trees
- `--synthesis-prompt-len-multiplier`: Scale unique prompt lengths
- `--synthesis-max-isl`: Filter out traces exceeding max input length
- `--synthesis-max-osl`: Cap output length for traces exceeding max

### User-Centric Timing Mode

Simulate realistic multi-turn conversations with controlled per-user timing for KV cache TTL testing. See the [User-Centric Timing Tutorial](tutorials/user-centric-timing.md) for detailed configuration.

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --user-centric-rate 1.0 \
  --num-users 15 \
  --session-turns-mean 20 \
  --shared-system-prompt-length 1000 \
  --tokenizer Qwen/Qwen3-0.6B
```

**Key Features:**
- **Controlled per-user turn gaps**: Each user waits exactly `num_users / QPS` seconds between turns
- **Shared system prompt**: Test prefix caching benefits with `--shared-system-prompt-length`
- **Virtual history**: Immediate steady-state without cold-start transient
- **Cache TTL testing**: Verify KV cache retention at specific time intervals

---

## Coming Soon

AIPerf is actively developing new capabilities:

### Kubernetes-Native Benchmarking
- **Distributed load generation**: Deploy multiple load-tester pods to simulate thousands of concurrent users
- **Large-scale workloads**: Test production-scale traffic patterns without client-side bottlenecks
- **Automated orchestration**: Kubernetes operators to manage benchmark lifecycles and resource allocation

---
