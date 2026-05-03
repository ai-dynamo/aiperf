<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Token Count & Throughput Discrepancy Analysis Between Client and Server

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Current State in AIPerf](#2-current-state-in-aiperf)
3. [Throughput Discrepancy: Client vs Server](#3-throughput-discrepancy-client-vs-server)
4. [Speculative Decoding Detection](#4-speculative-decoding-detection)
5. [Token Accounting for Reasoning Models](#5-token-accounting-for-reasoning-models)
6. [Prefix Caching Impact on Token Counts](#6-prefix-caching-impact-on-token-counts)
7. [Request-Level vs Aggregate Reconciliation](#7-request-level-vs-aggregate-reconciliation)
8. [Iteration Efficiency Analysis](#8-iteration-efficiency-analysis)
9. [Tokenizer Mismatch Root Cause Analysis](#9-tokenizer-mismatch-root-cause-analysis)
10. [Goodput vs Raw Throughput Correlation](#10-goodput-vs-raw-throughput-correlation)
11. [Unified Discrepancy Framework](#11-unified-discrepancy-framework)
12. [AIPerf Implementation Guidance](#12-aiperf-implementation-guidance)
13. [References](#13-references)

---

## 1. Introduction

LLM inference benchmarking requires precise token accounting. A benchmark tool
measures performance from the client side: it counts tokens in the prompt,
counts tokens in the response, and divides by elapsed time. Meanwhile, the
inference server has its own internal accounting exposed via Prometheus metrics.
When these two perspectives disagree, the benchmark results are unreliable --
but the disagreement itself contains diagnostic signal.

This document systematically analyzes every category of discrepancy between
client-measured and server-measured token counts and throughput. For each
category, we provide:

- **Reconciliation formulas** that express the expected relationship
- **Detection algorithms** that identify when the relationship breaks
- **Decision trees** for root cause analysis
- **AIPerf-specific implementation guidance** referencing existing code

### Why This Matters

A benchmark that reports 500 output tokens/sec on the client side while the
server internally processed 650 generation tokens/sec is not wrong -- it is
revealing that 23% of server-side token work was invisible to the client.
Possible explanations include speculative decoding waste, retried requests, or
prompt caching efficiencies. Each has different implications for capacity
planning and cost modeling.

### Scope

This analysis covers four inference backends supported by AIPerf: **vLLM**,
**SGLang**, **TensorRT-LLM (TRT-LLM)**, and **NVIDIA Dynamo**. Available
Prometheus metrics vary by backend; we identify which analyses are possible for
which backends.

---

## 2. Current State in AIPerf

AIPerf already implements two discrepancy detectors. Understanding their
design informs where the new analyses fit.

### 2.1 OSL Mismatch Detector

**Location**: `src/aiperf/metrics/types/osl_mismatch_metrics.py`

Compares the requested `max_tokens` (output sequence length) against the actual
number of tokens generated. Detects early EOS termination.

```
Diff % = ((actual_osl - requested_osl) / requested_osl) * 100
```

The threshold is adaptive: `min(requested_osl * pct_threshold / 100,
max_token_threshold)`. This caps the absolute token tolerance for large OSL
values. A request for 2000 tokens uses a 50-token ceiling rather than the
percentage-derived 100 tokens.

**What it catches**: Server hitting EOS before requested length. This is a
content-level discrepancy, not a measurement discrepancy.

**What it misses**: The server may have generated the correct number of tokens
but the client may have miscounted them due to tokenizer differences.

### 2.2 Usage Discrepancy Detector

**Location**: `src/aiperf/metrics/types/usage_diff_metrics.py`

Compares API-reported usage tokens (`response.usage.prompt_tokens`,
`response.usage.completion_tokens`) against client-computed token counts via
local tokenizer. Three per-request diff metrics exist:

| Metric | Formula |
|---|---|
| `usage_prompt_tokens_diff_pct` | `abs((usage_prompt - client_ISL) / client_ISL) * 100` |
| `usage_completion_tokens_diff_pct` | `abs((usage_completion - client_OSL) / client_OSL) * 100` |
| `usage_reasoning_tokens_diff_pct` | `abs((usage_reasoning - client_reasoning) / client_reasoning) * 100` |

An aggregate `usage_discrepancy_count` counts records where ANY diff exceeds
`AIPERF_METRICS_USAGE_PCT_DIFF_THRESHOLD` (default 10%).

**What it catches**: Tokenizer mismatch, API special tokens, preprocessing
differences.

**What it misses**: Aggregate-level discrepancies (per-request diffs may be
small but systematic), server-side token work invisible to the API response
(speculative decoding), and throughput discrepancies.

### 2.3 Gap Analysis

The two detectors are **request-scoped** and **token-count-only**. The analyses
in this document extend the framework to:

- **Aggregate reconciliation** (sum of client records vs server counters)
- **Throughput discrepancy** (tokens/sec on each side)
- **Server-internal phenomena** (speculative decoding, prefix caching, iteration batching)
- **Cross-domain correlation** (token counts to latency, throughput to goodput)

---

## 3. Throughput Discrepancy: Client vs Server

### 3.1 Definition

Client-measured throughput and server-measured throughput use fundamentally
different accounting methods.

**Client-side output token throughput** (`output_token_throughput`):

```
client_output_tps = sum(output_sequence_length for all requests) / benchmark_duration_seconds
```

This is computed in `OutputTokenThroughputMetric` as
`TotalOutputSequenceLengthMetric / BenchmarkDurationMetric`.

**Server-side generation token rate** (`vllm:generation_tokens`):

```
server_gen_tps = delta(generation_tokens) / time_range_seconds
```

Where `delta()` is the change in the counter value between the first and last
Prometheus scrape within the profiling window. Available as `stats.rate` in
the server metrics JSON export.

### 3.2 Reconciliation Formula

In a perfect system with no losses, no speculative decoding, and identical
tokenizers:

```
server_gen_tps == client_output_tps
```

In practice:

```
server_gen_tps = client_output_tps
                 + speculative_waste_tps
                 + retried_tokens_tps
                 + time_alignment_error_tps
                 - missed_tokens_tps
```

Where:
- `speculative_waste_tps`: Tokens generated speculatively then rejected
- `retried_tokens_tps`: Tokens from requests that failed and were retried
  (client sees only the successful attempt)
- `time_alignment_error_tps`: Artifact of non-synchronized clocks and
  different measurement windows
- `missed_tokens_tps`: Tokens the client failed to count (e.g., truncated
  streams, connection drops)

### 3.3 Throughput Discrepancy Ratio

```
throughput_discrepancy_ratio = server_gen_tps / client_output_tps
```

| Ratio | Interpretation |
|---|---|
| 1.00 +/- 0.02 | Healthy -- server and client agree within measurement noise |
| 1.05 -- 1.15 | Mild discrepancy -- likely tokenizer differences or time alignment |
| 1.15 -- 2.00 | Significant -- speculative decoding or request retries |
| > 2.00 | Severe -- speculative decoding with low acceptance rate, or major accounting error |
| < 0.95 | Anomalous -- client over-counting or server counter reset |

### 3.4 Detection Algorithm

```python
def detect_throughput_discrepancy(
    client_output_tps: float,
    server_gen_rate: float,
    threshold_pct: float = 5.0,
) -> ThroughputDiscrepancyResult:
    """Compare client and server throughput measurements.

    Args:
        client_output_tps: Client-measured output tokens per second.
        server_gen_rate: Server-reported generation_tokens rate (stats.rate).
        threshold_pct: Percentage difference threshold for flagging.

    Returns:
        Result with ratio, absolute difference, and diagnostic category.
    """
    if client_output_tps <= 0 or server_gen_rate <= 0:
        return ThroughputDiscrepancyResult(
            ratio=float('nan'),
            category="INSUFFICIENT_DATA",
        )

    ratio = server_gen_rate / client_output_tps
    diff_pct = abs(ratio - 1.0) * 100

    if diff_pct <= threshold_pct:
        category = "HEALTHY"
    elif ratio > 1.0 + threshold_pct / 100:
        category = "SERVER_EXCESS"  # Server processed more tokens than client received
    else:
        category = "CLIENT_EXCESS"  # Client counted more than server reported

    return ThroughputDiscrepancyResult(
        ratio=ratio,
        diff_pct=diff_pct,
        category=category,
        client_tps=client_output_tps,
        server_tps=server_gen_rate,
    )
```

### 3.5 Prompt Token Throughput Discrepancy

The same analysis applies to prompt (prefill) tokens:

```
client_prompt_tps = sum(input_sequence_length) / benchmark_duration
server_prompt_tps = delta(vllm:prompt_tokens) / time_range
```

With prefix caching, these will diverge -- see Section 6.

### 3.6 Total Token Throughput

AIPerf's `total_token_throughput` metric sums input and output tokens:

```
client_total_tps = (total_input_tokens + total_output_tokens) / duration
server_total_tps = delta(prompt_tokens + generation_tokens) / time_range
```

### 3.7 Backend Availability

| Backend | Generation Token Counter | Prompt Token Counter |
|---|---|---|
| vLLM | `vllm:generation_tokens` (stats.rate) | `vllm:prompt_tokens` (stats.rate) |
| SGLang | `sglang:gen_throughput` (gauge, tokens/s) | Not available as counter |
| TRT-LLM | Not available | Not available |
| Dynamo | `dynamo_frontend_output_tokens` (stats.rate) | Not available as counter |

**Note**: SGLang exposes `sglang:gen_throughput` as a **gauge** (instantaneous
tokens/s) rather than a monotonic counter. This means the "rate" is
pre-computed by the server and may use a different averaging window than
AIPerf's profiling duration. Comparison requires care:

```
# SGLang: use the gauge average directly
server_gen_tps_sglang = sglang_gen_throughput_stats.avg

# vLLM: compute from counter delta
server_gen_tps_vllm = vllm_generation_tokens_stats.rate
```

---

## 4. Speculative Decoding Detection

### 4.1 Background

Speculative decoding uses a small "draft" model to propose multiple tokens at
once, then the large "target" model verifies them in parallel. Rejected tokens
represent wasted computation. From the client's perspective, only accepted tokens
appear in the response. From the server's perspective, ALL tokens (accepted +
rejected) are processed.

### 4.2 Detection Signal

The primary signal is the throughput discrepancy ratio from Section 3:

```
speculation_ratio = server_generation_tokens_rate / client_output_token_throughput
```

If `speculation_ratio > 1.0`, the server is generating more tokens than the
client receives. The **speculative acceptance rate** can be estimated:

```
estimated_acceptance_rate = 1.0 / speculation_ratio
```

For example, if the server generates 1000 tokens/sec and the client receives
700 tokens/sec, the estimated acceptance rate is 70%.

### 4.3 SGLang Direct Metrics

SGLang exposes speculative decoding metrics directly:

| Metric | Description |
|---|---|
| `sglang:spec_accept_rate` | Acceptance rate (accepted / total draft tokens in batch) |
| `sglang:spec_accept_length` | Average number of consecutive accepted tokens |

When these metrics are available, they provide ground truth:

```
# Direct from SGLang
true_acceptance_rate = sglang_spec_accept_rate_stats.avg

# Cross-validate with throughput ratio
estimated_acceptance_rate = client_output_tps / server_gen_tps
discrepancy = abs(true_acceptance_rate - estimated_acceptance_rate)
```

A large discrepancy between the two estimates suggests other factors (retries,
tokenizer mismatch) are also contributing.

### 4.4 Detection Algorithm

```python
def detect_speculative_decoding(
    client_output_tps: float,
    server_gen_rate: float | None,
    sglang_spec_accept_rate: float | None = None,
    sglang_spec_accept_length: float | None = None,
) -> SpeculativeDecodingResult:
    """Detect speculative decoding and estimate acceptance rate.

    Decision tree:
    1. If sglang:spec_accept_rate is available -> confirmed speculative decoding
    2. If server_gen_rate / client_output_tps > 1.15 -> likely speculative decoding
    3. If ratio is 1.0 +/- 0.05 -> no speculative decoding detected
    """
    result = SpeculativeDecodingResult()

    # Direct detection via SGLang metrics
    if sglang_spec_accept_rate is not None:
        result.confirmed = True
        result.acceptance_rate = sglang_spec_accept_rate
        result.avg_accept_length = sglang_spec_accept_length
        result.source = "sglang_direct"
        return result

    # Indirect detection via throughput ratio
    if server_gen_rate is not None and client_output_tps > 0:
        ratio = server_gen_rate / client_output_tps
        if ratio > 1.15:
            result.likely = True
            result.estimated_acceptance_rate = 1.0 / ratio
            result.throughput_ratio = ratio
            result.source = "throughput_ratio"
            # Waste rate: what fraction of server work was discarded
            result.waste_rate = 1.0 - (1.0 / ratio)

    return result
```

### 4.5 Cost Implications

Speculative decoding waste directly affects cost efficiency:

```
effective_cost_per_token = actual_cost_per_token / acceptance_rate
```

If the acceptance rate is 70%, the effective cost per useful output token is
1.43x the raw cost per token. Reporting this alongside throughput gives users
a complete picture.

### 4.6 Speculative Decoding Efficiency Profile

| Acceptance Rate | Waste Rate | Latency Impact | Throughput Impact |
|---|---|---|---|
| > 90% | < 10% | Strong improvement (fewer iterations) | Moderate improvement |
| 70% -- 90% | 10% -- 30% | Good improvement | May be neutral (waste offsets gains) |
| 50% -- 70% | 30% -- 50% | Marginal improvement | Likely negative (waste dominates) |
| < 50% | > 50% | No improvement or regression | Negative |

---

## 5. Token Accounting for Reasoning Models

### 5.1 Background

Reasoning models (e.g., OpenAI o1, DeepSeek-R1, Qwen3 in thinking mode)
generate internal chain-of-thought tokens that may or may not appear in the
response stream. The API reports these in `usage.completion_tokens_details.reasoning_tokens`.

### 5.2 Token Flow Diagram

```
Server generates:
  [reasoning_tokens] + [visible_completion_tokens] = total_completion_tokens

API reports in usage field:
  usage.completion_tokens = reasoning_tokens + visible_completion_tokens
  usage.completion_tokens_details.reasoning_tokens = reasoning_tokens

Client observes in stream:
  Case A (visible reasoning): reasoning tokens appear with <think>...</think> tags
  Case B (hidden reasoning): reasoning tokens are NOT streamed to client
```

### 5.3 Discrepancy Scenarios

**Scenario 1: Visible reasoning (thinking mode enabled)**

The model streams reasoning tokens with `<think>...</think>` delimiters.
AIPerf's `ReasoningTokenCountMetric` counts these from the parsed response.

```
client_reasoning = count(tokens between <think> and </think>)
server_reasoning = usage.completion_tokens_details.reasoning_tokens
```

Expected: `client_reasoning == server_reasoning` (within tokenizer tolerance).

**Scenario 2: Hidden reasoning (thinking mode not exposed)**

The server generates reasoning tokens internally but does not stream them.
The client sees only visible completion tokens.

```
client_output_tokens = count(visible streamed tokens)   # excludes reasoning
server_completion_tokens = usage.completion_tokens       # includes reasoning
```

This creates a systematic discrepancy:

```
hidden_reasoning_tokens = server_completion_tokens - client_output_tokens
```

If `hidden_reasoning_tokens > 0` and `usage.reasoning_tokens > 0`, the model
is using hidden chain-of-thought.

**Scenario 3: Partial visibility**

Some servers stream a summary of the reasoning but not all reasoning tokens.
The client sees fewer reasoning tokens than the server reports.

```
reasoning_visibility_ratio = client_reasoning_tokens / usage_reasoning_tokens
```

| Ratio | Interpretation |
|---|---|
| 1.0 | Full visibility -- all reasoning tokens streamed |
| 0.0 | Hidden reasoning -- no reasoning tokens in stream |
| 0.0 -- 1.0 | Partial visibility -- summarized reasoning |
| > 1.0 | Client over-counting (likely tokenizer mismatch) |

### 5.4 Reconciliation Formula

For reasoning models, the full token reconciliation is:

```
usage_completion_tokens = visible_output_tokens + reasoning_tokens

Where:
  visible_output_tokens = client_OSL (output_sequence_length)
  reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
```

Rearranged as a consistency check:

```
expected_completion = client_OSL + usage_reasoning_tokens
actual_completion = usage_completion_tokens
reasoning_consistency_error = abs(actual_completion - expected_completion) / actual_completion
```

If `reasoning_consistency_error > threshold`, there are tokens unaccounted for
-- possibly system tokens, formatting tokens, or counting errors.

### 5.5 Throughput Impact of Reasoning

Reasoning tokens consume GPU compute but may not contribute to user-visible
output. The "visible throughput" vs "total throughput" distinction matters:

```
visible_output_tps = sum(client_OSL) / duration          # what the user sees
total_completion_tps = sum(usage_completion_tokens) / duration  # what the server computes
reasoning_overhead = 1 - (visible_output_tps / total_completion_tps)
```

A `reasoning_overhead` of 0.60 means 60% of the server's completion token work
was chain-of-thought reasoning. This is expected for reasoning models but
should be reported transparently.

### 5.6 Detection Algorithm

```python
def analyze_reasoning_token_accounting(
    client_osl_values: NDArray[np.int64],
    client_reasoning_values: NDArray[np.int64],
    usage_completion_values: NDArray[np.int64],
    usage_reasoning_values: NDArray[np.int64],
) -> ReasoningAccountingResult:
    """Analyze token accounting for reasoning models.

    Per-request analysis:
    1. Check if reasoning tokens exist (usage_reasoning > 0)
    2. Compare client vs usage reasoning counts
    3. Check completion token consistency

    Aggregate analysis:
    4. Compute reasoning overhead ratio
    5. Detect hidden vs visible reasoning
    """
    has_reasoning = np.any(usage_reasoning_values > 0)
    if not has_reasoning:
        return ReasoningAccountingResult(has_reasoning=False)

    # Per-request consistency
    expected_completion = client_osl_values + usage_reasoning_values
    consistency_error = np.abs(
        usage_completion_values - expected_completion
    ) / np.maximum(usage_completion_values, 1)

    # Reasoning visibility
    visibility_mask = usage_reasoning_values > 0
    if np.any(visibility_mask):
        visibility_ratios = np.where(
            visibility_mask,
            client_reasoning_values / np.maximum(usage_reasoning_values, 1),
            np.nan,
        )
        avg_visibility = float(np.nanmean(visibility_ratios))
    else:
        avg_visibility = float('nan')

    # Aggregate overhead
    total_visible = int(np.sum(client_osl_values))
    total_completion = int(np.sum(usage_completion_values))
    reasoning_overhead = 1.0 - (total_visible / max(total_completion, 1))

    return ReasoningAccountingResult(
        has_reasoning=True,
        avg_visibility_ratio=avg_visibility,
        reasoning_overhead=reasoning_overhead,
        avg_consistency_error=float(np.mean(consistency_error)),
        total_reasoning_tokens=int(np.sum(usage_reasoning_values)),
        total_visible_tokens=total_visible,
    )
```

---

## 6. Prefix Caching Impact on Token Counts

### 6.1 Background

Prefix caching allows the server to reuse KV-cache entries for shared prompt
prefixes across requests. When a cache hit occurs, the server skips
recomputing those tokens. This has two effects:

1. **Latency**: TTFT drops dramatically for cache hits (skip prefill)
2. **Token counting**: Server may report fewer prompt tokens "processed"
   because cached tokens did not require computation

### 6.2 Client vs Server Prompt Token Discrepancy

The client always sends the full prompt. Its ISL (input sequence length) is
the full tokenized prompt length. The server, with prefix caching, may only
process the non-cached suffix:

```
client_ISL = full_prompt_token_count
server_prompt_tokens_processed = full_prompt_token_count - cache_hit_tokens
```

This means:

```
server_prompt_rate < client_prompt_rate  (when caching is active)
```

### 6.3 Prefix Cache Metrics by Backend

| Backend | Cache Hit Metric | Cache Query Metric | Hit Rate Computation |
|---|---|---|---|
| vLLM | `vllm:prefix_cache_hits` | `vllm:prefix_cache_queries` | `hits / queries` |
| vLLM | `vllm:cache_config_info` labels | — | `enable_prefix_caching` label = True/False |
| SGLang | — | — | `sglang:cache_hit_rate` (gauge, direct ratio) |
| Dynamo | `dynamo_component_kvstats_gpu_prefix_cache_hit_rate` | — | Direct ratio gauge |
| TRT-LLM | Not available | Not available | Not available |

### 6.4 Reconciliation Formula

**vLLM**:

```
prefix_cache_hit_rate = delta(vllm:prefix_cache_hits) / delta(vllm:prefix_cache_queries)
tokens_saved = delta(vllm:prefix_cache_hits)
expected_prompt_delta = sum(client_ISL) - tokens_saved
actual_prompt_delta = delta(vllm:prompt_tokens)

prefix_cache_accounting_error = abs(expected_prompt_delta - actual_prompt_delta)
    / max(expected_prompt_delta, 1)
```

**Note**: The relationship between `vllm:prompt_tokens` and prefix cache hits
depends on vLLM's internal accounting. Some vLLM versions count all prompt
tokens in the counter regardless of caching; others count only tokens that
required computation. Empirical validation is required.

**SGLang**:

```
cache_hit_rate = sglang:cache_hit_rate stats.avg
estimated_tokens_saved = sum(client_ISL) * cache_hit_rate
```

### 6.5 Detection: Is Prefix Caching Active?

```python
def detect_prefix_caching(
    server_metrics: ServerMetricsResults,
    backend: str,
) -> PrefixCachingResult:
    """Detect whether prefix caching is active and quantify its impact.

    Decision tree:
    1. Check vllm:cache_config_info for enable_prefix_caching label
    2. Check vllm:prefix_cache_hits counter -- any delta > 0 means active
    3. Check sglang:cache_hit_rate gauge -- avg > 0 means active
    4. Check dynamo_component_kvstats_gpu_prefix_cache_hit_rate
    5. If prompt_token_rate << client_prompt_rate, infer caching
    """
    # Backend-specific checks
    if backend == "vllm":
        cache_config = get_metric(server_metrics, "vllm:cache_config_info")
        if cache_config:
            for series in cache_config.series:
                if series.labels.get("enable_prefix_caching") == "True":
                    return PrefixCachingResult(
                        active=True,
                        source="config_label",
                        hit_rate=compute_vllm_hit_rate(server_metrics),
                    )

        hits = get_counter_total(server_metrics, "vllm:prefix_cache_hits")
        if hits and hits > 0:
            queries = get_counter_total(server_metrics, "vllm:prefix_cache_queries")
            return PrefixCachingResult(
                active=True,
                source="counter_delta",
                hit_rate=hits / max(queries, 1) if queries else None,
                tokens_saved=int(hits),
            )

    elif backend == "sglang":
        hit_rate = get_gauge_avg(server_metrics, "sglang:cache_hit_rate")
        if hit_rate and hit_rate > 0.01:
            return PrefixCachingResult(
                active=True,
                source="gauge",
                hit_rate=hit_rate,
            )

    elif backend == "dynamo":
        hit_rate = get_gauge_avg(
            server_metrics,
            "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",
        )
        if hit_rate and hit_rate > 0.01:
            return PrefixCachingResult(
                active=True,
                source="gauge",
                hit_rate=hit_rate,
            )

    return PrefixCachingResult(active=False)
```

### 6.6 Impact on Throughput Interpretation

When prefix caching is active, the client-measured prompt throughput
(`sum(ISL) / duration`) overstates the server's actual prefill work. Users
comparing throughput across runs with and without prefix caching must account
for this:

```
adjusted_prompt_throughput = client_prompt_tps * (1 - cache_hit_rate)
```

This `adjusted_prompt_throughput` represents the actual computational work the
server performed on prompt tokens. It is the fairer metric for comparing
server capacity across configurations.

---

## 7. Request-Level vs Aggregate Reconciliation

### 7.1 The Reconciliation Principle

Over a well-defined time window, the sum of per-request token counts from the
client should equal the server's cumulative counters:

```
sum(client_output_tokens[i] for i in profiling_window) == delta(server_generation_tokens)
sum(client_input_tokens[i] for i in profiling_window) == delta(server_prompt_tokens)
```

When they do not match, one of several conditions holds:
1. **Missed requests**: Some server-completed requests are not in the client's records
2. **Duplicate counting**: Client counted a request twice (retry without dedup)
3. **Time alignment error**: Client and server windows do not cover the same requests
4. **Counter reset**: Server restarted during the benchmark
5. **In-flight requests**: Requests that started but did not finish within the window

### 7.2 Time Alignment Challenge

AIPerf uses nanosecond timestamps (`time.time_ns()`) for client events and
Prometheus scrape timestamps for server metrics. These clocks are independent:

```
Client timeline:  |--warmup--|---profiling_window---|--cooldown--|
Server timeline:  ...|--scrape_window--|...

profiling_start_ns != first_scrape_after_start_ns
profiling_end_ns != last_scrape_before_end_ns
```

The `ServerMetricsAccumulator` handles this with `TimeRangeFilter`:
- Reference point: last scrape before `profiling_start_ns` (baseline for deltas)
- End: `max(profiling_end_ns, last_update_ns)` (includes final collection)

Even so, scrapes at 333ms intervals mean up to 333ms of tokens may be
misattributed.

### 7.3 Reconciliation Algorithm

```python
def reconcile_token_counts(
    client_records: list[ParsedResponseRecord],
    server_metrics: ServerMetricsResults,
    profiling_start_ns: int,
    profiling_end_ns: int,
) -> ReconciliationResult:
    """Reconcile client per-request tokens against server aggregate counters.

    Steps:
    1. Sum client-side tokens from all records in profiling window
    2. Extract server counter deltas for same window
    3. Compare and categorize discrepancy
    4. Estimate time alignment error contribution
    """
    # Step 1: Client aggregates
    client_output_total = sum(
        r.output_sequence_length for r in client_records
        if profiling_start_ns <= r.start_ns < profiling_end_ns
    )
    client_input_total = sum(
        r.input_sequence_length for r in client_records
        if profiling_start_ns <= r.start_ns < profiling_end_ns
    )

    # Step 2: Server aggregates (from counter stats.total)
    server_gen_total = get_counter_total(server_metrics, "vllm:generation_tokens")
    server_prompt_total = get_counter_total(server_metrics, "vllm:prompt_tokens")

    # Step 3: Compare
    if server_gen_total is not None:
        output_diff = server_gen_total - client_output_total
        output_diff_pct = output_diff / max(client_output_total, 1) * 100
    else:
        output_diff = None
        output_diff_pct = None

    if server_prompt_total is not None:
        input_diff = server_prompt_total - client_input_total
        input_diff_pct = input_diff / max(client_input_total, 1) * 100
    else:
        input_diff = None
        input_diff_pct = None

    # Step 4: Estimate time alignment contribution
    # Maximum tokens that could be misattributed due to scrape interval
    scrape_interval_s = 0.333  # AIPerf default
    client_output_rate = client_output_total / max(
        (profiling_end_ns - profiling_start_ns) / 1e9, 1
    )
    time_alignment_max_error = client_output_rate * scrape_interval_s

    return ReconciliationResult(
        client_output_total=client_output_total,
        client_input_total=client_input_total,
        server_gen_total=server_gen_total,
        server_prompt_total=server_prompt_total,
        output_diff=output_diff,
        output_diff_pct=output_diff_pct,
        input_diff=input_diff,
        input_diff_pct=input_diff_pct,
        time_alignment_max_error=time_alignment_max_error,
    )
```

### 7.4 Decision Tree for Reconciliation Failures

```
Is output_diff_pct > threshold?
├── YES: server_gen_total > client_output_total?
│   ├── YES: Server processed more tokens than client received
│   │   ├── Is speculative decoding detected? → Speculative waste
│   │   ├── Are there failed/retried requests? → Retry overhead
│   │   └── Is diff within time_alignment_max_error? → Time alignment noise
│   └── NO: Client counted more tokens than server reports
│       ├── Is there a tokenizer mismatch? → Client over-counting
│       ├── Is server_gen_total == 0? → Counter not exposed by backend
│       └── Was there a server restart? → Counter reset
└── NO: Reconciliation passes — counts are consistent
```

### 7.5 Request Count Reconciliation

Beyond token counts, request counts should also reconcile:

```
client_request_count = len(successful_records_in_window)
server_request_count = delta(vllm:request_success) or delta(dynamo_frontend_requests)
```

| Scenario | Indication |
|---|---|
| `server > client` | Server completed requests not captured by client (e.g., other clients) |
| `client > server` | Client retried requests that server counted as one |
| `server == client` | Clean accounting |

**Warning**: If other clients are sending requests to the same server during
the benchmark, server counters will include their tokens. AIPerf cannot
isolate its own traffic from server-side counters unless label filtering is
available.

---

## 8. Iteration Efficiency Analysis

### 8.1 Background

vLLM exposes `vllm:iteration_tokens_total`, a histogram of tokens processed
per scheduler iteration (engine step). Each iteration processes a batch of
tokens from multiple requests simultaneously. The number of tokens per
iteration is effectively the batch token count.

### 8.2 Available Metrics

| Metric | Type | Buckets |
|---|---|---|
| `vllm:iteration_tokens_total` | histogram | 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, +Inf |

From the histogram statistics:
- `stats.count` = total number of iterations during profiling
- `stats.sum` = total tokens processed across all iterations
- `stats.avg` = average tokens per iteration = **effective batch token size**

### 8.3 Derived Metrics

**Effective batch token size:**

```
effective_batch_tokens = iteration_tokens_stats.avg
```

**Iteration rate (engine steps per second):**

```
iteration_rate = iteration_tokens_stats.count_rate
```

**Throughput decomposition:**

```
server_throughput = iteration_rate * effective_batch_tokens
```

This decomposition reveals whether throughput is limited by:
- **Iteration rate**: GPU compute bound (more FLOPs per step needed)
- **Batch token size**: Memory bound (KV-cache full, cannot batch more tokens)

### 8.4 Correlation with Latency

Higher batch token counts increase throughput but also increase per-iteration
latency due to larger attention matrices. The relationship is:

```
per_iteration_time = 1.0 / iteration_rate  (seconds)

# Approximate model:
per_iteration_time ~ alpha + beta * effective_batch_tokens
```

Where `alpha` is fixed overhead and `beta` is the per-token cost. Plotting
`per_iteration_time` vs `effective_batch_tokens` across timeslices reveals the
system's efficiency curve.

### 8.5 Efficiency Score

```
# Theoretical max tokens per iteration (from model config)
max_batch_tokens = vllm_model_max_num_batched_tokens  # from cache_config_info

# Efficiency: how close to max batch utilization
batch_efficiency = effective_batch_tokens / max_batch_tokens

# Iteration efficiency: how well does batching translate to throughput
throughput_per_batch_token = client_output_tps / effective_batch_tokens
```

| Batch Efficiency | Meaning |
|---|---|
| > 0.8 | Near-optimal batching -- KV cache well utilized |
| 0.4 -- 0.8 | Moderate -- some capacity headroom |
| < 0.4 | Under-utilized -- increase concurrency or request rate |

### 8.6 Detection Algorithm

```python
def analyze_iteration_efficiency(
    server_metrics: ServerMetricsResults,
    client_output_tps: float,
    client_concurrent_requests: float,
) -> IterationEfficiencyResult:
    """Analyze engine iteration efficiency from vllm:iteration_tokens_total.

    Decomposes throughput into iteration_rate * batch_size and correlates
    with client-measured concurrency and throughput.
    """
    iter_stats = get_histogram_stats(server_metrics, "vllm:iteration_tokens_total")
    if iter_stats is None:
        return IterationEfficiencyResult(available=False)

    effective_batch_tokens = iter_stats.avg
    iteration_rate = iter_stats.count_rate
    total_iterations = iter_stats.count

    # Throughput decomposition
    server_decomposed_tps = iteration_rate * effective_batch_tokens

    # Cross-validate with client throughput
    throughput_ratio = server_decomposed_tps / max(client_output_tps, 1e-9)

    # Batch efficiency (if max_batch_tokens available from config)
    max_batch = get_config_value(server_metrics, "vllm:cache_config_info",
                                  "max_num_batched_tokens")
    batch_efficiency = (
        effective_batch_tokens / float(max_batch) if max_batch else None
    )

    return IterationEfficiencyResult(
        available=True,
        effective_batch_tokens=effective_batch_tokens,
        iteration_rate=iteration_rate,
        total_iterations=total_iterations,
        server_decomposed_tps=server_decomposed_tps,
        throughput_ratio=throughput_ratio,
        batch_efficiency=batch_efficiency,
    )
```

---

## 9. Tokenizer Mismatch Root Cause Analysis

### 9.1 Background

AIPerf's `UsageDiscrepancyCountMetric` flags requests where client-computed
and API-reported token counts differ by more than a threshold. But the
aggregate pattern of these differences reveals the root cause.

### 9.2 Bias Classification

Collect the signed (not absolute) differences:

```
signed_prompt_diff = usage_prompt_tokens - client_ISL    (per request)
signed_completion_diff = usage_completion_tokens - client_OSL  (per request)
```

**Note**: The current `UsagePromptTokensDiffMetric` uses absolute differences.
A signed variant would enable bias analysis.

### 9.3 Root Cause Decision Tree

```
Compute signed_diff statistics:
├── mean(signed_prompt_diff) > 0 consistently (systematic positive bias)
│   ├── std(signed_prompt_diff) small (< 2 tokens) → Server adds fixed special tokens
│   │   → Root cause: System prompt tokens, BOS/EOS tokens, chat template overhead
│   │   → Fix: Account for template tokens in client tokenization
│   ├── signed_prompt_diff correlates with ISL → Proportional tokenizer divergence
│   │   → Root cause: Different vocabulary (BPE merge differences)
│   │   → Fix: Use --use-server-token-count
│   └── signed_prompt_diff varies randomly → Mixed causes
│
├── mean(signed_prompt_diff) < 0 consistently (systematic negative bias)
│   ├── Client over-counts → Client tokenizer splits more aggressively
│   │   → Root cause: Client using older/different tokenizer version
│   │   → Fix: Update tokenizer to match model
│   └── Server under-counts → Server may use prompt compression
│
├── mean(signed_prompt_diff) ~ 0 but std is large (random variance)
│   ├── Large diffs on specific content → Encoding edge cases
│   │   → Root cause: Unicode, whitespace, multi-byte characters
│   │   → Fix: Investigate outlier requests
│   └── Large diffs on all content → Fundamental tokenizer mismatch
│       → Fix: Use --use-server-token-count
│
└── signed_prompt_diff ~ 0 and std ~ 0 → No discrepancy
    → Tokenizers match
```

### 9.4 Statistical Analysis

```python
def analyze_tokenizer_mismatch(
    signed_prompt_diffs: NDArray[np.float64],
    signed_completion_diffs: NDArray[np.float64],
    client_isl_values: NDArray[np.int64],
    client_osl_values: NDArray[np.int64],
) -> TokenizerMismatchAnalysis:
    """Classify tokenizer mismatch root cause from signed differences.

    Computes:
    1. Bias (mean signed diff) -- systematic over/under-counting
    2. Variance (std of signed diff) -- consistency of mismatch
    3. Correlation with sequence length -- proportional vs fixed offset
    4. Outlier analysis -- encoding edge cases
    """
    # Prompt analysis
    prompt_bias = float(np.mean(signed_prompt_diffs))
    prompt_std = float(np.std(signed_prompt_diffs))

    # Completion analysis
    completion_bias = float(np.mean(signed_completion_diffs))
    completion_std = float(np.std(signed_completion_diffs))

    # Correlation: does diff scale with sequence length?
    if len(signed_prompt_diffs) > 10:
        prompt_corr = float(np.corrcoef(
            signed_prompt_diffs, client_isl_values.astype(np.float64)
        )[0, 1])
    else:
        prompt_corr = float('nan')

    if len(signed_completion_diffs) > 10:
        completion_corr = float(np.corrcoef(
            signed_completion_diffs, client_osl_values.astype(np.float64)
        )[0, 1])
    else:
        completion_corr = float('nan')

    # Classify root cause
    if abs(prompt_bias) < 1.0 and prompt_std < 1.0:
        prompt_cause = "MATCH"
    elif abs(prompt_bias) > 1.0 and prompt_std < 2.0:
        prompt_cause = "FIXED_OFFSET"  # System tokens
    elif abs(prompt_corr) > 0.7:
        prompt_cause = "PROPORTIONAL"  # Different vocabulary
    elif prompt_std > 5.0:
        prompt_cause = "RANDOM_VARIANCE"  # Edge cases
    else:
        prompt_cause = "MIXED"

    return TokenizerMismatchAnalysis(
        prompt_bias=prompt_bias,
        prompt_std=prompt_std,
        prompt_corr=prompt_corr,
        prompt_cause=prompt_cause,
        completion_bias=completion_bias,
        completion_std=completion_std,
        completion_corr=completion_corr,
    )
```

### 9.5 Common Root Causes by Server

| Server | Common Cause | Typical Diff | Notes |
|---|---|---|---|
| vLLM | Chat template tokens | +1 to +5 prompt tokens | BOS/EOS tokens added by server |
| SGLang | Identical (uses HF tokenizer) | 0 | Rare discrepancies |
| TRT-LLM | Tokenizer version mismatch | Variable | TRT-LLM may use a compiled tokenizer |
| OpenAI API | tiktoken vs HF tokenizer | +2 to +10 | Special tokens, system prompt injection |

### 9.6 The `--use-server-token-count` Escape Hatch

When tokenizer mismatch is confirmed, AIPerf's `--use-server-token-count` flag
bypasses client tokenization entirely. This:
- Uses `usage.prompt_tokens` as ISL
- Uses `usage.completion_tokens` as OSL
- Disables all usage diff metrics (they would all be 0%)
- Provides accurate token counts at the cost of losing tokenizer-independent validation

The decision to use this flag should be based on the root cause analysis above:
if the bias is a fixed offset from special tokens, it may be better to adjust
the client tokenizer configuration rather than lose independent validation.

---

## 10. Goodput vs Raw Throughput Correlation

### 10.1 Definitions

**Raw throughput**: All tokens produced, regardless of quality or SLO compliance.

```
raw_throughput = output_token_throughput
              = sum(output_tokens) / benchmark_duration
```

**Goodput**: Only requests meeting SLO thresholds, divided by time.

```
goodput = good_request_count / benchmark_duration
```

AIPerf computes goodput in `GoodputMetric` using `GoodRequestCountMetric`
which counts requests satisfying all configured SLO thresholds
(`--goodput "time_to_first_token:100 inter_token_latency:3.40"`).

**Effective throughput**: Time-weighted instantaneous throughput from sweep-line
analysis (`effective_throughput` in `SweepMetricSpec`).

### 10.2 The Goodput-Throughput Gap

```
goodput_ratio = goodput / request_throughput
token_goodput_ratio = good_request_token_throughput / raw_throughput
```

| Scenario | goodput_ratio | Interpretation |
|---|---|---|
| Close to 1.0 | System meets SLOs for nearly all requests |
| 0.5 -- 0.9 | Moderate SLO violations -- tail latency issues |
| < 0.5 | System is overloaded -- raw throughput is misleading |
| 0.0 | No requests meet SLOs -- throughput number is meaningless |

### 10.3 Server-Side Context

The server does not know client SLOs. But server metrics provide context for
why goodput may differ from raw throughput:

```
High raw throughput + Low goodput:
├── vllm:num_requests_waiting consistently high → Queue saturation
│   → Reduce concurrency or add capacity
├── vllm:kv_cache_usage_perc approaching 1.0 → Memory pressure
│   → Reduce max_model_len or batch size
├── vllm:num_preemptions > 0 → Active preemption
│   → Requests being paused/restarted → latency spikes
└── vllm:inter_token_latency_seconds p99 >> p50 → Tail latency
    → Batch interference at high concurrency
```

### 10.4 Token-Level Goodput

The current `GoodputMetric` counts requests, not tokens. A token-level goodput
would be:

```
token_goodput = sum(output_tokens for good_requests) / duration
token_waste = sum(output_tokens for bad_requests) / duration
raw_throughput = token_goodput + token_waste
```

This reveals how many tokens were "wasted" on requests that ultimately did not
meet SLOs. A high `token_waste` ratio with a system that is SLO-bound
indicates the server is doing useful work but too slowly.

### 10.5 Cross-Domain Correlation

Combining client goodput with server queue metrics provides a diagnostic
correlation:

```python
def correlate_goodput_with_server_state(
    goodput_ratio: float,
    server_metrics: ServerMetricsResults,
) -> GoodputCorrelation:
    """Correlate goodput ratio with server-side state indicators.

    Identifies the server-side bottleneck most likely causing SLO violations.
    """
    queue_depth_avg = get_gauge_avg(server_metrics, "vllm:num_requests_waiting")
    cache_usage_max = get_gauge_max(server_metrics, "vllm:kv_cache_usage_perc")
    preemptions = get_counter_total(server_metrics, "vllm:num_preemptions")
    running_avg = get_gauge_avg(server_metrics, "vllm:num_requests_running")

    bottlenecks = []

    if cache_usage_max and cache_usage_max > 0.9:
        bottlenecks.append(Bottleneck(
            type="KV_CACHE_PRESSURE",
            severity="HIGH",
            metric="vllm:kv_cache_usage_perc",
            value=cache_usage_max,
            recommendation="Reduce max_model_len or increase gpu_memory_utilization",
        ))

    if preemptions and preemptions > 0:
        bottlenecks.append(Bottleneck(
            type="PREEMPTION",
            severity="HIGH",
            metric="vllm:num_preemptions",
            value=preemptions,
            recommendation="Reduce concurrency or increase memory",
        ))

    if queue_depth_avg and queue_depth_avg > running_avg * 2:
        bottlenecks.append(Bottleneck(
            type="QUEUE_SATURATION",
            severity="MEDIUM",
            metric="vllm:num_requests_waiting",
            value=queue_depth_avg,
            recommendation="Reduce request rate or add inference replicas",
        ))

    return GoodputCorrelation(
        goodput_ratio=goodput_ratio,
        bottlenecks=bottlenecks,
        primary_bottleneck=bottlenecks[0] if bottlenecks else None,
    )
```

### 10.6 Effective Throughput vs Server Throughput

AIPerf's sweep-line `effective_throughput` is a time-weighted throughput that
accounts for concurrency variation over time. Comparing it against the server's
steady-state generation rate reveals client-side overhead:

```
client_overhead_ratio = effective_throughput / server_gen_rate
```

If `client_overhead_ratio < 1.0`, the client is observing lower throughput than
the server produces. Possible causes:
- Network latency between client and server (tokens produced but not yet delivered)
- Client-side processing overhead (tokenization, record creation)
- Measurement window differences

---

## 11. Unified Discrepancy Framework

### 11.1 Layered Discrepancy Model

Discrepancies exist at multiple layers. Each layer has its own measurement
point and reconciliation formula:

```
Layer 0: Token Identity
  Client tokenizer output == Server tokenizer output?
  → Measured by: usage_*_diff_pct metrics (existing)

Layer 1: Request Token Counts
  Client per-request tokens == API usage field tokens?
  → Measured by: UsageDiscrepancyCountMetric (existing)

Layer 2: Aggregate Token Counts
  sum(client per-request tokens) == server counter delta?
  → Measured by: ReconciliationResult (proposed, Section 7)

Layer 3: Throughput Rates
  client_tokens / client_duration == server_tokens / server_duration?
  → Measured by: ThroughputDiscrepancyResult (proposed, Section 3)

Layer 4: Effective Work
  tokens_received_by_client == tokens_computed_by_server?
  → Measured by: SpeculativeDecodingResult (proposed, Section 4)
  → Measured by: PrefixCachingResult (proposed, Section 6)

Layer 5: Useful Work
  tokens_meeting_SLO / total_tokens?
  → Measured by: GoodputCorrelation (proposed, Section 10)
```

### 11.2 Comprehensive Discrepancy Report

A single report combining all layers:

```python
@dataclass(frozen=True)
class DiscrepancyReport:
    """Comprehensive client-server discrepancy analysis."""

    # Layer 0-1: Per-request token mismatch (existing metrics)
    usage_discrepancy_count: int
    usage_discrepancy_pct: float
    tokenizer_mismatch_analysis: TokenizerMismatchAnalysis | None

    # Layer 2: Aggregate reconciliation
    output_token_reconciliation: ReconciliationResult | None
    input_token_reconciliation: ReconciliationResult | None
    request_count_reconciliation: RequestCountReconciliation | None

    # Layer 3: Throughput discrepancy
    output_throughput_discrepancy: ThroughputDiscrepancyResult | None
    prompt_throughput_discrepancy: ThroughputDiscrepancyResult | None

    # Layer 4: Server-internal phenomena
    speculative_decoding: SpeculativeDecodingResult | None
    prefix_caching: PrefixCachingResult | None
    reasoning_accounting: ReasoningAccountingResult | None

    # Layer 5: Quality correlation
    goodput_correlation: GoodputCorrelation | None
    iteration_efficiency: IterationEfficiencyResult | None

    # Meta
    backend: str
    warnings: list[str]

    @property
    def has_significant_discrepancy(self) -> bool:
        """Check if any layer reports a significant discrepancy."""
        if self.output_throughput_discrepancy and \
                self.output_throughput_discrepancy.diff_pct > 10:
            return True
        if self.usage_discrepancy_pct > 20:
            return True
        if self.speculative_decoding and self.speculative_decoding.likely:
            return True
        return False
```

### 11.3 Discrepancy Severity Matrix

| Layer | Metric | Low (Informational) | Medium (Warning) | High (Error) |
|---|---|---|---|---|
| 0 | Token identity | < 2% diff | 2-10% diff | > 10% diff |
| 1 | Per-request count | < 5% requests flagged | 5-20% flagged | > 20% flagged |
| 2 | Aggregate reconciliation | < 2% diff | 2-10% diff | > 10% diff |
| 3 | Throughput rate | < 5% ratio diff | 5-15% ratio diff | > 15% ratio diff |
| 4 | Speculative waste | < 10% waste | 10-30% waste | > 30% waste |
| 5 | Goodput ratio | > 0.9 | 0.5-0.9 | < 0.5 |

### 11.4 Automated Diagnosis Flow

```
1. Collect all available metrics (client + server)
2. For each layer:
   a. Compute discrepancy metric
   b. Classify severity
   c. If HIGH, run root cause analysis for that layer
3. Cross-correlate findings:
   - If Layer 3 (throughput) discrepancy + Layer 4 (speculative) detected
     → Report "Speculative decoding explains throughput gap"
   - If Layer 2 (aggregate) discrepancy + Layer 0 (tokenizer) bias
     → Report "Tokenizer mismatch accumulates to N tokens over M requests"
   - If Layer 5 (goodput) low + server queue high
     → Report "SLO violations caused by queue saturation"
4. Generate unified report with primary diagnosis
```

---

## 12. AIPerf Implementation Guidance

### 12.1 Architecture Overview

The analyses in this document fall into three implementation categories:

**Category A: Standalone post-processing (no new data collection)**

These analyses use data already collected by AIPerf. They can be implemented
as post-processors or exporters that operate on existing `MetricsSummary` and
`ServerMetricsResults` objects.

| Analysis | Inputs | Output |
|---|---|---|
| Throughput discrepancy (Section 3) | `output_token_throughput` + `vllm:generation_tokens` stats.rate | Ratio + classification |
| Aggregate reconciliation (Section 7) | Sum of per-request tokens + server counter stats.total | Diff + decision tree |
| Iteration efficiency (Section 8) | `vllm:iteration_tokens_total` stats | Efficiency metrics |
| Goodput correlation (Section 10) | `goodput` + server gauge/counter stats | Bottleneck identification |

**Category B: New derived metrics (extend MetricRegistry)**

These require new metric classes in `src/aiperf/metrics/types/`.

| Metric | Type | Formula |
|---|---|---|
| `signed_usage_prompt_diff` | BaseRecordMetric | `usage_prompt_tokens - client_ISL` (signed, not absolute) |
| `signed_usage_completion_diff` | BaseRecordMetric | `usage_completion_tokens - client_OSL` (signed) |
| `reasoning_visibility_ratio` | BaseRecordMetric | `client_reasoning / usage_reasoning` |
| `reasoning_overhead_pct` | BaseDerivedMetric | `(1 - visible_output_tps / total_completion_tps) * 100` |

**Category C: Cross-domain correlation (new analysis module)**

These require joining data from `MetricsAccumulator` and
`ServerMetricsAccumulator`. Implementation options:

1. **Post-export analysis**: Run after both accumulators export, operate on
   the exported JSON/summary objects. Simplest, no cross-accumulator coupling.

2. **SummaryContext extension**: Pass server metrics results into the
   `SummaryContext` so the `MetricsAccumulator` can access them during
   summarization. More coupled but enables per-request correlation.

3. **Dedicated CorrelationAnalyzer**: A new post-processor that receives both
   accumulator outputs and produces the `DiscrepancyReport`. Clean separation
   of concerns.

**Recommended**: Option 3 (CorrelationAnalyzer) because it follows the existing
`AnalyzerProtocol` pattern used by `SteadyStateAnalyzer`.

### 12.2 Data Flow

```
RecordProcessor
  ├── MetricsAccumulator
  │     ├── per-request metrics (ISL, OSL, usage_*, reasoning_*, diffs)
  │     └── aggregate metrics (throughput, goodput, sweep stats)
  │
  └── ServerMetricsAccumulator
        └── Prometheus time-series (counters, gauges, histograms)

RecordsManager
  ├── export_results() → MetricsSummary + ServerMetricsResults
  └── SummaryContext (both outputs available)

DiscrepancyAnalyzer (new)
  ├── Input: MetricsSummary + ServerMetricsResults
  ├── Runs all Section 3-10 analyses
  └── Output: DiscrepancyReport

ConsoleDiscrepancyExporter (new)
  ├── Input: DiscrepancyReport
  └── Output: Rich panel with findings + recommendations
```

### 12.3 Metric Access Patterns

**From MetricsSummary (client-side)**:

```python
# Per-request values (from ColumnStore via MetricResult)
output_tps = summary.results[OutputTokenThroughputMetric.tag]  # MetricResult
total_output = summary.results[TotalOutputSequenceLengthMetric.tag]
total_input = summary.results[TotalInputSequenceLengthMetric.tag]
duration = summary.results[BenchmarkDurationMetric.tag]
goodput = summary.results.get(GoodputMetric.tag)

# Usage tokens (when --use-server-token-count is NOT set)
total_usage_prompt = summary.results.get(TotalUsagePromptTokensMetric.tag)
total_usage_completion = summary.results.get(TotalUsageCompletionTokensMetric.tag)
usage_discrepancy = summary.results.get(UsageDiscrepancyCountMetric.tag)
```

**From ServerMetricsResults (server-side)**:

```python
# Access pattern for server metrics
for endpoint_display, endpoint_summary in results.endpoint_summaries.items():
    metrics = endpoint_summary.metrics

    # Counter: generation tokens
    gen_tokens = metrics.get("vllm:generation_tokens")
    if gen_tokens:
        for series in gen_tokens.series:
            gen_rate = series.stats.rate           # tokens/sec
            gen_total = series.stats.total         # total delta

    # Histogram: iteration tokens
    iter_tokens = metrics.get("vllm:iteration_tokens_total")
    if iter_tokens:
        for series in iter_tokens.series:
            avg_batch = series.stats.avg           # avg tokens/iteration
            iteration_count = series.stats.count   # total iterations

    # Gauge: cache usage
    cache = metrics.get("vllm:kv_cache_usage_perc")
    if cache:
        for series in cache.series:
            max_usage = series.stats.max           # peak cache utilization
```

### 12.4 Export Integration

The discrepancy report should be available in all export formats:

**Console**: A Rich panel similar to `ConsoleOSLMismatchExporter` and
`ConsoleUsageDiscrepancyExporter`, but covering all layers. Only shown when
significant discrepancies are detected.

**JSON**: A `discrepancy_analysis` section in the profile export JSON
with structured results from each layer.

**CSV**: Summary row with key discrepancy metrics
(throughput_ratio, reconciliation_diff_pct, etc.).

### 12.5 Configuration

New environment variables following the existing pattern:

| Variable | Default | Description |
|---|---|---|
| `AIPERF_DISCREPANCY_THROUGHPUT_THRESHOLD_PCT` | 5.0 | Throughput ratio diff % to flag |
| `AIPERF_DISCREPANCY_RECONCILIATION_THRESHOLD_PCT` | 5.0 | Aggregate token diff % to flag |
| `AIPERF_DISCREPANCY_ENABLED` | true | Enable/disable correlation analysis |

These should live in a `_DiscrepancySettings` class following the existing
`_SteadyStateSettings` pattern in `src/aiperf/common/environment.py`.

### 12.6 Testing Strategy

**Unit tests** (Category B metrics):

```python
@pytest.mark.parametrize("usage_prompt,client_isl,expected_signed_diff", [
    (105, 100, 5),        # Server counts more (special tokens)
    (95, 100, -5),        # Client counts more
    (100, 100, 0),        # Perfect match
    (200, 100, 100),      # Severe mismatch
])
def test_signed_usage_prompt_diff(usage_prompt, client_isl, expected_signed_diff):
    ...
```

**Unit tests** (Category A/C analyses):

```python
def test_throughput_discrepancy_healthy():
    result = detect_throughput_discrepancy(
        client_output_tps=500.0,
        server_gen_rate=510.0,
        threshold_pct=5.0,
    )
    assert result.category == "HEALTHY"
    assert result.ratio == pytest.approx(1.02, abs=0.01)

def test_throughput_discrepancy_speculative():
    result = detect_throughput_discrepancy(
        client_output_tps=500.0,
        server_gen_rate=750.0,
    )
    assert result.category == "SERVER_EXCESS"
    assert result.ratio == pytest.approx(1.5, abs=0.01)
```

**Synthetic validation** (extending the existing synthetic test suite):

```python
DISCREPANCY_PROFILES = [
    # (name, client_tps, server_tps, expected_category)
    ("no_discrepancy", 500, 505, "HEALTHY"),
    ("speculative_decoding", 500, 750, "SERVER_EXCESS"),
    ("tokenizer_overcounting", 500, 480, "CLIENT_EXCESS"),
    ("prefix_caching_active", 1000, 600, "CLIENT_EXCESS"),  # prompt tokens
]
```

### 12.7 Implementation Priority

| Priority | Analysis | Effort | Value | Dependencies |
|---|---|---|---|---|
| P1 | Throughput discrepancy (Section 3) | Low | High | ServerMetricsResults access |
| P1 | Aggregate reconciliation (Section 7) | Low | High | Same |
| P2 | Tokenizer mismatch classification (Section 9) | Medium | High | Signed diff metrics |
| P2 | Speculative decoding detection (Section 4) | Low | Medium | Throughput discrepancy |
| P3 | Goodput-server correlation (Section 10) | Medium | Medium | Goodput metric + server stats |
| P3 | Iteration efficiency (Section 8) | Low | Medium | vLLM-specific |
| P4 | Prefix caching impact (Section 6) | Medium | Medium | Backend-specific detection |
| P4 | Reasoning token accounting (Section 5) | Medium | Low-Medium | Reasoning model usage |

---

## 13. References

### Academic

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from
   Transformers via Speculative Decoding." *ICML 2023.*
   — Foundation for speculative decoding analysis.

2. Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with
   Speculative Sampling." *arXiv:2302.01318.*
   — Speculative sampling acceptance rate theory.

3. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention." *SOSP 2023.*
   — vLLM architecture, KV-cache management, and prefix caching.

4. Zheng, L., et al. (2024). "SGLang: Efficient Execution of Structured
   Language Model Programs." *arXiv:2312.07104.*
   — RadixAttention prefix caching, SGLang architecture.

5. Agrawal, A., et al. (2024). "Taming Throughput-Latency Tradeoff in LLM
   Inference with Sarathi-Serve." *OSDI 2024.*
   — Chunked prefill, iteration-level scheduling analysis.

### Industry

6. MLPerf Inference Benchmark Suite, MLCommons.
   https://mlcommons.org/benchmarks/inference-datacenter/
   — Standard benchmark methodology and reporting requirements.

7. NVIDIA TensorRT-LLM Documentation.
   https://nvidia.github.io/TensorRT-LLM/
   — TRT-LLM metric definitions and server architecture.

8. vLLM Metrics Documentation.
   https://docs.vllm.ai/en/latest/serving/metrics.html
   — Prometheus metric definitions for vLLM.

9. Dean, J. & Barroso, L.A. (2013). "The Tail at Scale." *Communications
   of the ACM, 56(2).*
   — Tail latency analysis, goodput vs throughput in distributed systems.

### AIPerf Internal

10. `src/aiperf/metrics/types/osl_mismatch_metrics.py` — OSL mismatch detector
11. `src/aiperf/metrics/types/usage_diff_metrics.py` — Usage discrepancy detector
12. `src/aiperf/server_metrics/accumulator.py` — Server metrics accumulator
13. `src/aiperf/server_metrics/data_collector.py` — Prometheus collection
14. `src/aiperf/analysis/sweep.py` — Sweep-line throughput algorithms
15. `src/aiperf/metrics/types/goodput_metric.py` — Goodput metric
16. `docs/server_metrics/server_metrics_reference.md` — Server metric definitions
