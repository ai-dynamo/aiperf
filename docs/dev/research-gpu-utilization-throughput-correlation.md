<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Utilization & Throughput Efficiency Correlation

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Background: What GPU Utilization Actually Measures](#2-background-what-gpu-utilization-actually-measures)
3. [GPU Utilization as Throughput Ceiling Indicator](#3-gpu-utilization-as-throughput-ceiling-indicator)
4. [The Roofline Model for LLM Inference](#4-the-roofline-model-for-llm-inference)
5. [Power-Performance Curve](#5-power-performance-curve)
6. [Memory Bandwidth Bottleneck Detection](#6-memory-bandwidth-bottleneck-detection)
7. [Batch Efficiency Analysis](#7-batch-efficiency-analysis)
8. [Thermal Throttling Detection](#8-thermal-throttling-detection)
9. [Energy Efficiency Metrics](#9-energy-efficiency-metrics)
10. [Multi-GPU Scaling](#10-multi-gpu-scaling)
11. [Temporal Phase Detection](#11-temporal-phase-detection)
12. [Correlation Framework: Joining Two Time Domains](#12-correlation-framework-joining-two-time-domains)
13. [AIPerf Implementation Guidance](#13-aiperf-implementation-guidance)
14. [Formulas Reference](#14-formulas-reference)
15. [Open Questions](#15-open-questions)
16. [References](#16-references)

---

## 1. Motivation

LLM inference benchmarking currently produces two largely independent metric
streams: **client-side performance** (throughput, latency, concurrency) and
**GPU telemetry** (utilization, power, temperature, memory). Each stream is
informative on its own, but their correlation reveals insights that neither
stream can provide alone:

- **Saturation detection**: throughput plateaus while GPU utilization climbs
  from 70% to 95% — the server is approaching its ceiling.
- **Bottleneck discrimination**: high `mem_utilization` with moderate
  `gpu_utilization` reveals a memory-bandwidth-bound decode phase, not a
  compute-bound one.
- **Efficiency optimization**: the operating point where tokens/second/watt is
  maximized is often far below peak throughput.
- **Anomaly diagnosis**: a sudden throughput drop correlated with a temperature
  spike points to thermal throttling, not a software regression.

Today, AIPerf collects both streams in parallel — the `GPUTelemetryAccumulator`
stores hierarchical numpy-backed time series from DCGM/pynvml, while the
`MetricsAccumulator` stores per-request records in a `ColumnStore` with
vectorized sweep-line algorithms producing instantaneous throughput and
concurrency curves. Both accumulators share the same nanosecond-precision
wall-clock timeline. The infrastructure for correlation is present; the
analysis is not.

This document researches the statistical methods, domain models, and
implementation strategies for cross-stream correlation analysis in AIPerf.

---

## 2. Background: What GPU Utilization Actually Measures

### 2.1 DCGM Field Semantics

The three utilization metrics reported by DCGM have fundamentally different
meanings that are frequently confused:

```
+-------------------------------------------------------------------+
|  Metric              |  DCGM Field              |  What It Means  |
+-------------------------------------------------------------------+
|  gpu_utilization     |  DCGM_FI_DEV_GPU_UTIL    |  % of time at   |
|  (0-100%)            |                          |  least one      |
|                      |                          |  kernel active   |
+-------------------------------------------------------------------+
|  sm_utilization      |  DCGM_FI_PROF_SM_ACTIVE  |  Fraction of    |
|  (0-100%)            |                          |  SMs that have   |
|                      |                          |  at least one    |
|                      |                          |  active warp     |
+-------------------------------------------------------------------+
|  mem_utilization     |  DCGM_FI_DEV_MEM_COPY_   |  % of time      |
|  (0-100%)            |  UTIL                    |  memory bus      |
|                      |                          |  is active       |
+-------------------------------------------------------------------+
```

**Critical distinction**: `gpu_utilization` is a **temporal** occupancy metric.
A single tiny kernel running for 99% of the sample period yields
`gpu_utilization = 99%` even if only 1 of 144 SMs is active. In contrast,
`sm_utilization` measures **spatial** occupancy — how many SMs are actually
doing work. For LLM inference:

- `gpu_utilization` saturates early (as soon as the inference engine keeps the
  GPU busy, which happens at low batch sizes).
- `sm_utilization` continues climbing as batch size increases, reflecting actual
  compute parallelism.
- `mem_utilization` tracks the memory subsystem independently.

This means `sm_utilization` is the **primary correlate** for throughput
analysis, while `gpu_utilization` is more useful as a binary "is the GPU
idle?" indicator.

### 2.2 Sampling Rates and Aliasing

DCGM reports utilization as averages over the **sample period** (typically
100ms-1s, controlled by `DCGM_FI_DEV_GPU_UTIL` update frequency). AIPerf's
default collection interval is `Environment.GPU.COLLECTION_INTERVAL` seconds.
This creates two sources of temporal aliasing:

1. **DCGM averaging window**: A burst of 50ms compute followed by 50ms idle
   appears as "50% utilization" — the temporal structure is lost.
2. **Collection interval**: If AIPerf collects every 1s but DCGM averages over
   100ms windows, we're sampling a 100ms average at 1s intervals, potentially
   missing short utilization spikes.

For correlation analysis, both streams should be aligned to the **coarser**
resolution. In practice, this means resampling the sweep-line throughput
curves to match DCGM's collection cadence (see Section 12).

### 2.3 Counter vs. Gauge Metrics

GPU telemetry metrics fall into two categories (defined in
`GPU_TELEMETRY_COUNTER_METRICS` in `constants.py`):

- **Gauges** (`gpu_utilization`, `sm_utilization`, `mem_utilization`,
  `gpu_power_usage`, `gpu_memory_used`, `gpu_temperature`): Instantaneous
  values. Statistics (mean, percentiles) are meaningful.
- **Counters** (`energy_consumption`, `xid_errors`, `power_violation`):
  Monotonically increasing cumulative values. Only deltas are meaningful.

Correlation analysis must handle these differently. Gauge metrics can be
directly correlated with throughput time series. Counter metrics require
delta computation (change per time interval) before correlation.

---

## 3. GPU Utilization as Throughput Ceiling Indicator

### 3.1 The Saturation Curve

As client-side load increases (more concurrent requests), GPU utilization
increases and throughput increases — up to a point. Beyond saturation, adding
more load increases queuing delay without increasing throughput. The
characteristic shape is:

```
  throughput                           gpu_utilization (sm)
  (tok/s)                              (%)
    ^                                    ^
    |          .......                   |               ............
    |       ../                          |            ../
    |     ./                             |          ./
    |   ./                               |        ./
    |  /                                 |      ./
    | /                                  |    ./
    |/                                   |  ./
    +---------> concurrency              +---------> concurrency
      "knee"                               "knee"
```

The **knee** of the throughput curve — where marginal throughput gain per unit
of concurrency drops below a threshold — corresponds to a characteristic GPU
utilization level. Empirically, for transformer inference:

- **Prefill phase** (compute-bound): knee at `sm_utilization` ~ 70-85%
- **Decode phase** (memory-bandwidth-bound): knee at `mem_utilization` ~ 60-80%
  with `sm_utilization` often still below 50%

### 3.2 Detecting Saturation

Given aligned time series of throughput `T(t)` and sm_utilization `U(t)`:

```
Marginal throughput efficiency:

    MTE(t) = dT/dU = [T(t+dt) - T(t)] / [U(t+dt) - U(t)]
```

When `MTE(t)` drops below a threshold (e.g., 10% of its initial value), the
system is saturated. This is the "diminishing returns" signal.

**Pseudocode**:

```python
def detect_saturation_point(
    throughput_ts: NDArray[np.float64],
    throughput_vals: NDArray[np.float64],
    util_ts: NDArray[np.float64],
    util_vals: NDArray[np.float64],
    mte_threshold_fraction: float = 0.10,
) -> float | None:
    """Detect the timestamp where throughput saturates relative to GPU utilization.

    Returns the timestamp of the saturation point, or None if no saturation detected.
    """
    # Align time series to common grid (DCGM cadence)
    common_ts = util_ts  # DCGM is the coarser signal
    tput_aligned = step_lookup(throughput_ts, throughput_vals, common_ts)
    util_aligned = util_vals

    # Compute marginal throughput efficiency
    dt = np.diff(tput_aligned)
    du = np.diff(util_aligned)

    # Avoid division by zero where utilization is flat
    valid = np.abs(du) > 0.5  # at least 0.5% utilization change
    mte = np.zeros_like(dt)
    mte[valid] = dt[valid] / du[valid]

    # Smooth with rolling window to reduce noise
    window = min(5, len(mte) // 3)
    if window > 0:
        kernel = np.ones(window) / window
        mte_smooth = np.convolve(mte, kernel, mode="same")
    else:
        mte_smooth = mte

    # Find initial MTE (first quartile of positive values)
    positive_mte = mte_smooth[mte_smooth > 0]
    if len(positive_mte) < 3:
        return None
    initial_mte = np.percentile(positive_mte, 75)

    # Saturation = first sustained drop below threshold
    threshold = initial_mte * mte_threshold_fraction
    below = mte_smooth < threshold
    # Require 3 consecutive points below threshold
    for i in range(len(below) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            return float(common_ts[i])

    return None
```

### 3.3 Regime Classification

Cross-referencing `sm_utilization` and `mem_utilization` at the saturation
point classifies the bottleneck:

| sm_util | mem_util | Regime | Typical Phase |
|---------|----------|--------|---------------|
| High (>70%) | Low (<50%) | Compute-bound | Prefill with large batch |
| Low (<50%) | High (>60%) | Memory-bandwidth-bound | Decode (autoregressive) |
| High (>70%) | High (>60%) | Both saturated | High-batch decode or long-context prefill |
| Low (<40%) | Low (<40%) | Neither saturated | Queuing elsewhere (CPU, network, scheduling) |

---

## 4. The Roofline Model for LLM Inference

### 4.1 Classical Roofline

The roofline model (Williams, Waterman, Patterson 2009) bounds achievable
performance by two hardware limits: peak compute (FLOP/s) and peak memory
bandwidth (bytes/s). The crossover point — the **ridge point** — is
determined by the operational intensity of the workload:

```
  Performance                    Roofline
  (FLOP/s)                      _______________
    ^                          /
    |                         /  peak compute
    |                        /
    |                       /
    |                      /
    |                     /
    |                    /
    |                   /
    |  peak BW slope   /
    | (bytes/s)       /
    +-----------------------------> Operational Intensity
                     ^               (FLOP/byte)
                ridge point
```

For LLM inference, the operational intensity varies dramatically between
phases:

### 4.2 Prefill Phase Roofline

The prefill phase processes the entire input prompt in one forward pass.
For a transformer with:
- `L` layers, `H` hidden dimension, `S` sequence length, `V` vocab size
- Batch size `B`

**Compute**: ~ `2 * B * S * H^2 * L * 4` FLOP (approximate, includes
attention + FFN + layer norm)

**Memory traffic**: ~ `2 * H^2 * L * 4` bytes (model weights, read once
per token)

**Operational intensity**: ~ `B * S` FLOP/byte (increases linearly with
batch size and sequence length)

At typical LLM scales (H=4096, L=32, S=2048, B=8):
- OI ~ 16,384 FLOP/byte
- This is well above the ridge point for any modern GPU
- Prefill is **compute-bound** at reasonable batch sizes

### 4.3 Decode Phase Roofline

The decode phase generates one token at a time per request, reading the
full model weights for each token:

**Compute per token**: ~ `2 * B * H^2 * L * 4` FLOP

**Memory traffic per token**: ~ `2 * H^2 * L * 4` bytes (same weights, but
S=1 per step)

**Operational intensity**: ~ `B` FLOP/byte (depends ONLY on batch size)

At batch size B=1: OI ~ 1 FLOP/byte — firmly memory-bandwidth-bound.
At batch size B=128: OI ~ 128 FLOP/byte — approaching the ridge point.

This is why **batching is critical for decode efficiency** and why
`mem_utilization` is the primary constraint during decode.

### 4.4 Mapping to AIPerf Metrics

The roofline model maps to AIPerf observables:

```
  Roofline Concept          AIPerf Observable
  ──────────────────────    ─────────────────────────────────
  Operational Intensity     Not directly measured (derived from
                            effective_concurrency + ISL/OSL)
  Achieved Compute          sm_utilization (proxy)
  Achieved Bandwidth        mem_utilization (proxy)
  Ridge Point               sm_util / mem_util crossover
  Compute Ceiling           sm_utilization approaching 100%
  Bandwidth Ceiling         mem_utilization approaching 100%
```

**Pseudocode for roofline classification**:

```python
@dataclass
class RooflineClassification:
    """Per-sample roofline regime classification."""
    timestamp_ns: int
    regime: str  # "compute_bound", "memory_bound", "balanced", "underutilized"
    sm_utilization: float
    mem_utilization: float
    effective_throughput: float
    operational_intensity_proxy: float  # concurrency * avg_osl


def classify_roofline_regime(
    sm_util: float,
    mem_util: float,
    sm_threshold: float = 70.0,
    mem_threshold: float = 60.0,
) -> str:
    """Classify the operating regime from utilization pair."""
    if sm_util >= sm_threshold and mem_util < mem_threshold:
        return "compute_bound"
    elif sm_util < sm_threshold and mem_util >= mem_threshold:
        return "memory_bound"
    elif sm_util >= sm_threshold and mem_util >= mem_threshold:
        return "balanced"
    else:
        return "underutilized"
```

### 4.5 Roofline ASCII Model for LLM Inference

A more detailed model that accounts for prefill vs decode:

```
  Effective Throughput (tok/s)
    ^
    |
    |  Prefill Roofline (compute ceiling)
    |  ────────────────────────────────────────
    |                                  /
    |                                 /   Prefill: OI scales with B*S
    |                                /
    |                               /
    |  ────────────────────────────/──── Decode Roofline (BW ceiling)
    |                             /
    |                            /   Decode: OI scales with B only
    |                           /
    |                          /
    |                         /
    |                        /
    |  Memory BW slope      /
    |                      /
    +──────────────────────────────────────> Operational Intensity
                                              (FLOP/byte)
         B=1    B=8   B=32   B=128         B=1024
        (decode)              (decode)      (prefill)
```

The key insight: **prefill and decode have separate rooflines** because their
operational intensities differ by a factor of ~S (sequence length). A single
GPU serving mixed prefill+decode traffic operates in a regime that alternates
between the two curves depending on the batch scheduler's decisions.

---

## 5. Power-Performance Curve

### 5.1 The Sublinear Relationship

GPU power consumption scales sublinearly with throughput. Doubling the
throughput does NOT double the power — but it more than doubles the marginal
power per token. This creates a characteristic diminishing-returns curve:

```
  Power (W)
    ^
    |                              .........
    |                         ....·
    |                     ...·
    |                  ..·
    |               ..·
    |            ..·
    |          .·
    |        .·
    |      .·         Power scales ~T^0.5 to T^0.7
    |    .·           (sublinear)
    |  .·
    | ·
    +─────────────────────────────> Throughput (tok/s)

  Tokens/Watt
    ^
    | ·
    | ·.
    |  ·..
    |    ·...
    |       ·....
    |            ·.......
    |                    ·...............
    |                                   ·..........
    +─────────────────────────────> Throughput (tok/s)
```

The **tokens-per-watt** metric decreases monotonically with throughput.
The optimal operating point depends on the cost function:

- **Minimize TCO**: operate at the throughput where `throughput / power` is
  maximized (the tangent from the origin to the power curve).
- **Maximize throughput**: operate at the knee (ignoring power cost).
- **SLA-constrained**: operate at the highest throughput that meets latency
  SLAs, then evaluate power cost at that point.

### 5.2 Power-Performance Ratio

```
PPR(t) = effective_throughput(t) / gpu_power_usage(t)     [tokens/s/W]
```

This is the instantaneous power-performance ratio. Over the steady-state
window:

```
PPR_steady = avg(effective_throughput) / avg(gpu_power_usage)     [tokens/s/W]
```

Or equivalently using energy:

```
PPR_steady = total_tokens / energy_delta     [tokens/J]
```

Where `energy_delta` is the cumulative `energy_consumption` counter delta
over the steady-state window.

### 5.3 Pseudocode

```python
@dataclass(frozen=True)
class PowerPerformanceResult:
    """Power-performance analysis result."""
    avg_throughput_tok_s: float
    avg_power_w: float
    tokens_per_watt: float
    tokens_per_joule: float
    energy_delta_j: float
    total_tokens: int
    optimal_operating_pct: float  # utilization at max tokens/watt


def compute_power_performance(
    throughput_ts: NDArray[np.float64],
    throughput_vals: NDArray[np.float64],
    power_ts: NDArray[np.int64],
    power_vals: NDArray[np.float64],
    energy_start_mj: float,
    energy_end_mj: float,
    total_tokens: int,
    window_start_ns: int,
    window_end_ns: int,
) -> PowerPerformanceResult:
    """Compute power-performance metrics over the steady-state window.

    Args:
        throughput_ts: Sweep-line throughput timestamps (ns).
        throughput_vals: Throughput values (tokens/ns).
        power_ts: DCGM power timestamps (ns).
        power_vals: Power values (watts).
        energy_start_mj: Cumulative energy at window start (MJ).
        energy_end_mj: Cumulative energy at window end (MJ).
        total_tokens: Total output tokens in window.
        window_start_ns: Steady-state window start.
        window_end_ns: Steady-state window end.
    """
    # Time-weighted average throughput (tokens/s)
    tput_stats = compute_time_weighted_stats(
        throughput_ts, throughput_vals * NANOS_PER_SECOND,
        float(window_start_ns), float(window_end_ns),
    )

    # Time-weighted average power (W) — DCGM gauge, use segment durations
    power_mask = (power_ts >= window_start_ns) & (power_ts < window_end_ns)
    windowed_power = power_vals[power_mask]
    windowed_power_ts = power_ts[power_mask]

    if len(windowed_power) < 2:
        avg_power = float(np.mean(windowed_power)) if len(windowed_power) > 0 else 0.0
    else:
        # Duration-weighted mean
        durations = np.diff(windowed_power_ts).astype(np.float64)
        avg_power = float(np.average(windowed_power[:-1], weights=durations))

    # Energy delta (MJ -> J)
    energy_delta_j = (energy_end_mj - energy_start_mj) * 1e6

    tokens_per_watt = tput_stats.avg / avg_power if avg_power > 0 else 0.0
    tokens_per_joule = total_tokens / energy_delta_j if energy_delta_j > 0 else 0.0

    return PowerPerformanceResult(
        avg_throughput_tok_s=tput_stats.avg,
        avg_power_w=avg_power,
        tokens_per_watt=tokens_per_watt,
        tokens_per_joule=tokens_per_joule,
        energy_delta_j=energy_delta_j,
        total_tokens=total_tokens,
        optimal_operating_pct=0.0,  # computed via sweep below
    )
```

### 5.4 Finding the Optimal Operating Point

Given a sweep over concurrency levels (from a load-ramp benchmark), the
optimal operating point maximizes `throughput / power`:

```python
def find_optimal_operating_point(
    concurrency_levels: NDArray[np.float64],
    throughput_at_level: NDArray[np.float64],
    power_at_level: NDArray[np.float64],
) -> int:
    """Return index of the concurrency level with maximum tokens/watt."""
    efficiency = np.zeros_like(throughput_at_level)
    nonzero = power_at_level > 0
    efficiency[nonzero] = throughput_at_level[nonzero] / power_at_level[nonzero]
    return int(np.argmax(efficiency))
```

This requires a load-ramp or multi-concurrency benchmark. For a single
fixed-concurrency run, the power-performance ratio is a point estimate,
not an optimization.

---

## 6. Memory Bandwidth Bottleneck Detection

### 6.1 The Decode Phase Problem

Autoregressive decode is fundamentally memory-bandwidth-bound at low batch
sizes. Each token generation step requires reading the entire model from
HBM to compute a single output per active request. The arithmetic intensity
is:

```
AI_decode = 2 * B / (2 * element_size) = B / element_size     [FLOP/byte]
```

For FP16 (element_size = 2 bytes): AI = B/2. At batch size 1, AI = 0.5
FLOP/byte — far below the ridge point of any modern GPU (~100-200 for
A100/H100).

This means **the GPU compute units are mostly idle during decode**, waiting
for memory reads. The observable signature:

```
  Decode phase:     sm_utilization ~ 20-40%
                    mem_utilization ~ 60-90%
                    gpu_utilization ~ 90%+ (kernels run, just slowly)
```

### 6.2 The Prefill Phase Contrast

Prefill processes `S` tokens in parallel, so AI = B*S/element_size. Even at
batch size 1 with sequence length 2048: AI = 1024 for FP16 — deeply
compute-bound:

```
  Prefill phase:    sm_utilization ~ 70-95%
                    mem_utilization ~ 30-50%
                    gpu_utilization ~ 95%+
```

### 6.3 Discriminating Prefill vs Decode from Telemetry

The ratio `sm_utilization / mem_utilization` is a strong discriminator:

```
  Compute-Memory Ratio:
    CMR(t) = sm_utilization(t) / mem_utilization(t)

    CMR > 1.5  =>  Prefill-dominant workload at time t
    CMR < 0.7  =>  Decode-dominant workload at time t
    0.7 <= CMR <= 1.5  =>  Mixed or balanced
```

**Pseudocode**:

```python
@dataclass
class BandwidthBottleneckResult:
    """Memory bandwidth bottleneck analysis."""
    avg_cmr: float  # compute-memory ratio
    prefill_fraction: float  # fraction of time in prefill-dominant regime
    decode_fraction: float  # fraction of time in decode-dominant regime
    mixed_fraction: float
    bottleneck_regime: str  # "compute_bound", "memory_bound", "mixed"
    peak_mem_utilization: float
    peak_sm_utilization: float


def analyze_bandwidth_bottleneck(
    sm_util_ts: NDArray[np.int64],
    sm_util_vals: NDArray[np.float64],
    mem_util_ts: NDArray[np.int64],
    mem_util_vals: NDArray[np.float64],
    prefill_cmr_threshold: float = 1.5,
    decode_cmr_threshold: float = 0.7,
) -> BandwidthBottleneckResult:
    """Classify workload regime from SM and memory utilization time series.

    Aligns both time series to the coarser grid (they typically share
    timestamps from the same DCGM scrape) and computes the compute-memory
    ratio at each sample.
    """
    # Align to common timestamps (should be identical from same DCGM scrape)
    common_ts = np.intersect1d(sm_util_ts, mem_util_ts)
    if len(common_ts) == 0:
        # Fallback: use all sm_util timestamps, interpolate mem_util
        common_ts = sm_util_ts

    sm_aligned = np.interp(common_ts, sm_util_ts, sm_util_vals)
    mem_aligned = np.interp(common_ts, mem_util_ts, mem_util_vals)

    # Avoid division by zero
    safe_mem = np.maximum(mem_aligned, 1.0)  # floor at 1%
    cmr = sm_aligned / safe_mem

    # Classify each sample
    prefill_mask = cmr > prefill_cmr_threshold
    decode_mask = cmr < decode_cmr_threshold
    mixed_mask = ~prefill_mask & ~decode_mask

    # Duration-weighted fractions
    if len(common_ts) > 1:
        durations = np.diff(common_ts).astype(np.float64)
        total_dur = durations.sum()
        if total_dur > 0:
            prefill_frac = float(durations[prefill_mask[:-1]].sum() / total_dur)
            decode_frac = float(durations[decode_mask[:-1]].sum() / total_dur)
            mixed_frac = float(durations[mixed_mask[:-1]].sum() / total_dur)
        else:
            prefill_frac = decode_frac = mixed_frac = 0.0
    else:
        n = len(cmr)
        prefill_frac = prefill_mask.sum() / n if n > 0 else 0.0
        decode_frac = decode_mask.sum() / n if n > 0 else 0.0
        mixed_frac = mixed_mask.sum() / n if n > 0 else 0.0

    # Overall regime
    if prefill_frac > 0.6:
        regime = "compute_bound"
    elif decode_frac > 0.6:
        regime = "memory_bound"
    else:
        regime = "mixed"

    return BandwidthBottleneckResult(
        avg_cmr=float(np.mean(cmr)),
        prefill_fraction=prefill_frac,
        decode_fraction=decode_frac,
        mixed_fraction=mixed_frac,
        bottleneck_regime=regime,
        peak_mem_utilization=float(np.max(mem_aligned)),
        peak_sm_utilization=float(np.max(sm_aligned)),
    )
```

### 6.4 Cross-Referencing with AIPerf Sweep Metrics

The bandwidth bottleneck analysis becomes more powerful when cross-referenced
with AIPerf's existing sweep metrics:

- **`effective_prefill_throughput`** correlates with `sm_utilization`
  (compute-bound — higher SM occupancy means faster prefill)
- **`effective_throughput`** (generation phase) correlates with
  `mem_utilization` (bandwidth-bound — decode is the bottleneck)
- **`effective_prefill_concurrency`** vs `effective_generation_concurrency`
  reveals the phase mix at any point in time

When `effective_prefill_concurrency / effective_generation_concurrency > 0.5`,
the server is spending significant time in prefill, and `sm_utilization`
should be the primary efficiency indicator. When the ratio is low, the server
is decode-dominated, and `mem_utilization` matters more.

---

## 7. Batch Efficiency Analysis

### 7.1 The Concurrency-Utilization Relationship

As client concurrency increases, the inference server batches more requests
together, increasing GPU utilization. The relationship follows a concave
curve that eventually flattens:

```
  sm_utilization (%)
    ^
    |                         .............
    |                     ...·
    |                  ..·
    |               ..·
    |            ..·
    |          .·
    |        .·
    |      .·
    |    .·
    |  .·
    | ·
    +───────────────────────────> effective_concurrency
```

The slope `d(sm_util) / d(concurrency)` represents how efficiently additional
load translates to GPU utilization. A steep initial slope followed by a
plateau means:
- Small batches dramatically improve GPU efficiency
- Large batches yield diminishing returns on utilization

### 7.2 Batch Efficiency Metric

Define **batch efficiency** as the throughput achieved per unit of GPU
utilization:

```
BE(t) = effective_throughput(t) / sm_utilization(t)     [tokens/s/%]
```

This measures how effectively the GPU translates utilization into useful
work. It typically follows an inverted-U shape:

```
  Batch Efficiency (tok/s/%)
    ^
    |      ...
    |    .·   ·..
    |  .·       ·...
    | ·             ·.....
    |·                    ·..........
    +───────────────────────────────> effective_concurrency
       ^
       optimal batch
       efficiency point
```

The peak of this curve is the **optimal operating point** — where each
additional percent of GPU utilization produces the most tokens. Beyond this
point, contention effects (memory pressure, scheduling overhead, KV cache
eviction) reduce per-utilization efficiency.

### 7.3 Per-User Throughput Degradation

AIPerf already computes `effective_throughput_per_user` via sweep-line
division. Correlating this with GPU utilization reveals the **user
experience cost** of high utilization:

```
  per_user_throughput (tok/s/user)
    ^
    | ·
    |  ·.
    |    ·..
    |       ·...
    |           ·.....
    |                 ·...........
    |                             ·..............
    +───────────────────────────────────────────> sm_utilization (%)
```

This curve is always monotonically decreasing. The interesting question is
the **rate of degradation**: a shallow curve means the server handles
batching well (e.g., continuous batching with efficient scheduling). A steep
curve means heavy contention.

**Degradation rate**:

```
DR = -d(per_user_throughput) / d(sm_utilization)
```

High DR at low utilization suggests scheduling inefficiency. High DR at high
utilization is expected (physical limits).

### 7.4 Pseudocode

```python
@dataclass
class BatchEfficiencyResult:
    """Batch efficiency analysis results."""
    optimal_concurrency: float
    optimal_sm_utilization: float
    peak_batch_efficiency: float  # tokens/s/%
    utilization_elasticity: float  # % throughput change / % utilization change
    per_user_degradation_rate: float  # tokens/s/user per % utilization


def analyze_batch_efficiency(
    throughput_ts: NDArray[np.float64],
    throughput_vals: NDArray[np.float64],
    concurrency_ts: NDArray[np.float64],
    concurrency_vals: NDArray[np.float64],
    sm_util_ts: NDArray[np.int64],
    sm_util_vals: NDArray[np.float64],
    per_user_tput_ts: NDArray[np.float64],
    per_user_tput_vals: NDArray[np.float64],
) -> BatchEfficiencyResult:
    """Analyze batch efficiency across the throughput-utilization space."""
    # Align to DCGM cadence
    tput_at_dcgm = step_lookup(throughput_ts, throughput_vals, sm_util_ts)
    conc_at_dcgm = step_lookup(concurrency_ts, concurrency_vals, sm_util_ts)
    puu_at_dcgm = step_lookup(per_user_tput_ts, per_user_tput_vals, sm_util_ts)

    # Convert throughput from tokens/ns to tokens/s
    tput_tok_s = tput_at_dcgm * NANOS_PER_SECOND

    # Batch efficiency: throughput / utilization
    safe_util = np.maximum(sm_util_vals, 1.0)
    batch_eff = tput_tok_s / safe_util

    # Find optimal point
    best_idx = int(np.argmax(batch_eff))

    # Utilization elasticity (% throughput change per % utilization change)
    # Using log-log regression: log(throughput) = alpha + beta * log(utilization)
    valid = (sm_util_vals > 5.0) & (tput_tok_s > 0)
    if valid.sum() >= 5:
        log_util = np.log(sm_util_vals[valid])
        log_tput = np.log(tput_tok_s[valid])
        # Simple least-squares slope
        elasticity = float(
            np.polyfit(log_util, log_tput, 1)[0]
        )
    else:
        elasticity = 0.0

    # Per-user degradation rate (linear regression of per_user_tput vs sm_util)
    puu_tok_s = puu_at_dcgm * NANOS_PER_SECOND
    valid_puu = (sm_util_vals > 5.0) & (puu_tok_s > 0)
    if valid_puu.sum() >= 5:
        slope = float(np.polyfit(sm_util_vals[valid_puu], puu_tok_s[valid_puu], 1)[0])
    else:
        slope = 0.0

    return BatchEfficiencyResult(
        optimal_concurrency=float(conc_at_dcgm[best_idx]),
        optimal_sm_utilization=float(sm_util_vals[best_idx]),
        peak_batch_efficiency=float(batch_eff[best_idx]),
        utilization_elasticity=elasticity,
        per_user_degradation_rate=slope,
    )
```

### 7.5 Interpretation Guide

| Elasticity | Meaning |
|-----------|---------|
| beta ~ 1.0 | Linear scaling — throughput grows proportionally with utilization. Ideal. |
| beta ~ 0.5 | Square-root scaling — typical for memory-bound decode. |
| beta > 1.0 | Superlinear — batching provides superlinear speedup (common early in the curve). |
| beta < 0.3 | Severely diminishing returns — likely saturated or thrashing. |

---

## 8. Thermal Throttling Detection

### 8.1 Throttling Mechanisms

NVIDIA GPUs implement multiple throttling mechanisms:

1. **Thermal throttling**: When `gpu_temperature` approaches TjMax (typically
   83-90C for data center GPUs), the GPU reduces clock frequency to limit heat
   generation.
2. **Power throttling**: When instantaneous power exceeds the power limit
   (TDP), the GPU reduces clocks. Tracked by the `power_violation` counter
   (cumulative microseconds of throttling).
3. **Reliability throttling**: At extreme temperatures (approaching thermal
   shutdown at ~100C), more aggressive throttling kicks in.

### 8.2 Observable Signature

Thermal throttling creates a characteristic pattern in the telemetry:

```
  Temperature (C)       Power (W)          Throughput (tok/s)
    ^                     ^                    ^
    |     ...TjMax...     |  ....TDP....       |  ........
    |    /               |  /                  |          \
    |   /                | /                   |           \
    |  /                 |/                    |            \
    | /                  |                     |             \
    |/                   |                     |              \.
    +──> time            +──> time             +──────────> time
```

The key correlation: **throughput drop lagging a temperature/power ceiling
by a few seconds** (the GPU controller's feedback loop).

### 8.3 Detection Algorithm

```python
@dataclass
class ThrottlingEvent:
    """A detected throttling event."""
    start_ns: int
    end_ns: int
    trigger: str  # "thermal", "power", "both"
    peak_temperature_c: float
    throughput_drop_pct: float
    power_violation_us: float


def detect_thermal_throttling(
    temp_ts: NDArray[np.int64],
    temp_vals: NDArray[np.float64],
    power_ts: NDArray[np.int64],
    power_vals: NDArray[np.float64],
    power_violation_ts: NDArray[np.int64],
    power_violation_vals: NDArray[np.float64],
    throughput_ts: NDArray[np.float64],
    throughput_vals: NDArray[np.float64],
    thermal_threshold_c: float = 83.0,
    power_violation_threshold_us: float = 1000.0,
) -> list[ThrottlingEvent]:
    """Detect periods where GPU throttling correlates with throughput drops.

    Strategy:
    1. Find intervals where temperature exceeds threshold or power_violation
       counter increments significantly.
    2. Check if throughput drops during or shortly after those intervals.
    3. Report correlated events.
    """
    events: list[ThrottlingEvent] = []

    # Detect thermal events (temperature above threshold)
    thermal_mask = temp_vals >= thermal_threshold_c
    thermal_regions = _find_contiguous_regions(thermal_mask)

    for start_idx, end_idx in thermal_regions:
        event_start = int(temp_ts[start_idx])
        event_end = int(temp_ts[min(end_idx, len(temp_ts) - 1)])

        # Check power violations in the same window
        pv_mask = (power_violation_ts >= event_start) & (power_violation_ts <= event_end)
        pv_delta = 0.0
        if pv_mask.sum() >= 2:
            pv_delta = float(power_violation_vals[pv_mask][-1] - power_violation_vals[pv_mask][0])

        # Check throughput impact: compare during-event vs pre-event
        pre_window = (throughput_ts >= event_start - 5e9) & (throughput_ts < event_start)
        during_window = (throughput_ts >= event_start) & (throughput_ts <= event_end)

        pre_tput = throughput_vals[pre_window]
        during_tput = throughput_vals[during_window]

        if len(pre_tput) > 0 and len(during_tput) > 0:
            pre_avg = float(np.mean(pre_tput))
            during_avg = float(np.mean(during_tput))
            drop_pct = (pre_avg - during_avg) / pre_avg * 100 if pre_avg > 0 else 0.0
        else:
            drop_pct = 0.0

        trigger = "both" if pv_delta > power_violation_threshold_us else "thermal"

        events.append(ThrottlingEvent(
            start_ns=event_start,
            end_ns=event_end,
            trigger=trigger,
            peak_temperature_c=float(np.max(temp_vals[start_idx:end_idx + 1])),
            throughput_drop_pct=drop_pct,
            power_violation_us=pv_delta,
        ))

    return events


def _find_contiguous_regions(
    mask: NDArray[np.bool_],
) -> list[tuple[int, int]]:
    """Find start/end indices of contiguous True regions in a boolean mask."""
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), (ends - 1).tolist(), strict=True))
```

### 8.4 Power Violation Counter Analysis

The `power_violation` field in DCGM is a **cumulative counter** measured in
microseconds. It tracks the total time the GPU has been power-throttled since
boot. AIPerf already handles this as a counter metric (in
`GPU_TELEMETRY_COUNTER_METRICS`), computing the delta over the profiling
window.

For correlation analysis, we need the **rate of power violation** — the
derivative of the counter:

```
power_throttle_rate(t) = d(power_violation) / dt     [us/s = dimensionless ratio]
```

A `power_throttle_rate` of 0.5 means the GPU was power-throttled 50% of
the time. This should correlate strongly with throughput degradation.

### 8.5 Thermal Headroom Metric

Rather than waiting for throttling to occur, a proactive metric:

```
thermal_headroom(t) = TjMax - gpu_temperature(t)     [C]
```

When `thermal_headroom < 5C`, the system is at risk. Correlating
`thermal_headroom` trends with throughput trends can predict future
performance degradation before it happens.

---

## 9. Energy Efficiency Metrics

### 9.1 Joules Per Token

The fundamental energy efficiency metric:

```
J_per_token = energy_delta_J / total_output_tokens

Where:
    energy_delta_J = (energy_end_MJ - energy_start_MJ) * 1e6
    total_output_tokens = sum of output_tokens across all requests
```

This metric captures the **full cost** of generating tokens, including:
- Compute for both prefill and decode
- Memory access energy
- Idle power during scheduling gaps
- Cooling overhead (indirectly, through higher base power)

### 9.2 Joules Per Request

For workloads where request count matters more than token count:

```
J_per_request = energy_delta_J / total_requests
```

This is useful for comparing configurations with different output lengths
(e.g., classification vs generation).

### 9.3 Energy Breakdown by Phase

If we can distinguish prefill and decode time (using `effective_prefill_concurrency`
and `effective_generation_concurrency` from AIPerf's sweep data), we can
estimate the energy split:

```
  Total Energy = E_prefill + E_decode + E_idle

  E_prefill ~ avg_power * total_prefill_time
  E_decode  ~ avg_power * total_decode_time
  E_idle    ~ idle_power * total_idle_time
```

Where `total_prefill_time` and `total_decode_time` can be estimated from
the sweep curves:

```python
def estimate_phase_energy_split(
    prefill_conc_ts: NDArray[np.float64],
    prefill_conc: NDArray[np.float64],
    gen_conc_ts: NDArray[np.float64],
    gen_conc: NDArray[np.float64],
    power_ts: NDArray[np.int64],
    power_vals: NDArray[np.float64],
    window_start_ns: float,
    window_end_ns: float,
) -> tuple[float, float, float]:
    """Estimate energy split between prefill, decode, and idle phases.

    Returns (E_prefill_J, E_decode_J, E_idle_J).

    Note: This is an approximation because prefill and decode overlap in
    continuous batching. The "active" time for each phase is estimated from
    the fraction of time where the respective concurrency is > 0.
    """
    total_duration_s = (window_end_ns - window_start_ns) / NANOS_PER_SECOND

    # Duration where prefill is active (concurrency > 0)
    prefill_active_frac = _active_fraction(
        prefill_conc_ts, prefill_conc, window_start_ns, window_end_ns
    )
    decode_active_frac = _active_fraction(
        gen_conc_ts, gen_conc, window_start_ns, window_end_ns
    )

    # Average power over the window
    power_mask = (power_ts >= window_start_ns) & (power_ts < window_end_ns)
    avg_power_w = float(np.mean(power_vals[power_mask])) if power_mask.any() else 0.0

    # Rough split (prefill and decode can overlap, so fractions may sum > 1)
    # Normalize to sum to 1
    total_active = prefill_active_frac + decode_active_frac
    idle_frac = max(0.0, 1.0 - max(prefill_active_frac, decode_active_frac))

    if total_active > 0:
        prefill_weight = prefill_active_frac / total_active
        decode_weight = decode_active_frac / total_active
    else:
        prefill_weight = decode_weight = 0.5

    total_energy_j = avg_power_w * total_duration_s
    active_energy_j = total_energy_j * (1.0 - idle_frac)

    return (
        active_energy_j * prefill_weight,
        active_energy_j * decode_weight,
        total_energy_j * idle_frac,
    )
```

### 9.4 Comparative Energy Analysis

For benchmarking across configurations (different models, quantization
levels, batch sizes, tensor parallelism), the key comparison metrics are:

| Metric | Formula | Use Case |
|--------|---------|----------|
| J/token | energy / output_tokens | Compare generation efficiency |
| J/request | energy / requests | Compare end-to-end efficiency |
| tokens/kWh | output_tokens / (energy_J / 3.6e6) | Data center capacity planning |
| W/tok/s | avg_power / throughput | Marginal power cost of throughput |

### 9.5 TCO Implications

At data center scale, energy cost is a significant fraction of TCO:

```
Annual energy cost = avg_power_kW * hours_per_year * electricity_rate_per_kWh
                   = (avg_power / 1000) * 8760 * rate

Annual token capacity = throughput_tok_s * 3600 * 8760 * utilization_fraction

Cost per million tokens = annual_energy_cost / (annual_token_capacity / 1e6)
```

This connects benchmark-level metrics to business-level decisions.

---

## 10. Multi-GPU Scaling

### 10.1 Tensor Parallelism Utilization Balance

In tensor-parallel (TP) deployments, the model is split across N GPUs. Each
GPU should have approximately equal utilization. Imbalance indicates:
- Uneven layer distribution (some GPUs have more work)
- Communication bottlenecks (AllReduce stalling some GPUs)
- Memory pressure differences (different KV cache sizes)

**Imbalance metric** (coefficient of variation):

```
util_cv = std(sm_util_per_gpu) / mean(sm_util_per_gpu)
```

A `util_cv < 0.05` (5%) indicates good balance. A `util_cv > 0.15` indicates
significant imbalance worth investigating.

### 10.2 Cross-GPU Utilization Analysis

AIPerf's `GPUTelemetryAccumulator` stores telemetry hierarchically:
`dcgm_url -> gpu_uuid -> metric_time_series`. For multi-GPU analysis,
we need to aggregate across GPUs within the same endpoint:

```python
@dataclass
class MultiGPUScalingResult:
    """Multi-GPU scaling efficiency analysis."""
    num_gpus: int
    utilization_cv: float  # coefficient of variation across GPUs
    utilization_balance_score: float  # 1.0 = perfect, 0.0 = one GPU does all work
    throughput_scaling_efficiency: float  # actual throughput / (single_gpu_throughput * N)
    max_utilization_gap: float  # max - min utilization across GPUs
    gpu_utilizations: dict[str, float]  # per-GPU average utilization


def analyze_multi_gpu_scaling(
    hierarchy: TelemetryHierarchy,
    throughput_ts: NDArray[np.float64],
    throughput_vals: NDArray[np.float64],
    window_start_ns: int,
    window_end_ns: int,
    single_gpu_throughput: float | None = None,
) -> MultiGPUScalingResult | None:
    """Analyze utilization balance and scaling efficiency across GPUs.

    Args:
        hierarchy: TelemetryHierarchy from GPUTelemetryAccumulator.
        throughput_ts: Sweep-line throughput timestamps.
        throughput_vals: Throughput values (tokens/ns).
        window_start_ns: Analysis window start.
        window_end_ns: Analysis window end.
        single_gpu_throughput: Optional baseline for scaling efficiency calculation.

    Returns:
        MultiGPUScalingResult or None if fewer than 2 GPUs.
    """
    gpu_avg_utils: dict[str, float] = {}

    for dcgm_url, gpus in hierarchy.dcgm_endpoints.items():
        for gpu_uuid, gpu_data in gpus.items():
            sm_arr = gpu_data.time_series.get_metric_array("sm_utilization")
            if sm_arr is None or len(sm_arr) == 0:
                continue

            # Filter to window
            ts = gpu_data.time_series.timestamps
            mask = (ts >= window_start_ns) & (ts < window_end_ns)
            windowed = sm_arr[mask]

            if len(windowed) > 0:
                gpu_id = f"{dcgm_url}:gpu{gpu_data.metadata.gpu_index}"
                gpu_avg_utils[gpu_id] = float(np.mean(windowed))

    num_gpus = len(gpu_avg_utils)
    if num_gpus < 2:
        return None

    utils = np.array(list(gpu_avg_utils.values()))
    mean_util = float(np.mean(utils))
    std_util = float(np.std(utils))
    cv = std_util / mean_util if mean_util > 0 else 0.0

    # Balance score: 1 - normalized CV (clamped to [0, 1])
    balance_score = max(0.0, 1.0 - cv)

    # Scaling efficiency (if baseline provided)
    if single_gpu_throughput is not None and single_gpu_throughput > 0:
        tput_stats = compute_time_weighted_stats(
            throughput_ts, throughput_vals,
            float(window_start_ns), float(window_end_ns),
        )
        actual_throughput = tput_stats.avg * NANOS_PER_SECOND
        ideal_throughput = single_gpu_throughput * num_gpus
        scaling_eff = actual_throughput / ideal_throughput
    else:
        scaling_eff = 0.0  # unknown without baseline

    return MultiGPUScalingResult(
        num_gpus=num_gpus,
        utilization_cv=cv,
        utilization_balance_score=balance_score,
        throughput_scaling_efficiency=scaling_eff,
        max_utilization_gap=float(np.max(utils) - np.min(utils)),
        gpu_utilizations=gpu_avg_utils,
    )
```

### 10.3 Communication Overhead Detection

In TP deployments, inter-GPU communication (AllReduce, AllGather) creates
"gaps" where all GPUs wait for the slowest. These manifest as synchronized
utilization dips:

```
  GPU 0 sm_util:  ████████  ██████████  ████████  ██████████
  GPU 1 sm_util:  ████████  ██████████  ████████  ██████████
  GPU 2 sm_util:  ████████  ██████████  ████████  ██████████
  GPU 3 sm_util:  ████████  ██████████  ████████  ██████████
                          ^^          ^^          ^^
                       AllReduce   AllReduce   AllReduce
                        gaps        gaps        gaps
```

**Detection**: Compute the cross-correlation of utilization time series
across GPUs. High cross-correlation (>0.8) with periodic dips suggests
communication overhead. The duty cycle of the dips estimates the
communication fraction:

```
comm_fraction = 1.0 - (avg_util_with_dips / avg_util_peaks_only)
```

### 10.4 Pipeline Parallelism Signatures

Pipeline-parallel (PP) deployments create a different utilization pattern:
**staggered** utilization across stages:

```
  GPU 0 (stage 0):  ████    ████    ████    ████
  GPU 1 (stage 1):    ████    ████    ████    ████
  GPU 2 (stage 2):      ████    ████    ████    ████
  GPU 3 (stage 3):        ████    ████    ████    ████
                    ─────────────────────────────────> time
```

The cross-correlation between adjacent stages should show a lag equal to the
per-stage latency. High lag-correlation with low zero-lag correlation is the
signature of PP (vs TP which shows high zero-lag correlation).

---

## 11. Temporal Phase Detection

### 11.1 Workload Phases from GPU Utilization

A typical LLM benchmark run has three to five temporal phases visible in
GPU utilization:

```
  sm_utilization (%)
    ^
    |                    ┌────────────────────────────┐
    |                    │                            │
    |                    │      Steady State          │
    |                 ┌──┘                            └──┐
    |              ┌──┘                                  └──┐
    |           ┌──┘                                        └──┐
    |        ┌──┘    Ramp                           Ramp       └──┐
    |     ┌──┘       Up                             Down          └──┐
    |  ┌──┘                                                          └──┐
    | ─┘  Warm                                                    Cool   └─
    |     Up                                                      Down
    +──────────────────────────────────────────────────────────────────> time
```

Phases:
1. **Warm-up**: JIT compilation, CUDA context initialization, KV cache
   allocation. Utilization is erratic and low.
2. **Ramp-up**: Client load increases, utilization climbs steadily.
3. **Steady state**: Stable utilization (what AIPerf's CUSUM+MSER-5 detects).
4. **Ramp-down**: Client stops sending requests, in-flight requests drain.
5. **Cool-down**: Final cleanup, utilization drops to near zero.

### 11.2 Phase Detection Using GPU Utilization Change Points

AIPerf's existing steady-state detection (CUSUM + MSER-5) operates on
**client-side metrics** (concurrency, latency, throughput, TTFT). Adding
GPU utilization as a **fifth signal** to the multi-signal detection in
`detect_steady_state_window()` provides server-side validation:

```python
def detect_steady_state_with_gpu_utilization(
    concurrency_ts: NDArray[np.float64],
    concurrency: NDArray[np.float64],
    latency_values: NDArray[np.float64],
    latency_start_ns: NDArray[np.float64],
    ttft_values: NDArray[np.float64],
    ttft_start_ns: NDArray[np.float64],
    throughput_ts: NDArray[np.float64],
    throughput: NDArray[np.float64],
    # New: GPU utilization signal
    sm_util_ts: NDArray[np.int64],
    sm_util_vals: NDArray[np.float64],
    min_window_pct: float = 10.0,
) -> DetectionResult:
    """Multi-signal steady-state detection with GPU utilization as fifth signal.

    The GPU utilization signal serves as server-side validation:
    - If client-side signals detect steady state at [t1, t2], but GPU
      utilization is still climbing during [t1, t1+delta], the ramp-up
      hasn't truly ended (the server is still warming up internally).
    - If GPU utilization drops before t2, the server started degrading
      before the client noticed.
    """
    # Existing 4-signal detection
    client_start, client_end = detect_steady_state_window(
        concurrency_ts, concurrency,
        latency_values, latency_start_ns,
        ttft_values, ttft_start_ns,
        throughput_ts, throughput,
        min_window_pct=min_window_pct,
    )

    # GPU utilization boundaries via MSER-5
    gpu_start, gpu_end = mser5_boundary_ns(
        sm_util_vals, sm_util_ts.astype(np.float64)
    )

    # Intersection: steady state must satisfy BOTH client and GPU constraints
    final_start = max(client_start, gpu_start)
    final_end = min(client_end, gpu_end)

    # Ensure minimum window
    total_duration = float(concurrency_ts[-1] - concurrency_ts[0])
    min_duration = total_duration * (min_window_pct / 100.0)
    if final_end - final_start < min_duration:
        # Fall back to client-only detection
        return DetectionResult(client_start, client_end, "cusum_mser5")

    return DetectionResult(final_start, final_end, "cusum_mser5_gpu")
```

### 11.3 GPU Utilization Stability Metric

Within the detected steady-state window, GPU utilization should be stable.
Quantify stability using the coefficient of variation:

```
GPU_stability = 1.0 - CV(sm_utilization_in_window)

Where CV = std / mean
```

A `GPU_stability > 0.95` (CV < 5%) indicates a truly steady state on the
server side. Lower values suggest the server is experiencing periodic
fluctuations (e.g., KV cache evictions, GC pauses, scheduling jitter) that
the client may not directly observe.

### 11.4 Utilization Change-Point Detection

Beyond the binary steady-state/non-steady-state classification, we can use
utilization change points to detect **mid-run events**:

- **KV cache eviction**: Sudden utilization drop + spike as cache is
  reorganized and recomputation begins.
- **Model offloading**: Utilization drops to near zero as weights are swapped.
- **Server auto-scaling**: New GPU instances coming online (multi-instance).

These events would be detected by running PELT (from the `proposal-advanced-analysis.md`) or simpler sliding-window variance analysis
on the `sm_utilization` time series:

```python
def detect_utilization_anomalies(
    sm_util_ts: NDArray[np.int64],
    sm_util_vals: NDArray[np.float64],
    window_size: int = 10,
    zscore_threshold: float = 3.0,
) -> list[tuple[int, float, str]]:
    """Detect anomalous utilization changes using sliding-window z-scores.

    Returns list of (timestamp_ns, z_score, direction) tuples.
    """
    if len(sm_util_vals) < window_size * 2:
        return []

    anomalies = []
    for i in range(window_size, len(sm_util_vals) - window_size):
        before = sm_util_vals[i - window_size:i]
        after = sm_util_vals[i:i + window_size]

        mean_before = np.mean(before)
        std_before = np.std(before)

        if std_before < 1.0:
            std_before = 1.0  # minimum std to avoid false positives on flat signals

        z = (np.mean(after) - mean_before) / std_before

        if abs(z) > zscore_threshold:
            direction = "increase" if z > 0 else "decrease"
            anomalies.append((int(sm_util_ts[i]), float(z), direction))

    return anomalies
```

---

## 12. Correlation Framework: Joining Two Time Domains

### 12.1 The Time Alignment Problem

The fundamental challenge in correlating GPU telemetry with client-side
metrics is that they live in different time domains:

| Property | Client-Side (Sweep) | GPU Telemetry (DCGM) |
|----------|--------------------|-----------------------|
| **Time representation** | Float64 nanoseconds | Int64 nanoseconds |
| **Sampling** | Event-driven (per request boundary) | Periodic (fixed interval) |
| **Data structure** | Step functions (SweepCurves) | Point samples (GpuMetricTimeSeries) |
| **Resolution** | Sub-microsecond (nanosecond events) | 100ms-1s (DCGM scrape interval) |
| **Storage** | ColumnStore (session-indexed) | TelemetryHierarchy (GPU-indexed) |
| **Time source** | Client clock (may differ from server) | Server/GPU clock |

### 12.2 Alignment Strategy

The alignment must:
1. Resample the high-resolution sweep curves to the DCGM cadence
2. Handle clock skew between client and server (if they run on different
   machines)
3. Preserve the step-function semantics of sweep data

**Step 1: Resample sweep curves to DCGM timestamps**

The sweep-line outputs are step functions: value `v[i]` holds from `ts[i]`
to `ts[i+1]`. We can evaluate them at any timestamp using binary search
(this is exactly `_step_lookup` from `sweep.py`):

```python
def align_sweep_to_telemetry(
    sweep_ts: NDArray[np.float64],
    sweep_vals: NDArray[np.float64],
    telemetry_ts: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Evaluate a sweep step function at DCGM sample timestamps.

    Uses _step_lookup: O(n log n) via np.searchsorted.
    """
    return _step_lookup(sweep_ts, sweep_vals, telemetry_ts.astype(np.float64))
```

**Step 2: Handle clock skew**

When the client and GPU are on different machines, their clocks may differ
by milliseconds to seconds. For correlation analysis, absolute time alignment
matters less than **relative** trends. Two approaches:

1. **Cross-correlation offset**: Compute the lag that maximizes
   cross-correlation between throughput and sm_utilization. Use this lag
   as the clock offset. Works when both signals have clear ramp-up patterns.

2. **Normalize to relative time**: Subtract the first timestamp from both
   time series, so both start at t=0. This eliminates clock offset but
   requires both streams to start at approximately the same wall-clock time
   (which is true in AIPerf — both start at profiling start).

**Step 3: Build aligned dataset**

```python
@dataclass
class AlignedCorrelationData:
    """Temporally aligned client-side and GPU telemetry data."""
    timestamps_ns: NDArray[np.int64]  # common timeline (DCGM cadence)

    # Client-side (resampled to DCGM cadence)
    effective_throughput: NDArray[np.float64]  # tokens/s
    effective_concurrency: NDArray[np.float64]  # requests
    effective_prefill_throughput: NDArray[np.float64]  # tokens/s
    tokens_in_flight: NDArray[np.float64]  # tokens
    throughput_per_user: NDArray[np.float64]  # tokens/s/user

    # GPU telemetry (native DCGM cadence)
    gpu_utilization: NDArray[np.float64]  # %
    sm_utilization: NDArray[np.float64]  # %
    mem_utilization: NDArray[np.float64]  # %
    gpu_power_usage: NDArray[np.float64]  # W
    gpu_temperature: NDArray[np.float64]  # C
    gpu_memory_used: NDArray[np.float64]  # GB


def build_aligned_data(
    sweeps: SweepCurves,
    telemetry_data: GpuMetricTimeSeries,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
) -> AlignedCorrelationData:
    """Build a temporally aligned dataset from sweep curves and GPU telemetry.

    Resamples sweep step functions to DCGM timestamps using step_lookup.
    Optionally filters to a time window (e.g., steady-state window).
    """
    dcgm_ts = telemetry_data.timestamps

    # Optional time windowing
    if window_start_ns is not None and window_end_ns is not None:
        mask = (dcgm_ts >= window_start_ns) & (dcgm_ts < window_end_ns)
        dcgm_ts = dcgm_ts[mask]
    else:
        mask = np.ones(len(dcgm_ts), dtype=bool)

    dcgm_float = dcgm_ts.astype(np.float64)

    # Resample sweep curves to DCGM cadence
    throughput = _step_lookup(sweeps.throughput_ts, sweeps.throughput, dcgm_float)
    concurrency = _step_lookup(sweeps.concurrency_ts, sweeps.concurrency, dcgm_float)
    prefill_tput = _step_lookup(
        sweeps.prefill_throughput_ts, sweeps.prefill_throughput, dcgm_float
    )
    tif = _step_lookup(sweeps.tokens_in_flight_ts, sweeps.tokens_in_flight, dcgm_float)
    tpu = _step_lookup(
        sweeps.throughput_per_user_ts, sweeps.throughput_per_user, dcgm_float
    )

    # Scale throughput from tokens/ns to tokens/s
    throughput_tok_s = throughput * NANOS_PER_SECOND
    prefill_tput_tok_s = prefill_tput * NANOS_PER_SECOND
    tpu_tok_s = tpu * NANOS_PER_SECOND

    # GPU telemetry (already at DCGM cadence)
    def safe_metric(name: str) -> NDArray[np.float64]:
        arr = telemetry_data.get_metric_array(name)
        if arr is None:
            return np.full(len(dcgm_ts), np.nan)
        return arr[mask] if mask is not None else arr

    return AlignedCorrelationData(
        timestamps_ns=dcgm_ts,
        effective_throughput=throughput_tok_s,
        effective_concurrency=concurrency,
        effective_prefill_throughput=prefill_tput_tok_s,
        tokens_in_flight=tif,
        throughput_per_user=tpu_tok_s,
        gpu_utilization=safe_metric("gpu_utilization"),
        sm_utilization=safe_metric("sm_utilization"),
        mem_utilization=safe_metric("mem_utilization"),
        gpu_power_usage=safe_metric("gpu_power_usage"),
        gpu_temperature=safe_metric("gpu_temperature"),
        gpu_memory_used=safe_metric("gpu_memory_used"),
    )
```

### 12.3 Correlation Metrics

Once aligned, compute standard correlation coefficients:

**Pearson correlation** (linear relationship):
```
r = cov(X, Y) / (std(X) * std(Y))
```

**Spearman rank correlation** (monotonic relationship, robust to outliers):
```
rho = pearson(rank(X), rank(Y))
```

Both are O(n log n) for n samples.

**Recommended correlation pairs**:

| Pair | Expected r | Interpretation |
|------|-----------|----------------|
| `effective_throughput` vs `sm_utilization` | +0.7 to +0.95 | Higher SM occupancy drives throughput |
| `effective_throughput` vs `mem_utilization` | +0.5 to +0.9 | Bandwidth correlates with throughput (decode) |
| `effective_concurrency` vs `gpu_utilization` | +0.8 to +0.98 | More load = more GPU activity |
| `throughput_per_user` vs `sm_utilization` | -0.3 to -0.8 | Per-user throughput degrades with load |
| `sm_utilization` vs `gpu_power_usage` | +0.85 to +0.99 | Power tracks compute activity closely |
| `gpu_temperature` vs `gpu_power_usage` | +0.7 to +0.95 | Temperature lags power (thermal inertia) |
| `effective_throughput` vs `gpu_temperature` | +0.3 to +0.7 | Weak: temperature is a lagged effect |

**Pseudocode**:

```python
@dataclass
class CorrelationResult:
    """Pairwise correlation between two metric time series."""
    metric_a: str
    metric_b: str
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    n_samples: int
    interpretation: str


def compute_correlation_matrix(
    aligned: AlignedCorrelationData,
) -> list[CorrelationResult]:
    """Compute pairwise correlations between all client-GPU metric pairs."""
    from scipy import stats  # or pure numpy implementation

    client_metrics = {
        "effective_throughput": aligned.effective_throughput,
        "effective_concurrency": aligned.effective_concurrency,
        "throughput_per_user": aligned.throughput_per_user,
        "tokens_in_flight": aligned.tokens_in_flight,
    }
    gpu_metrics = {
        "sm_utilization": aligned.sm_utilization,
        "mem_utilization": aligned.mem_utilization,
        "gpu_utilization": aligned.gpu_utilization,
        "gpu_power_usage": aligned.gpu_power_usage,
        "gpu_temperature": aligned.gpu_temperature,
    }

    results = []
    for c_name, c_vals in client_metrics.items():
        for g_name, g_vals in gpu_metrics.items():
            # Filter NaN pairs
            valid = ~(np.isnan(c_vals) | np.isnan(g_vals))
            if valid.sum() < 5:
                continue

            cv = c_vals[valid]
            gv = g_vals[valid]

            pr, pp = stats.pearsonr(cv, gv)
            sr, sp = stats.spearmanr(cv, gv)

            results.append(CorrelationResult(
                metric_a=c_name,
                metric_b=g_name,
                pearson_r=float(pr),
                pearson_p_value=float(pp),
                spearman_rho=float(sr),
                spearman_p_value=float(sp),
                n_samples=int(valid.sum()),
                interpretation=_interpret_correlation(c_name, g_name, float(sr)),
            ))

    return results


def _interpret_correlation(metric_a: str, metric_b: str, rho: float) -> str:
    """Generate human-readable interpretation of a correlation coefficient."""
    strength = (
        "strong" if abs(rho) > 0.7
        else "moderate" if abs(rho) > 0.4
        else "weak"
    )
    direction = "positive" if rho > 0 else "negative"
    return f"{strength} {direction} correlation (rho={rho:.3f})"
```

### 12.4 Lag-Correlation Analysis

Some correlations have a temporal lag (e.g., temperature lags power by
several seconds due to thermal inertia). Cross-correlation analysis finds
the optimal lag:

```python
def cross_correlation_lag(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int = 20,
) -> tuple[int, float]:
    """Find the lag that maximizes cross-correlation between x and y.

    Returns (optimal_lag, correlation_at_optimal_lag).
    Positive lag means y lags x (y shifted right).
    """
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

    n = len(x_norm)
    best_lag = 0
    best_corr = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_slice = x_norm[:n - lag]
            y_slice = y_norm[lag:]
        else:
            x_slice = x_norm[-lag:]
            y_slice = y_norm[:n + lag]

        if len(x_slice) < 5:
            continue

        corr = float(np.dot(x_slice, y_slice) / len(x_slice))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr
```

The lag in DCGM samples can be converted to seconds:
`lag_seconds = lag * collection_interval`.

---

## 13. AIPerf Implementation Guidance

### 13.1 Architecture: CorrelationAnalyzer Plugin

Following the existing `AnalyzerProtocol` pattern (from
`SteadyStateAnalyzer`), the correlation analysis should be implemented as an
optional analyzer plugin:

```
plugins.yaml:
  analyzer:
    gpu_correlation:
      class: aiperf.analysis.gpu_correlation.GPUCorrelationAnalyzer
      description: GPU utilization & throughput correlation analysis
      metadata:
        requires: [gpu_telemetry]
```

The analyzer would:
1. Accept references to both `MetricsAccumulator` and `GPUTelemetryAccumulator`
   (via `SummaryContext`)
2. Build `AlignedCorrelationData` from their shared timeline
3. Compute all correlation analyses (saturation, bandwidth, power, thermal, etc.)
4. Return a `GPUCorrelationSummary` result object

### 13.2 Data Flow

```
  MetricsAccumulator            GPUTelemetryAccumulator
       │                               │
       │  ColumnStore                   │  TelemetryHierarchy
       │  SweepCurves                   │  GpuMetricTimeSeries
       │                               │
       └──────────┐     ┌──────────────┘
                  │     │
                  v     v
          GPUCorrelationAnalyzer
                  │
                  │  build_aligned_data()
                  │  compute_correlation_matrix()
                  │  detect_saturation_point()
                  │  analyze_bandwidth_bottleneck()
                  │  compute_power_performance()
                  │  detect_thermal_throttling()
                  │  analyze_batch_efficiency()
                  │  analyze_multi_gpu_scaling()
                  │
                  v
          GPUCorrelationSummary
                  │
                  ├──> ConsoleGPUCorrelationExporter
                  ├──> GPUCorrelationJsonExporter
                  └──> GPUCorrelationCsvExporter
```

### 13.3 Result Model

```python
@dataclass
class GPUCorrelationSummary:
    """Complete GPU-throughput correlation analysis results."""

    # Core correlation matrix
    correlations: list[CorrelationResult]

    # Saturation analysis
    saturation_timestamp_ns: int | None  # None if no saturation detected
    saturation_sm_utilization: float | None
    saturation_throughput: float | None

    # Roofline classification
    dominant_regime: str  # "compute_bound", "memory_bound", "mixed"
    prefill_fraction: float
    decode_fraction: float

    # Power-performance
    power_performance: PowerPerformanceResult | None

    # Thermal throttling
    throttling_events: list[ThrottlingEvent]

    # Batch efficiency
    batch_efficiency: BatchEfficiencyResult | None

    # Multi-GPU scaling
    multi_gpu: MultiGPUScalingResult | None

    # Energy efficiency
    joules_per_token: float | None
    joules_per_request: float | None

    # Metadata
    num_aligned_samples: int
    window_start_ns: int
    window_end_ns: int
    num_gpus: int

    def to_json(self) -> dict[str, Any]:
        """Serialize to structured JSON for export."""
        ...

    def to_csv(self) -> list[dict[str, Any]]:
        """Serialize to flat CSV rows for tabular export."""
        ...
```

### 13.4 CLI Configuration

Following the existing `SteadyStateConfig` pattern:

```python
class GPUCorrelationConfig(BaseConfig):
    """Configuration for GPU correlation analysis."""

    enabled: bool = Field(
        default=False,
        description="Enable GPU utilization & throughput correlation analysis",
    )
    thermal_threshold_c: float = Field(
        default=83.0,
        description="Temperature threshold (C) for thermal throttling detection",
    )
    saturation_mte_fraction: float = Field(
        default=0.10,
        description="Marginal throughput efficiency fraction for saturation detection",
    )
    sm_threshold: float = Field(
        default=70.0,
        description="SM utilization threshold (%) for compute-bound classification",
    )
    mem_threshold: float = Field(
        default=60.0,
        description="Memory utilization threshold (%) for memory-bound classification",
    )
```

**CLI flags**:
- `--gpu-correlation` — enable correlation analysis
- `--gpu-correlation-thermal-threshold 83` — override thermal threshold
- `--gpu-correlation-sm-threshold 70` — override SM threshold

**Environment variables**:
- `AIPERF_GPU_CORRELATION=1`
- `AIPERF_GPU_CORRELATION_THERMAL_THRESHOLD=83`

### 13.5 Export Integration

Three export formats, following existing patterns:

**Console** (summary table):
```
GPU Correlation Analysis
========================
Dominant Regime:          memory_bound (72% decode, 18% prefill, 10% mixed)
Saturation Point:         3847 tok/s at 78% SM utilization
Power Performance:        12.4 tokens/watt (avg 287W)
Energy Efficiency:        0.081 J/token
Thermal Throttling:       None detected
Batch Efficiency Peak:    49.3 tok/s/% at concurrency 24
Multi-GPU Balance:        0.97 (4 GPUs, CV=0.03)

Key Correlations:
  throughput vs sm_util:      rho=+0.91 (strong positive)
  throughput vs mem_util:     rho=+0.84 (strong positive)
  per_user_tput vs sm_util:   rho=-0.67 (moderate negative)
  temperature vs power:       rho=+0.94 (strong positive, lag=3 samples)
```

**JSON** (structured):
```json
{
  "gpu_correlation": {
    "dominant_regime": "memory_bound",
    "phase_fractions": {
      "prefill": 0.18,
      "decode": 0.72,
      "mixed": 0.10
    },
    "saturation": {
      "detected": true,
      "timestamp_ns": 1738281600000000000,
      "sm_utilization_pct": 78.0,
      "throughput_tok_s": 3847.0
    },
    "power_performance": {
      "tokens_per_watt": 12.4,
      "tokens_per_joule": 12.4,
      "avg_power_w": 287.0,
      "joules_per_token": 0.081,
      "joules_per_request": 32.4
    },
    "correlations": [
      {
        "metric_a": "effective_throughput",
        "metric_b": "sm_utilization",
        "spearman_rho": 0.91,
        "p_value": 1.2e-15,
        "interpretation": "strong positive"
      }
    ],
    "thermal_throttling": {
      "events": [],
      "total_power_violation_us": 0
    },
    "batch_efficiency": {
      "optimal_concurrency": 24,
      "optimal_sm_utilization": 65.0,
      "peak_efficiency_tok_s_pct": 49.3,
      "utilization_elasticity": 0.72
    },
    "multi_gpu": {
      "num_gpus": 4,
      "utilization_cv": 0.03,
      "balance_score": 0.97,
      "max_utilization_gap_pct": 4.2
    }
  }
}
```

**CSV** (flat rows for each correlation pair and summary metrics).

### 13.6 Dependency Considerations

The correlation analysis depends on:

1. **GPU telemetry enabled** (`--gpu-telemetry` not disabled) — required for
   any GPU metrics.
2. **SM utilization available** (`DCGM_FI_PROF_SM_ACTIVE` in DCGM exporter) —
   this is a **profiling metric** that requires DCGM to be configured with
   profiling enabled. Not all deployments have this. Fallback to
   `gpu_utilization` with degraded accuracy.
3. **Sufficient overlap** — at least 10 aligned data points for meaningful
   correlation (5 minimum for statistical significance, 10 for robustness).
4. **Sweep curves computed** — requires at least a few completed requests
   in the MetricsAccumulator.

The analyzer should gracefully degrade when signals are missing:

```python
def _check_prerequisites(self, aligned: AlignedCorrelationData) -> list[str]:
    """Check which analyses are possible given available data."""
    available = []
    if not np.all(np.isnan(aligned.sm_utilization)):
        available.append("roofline")
        available.append("batch_efficiency")
        available.append("saturation")
    if not np.all(np.isnan(aligned.mem_utilization)):
        available.append("bandwidth_bottleneck")
    if not np.all(np.isnan(aligned.gpu_power_usage)):
        available.append("power_performance")
    if not np.all(np.isnan(aligned.gpu_temperature)):
        available.append("thermal_throttling")
    return available
```

### 13.7 Performance Considerations

The correlation analysis runs **once** at the end of the benchmark (during
`export_results()`), not in real-time. Performance requirements:

- **Time complexity**: O(n log n) for alignment (searchsorted), O(n) for
  correlation computation. Total: O(n log n) where n is the number of DCGM
  samples (typically 100-10000 for a 10s-2hr benchmark).
- **Memory**: Aligned dataset is ~10 float64 arrays of length n. At 10000
  samples, this is ~800KB. Negligible.
- **No scipy dependency for core path**: Use numpy-only implementations for
  Pearson (dot product), Spearman (argsort), and p-values (t-distribution
  approximation). The scipy import in the pseudocode above is illustrative.

### 13.8 Numpy-Only Correlation Implementations

To avoid adding a scipy dependency (keeping with AIPerf's numpy-only analysis
philosophy), here are pure numpy implementations:

```python
def pearson_r(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Pearson correlation coefficient with p-value (numpy-only).

    p-value uses the t-distribution approximation:
        t = r * sqrt((n-2) / (1-r^2))
    with n-2 degrees of freedom.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    sx, sy = np.sqrt(np.dot(dx, dx)), np.sqrt(np.dot(dy, dy))

    if sx == 0 or sy == 0:
        return 0.0, 1.0

    r = float(np.dot(dx, dy) / (sx * sy))

    # t-statistic for significance test
    if abs(r) >= 1.0:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r * r))

    # Two-tailed p-value from t-distribution (approximation)
    # Using the regularized incomplete beta function approximation
    p_value = _t_distribution_p_value(abs(t_stat), n - 2)
    return r, p_value


def spearman_rho(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Spearman rank correlation (numpy-only). Delegates to pearson_r on ranks."""
    return pearson_r(_rank(x), _rank(y))


def _rank(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute fractional ranks (average rank for ties)."""
    order = np.argsort(arr)
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _t_distribution_p_value(t: float, df: int) -> float:
    """Approximate two-tailed p-value from t-distribution.

    Uses the approximation: p ~ 2 * (1 - Phi(t * (1 - 1/(4*df))))
    where Phi is the standard normal CDF. Accurate for df > 5.
    """
    # Adjusted t for normal approximation
    t_adj = t * (1.0 - 1.0 / (4.0 * max(df, 1)))
    # Normal CDF via error function
    p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(t_adj / np.sqrt(2.0))))
    return max(0.0, min(1.0, p))


def _erf(x: float) -> float:
    """Error function approximation (Abramowitz & Stegun 7.1.26). |error| < 1.5e-7."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y
```

---

## 14. Formulas Reference

### 14.1 Efficiency Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| Tokens per watt | `throughput_tok_s / power_W` | tok/s/W |
| Tokens per joule | `total_tokens / energy_delta_J` | tok/J |
| Joules per token | `energy_delta_J / total_tokens` | J/tok |
| Joules per request | `energy_delta_J / total_requests` | J/req |
| Tokens per kWh | `total_tokens / (energy_J / 3.6e6)` | tok/kWh |
| Batch efficiency | `throughput_tok_s / sm_utilization_pct` | tok/s/% |
| Power-performance ratio | `throughput_tok_s / power_W` | tok/s/W |

### 14.2 Utilization Metrics

| Metric | Formula | Unit |
|--------|---------|------|
| Compute-memory ratio (CMR) | `sm_utilization / mem_utilization` | dimensionless |
| Marginal throughput efficiency | `dT/dU` | tok/s/% |
| Utilization elasticity | `d(log T) / d(log U)` | dimensionless |
| Thermal headroom | `TjMax - gpu_temperature` | C |
| Power throttle rate | `d(power_violation) / dt` | us/s |
| GPU stability (CV) | `1 - std(sm_util) / mean(sm_util)` | dimensionless |
| Utilization balance (CV) | `std(sm_util_per_gpu) / mean(sm_util_per_gpu)` | dimensionless |

### 14.3 Roofline Parameters

| Parameter | Prefill Phase | Decode Phase |
|-----------|--------------|--------------|
| Operational intensity | `B * S / elem_size` | `B / elem_size` |
| Primary bottleneck | Compute (SM) | Memory bandwidth |
| Key utilization metric | `sm_utilization` | `mem_utilization` |
| Scaling with batch size | Superlinear | Linear |
| Saturation indicator | `sm_util > 85%` | `mem_util > 80%` |

### 14.4 Correlation Interpretation

| |rho| Range | Strength | Confidence at n=100 |
|------------|----------|---------------------|
| 0.9 - 1.0 | Very strong | p < 1e-30 |
| 0.7 - 0.9 | Strong | p < 1e-15 |
| 0.4 - 0.7 | Moderate | p < 1e-4 |
| 0.2 - 0.4 | Weak | p < 0.05 |
| 0.0 - 0.2 | Negligible | Not significant |

### 14.5 Energy Conversion Constants

```
1 MJ  = 1,000,000 J
1 kWh = 3,600,000 J = 3.6 MJ
1 W   = 1 J/s
```

DCGM reports `energy_consumption` in **millijoules** (mJ), which AIPerf
scales by `1e-9` to **megajoules** (MJ) in `SCALING_FACTORS`. For
correlation analysis, convert to joules:

```
energy_J = energy_consumption_MJ * 1e6
```

---

## 15. Open Questions

### 15.1 SM Utilization Availability

`DCGM_FI_PROF_SM_ACTIVE` is a **profiling metric** that requires DCGM to be
running with profiling enabled (`dcgmi profile --set-config`). In Kubernetes
deployments with the GPU operator, this may not be enabled by default. The
fallback to `gpu_utilization` (which is always available) severely degrades
the analysis — should we:

1. Warn the user and proceed with `gpu_utilization`?
2. Skip roofline/batch efficiency analysis entirely?
3. Provide a "quality of analysis" indicator in the output?

**Recommendation**: Option 3 — include a `signal_quality` field in the
result indicating which metrics were available and how that affects
confidence in the analysis.

### 15.2 Clock Synchronization

When benchmarking remote inference servers (the common case), the client
machine's clock and the GPU server's clock may differ by tens of
milliseconds (NTP accuracy) to seconds (unsynchronized clocks). For
correlation analysis:

- **Does absolute time alignment matter?** For overall correlation (Pearson/Spearman
  over the full window), sub-second misalignment is irrelevant because both
  signals change on timescales of seconds.
- **Does it matter for event detection?** For thermal throttling detection
  (Section 8), a few seconds of misalignment could cause false negatives.
  The lag-correlation analysis (Section 12.4) handles this automatically.

**Recommendation**: Document that clock synchronization within 1 second is
assumed. Add a `clock_offset_estimate_ns` field computed from
cross-correlation of the ramp-up patterns.

### 15.3 Multi-Endpoint Aggregation

When AIPerf targets multiple inference endpoints (load-balanced deployment),
each endpoint may have its own DCGM source. The correlation analysis needs
to decide:

1. **Per-endpoint analysis**: Separate correlation per endpoint. Most accurate
   but verbose.
2. **Aggregate analysis**: Average utilization across all endpoints, correlate
   with total throughput. Simpler but loses per-endpoint detail.
3. **Both**: Aggregate for summary, per-endpoint for detailed output.

**Recommendation**: Option 3, with aggregate in the console output and
per-endpoint in JSON/CSV exports.

### 15.4 Causal vs. Correlational

Correlation does not imply causation. Specifically:

- Throughput and utilization both increase with load — their correlation may
  be driven by a **common cause** (load level) rather than one causing the
  other.
- To establish causal direction (does higher utilization *cause* higher
  throughput, or does higher throughput *cause* higher utilization?), we would
  need **Granger causality** tests or **instrumental variable** analysis, which
  is beyond the scope of this initial implementation.

**Recommendation**: Use careful language in outputs: "correlated with", not
"caused by". Add a note in documentation.

### 15.5 Interaction with Steady-State Analysis

The correlation analysis is most meaningful within the steady-state window
(where both signals are stable). Outside steady state, the correlation
structure is dominated by the ramp-up/ramp-down trends, which inflates
correlation coefficients artificially (the "common trend" problem).

**Recommendation**: Always compute correlations on the steady-state window
when available. If steady-state analysis is disabled, use the full profiling
window but add a warning about potential trend inflation.

### 15.6 Continuous Batching Complications

Modern inference servers use **continuous batching** (iteration-level
scheduling), which means:
- The batch composition changes at every iteration
- Prefill and decode requests are mixed within the same batch
- GPU utilization reflects the aggregate of all concurrent operations

This makes it difficult to attribute utilization to specific phases
(prefill vs decode). The CMR approach (Section 6.3) provides a
probabilistic classification but cannot distinguish individual request
phases within a mixed batch.

**Recommendation**: Acknowledge this limitation. The phase classification
represents the **dominant mode** at each DCGM sample, not a precise
decomposition.

---

## 16. References

1. **Williams, S., Waterman, A., Patterson, D.** (2009). "Roofline: An
   Insightful Visual Performance Model for Multicore Architectures."
   Communications of the ACM, 52(4), pp. 65-76.
   *The original roofline model paper.*

2. **NVIDIA DCGM Documentation** — "DCGM Field Identifiers."
   https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/
   *Definitions of DCGM_FI_DEV_GPU_UTIL, DCGM_FI_PROF_SM_ACTIVE, etc.*

3. **Yu, G.-I., Jeong, J.S., Kim, G.-W., Kim, S., Chun, B.-G.** (2022).
   "Orca: A Distributed Serving System for Transformer-Based Generative
   Models." Proceedings of OSDI.
   *Continuous batching (iteration-level scheduling) for LLM inference.*

4. **Agrawal, A., et al.** (2024). "Sarathi-Serve: A Disaggregated Architecture
   for Efficient LLM Serving." arXiv:2403.02310.
   *Prefill-decode disaggregation and its effect on GPU utilization patterns.*

5. **Pope, R., et al.** (2023). "Efficiently Scaling Transformer Inference."
   Proceedings of MLSys.
   *Analysis of memory-bandwidth vs compute bottlenecks in transformer inference.*

6. **Dean, J. and Barroso, L.A.** (2013). "The Tail at Scale."
   Communications of the ACM, 56(2), pp. 74-80.
   *Tail latency analysis at scale — relevant to understanding throughput-latency
   tradeoffs under high GPU utilization.*

7. **Kwon, W., et al.** (2023). "Efficient Memory Management for Large Language
   Model Serving with PagedAttention." Proceedings of SOSP.
   *vLLM's PagedAttention — KV cache management affects memory utilization patterns.*

8. **NVIDIA** (2024). "NVIDIA H100 Tensor Core GPU Architecture Whitepaper."
   *Hardware specifications for roofline parameters: peak FLOP/s, HBM bandwidth,
   TjMax, TDP.*

9. **Killick, R., Fearnhead, P., and Eckley, I.A.** (2012). "Optimal
   Detection of Changepoints with a Linear Computational Cost." Journal of
   the American Statistical Association, 107(500), pp. 1590-1598.
   *PELT algorithm referenced for utilization change-point detection.*

10. **NVIDIA** (2024). "Data Center GPU Manager (DCGM) User Guide: Profiling
    Metrics." Section on `DCGM_FI_PROF_SM_ACTIVE` vs `DCGM_FI_DEV_GPU_UTIL`.
    *Critical distinction between temporal occupancy and spatial occupancy.*
