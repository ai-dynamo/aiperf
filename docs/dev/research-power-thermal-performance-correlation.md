<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Power & Thermal Throttling Impact on Inference Performance

**Status**: Research

**Authors**: Anthony Casagrande

**Category**: Research — Correlation Analysis

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: GPU Power and Thermal Architecture](#2-background-gpu-power-and-thermal-architecture)
3. [Available Telemetry in AIPerf](#3-available-telemetry-in-aiperf)
4. [Power Throttling Detection & Quantification](#4-power-throttling-detection--quantification)
5. [Thermal Trajectory Prediction](#5-thermal-trajectory-prediction)
6. [Power-Limited vs Thermal-Limited Regimes](#6-power-limited-vs-thermal-limited-regimes)
7. [Energy Efficiency Over Time](#7-energy-efficiency-over-time)
8. [Cooling Adequacy Assessment](#8-cooling-adequacy-assessment)
9. [Multi-GPU Thermal Cascading](#9-multi-gpu-thermal-cascading)
10. [Workload-Dependent Power Profiles](#10-workload-dependent-power-profiles)
11. [Data Center Implications](#11-data-center-implications)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [References](#13-references)

---

## 1. Introduction

LLM inference benchmarking assumes that the hardware under test operates at
stable performance throughout the measurement window. This assumption breaks
down under two conditions: **power throttling** (the GPU hits its TDP limit and
the firmware reduces clock frequencies) and **thermal throttling** (the GPU die
temperature exceeds its thermal design threshold and the firmware progressively
reduces performance to prevent damage).

Both phenomena are invisible to the client-side measurement stack unless GPU
telemetry is collected alongside inference metrics. A benchmark that reports
"P99 latency = 142ms" without checking whether the GPU throttled during the
measurement is reporting a number that may not be reproducible — or worse, may
be optimistic relative to sustained production workloads.

This document researches the correlation between GPU power/thermal behavior and
LLM inference performance, focusing on:

- **Detection**: How to identify when throttling occurs during a benchmark
- **Quantification**: How much performance is lost to throttling
- **Prediction**: Whether throttling can be anticipated from early telemetry
- **Causation**: Distinguishing power-limited from thermal-limited regimes
- **Efficiency**: How energy efficiency degrades over time
- **Scale**: Multi-GPU thermal interactions and data center power management

All analysis is grounded in the DCGM telemetry metrics that AIPerf already
collects, with concrete formulas and implementation guidance for extending the
existing accumulator pipeline.

### Why This Matters for LLM Inference

LLM inference has a distinctive power profile compared to training:

1. **Phase-dependent power draw**: Prefill (attention matrix computation) is
   compute-bound and draws near-TDP power. Decode (autoregressive token
   generation) is memory-bandwidth-bound and draws 60-80% TDP. The ratio of
   prefill to decode changes dynamically with request mix.

2. **Latency sensitivity**: Unlike training (where throughput is king), inference
   SLOs are latency-bound. A 5% clock reduction from throttling translates
   directly to 5-15% latency increase, potentially violating P99 SLOs.

3. **Long-running workloads**: Production inference servers run continuously.
   A 5-minute benchmark may never reach thermal steady state, making its
   results non-representative of sustained performance.

4. **Dense deployments**: Data center GPU servers pack 8 GPUs into a single
   chassis. Thermal coupling between GPUs means one GPU's heat output affects
   its neighbors.

---

## 2. Background: GPU Power and Thermal Architecture

### 2.1 GPU Power Delivery

Modern NVIDIA data center GPUs (A100, H100, H200, B200) have a power delivery
architecture with several relevant limits:

```
                        ┌─────────────────────────────────────┐
                        │          GPU Board Power            │
                        │                                     │
  PCIe slot power ──────┤   ┌──────────┐   ┌──────────────┐  │
  (75W max)             │   │ Voltage   │   │ Power        │  │
                        │   │ Regulators│──>│ Management   │  │
  Aux power connectors ─┤   │ (VRMs)    │   │ Controller   │  │
  (150-600W)            │   └──────────┘   └──────┬───────┘  │
                        │                          │          │
                        │   ┌──────────────────────▼───────┐  │
                        │   │  GPU Die                     │  │
                        │   │  ┌─────┐ ┌─────┐ ┌────────┐ │  │
                        │   │  │ SMs │ │ HBM │ │ NVLink │ │  │
                        │   │  │     │ │ I/O │ │        │ │  │
                        │   │  └─────┘ └─────┘ └────────┘ │  │
                        │   └──────────────────────────────┘  │
                        │                                     │
                        │   ┌──────────────────────────────┐  │
                        │   │  HBM Memory Stacks           │  │
                        │   │  (separate power rail)       │  │
                        │   └──────────────────────────────┘  │
                        └─────────────────────────────────────┘
```

**Key power limits:**

| GPU     | TDP (W) | Typical Idle (W) | Prefill Peak (W) | Decode Typical (W) |
|---------|---------|-------------------|-------------------|--------------------|
| A100    | 300-400 | 50-75             | 290-380           | 180-240            |
| H100    | 350-700 | 60-100            | 330-680           | 210-420            |
| H200    | 700     | 80-120            | 650-690           | 350-500            |
| B200    | 1000    | 100-150           | 900-980           | 500-700            |

### 2.2 Thermal Management

GPU thermal management operates in three zones:

```
Temperature (°C)
    │
 95 ├─────────────────────────────────── TjMax (thermal shutdown)
    │                              ▓▓▓▓▓ Aggressive throttling zone
 85 ├─────────────────────────────────── Thermal throttle threshold
    │                         ░░░░░░░░░░ Progressive clock reduction
 75 ├─────────────────────────────────── Clock boost limit
    │                    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒ Reduced boost headroom
 65 ├─────────────────────────────────── Nominal operating range
    │         ████████████████████████████ Full boost clock available
 40 ├─────────────────────────────────── Idle temperature
    │
    └──────────────────────────────────── Time →
```

The firmware continuously adjusts GPU clock frequency based on both power draw
and temperature. The effective clock is:

```
f_effective = min(f_boost, f_power_limit, f_thermal_limit)

where:
  f_boost         = maximum boost clock (spec sheet value)
  f_power_limit   = max clock achievable within TDP
  f_thermal_limit = max clock achievable within thermal envelope
```

### 2.3 DCGM Counter Semantics

The `power_violation` field reported by DCGM is a **cumulative counter** measured
in microseconds. It represents the total duration the GPU has been clock-gated
due to exceeding its power limit since the driver was loaded. Key properties:

- **Monotonically increasing**: The counter only goes up (except on driver reset)
- **Microsecond granularity**: Each microsecond of throttling adds 1 to the counter
- **Per-GPU**: Each GPU has its own independent counter
- **Includes all power-related throttling**: Both sustained and transient events

The `energy_consumption` field is also cumulative, measured in millijoules by
DCGM (converted to megajoules by AIPerf's `DCGMTelemetryCollector`). The
relationship between instantaneous power and cumulative energy is:

```
P(t) = dE/dt

E(t₂) - E(t₁) = ∫[t₁ to t₂] P(t) dt
```

---

## 3. Available Telemetry in AIPerf

### 3.1 GPU Telemetry (DCGM)

AIPerf collects these metrics via the `GPUTelemetryAccumulator` (implemented in
`src/aiperf/gpu_telemetry/accumulator.py`), with field definitions in
`src/aiperf/gpu_telemetry/constants.py`:

| Metric              | Type    | Unit         | DCGM Field                          | Semantics                |
|---------------------|---------|--------------|--------------------------------------|--------------------------|
| `gpu_power_usage`   | Gauge   | W            | `DCGM_FI_DEV_POWER_USAGE`           | Instantaneous power draw |
| `energy_consumption`| Counter | MJ           | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | Cumulative energy     |
| `power_violation`   | Counter | microseconds | `DCGM_FI_DEV_POWER_VIOLATION`       | Cumulative throttle time |
| `gpu_temperature`   | Gauge   | °C           | `DCGM_FI_DEV_GPU_TEMP`              | GPU die temperature      |
| `gpu_utilization`   | Gauge   | %            | `DCGM_FI_DEV_GPU_UTIL`              | GPU compute utilization  |
| `sm_utilization`    | Gauge   | %            | `DCGM_FI_PROF_SM_ACTIVE`            | SM active percentage     |
| `gpu_memory_used`   | Gauge   | GB           | `DCGM_FI_DEV_FB_USED`               | Framebuffer memory used  |

Counter metrics (`energy_consumption`, `power_violation`, `xid_errors`) are
handled specially by `GpuMetricTimeSeries.to_metric_result_filtered()` — they
report the delta between a baseline measurement before profiling start and the
final measurement, rather than statistical summaries.

### 3.2 Client-Side Metrics

Collected by the `MetricsAccumulator` via `ColumnStore`:

| Metric                       | Unit       | Relevance to Power/Thermal          |
|------------------------------|------------|--------------------------------------|
| `request_latency`            | ms         | Directly affected by clock throttling|
| `time_to_first_token`        | ms         | Prefill-phase indicator              |
| `inter_token_latency`        | ms         | Decode-phase indicator               |
| `effective_throughput`       | tokens/sec | Inverse relationship to throttling   |
| `effective_concurrency`      | requests   | Load metric (drives power)           |
| `tokens_in_flight`           | tokens     | Active work (drives power)           |
| `effective_prefill_concurrency` | requests | Prefill load (high power phase)   |
| `effective_generation_concurrency` | requests | Decode load (lower power phase) |

### 3.3 Data Collection Architecture

```
┌────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ DCGM Exporter  │────>│ GPUTelemetry     │────>│ GPUTelemetry       │
│ (Prometheus)   │HTTP │ Manager          │ ZMQ │ Accumulator        │
│ /metrics       │     │ (collector loop) │     │ (TelemetryHierarchy│
└────────────────┘     └──────────────────┘     │  └─ GpuMetric      │
                                                │     TimeSeries)    │
┌────────────────┐     ┌──────────────────┐     └────────────────────┘
│ LLM Server     │────>│ Workers          │────>┌────────────────────┐
│ (inference)    │HTTP │ (send requests)  │ ZMQ │ MetricsAccumulator │
└────────────────┘     └──────────────────┘     │ (ColumnStore)      │
                                                └────────────────────┘
```

Both accumulators run in the same `RecordProcessor` process and share the same
wall-clock timeline. The `timestamp_ns` fields in `TelemetryRecord` and
`ColumnStore` are directly comparable, enabling cross-correlation between GPU
telemetry and client-side metrics.

---

## 4. Power Throttling Detection & Quantification

### 4.1 Delta Analysis of the power_violation Counter

The `power_violation` counter is cumulative microseconds of throttling. The
fundamental analysis is delta computation over a time window:

```
throttle_rate(t₁, t₂) = Δ(power_violation) / Δ(time)
                       = (PV(t₂) - PV(t₁)) / (t₂ - t₁)
```

Where:
- `PV(t)` is the `power_violation` counter value at time `t`
- `throttle_rate` is dimensionless (microseconds of throttling per microsecond
  of wall time), ranging from 0.0 (no throttling) to 1.0 (continuous throttling)

**Interpretation:**

| Throttle Rate | Meaning                                        | Expected Impact        |
|---------------|------------------------------------------------|------------------------|
| 0.00          | No power throttling occurred                   | None                   |
| 0.00 - 0.01   | Transient throttling (< 1% of time)           | Negligible             |
| 0.01 - 0.10   | Intermittent throttling                        | 1-10% throughput loss  |
| 0.10 - 0.50   | Significant throttling                         | 10-40% throughput loss |
| 0.50 - 1.00   | Severe/continuous throttling                   | 40-80% throughput loss |

### 4.2 Windowed Throttle Rate Computation

For time-series analysis, compute the throttle rate over sliding windows to
identify when throttling begins and how it evolves:

```python
def compute_throttle_rate_series(
    timestamps_ns: NDArray[np.int64],
    power_violation: NDArray[np.float64],
    window_size: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute windowed throttle rate from cumulative power_violation counter.

    Args:
        timestamps_ns: Sorted telemetry timestamps in nanoseconds.
        power_violation: Cumulative power_violation values (microseconds).
        window_size: Number of samples per sliding window.

    Returns:
        (window_centers_ns, throttle_rates) — aligned time series.
    """
    if len(timestamps_ns) < window_size + 1:
        return np.array([]), np.array([])

    # Delta over each window
    dt_ns = timestamps_ns[window_size:] - timestamps_ns[:-window_size]
    dt_us = dt_ns / 1e3  # nanoseconds -> microseconds

    dpv = power_violation[window_size:] - power_violation[:-window_size]

    # Guard against zero time intervals
    valid = dt_us > 0
    rates = np.zeros_like(dpv)
    rates[valid] = dpv[valid] / dt_us[valid]

    # Clamp to [0, 1] (counter should be monotonic, but handle resets)
    rates = np.clip(rates, 0.0, 1.0)

    centers = (timestamps_ns[window_size:] + timestamps_ns[:-window_size]) // 2
    return centers, rates
```

### 4.3 Throttle-Throughput Correlation

The core hypothesis: when throttle rate increases, throughput decreases. The
Pearson correlation coefficient between windowed throttle rate and windowed
throughput quantifies this relationship:

```
ρ(throttle_rate, throughput) = Cov(R, T) / (σ_R × σ_T)
```

Where `R` is the throttle rate series and `T` is the throughput sweep series
(from `throughput_sweep()` in `src/aiperf/analysis/sweep.py`).

**Cross-correlation with lag**: Power throttling may have a delayed effect on
throughput due to request queuing and batching. The cross-correlation function
identifies the optimal lag:

```python
def cross_correlate_throttle_throughput(
    throttle_ts: NDArray[np.float64],
    throttle_rate: NDArray[np.float64],
    throughput_ts: NDArray[np.float64],
    throughput: NDArray[np.float64],
    max_lag_samples: int = 20,
) -> tuple[int, float]:
    """Find the lag at which throttle rate best predicts throughput drop.

    Returns:
        (optimal_lag, correlation_at_lag) — negative lag means throttling
        leads throughput drop.
    """
    # Interpolate to common time grid
    common_ts = np.union1d(throttle_ts, throughput_ts)
    r_interp = np.interp(common_ts, throttle_ts, throttle_rate)
    t_interp = np.interp(common_ts, throughput_ts, throughput)

    # Normalize
    r_norm = (r_interp - np.mean(r_interp)) / (np.std(r_interp) + 1e-12)
    t_norm = (t_interp - np.mean(t_interp)) / (np.std(t_interp) + 1e-12)

    # Compute cross-correlation for each lag
    n = len(common_ts)
    best_lag, best_corr = 0, 0.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            corr = np.dot(r_norm[:n - lag], t_norm[lag:]) / (n - abs(lag))
        else:
            corr = np.dot(r_norm[-lag:], t_norm[:n + lag]) / (n - abs(lag))
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr

    return best_lag, best_corr
```

Expected result: strong negative correlation (ρ < -0.7) with a small negative
lag (throttling leads throughput drop by a few collection intervals).

### 4.4 Throttle Event Detection

Beyond continuous throttle rate, discrete throttle events can be detected from
the derivative of the power_violation counter:

```
Event detection:
  1. Compute Δ(power_violation) between consecutive samples
  2. A throttle event starts when Δ > threshold (e.g., > 100 µs per interval)
  3. A throttle event ends when Δ returns to 0 for N consecutive samples

Throttle event timeline:
  time ──────────────────────────────────────────────────►
  PV    0  0  0  150  300  280  50  0  0  0  400  380  0
                  ▲────────────────▲        ▲─────────▲
                  Event 1           End     Event 2    End
                  duration: 780µs           duration: 780µs
```

Each detected event can be correlated with the concurrent client-side metrics
to measure per-event impact:

```python
@dataclass(frozen=True, slots=True)
class ThrottleEvent:
    """A discrete power throttling event."""

    start_ns: int        # Wall-clock start
    end_ns: int          # Wall-clock end
    total_throttle_us: float  # Total throttling microseconds during event
    peak_rate: float     # Peak throttle rate during event
    mean_power_w: float  # Mean power draw during event
    mean_temp_c: float   # Mean temperature during event
```

### 4.5 Quantifying Performance Impact

Given a set of throttle events, the aggregate performance impact is:

```
throughput_loss_fraction = 1 - (throughput_during_throttling / throughput_outside_throttling)

latency_inflation = mean_latency_during_throttling / mean_latency_outside_throttling
```

This requires segmenting the client-side time series into throttled and
non-throttled windows, then comparing the distributions:

```
                  Non-throttled       Throttled        Non-throttled
Throughput   ████████████████████░░░░░░░░░░░░████████████████████
             ~1200 tok/s           ~900 tok/s        ~1200 tok/s
                                   (-25%)

P99 latency  ────────────────────────────────────────────────────
             ~140ms               ~190ms            ~140ms
                                   (+36%)
```

---

## 5. Thermal Trajectory Prediction

### 5.1 Lumped Thermal Model

GPU thermal behavior under constant load follows Newton's Law of Cooling
(a first-order linear ODE). The GPU die is treated as a single thermal mass
with a thermal resistance to the ambient environment:

```
C × dT/dt = P(t) - (T(t) - T_ambient) / R_th

where:
  C        = thermal capacitance of GPU die + heatsink (J/°C)
  T(t)     = GPU temperature at time t
  T_ambient = ambient / inlet air temperature
  P(t)     = power dissipation at time t
  R_th     = thermal resistance die-to-ambient (°C/W)
```

For constant power `P`, the solution is the classic exponential approach:

```
T(t) = T_ambient + P × R_th × (1 - e^(-t/τ)) + (T_0 - T_ambient) × e^(-t/τ)

where:
  τ = R_th × C     (thermal time constant, typically 30-120 seconds for GPUs)
  T_0              (initial temperature)
  T_steady = T_ambient + P × R_th   (steady-state temperature)
```

### 5.2 Parameter Estimation from Telemetry

Given a time series of `(timestamp_ns, gpu_temperature, gpu_power_usage)`, we
can estimate the thermal model parameters by fitting the exponential model to
observed data:

```python
def estimate_thermal_parameters(
    timestamps_ns: NDArray[np.int64],
    temperatures: NDArray[np.float64],
    power_watts: NDArray[np.float64],
    ambient_temp: float = 25.0,
) -> tuple[float, float, float]:
    """Estimate thermal model parameters from telemetry data.

    Uses linearized least-squares on the exponential model:
      T(t) = T_steady - (T_steady - T_0) × e^(-t/τ)

    For the warm-up phase (rising temperature), this can be linearized:
      ln(T_steady - T(t)) = ln(T_steady - T_0) - t/τ

    Args:
        timestamps_ns: Sorted telemetry timestamps.
        temperatures: GPU die temperature series (°C).
        power_watts: Instantaneous power draw series (W).
        ambient_temp: Assumed ambient temperature (°C).

    Returns:
        (tau_seconds, r_thermal, t_steady) — time constant, thermal resistance,
        predicted steady-state temperature.
    """
    # Convert timestamps to seconds from start
    t_sec = (timestamps_ns - timestamps_ns[0]).astype(np.float64) / 1e9

    # Estimate steady-state from mean power and asymptotic temperature
    mean_power = np.mean(power_watts)

    # Use the last 20% of data to estimate R_th if temperature is stabilizing
    tail_start = int(0.8 * len(temperatures))
    t_tail_mean = np.mean(temperatures[tail_start:])

    # R_th = (T_steady - T_ambient) / P_mean
    r_thermal = (t_tail_mean - ambient_temp) / mean_power if mean_power > 0 else 0.0
    t_steady = ambient_temp + mean_power * r_thermal

    # Estimate tau from linearized exponential fit on rising portion
    rising_mask = temperatures < 0.95 * t_steady
    if np.sum(rising_mask) > 5:
        delta_t = t_steady - temperatures[rising_mask]
        delta_t = np.maximum(delta_t, 0.1)  # avoid log(0)
        log_delta = np.log(delta_t)

        # Linear regression: log_delta = a - t/tau
        t_rising = t_sec[rising_mask]
        n = len(t_rising)
        if n > 2:
            slope = (n * np.dot(t_rising, log_delta) - np.sum(t_rising) * np.sum(log_delta)) / \
                    (n * np.dot(t_rising, t_rising) - np.sum(t_rising)**2 + 1e-12)
            tau_seconds = -1.0 / slope if slope < 0 else 60.0  # default 60s
        else:
            tau_seconds = 60.0
    else:
        tau_seconds = 60.0  # default if already near steady state

    return tau_seconds, r_thermal, t_steady
```

### 5.3 Throttle Threshold Prediction

Given estimated parameters, predict when the GPU will reach the thermal
throttling threshold:

```
Time to throttle threshold:
  t_throttle = -τ × ln((T_threshold - T_steady) / (T_0 - T_steady))

Only valid when T_steady > T_threshold (i.e., the GPU will eventually throttle)
If T_steady ≤ T_threshold, the GPU will stabilize below the threshold.
```

```
Temperature trajectory prediction:

  T(°C)
  90 ┤                                         ╌╌╌╌╌ T_throttle = 85°C
     │                                   ╱─────
  80 ┤                              ╱───╱      Predicted (exponential fit)
     │                         ╱───╱
  70 ┤                    ╱───╱
     │               ●───╱
  60 ┤          ●───●╱             ● = observed data points
     │     ●───●
  50 ┤●───●                        ↑
     │                         t_throttle predicted here
  40 ┤
     └────┬────┬────┬────┬────┬────┬────┬────── Time (seconds)
          0   30   60   90  120  150  180  210
```

### 5.4 Multi-Phase Thermal Model

LLM inference power draw is not constant — it varies with the prefill/decode
mix. A piecewise thermal model accounts for changing power:

```
For time intervals [t_k, t_{k+1}] with approximately constant power P_k:

  T(t) = T_ambient + P_k × R_th + (T(t_k) - T_ambient - P_k × R_th) × e^(-(t - t_k)/τ)
```

This can be computed iteratively across DCGM collection intervals:

```python
def predict_thermal_trajectory(
    timestamps_ns: NDArray[np.int64],
    power_watts: NDArray[np.float64],
    initial_temp: float,
    tau: float,
    r_thermal: float,
    ambient_temp: float = 25.0,
) -> NDArray[np.float64]:
    """Predict temperature trajectory from power series and thermal model.

    Uses piecewise-constant power assumption between telemetry samples.

    Args:
        timestamps_ns: Future (or historical) timestamps to predict at.
        power_watts: Power draw at each timestamp (piecewise-constant).
        initial_temp: Starting temperature.
        tau: Thermal time constant (seconds).
        r_thermal: Thermal resistance (°C/W).
        ambient_temp: Ambient temperature (°C).

    Returns:
        Predicted temperature at each timestamp.
    """
    temps = np.empty(len(timestamps_ns), dtype=np.float64)
    temps[0] = initial_temp
    t_sec = timestamps_ns.astype(np.float64) / 1e9

    for i in range(1, len(timestamps_ns)):
        dt = t_sec[i] - t_sec[i - 1]
        p = power_watts[i - 1]
        t_ss = ambient_temp + p * r_thermal  # local steady-state
        decay = np.exp(-dt / tau) if tau > 0 else 0.0
        temps[i] = t_ss + (temps[i - 1] - t_ss) * decay

    return temps
```

### 5.5 Thermal Time Constant by GPU Generation

Empirical thermal time constants vary significantly by GPU form factor and
cooling solution:

| GPU / Form Factor     | Typical τ (seconds) | Cooling Type        |
|----------------------|---------------------|---------------------|
| A100 PCIe             | 60-90               | Passive heatsink    |
| A100 SXM4             | 45-70               | Active heatsink     |
| H100 SXM5             | 40-60               | Vapor chamber       |
| H100 PCIe             | 55-80               | Passive heatsink    |
| H200 SXM5             | 35-55               | Enhanced vapor      |
| B200 (liquid cooled)  | 15-25               | Liquid cooling      |
| B200 (air cooled)     | 40-65               | Air cooling         |

Liquid-cooled GPUs have dramatically shorter time constants (lower R_th), meaning
they reach thermal steady state faster. This is relevant for benchmark duration
recommendations (Section 8).

---

## 6. Power-Limited vs Thermal-Limited Regimes

### 6.1 Two Distinct Throttling Mechanisms

Power and thermal throttling have fundamentally different signatures in
telemetry data:

```
Power-Limited:                          Thermal-Limited:

Power (W)                               Temperature (°C)
TDP ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─           T_throttle ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    ████████████████████████                              ████████████████
    █                      █                         ████
    █      Clamped at TDP  █                    ████
    █                      █               ████
    ████████████████████████          ████
                                 ████
────────────────────────────  ────────────────────────────
         Time                          Time

Throughput Impact:                      Throughput Impact:
    ┌─────────────────────┐                 ────────────
    │                     │ Step           ╲
    │  Immediate drop     │ function        ╲  Gradual
    │  when TDP reached   │                  ╲  degradation
    └─────────────────────┘                   ╲─────────
```

### 6.2 Regime Classification Algorithm

Classify the throttling regime by examining the co-evolution of power,
temperature, and the power_violation counter:

```python
from enum import StrEnum

class ThrottleRegime(StrEnum):
    NONE = "none"                      # No throttling detected
    POWER_LIMITED = "power_limited"    # TDP capping active
    THERMAL_LIMITED = "thermal_limited"  # Thermal throttling active
    BOTH = "power_and_thermal"         # Both mechanisms active


def classify_throttle_regime(
    power_watts: NDArray[np.float64],
    temperatures: NDArray[np.float64],
    throttle_rate: NDArray[np.float64],
    tdp_watts: float,
    thermal_threshold_c: float = 83.0,
    power_proximity_pct: float = 95.0,
) -> ThrottleRegime:
    """Classify the dominant throttling regime during a measurement window.

    Heuristic:
    - Power-limited: power near TDP AND throttle_rate > 0 AND temperature below threshold
    - Thermal-limited: temperature near/above threshold AND power may be reduced
    - Both: power at TDP AND temperature at threshold
    - None: neither condition met

    Args:
        power_watts: Instantaneous power draw series.
        temperatures: GPU die temperature series.
        throttle_rate: Computed throttle rate series (0-1).
        tdp_watts: GPU TDP in watts.
        thermal_threshold_c: Temperature above which thermal throttling occurs.
        power_proximity_pct: % of TDP above which power-limiting is likely.

    Returns:
        ThrottleRegime enum indicating the dominant regime.
    """
    power_limit = tdp_watts * power_proximity_pct / 100.0

    near_tdp = np.mean(power_watts > power_limit) > 0.3  # >30% of time near TDP
    near_thermal = np.mean(temperatures > thermal_threshold_c) > 0.1  # >10% time hot
    is_throttling = np.mean(throttle_rate > 0.01) > 0.05  # >5% of time throttling

    if near_tdp and near_thermal and is_throttling:
        return ThrottleRegime.BOTH
    elif near_tdp and is_throttling and not near_thermal:
        return ThrottleRegime.POWER_LIMITED
    elif near_thermal:
        return ThrottleRegime.THERMAL_LIMITED
    else:
        return ThrottleRegime.NONE
```

### 6.3 Correlation Signatures by Regime

The cross-correlation structure differs between regimes:

**Power-limited:**
```
Correlation matrix:
                  power    temp    throttle   throughput   latency
power            1.00     0.3-0.5   0.8-0.95  -0.7--0.9    0.7-0.9
temperature      0.3-0.5  1.00      0.2-0.4   -0.1--0.3    0.1-0.3
throttle_rate    0.8-0.95 0.2-0.4   1.00      -0.8--0.95   0.8-0.95
throughput      -0.7--0.9 ...       ...        1.00        -0.85--0.95
latency          0.7-0.9  ...       ...       -0.85--0.95   1.00

Key: Strong power-throttle-throughput axis. Temperature is a bystander.
```

**Thermal-limited:**
```
Correlation matrix:
                  power    temp    throttle   throughput   latency
power            1.00     0.6-0.8   0.5-0.7   -0.4--0.6    0.4-0.6
temperature      0.6-0.8  1.00      0.7-0.9   -0.6--0.85   0.6-0.85
throttle_rate    0.5-0.7  0.7-0.9   1.00      -0.7--0.9    0.7-0.9
throughput      -0.4--0.6 ...       ...        1.00        -0.85--0.95
latency          0.4-0.6  ...       ...       -0.85--0.95   1.00

Key: Strong temperature-throttle-throughput axis. Power may actually decrease
     (GPU is shedding power to reduce temperature).
```

### 6.4 Time-Domain Signatures

```
Power-Limited Event:
  t₀           t₁                t₂            t₃
  │            │                 │             │
  │  Load      │  Power hits    │  Throttle   │  Load
  │  ramps up  │  TDP limit     │  active     │  decreases
  │            │                 │             │
  power  ▁▃▅▇██████████████████████████████████▇▅▃▁
  temp   ▁▁▂▃▃▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▃▂▁
  throt  ▁▁▁▁▁▅████████████████████████████████▅▁▁▁
  tput   ████████▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆█████

  Characteristic: Power saturates first, temperature follows slowly.
  Throttle coincides with power saturation, NOT temperature peak.

Thermal-Limited Event:
  t₀           t₁                t₂            t₃
  │            │                 │             │
  │  Temp      │  Temp crosses  │  Firmware   │  Temp
  │  rising    │  threshold     │  reduces    │  stabilizes
  │            │                 │  clocks     │
  power  ████████████████████████▇▆▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅
  temp   ▁▂▃▄▅▆▇████████████████████████████████████
  throt  ▁▁▁▁▁▁▁▁▁▁▂▃▅▆▇████████████████████████████
  tput   ██████████████▇▇▆▆▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄

  Characteristic: Temperature rises gradually. Throttling is progressive.
  Power actually DROPS (clock reduction reduces compute, reducing power).
```

---

## 7. Energy Efficiency Over Time

### 7.1 Joules per Token as an Efficiency Metric

Energy efficiency for LLM inference is best expressed as energy per output
token:

```
η(t) = P(t) / throughput(t)    [Joules per token]

where:
  P(t)          = instantaneous power draw (watts = joules/second)
  throughput(t) = output tokens per second at time t
```

For cumulative measurement over a window:

```
η_window = ΔE / total_tokens_generated

where:
  ΔE = delta(energy_consumption) over the window (Joules)
  total_tokens_generated = sum of output tokens in the window
```

### 7.2 Efficiency Degradation Trend

Even before explicit throttling, energy efficiency degrades as temperature
rises because:

1. **Leakage current increases exponentially with temperature**: At higher
   temperatures, more power is wasted as leakage (does no useful work).
2. **Voltage regulators become less efficient**: Higher junction temperatures
   reduce VRM efficiency.
3. **Memory bandwidth degrades**: HBM thermal throttling can reduce bandwidth
   before the GPU reports throttling.

The empirical model for leakage power:

```
P_leakage(T) = P_leakage_ref × 2^((T - T_ref) / k)

where:
  k ≈ 10-15°C (doubling temperature for leakage)
  T_ref = reference temperature (e.g., 25°C)
  P_leakage_ref = leakage power at reference temperature

Total power = P_dynamic + P_leakage(T)
```

This means the same workload draws more power at higher temperatures, leaving
less headroom before hitting TDP:

```
Efficiency over a 30-minute benchmark:

  η (J/token)
  0.8 ┤
      │  ●                                              Warm-up
  0.7 ┤    ● ●                                          transient
      │        ● ● ● ● ●                               (ignore this
  0.6 ┤                    ● ● ●                         region)
      │                          ● ●
  0.5 ┤                               ● ● ● ● ●
      │                                          ● ● ● ● ● ●  Degrading
  0.4 ┤                                                        efficiency
      │
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──── Time (min)
         0  2  4  6  8  10 12 14 16 18 20 22 24 26 28 30
              steady-state window ───────────────────►

  The trend within the steady-state window reveals whether the benchmark
  duration was sufficient for thermal stabilization.
```

### 7.3 Trending Analysis

Detect efficiency degradation by applying the same `batch_means_trend_test()`
from `src/aiperf/analysis/stationarity.py` to the J/token time series:

```python
def compute_efficiency_trend(
    power_ts: NDArray[np.int64],
    power_watts: NDArray[np.float64],
    throughput_ts: NDArray[np.float64],
    throughput: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Compute energy efficiency time series and its trend.

    Args:
        power_ts: Timestamps for power readings (ns).
        power_watts: Instantaneous power (W).
        throughput_ts: Timestamps for throughput sweep values (ns).
        throughput: Throughput values (tokens/sec).

    Returns:
        (efficiency_ts, efficiency_j_per_tok, spearman_rho, p_value)
    """
    # Interpolate throughput to power timestamps
    tput_interp = np.interp(
        power_ts.astype(np.float64),
        throughput_ts,
        throughput,
    )

    # Avoid division by zero
    valid = tput_interp > 0
    efficiency = np.full_like(power_watts, np.nan)
    efficiency[valid] = power_watts[valid] / tput_interp[valid]

    # Trend test on valid efficiency values
    from aiperf.analysis.stationarity import batch_means_trend_test
    valid_efficiency = efficiency[~np.isnan(efficiency)]

    if len(valid_efficiency) < 20:
        return power_ts, efficiency, 0.0, 1.0

    rho, p_value = batch_means_trend_test(valid_efficiency)
    return power_ts, efficiency, rho, p_value
```

**Interpretation:**

| Spearman ρ | p-value | Meaning                                              |
|-----------|---------|------------------------------------------------------|
| < 0.1      | > 0.05 | No trend — efficiency is stable                      |
| 0.1 - 0.3  | < 0.05 | Mild upward trend — leakage increasing               |
| 0.3 - 0.6  | < 0.01 | Moderate trend — thermal effects visible              |
| > 0.6      | < 0.001| Strong trend — likely approaching throttle threshold  |

A positive ρ means J/token is increasing (efficiency is degrading). A negative ρ
means J/token is decreasing (efficiency is improving, typical during warm-up as
caches fill and batching stabilizes).

---

## 8. Cooling Adequacy Assessment

### 8.1 Problem Statement

If the GPU temperature has not stabilized by the end of a benchmark, the
reported results are not representative of sustained production performance.
The throughput and latency measured during a still-warming GPU will be better
than what the GPU delivers once it reaches thermal equilibrium.

### 8.2 Temperature Stabilization Test

A GPU is considered thermally stabilized when the rate of temperature change
falls below a threshold:

```
dT/dt < ε  for the final k% of the benchmark

where:
  ε = 0.1°C per minute (practical threshold for data center GPUs)
  k = 20% (analyze the last 20% of the benchmark duration)
```

```python
def assess_cooling_adequacy(
    timestamps_ns: NDArray[np.int64],
    temperatures: NDArray[np.float64],
    rate_threshold_c_per_min: float = 0.1,
    tail_fraction: float = 0.2,
) -> tuple[bool, float, float]:
    """Assess whether GPU temperature has stabilized.

    Fits a linear trend to the tail of the temperature series.
    If the slope exceeds the threshold, the GPU is still warming up
    and benchmark results may not represent sustained performance.

    Args:
        timestamps_ns: Sorted telemetry timestamps.
        temperatures: GPU die temperature series (°C).
        rate_threshold_c_per_min: Max acceptable rate of temperature change.
        tail_fraction: Fraction of data to use for stability assessment.

    Returns:
        (is_stable, rate_c_per_min, predicted_steady_state_c)
    """
    n = len(timestamps_ns)
    tail_start = int(n * (1 - tail_fraction))
    if tail_start >= n - 2:
        return True, 0.0, float(temperatures[-1])

    t_tail = timestamps_ns[tail_start:]
    temp_tail = temperatures[tail_start:]

    # Convert to minutes from tail start
    t_min = (t_tail - t_tail[0]).astype(np.float64) / 6e10

    # Linear regression for slope
    n_tail = len(t_min)
    slope = (
        n_tail * np.dot(t_min, temp_tail) - np.sum(t_min) * np.sum(temp_tail)
    ) / (n_tail * np.dot(t_min, t_min) - np.sum(t_min) ** 2 + 1e-12)

    # Predict steady state from exponential extrapolation
    current_temp = float(temp_tail[-1])
    # If still rising, estimate how much higher it will go
    # using the thermal model: T_steady ≈ T_current + slope × τ
    tau_minutes = 1.0  # conservative estimate
    predicted_steady = current_temp + slope * tau_minutes

    is_stable = abs(slope) < rate_threshold_c_per_min
    return is_stable, float(slope), predicted_steady
```

### 8.3 Minimum Benchmark Duration Recommendations

Based on the thermal time constant τ, the minimum benchmark duration for
thermal representativeness is:

```
t_min = 3τ    (reaches 95% of steady-state temperature)
t_good = 5τ   (reaches 99.3% of steady-state temperature)

For air-cooled H100 (τ ≈ 50s):
  t_min  = 150s (2.5 minutes)
  t_good = 250s (4.2 minutes)

For liquid-cooled B200 (τ ≈ 20s):
  t_min  = 60s  (1 minute)
  t_good = 100s (1.7 minutes)
```

These are minimums from a thermal perspective only. Steady-state detection
(CUSUM + MSER-5) will also enforce sufficient data for statistical validity.

### 8.4 Integration with Steady-State Analysis

The cooling adequacy assessment complements the existing steady-state detection
in `src/aiperf/post_processors/steady_state_analyzer.py`. Currently, the
`SteadyStateAnalyzer` operates on four signals: concurrency, latency, TTFT,
and throughput. Adding a fifth signal — GPU temperature stability — would
strengthen the detection:

```
Current signals:
  1. CUSUM on concurrency curve (load perspective)
  2. MSER-5 on latency (performance perspective)
  3. MSER-5 on TTFT (prefill performance perspective)
  4. CUSUM on throughput curve (output perspective)

Proposed addition:
  5. Temperature stabilization test (hardware perspective)

Combined boundary:
  ramp_up_end   = max(all signal starts, temperature_stabilization_time)
  ramp_down_start = min(all signal ends)
```

This would prevent the steady-state window from starting before the GPU has
thermally stabilized, which can happen when the load ramp-up is fast but
the GPU is still warming.

### 8.5 Cooling Adequacy Report

```
Cooling Adequacy Assessment
───────────────────────────────────────────────────────────
GPU 0 (NVIDIA H100 SXM):
  Final temperature:       78.2°C
  Temperature trend:       +0.03°C/min (STABLE)
  Estimated time constant: 48.3s
  Benchmark duration:      300s (6.2τ — SUFFICIENT)
  Predicted steady-state:  78.5°C
  Thermal headroom:        6.8°C to throttle threshold

GPU 1 (NVIDIA H100 SXM):
  Final temperature:       81.4°C
  Temperature trend:       +0.42°C/min (STILL RISING)
  Estimated time constant: 52.1s
  Benchmark duration:      300s (5.8τ — MARGINAL)
  Predicted steady-state:  84.1°C
  Thermal headroom:        0.9°C to throttle threshold
  ⚠ WARNING: GPU 1 will likely throttle under sustained load.
             Consider extending benchmark duration or checking airflow.
───────────────────────────────────────────────────────────
```

---

## 9. Multi-GPU Thermal Cascading

### 9.1 Thermal Coupling in Dense GPU Systems

Data center GPU servers (DGX, HGX) pack 4-8 GPUs into a single chassis with
shared airflow. The thermal behavior of one GPU affects its neighbors:

```
DGX H100 Layout (top view, airflow left to right):

  Airflow →
  ┌─────────────────────────────────────────────────┐
  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐            │
  │  │GPU 0│  │GPU 1│  │GPU 2│  │GPU 3│   Inlet     │
  │  │ 72°C│  │ 75°C│  │ 78°C│  │ 81°C│   side      │
  │  └─────┘  └─────┘  └─────┘  └─────┘            │
  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐            │
  │  │GPU 4│  │GPU 5│  │GPU 6│  │GPU 7│   Exhaust   │
  │  │ 73°C│  │ 76°C│  │ 79°C│  │ 83°C│   side      │
  │  └─────┘  └─────┘  └─────┘  └─────┘            │
  └─────────────────────────────────────────────────┘

  Note: GPUs downstream in airflow run 3-10°C hotter than upstream GPUs
  with identical workload. GPU 7 is the "hottest seat" in the chassis.
```

### 9.2 Cross-GPU Temperature Correlation

The temperature of downstream GPUs depends on the power dissipation of
upstream GPUs. This creates a coupling matrix:

```
T_i(t) = T_ambient + Σ_j [P_j(t) × R_thermal(j→i)] × (1 - e^(-t/τ_i))

where:
  R_thermal(j→i) = thermal coupling resistance from GPU j to GPU i
  R_thermal(i→i) = self-heating resistance (dominant term)
  R_thermal(j→i) for j ≠ i = cross-coupling resistance (smaller, nonzero for neighbors)
```

In practice, for airflow-cooled systems:

```
Coupling matrix R_thermal (schematic, arbitrary units):

        GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7
GPU 0  [1.00   0.05   0.01   0.00   0.03   0.01   0.00   0.00]
GPU 1  [0.15   1.00   0.05   0.01   0.01   0.03   0.01   0.00]
GPU 2  [0.05   0.15   1.00   0.05   0.00   0.01   0.03   0.01]
GPU 3  [0.01   0.05   0.15   1.00   0.00   0.00   0.01   0.03]
GPU 4  [0.10   0.01   0.00   0.00   1.00   0.05   0.01   0.00]
GPU 5  [0.01   0.10   0.01   0.00   0.15   1.00   0.05   0.01]
GPU 6  [0.00   0.01   0.10   0.01   0.05   0.15   1.00   0.05]
GPU 7  [0.00   0.00   0.01   0.10   0.01   0.05   0.15   1.00]

Note: Asymmetric — upstream GPUs heat downstream, not vice versa.
Diagonal is dominant (self-heating). Off-diagonal decays with distance.
```

### 9.3 Staggered Throttling Detection

When GPUs are thermally coupled, throttling events propagate through the
chassis with a time delay:

```
Timeline of cascading throttle events:

GPU 0   ────────────────────────────────────────────────────
GPU 1   ────────────────────────────────────────────────────
GPU 2   ──────────────────────────────────█████████─────────  throttle at t=120s
GPU 3   ────────────────────────████████████████████████████  throttle at t=100s
GPU 4   ────────────────────────────────────────────────────
GPU 5   ────────────────────────────────────────────────────
GPU 6   ────────────────────────────────────████████████████  throttle at t=130s
GPU 7   ──────────────────────███████████████████████████████  throttle at t=90s

Pattern: GPU 7 (hottest position) throttles first. Its reduced throughput
shifts load to other GPUs, which may accelerate their approach to throttling.
```

### 9.4 Cross-GPU Correlation Analysis

```python
def compute_cross_gpu_thermal_correlation(
    gpu_temperatures: dict[str, NDArray[np.float64]],
    gpu_timestamps: dict[str, NDArray[np.int64]],
) -> NDArray[np.float64]:
    """Compute pairwise temperature correlation between GPUs.

    Interpolates all GPU temperature series to a common time grid,
    then computes the Pearson correlation matrix.

    Args:
        gpu_temperatures: Dict of gpu_uuid -> temperature series.
        gpu_timestamps: Dict of gpu_uuid -> timestamp series.

    Returns:
        N x N correlation matrix where N = number of GPUs.
    """
    gpu_ids = sorted(gpu_temperatures.keys())
    n_gpus = len(gpu_ids)

    if n_gpus < 2:
        return np.eye(n_gpus)

    # Build common time grid from union of all timestamps
    all_ts = np.concatenate([gpu_timestamps[g] for g in gpu_ids])
    common_ts = np.unique(all_ts)

    # Interpolate each GPU's temperature to common grid
    interp_temps = np.empty((n_gpus, len(common_ts)), dtype=np.float64)
    for i, gpu_id in enumerate(gpu_ids):
        interp_temps[i] = np.interp(
            common_ts.astype(np.float64),
            gpu_timestamps[gpu_id].astype(np.float64),
            gpu_temperatures[gpu_id],
        )

    # Pearson correlation matrix
    return np.corrcoef(interp_temps)
```

### 9.5 Identifying the Thermal Bottleneck GPU

In a multi-GPU inference setup, the overall system throughput is limited by the
slowest GPU. Thermal cascading means the "hottest seat" GPU determines system
performance:

```
System throughput = min(GPU_i throughput for all i)

If GPU 7 throttles to 80% performance while others maintain 100%:
  System throughput = 80% (for tensor-parallel workloads)

The thermal bottleneck GPU is:
  bottleneck_gpu = argmax(temperature) across all GPUs
  OR
  bottleneck_gpu = argmax(throttle_rate) across all GPUs
```

---

## 10. Workload-Dependent Power Profiles

### 10.1 Prefill vs Decode Power Characteristics

LLM inference has two distinct computational phases with very different power
profiles:

```
Prefill Phase (prompt processing):
  - Operation: Dense matrix multiplication (QKV projection, attention, FFN)
  - Bound by: Compute (SM utilization > 90%)
  - Power: Near TDP (>90% of max power)
  - Duration: Proportional to prompt length
  - Memory: Read weights, write KV cache

Decode Phase (token generation):
  - Operation: Small matrix-vector multiply + KV cache reads
  - Bound by: Memory bandwidth (mem_utilization > 80%)
  - Power: 60-80% of TDP
  - Duration: Proportional to output length
  - Memory: Read weights + growing KV cache, write 1 token
```

### 10.2 Power Profile Model

The instantaneous power draw of a GPU serving LLM inference can be modeled as:

```
P(t) = P_idle + P_prefill × n_prefill(t) / n_prefill_max +
       P_decode × n_decode(t) / n_decode_max

where:
  P_idle       = idle power (leakage + fans + HBM refresh)
  P_prefill    = additional power for full prefill utilization
  P_decode     = additional power for full decode utilization
  n_prefill(t) = number of prefill requests at time t
  n_decode(t)  = number of decode requests at time t
  n_prefill_max = max concurrent prefill capacity
  n_decode_max  = max concurrent decode capacity
```

### 10.3 Cross-Correlation with Concurrency Phases

AIPerf's sweep metrics already decompose concurrency into prefill and decode
phases via `effective_prefill_concurrency` and `effective_generation_concurrency`
(from `src/aiperf/analysis/sweep.py`). The cross-correlation with power reveals
the workload structure:

```python
def analyze_workload_power_profile(
    power_ts: NDArray[np.int64],
    power_watts: NDArray[np.float64],
    prefill_conc_ts: NDArray[np.float64],
    prefill_concurrency: NDArray[np.float64],
    decode_conc_ts: NDArray[np.float64],
    decode_concurrency: NDArray[np.float64],
) -> dict[str, float]:
    """Decompose power into prefill and decode contributions.

    Uses ordinary least squares to fit:
      P(t) = β₀ + β_prefill × prefill_conc(t) + β_decode × decode_conc(t) + ε

    Args:
        power_ts: Timestamps for power readings.
        power_watts: Instantaneous power draw (W).
        prefill_conc_ts: Timestamps for prefill concurrency.
        prefill_concurrency: Prefill concurrency values.
        decode_conc_ts: Timestamps for decode concurrency.
        decode_concurrency: Decode concurrency values.

    Returns:
        Dict with 'idle_power', 'prefill_power_per_request',
        'decode_power_per_request', 'r_squared'.
    """
    # Interpolate concurrency to power timestamps
    t_float = power_ts.astype(np.float64)
    pc = np.interp(t_float, prefill_conc_ts, prefill_concurrency)
    dc = np.interp(t_float, decode_conc_ts, decode_concurrency)

    # OLS regression: P = β₀ + β₁×prefill + β₂×decode
    n = len(power_watts)
    X = np.column_stack([np.ones(n), pc, dc])
    # Normal equation: β = (X'X)^(-1) X'y
    XtX = X.T @ X
    Xty = X.T @ power_watts
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {"idle_power": float(np.mean(power_watts)),
                "prefill_power_per_request": 0.0,
                "decode_power_per_request": 0.0,
                "r_squared": 0.0}

    # R-squared
    y_pred = X @ beta
    ss_res = np.sum((power_watts - y_pred) ** 2)
    ss_tot = np.sum((power_watts - np.mean(power_watts)) ** 2)
    r_sq = 1 - ss_res / (ss_tot + 1e-12)

    return {
        "idle_power": float(beta[0]),
        "prefill_power_per_request": float(beta[1]),
        "decode_power_per_request": float(beta[2]),
        "r_squared": float(r_sq),
    }
```

### 10.4 Power Fluctuation Spectrum

The frequency content of power fluctuations reveals the workload structure:

```
Power spectral density (schematic):

  PSD
  │
  │  ●        Dominant frequency = batch scheduling frequency
  │  │●       (how often new prefill batches are formed)
  │  │ ●
  │  │  ●●
  │  │    ●●●
  │  │       ●●●●●●●●
  │  │                ●●●●●●●●●●●●●●●●●●●●●●
  └──┴──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──── Frequency (Hz)
        0  0.1 0.5 1  2  5  10

Low frequency (< 0.1 Hz): Load level changes, ramp-up/ramp-down
Mid frequency (0.1 - 2 Hz): Batch scheduling, prefill/decode cycling
High frequency (> 2 Hz): Individual request-level variation
```

### 10.5 Prefill-Heavy Workloads and Throttling Risk

Workloads with a high prefill-to-decode ratio (short outputs, long prompts)
sustain higher average power and are more likely to trigger throttling:

```
Scenario comparison (same total requests):

Scenario A: Prompt=128 tokens, Output=512 tokens
  Time profile: |P|------D------|  (P = prefill, D = decode)
  Prefill fraction: ~10% of time
  Average power: ~70% TDP
  Throttle risk: LOW

Scenario B: Prompt=4096 tokens, Output=64 tokens
  Time profile: |----P----|D|
  Prefill fraction: ~80% of time
  Average power: ~92% TDP
  Throttle risk: HIGH

Scenario C: Summarization (prompt=8192, output=256)
  Time profile: |--------P--------|--D--|
  Prefill fraction: ~85% of time
  Average power: ~94% TDP
  Throttle risk: VERY HIGH — near-continuous prefill at TDP
```

---

## 11. Data Center Implications

### 11.1 Power Capping

Data center operators often impose power caps below GPU TDP to manage rack-level
power budgets:

```
Example: 40kW rack power limit

Without power cap:
  8 × H100 SXM at 700W TDP = 5,600W GPU power
  + ~1,200W system overhead = 6,800W per node
  Nodes per rack: 40,000 / 6,800 = 5.88 → 5 nodes = 40 GPUs

With power cap at 500W per GPU:
  8 × 500W = 4,000W GPU power
  + ~1,200W system overhead = 5,200W per node
  Nodes per rack: 40,000 / 5,200 = 7.69 → 7 nodes = 56 GPUs

Tradeoff: 40% more GPUs but each at ~85% peak performance
Net capacity change depends on throttling behavior.
```

### 11.2 Power Cap Impact on Inference SLOs

When a GPU is power-capped below its natural demand, the `power_violation`
counter accumulates continuously. The performance impact is proportional to
how far below demand the cap is set:

```
Performance vs Power Cap:

  Relative
  Performance (%)
  100 ┤─────────────●
      │              \
   95 ┤               ●          Prefill-bound (compute-limited)
      │                \
   90 ┤                 ●
      │                  \●
   85 ┤                    ●
      │                     \
   80 ┤                      ●
      │                       \
   75 ┤                        ●  Decode-bound (memory-limited, less affected)
      │
   70 ┤
      └──┬──┬──┬──┬──┬──┬──┬──┬── Power Cap (% of TDP)
         60 65 70 75 80 85 90 95 100

  Two curves: prefill performance drops earlier because it is compute-bound
  and sensitive to clock frequency. Decode is memory-bound and less affected
  until severe power caps reduce memory controller frequency.
```

### 11.3 P99 Latency Stability Under Power Caps

Power capping creates a specific pattern in the latency distribution:

```
Without power cap (full TDP):
  Latency distribution: narrow, low variance
  │
  │    ████
  │   ██████
  │  ████████
  │ ██████████
  │████████████
  └──────────────── Latency (ms)
     100  150  200

With aggressive power cap (70% TDP):
  Latency distribution: bimodal (prefill-limited tail)
  │
  │   ███                 █
  │  █████               ███
  │ ███████             █████
  │█████████           ███████
  └──────────────────────────── Latency (ms)
     120  180  240  300  360

  The second mode appears because prefill phases are elongated by clock
  throttling, while decode phases are relatively unaffected. Requests
  with longer prompts form the right tail.
```

### 11.4 Power Cap Headroom Analysis

```python
def analyze_power_cap_headroom(
    power_watts: NDArray[np.float64],
    power_cap_watts: float,
    latency_p99: float,
    latency_slo_ms: float,
) -> dict[str, float]:
    """Analyze power cap headroom relative to SLO compliance.

    Args:
        power_watts: Instantaneous power draw series.
        power_cap_watts: Configured power cap (or TDP if no cap).
        latency_p99: Observed P99 latency (ms).
        latency_slo_ms: Target SLO for P99 latency (ms).

    Returns:
        Dict with power headroom metrics.
    """
    mean_power = float(np.mean(power_watts))
    peak_power = float(np.max(power_watts))
    p99_power = float(np.percentile(power_watts, 99))

    # How often are we at the power cap?
    at_cap_fraction = float(np.mean(power_watts > power_cap_watts * 0.98))

    # Power headroom: how much room before we hit the cap?
    headroom_watts = power_cap_watts - peak_power
    headroom_pct = headroom_watts / power_cap_watts * 100

    # SLO headroom: how close is P99 latency to the SLO?
    slo_headroom_ms = latency_slo_ms - latency_p99
    slo_headroom_pct = slo_headroom_ms / latency_slo_ms * 100

    return {
        "mean_power_watts": mean_power,
        "peak_power_watts": peak_power,
        "p99_power_watts": p99_power,
        "power_cap_watts": power_cap_watts,
        "power_headroom_watts": headroom_watts,
        "power_headroom_pct": headroom_pct,
        "time_at_cap_fraction": at_cap_fraction,
        "latency_p99_ms": latency_p99,
        "latency_slo_ms": latency_slo_ms,
        "slo_headroom_ms": slo_headroom_ms,
        "slo_headroom_pct": slo_headroom_pct,
    }
```

### 11.5 Rack-Level Thermal Interactions

Beyond individual chassis thermal coupling, rack-level effects include:

1. **Hot aisle / cold aisle**: GPUs in the same rack share airflow. If one
   server runs hotter, its exhaust preheats the intake of the server above it.

2. **Power distribution**: Rack-level PDUs have maximum amperage. If all
   servers simultaneously prefill (e.g., a burst of long prompts), the
   aggregate power draw may exceed PDU capacity, triggering rack-level
   power management.

3. **Cooling capacity**: CRAC/CRAH units have finite cooling capacity. During
   heatwaves or cooling failures, inlet air temperature rises, reducing all
   GPUs' thermal headroom simultaneously.

These effects are outside AIPerf's direct measurement scope but should be
documented as context for interpreting benchmark results that show unexpected
throttling.

---

## 12. Implementation Roadmap

### 12.1 Phase 1: Throttle Detection (Low Effort)

Extend the existing `GPUTelemetryAccumulator` export to include derived
throttling metrics.

**New derived metrics from existing counters:**

| Metric                  | Computation                               | Type    |
|-------------------------|-------------------------------------------|---------|
| `throttle_rate`         | Δ(power_violation) / Δ(time)              | Derived |
| `energy_per_token`      | Δ(energy_consumption) / total_output_tokens | Derived |
| `mean_power_headroom_w` | TDP - mean(gpu_power_usage)               | Derived |
| `peak_power_headroom_w` | TDP - max(gpu_power_usage)                | Derived |

**Where it fits:** These metrics can be computed in `export_results()` of the
`GPUTelemetryAccumulator` (in `src/aiperf/gpu_telemetry/accumulator.py`) using
existing data from `GpuMetricTimeSeries`. The `power_violation` counter delta
is already computed for the counter metric; the throttle rate is simply
normalizing it by the time window duration.

**Integration with existing counter handling:**

The `to_metric_result_filtered()` method in
`src/aiperf/common/models/telemetry_models.py` already handles counter deltas.
The throttle rate is:

```python
throttle_rate = power_violation_delta_us / (duration_ns / 1e3)
```

### 12.2 Phase 2: Cooling Adequacy Warning (Low Effort)

Add a temperature stability check to the export pipeline.

**Implementation:** A standalone function that analyzes the `gpu_temperature`
time series from `GpuMetricTimeSeries` and returns a stability assessment.
Called during `export_results()` and included in the JSON/CSV output.

**Reporting:** Add to `SteadyStateSummary.to_json()` output as a new
`thermal_stability` group alongside the existing `quality`, `stationarity`,
`cross_validation`, and `bootstrap` groups.

### 12.3 Phase 3: Throttle-Throughput Correlation (Medium Effort)

Add an optional `ThermalAnalyzer` implementing `AnalyzerProtocol` (similar to
`SteadyStateAnalyzer`).

**Data requirements:** Needs access to both GPU telemetry time series and
client-side metric time series. This requires either:
- (a) The analyzer reads from both accumulators via `SummaryContext`, or
- (b) The analyzer receives pre-computed sweep curves and telemetry arrays.

Option (b) is preferred — keeps the analyzer focused on computation rather
than data access, consistent with `SteadyStateAnalyzer`'s pattern of
receiving `SweepCurves` from `MetricsAccumulator`.

**Output:** A `ThermalCorrelationSummary` containing:
- Throttle regime classification
- Power-throughput Pearson ρ and optimal lag
- Temperature-throughput Pearson ρ
- Efficiency trend (Spearman ρ, p-value)
- Cross-GPU temperature correlation matrix (if multi-GPU)

### 12.4 Phase 4: Workload Power Decomposition (Medium Effort)

Compute the prefill/decode power decomposition using the OLS regression from
Section 10.3.

**Data requirements:** Needs `effective_prefill_concurrency` and
`effective_generation_concurrency` sweep curves from `MetricsAccumulator`
alongside the GPU power time series.

**Output:** Per-GPU power decomposition coefficients (idle power,
prefill watts per request, decode watts per request) plus R-squared
goodness of fit.

### 12.5 Phase 5: Multi-GPU Thermal Cascading (High Effort)

Analyze cross-GPU thermal interactions using the correlation matrix approach.

**Data requirements:** Temperature time series from all GPUs, requiring the
`TelemetryHierarchy` to expose per-GPU time series arrays. This is already
structured correctly — `dcgm_endpoints -> gpu_uuid -> GpuTelemetryData -> time_series`.

**Output:**
- Cross-GPU temperature correlation matrix
- Thermal bottleneck GPU identification
- Staggered throttle event detection
- Recommended GPU placement adjustments (if correlation exceeds threshold)

### 12.6 Phase 6: Data Center Power Cap Analysis (Low Effort, Optional)

Add CLI parameter for specifying GPU TDP or power cap, enabling headroom
analysis.

**New configuration:**

```python
class PowerAnalysisConfig(BaseConfig):
    """Configuration for power and thermal analysis."""

    gpu_tdp_watts: float | None = Field(
        default=None,
        description="GPU TDP in watts. Auto-detected from GPU model if not specified.",
    )
    power_cap_watts: float | None = Field(
        default=None,
        description="Configured power cap in watts (if lower than TDP).",
    )
    thermal_throttle_threshold_c: float = Field(
        default=83.0,
        description="Temperature threshold above which thermal throttling occurs.",
    )
    ambient_temperature_c: float = Field(
        default=25.0,
        description="Ambient / inlet air temperature for thermal modeling.",
    )
```

### 12.7 Summary Table

| Phase | Feature                        | Effort | Dependencies         | New Files                  |
|-------|-------------------------------|--------|----------------------|---------------------------|
| P1    | Throttle detection metrics     | Low    | None                 | None (extend accumulator) |
| P2    | Cooling adequacy warning       | Low    | P1                   | None (extend export)      |
| P3    | Throttle-throughput correlation | Medium | P1, sweep curves     | `analysis/thermal.py`     |
| P4    | Workload power decomposition   | Medium | P3, sweep curves     | Extend `thermal.py`       |
| P5    | Multi-GPU thermal cascading    | High   | P3, multi-GPU data   | Extend `thermal.py`       |
| P6    | Data center power cap analysis | Low    | P1                   | Config extension          |

### 12.8 Testing Strategy

All new analysis functions should be tested against synthetic telemetry data
with known properties, following the pattern established by the steady-state
synthetic validation suite:

**Synthetic profiles:**

| Profile               | Characteristics                                    | Expected Result              |
|-----------------------|----------------------------------------------------|------------------------------|
| `constant_power`      | Flat power at 80% TDP, no throttling               | regime=NONE, ρ≈0            |
| `power_capped`        | Power at TDP, throttle_rate=0.3, step throughput    | regime=POWER_LIMITED, ρ<-0.7|
| `thermal_ramp`        | Exponential temp rise, gradual throughput drop       | regime=THERMAL_LIMITED       |
| `mixed_throttling`    | Power cap + thermal ramp                            | regime=BOTH                 |
| `prefill_heavy`       | High prefill ratio, oscillating power               | decomposition R²>0.8       |
| `decode_heavy`        | Low power, memory-bound                             | decomposition R²>0.8       |
| `multi_gpu_cascade`   | 8 GPUs, staggered temperatures                      | correlation matrix matches  |
| `stable_cooling`      | Temperature stabilized, no trend                    | is_stable=True              |
| `inadequate_cooling`  | Temperature still rising at end                     | is_stable=False             |
| `efficiency_degrading`| J/token increasing over time                        | positive Spearman ρ         |

---

## 13. References

### GPU Architecture and Thermal Management

1. NVIDIA DCGM Documentation — Field identifiers and counter semantics for
   `DCGM_FI_DEV_POWER_VIOLATION`, `DCGM_FI_DEV_GPU_TEMP`, and related fields.

2. NVIDIA GPU Boost Technology — Dynamic clock frequency adjustment based on
   power and thermal constraints. Describes the interaction between TDP limits,
   thermal limits, and effective clock frequency.

3. NVIDIA Data Center GPU Specifications — TDP values, TjMax, and thermal
   design parameters for A100, H100, H200, and B200 GPUs.

### Thermal Modeling

4. Incropera, F.P., DeWitt, D.P., "Fundamentals of Heat and Mass Transfer"
   — Lumped capacitance thermal model (Newton's Law of Cooling applied to
   electronic components with Biot number < 0.1).

5. Bar-Cohen, A., "Thermal Management of Microelectronics" — Thermal
   resistance networks for multi-chip modules, applicable to multi-GPU
   thermal coupling analysis.

6. ASHRAE TC 9.9, "Thermal Guidelines for Data Processing Environments" —
   Recommended inlet air temperature ranges (A1: 15-32°C) and humidity
   limits for data center equipment.

### Correlation Analysis

7. Pearson, K., "Notes on Regression and Inheritance in the Case of Two
   Parents" — Proceedings of the Royal Society of London, 1895. Foundation
   for the Pearson correlation coefficient used in throttle-throughput
   correlation.

8. Spearman, C., "The Proof and Measurement of Association between Two
   Things" — American Journal of Psychology, 1904. Rank correlation used
   in trend detection for efficiency degradation.

### LLM Inference Performance

9. Kwon, W., et al., "Efficient Memory Management for Large Language Model
   Serving with PagedAttention" — SOSP 2023 (vLLM paper). Describes the
   prefill/decode phase distinction and memory management that affects
   power profiles.

10. NVIDIA TensorRT-LLM Documentation — In-flight batching, continuous
    batching, and the impact of batch scheduling on power draw patterns.

### Benchmarking Methodology

11. Dean, J. and Barroso, L.A., "The Tail at Scale" — Communications of
    the ACM, 2013. Motivation for analyzing tail latency in the context
    of power throttling events.

12. Law, A.M. and Kelton, W.D., "Simulation Modeling and Analysis" —
    Steady-state estimation methods (MSER, CUSUM) applied in AIPerf's
    existing steady-state detection, relevant to thermal steady-state
    assessment.

### Energy Efficiency

13. Patterson, D., et al., "Carbon Emissions and Large Neural Network
    Training" — 2021. Energy efficiency metrics for AI workloads,
    motivating joules-per-token measurement.

14. Strubell, E., et al., "Energy and Policy Considerations for Deep
    Learning in NLP" — ACL 2019. Early work on energy measurement for
    neural network computation, applicable to inference workloads.

### Data Center Power Management

15. Barroso, L.A., Holzle, U., and Ranganathan, P., "The Datacenter as
    a Computer: Designing Warehouse-Scale Machines" — 3rd Edition, 2018.
    Rack-level power management, power capping strategies, and their
    impact on workload performance.

16. Fan, X., Weber, W.D., and Barroso, L.A., "Power Provisioning for a
    Warehouse-sized Computer" — ISCA 2007. Foundation for understanding
    power capping decisions at rack and cluster scale.
