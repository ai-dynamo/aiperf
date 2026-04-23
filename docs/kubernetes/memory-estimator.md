---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Memory Estimator
---

# Memory Estimator

AIPerf ships a static memory estimator that predicts the peak and steady-state
RSS of every pod in a Kubernetes deployment from the benchmark configuration.
It is the source of truth that drives resource-request/limit generation during
`aiperf kube generate`, `aiperf kube profile`, and operator preflight checks.

If your `records-manager` OOMs at 500k concurrency, if workers get killed during
ramp, or if you need to justify a resource bump — read this page first.

---

## Purpose

The estimator answers three related questions:

1. **Will this workload fit in the current memory limits?** Each `PodEstimate`
   carries both the projected peak RSS and the configured K8s limit; the
   `headroom_pct` / `at_risk` properties flag OOM risk before a job is
   submitted.
2. **What limits should I set?** `recommended_request_mib` (steady-state × 1.2)
   and `recommended_limit_mib` (peak × 1.3) come straight off the estimate.
3. **Where is the memory actually going?** Per-component breakdowns identify
   whether `RecordsManager` growable arrays, `RecordProcessor` tokenizer caches,
   or in-flight `Worker` records dominate — so tuning targets the right knob.

The model is purely static: formulas are derived from code inspection and
calibrated against real RSS measurements via
`scripts/calibrate_memory_estimates.py`. No runtime profiling is required at
estimate time.

**Consumers of estimator output:**

- `aiperf kube generate` — sets pod `resources.requests` / `resources.limits`.
- `aiperf kube profile` — prints the full estimate and warnings as part of the
  preflight report.
- Operator preflight — rejects jobs whose projected peak exceeds the namespace
  quota.

---

## Inputs

All inputs are captured in `MemoryEstimationParams`
(`src/aiperf/kubernetes/_memory_estimator/params.py`). They fall into four
groups:

### Topology

| Field | Meaning |
|---|---|
| `total_workers` | Total worker processes across all pods. |
| `workers_per_pod` | Worker processes per worker pod. |
| `num_worker_pods` | Worker pod replica count (`ceil(total_workers / workers_per_pod)`). |
| `record_processors_per_pod` | Record processors per worker pod (`workers_per_pod // RECORD_PROCESSOR_SCALE_FACTOR`). |

### Load profile

| Field | Meaning |
|---|---|
| `max_concurrency` | Peak in-flight request concurrency across all workers. |
| `total_requests` | Total requests for the entire benchmark (sum across phases). |
| `total_benchmark_duration_s` | Total benchmark wall-clock seconds. |

### Dataset shape

| Field | Meaning |
|---|---|
| `dataset_count` | Number of conversations pre-synthesized. |
| `avg_isl_tokens` | Weighted-mean input sequence length in tokens. |
| `avg_osl_tokens` | Weighted-mean output sequence length in tokens. |
| `max_turns` | Maximum conversation turns (1 for single-turn). |
| `streaming` | SSE vs. buffered-text responses (changes per-chunk accounting). |

### Observability

| Field | Meaning |
|---|---|
| `num_gpus`, `gpu_sample_interval_s`, `num_gpu_metrics` | DCGM sampling shape. |
| `num_server_metrics_endpoints`, `server_metrics_scrape_interval_s` | Prometheus scrape shape. |
| `est_unique_metric_series`, `est_histogram_metrics`, `est_histogram_buckets` | Per-endpoint series cardinality. |
| `num_models`, `num_standard_metrics`, `export_http_trace` | Per-RP tokenizer count, per-record metrics, trace export flag. |

`MemoryEstimationParams.from_config(config, total_workers, workers_per_pod,
connections_per_worker)` is the normal entry point: it derives all of the above
from an `AIPerfConfig` plus three CLI flags.

---

## Per-component model

Every per-process estimate returns a `ComponentEstimate` (see
`src/aiperf/kubernetes/_memory_estimator/estimates.py`) with four numeric
fields — `base_mib`, `variable_mib`, `peak_mib`, and a derived
`steady_state_mib = base_mib + variable_mib`.

Two universal baselines ride along on every process:

- `_PYTHON_SUBPROCESS_BASE_MIB = 35` — fresh interpreter + stdlib + GC arenas
  for control-plane subprocesses forked from `SystemController`.
- `_PYTHON_CHILD_SUBPROCESS_BASE_MIB = 18` — Worker/RP subprocesses forked from
  the WorkerProcessManager share module pages via copy-on-write; effective
  private RSS is about half of a fresh process.

Each service adds a per-service overhead from `_SERVICE_BASE_MIB` (e.g.
`records_manager: 40`, `dataset_manager: 30`, `worker: 12`,
`record_processor: 10`).

### RecordsManager

Accumulates one metric record per request for the lifetime of the benchmark.
The dominant cost is one `GrowableArray` per metric tag, each holding one
`float64` per request.

Source: `_estimate_records_manager`, `components.py`.

$$\text{variable\_mib} = \text{num\_metrics} \times \text{ceil\_pow2}(\text{total\_requests}) \times 8\text{B} \times 1.3$$

Key constants:

- `_FLOAT64_BYTES = 8` — numpy element width.
- `_GROWABLE_ARRAY_OVERHEAD = 1.3` — calibrated doubling waste factor
  (measured 1.05x–1.64x across scales).
- `_DEFAULT_NUM_STANDARD_METRICS = 25` — TTFT, TPOT, ITL, E2E, throughput, etc.
- `ceil_pow2` rounds capacity up to the next power of two (the doubling
  allocator's actual footprint, not the logical request count).

**Peak** applies a 10% finalization overhead on top of variable. Tracker
overhead (`WorkerProcessingStats` per worker) is a flat ~1 MiB.

**Warning:** if `variable_mib > 500` the estimator flags the result.

**Scales with:** `total_requests` (linearly rounded up to a power of two) and
`num_standard_metrics`.

### DatasetManager

Two memory regimes. During generation the full dataset is materialized as
Pydantic `Conversation` objects; at steady state only the mmap index survives.

Source: `_estimate_dataset_manager`, `components.py`.

$$\text{bytes\_per\_turn} = 1500 + (\text{ISL} + \text{OSL}) \times 16$$
$$\text{peak} = \text{base} + \text{dataset\_count} \times \text{max\_turns} \times \text{bytes\_per\_turn}$$
$$\text{steady} = \text{base} + \text{dataset\_count} \times 16\text{B}$$

Key constants:

- `16 bytes/token` — effective token cost after Pydantic model wrappers
  (~1 KiB per `Turn`), Python string headers, and the ~3x multiplier measured
  against ISL=100K OSL=73K with 100 entries (~297 MiB PSS).
- `_MMAP_INDEX_ENTRY_BYTES = 16` — per-conversation index entry at steady
  state.

**Scales with:** `dataset_count`, `max_turns`, `avg_isl + avg_osl`. Generation
peak dominates; steady state is negligible.

### Worker

One process per worker. Memory = connection pool + in-flight request records
+ session cache (multi-turn only).

Source: `_estimate_worker`, `components.py`. Shares the
`_per_request_bytes(avg_isl, avg_osl, streaming)` helper with RecordProcessor
so the two stay consistent.

Per in-flight request:

$$\text{per\_request} = \underbrace{1600}_{\text{record base}} + \underbrace{(400 + \text{ISL} \times 4)}_{\text{turn}} + \text{response}$$

Response depends on streaming mode:

$$\text{response}_{\text{SSE}} = 200 + \text{OSL} \times 200 \qquad \text{response}_{\text{text}} = 400 + \text{OSL} \times 4$$

Pod-level:

$$\text{variable} = \text{pool} + \text{concurrency\_per\_worker} \times \text{per\_request} + \text{session\_cache}$$

Key constants (`constants.py`):

- `_REQUEST_RECORD_BASE_BYTES = 1600` — Pydantic `RequestRecord` shell.
- `_TURN_BASE_BYTES = 400`, `_TURN_BYTES_PER_TOKEN = 4`.
- `_SSE_MESSAGE_BASE_BYTES = 200`, `_SSE_BYTES_PER_CHUNK = 200` — SSE per-token
  cost is ~50x buffered text because every chunk is a dataclass object plus a
  short JSON string.
- `_TEXT_RESPONSE_BASE_BYTES = 400`, `_TEXT_RESPONSE_BYTES_PER_TOKEN = 4`.
- `_BYTES_PER_CONNECTION = 1024` — aiohttp per-connection kernel + userspace
  buffers.

Session cache activates when `max_turns > 1`: prior-turn prompts stay resident
for the session duration.

**Scales with:** `max_concurrency / total_workers`, `avg_osl`, streaming mode,
`max_turns`, `connections_per_worker`.

### RecordProcessor

One process per RP; `record_processors_per_pod` copies live in each worker
pod. Memory = tokenizer cache + in-flight records + raw-batch and
export-batch buffers.

Source: `_estimate_record_processor`, `components.py`.

$$\text{variable} = \underbrace{\text{num\_models} \times 150}_{\text{tokenizer}} + \underbrace{\text{rp\_queue\_depth} \times \text{per\_record}}_{\text{inflight}} + \text{raw\_buf} + \text{export\_buf}$$

Key constants:

- `_TOKENIZER_CACHE_MIB = 150` — per distinct model (GPT-2 ~73 MiB, Llama-3
  ~50–100 MiB, large SentencePiece models ~150 MiB).
- `_RAW_BATCH_SIZE = 10`, `_EXPORT_BATCH_SIZE = 100`,
  `_EXPORT_BYTES_PER_RECORD = 1100` (module-local in `components.py`).

**Queue-depth amplification at high token counts** — the
`_rp_queue_depth(conc_per_rp, isl, osl)` helper in `estimator.py` models the
fact that tokenization becomes the bottleneck for ISL+OSL > 10K. Records pile
up in the RP's unbounded ZMQ pull queue as fully deserialized Python objects:

$$\text{rp\_queue\_depth} = \begin{cases} \text{conc\_per\_rp} \times \min\!\left(\tfrac{\text{ISL}+\text{OSL}}{10000}, 10\right) & \text{if ISL}+\text{OSL} > 10{,}000 \\ \text{conc\_per\_rp} & \text{otherwise} \end{cases}$$

Calibrated against PSS: at ISL+OSL=173K the queue reaches ~150 records per RP
(10x base). This is the mechanism by which large-token benchmarks OOM worker
pods even at moderate concurrency.

**Warnings:** triggers on `inflight_mib > 50` (high token pressure) or
`tokenizer_mib > 450` (too many models loaded per RP).

**Scales with:** `num_models`, `concurrency_per_rp` (amplified by token
count), streaming mode, `avg_isl + avg_osl`.

### ServerMetrics

Prometheus scrape history, one time series per metric per endpoint, held in
growable arrays identical in shape to RecordsManager.

Source: `_estimate_server_metrics`, `components.py`.

$$\text{scalar\_bytes} = \text{scalar\_count} \times \text{ceil\_pow2}(\text{n\_scrapes}) \times 16\text{B}$$
$$\text{hist\_bytes} = \text{hist\_count} \times \text{ceil\_pow2}(\text{n\_scrapes}) \times (24 + \text{buckets} \times 8)\text{B}$$
$$\text{variable} = \text{num\_endpoints} \times (\text{scalar} + \text{hist} + \text{fetch}) \times 1.3$$

Key constants (`constants.py`):

- `_DEFAULT_SCRAPE_INTERVAL_S = 5.0`
- `_DEFAULT_UNIQUE_METRIC_SERIES = 200`,
  `_DEFAULT_HISTOGRAM_METRICS = 20`,
  `_DEFAULT_HISTOGRAM_BUCKETS = 10`.

Returns zero if `num_endpoints == 0` (server metrics disabled).

**Scales with:** `num_endpoints`, `duration_s / scrape_interval_s`,
`unique_series`, `histogram_buckets`.

### GPUTelemetry

DCGM samples held as columnar numpy arrays, one per metric per GPU.

Source: `_estimate_gpu_telemetry`, `components.py`.

$$\text{per\_gpu\_bytes} = \text{ceil\_pow2}(\text{n\_samples}) \times 8 + \text{num\_metrics} \times \text{ceil\_pow2}(\text{n\_samples}) \times 8$$
$$\text{variable} = \text{num\_gpus} \times \text{per\_gpu\_bytes} \times 1.3$$

Key constants:

- `_DEFAULT_GPU_METRICS = 12` (DCGM default set).
- `gpu_sample_interval_s = 1.0` (default in `params.py`).
- `_GROWABLE_ARRAY_OVERHEAD = 1.3` — applied as a single multiplier to the
  total GPU telemetry footprint (same constant as RecordsManager).

Returns zero if no DCGM URLs are configured.

**Scales with:** `num_gpus`, `duration_s / sample_interval_s`,
`num_gpu_metrics`.

### Fixed-overhead services

`SystemController`, `TimingManager`, `APIService`, `ResultsSidecar` (on the
controller pod) and `WorkerGroupManager` (on each worker pod) all use
`_estimate_fixed_service`: a flat `_SERVICE_BASE_MIB[name] +
_PYTHON_SUBPROCESS_BASE_MIB` with `variable_mib = 0`. These do not scale with
workload.

Plus 3 ZMQ proxies at 5 MiB each (`_ZMQ_PROXY_MIB = 5`,
`_NUM_ZMQ_PROXIES = 3`) on the controller pod.

---

## Output schema

`estimate_memory(config, ...)` returns a `ClusterMemoryEstimate`:

```
ClusterMemoryEstimate
├── params: MemoryEstimationParams        # echoed inputs
├── controller: PodEstimate
│   ├── components: list[ComponentEstimate]   # per-service breakdown
│   ├── current_limit_mib: float              # configured K8s limit
│   ├── replicas: int
│   ├── total_steady_state_mib (property)
│   ├── total_peak_mib (property)
│   ├── recommended_request_mib (property)    # steady x 1.2
│   ├── recommended_limit_mib (property)      # peak x 1.3
│   ├── headroom_pct (property)
│   └── at_risk (property)                    # headroom < 15%
├── worker_pod: PodEstimate                   # single-pod, multiplied by replicas
├── operator: PodEstimate                     # fixed 256 MiB
├── warnings: list[str]                       # ordered by severity
└── recommendations: list[str]                # actionable tuning suggestions
```

The `replicas` field on `worker_pod` is the number of worker pods; use
`worker_pod.total_steady_state_mib * worker_pod.replicas` for the cluster-wide
worker footprint.

Every `ComponentEstimate` carries a `formula` string and a `dominant_factor`
string for display purposes — `format_estimate()` in
`formatting.py` renders the full human-readable report, which is what
`aiperf kube profile` prints.

---

## OOM risk warnings

The estimator attaches warnings at two layers.

### Per-component (attached to `ComponentEstimate.warning`)

| Component | Trigger | Message pattern |
|---|---|---|
| RecordsManager | `variable_mib > 500` | `"variable memory is X MiB ... consider reducing request count"` |
| RecordProcessor | `inflight_mib > 50` | `"in-flight records use X MiB ... driven by <mode> ISL=... OSL=... at concurrency N"` |
| RecordProcessor | `tokenizer_mib > 450` | `"tokenizer cache is X MiB ... consider reducing model count"` |

### Per-cluster (appended to `ClusterMemoryEstimate.warnings`)

| Warning | Trigger (constant) | Meaning |
|---|---|---|
| Controller pod at risk | `headroom_pct < 15%` (`_HEADROOM_WARNING_PCT`) | Peak is within 15% of limit; OOM likely. |
| RecordsManager dominates controller | RM > 50% of controller limit (`_RECORDS_MANAGER_WARN_PCT`) | Request count is the single largest driver. |
| Worker pod at risk | `headroom_pct < 15%` | Per-pod peak too close to limit. |
| High request volume | `total_requests > 500_000` | Expect significant metric array storage. |
| Many models per RP | `num_models * 150 > 450 MiB` | Tokenizer cache will dominate each RP. |
| HTTP trace at scale | `export_http_trace and total_requests > 10_000` | Per-chunk trace data accumulates unboundedly. |
| Multi-turn with heavy concurrency | `max_turns > 1 and sessions_per_worker > 100` | Session cache may grow significantly. |

`recommendations` are built by `_build_recommendations(est)` — it prints the
specific `recommended_limit_mib` values to bump to, or confirms that current
limits have adequate headroom.

---

## How to run it

### Programmatic

```python
from aiperf.kubernetes.memory_estimator import estimate_memory, format_estimate
from aiperf.config.config import AIPerfConfig

config = AIPerfConfig.from_yaml("bench.yaml")
est = estimate_memory(
    config,
    total_workers=10,
    workers_per_pod=None,      # None = use config.runtime.workers_per_pod
    connections_per_worker=200,
)
print(format_estimate(est))

if est.controller.at_risk:
    raise SystemExit(
        f"Controller would OOM: need {est.controller.recommended_limit_mib} MiB, "
        f"have {est.controller.current_limit_mib:.0f} MiB"
    )
```

### Via CLI

`aiperf kube profile` derives `MemoryEstimationParams.from_config(...)` from
the rendered config and prints the full report — including per-pod tables,
warnings, and recommendations — before any cluster resources are created.

`aiperf kube generate` uses the same estimate to populate `resources.requests`
(from `recommended_request_mib`) and `resources.limits` (from
`recommended_limit_mib`) in the rendered manifests.

The operator preflight step (`src/aiperf/operator/`) runs the estimator one
more time at admission and rejects jobs whose projected peak exceeds the
namespace's available quota.

---

## Tuning recipes

### "My records-manager OOMs at 500k concurrency"

1. Run `aiperf kube profile` and look at the `RecordsManager` row in the
   Controller Pod table.
2. If `RecordsManager uses N% of controller limit` appears in warnings, the
   growable arrays are the cause. At 500k requests × 25 metrics × 8 bytes × 1.3
   overhead × `ceil_pow2(500_000) = 524_288`, expect roughly 130 MiB just for
   metric arrays — plus base, dataset manager, server metrics, etc.
3. Bump `AIPERF_K8S_RECORDS_MANAGER_MEMORY` (and its CPU sibling) on the
   operator to at least `recommended_limit_mib`. Controller resource keys are
   listed in `CONTROLLER_RESOURCE_KEYS` in `aiperf.kubernetes.environment`.

### "Workers OOM mid-ramp on a large-token workload"

Check the RecordProcessor warning first. If `in-flight records use X MiB` fires
with `ISL+OSL > 10_000`, you have hit the tokenization-queue-depth
amplification — at ISL+OSL=173K the queue reaches 10x `conc_per_rp`. Options:

- Increase `workers_per_pod` to drop concurrency-per-RP.
- Increase `record_processors_per_pod` so each RP processes fewer records.
- Bump the worker pod's memory limit via the estimator's
  `recommended_limit_mib`.

### "Tokenizer cache dominates each RP"

Triggered by `num_models * 150 MiB > 450 MiB`. Either split models across
separate AIPerfJob CRs (one model per benchmark) or accept the higher
worker-pod memory limit — there is no per-process cache deduplication.

### "The estimator disagrees with measured RSS"

Constants are calibrated but static. Re-run
`scripts/calibrate_memory_estimates.py` against a real cluster run to update
the calibration constants; do not edit the formulas ad-hoc.

---

## References

- Public API: `src/aiperf/kubernetes/memory_estimator.py`
- Orchestrator: `src/aiperf/kubernetes/_memory_estimator/estimator.py`
- Per-component formulas: `src/aiperf/kubernetes/_memory_estimator/components.py`
- Calibration constants: `src/aiperf/kubernetes/_memory_estimator/constants.py`
- Result dataclasses: `src/aiperf/kubernetes/_memory_estimator/estimates.py`
- Param extraction: `src/aiperf/kubernetes/_memory_estimator/params.py`
- Formatter: `src/aiperf/kubernetes/_memory_estimator/formatting.py`
