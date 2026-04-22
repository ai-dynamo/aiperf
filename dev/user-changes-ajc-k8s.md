# What's New for You in `ajc/k8s`

A plain-English guide to everything that's different for the person actually using AIPerf, compared to the current `main` branch. No internals, no code.

---

## TL;DR

AIPerf used to be a single command you ran on one machine to benchmark an LLM endpoint. This branch turns it into a **Kubernetes-native benchmarking service** that you install once on a cluster and drive via CRs, a new `aiperf kube` CLI, a web dashboard, or a REST/WebSocket API. The old single-machine workflow still works — it's now one of several ways to run AIPerf.

---

## 1. Brand-New Ways to Run a Benchmark

### The old way (still works)
```bash
aiperf profile --model my-model --url http://endpoint ...
```
One process, one machine, results in a local folder. Unchanged.

### The new way (Kubernetes-native)
You install AIPerf on your cluster once (via Helm), then submit benchmarks the same way you submit any other Kubernetes workload:

```bash
kubectl apply -f my-benchmark.yaml   # where my-benchmark.yaml is an AIPerfJob
```

Or from the CLI:

```bash
aiperf kube profile my-benchmark.yaml
```

The cluster spins up the pods, runs the benchmark, stores the results, and exposes everything through a web dashboard. You don't need a workstation that's big enough to generate the load.

### What `aiperf kube` gives you

A new subcommand family. Each is a normal `aiperf kube <thing>` command:

| Command | What it does |
|---|---|
| `aiperf kube init` | Scaffold a new benchmark YAML file interactively |
| `aiperf kube preflight` | Check the cluster is healthy and has what the benchmark needs before you submit |
| `aiperf kube validate` | Type-check a benchmark YAML without running it |
| `aiperf kube generate` | Render the raw Kubernetes manifests (for GitOps / review) |
| `aiperf kube profile` | Submit a benchmark and stream progress |
| `aiperf kube list` | Show all benchmarks in the cluster with phase/progress/QPS |
| `aiperf kube watch` | Live-stream one benchmark's state |
| `aiperf kube logs` | Tail logs from any pod in a running benchmark |
| `aiperf kube attach` | Attach an interactive terminal for debugging |
| `aiperf kube dashboard` | Open the web dashboard in your browser |
| `aiperf kube results` | Download finished results locally |
| `aiperf kube debug` | Run a guided troubleshooting walk-through on a failed benchmark |

You no longer need to know `kubectl` to use AIPerf on Kubernetes.

---

## 2. New Web Dashboard

The operator now serves a full web UI (NVIDIA-styled) at the cluster endpoint:

- **Dashboard** — at-a-glance view of active and recent benchmarks.
- **Jobs list** — every benchmark ever run, filterable/sortable.
- **Job detail** — live charts (latency, throughput, concurrency), condition timeline, pod status, error surfaces, downloadable artifacts.
- **Leaderboard** — rank benchmarks by any metric across runs.
- **Compare** — pick two or more runs and see them side-by-side.
- **History** — chronological view across all runs.

Real-time updates are pushed over WebSocket, so numbers tick live without refreshing.

You can also run `aiperf kube dashboard` and it will open your browser to the right place with port-forwarding handled for you.

---

## 3. New Config Format (v2.0)

The biggest YAML-facing change. The old "one giant flat config" is gone. The new format is organized around **phases** — distinct stages of a benchmark (warmup, profiling, cooldown, etc.), each with its own load pattern.

### Old config (abbreviated)
```yaml
model: llama-3-8b
endpoint: { url: http://..., streaming: true }
concurrency: 128
request_count: 10000
```

### New config (abbreviated)
```yaml
models: [llama-3-8b]
endpoint:
  urls: [http://...]
  streaming: true
datasets:
  main:
    type: synthetic
    count: 10000
    prompts: {isl: 512}
phases:
  warmup:
    type: concurrency
    dataset: main
    concurrency: 8
    duration: 30s
  profiling:
    type: concurrency
    dataset: main
    concurrency: 128
    requests: 10000
```

### What's new about it
- **Phases**: Instead of a single benchmark configuration, you define an ordered list of named phases. Warmup is no longer a special flag — it's just a phase.
- **Multiple datasets**: You can define several named datasets and route each phase to a different one.
- **Phase types** (pick one per phase):
  - `concurrency` — classic constant-concurrency load.
  - `constant` — fixed request-rate (requests/second) load.
  - `poisson` — requests arrive on a Poisson process.
  - `gamma` — gamma-distributed inter-arrival times.
  - `fixed_schedule` — replay a timing trace exactly.
  - `user_centric` — simulate N concurrent users with think-time.
- **Duration strings**: Time fields now accept human-friendly values: `"30s"`, `"5m"`, `"2h"`. Plain numbers still work as seconds.
- **Strict schema**: Typos in field names are now caught at validation time instead of silently being ignored.
- **Shorthand still works**: `models: [my-model]` instead of a full object, single-phase dict instead of the long form, etc.

### Sweeps are first-class
You can now ask for a **grid sweep**, a **scenario list**, or a **sequential sweep** directly in the config:

```yaml
sweep:
  type: grid
  over:
    concurrency: [8, 16, 32, 64, 128, 256]
    input_tokens: [128, 512, 2048]
```

The operator expands these into child runs automatically and rolls the results up into the leaderboard.

### Ready-made templates
New `aiperf config` command gives you starter templates out of the box:
- Multimodal vision
- Public dataset (ShareGPT, BurstGPT, Mooncake, etc.)
- Request cancellation
- Scenario workload profiles
- Sweep with statistical distributions
- Trace replay
- Warmup + profiling

Use them with:
```bash
aiperf config template trace_replay > my-config.yaml
```

### Validation tools
- `aiperf config validate my-config.yaml` — check a file before submitting.
- A full JSON Schema is now published, so your editor can auto-complete and red-underline mistakes as you type.

---

## 4. Convergence / Adaptive Stopping

New feature for multi-run benchmarks. Instead of picking a fixed number of runs and hoping it's enough:

```yaml
num_profile_runs: 20
convergence_metric: p99_latency
convergence_tolerance: 0.02   # stop when results stabilize to within 2%
```

AIPerf will run additional trials until the chosen metric converges within tolerance (or hits `num_profile_runs`), then stop automatically. Useful for runs where you don't want to overpay for unnecessary iterations.

---

## 5. New HTTP/2 Transport

Previously only HTTP/1.1 was supported. Now you can opt into HTTP/2:

```yaml
endpoint:
  transport: http2
```

Benefits:
- **Connection multiplexing**: ~100 concurrent streams per TCP connection, so you use far fewer sockets.
- **Automatic fallback**: If the endpoint doesn't speak HTTP/2, transparently falls back to HTTP/1.1.
- **Sticky sessions** (opt-in): Pin a simulated user to a single backend connection through load balancers, so conversational benchmarks don't get round-robined between backends.

The old HTTP/1.1 transport remains the default and is unchanged.

---

## 6. Per-Record CSV Export

In addition to JSONL per-record export, there's now a CSV record-export format with flat columns, which opens directly in Excel/pandas/Sheets without parsing:

```yaml
output:
  export_level: records
  record_export_format: csv   # or jsonl
```

Related: a new `--export-per-chunk-data` flag lets you export raw per-chunk streaming data (SSE chunk timings) for deep-dive analysis.

---

## 7. Secrets Handling

API keys and authentication tokens in your config are now **redacted** from:
- Console output
- Log files
- The config echo in CR status
- Any artifact that leaves the cluster

You can put `api_key: "sk-..."` in your config without worrying about it leaking into a log aggregator or a ticket attachment.

---

## 8. The `AIPerfJob` Custom Resource

If you live in Kubernetes, AIPerf benchmarks are now a first-class resource:

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-benchmark
spec:
  benchmark:
    models: [llama-3-8b]
    endpoint: { urls: [http://my-endpoint.svc:8000] }
    phases: {...}
  image: nvcr.io/...
  scheduling: {...}
```

`kubectl get aiperfjobs` (or the short name `kubectl get apj`) now shows:

```
NAME           PHASE      STAGE       PROGRESS       QPS     AGE
my-benchmark   Running    Profiling   4200/10000     850     2m
```

### Lifecycle features
- **Cancellation**: `kubectl delete aiperfjob my-benchmark` cooperatively cancels a running benchmark — it finishes in-flight requests, exports what it has, and cleans up.
- **Timeouts**: Per-job timeout enforced by the operator.
- **TTL**: Old results are auto-cleaned after a configurable number of days.
- **Pod restart detection**: If a worker pod crash-loops, the operator flags the job as failed instead of hanging.
- **Debugging failed pods**: Opt-in `keepFailedPods: true` preserves failed pod attempts so you can `kubectl logs` them post-mortem.

### Status conditions
The CR status carries structured conditions that CI/CD and GitOps tools can read directly:
- `ConfigValid` — did validation pass?
- `EndpointReachable` — can we reach the target?
- `ResourcesCreated` — did pods spin up?
- `PreflightHasWarnings` — non-fatal issues the user should see.
- `Completed` / `Failed` — terminal state.

---

## 9. Installation

### Before
```bash
pip install aiperf
aiperf profile ...
```

### Now (local)
Still works. `uv add aiperf` or equivalent, run `aiperf` locally.

### Now (cluster)
```bash
helm install aiperf-operator oci://.../aiperf-operator \
  --set image.repository=... --set image.tag=...
```

The Helm chart installs:
- The AIPerf operator (a long-running deployment).
- The `AIPerfJob` CRD.
- RBAC (cluster role + bindings).
- A pod disruption budget, PVC for results, service, service account.
- Two tiny test hooks (`helm test`) that verify the install worked.

One install; all benchmarks thereafter are just CRs.

---

## 10. Progress Monitoring & APIs

Previously: logs scrolling in a terminal.

Now: multiple ways to observe a running benchmark.

- **Web dashboard**: live charts, already described.
- **REST API**: `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/metrics`, `GET /api/results/{id}`, `GET /api/server-metrics`, `GET /api/dataset/...`, etc.
- **Prometheus endpoint**: Scrape-ready metrics endpoint so your existing Grafana dashboards can pull AIPerf metrics like any other service.
- **WebSocket stream**: `/ws` pushes every internal event in real time — what the dashboard uses, but also available for your own tooling.
- **`aiperf kube watch`**: live terminal view.
- **`kubectl get apj -w`**: classic Kubernetes way.

---

## 11. Environment Variable Tuning

Several new `AIPERF_*` environment variables you can set on the operator (via Helm `values.yaml`) or on workers (via CR `env`) for cluster-specific tuning. Not exhaustive, but representative:

- `AIPERF_DATASET_DOWNLOAD_MAX_RETRIES` — tolerance for flaky object storage.
- `AIPERF_DATASET_DOWNLOAD_RETRY_DELAY` — retry backoff for dataset downloads.
- `AIPERF_DEVELOPER_MEMORY_PROFILE_ENABLED` — live memory-leak diagnostics.
- `AIPERF_RECORD_INGEST_BATCH_SIZE` / `AIPERF_RECORD_INGEST_BATCH_FLUSH_INTERVAL` — tune record ingestion throughput.
- `AIPERF_SERVICE_POD_FAILURE_ABORT_THRESHOLD_PERCENT` — how many pod failures before the operator gives up on a job.
- `AIPERF_SERVICE_HEARTBEAT_MISSED_THRESHOLD` — controller tolerance for missed worker heartbeats.
- `AIPERF_JOB_TIMEOUT_SECONDS` — operator-level safety net on runaway benchmarks.
- `AIPERF_RESULTS_TTL_DAYS` — how long results hang around before cleanup.
- `AIPERF_POD_RESTART_THRESHOLD` — crash-loop tolerance.

Everything tunable lives in a single file you can introspect with `aiperf --show-env` (auto-generated docs: `docs/environment-variables.md`).

---

## 12. Kubernetes-Specific Features

- **JobSet under the hood**: Benchmark workloads are expressed as JobSets, so they work with Kueue (GPU quota, preemption), node selectors, taints/tolerations, and topology-aware scheduling out of the box.
- **Scheduling knobs in the CR**: `podTemplate`, `scheduling`, `resourceMode` (`guaranteed` / `burstable` / `none`), `imagePullPolicy` — all surfaced at the top level of the CR.
- **Namespace isolation**: Operator runs cluster-wide but can enforce per-namespace RBAC.
- **Dataset download once-per-pod**: In K8s mode, the dataset is downloaded once per pod and shared by all worker containers in that pod — massively reduces network egress when you run many workers.

---

## 13. Improved Reliability (user-visible)

- **Pod crashes don't kill the benchmark**: If a worker pod dies, Kubernetes restarts it and the benchmark recovers (credits are reconciled). Previously this would hang or corrupt the run.
- **Faster shutdowns**: Cancelling a benchmark is bounded-time — no more "stuck in terminating" pods.
- **Deterministic completion**: The operator makes an explicit "completion claim" on the CR, so you get exactly one terminal state (no flicker between Complete and Running).
- **Higher concurrency ceilings**: The internal metrics pipeline was overhauled; benchmarks at 100k+ concurrency are now practical where previously they'd starve the controller.

---

## 14. Bigger Dataset / Benchmark Coverage

New dataset + loader options exposed through config:

- **Public datasets**: ShareGPT, BurstGPT, HellaSwag, MMLU, GPQA-Diamond, SpeedBench, MathBench, MMVU, HuggingFace instruction-response, HuggingFace conversation.
- **Mooncake trace replay**: Drive load from a recorded trace file with exact timing.
- **Composed datasets**: Compose multiple datasets into a single run.
- **Random pool**: Sample randomly from a pool of prompts for better variance.
- **Multimodal**: Vision and audio inputs are supported end-to-end.
- **HuggingFace subset filter**: `--hf-subset` CLI flag to pick a subset of an HF dataset.
- **New endpoint**: OpenAI `/v1/responses` endpoint is supported.
- **Built-in tiktoken tokenizer**: No longer needs an external HF tokenizer for OpenAI-style endpoints.

---

## 15. Analysis / Scripts

New analysis scripts shipped under `scripts/` and `tools/`:
- `analyze_profile_export.py` / `analyze_profile_export_legacy.py` — post-hoc analysis of exported runs.
- `render_token_throughput_html.py` — standalone HTML report generator.
- Multiple mock-server + kind-cluster configs for local testing (in `dev/deploy/`).

---

## 16. Things That Went Away (or Changed)

- The old `user_config.yaml` / `service_config.json` split is gone. There's **one** YAML file now: your `AIPerfConfig`.
- The CLI flag `--export-formats` is respected differently — record export format is now `record_export_format: csv|jsonl` in config.
- Some environment-variable names were reorganized (all `AIPERF_*`, but subsystem prefixes tightened).
- The one-pod-per-service Kubernetes topology is replaced with a shared-pod-with-sibling-containers topology — you won't see this unless you're inspecting pods, but it means **far fewer pods for the same load**.

---

## 17. Scale Tested

This branch has been run against clusters at **1M concurrent user simulation** on DGX-class hardware. Prior iterations on `main` capped out around 100k before the controller became a bottleneck. Concrete throughput numbers depend on your endpoint, but the benchmarking tool itself is no longer the bottleneck at typical data-center scale.

---

## Quick Migration Guide

| If you used to do… | Now do… |
|---|---|
| `aiperf profile --model X --url Y --concurrency 128` | Still works. Or write an AIPerfJob CR. |
| Edit `user_config.yaml` | Edit your new v2.0 AIPerfConfig YAML (use `aiperf config template` to start) |
| Grep logs in the terminal | Open the web dashboard, or `aiperf kube watch` |
| Shell script a sweep | Put the sweep in config under `sweep:` — one submission, one leaderboard |
| Install AIPerf on a beefy VM | `helm install` once; submit AIPerfJob CRs from anywhere |
| Manually tokenize | Tokenizer is built-in (tiktoken for OpenAI-compatible) |
| Download + parse HuggingFace yourself | `datasets.X.type: public, source: sharegpt` etc. |
| Figure out concurrency ceiling by trial | `convergence_metric: p99_latency` — runs until stable |

---

## Where to Look Next

- `docs/kubernetes/getting-started.md` — first-time setup walkthrough.
- `docs/kubernetes/configuration.md` — full config reference.
- `docs/kubernetes/monitoring.md` — dashboards, Prometheus, alerts.
- `docs/kubernetes/production.md` — hardening for production use.
- `docs/kubernetes/ai-deployment-guide.md` / `ai-debugging-guide.md` — troubleshooting.
- `docs/tutorials/distributions.md`, `sweeps.md`, `template-endpoint.md` — worked examples.
- `aiperf config template --list` — starter configs for every major use case.
- `aiperf kube --help` — full Kubernetes CLI reference.
