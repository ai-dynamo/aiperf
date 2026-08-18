---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Getting Started on Kubernetes
---

# Getting Started with AIPerf on Kubernetes

> **AI Agents (Claude, Copilot, Cursor, etc.):** For diagnosing failures, see the [AI Agent Debugging Guide](ai-debugging-guide.md).

This guide walks you through benchmarking an NVIDIA Dynamo inference server on Kubernetes using AIPerf. By the end, you will have a cluster running, both operators installed, a benchmark executed against a Dynamo deployment, and your results downloaded.

The same workflow applies to any OpenAI-compatible endpoint (vLLM, TRT-LLM, SGLang) -- just change the endpoint URL and model name.

---


## Preparing a cluster

`aiperf kube setup` installs everything a cluster needs before its first
benchmark. It is idempotent -- each step checks first, so re-running against a
prepared cluster reports what is already there instead of failing.

```bash
# See what is missing without touching the cluster
aiperf kube setup --dry-run

# Install JobSet, the namespaces, and the AIPerf operator
aiperf kube setup

# Prerequisites only, with JobSet pinned
aiperf kube setup --jobset-version v0.5.2 --skip-operator
```

It installs, in order: the **JobSet CRD** (AIPerf runs every benchmark as a
JobSet), the **operator and benchmark namespaces**, and the **AIPerf
operator** Helm chart that owns the AIPerfJob and AIPerfSweep CRDs. Point
`--chart` at an alternate chart directory only when testing a customized
operator chart; normal package installs use the chart bundled in the AIPerf
wheel. A missing chart is fatal, so `setup` cannot report a ready cluster after
skipping the operator. Follow it with `aiperf kube preflight` to confirm the
cluster is ready. The CLI creates the resolved benchmark namespace first, then
installs the chart with `benchmarkNamespace.create=false` and
`benchmarkNamespace.name` set to that namespace. This lets Helm own the
namespace-local benchmark RBAC without trying to adopt a namespace it did not
create; `aiperf kube setup --namespace team-benchmarks` therefore prepares and
authorizes `team-benchmarks`, not the chart's default namespace.

## Cluster Setup

If you already have a Kubernetes cluster with GPU nodes and `kubectl` configured, skip to [Prerequisites](#prerequisites).

### Option A: Use an Existing Cluster

Any Kubernetes v1.27+ cluster with NVIDIA GPUs works. You need:

- `kubectl` configured to talk to the cluster
- GPU nodes with the NVIDIA device plugin installed
- Permissions to install CRDs and create namespaces

Verify your connection:

```bash
kubectl cluster-info
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

If GPUs show up, skip to [Prerequisites](#prerequisites).

### Option B: Create a Local Kind Cluster with GPU Passthrough

[Kind](https://kind.sigs.k8s.io/) runs Kubernetes inside Docker containers on your local machine. With the NVIDIA container runtime configured as Docker's default, a Kind node can see the host's GPUs.

**Host requirements:**
- Docker with the NVIDIA container runtime as the default runtime
- NVIDIA drivers installed on the host
- `kind` CLI installed (`go install sigs.k8s.io/kind@latest` or [releases](https://kind.sigs.k8s.io/docs/user/quick-start/#installation))

#### One-time Docker setup

If you haven't already configured Docker for GPU passthrough:

```bash
# Enable volume-mount-based GPU injection
sudo nvidia-ctk config --in-place \
  --set accept-nvidia-visible-devices-as-volume-mounts=true

# Set nvidia as the default Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# Restart Docker to pick up changes
sudo systemctl restart docker
```

Verify:

```bash
docker info 2>/dev/null | grep "Default Runtime"
# Should show: Default Runtime: nvidia
```

#### Cluster setup

Create the cluster:

```bash
kind create cluster --name aiperf
```

Install the NVIDIA device plugin so GPUs become an allocatable resource, and JobSet, which the AIPerf operator uses to run benchmark pods:

```bash
kubectl --context kind-aiperf apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/deployments/static/nvidia-device-plugin.yml

kubectl --context kind-aiperf apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/latest/download/manifests.yaml
```

Build the AIPerf image locally and load it into the Kind node so pods can pull it without a registry:

```bash
docker build -t aiperf:local .
kind load docker-image aiperf:local --name aiperf
```

Verify GPUs are allocatable:

```bash
kubectl --context kind-aiperf get nodes \
  -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'
# Should show: 1 (or more)
```

#### Teardown

To delete the cluster when you are done:

```bash
kind delete cluster --name aiperf
```

---

## Prerequisites

At this point you should have:

- A Kubernetes cluster with `kubectl` configured
- GPU nodes with the NVIDIA device plugin
- [JobSet](https://github.com/kubernetes-sigs/jobset) installed
- [Helm](https://helm.sh/) v3 installed locally
- AIPerf installed locally (`uv tool install aiperf`, or `uv sync` from a source checkout)
- Access to NGC container registry (`nvcr.io/nvidia/ai-dynamo`)

Run the preflight checker to verify:

```bash
aiperf kube preflight
```

---

## Step 1: Install the Operators

You need two operators: the **Dynamo operator** (manages inference servers) and the **AIPerf operator** (manages benchmarks).

### Install the Dynamo Operator

```bash
# Install Dynamo platform (CRDs are bundled into the platform chart in v1.x)
helm install dynamo-platform \
  oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
  --version 1.1.0 \
  --namespace dynamo-system \
  --create-namespace \
  --set dynamo-operator.webhook.enabled=false \
  --set grove.enabled=false \
  --set kai-scheduler.enabled=false
```

Verify the Dynamo operator is running:

```bash
kubectl get pods -n dynamo-system
```

### Install the AIPerf Operator

```bash
helm install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --create-namespace
```

For Kind clusters using a locally built image, override the image:

```bash
helm install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system \
  --create-namespace \
  --set image.repository=aiperf \
  --set image.tag=local \
  --set image.pullPolicy=Never
```

Verify it is running:

```bash
kubectl get pods -n aiperf-system
```

You should see `2/2` containers ready by default (operator + results-server). With the optional Plotly dashboard enabled (`dashboard.enabled=true`), the count becomes `3/3` (operator + results-server + dashboard).

---

## Step 2: Deploy a Dynamo Inference Server

Create a `DynamoGraphDeployment`. This example deploys Qwen3-0.6B in aggregated mode using the vLLM backend:

```yaml
# dynamo-server.yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-agg
  namespace: dynamo-server
spec:
  services:
    Frontend:
      dynamoNamespace: dynamo-agg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
    VllmWorker:
      dynamoNamespace: dynamo-agg
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        runtimeClassName: nvidia
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: DYN_SYSTEM_PORT
              value: "9090"
```

Apply it:

```bash
kubectl create namespace dynamo-server
kubectl apply -f dynamo-server.yaml
```

Wait for the server to be ready (model loading can take a few minutes):

```bash
# Watch pods until all are Running
kubectl get pods -n dynamo-server -w
```

Verify the endpoint is healthy:

```bash
kubectl run curl-test --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://dynamo-agg-frontend.dynamo-server.svc:8000/v1/models
```

---

## Step 3: Run Your First Benchmark

Now benchmark the Dynamo server. The Dynamo endpoint URL follows the pattern `http://{deployment-name}-frontend.{namespace}.svc:8000/v1`:

```bash
aiperf kube profile \
  --model Qwen/Qwen3-0.6B \
  --url http://dynamo-agg-frontend.dynamo-server.svc:8000/v1 \
  --image nvcr.io/nvidia/aiperf:latest \
  --total-workers 10 \
  --request-count 500 \
  --concurrency 50 \
  --streaming
```

On a Kind cluster with a locally built image, point `--image` at the loaded tag and disable pulling. Substitute the Service URL of whatever OpenAI-compatible endpoint you are testing against:

```bash
aiperf kube profile \
  --model Qwen/Qwen3-0.6B \
  --url http://my-endpoint.default.svc:8000 \
  --image aiperf:local \
  --image-pull-policy Never \
  --total-workers 10 \
  --request-count 200 \
  --concurrency 5 \
  --streaming
```

What happens:

1. AIPerf builds an `AIPerfJob` custom resource from your flags
2. Submits it to the cluster via the AIPerf operator
3. The operator creates a ConfigMap, RBAC, and a JobSet with a controller pod and worker pods
4. Workers send requests to the Dynamo frontend
5. AIPerf attaches to the controller and streams live progress to your terminal

You will see real-time output showing request throughput, latency percentiles, and progress.

Press **Ctrl+C** to detach. The benchmark continues running in the cluster. To cancel it, use `aiperf kube attach` and press **Ctrl+C** there (which sends a cancellation signal), or patch the CR directly:

```bash
kubectl patch aiperfjob <name> -n aiperf-benchmarks --type=merge -p '{"spec":{"cancel":true}}'
```

---

## Step 4: Using a Config File

For repeatable benchmarks, use an AIPerfJob YAML file. Generate a starter template:

```bash
aiperf kube init --output benchmark.yaml
```

Edit it for your Dynamo deployment:

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: dynamo-benchmark
spec:
  benchmark:
    models:
      - "Qwen/Qwen3-0.6B"
    endpoint:
      urls:
        - "http://dynamo-agg-frontend.dynamo-server.svc:8000/v1"
      streaming: true
    datasets:
      - name: main
        type: synthetic
        entries: 1000
        prompts:
          isl:
            mean: 512
            stddev: 0
          osl:
            mean: 128
            stddev: 0
    phases:
      - name: warmup
        kind: warmup
        type: concurrency
        concurrency: 10
        requests: 20
      - name: profiling
        kind: profiling
        type: concurrency
        concurrency: 50
        requests: 500
```

Validate the config before deploying:

```bash
aiperf kube validate benchmark.yaml
```

Run it:

```bash
aiperf kube profile --config benchmark.yaml --image nvcr.io/nvidia/aiperf:latest
```

Or apply it directly with kubectl (the operator picks it up automatically):

```bash
kubectl apply -f benchmark.yaml
```

---

## Step 5: Disaggregated Inference

Dynamo's disaggregated mode separates prefill and decode into different pods for better GPU utilization. To benchmark a disaggregated deployment:

1. Deploy Dynamo in disaggregated mode (separate prefill and decode workers with KV cache transfer):

```yaml
# dynamo-disagg.yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: dynamo-disagg
  namespace: dynamo-server
spec:
  services:
    Frontend:
      dynamoNamespace: dynamo-disagg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: DYN_ROUTER_MODE
              value: "kv"
    VllmPrefillWorker:
      dynamoNamespace: dynamo-disagg
      componentType: worker
      subComponentType: prefill
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        runtimeClassName: nvidia
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--is-prefill-worker"
            - "--connector"
            - "kvbm"
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: DYN_KVBM_CPU_CACHE_GB
              value: "1"
    VllmDecodeWorker:
      dynamoNamespace: dynamo-disagg
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        runtimeClassName: nvidia
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--is-decode-worker"
          env:
            - name: POD_UID
              valueFrom:
                fieldRef:
                  fieldPath: metadata.uid
            - name: DYN_SYSTEM_PORT
              value: "9090"
```

2. Benchmark it -- the endpoint URL changes to match the deployment name:

```bash
aiperf kube profile \
  --model Qwen/Qwen3-0.6B \
  --url http://dynamo-disagg-frontend.dynamo-server.svc:8000/v1 \
  --image nvcr.io/nvidia/aiperf:latest \
  --total-workers 20 \
  --request-count 1000 \
  --concurrency 100 \
  --streaming
```

---

## Step 6: View Results

After the benchmark completes, retrieve your results:

```bash
aiperf kube results
```

This downloads the full results package including:

- `profile_export_aiperf.json` -- Summary metrics (throughput, latency percentiles, TTFT, ITL)
- `inputs.json` -- Dataset that was used
- `server_metrics_export.json` -- Dynamo server metrics (frontend throughput, KV cache stats, component latencies)

To retrieve results from a specific job:

```bash
aiperf kube results dynamo-benchmark
```

Results are stored on the operator's persistent volume by default, so `aiperf kube results` works even after benchmark pods are deleted. To retrieve directly from the benchmark pods instead (tries the controller API first, then falls back to `kubectl cp`):

```bash
aiperf kube results dynamo-benchmark --from-pods
```

### Dynamo Server Metrics

When benchmarking Dynamo, AIPerf discovers and collects Prometheus metrics from pods with the `nvidia.com/metrics-enabled=true` label (or a recognizable inference-server image). By default discovery only searches the **benchmark job's own namespace** — the namespace the chart-provisioned benchmark RBAC can list pods in. If Dynamo runs in a different namespace (e.g. `dynamo-server`), set `server_metrics.discovery.namespace: dynamo-server` in the benchmark config **and** grant pod-read access there by adding that namespace to the chart's `serverMetricsDiscoveryNamespaces` value — a plain string entry binds the benchmark namespaces' `default` ServiceAccount, so benchmark pods running under a custom `podTemplate.serviceAccountName` need the `{namespace, serviceAccounts}` entry form (or a manual RoleBinding); see the [server-metrics guide](../server-metrics/server-metrics.md#kubernetes-auto-discovery) for the full RBAC prerequisite table and how to tune or disable discovery. Discovered metrics include:

- **Frontend metrics** -- `dynamo_frontend_requests`, `dynamo_frontend_time_to_first_token_seconds`, `dynamo_frontend_inter_token_latency_seconds`, `dynamo_frontend_output_tokens`
- **Component metrics** -- Per-worker `dynamo_component_requests`, `dynamo_component_kvstats_gpu_cache_usage_percent`, `dynamo_component_kvstats_gpu_prefix_cache_hit_rate`

These are exported alongside the standard AIPerf metrics in the results package.

---

## Step 7: List and Manage Jobs

See all benchmark jobs across namespaces:

```bash
aiperf kube list
```

```
NAME                 NAMESPACE           PHASE      WORKERS  PROGRESS  THROUGHPUT  LATENCY  AGE
dynamo-benchmark     aiperf-benchmarks   Completed  10       100%      142.3 rps   187ms    5m
disagg-test          aiperf-benchmarks   Running    10       42%       98.1 rps    201ms    2m
```

Use `--wide` to see model and endpoint columns.

Filter by status:

```bash
aiperf kube list --running
aiperf kube list --completed
aiperf kube list --failed
```

Watch jobs with live refresh:

```bash
aiperf kube list --watch
```

---

## Web Dashboard

The AIPerf operator includes a built-in web dashboard for monitoring benchmarks and analyzing results.

Access it by port-forwarding to the operator:

```bash
kubectl port-forward -n aiperf-system deploy/aiperf-operator 8081:8081
```

Then open [http://localhost:8081](http://localhost:8081) in your browser.

The dashboard provides:

- **Dashboard** -- Overview with KPI cards, active jobs, and throughput trends
- **Jobs** -- Sortable table of all benchmark jobs with phase filters
- **Job Detail** -- Live metrics, charts, phase progress, and pod status for a single job
- **Leaderboard** -- Rank benchmark runs by any metric
- **Compare** -- Side-by-side comparison of multiple jobs
- **History** -- Time-series charts showing metrics across runs
- **Sweeps** -- AIPerfSweep listings and per-sweep variation drill-down

Use Ctrl+K to quickly search and navigate between jobs and pages.

---

## Dynamo Deployment Modes

| Mode | Description | Endpoint URL Pattern |
|------|-------------|---------------------|
| `agg` | Aggregated -- single workers handle prefill + decode | `http://dynamo-agg-frontend.dynamo-server.svc:8000/v1` |
| `agg-router` | Aggregated with KV-aware routing | `http://dynamo-agg-router-frontend.dynamo-server.svc:8000/v1` |
| `disagg` | Disaggregated -- separate prefill and decode workers | `http://dynamo-disagg-frontend.dynamo-server.svc:8000/v1` |

### Backends

Dynamo supports three inference backends. Change the worker image and command:

| Backend | Image | Worker Command |
|---------|-------|---------------|
| vLLM | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.0` | `python3 -m dynamo.vllm` |
| TRT-LLM | `nvcr.io/nvidia/ai-dynamo/trtllm-runtime:1.1.0` | `python3 -m dynamo.trtllm` |
| SGLang | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.0` | `python3 -m dynamo.sglang` |

---

## Without the Operator

If you cannot install the AIPerf operator (e.g., limited cluster permissions), AIPerf can deploy benchmarks directly using raw Kubernetes manifests. Use `--no-operator`:

```bash
aiperf kube profile \
  --model Qwen/Qwen3-0.6B \
  --url http://dynamo-agg-frontend.dynamo-server.svc:8000/v1 \
  --image nvcr.io/nvidia/aiperf:latest \
  --no-operator
```

This creates a Namespace, RBAC, ConfigMap, and JobSet directly. You lose operator features (automated monitoring, results storage, conditions) but the benchmark itself works the same way.

To generate the manifests without applying them (useful for GitOps):

```bash
aiperf kube generate --no-operator \
  --model Qwen/Qwen3-0.6B \
  --url http://dynamo-agg-frontend.dynamo-server.svc:8000/v1 \
  --image nvcr.io/nvidia/aiperf:latest \
  > manifests.yaml
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Check cluster readiness | `aiperf kube preflight` |
| Generate config template | `aiperf kube init` |
| Validate a config file | `aiperf kube validate benchmark.yaml` |
| Run a benchmark | `aiperf kube profile --config benchmark.yaml --image <img>` |
| Run without waiting | `aiperf kube profile ... --detach` |
| Preview without deploying | `aiperf kube profile ... --dry-run` |
| Attach to a running job | `aiperf kube attach` |
| Diagnose a job | `aiperf kube debug` |
| List all jobs | `aiperf kube list` |
| Get results | `aiperf kube results` |
| Get logs | `aiperf kube logs` |
| Debug a stuck job | `aiperf kube debug` |

---

## Next Steps

- [Deploy from a Source Checkout](source-checkout-deploy.md) -- Build and push AIPerf, install the operator with Helm, and run on a real cluster
- [Kubernetes Configuration Reference](configuration.md) -- All CRD fields, deployment options, and config patterns
- [Monitoring and Troubleshooting](monitoring.md) -- Watch, debug, and diagnose benchmark issues
- [Production Deployments](production.md) -- CI/CD, Kueue scheduling, secrets, and GitOps workflows

## Related Documentation

- [YAML Config Reference](../tutorials/yaml-config.md) -- General AIPerf YAML configuration
- [Sequence Length Distributions](../tutorials/sequence-distributions.md) -- ISL/OSL distribution configuration
- [Architecture](../architecture.md) -- AIPerf internal architecture
