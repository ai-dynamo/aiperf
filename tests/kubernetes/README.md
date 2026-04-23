# AIPerf Kubernetes E2E Test Suite

Async pytest-based end-to-end test suite for AIPerf Kubernetes deployments. Two test suites: **local** (Kind/Minikube + mock server) and **GPU** (real cluster + vLLM/Dynamo).

## Quick Start

### Local tests (Kind by default)

Local tests use **Kind** by default for faster cluster lifecycle. Set `--k8s-runtime=minikube`
to use Minikube instead. All settings can be configured via **CLI options** (`--k8s-*`) or
**environment variables** (`K8S_TEST_*`). CLI takes precedence over environment.

```bash
# Full run: build images, create cluster, run tests, clean up (uses Kind)
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu

# Use Minikube instead of Kind
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu --k8s-runtime=minikube

# Skip image builds (CLI or env var)
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu --k8s-skip-build
K8S_TEST_SKIP_BUILD=1 uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu

# Keep cluster after tests for debugging
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu --k8s-skip-cleanup

# Reuse existing cluster (fastest for iteration)
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu --k8s-reuse-cluster

# Combine options
uv run pytest tests/kubernetes/ -v --ignore=tests/kubernetes/gpu --k8s-reuse-cluster --k8s-skip-build
```

### GPU tests (Kind by default, with GPU passthrough)

GPU tests default to **Kind** with GPU passthrough (requires `nvidia-smi`,
`nvidia-ctk`, and Docker configured with `nvidia` as the default runtime).
Set `--gpu-runtime=minikube` to use Minikube, or `--gpu-context` to use a
pre-existing external cluster instead.

GPU-specific settings use `--gpu-*` flags with `GPU_TEST_*` env var equivalents.
Run `uv run pytest tests/kubernetes/gpu/ --help` for the full list.

```bash
# Create GPU cluster (Kind by default), deploy vLLM, run benchmarks
./tests/kubernetes/gpu/run_vllm_tests.sh

# Same via pytest directly (with CLI options)
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-stream-logs

# Reuse existing GPU cluster between runs
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-reuse-cluster

# Use an existing vLLM endpoint (skip vLLM deploy)
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-vllm-endpoint http://vllm:8000/v1

# Use a pre-existing external cluster instead of a local one
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-context my-cluster

# Use Minikube instead of Kind
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-runtime=minikube

# Change model and GPU memory utilization
uv run pytest tests/kubernetes/gpu/ -v -m gpu --gpu-model facebook/opt-125m --gpu-mem-util 0.3

# Run only Dynamo tests with single-GPU disaggregated mode
uv run pytest tests/kubernetes/gpu/ -v -s -m dynamo --gpu-dynamo-mode disagg-1gpu --log-cli-level=DEBUG

# Shell script also works with env vars
GPU_TEST_REUSE_CLUSTER=1 GPU_TEST_STREAM_LOGS=1 ./tests/kubernetes/gpu/run_vllm_tests.sh
```

## Prerequisites

- Docker
- Kind (default, see https://kind.sigs.k8s.io/) or Minikube (see https://minikube.sigs.k8s.io/docs/start/)
- kubectl
- uv (Python package manager)
- NVIDIA drivers + nvidia-container-toolkit (GPU tests)

## Test Structure

```
tests/kubernetes/
├── conftest.py                  # Package-scoped fixtures: cluster, images, mock server, operator, helm
├── helpers/
│   ├── benchmark.py             # BenchmarkConfig, BenchmarkDeployer, BenchmarkResult, BenchmarkMetrics
│   ├── cluster.py               # Local cluster management (Kind/Minikube backends)
│   ├── helm.py                  # HelmClient, HelmDeployer, HelmValues
│   ├── images.py                # Docker image building (ImageManager)
│   ├── kubectl.py               # Async kubectl wrapper (KubectlClient)
│   ├── kueue.py                 # Kueue install/config helpers
│   ├── log_streamer.py          # Background pod log streamer
│   ├── operator.py              # AIPerfJob CR management (OperatorDeployer)
│   ├── preflight.py             # Preflight checker
│   └── watchdog.py              # Benchmark watchdog for deadlock detection
├── test_benchmark.py            # Benchmark execution, lifecycle, endpoints
├── test_cli_commands.py         # CLI kube subcommand integration
├── test_completion_signal.py    # Completion signalling tests
├── test_deployment.py           # Cluster setup, pod lifecycle, config variations
├── test_edge_cases.py           # Validation, cleanup, diagnostics
├── test_helm.py                 # Helm chart install/upgrade, job lifecycle
├── test_kube_profile.py         # Profile/configuration tests
├── test_kueue_gang_scheduling.py  # Kueue gang-scheduling tests
├── test_kueue_integration.py    # Kueue admission/queue integration
├── test_metrics.py              # Metrics collection, validation, consistency
├── test_operator.py             # Operator CRD, jobs, conditions, scaling
├── test_recipes.py              # Example recipes / config presets
├── test_scaling.py              # Concurrency, worker scaling, resource config
└── gpu/
    ├── conftest.py              # Shared GPU fixtures: cluster, images, kubectl, settings
    ├── run_vllm_tests.sh        # Shell wrapper for vLLM tests
    ├── run_dynamo_tests.sh      # Shell wrapper for Dynamo tests
    ├── vllm/
    │   ├── conftest.py          # vLLM server deployment, benchmark configs
    │   ├── helpers.py           # VLLMDeployer, VLLMConfig, GPUBenchmarkDeployer
    │   ├── test_vllm_benchmark.py       # GPU benchmark completion, lifecycle, concurrency
    │   ├── test_vllm_endpoint_types.py  # Chat vs completions endpoint types
    │   └── test_vllm_metrics.py         # GPU metrics collection and reasonableness
    ├── trtllm/
    │   ├── conftest.py
    │   ├── helpers.py
    │   ├── test_trtllm_benchmark.py
    │   ├── test_trtllm_endpoint_types.py
    │   └── test_trtllm_metrics.py
    ├── sglang/
    │   ├── conftest.py
    │   ├── helpers.py
    │   ├── test_sglang_benchmark.py
    │   ├── test_sglang_endpoint_types.py
    │   └── test_sglang_metrics.py
    └── dynamo/
        ├── conftest.py          # Dynamo operator install, server deployment
        ├── helpers.py           # DynamoDeployer, DynamoConfig
        └── test_dynamo_benchmark.py     # Benchmark against Dynamo inference graphs
```

## Environment Variables

### Local tests (`--k8s-*` / `K8S_TEST_*`)

All local settings can be set via CLI (`--k8s-*`) or env var (`K8S_TEST_*`). CLI wins.

| CLI option | Env var | Description | Default |
|------------|---------|-------------|---------|
| `--k8s-cluster` | `K8S_TEST_CLUSTER` | Cluster name | `aiperf-<uuid>` (or `aiperf-pytest` with `--k8s-reuse-cluster`) |
| `--k8s-runtime` | `K8S_TEST_RUNTIME` | Cluster runtime: `kind` or `minikube` | `kind` |
| `--k8s-quick` | `K8S_TEST_QUICK` | Reuse cluster, skip build/load/cleanup/preflight | `false` |
| `--k8s-skip-build` | `K8S_TEST_SKIP_BUILD` | Skip image building | `false` |
| `--k8s-skip-cleanup` | `K8S_TEST_SKIP_CLEANUP` | Keep cluster after tests | `false` |
| `--k8s-reuse-cluster` | `K8S_TEST_REUSE_CLUSTER` | Reuse existing cluster | `false` |
| `--k8s-skip-load` | `K8S_TEST_SKIP_LOAD` | Skip loading images into cluster | `false` |
| `--k8s-skip-preflight` | `K8S_TEST_SKIP_PREFLIGHT` | Skip preflight checks | `false` |
| `--k8s-stream-logs` | `K8S_TEST_STREAM_LOGS` | Stream pod logs in real time | `false` |
| `--k8s-aiperf-image` | `K8S_TEST_AIPERF_IMAGE` | AIPerf image name | `aiperf:local` |
| `--k8s-mock-server-image` | `K8S_TEST_MOCK_SERVER_IMAGE` | Mock server image | `aiperf-mock-server:latest` |
| `--k8s-jobset-version` | `K8S_TEST_JOBSET_VERSION` | JobSet controller version | `v0.8.0` |
| `--k8s-kueue-version` | `K8S_TEST_KUEUE_VERSION` | Kueue controller version | (from `dev/versions.py`) |
| `--k8s-benchmark-timeout` | `K8S_TEST_BENCHMARK_TIMEOUT` | Benchmark timeout (seconds) | `300` |

### GPU tests (`--gpu-*` / `GPU_TEST_*`)

GPU tests register their own `--gpu-*` options (not `--k8s-*`). All settings
can be set via CLI or env var; CLI wins.

**Cluster management:**

| CLI option | Env var | Description | Default |
|------------|---------|-------------|---------|
| `--gpu-cluster` | `GPU_TEST_CLUSTER` | Local cluster name | `aiperf-gpu` |
| `--gpu-runtime` | `GPU_TEST_RUNTIME` | Cluster runtime: `kind` or `minikube` | `kind` |
| `--gpu-context` | `GPU_TEST_CONTEXT` | Use external cluster (skip local cluster) | (creates local cluster) |
| `--gpu-kubeconfig` | `GPU_TEST_KUBECONFIG` | Path to kubeconfig | system default |
| `--gpu-quick` | `GPU_TEST_QUICK` | Reuse cluster, skip build/cleanup/preflight | `false` |
| `--gpu-reuse-cluster` | `GPU_TEST_REUSE_CLUSTER` | Reuse existing cluster | `false` |
| `--gpu-skip-build` | `GPU_TEST_SKIP_BUILD` | Skip building aiperf image | `false` |
| `--gpu-skip-cleanup` | `GPU_TEST_SKIP_CLEANUP` | Keep resources/cluster after tests | `false` |
| `--gpu-skip-preflight` | `GPU_TEST_SKIP_PREFLIGHT` | Skip preflight checks | `false` |
| `--gpu-stream-logs` | `GPU_TEST_STREAM_LOGS` | Stream pod logs in real time | `false` |
| `--gpu-aiperf-image` | `GPU_TEST_AIPERF_IMAGE` | AIPerf image | `aiperf:latest` |
| `--gpu-benchmark-timeout` | `GPU_TEST_BENCHMARK_TIMEOUT` | Benchmark timeout (seconds) | `600` |

**vLLM / model:**

| CLI option | Env var | Description | Default |
|------------|---------|-------------|---------|
| `--gpu-vllm-image` | `GPU_TEST_VLLM_IMAGE` | vLLM image | `vllm/vllm-openai:latest` |
| `--gpu-model` | `GPU_TEST_MODEL` | Model name | `Qwen/Qwen3-0.6B` |
| `--gpu-count` | `GPU_TEST_GPU_COUNT` | GPUs per instance | `1` |
| `--gpu-max-model-len` | `GPU_TEST_MAX_MODEL_LEN` | Max context length | `4096` |
| `--gpu-mem-util` | `GPU_TEST_GPU_MEM_UTIL` | GPU memory utilization (0.0-1.0) | `0.5` |
| `--gpu-vllm-endpoint` | `GPU_TEST_VLLM_ENDPOINT` | Skip vLLM deploy, use existing | (deploys vLLM) |
| `--gpu-vllm-deploy-timeout` | `GPU_TEST_VLLM_DEPLOY_TIMEOUT` | vLLM deploy timeout (seconds) | `600` |

**K8s scheduling:**

| CLI option | Env var | Description | Default |
|------------|---------|-------------|---------|
| `--gpu-tolerations` | `GPU_TEST_TOLERATIONS` | JSON array of K8s tolerations | none |
| `--gpu-node-selector` | `GPU_TEST_NODE_SELECTOR` | JSON object of node selectors | none |
| `--gpu-hf-token-secret` | `GPU_TEST_HF_TOKEN_SECRET` | K8s secret with HF token | none |
| `--gpu-image-pull-secret` | `GPU_TEST_IMAGE_PULL_SECRET` | Image pull secret name | none |
| `--gpu-runtime-class` | `GPU_TEST_RUNTIME_CLASS` | RuntimeClass for GPU pods | `nvidia` |

**TRT-LLM / SGLang / Dynamo:**

| CLI option | Env var | Description | Default |
|------------|---------|-------------|---------|
| `--gpu-trtllm-image` | `GPU_TEST_TRTLLM_IMAGE` | TRT-LLM image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7` |
| `--gpu-trtllm-endpoint` | `GPU_TEST_TRTLLM_ENDPOINT` | Skip TRT-LLM deploy, use existing | (deploys TRT-LLM) |
| `--gpu-trtllm-deploy-timeout` | `GPU_TEST_TRTLLM_DEPLOY_TIMEOUT` | TRT-LLM deploy timeout (seconds) | `900` |
| `--gpu-sglang-image` | `GPU_TEST_SGLANG_IMAGE` | SGLang image | `lmsysorg/sglang:latest` |
| `--gpu-sglang-endpoint` | `GPU_TEST_SGLANG_ENDPOINT` | Skip SGLang deploy, use existing | (deploys SGLang) |
| `--gpu-sglang-deploy-timeout` | `GPU_TEST_SGLANG_DEPLOY_TIMEOUT` | SGLang deploy timeout (seconds) | `600` |
| `--gpu-dynamo-image` | `GPU_TEST_DYNAMO_IMAGE` | Dynamo runtime image | backend-specific default |
| `--gpu-dynamo-backend` | `GPU_TEST_DYNAMO_BACKEND` | Dynamo backend: `vllm\|trtllm\|sglang` | `vllm` |
| `--gpu-dynamo-mode` | `GPU_TEST_DYNAMO_MODE` | `agg\|disagg\|disagg-1gpu` | `disagg-1gpu` |
| `--gpu-dynamo-connectors` | `GPU_TEST_DYNAMO_CONNECTORS` | Comma-separated prefill connectors | none |
| `--gpu-dynamo-kvbm-cpu-cache-gb` | `GPU_TEST_DYNAMO_KVBM_CPU_CACHE_GB` | KVBM CPU cache GB for prefill workers | none |
| `--gpu-dynamo-endpoint` | `GPU_TEST_DYNAMO_ENDPOINT` | Skip Dynamo deploy, use existing | (deploys Dynamo) |
| `--gpu-dynamo-deploy-timeout` | `GPU_TEST_DYNAMO_DEPLOY_TIMEOUT` | Dynamo deploy timeout (seconds) | `600` |
| `--gpu-dynamo-version` | `GPU_TEST_DYNAMO_VERSION` | Dynamo Helm chart version | `0.9.0` |
| `--gpu-local-keygen` | `GPU_TEST_LOCAL_KEYGEN` | Create MPI SSH secret locally | `false` |

## Pytest Markers

| Marker | Auto-applied | Description |
|--------|-------------|-------------|
| `k8s` | All `tests/kubernetes/` tests | Requires a Kubernetes cluster |
| `k8s_slow` | No | Long-running stress/scaling tests |
| `gpu` | All `tests/kubernetes/gpu/` tests | Requires GPU cluster |
| `vllm` | All `tests/kubernetes/gpu/vllm/` tests | Requires vLLM server |
| `trtllm` | All `tests/kubernetes/gpu/trtllm/` tests | Requires TRT-LLM server |
| `sglang` | All `tests/kubernetes/gpu/sglang/` tests | Requires SGLang server |
| `dynamo` | All `tests/kubernetes/gpu/dynamo/` tests | Requires Dynamo deployment |

```bash
# Run only local (non-GPU) tests
uv run pytest tests/kubernetes/ -m "k8s and not gpu" -v

# Run GPU tests excluding Dynamo
uv run pytest tests/kubernetes/gpu/ -m "gpu and not dynamo" -v

# Skip slow tests
uv run pytest tests/kubernetes/ -m "not k8s_slow" -v
```

## Fixtures

All I/O fixtures are async, using `pytest_asyncio.fixture` with `scope="package"` and `loop_scope="package"`. Data-only fixtures use regular `@pytest.fixture`.

### Package-scoped (shared across an entire package, e.g. all tests under `tests/kubernetes/` or `tests/kubernetes/gpu/vllm/`)

| Fixture | Suite | Description |
|---------|-------|-------------|
| `local_cluster` | Local | Local cluster lifecycle (Kind or Minikube) |
| `gpu_cluster` | GPU | Minikube cluster with `--gpus=all` (or None for external) |
| `kubectl` | Both | Async `KubectlClient` for the target cluster |
| `built_images` | Local | Docker image builds |
| `loaded_images` | Local | Images loaded into cluster |
| `loaded_aiperf_image` | GPU | Builds and loads aiperf image into GPU cluster |
| `jobset_controller` | Both | JobSet CRD installation |
| `mock_server` | Local | Mock LLM server deployment |
| `k8s_ready` | Local | Aggregates cluster + images + JobSet + mock server |
| `gpu_cluster_ready` | GPU | Aggregates cluster + images + vLLM + JobSet |
| `benchmark_deployer` | Both | `BenchmarkDeployer` / `GPUBenchmarkDeployer` |
| `vllm_server` | GPU | vLLM deployment (or uses `GPU_TEST_VLLM_ENDPOINT`) |
| `vllm_endpoint_url` | GPU | vLLM endpoint URL string |
| `dynamo_server` | GPU | Dynamo deployment (or uses `GPU_TEST_DYNAMO_ENDPOINT`) |
| `dynamo_endpoint_url` | GPU | Dynamo endpoint URL string |
| `operator_deployer` | Local | Operator CRD installation |
| `operator_ready` | Local | Operator fully deployed |
| `helm_deployer` | Local | Helm chart setup |
| `helm_deployed` | Local | Helm chart installed |
| `deployed_small_benchmark_module` | Local | Pre-deployed benchmark result (read-only tests) |
| `deployed_gpu_benchmark_module` | GPU | Pre-deployed GPU benchmark result (read-only tests) |

### Function-scoped (fresh per test)

| Fixture | Suite | Description |
|---------|-------|-------------|
| `benchmark_config` | Local | Default config (5 concurrency, 50 requests) |
| `small_benchmark_config` | Local | Small config (2 concurrency, 10 requests) |
| `large_benchmark_config` | Local | Stress config (10 concurrency, 200 requests) |
| `gpu_benchmark_config` | GPU | GPU config (4 concurrency, 20 requests) |
| `small_gpu_benchmark_config` | GPU | Small GPU config (2 concurrency, 10 requests) |
| `deployed_benchmark` | Local | Deploys and waits for benchmark completion |
| `assert_metrics` | Both | Factory for metric assertions |
| `get_pod_logs` | Both | Factory for pod log retrieval |

## Writing Tests

### Async test with deployer

```python
@pytest.mark.asyncio
async def test_benchmark_creates_namespace(
    benchmark_deployer: BenchmarkDeployer,
    small_benchmark_config: BenchmarkConfig,
    kubectl: KubectlClient,
) -> None:
    """Verify benchmark creates its own namespace."""
    result = await benchmark_deployer.deploy(
        config=small_benchmark_config,
        wait_for_completion=True,
        timeout=300,
    )

    assert result.success
    assert await kubectl.namespace_exists(result.namespace)
```

### Read-only test against module-scoped result

```python
def test_benchmark_completes(deployed_small_benchmark_module: BenchmarkResult) -> None:
    """Verify benchmark completes successfully."""
    assert deployed_small_benchmark_module.success
    assert deployed_small_benchmark_module.metrics is not None
```

### Parameterized test

```python
@pytest.mark.parametrize(
    "concurrency",
    [
        param(1, id="concurrency-1"),
        param(4, id="concurrency-4"),
        param(8, id="concurrency-8"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_concurrency_levels(
    benchmark_deployer: BenchmarkDeployer,
    concurrency: int,
) -> None:
    """Test various concurrency levels."""
    config = BenchmarkConfig(concurrency=concurrency, request_count=20)
    result = await benchmark_deployer.deploy(config, wait_for_completion=True, timeout=300)
    assert result.success
```

## Helpers API

### BenchmarkConfig

```python
config = BenchmarkConfig(
    endpoint_url="http://mock-server:8000/v1",
    endpoint_type="chat",            # or "completions"
    model_name="mock-model",
    concurrency=5,
    request_count=50,
    warmup_request_count=5,
    image="aiperf:local",
    workers=2,
    input_sequence_min=50,
    input_sequence_max=100,
    output_tokens_min=10,
    output_tokens_max=50,
)
```

### BenchmarkResult

```python
result = await deployer.deploy(config, wait_for_completion=True, timeout=300)

result.success              # bool
result.error_message        # str | None
result.duration_seconds     # float
result.namespace            # str
result.jobset_name          # str
result.status               # JobSetStatus
result.pods                 # list[PodStatus]
result.controller_pod       # PodStatus
result.worker_pods          # list[PodStatus]
result.metrics.request_throughput       # float (req/s)
result.metrics.output_token_throughput  # float (tokens/s)
result.metrics.request_count            # int
result.metrics.request_latency_avg      # float (ms)
result.metrics.error_count              # int
```

### KubectlClient

All methods are async:

```python
kubectl = KubectlClient(context="aiperf-pytest")

await kubectl.apply(manifest)
await kubectl.delete("deployment", "my-app", namespace="default")
pods = await kubectl.get_pods("my-namespace")
jobset = await kubectl.get_jobset("my-jobset", "my-namespace")
logs = await kubectl.get_logs("pod-name", container="worker", namespace="ns")
await kubectl.wait_for_rollout("deployment", "my-app", namespace="default")
await kubectl.wait_for_jobset_completion("my-jobset", "my-namespace", timeout=300)
```

## Development Workflow

### Fast iteration (reuse cluster)

```bash
# First run: creates cluster
uv run pytest tests/kubernetes/test_deployment.py -v --k8s-reuse-cluster

# Subsequent runs: reuses cluster + skip builds
uv run pytest tests/kubernetes/test_deployment.py -v --k8s-reuse-cluster --k8s-skip-build

# When done: cleanup manually
kind delete cluster --name aiperf-pytest        # if using Kind (default)
minikube delete -p aiperf-pytest                # if using Minikube
```

### Debugging failed tests

```bash
# Keep cluster after failure
uv run pytest tests/kubernetes/test_benchmark.py -v -x --k8s-skip-cleanup

# Inspect cluster
kubectl get pods -A
kubectl get jobsets -A
kubectl logs -n <namespace> <pod> -c system-controller

# GPU: reuse cluster for fast iteration
GPU_TEST_REUSE_CLUSTER=1 ./tests/kubernetes/gpu/run_vllm_tests.sh

# GPU: keep resources and enable full debug logging
GPU_TEST_SKIP_CLEANUP=1 ./tests/kubernetes/gpu/run_vllm_tests.sh
```
