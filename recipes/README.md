# AIPerf Recipes

This directory holds ready-to-run `AIPerfJob` manifests for common model / engine / topology combinations. Each recipe is a self-contained Kubernetes Custom Resource that the AIPerf operator reconciles into a full benchmark job (controller, workers, dataset/timing managers, and results server).

Recipes are meant as starting points: pick the closest match, `kubectl apply` it to confirm end-to-end wiring against your inference deployment, then tune the `profiling`, `warmup`, `datasets`, and `slos` sections for your workload.

## Directory layout

```
recipes/<model>/<engine>/<topology>/aiperfjob.yaml
```

- `<model>` — the target model family (e.g. `llama-3-70b`, `qwen3-32b-fp8`, `deepseek-r1`).
- `<engine>` — the inference engine serving the model (`vllm` or `trtllm`).
- `<topology>` — the deployment shape (`agg`, `disagg`, `disagg-kv-router`, `disagg-multi-node`, `wide_ep/gb200`, `agg-embedding-cache`, ...).

## Recipe index

| Model | Engine | Topology | Path |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-R1 | TRT-LLM | disagg wide-EP (GB200) | [`deepseek-r1/trtllm/disagg/wide_ep/gb200/aiperfjob.yaml`](deepseek-r1/trtllm/disagg/wide_ep/gb200/aiperfjob.yaml) |
| nvidia/DeepSeek-V3.2-NVFP4 | TRT-LLM | agg (round-robin) | [`deepseek-v32-fp4/trtllm/agg-round-robin/aiperfjob.yaml`](deepseek-v32-fp4/trtllm/agg-round-robin/aiperfjob.yaml) |
| nvidia/DeepSeek-V3.2-NVFP4 | TRT-LLM | disagg (KV-router) | [`deepseek-v32-fp4/trtllm/disagg-kv-router/aiperfjob.yaml`](deepseek-v32-fp4/trtllm/disagg-kv-router/aiperfjob.yaml) |
| openai/gpt-oss-120b | TRT-LLM | agg | [`gpt-oss-120b/trtllm/agg/aiperfjob.yaml`](gpt-oss-120b/trtllm/agg/aiperfjob.yaml) |
| openai/gpt-oss-120b | TRT-LLM | disagg | [`gpt-oss-120b/trtllm/disagg/aiperfjob.yaml`](gpt-oss-120b/trtllm/disagg/aiperfjob.yaml) |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | vLLM | agg | [`llama-3-70b/vllm/agg/aiperfjob.yaml`](llama-3-70b/vllm/agg/aiperfjob.yaml) |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | vLLM | disagg (multi-node) | [`llama-3-70b/vllm/disagg-multi-node/aiperfjob.yaml`](llama-3-70b/vllm/disagg-multi-node/aiperfjob.yaml) |
| RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic | vLLM | disagg (single-node) | [`llama-3-70b/vllm/disagg-single-node/aiperfjob.yaml`](llama-3-70b/vllm/disagg-single-node/aiperfjob.yaml) |
| Qwen/Qwen3-235B-A22B-FP8 | TRT-LLM | agg | [`qwen3-235b-a22b-fp8/trtllm/agg/aiperfjob.yaml`](qwen3-235b-a22b-fp8/trtllm/agg/aiperfjob.yaml) |
| Qwen/Qwen3-235B-A22B-FP8 | TRT-LLM | disagg | [`qwen3-235b-a22b-fp8/trtllm/disagg/aiperfjob.yaml`](qwen3-235b-a22b-fp8/trtllm/disagg/aiperfjob.yaml) |
| Qwen/Qwen3-32B-FP8 | TRT-LLM | agg | [`qwen3-32b-fp8/trtllm/agg/aiperfjob.yaml`](qwen3-32b-fp8/trtllm/agg/aiperfjob.yaml) |
| Qwen/Qwen3-32B-FP8 | TRT-LLM | disagg | [`qwen3-32b-fp8/trtllm/disagg/aiperfjob.yaml`](qwen3-32b-fp8/trtllm/disagg/aiperfjob.yaml) |
| Qwen/Qwen3-32B | vLLM | agg (round-robin) | [`qwen3-32b/vllm/agg-round-robin/aiperfjob.yaml`](qwen3-32b/vllm/agg-round-robin/aiperfjob.yaml) |
| Qwen/Qwen3-32B | vLLM | disagg (KV-router) | [`qwen3-32b/vllm/disagg-kv-router/aiperfjob.yaml`](qwen3-32b/vllm/disagg-kv-router/aiperfjob.yaml) |
| Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 | vLLM | agg (embedding cache) | [`qwen3-vl-30b/vllm/agg-embedding-cache/aiperfjob.yaml`](qwen3-vl-30b/vllm/agg-embedding-cache/aiperfjob.yaml) |

## How to run a recipe

The endpoint `urls` in each manifest point at the in-cluster Service name of the corresponding inference deployment (e.g. `http://llama3-70b-agg-frontend:8000`). Bring up that deployment first, then apply the recipe in the same namespace:

```bash
# Direct kubectl apply:
kubectl apply -f recipes/llama-3-70b/vllm/agg/aiperfjob.yaml

# Or via the aiperf CLI (adds defaults, validation, and streaming logs):
aiperf kube profile -f recipes/llama-3-70b/vllm/agg/aiperfjob.yaml
```

Tail progress and pull results:

```bash
kubectl get aiperfjobs
aiperf kube logs <job-name>
aiperf kube results <job-name>
```

See [`docs/kubernetes/getting-started.md`](../docs/kubernetes/getting-started.md) for the full workflow and [`docs/kubernetes/configuration.md`](../docs/kubernetes/configuration.md) for the complete `AIPerfJob` schema.

## Version pinning

These recipes do not set a `spec.image` override, so they run against the `aiperf-operator` Deployment's configured image — currently a `k8s-arm64-*` tag on `nvcr.io/nvidian/dynamo-dev/aiperf`. That tag is pinned in `deploy/helm/aiperf-operator/values.yaml`. To benchmark a different AIPerf build without redeploying the operator, add `spec.image: <tag>` to the manifest before applying.

Some recipes reference model weights or mooncake traces under `/model-cache` or `/perf-cache`; those paths assume the `model-cache` / `perf-cache` PVCs described in [`docs/kubernetes/configuration.md`](../docs/kubernetes/configuration.md).
