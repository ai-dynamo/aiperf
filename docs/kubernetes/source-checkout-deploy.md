---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Source Checkout Deployment
---

# Deploy from a Source Checkout to a Real Kubernetes Cluster

This guide starts from an AIPerf source checkout and deploys that code to a real Kubernetes cluster. It builds a container image, pushes it to a registry, installs or upgrades the AIPerf operator with Helm, then runs a benchmark against a real inference endpoint.

This guide does **not** use the mock server path. Use it when you have a real OpenAI-compatible endpoint already running, or when you want to deploy a real Dynamo/vLLM endpoint first.

## Prerequisites

You need:

- An AIPerf source checkout.
- `kubectl` configured for the target cluster.
- Helm v3.
- Docker or another container builder that can build and push OCI images.
- Registry credentials for the image repository you will use.
- Permission to install CRDs, create the operator namespace, create benchmark namespaces, and create JobSet workloads.
- JobSet installed on the cluster. If it is not installed, install it before the AIPerf operator.
- A real OpenAI-compatible inference endpoint reachable from benchmark pods, or permission to deploy one.

Check the cluster first:

```bash
kubectl cluster-info
kubectl get nodes
kubectl api-resources | grep -i jobset
```

For GPU benchmarks, verify GPU resources are allocatable:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

## 1. Set up the checkout

Install the local development environment from the checkout:

```bash
make first-time-setup
```

Use the local CLI through `uv run` until the package is installed somewhere else:

```bash
uv run aiperf --help
uv run aiperf kube --help
```

Pick a tag that identifies the exact checkout you are deploying:

```bash
export AIPERF_TAG="$(git rev-parse --short HEAD)"
```

## 2. Build and push the AIPerf image

The benchmark image contains the native `aiperf` binary. The independently packaged operator has its own image; configure it through the operator chart separately from the submitted benchmark image.

### NGC-style registry

Use this shape for an NGC organization or private NVIDIA registry namespace:

```bash
export AIPERF_IMAGE="nvcr.io/<org>/aiperf:${AIPERF_TAG}"

docker build -t "${AIPERF_IMAGE}" .
docker push "${AIPERF_IMAGE}"
```

### GitHub Container Registry

Use this shape for GHCR:

```bash
export AIPERF_IMAGE="ghcr.io/<org>/aiperf:${AIPERF_TAG}"

docker build -t "${AIPERF_IMAGE}" .
docker push "${AIPERF_IMAGE}"
```

If your cluster nodes use a different architecture than the build host, build for the cluster platform:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t "${AIPERF_IMAGE}" \
  --push \
  .
```

## 3. Create an image pull secret if the registry is private

Skip this section if every node can pull the image without a secret.

Create the operator namespace first:

```bash
kubectl create namespace aiperf-system --dry-run=client -o yaml | kubectl apply -f -
```

For NGC-style registries:

```bash
kubectl create secret docker-registry aiperf-registry \
  --namespace aiperf-system \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}"
```

For GHCR:

```bash
kubectl create secret docker-registry aiperf-registry \
  --namespace aiperf-system \
  --docker-server=ghcr.io \
  --docker-username="${GITHUB_USER}" \
  --docker-password="${GITHUB_TOKEN}"
```

The Helm install below references this secret for the operator. Benchmark jobs also need pull access in their namespace. The chart creates the default benchmark namespace `aiperf-benchmarks`, but Kubernetes secrets are namespace-scoped, so create the same pull secret there too if you use the default namespace:

```bash
kubectl create namespace aiperf-benchmarks --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret docker-registry aiperf-registry \
  --namespace aiperf-benchmarks \
  --docker-server=<registry-host> \
  --docker-username=<username> \
  --docker-password=<token>
```

## 4. Install or upgrade the AIPerf operator

The operator is a separate Python distribution and image. Select its image
independently from the native benchmark image built above:

```bash
export AIPERF_OPERATOR_IMAGE="ghcr.io/<org>/aiperf-k8s-operator:<version>"
```

Install the operator:

```bash
helm upgrade --install aiperf-operator deploy/aiperf-k8s-operator/helm/aiperf-k8s-operator \
  --namespace aiperf-system \
  --create-namespace \
  --set image.repository="${AIPERF_OPERATOR_IMAGE%:*}" \
  --set image.tag="${AIPERF_OPERATOR_IMAGE##*:}" \
  --set image.pullPolicy=IfNotPresent
```

If the operator image is private, configure its pull secret separately. The
native `aiperf kube` submitter carries the benchmark image digest in each
submitted envelope; the operator chart never supplies a default benchmark
image or makes benchmark pods inherit its own image.

Wait for the operator and results server:

```bash
kubectl rollout status deploy/aiperf-operator -n aiperf-system --timeout=180s
kubectl get pods -n aiperf-system
```

Run Helm tests if the cluster can pull the chart's test image:

```bash
helm test aiperf-operator -n aiperf-system
```

## 5. Run against an existing real endpoint

Use this path when your inference server is already deployed in the cluster or reachable from the cluster network.

Set the endpoint and model:

```bash
export MODEL="Qwen/Qwen3-0.6B"
export ENDPOINT_URL="http://vllm.default.svc.cluster.local:8000/v1"
```

Run cluster-side preflight checks:

```bash
uv run aiperf kube preflight \
  --image "${AIPERF_IMAGE}" \
  --endpoint-url "${ENDPOINT_URL}" \
  --workers 8
```

Submit a benchmark:

```bash
uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --workers-max 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

For private benchmark images, pass the pull secret name when submitting jobs:

```bash
uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --image-pull-secrets aiperf-registry \
  --workers-max 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

For repeatable runs, put the benchmark configuration in YAML and pass `--config benchmark.yaml`; see [End-to-End Workflow](workflow.md) for the `init` -> `validate` -> `preflight` -> `profile` sequence.

## 6. Optional: deploy a real Dynamo/vLLM endpoint first

Skip this section if you already have a real endpoint.

Install the Dynamo platform chart if your cluster does not already have it:

```bash
helm upgrade --install dynamo-platform \
  oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
  --version 1.1.0 \
  --namespace dynamo-system \
  --create-namespace \
  --set dynamo-operator.webhook.enabled=false \
  --set grove.enabled=false \
  --set kai-scheduler.enabled=false
```

Deploy an aggregated Dynamo vLLM server by applying a `DynamoGraphDeployment` manifest such as the aggregated vLLM example in [Getting Started on Kubernetes](getting-started.md#step-2-deploy-a-dynamo-inference-server):

```bash
kubectl create namespace dynamo-server --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f dynamo-server.yaml
```

Wait for the frontend and worker pods to become ready:

```bash
kubectl get pods -n dynamo-server -w
```

Use the Dynamo service URL as the benchmark endpoint:

```bash
export ENDPOINT_URL="http://dynamo-agg-frontend.dynamo-server.svc:8000/v1"

uv run aiperf kube preflight \
  --image "${AIPERF_IMAGE}" \
  --endpoint-url "${ENDPOINT_URL}" \
  --workers 8

uv run aiperf kube profile \
  --model "${MODEL}" \
  --url "${ENDPOINT_URL}" \
  --image "${AIPERF_IMAGE}" \
  --workers-max 8 \
  --request-count 1000 \
  --concurrency 50 \
  --streaming
```

The `dev/kube.py deploy-dynamo` helper is intentionally scoped to AIPerf's local Kind/Minikube workflow and should not be used as the generic real-cluster path.

## 7. Monitor and retrieve results

Watch progress:

```bash
uv run aiperf kube watch
```

Reattach to a detached run:

```bash
uv run aiperf kube attach
```

Download results:

```bash
uv run aiperf kube results --output ./aiperf-results
```

Port-forward the results server and dashboard:

```bash
kubectl port-forward -n aiperf-system svc/aiperf-operator 8081:8081
```

Then open `http://localhost:8081`.

## 8. Upgrade after source changes

After changing source code, repeat the build and push with a new immutable tag:

```bash
export AIPERF_TAG="$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
export AIPERF_IMAGE="nvcr.io/<org>/aiperf:${AIPERF_TAG}"

docker build -t "${AIPERF_IMAGE}" .
docker push "${AIPERF_IMAGE}"

export AIPERF_IMAGE_REPOSITORY="${AIPERF_IMAGE%:*}"
export AIPERF_IMAGE_TAG="${AIPERF_IMAGE##*:}"

helm upgrade aiperf-operator deploy/aiperf-k8s-operator/helm/aiperf-k8s-operator \
  --namespace aiperf-system \
  --reuse-values \
  --set image.repository="${AIPERF_IMAGE_REPOSITORY}" \
  --set image.tag="${AIPERF_IMAGE_TAG}"
```

Wait for the new operator pod before submitting new jobs:

```bash
kubectl rollout status deploy/aiperf-operator -n aiperf-system --timeout=180s
```

## Troubleshooting

### ImagePullBackOff

Check the failing pod and events:

```bash
kubectl describe pod -n aiperf-system -l app.kubernetes.io/name=aiperf-operator
uv run aiperf kube debug
```

Common fixes:

- Confirm the image was pushed with the exact tag used by Helm or `aiperf kube profile`.
- Create the pull secret in both `aiperf-system` and the benchmark namespace.
- Pass `--image-pull-secrets aiperf-registry` to `aiperf kube profile` for private benchmark images.
- Use `--set image.pullPolicy=Always` while testing mutable tags; prefer immutable tags for normal use.

### Operator is running but jobs use an old image

The operator image and default benchmark image come from the Helm release. Check the rendered CRD default:

```bash
kubectl get crd aiperfjobs.aiperf.nvidia.com \
  -o jsonpath='{.spec.versions[?(@.name=="v1alpha1")].schema.openAPIV3Schema.properties.spec.properties.image.default}{"\n"}'
```

If you set `defaults.image`, it overrides the benchmark image independently of `image.repository` and `image.tag`.

### Endpoint check fails

Verify the endpoint from inside the cluster:

```bash
kubectl run curl-check \
  --rm -it \
  --restart=Never \
  --image=curlimages/curl:latest \
  -- curl -sf "${ENDPOINT_URL}/models"
```

If the server is still starting and you intentionally want the benchmark to wait until worker runtime, set `skipEndpointCheck: true` in the AIPerfJob YAML. Do not use this to hide a wrong service name or namespace.

### RBAC or namespace errors

The Helm chart creates benchmark RBAC for `benchmarkNamespace.name` and any namespaces listed in `benchmarkRbacNamespaces`. If you run jobs in a different namespace, either install the chart with that namespace configured or add the namespace to `benchmarkRbacNamespaces`:

```bash
helm upgrade aiperf-operator deploy/aiperf-k8s-operator/helm/aiperf-k8s-operator \
  --namespace aiperf-system \
  --reuse-values \
  --set 'benchmarkRbacNamespaces[0]=team-a-benchmarks'
```

## Related Documentation

- [Getting Started on Kubernetes](getting-started.md) -- First benchmark walkthrough and Dynamo manifest examples.
- [End-to-End Workflow](workflow.md) -- Full `init` -> `validate` -> `preflight` -> `profile` -> `results` lifecycle.
- [Production Deployments](production.md) -- CI/CD, Kueue, private registries, multi-tenancy, and operations patterns.
- [Kubernetes Configuration Reference](configuration.md) -- CRD fields, Helm values, and AIPerfJob configuration.
- [Monitoring and Troubleshooting](monitoring.md) -- Watch, debug, logs, and common failure modes.
