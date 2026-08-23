---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Source Checkout Deployment
---

# Deploy a Native Kubernetes Envelope from a Source Checkout

This guide covers the Kubernetes interface that the native `aiperf` binary
actually ships: validate one authored `native-k8s/v1` envelope, create its
role-specific bootstrap Secrets, and submit one `AIPerfJob`. The native command
does not translate profile flags or Config v2 into a Kubernetes envelope;
`aiperf kube init` and `generate` currently refuse. Prepare the envelope,
image-capability document, and opaque role bootstrap files before following the
submission steps below.

## Prerequisites

You need:

- an AIPerf source checkout and its development environment;
- `kubectl` and Helm v3 configured for the target cluster;
- permission to install CRDs and cluster-scoped operator RBAC;
- a default StorageClass, or an explicit StorageClass for the results PVC;
- a registry reference that every benchmark node can pull; and
- one valid native security bootstrap file for the controller, results
  sidecar, and every cell named by the envelope.

The submitted benchmark image must be available by immutable digest reference.
The current envelope has no image-pull-secret field, so a private benchmark
registry must be available through node-level registry credentials or a
credential provider.

Set up and inspect the checkout:

```bash
make first-time-setup
uv run aiperf kube --help
kubectl cluster-info
kubectl get storageclass
```

## 1. Build and publish the two images

The benchmark image and Python operator image are separate artifacts. Build and
push the benchmark image from the repository root, then record the digest
reference returned by the registry:

```bash
export BENCHMARK_TAG="$(git rev-parse --short HEAD)"
export BENCHMARK_REPOSITORY="ghcr.io/<org>/aiperf"

docker build -t "${BENCHMARK_REPOSITORY}:${BENCHMARK_TAG}" .
docker push "${BENCHMARK_REPOSITORY}:${BENCHMARK_TAG}"

export BENCHMARK_DIGEST="sha256:<64-lowercase-hex-digest>"
export BENCHMARK_IMAGE_REFERENCE="${BENCHMARK_REPOSITORY}@${BENCHMARK_DIGEST}"
```

`imageReference` in the envelope must be a lowercase
`registry/repository@sha256:<64hex>` reference. Its digest suffix must equal the
separate `imageDigest` value exactly. Tags and bare `sha256:...` values are
rejected, and the operator projects the accepted `imageReference` byte for byte
into every workload container.

Build and push the independently packaged operator:

```bash
export OPERATOR_REPOSITORY="ghcr.io/<org>/aiperf-k8s-operator"
export OPERATOR_TAG="${BENCHMARK_TAG}"

docker build \
  -f aiperf-k8s-operator/Dockerfile \
  -t "${OPERATOR_REPOSITORY}:${OPERATOR_TAG}" \
  aiperf-k8s-operator
docker push "${OPERATOR_REPOSITORY}:${OPERATOR_TAG}"
```

## 2. Install the operator

Install the chart and its pinned JobSet dependency. The chart creates the
operator Service and durable results PVC, but it does not create benchmark
namespaces or choose a benchmark image.

```bash
helm upgrade --install aiperf-operator \
  deploy/aiperf-k8s-operator/helm/aiperf-k8s-operator \
  --namespace aiperf-system \
  --create-namespace \
  --set image.repository="${OPERATOR_REPOSITORY}" \
  --set image.tag="${OPERATOR_TAG}" \
  --set image.pullPolicy=IfNotPresent

kubectl rollout status \
  deployment/aiperf-operator \
  --namespace aiperf-system \
  --timeout=180s
kubectl get pods \
  --namespace aiperf-system \
  --selector app.kubernetes.io/instance=aiperf-operator
```

Set `persistence.storageClass`, `persistence.size`, or
`persistence.accessModes` during the first install when the defaults do not
match the cluster. The Deployment uses a `Recreate` rollout so two operator
pods never contend for the default ReadWriteOnce volume.

## 3. Prepare the authored inputs

Create the namespace and the ConfigMap named by the envelope's `configRef`:

```bash
export BENCHMARK_NAMESPACE="bench"
kubectl create namespace "${BENCHMARK_NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap config-1 \
  --namespace "${BENCHMARK_NAMESPACE}" \
  --from-file=benchmark.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

Compute the digest authored as `configRef.sha256` from the same complete maps.
For the single `benchmark.yaml` key above:

```bash
CONFIG_SHA256=$(jq -S -c -j -n --rawfile benchmark benchmark.yaml \
  '{binaryData:{},data:{"benchmark.yaml":$benchmark}}' \
  | sha256sum | cut -d' ' -f1)
```

The operator verifies this canonical content and creates an immutable
per-incarnation snapshot. Workload pods mount the snapshot rather than
`config-1`, so later source replacement cannot affect the run.

Author `controller-envelope.json` against
`contracts/native-k8s/v1/controller-envelope.schema.json`. The checked-in
`contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json` shows the
shape, but its placeholder digests and bootstrap bytes are not deployment
credentials. A deployable envelope must contain:

- the exact namespace, AIPerfJob name, and unique run ID;
- `imageDigest` and the matching digest-qualified `imageReference`;
- the fixed `controller`, `cell`, and `results-sidecar` commands, argv, and
  environment;
- one bootstrap reference for the controller;
- one numbered `cellBootstraps` reference per cell; and
- the SHA-256 digest of each corresponding local bootstrap file; and
- the complete ConfigMap content digest in `configRef.sha256`.

The bootstrap files are opaque cellular-execution material produced with the
envelope. Do not substitute random bytes, reuse them between runs, add them to
the envelope, or commit them to source control. The results sidecar has no
bootstrap. For a one-cell envelope, set local paths for the controller and cell:

```bash
export CONTROLLER_BOOTSTRAP=/secure/run-1/controller.bootstrap
export CELL_0_BOOTSTRAP=/secure/run-1/cell-0.bootstrap
```

Create `image-capabilities.json` from inspection of that exact benchmark image:

```json
{
  "contractVersion": "native-k8s/v1",
  "imageDigest": "sha256:<same-64-lowercase-hex-digest>",
  "cellular": true,
  "resultsSidecar": true,
  "hierarchicalAggregation": false
}
```

## 4. Validate, preflight, and submit

Validation is local and creates no Kubernetes objects:

```bash
uv run aiperf kube validate \
  --envelope controller-envelope.json \
  --image-capabilities image-capabilities.json
```

Check authenticated Kubernetes API reachability separately:

```bash
uv run aiperf kube preflight \
  --namespace "${BENCHMARK_NAMESPACE}"
```

Submit the envelope and the exact local bootstrap files. Repeat
`--bootstrap-material cell-N=...` for every declared cell:

```bash
uv run aiperf kube profile \
  --envelope controller-envelope.json \
  --image-capabilities image-capabilities.json \
  --bootstrap-material "controller=${CONTROLLER_BOOTSTRAP}" \
  --bootstrap-material "cell-0=${CELL_0_BOOTSTRAP}"
```

Use `--kubeconfig <path>` and `--context <name>` on any command when the
defaults are not the desired cluster. The namespace for submission comes from
the envelope; a separate `--namespace` flag does not override it. The CLI
creates every immutable bootstrap Secret, creates the `AIPerfJob`, and then
binds each Secret to the returned object UID. A failed admission removes newly
created Secrets; a failed owner-binding step removes the CR and newly created
Secrets. The operator refuses JobSet creation until every binding is present
and exact.

## 5. Observe the run and retrieve results

Use the envelope's `jobId`, namespace, and trusted `runId`:

```bash
uv run aiperf kube list --namespace "${BENCHMARK_NAMESPACE}"
uv run aiperf kube show job-1 --namespace "${BENCHMARK_NAMESPACE}"
uv run aiperf kube watch --namespace "${BENCHMARK_NAMESPACE}"
uv run aiperf kube logs job-1 --namespace "${BENCHMARK_NAMESPACE}"

uv run aiperf kube results job-1 \
  --namespace "${BENCHMARK_NAMESPACE}" \
  --run-id run-1 \
  --output-directory ./aiperf-results/run-1
```

The CLI retrieves the durable result through the locally selected Kubernetes
Service proxy. It addresses the persisted namespace/job/run triple directly and
does not need the AIPerfJob, a Secret, or an application-level read capability;
the caller's Kubernetes identity authorizes the Service-proxy request. Results
remain available after the producer or AIPerfJob is deleted and after an
operator restart. If the chart was installed with a nondefault release namespace
or API Service name, add `--operator-namespace <namespace>` and
`--operator-service <name>`.

For API health diagnostics:

```bash
kubectl port-forward \
  --namespace aiperf-system \
  service/aiperf-k8s-operator 8080:8080
curl --fail http://127.0.0.1:8080/healthz
```

## Troubleshooting

### The envelope is rejected before cluster access

Run `aiperf kube validate` and fix the first contract error. Common causes are
a missing `imageReference`, a tag or bare digest used as the reference, a
reference digest that differs from `imageDigest`, a capability-document digest
mismatch, duplicate bootstrap Secret names, or a missing cell bootstrap.

### ImagePullBackOff

Inspect the JobSet pods and events:

```bash
kubectl get pods --namespace "${BENCHMARK_NAMESPACE}"
kubectl describe pod --namespace "${BENCHMARK_NAMESPACE}" <pod-name>
```

Confirm that `imageReference` exists in the registry by that exact digest and
that the node can authenticate to the registry. Changing or preloading a tag
does not satisfy an envelope that names a different immutable digest.

### Submission receives 403

The chart's ClusterRole is intentionally narrow but cluster-wide because the
operator reconciles user-selected namespaces. Confirm the installed release
has the expected ClusterRoleBinding and that admission policy permits the
operator to create JobSets, per-run ServiceAccounts/Roles/RoleBindings, and
immutable Secrets/ConfigMaps in the target namespace. The chart has no
`benchmarkRbacNamespaces` value.

## Related documentation

- [Preflight Checks](preflight.md)
- [RBAC and Security](rbac-security.md)
- [Results Sidecars](sidecars.md)
- [Native Kubernetes control-plane isolation](../specs/kubernetes-control-plane-isolation.md)
