---
name: aiperf-kube-setup
description: Use when preparing a Kubernetes cluster for AIPerf - installing or upgrading the aiperf-operator Helm chart, setting up a local Kind cluster with GPU passthrough, wiring JobSet and the NVIDIA device plugin, sizing the results PVC, or loading a locally built AIPerf image.
---

# Preparing a Cluster for AIPerf

Everything that happens once per cluster, before any benchmark is submitted.

**Related skills:** `aiperf-kube-run` (submit a benchmark once this is done),
`aiperf-kube-triage` (operator installed but jobs misbehave).

## Cluster prerequisites

| Requirement | Verify |
|---|---|
| Kubernetes v1.24+ with `kubectl` configured | `kubectl cluster-info` |
| NVIDIA device plugin (GPU nodes) | `kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu` |
| JobSet controller | `kubectl get crd jobsets.jobset.x-k8s.io` |
| Helm v3, AIPerf CLI locally | `helm version`, `aiperf --version` |

`aiperf kube preflight` checks connectivity, API versions, RBAC, node capacity
vs worker projection, image pullability, and endpoint reachability in one shot.
Run it before and after install; `-o json` for CI.

## Install the operator

```bash
helm install aiperf-operator deploy/helm/aiperf-operator \
    --namespace aiperf-system --create-namespace
kubectl get pods -n aiperf-system
```

Expect `2/2` ready (operator + results-server), or `3/3` with
`--set dashboard.enabled=true`.

Key knobs in `deploy/helm/aiperf-operator/values.yaml`:

| Value | Default | Why you'd change it |
|---|---|---|
| `image.repository` / `image.tag` / `image.pullPolicy` | `nvcr.io/nvidia/aiperf` / chart appVersion / `IfNotPresent` | local image, pinned release |
| `storage.size` / `storage.storageClassName` / `storage.accessMode` | `1Ti` / cluster default / `ReadWriteOnce` | results PVC sizing |
| `benchmarkNamespace.create` / `.name` | `true` / `aiperf-benchmarks` | pre-provisioned tenant namespace |
| `benchmarkRbacNamespaces` | `[]` | additional namespaces the operator may run jobs in |
| `operator.watchNamespaces` | `[]` (all) | scope the operator |
| `kueue.defaultQueueName` / `kueue.createQueues` | `""` / `false` | gang-scheduled admission |
| `dashboard.enabled` | `false` | Plotly results UI |
| `serviceMonitor.enabled` | `false` | Prometheus scraping of operator metrics |
| `operator.env` | — | per-container resource overrides (`AIPERF_K8S_*`) |
| `rbac.create`, `networkPolicy.enabled` | `true`, `false` | hardening; see `docs/kubernetes/rbac-security.md` |

Full matrix: `docs/kubernetes/configuration.md`; hardening:
`docs/kubernetes/production.md`.

## Local Kind cluster with GPUs

One-time host setup (NVIDIA container runtime as Docker's default):

```bash
sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
docker info 2>/dev/null | grep "Default Runtime"   # -> nvidia
```

Cluster:

```bash
kind create cluster --name aiperf
kubectl --context kind-aiperf apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/deployments/static/nvidia-device-plugin.yml
kubectl --context kind-aiperf apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/latest/download/manifests.yaml

docker build -t aiperf:local .
kind load docker-image aiperf:local --name aiperf

helm install aiperf-operator deploy/helm/aiperf-operator \
    --namespace aiperf-system --create-namespace \
    --set image.repository=aiperf --set image.tag=local --set image.pullPolicy=Never
```

Teardown: `kind delete cluster --name aiperf`.

**Always pass `--context kind-<name>` (or `--kube-context`) explicitly** rather
than switching the active context — it keeps a Kind session from leaking into a
real cluster.

## Rules that bite

- **A locally built image needs `pullPolicy: Never` AND `kind load`.** Skipping
  either yields `ErrImagePull` / `ImagePullBackOff` on every benchmark pod, not
  on the operator.
- **`kind load` must be re-run after every rebuild.** The tag stays the same, so
  nothing signals staleness — pods silently run old code.
- **JobSet is a hard dependency of operator mode.** Without the CRD the operator
  installs fine and every job fails at JobSet creation.
- **`storage.accessMode: ReadWriteOnce` pins the operator to one node.** Use a
  RWX class for multi-replica or node-failure tolerance.
- **`--set benchmarkNamespace.create=false` requires the namespace to exist**
  with the operator's RBAC already granted there (`benchmarkRbacNamespaces`).
- **Upgrading the chart does not restart running benchmarks**; in-flight
  AIPerfJobs keep their old controller image.

## Verify the install

```bash
kubectl get crd | grep aiperf.nvidia.com          # aiperfjobs + aiperfsweeps
kubectl get pods -n aiperf-system
aiperf kube preflight -o json | jq '.checks[] | select(.status != "pass")'
```

CRDs are generated from Pydantic models — never hand-edit
`deploy/helm/aiperf-operator/templates/crd-*.yaml`; run
`uv run python tools/generate_crd.py` (`--check` in CI).
