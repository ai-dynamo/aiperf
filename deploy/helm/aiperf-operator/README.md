<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# aiperf-operator Helm chart

Deploys the AIPerf Kubernetes operator, its CRD (`aiperfjobs.aiperf.nvidia.com`), the results-server sidecar, and optional RBAC / storage / ingress / NetworkPolicy / Kueue / ServiceMonitor resources.

```bash
helm install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system --create-namespace
```

After install, submit a benchmark with `AIPerfJob` — see [`templates/NOTES.txt`](./templates/NOTES.txt) or [`docs/kubernetes/getting-started.md`](../../../docs/kubernetes/getting-started.md).

## Values

All values are documented inline in [`values.yaml`](./values.yaml). Common overrides:

| Value | Purpose |
|---|---|
| `image.repository` / `image.tag` | Operator image (defaults to `nvcr.io/nvidia/aiperf:<appVersion>`). Override `image.tag` for dev builds — it automatically propagates to `AIPERF_DEFAULT_IMAGE` and the CRD schema default. |
| `defaults.image` | Override the benchmark-pod image independently of the operator image. Leave empty to derive from `image.repository`/`image.tag`. |
| `operator.watchNamespaces` | Namespaces to watch for `AIPerfJob` CRs. Empty list (default) watches all namespaces. |
| `benchmarkNamespace.name` / `.create` | Namespace where chart creates `AIPerfJob` RBAC (and optionally the namespace itself). |
| `benchmarkRbacNamespaces` | Extra namespaces to provision benchmark RBAC in (e.g. multi-team setups). |
| `storage.enabled` | Back the results volume with a `PersistentVolumeClaim` instead of an `emptyDir`. |
| `kueue.createQueues` | Chart-manage `ResourceFlavor` / `ClusterQueue` / `LocalQueue`. Gated on the `kueue.x-k8s.io/v1beta1` API being present. |
| `serviceMonitor.enabled` | Create a Prometheus Operator `ServiceMonitor`. Gated on `monitoring.coreos.com/v1`. |
| `ingress.enabled` | Expose the results-server HTTP API outside the cluster. |
| `networkPolicy.enabled` | Lock down the operator pod's ingress/egress. |
| `tests.image` | Image used by `helm test` hook pods. Defaults to `alpine/k8s:1.33.11` (provides `kubectl` + `curl`). |

## Release name guidance

The default fullname helper produces `<release>-<chart>-<role>`. Installing with release name `aiperf` yields resource names like `aiperf-aiperf-operator-…`. Three ways to avoid the duplicated prefix:

1. Use `aiperf-operator` as the release name (the helper collapses when release contains the chart name):
   ```bash
   helm install aiperf-operator deploy/helm/aiperf-operator -n aiperf-system
   ```
2. Set `fullnameOverride` for full control:
   ```bash
   helm install aiperf deploy/helm/aiperf-operator -n aiperf-system \
     --set fullnameOverride=aiperf-operator
   ```
3. Set `nameOverride` to change just the chart-name portion (rarely what you want).

## Uninstall + reinstall behavior

Two resources are annotated `helm.sh/resource-policy: keep` and intentionally survive `helm uninstall`:

- `CustomResourceDefinition aiperfjobs.aiperf.nvidia.com`
- `Namespace <benchmarkNamespace.name>` (default: `aiperf-benchmarks`)

This is so a `helm uninstall` while a benchmark is still running does not drop the live CRs or their namespace. Helm leaves its release-metadata annotations (`meta.helm.sh/release-name`, `meta.helm.sh/release-namespace`) on both — which has a concrete side effect:

**Re-installing under a different release name will fail** with `invalid ownership metadata` on the preserved CRD and namespace. If you need to switch release names, clean up manually first:

```bash
kubectl delete crd aiperfjobs.aiperf.nvidia.com      # only if no live AIPerfJob CRs
kubectl delete namespace aiperf-benchmarks            # only if empty
# then install under the new release name
```

Re-installing under the **same** release name works without any cleanup.

## Running `helm test`

```bash
helm test aiperf-operator -n aiperf-system --logs
```

Two hook pods run:
- `…-test-crd` — verifies the CRD is registered and serving `v1alpha1`.
- `…-test-health` — resolves the operator pod IP and hits `/healthz`.

Both use a dedicated test `ServiceAccount` (`…-tests`) with minimal RBAC: `get` on the AIPerfJob CRD (name-restricted) and `get`/`list` on pods in the release namespace. The test SA and RBAC are themselves `helm.sh/hook: test` resources and are cleaned up alongside the pods.

Hook pods have `helm.sh/hook-delete-policy: before-hook-creation,hook-succeeded,hook-failed`. Use `helm test … --logs` to capture output before deletion; without `--logs`, `kubectl logs` the pods while they are still `Completed` / `Failed` (there's a brief window before `hook-failed` takes effect on the next test run).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `helm test` times out / fails on fresh install | `tests.image` not pullable (airgapped / proxied cluster) | Preload with `kind load docker-image alpine/k8s:1.33.11 --name <cluster>` or `docker save` → node import; or override `tests.image` to a reachable image with `kubectl` + `curl`. |
| `invalid ownership metadata` on reinstall | Previous uninstall kept the CRD/namespace with the old release's metadata | See [Uninstall + reinstall behavior](#uninstall--reinstall-behavior). |
| Operator pod stuck `ImagePullBackOff` | `image.repository` / `image.tag` not reachable from the cluster | Check `imagePullSecrets`; preload with `kind load docker-image` on kind clusters. |
| `AIPERF_DEFAULT_IMAGE` and operator image differ | `defaults.image` set explicitly to a different image | Either unset `defaults.image` (empty → computed from `image.*`) or set both to match. |
| `helm test --logs` returns `unable to get pod logs` | Previous chart versions annotated test RBAC as `helm.sh/hook: test`, causing Helm to try fetching logs for non-Pod resources | Fixed in current chart — test RBAC is a regular resource. No action on a fresh install. |
