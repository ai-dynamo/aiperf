<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# aiperf-operator Helm chart

Deploys the AIPerf Kubernetes operator, its two CRDs (`aiperfjobs.aiperf.nvidia.com` and `aiperfsweeps.aiperf.nvidia.com`), the results-server sidecar, an optional dashboard sidecar, and optional RBAC / storage / ingress / NetworkPolicy / Kueue / ServiceMonitor resources.

Both CRD templates are generated from the `AIPerfJobSpec` / `AIPerfSweepSpec` Pydantic models by `tools/generate_crd.py`. Do not hand-edit `templates/crd-aiperfjob.yaml` or `templates/crd-aiperfsweep.yaml`.

```bash
helm install aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system --create-namespace
```

After install, submit a benchmark with `AIPerfJob` — see [`templates/NOTES.txt`](./templates/NOTES.txt) or [`docs/kubernetes/getting-started.md`](../../../docs/kubernetes/getting-started.md).

Benchmarks run in the release namespace. The chart creates no `Namespace` objects of its own; it provisions benchmark RBAC in the namespace you install into, plus any already-existing namespaces you list in `benchmarkRbacNamespaces`.

## Values

All values are documented inline in [`values.yaml`](./values.yaml). Common overrides:

| Value | Purpose |
|---|---|
| `image.repository` / `image.tag` | Operator image (defaults to `nvcr.io/nvidia/aiperf:<appVersion>`). Override `image.tag` for dev builds — it automatically propagates to the CRD schema default for `spec.image`. |
| `defaults.image` | Override the benchmark-pod image independently of the operator image. Leave empty to derive from `image.repository`/`image.tag`. |
| `operator.watchNamespaces` | Namespaces to watch for `AIPerfJob` and `AIPerfSweep` CRs (`--namespace=<ns>` per entry). Empty list (default) passes `--all-namespaces`. |
| `benchmarkRbacNamespaces` | Extra, already-existing namespaces to provision benchmark RBAC in, beyond the release namespace (e.g. multi-team setups). The chart does not create them. |
| `serverMetricsDiscoveryNamespaces` | Existing inference namespaces (e.g. a Dynamo deployment) where benchmark pods get read-only `pods` RBAC for cross-namespace server-metrics endpoint discovery. A plain string entry binds the benchmark namespaces' `default` ServiceAccount; use `{namespace, serviceAccounts: [...]}` to bind custom ServiceAccounts (`podTemplate.serviceAccountName`). Namespaces are not created by the chart. |
| `serviceAccount.create` / `.name` | Default `create: true` provisions the operator ServiceAccount and binds it to the chart's ClusterRole. With `create: false` the chart binds no RBAC, so `serviceAccount.name` becomes **required** — it must name a pre-provisioned account already bound to the operator's ClusterRole. The render fails otherwise, rather than silently using the namespace `default` account, which has no operator permissions. |
| `storage.enabled` | Default `true`: back the results volume with a `PersistentVolumeClaim` so benchmark results survive operator restarts. Set `false` to fall back to `emptyDir` (ephemeral, useful for clusters with no default `StorageClass`). |
| `kueue.createQueues` | Chart-manage `ResourceFlavor` / `ClusterQueue` / `LocalQueue`. Gated on the `kueue.x-k8s.io/v1beta1` API being present. |
| `serviceMonitor.enabled` | Create a Prometheus Operator `ServiceMonitor`. Gated on `monitoring.coreos.com/v1`. |
| `ingress.enabled` | Expose the results-server HTTP API outside the cluster. |
| `networkPolicy.enabled` | Lock down the operator pod's ingress/egress. |
| `dashboard.enabled` | Add the Plotly Dash sidecar (off by default). results-server reverse-proxies `/dashboard/*` to it on `dashboard.port` (8082). |
| `tests.enabled` | Default `true`: render the `helm test` hook pods plus the dedicated test ServiceAccount, Role, RoleBinding, ClusterRole, and ClusterRoleBinding. Set `false` to render none of them — this is the only way to make the chart emit zero cluster-scoped RBAC, because the CRD check reads cluster-scoped resources. Deliberately independent of `rbac.create`; see [`helm test` and cluster-scoped RBAC](#helm-test-and-cluster-scoped-rbac). |
| `tests.image.repository` / `.tag` | Image used by `helm test` hook pods. Defaults to `alpine/k8s:1.33.11` (provides `kubectl` + `curl`). |

## Release name guidance

The default fullname helper produces `<release>-<chart>`, and individual resources append a role suffix (`-results`, `-benchmark`, `-tests`, …). Installing with release name `aiperf` yields resource names like `aiperf-aiperf-operator-…`. Three ways to avoid the duplicated prefix:

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

These resources are annotated `helm.sh/resource-policy: keep` and intentionally survive `helm uninstall`:

- `CustomResourceDefinition aiperfjobs.aiperf.nvidia.com`
- `CustomResourceDefinition aiperfsweeps.aiperf.nvidia.com`

This is so a `helm uninstall` while a benchmark is still running does not drop the live CRs. Helm leaves its release-metadata annotations (`meta.helm.sh/release-name`, `meta.helm.sh/release-namespace`) on them — which has a concrete side effect:

**Re-installing under a different release name will fail** with `invalid ownership metadata` on the preserved CRDs. If you need to switch release names, clean up manually first:

```bash
kubectl delete crd aiperfjobs.aiperf.nvidia.com      # only if no live AIPerfJob CRs
kubectl delete crd aiperfsweeps.aiperf.nvidia.com    # only if no live AIPerfSweep CRs
# then install under the new release name
```

Re-installing under the **same** release name works without any cleanup.

## Running `helm test`

```bash
helm test aiperf-operator -n aiperf-system --logs
```

Two hook pods run:
- `…-test-crd` — verifies both CRDs are registered and serving `v1alpha1`.
- `…-test-health` — resolves the operator pod IP, hits kopf's `/healthz` on 8080, then the results-server's `/api/v1/jobs` on `resultsServer.port`.

Both use a dedicated test `ServiceAccount` (`…-tests`) with minimal RBAC: `get` on the two AIPerf CRDs (name-restricted) and `get`/`list` on pods in the release namespace. The test SA and RBAC are ordinary chart-managed resources, not test-phase hooks, so `helm test --logs` only iterates the actual Pod hooks; they are removed by `helm uninstall`.

Hook pods carry only `helm.sh/hook-delete-policy: before-hook-creation`, so they persist after the run and are deleted at the start of the next `helm test`. Use `helm test … --logs` to stream output, or `kubectl logs` the `Completed` / `Failed` pods any time before the next test run.

### `helm test` and cluster-scoped RBAC

`rbac.create=false` suppresses the operator's own `ClusterRole` and `ClusterRoleBinding`, but not the test hooks' — the `…-tests` `ClusterRole`/`ClusterRoleBinding` still render, because pre-provisioning operator RBAC out-of-band says nothing about the chart's smoke-test scaffolding. Use `tests.enabled=false` to drop them:

```bash
helm template aiperf-operator deploy/helm/aiperf-operator -n aiperf-system \
  --set rbac.create=false \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aiperf-operator-sa \
  --set tests.enabled=false \
  | grep -c '^kind: Cluster'   # -> 0
```

`tests.enabled` gates the hook Pods and their RBAC together, so you never get Pods referencing a ServiceAccount that was not created. Keeping `helm test` and reaching zero cluster-scoped RBAC are mutually exclusive: `…-test-crd` runs `kubectl get crd`, and CustomResourceDefinitions are cluster-scoped. With the hooks off, verify the install out-of-band with `kubectl get crd aiperfjobs.aiperf.nvidia.com aiperfsweeps.aiperf.nvidia.com` and `aiperf kube preflight`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `helm test` times out / fails on fresh install | `tests.image` not pullable (airgapped / proxied cluster) | Preload with `kind load docker-image alpine/k8s:1.33.11 --name <cluster>` or `docker save` → node import; or override `tests.image.repository` / `tests.image.tag` to a reachable image with `kubectl` + `curl`. |
| `invalid ownership metadata` on reinstall | Previous uninstall kept the CRDs with the old release's metadata | See [Uninstall + reinstall behavior](#uninstall--reinstall-behavior). |
| Operator pod stuck `ImagePullBackOff` | `image.repository` / `image.tag` not reachable from the cluster | Check `imagePullSecrets`; preload with `kind load docker-image` on kind clusters. |
| Benchmark-pod image (CRD `spec.image` default) and operator image differ | `defaults.image` set explicitly to a different image | Either unset `defaults.image` (empty → computed from `image.*`) or set both to match. |
| `helm test --logs` returns `unable to get pod logs` | Previous chart versions annotated test RBAC as `helm.sh/hook: test`, causing Helm to try fetching logs for non-Pod resources | Fixed in current chart — test RBAC is a regular resource. No action on a fresh install. |
| `helm test` prints `TEST SUITE: None` and exits 0 | Release was installed or upgraded with `tests.enabled=false`, so no `helm.sh/hook: test` Pod is in the deployed revision | Expected. `helm upgrade --set tests.enabled=true` if you want the hooks, or run the equivalent checks out-of-band (see [`helm test` and cluster-scoped RBAC](#helm-test-and-cluster-scoped-rbac)). |
