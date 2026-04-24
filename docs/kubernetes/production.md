---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Production Deployments
---

# Production Deployments

This guide covers patterns for running AIPerf benchmarks in production environments -- CI/CD pipelines, Kueue-managed clusters, private registries, GitOps workflows, and multi-tenant setups.

---

## CI/CD Integration

### Detach Mode

In non-interactive environments (CI pipelines, cron jobs), AIPerf automatically detaches after deploying. You can also force this explicitly:

```bash
aiperf kube profile \
  --config benchmark.yaml \
  --image nvcr.io/nvidia/aiperf:latest \
  --detach
```

After deploying, poll for completion:

```bash
# Wait for the job to complete
while true; do
  PHASE=$(aiperf kube list my-benchmark 2>/dev/null | awk 'NR==2{print $3}')
  [ "$PHASE" = "Completed" ] || [ "$PHASE" = "Failed" ] && break
  sleep 30
done

# Download results
aiperf kube results my-benchmark --output ./artifacts
```

### JSON-Based Monitoring

For automated pipelines, use JSON output:

```bash
# Preflight check with exit code
aiperf kube preflight -o json
echo "Exit code: $?"    # 0 = all checks passed, 1 = failures

# Validation with structured output
aiperf kube validate -o json benchmark.yaml

# Live monitoring as NDJSON
aiperf kube watch --output json --interval 10
```

### Example GitHub Actions Workflow

```yaml
jobs:
  benchmark:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - name: Preflight check
        run: |
          aiperf kube preflight \
            --endpoint-url http://dynamo-agg-frontend.dynamo-server.svc:8000/v1 \
            -o json

      - name: Run benchmark
        run: |
          aiperf kube profile \
            --config benchmark.yaml \
            --image nvcr.io/nvidia/aiperf:${{ github.sha }} \
            --name "ci-${{ github.run_number }}" \
            --detach

      - name: Wait for completion
        run: |
          aiperf kube watch ci-${{ github.run_number }} \
            --output json 2>/dev/null | while read line; do
            PHASE=$(echo "$line" | jq -r '.phase // empty')
            [ "$PHASE" = "Completed" ] && exit 0
            [ "$PHASE" = "Failed" ] && exit 1
          done

      - name: Collect results
        if: always()
        run: aiperf kube results ci-${{ github.run_number }} --output ./artifacts

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: benchmark-results
          path: ./artifacts/
```

---

## GitOps Workflows

### Generate Manifests for Version Control

Instead of deploying from the CLI, generate Kubernetes manifests and commit them to your GitOps repository:

```bash
# Generate an AIPerfJob CR (operator mode)
aiperf kube generate --operator \
  --config benchmark.yaml \
  --image nvcr.io/nvidia/aiperf:latest \
  > deploy/aiperfjob.yaml

# Or generate raw manifests (no operator needed)
aiperf kube generate --no-operator \
  --config benchmark.yaml \
  --image nvcr.io/nvidia/aiperf:latest \
  > deploy/manifests.yaml
```

Commit the generated YAML and let ArgoCD, Flux, or your GitOps tool apply it.

### Operator Mode vs. Raw Manifests

| Feature | Operator Mode | Raw Manifests |
|---------|--------------|---------------|
| Output | Single AIPerfJob CR | Namespace + RBAC + ConfigMap + JobSet |
| Requires | Operator installed | Only JobSet CRD |
| Monitoring | Operator tracks phase/progress | Manual pod watching |
| Results | Stored on operator PVC | Must retrieve before TTL |
| Cancellation | `spec.cancel: true` | Delete the JobSet |
| Conditions | 7 status conditions populated | None |

For production use, the operator mode is recommended.

---

## Kueue Gang-Scheduling

If your cluster uses [Kueue](https://kueue.sigs.k8s.io/) for resource management, AIPerf integrates with it for quota-managed gang-scheduling.

### Submit to a Kueue Queue

```bash
aiperf kube profile \
  --config benchmark.yaml \
  --image nvcr.io/nvidia/aiperf:latest \
  --queue-name gpu-benchmarks \
  --priority-class high-priority
```

Or in YAML:

```yaml
spec:
  scheduling:
    queueName: gpu-benchmarks
    priorityClass: high-priority
```

When a queue is specified:

1. The JobSet is created in a suspended state
2. Kueue evaluates quota availability
3. Once resources are available, Kueue unsuspends the JobSet
4. The operator detects the transition and monitors normally

The `watch` command shows Kueue suspension status:

```bash
aiperf kube watch my-benchmark
# Shows "Waiting for Kueue admission" during suspension
```

---

## Private Registries

### Image Pull Secrets

If your AIPerf image is in a private registry:

```bash
# Create the pull secret
kubectl create secret docker-registry my-registry \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='YOUR_TOKEN' \
  -n aiperf-benchmarks

# Reference it when deploying
aiperf kube profile ... --image-pull-secrets my-registry
```

Or in YAML:

```yaml
spec:
  podTemplate:
    imagePullSecrets:
      - my-registry
```

### API Keys and Secrets

Pass API keys to the benchmark pods without embedding them in the config:

```bash
# Create a secret with your API key
kubectl create secret generic llm-api-key \
  --from-literal=api-key='sk-...' \
  -n aiperf-benchmarks

# Reference it as an environment variable
aiperf kube profile ... \
  --env-from-secrets 'OPENAI_API_KEY=llm-api-key/api-key'
```

Or in YAML:

```yaml
spec:
  podTemplate:
    env:
      - name: OPENAI_API_KEY
        valueFrom:
          secretKeyRef:
            name: llm-api-key
            key: api-key
```

### Mounting Secret Files

For secrets that need to be files (e.g., certificates, tokens):

```bash
aiperf kube profile ... \
  --secret-mounts '[{"name": "tls-cert", "mount_path": "/certs/ca.pem", "sub_path": "ca.pem"}]'
```

---

## Node Placement

### Target Specific GPUs

```bash
aiperf kube profile ... \
  --node-selector '{"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"}'
```

### Tolerate GPU Taints

Many clusters taint GPU nodes. Add tolerations so benchmark pods can schedule there:

```bash
aiperf kube profile ... \
  --tolerations '[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]'
```

Or in YAML:

```yaml
spec:
  podTemplate:
    nodeSelector:
      nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

---

## Scaling Workers

AIPerf distributes workers across pods automatically. The `--workers-max` flag sets the total number of workers. The system places 10 workers per pod by default:

| `--workers-max` | Pods Created | Workers Per Pod |
|-----------------|-------------|-----------------|
| 10 | 1 | 10 |
| 50 | 5 | 10 |
| 100 | 10 | 10 |
| 200 | 20 | 10 |

Each worker maintains up to `connectionsPerWorker` concurrent requests (default: 100). The CLI auto-computes this from your concurrency and worker count: `connectionsPerWorker = ceil(concurrency / workers)`.

For high-concurrency benchmarks, scale workers rather than increasing connections per worker. This distributes the load across pods and nodes.

---

## Results Server

The operator includes a results server sidecar that provides HTTP access to stored results. This powers `aiperf kube results` (which retrieves from the operator's PVC by default) and provides DuckDB-powered analytics endpoints.

### Available Endpoints

After the operator is running:

```bash
# Port-forward to the results server
kubectl port-forward -n aiperf-system svc/aiperf-operator 8081:8081

# List stored results
curl localhost:8081/api/v1/results

# Get summary for a specific job
curl localhost:8081/api/v1/analytics/summary/aiperf-benchmarks/my-benchmark

# Leaderboard across all benchmarks
curl localhost:8081/api/v1/analytics/leaderboard

# Compare two runs
curl localhost:8081/api/v1/analytics/compare?jobs=run-a,run-b

# Job history
curl localhost:8081/api/v1/analytics/history?model=Qwen/Qwen3-0.6B
```

### Storage Configuration

Results are stored on the operator's PVC. Configure retention in the Helm values:

```yaml
operator:
  env:
    resultsTtlDays: "30"           # auto-cleanup after 30 days
    resultsCompressOnDisk: "true"  # zstd compression

storage:
  enabled: true                    # default is false; must be true to provision the PVC
  size: 1Ti                        # only used when enabled: true
  storageClassName: ""             # cluster default; only used when enabled: true
```

### Exposing Results Outside the Cluster (Ingress)

By default the results server is reachable only via `ClusterIP + kubectl port-forward`. To expose it through an Ingress (e.g. for a shared dashboard link), enable `ingress.*` in the Helm values:

```yaml
ingress:
  enabled: true
  className: "nginx"          # empty uses the cluster default IngressClass
  annotations: {}
  hosts:
    - host: aiperf.example.com
      paths:
        - path: /
          pathType: Prefix    # backend port defaults to resultsServer.port (8081)
  tls: []                     # optional: list of {hosts: [...], secretName: ...}
```

The template lives at `deploy/helm/aiperf-operator/templates/ingress.yaml` and routes to the operator Service on `resultsServer.port`. Each path may override the backend port via `portNumber`.

---

## Environment Variables

Fine-tune benchmark behavior with environment variables in the pod template:

```yaml
spec:
  podTemplate:
    env:
      - name: AIPERF_HTTP_CONNECTION_LIMIT
        value: "200"
      - name: AIPERF_HTTP_TIMEOUT
        value: "120"
      - name: AIPERF_K8S_HEALTH_STARTUP_PERIOD_SECONDS
        value: "30"
      - name: AIPERF_K8S_HEALTH_STARTUP_FAILURE_THRESHOLD
        value: "60"
```

See the [Environment Variables Reference](../environment-variables.md) for the full list.

---

## Multi-Tenant Clusters

### Namespace Isolation

By default, benchmarks run in `aiperf-benchmarks`. For multi-tenant setups, use separate namespaces:

```bash
aiperf kube profile ... --namespace team-a-benchmarks
aiperf kube profile ... --namespace team-b-benchmarks
```

The operator watches all namespaces for AIPerfJob CRs. Each job gets its own RBAC scoped to its namespace.

### Resource Quotas

The preflight checker validates resource quotas before deploying:

```bash
aiperf kube preflight --namespace team-a-benchmarks --workers 20
```

If the namespace has a ResourceQuota, preflight projects the total CPU and memory requirements and warns if deployment would exceed the quota.

---

## Operator Management

### Upgrading

```bash
helm upgrade aiperf-operator deploy/helm/aiperf-operator \
  --namespace aiperf-system
```

### Uninstalling

```bash
helm uninstall aiperf-operator --namespace aiperf-system
```

The CRD and any existing AIPerfJob resources are preserved during uninstall. To remove everything:

```bash
kubectl delete crd aiperfjobs.aiperf.nvidia.com
kubectl delete namespace aiperf-system
```

### Monitoring the Operator

Check operator logs:

```bash
kubectl logs -n aiperf-system -l app.kubernetes.io/name=aiperf-operator -f
```

The operator emits structured logs with job events, phase transitions, and error details.

### Debugging Failed Jobs

By default the operator cleans up JobSet pods once a job terminates, which makes postmortems hard. Set `keepFailedPods: true` on the AIPerfJob spec to preserve failed pod attempts for inspection:

```yaml
spec:
  keepFailedPods: true   # default: false; retains failed JobSet pods for kubectl logs/exec
```

This changes JobSet lifecycle (completed pods still get reaped via `ttlSecondsAfterFinished`); enable it only while diagnosing, then remove for steady-state runs.

### Deploying Before the Endpoint Is Reachable

The operator runs a TCP/HTTP reachability probe against `spec.benchmark.endpoint` before spinning up workers. If the inference server isn't live yet at deploy time (e.g. GitOps applies the CR before the model pod is Ready), set `skipEndpointCheck: true` to bypass the probe:

```yaml
spec:
  skipEndpointCheck: true   # default: false; skips the operator-side endpoint reachability probe
```

Workers will still fail fast if the endpoint never comes up -- this only suppresses the upfront check.

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Kubernetes Configuration](configuration.md) -- All CRD fields and deployment options
- [Monitoring and Troubleshooting](monitoring.md) -- Watch, debug, and diagnose issues
- [Environment Variables](../environment-variables.md) -- All AIPERF_* environment variables
