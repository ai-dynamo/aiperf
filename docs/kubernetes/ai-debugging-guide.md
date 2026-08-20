---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: AI Agent Debugging Guide
---

# AI Agent Debugging Guide for AIPerf on Kubernetes

This guide is written for AI coding agents (Claude, Copilot, Cursor, etc.) that need to diagnose and fix AIPerf Kubernetes benchmark issues. Every command produces machine-parseable output. Every decision point has explicit criteria. No ambiguity.

---

## How to Use This Guide

1. Start at [Triage](#triage) to classify the problem
2. Follow the decision tree for your problem class
3. Each section gives you the exact command, the JSON schema of the output, and the decision logic
4. All commands use `--output json` or equivalent for structured output

---

## Triage

Run this command first. It gives you everything you need to classify the problem:

```bash
kubectl get aiperfjob <NAME> -n <NS> -o json | python3 -c "
import sys, json
st = json.load(sys.stdin).get('status', {})
w = st.get('workers', {})
print(json.dumps({
    'phase': st.get('phase'),
    'subPhase': st.get('subPhase'),
    'currentPhase': st.get('currentPhase'),
    'workers_ready': w.get('ready', 0),
    'workers_total': w.get('total', 0),
    'conditions': st.get('conditions', []),
    'error': st.get('error'),
}, indent=2))
"

# Pod-level triage (container states, recent events, node pressure)
aiperf kube debug --job <NAME> --namespace <NS>
```

### Decision Tree

Branch on `status.phase` from the CR, then on what `aiperf kube debug` reports.
(The `health` / `error_rate` rollups came from the removed live-watch command;
derive the equivalent from pod state and metrics as shown below.)

```
phase == "Completed"
  -> Benchmark finished. Go to [Collect Results](#collect-results).

phase == "Failed"
  -> Go to [Failed Job](#failed-job).

phase == "Cancelled"
  -> Job was cancelled. Check if intentional. No action needed.

phase == "Pending" for more than ~60s
  -> Go to [Stuck in Pending](#stuck-in-pending).

phase == "Queued"
  -> Job is waiting for Kueue admission. Go to [Kueue Issues](#kueue-issues).

phase == "Initializing" for more than ~120s
  -> Go to [Stuck Initializing](#stuck-initializing).

phase == "Running", and `aiperf kube debug` shows a pod with restarts > 3
  -> Go to [Crash Loop](#crash-loop).

phase == "Running", and `aiperf kube debug` reports an OOM-killed pod
  -> Go to [OOM Kills](#problem-oom-kills).

phase == "Running", but status.phases.*.requestsCompleted is not advancing
  -> Go to [Stalled Benchmark](#stalled-benchmark).

phase == "Running", and error_count / request_count > 0.05 in
status.liveMetrics.metrics
  -> Go to [High Error Rate](#high-error-rate).

phase == "Running" and none of the above
  -> Benchmark is running normally. Monitor with:
     aiperf kube attach
```

---

## Problem: Failed Job

### Gather Information

```bash
# Get the error message from the CR status
kubectl get aiperfjob <JOB_NAME> -n <NAMESPACE> -o json | \
  python3 -c "import sys,json; s=json.load(sys.stdin)['status']; print(json.dumps({'phase':s.get('phase'),'error':s.get('error'),'conditions':s.get('conditions',[])}, indent=2))"
```

```bash
# Get controller pod logs (last 50 lines)
aiperf kube logs <JOB_ID> --container control-plane --tail 50
```

```bash
# Run full diagnostics
aiperf kube debug --job-id <JOB_ID> --verbose
```

### Common Failure Patterns

| Error contains | Root cause | Fix |
|---|---|---|
| `preflight` | Cluster validation failed | Run `aiperf kube preflight -o json` and fix failing checks |
| `endpoint` or `health check` | Inference server unreachable | Verify endpoint URL resolves from inside the cluster |
| `timeout` | Benchmark exceeded `timeoutSeconds` | Increase `spec.timeoutSeconds` or set to `0` |
| `ConfigMap` or `size` | Config too large for K8s 1MiB limit | Reduce config size |
| `image` or `pull` | Container image not accessible | Check image name and pull secrets |
| `RBAC` or `forbidden` | Missing permissions | Check service account and role bindings |

---

## Problem: Stuck in Pending

Pods cannot be scheduled. Get the reason:

```bash
# Check pod events for scheduling failures
kubectl get pods -n <NAMESPACE> -l aiperf.nvidia.com/job-id=<JOB_ID> -o json | \
  python3 -c "
import sys, json
pods = json.load(sys.stdin)['items']
for pod in pods:
    name = pod['metadata']['name']
    conditions = pod['status'].get('conditions', [])
    for c in conditions:
        if c.get('type') == 'PodScheduled' and c.get('status') == 'False':
            print(json.dumps({'pod': name, 'reason': c.get('reason'), 'message': c.get('message')}))
"
```

```bash
# Check node resources
kubectl get nodes -o json | python3 -c "
import sys, json
nodes = json.load(sys.stdin)['items']
for n in nodes:
    alloc = n['status']['allocatable']
    print(json.dumps({
        'node': n['metadata']['name'],
        'cpu': alloc.get('cpu'),
        'memory': alloc.get('memory'),
        'gpu': alloc.get('nvidia.com/gpu', '0'),
    }))
"
```

### Decision Logic

| Scheduling message contains | Fix |
|---|---|
| `Insufficient cpu` or `Insufficient memory` | Reduce worker count: `--total-workers <lower_number>` |
| `nvidia.com/gpu` | No available GPU nodes. Wait or add capacity. |
| `didn't match Pod's node affinity/selector` | Fix `spec.podTemplate.nodeSelector` to match existing nodes |
| `had untolerated taint` | Add tolerations to `spec.podTemplate.tolerations` |
| `quota` | Namespace ResourceQuota exhausted. Request more or use different namespace. |

---

## Problem: Kueue Issues

```bash
# Check if the workload is admitted
kubectl get workloads -n <NAMESPACE> -o json | python3 -c "
import sys, json
items = json.load(sys.stdin)['items']
for w in items:
    name = w['metadata']['name']
    conditions = w.get('status', {}).get('conditions', [])
    admitted = any(c['type'] == 'Admitted' and c['status'] == 'True' for c in conditions)
    print(json.dumps({'workload': name, 'admitted': admitted, 'conditions': [{'type': c['type'], 'status': c['status'], 'message': c.get('message','')} for c in conditions]}))
"
```

```bash
# Check ClusterQueue capacity
kubectl get clusterqueues -o json | python3 -c "
import sys, json
items = json.load(sys.stdin)['items']
for q in items:
    print(json.dumps({
        'name': q['metadata']['name'],
        'flavors': q.get('status', {}).get('flavorsReservation', []),
        'pending': q.get('status', {}).get('pendingWorkloads', 0),
    }))
"
```

If the workload is not admitted, the queue is full. Wait for other workloads to complete, or adjust priority with `spec.scheduling.priorityClass`.

---

## Problem: Stuck Initializing

Workers are starting but not all are ready yet.

```bash
# Check which pods are not ready
kubectl get pods -n <NAMESPACE> -l aiperf.nvidia.com/job-id=<JOB_ID> -o json | \
  python3 -c "
import sys, json
pods = json.load(sys.stdin)['items']
for pod in pods:
    containers = pod['status'].get('containerStatuses', [])
    for c in containers:
        if not c.get('ready', False):
            waiting = c.get('state', {}).get('waiting', {})
            print(json.dumps({
                'pod': pod['metadata']['name'],
                'container': c['name'],
                'ready': False,
                'waiting_reason': waiting.get('reason', 'unknown'),
                'waiting_message': waiting.get('message', ''),
                'restarts': c.get('restartCount', 0),
            }))
"
```

| Waiting reason | Fix |
|---|---|
| `ContainerCreating` | Normal -- image is pulling. Wait. |
| `ImagePullBackOff` | Image does not exist or no pull secret. Fix image or add `--image-pull-secrets`. |
| `CrashLoopBackOff` | Container crashes on startup. Check logs with `aiperf kube logs --container <name>`. |
| `CreateContainerConfigError` | Missing ConfigMap, Secret, or volume. Check the pod events. |

---

## Problem: Crash Loop

A pod is restarting repeatedly (>3 restarts).

```bash
# Get logs from the previous (crashed) container
kubectl logs -n <NAMESPACE> <POD_NAME> --previous -c <CONTAINER_NAME> --tail=50
```

```bash
# Get the exit code
kubectl get pod -n <NAMESPACE> <POD_NAME> -o json | python3 -c "
import sys, json
pod = json.load(sys.stdin)
for c in pod['status'].get('containerStatuses', []):
    term = c.get('lastState', {}).get('terminated', {})
    if term:
        print(json.dumps({
            'container': c['name'],
            'exit_code': term.get('exitCode'),
            'reason': term.get('reason'),
            'message': term.get('message', ''),
        }))
"
```

| Exit code | Meaning | Fix |
|---|---|---|
| 137 | SIGKILL (OOM or external kill) | Increase memory limits. See [OOM Kills](#problem-oom-kills). |
| 1 | Application error | Read the logs. Common: bad config, missing model, endpoint unreachable. |
| 2 | Python syntax/import error | Image may be wrong version. Verify `--image`. |

---

## Problem: OOM Kills

A pod was killed because it exceeded its memory limit.

```bash
# Confirm OOM and get memory limits
kubectl get pod -n <NAMESPACE> <POD_NAME> -o json | python3 -c "
import sys, json
pod = json.load(sys.stdin)
for c in pod['spec']['containers']:
    limits = c.get('resources', {}).get('limits', {})
    print(json.dumps({'container': c['name'], 'memory_limit': limits.get('memory', 'none')}))
for c in pod['status'].get('containerStatuses', []):
    term = c.get('lastState', {}).get('terminated', {})
    if term.get('reason') == 'OOMKilled':
        print(json.dumps({'container': c['name'], 'oom_killed': True}))
"
```

### Fixes (in priority order)

1. **Reduce connections per worker** -- Lower `spec.connectionsPerWorker` (default: 100). Each connection holds request/response buffers in memory.

2. **Increase workers, reduce per-pod** -- Use more pods with fewer workers each. Lower `spec.benchmark.runtime.workersPerPod` in the CR (the cluster-wide worker total stays `--total-workers`; this knob only controls how that total is fanned across pods).

3. **Override memory limits** via environment variables:
   ```yaml
   spec:
     podTemplate:
       env:
         - name: AIPERF_K8S_WORKER_POD_MEMORY
           value: "8192Mi"
   ```

---

## Problem: Stalled Benchmark

Running phase but no progress (0 throughput, 0 requests completed).

```bash
# Check Dynamo endpoint reachability from inside the cluster
# URL pattern: http://{deploy-name}-frontend.{namespace}.svc:8000/v1/models
kubectl run aiperf-curl-test --rm -it --restart=Never \
  --image=curlimages/curl -- \
  curl -s -o /dev/null -w '{"http_code":%{http_code},"time_total":%{time_total}}' \
  <ENDPOINT_URL>/models
```

```bash
# Check controller logs for endpoint errors
aiperf kube logs <JOB_ID> --container control-plane --tail 30 2>&1 | grep -i "error\|timeout\|refused\|unreachable"
```

| Symptom | Fix |
|---|---|
| `curl` returns `http_code: 0` or `Connection refused` | Endpoint URL wrong or Dynamo frontend not running. Verify URL pattern: `http://{deploy-name}-frontend.{namespace}.svc:8000/v1`. Check Dynamo pods: `kubectl get pods -n dynamo-server`. |
| `curl` returns `http_code: 200` but benchmark stalled | Workers may not be connecting. Check ZMQ connectivity in controller logs. |
| `curl` times out | Network policy blocking traffic, or Dynamo workers are still loading the model. Check pod logs: `kubectl logs -n dynamo-server -l app.kubernetes.io/managed-by=dynamo-operator`. |

---

## Problem: High Error Rate

More than 5% of requests are failing.

```bash
# Get live metrics with error breakdown
kubectl get aiperfjob <NAME> -n <NS> \
  -o jsonpath='{.status.liveMetrics.metrics}' | python3 -m json.tool
```

| Error rate range | Likely cause | Fix |
|---|---|---|
| 5-20% | Endpoint overloaded | Reduce `concurrency` in the phase config |
| 20-50% | Model or endpoint errors | Check endpoint logs for 500/503 errors |
| >50% | Endpoint down or misconfigured | Verify model name matches what the server is serving |
| 100% | Wrong endpoint URL or auth required | Fix URL or add API key via `--env-from-secrets` |

---

## Collect Results

```bash
# Download all artifacts (default: fetched from operator storage,
# which works even after pods are deleted)
aiperf kube results <JOB_ID> --output ./artifacts

# Retrieve directly from benchmark pods instead of operator storage
aiperf kube results <JOB_ID> --from-pods --output ./artifacts

# Read the summary metrics
cat ./artifacts/profile_export_aiperf.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(json.dumps({
    'request_throughput': data.get('request_throughput'),
    'request_latency_avg': data.get('request_latency_avg'),
    'request_latency_p99': data.get('request_latency_p99'),
    'ttft_avg': data.get('time_to_first_token_avg'),
    'ttft_p99': data.get('time_to_first_token_p99'),
    'itl_avg': data.get('inter_token_latency_avg'),
    'output_token_throughput': data.get('output_token_throughput'),
    'total_requests': data.get('request_count'),
    'error_count': data.get('error_count'),
}, indent=2))
"
```

## Preflight JSON Schema

Output from `aiperf kube preflight -o json`:

```json
{
  "passed": true,
  "has_warnings": false,
  "checks": [
    {
      "name": "Cluster Connectivity",
      "status": "pass",
      "message": "Connected to cluster",
      "details": [],
      "hints": [],
      "duration_ms": 45.2
    },
    {
      "name": "JobSet CRD",
      "status": "fail",
      "message": "JobSet CRD not found",
      "details": ["API group jobset.x-k8s.io not registered"],
      "hints": ["Install JobSet: kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/v0.5.2/manifests.yaml"],
      "duration_ms": 12.1
    }
  ]
}
```

### Check Statuses

| Status | Meaning | Agent action |
|---|---|---|
| `pass` | Check passed | No action |
| `fail` | Check failed, deployment will fail | Must fix before deploying. Read `hints`. |
| `warn` | Potential issue | Review but not blocking |
| `skip` | Check not applicable | Ignore |
| `info` | Informational | Log for context |

### Agent Decision Logic

```python
preflight = json.loads(subprocess.check_output(["aiperf", "kube", "preflight", "-o", "json"]))
if not preflight["passed"]:
    for check in preflight["checks"]:
        if check["status"] == "fail":
            # Apply hints[0] if available, otherwise report to user
            if check["hints"]:
                print(f"Fix: {check['hints'][0]}")
            else:
                print(f"BLOCKED: {check['name']}: {check['message']}")
    sys.exit(1)
```

---

## Validate JSON Schema

Output from `aiperf kube validate -o json benchmark.yaml`:

```json
[
  {
    "path": "benchmark.yaml",
    "passed": true,
    "errors": [],
    "warnings": ["Unknown spec field: foo"]
  }
]
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Get structured triage snapshot | `kubectl get aiperfjob <NAME> -n <NS> -o json` |
| Get job phase and error | `kubectl get aiperfjob <NAME> -n <NS> -o jsonpath='{.status.phase} {.status.error}'` |
| Check preflight (JSON) | `aiperf kube preflight -o json` |
| Validate config (JSON) | `aiperf kube validate -o json <FILE>` |
| List all jobs (kubectl) | `kubectl get aiperfjobs -A -o json` |
| Get pod statuses | `kubectl get pods -n <NS> -l aiperf.nvidia.com/job-id=<ID> -o json` |
| Get controller logs | `aiperf kube logs <ID> --container control-plane --tail 50` |
| Get worker logs | `aiperf kube logs <ID> --container worker-group-manager --tail 50` |
| Get events | `kubectl get events -n <NS> --sort-by=.lastTimestamp -o json` |
| Cancel a job | `kubectl patch aiperfjob <NAME> -n <NS> --type=merge -p '{"spec":{"cancel":true}}'` |
| Delete a job | `kubectl delete aiperfjob <NAME> -n <NS>` |
| Download results (from operator, default) | `aiperf kube results <ID> --output ./artifacts` |
| Download directly from pods | `aiperf kube results <ID> --from-pods --output ./artifacts` |

---

## Related Documentation

- [Getting Started](getting-started.md) -- First benchmark walkthrough
- [Monitoring and Troubleshooting](monitoring.md) -- Human-readable monitoring guide
- [Kubernetes Configuration](configuration.md) -- All CRD fields and deployment options
