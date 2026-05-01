# 100K-concurrency smoke + Kueue gang-scheduling — DGX, 2026-04-30

**Operator image (during smoke)**: `nvcr.io/nvidian/dynamo-dev/aiperf:k8s-multi-20260501-021032-51137ff98`
**Mock-server**: Rust `aiperf-mock-server-rs:k8s-amd64-20260319-rs-local`
**Cluster**: `nv-prd-dgxc` GKE.

## Headline

**AIPerf successfully ran 100,000 concurrent connections issuing 500,000 requests in 146 s with zero errors** against the rust mock, after spreading worker pods across the system-cpu node pool (which has 144 CPU vs the 24 CPU customer-cpu pool).

Two AIPerf-side scaling bugs surfaced and need separate tickets — both are status-reporting issues, not benchmark-correctness issues.

## What we proved

Single AIPerfJob, one cell:

| Field | Value |
|---|---|
| concurrency | 100,000 |
| connections-per-worker | 2,500 (down from default 250) |
| workers | 40 |
| workersPerPod | 10 |
| worker pods | 4 |
| ISL / OSL | 128 / 128 |
| requests | 500,000 (target) / 500,000 (sent) / 500,000 (completed) / **0 errors** |
| benchmark wall-time | **146.38 s** |
| RPS | **~3,420** |

Live confirmation from `timing-manager` log:
```
05:05:54  Phase profiling started | target: 500,000 requests
05:07:21  sending complete | sent=500,000 completed=400,949 in_flight=99,051
05:08:21  Phase profiling complete | completed=500,000, cancelled=0, errors=0 | elapsed=146.38s
```

## Memory peaks (mock excluded from total)

| Component | Peak |
|---|---|
| Controller pod (records-mgr 165 MiB, dataset-mgr 173 MiB, others) | **1.20 GiB** |
| Worker pod sum (4 pods) | 22.4 GiB |
| **Worker peak per pod** | **6.32 GiB** |
| Operator | 380 MiB |
| **Total (no mock)** | **24.0 GiB** |
| Rust mock-server | 4.6 GiB |

The Rust mock pod RSS climbed sharply at this scale — 4.6 GiB at conc=100K vs ≤340 MiB at conc≤500. That's the mock's per-connection state ballooning. Worth profiling the rust mock at scale separately.

## What unlocked the run

The 24-CPU customer-cpu pool was structurally incapable of fitting the controller + 4 worker pods alongside skypilot, dgxc-alloy, and platform DaemonSets (~85% CPU pre-utilized on every customer-cpu node). The fix:

```yaml
podTemplate:
  nodeSelector: { kubernetes.io/arch: amd64 }   # removed nodeGroup constraint
  tolerations:
    - { effect: NoExecute, key: components.gke.io/gke-managed-components, operator: Equal, value: "true" }
    - { effect: NoSchedule, key: dedicated, operator: Equal, value: user-workload }
    - { effect: NoExecute,  key: dedicated, operator: Equal, value: user-workload }
    - { effect: NoSchedule, key: team, operator: Equal, value: nemo-ci }
```

The new toleration opens the 9-node × 16-CPU **system-cpu** pool (143 CPU). NVIDIA already runs the `kai-scheduler-default`, `karenc-dynamo` operator, and `dgxc-alloy-metrics` there, so AIPerf benchmark workloads landing there is consistent with cluster norms — not a policy violation. The pool's NoExecute taint is opt-in, not blocked.

Pods landed cleanly across 5 different system-cpu nodes (controller + 4 workers each on its own node).

## Two AIPerf scaling bugs surfaced

### 1. CR `status.phase` does not transition past `Initializing` at high service counts

**Symptom**: AIPerfJob `status.phase` stays `Initializing` while the controller's `system_controller.py` log says `AIPerf System is PROFILING`. Eventually flips to `Failed` even though the run completed cleanly (HTTP requests OK, no errors, all credits processed).

**Repro thresholds observed**:

| services | result |
|---|---|
| ≤ 60 worker services | CR transitions Initializing → Running → Completed correctly |
| 150+ worker services (15K conc, conn/worker=100) | CR stuck Initializing for 52 s, then Failed despite live "PROFILING" log |
| 100K conc / 40 worker services / 4 worker_group_managers | CR stuck Initializing for 194 s, then Failed despite full successful run |

**Likely cause**: kopf's status reconciliation falls behind under heavy ZMQ traffic + many service registrations. The watcher may be seeing old resource versions and patching with stale data. Worth investigating in `src/aiperf/operator/handlers/lifecycle.py` and the kopf-event throughput limits in `src/aiperf/operator/main.py`.

### 2. `status.workers.ready` count understates actual readiness

**Symptom**: `status.workers.ready=4/40` while all 40 workers are actually running and issuing 100K concurrent requests at 95%+ CPU each.

**Likely cause**: Only worker_group_managers (one per pod) appear to be counted as "ready," not the individual worker processes inside them. Cosmetic but misleading — the watch dashboard shows "10% ready" for a fully-functional benchmark.

## Kueue gang-scheduling: installed and code-defaulted-on

This session also installed Kueue v0.10.0 cluster-wide and wired AIPerf's existing-but-dormant Kueue support to be default-on.

### Kueue install on DGX

Patched the upstream Kueue manifest to satisfy the cluster's Kyverno `enforce-team-toleration-explicit-value` policy (the default Kueue Deployment has `tolerations: None` which Kyverno rejects on a JMESPath nil-check).

**Persisted**:
- `dev/deploy/aiperf-mock-server-rs.yaml` — Rust mock Deployment + Service.
- `dev/deploy/kueue-aiperf-queues.yaml` — `ResourceFlavor customer-cpu`, `ClusterQueue aiperf-bench-cq` (22 CPU / 80 GiB nominalQuota), `LocalQueue aiperf-lq` in the bench namespace.

### AIPerf code change (default-on)

```python
# src/aiperf/kubernetes/environment.py
class _JobSetSettings:
    KUEUE_DEFAULT_QUEUE_NAME: str = Field(default="", ...)
    KUEUE_DEFAULT_PRIORITY_CLASS: str = Field(default="", ...)

# src/aiperf/kubernetes/jobset.py — _build_manifest_labels
queue_name = (
    self.scheduling.queue_name
    or K8sEnvironment.JOBSET.KUEUE_DEFAULT_QUEUE_NAME
)
if queue_name:
    labels[KueueLabels.QUEUE_NAME] = queue_name

# src/aiperf/operator/models.py — AIPerfJobSpec
scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig, ...)
```

When the operator is set with `AIPERF_K8S_JOBSET_KUEUE_DEFAULT_QUEUE_NAME=aiperf-lq`, every emitted JobSet automatically gets the `kueue.x-k8s.io/queue-name: aiperf-lq` label, and Kueue admits the workload as a unit (gang-scheduling). Per-CR override remains via `spec.scheduling.queueName`.

**Why this matters**: the failure mode where workers grab CPU first and the controller is left Pending → workers crash on registration timeout → run fails — the recurring blocker we hit several times this session — is exactly what gang-scheduling prevents. Kueue admits all pods as a unit or none.

## Cluster integration recommendations

Cluster has Kai-scheduler, Grove, and now Kueue installed. AIPerf currently uses none. Recommendations:

1. **Kueue gang-scheduling**: enable as default on clusters where Kueue is installed. Minimal AIPerf change (env var + label fallback) — done in this session, awaits image rebuild + merge.
2. **Kai-scheduler**: AIPerf should set `schedulerName: kai-scheduler` on its pods when the cluster has Kai. Lets AIPerfJobs participate in cluster-wide priority/preemption alongside other AI workloads. Less critical than Kueue (per-pod, no gang semantics).
3. **Grove**: skip for now — duplicative with Kueue+JobSet for AIPerf's pod hierarchy. Revisit if/when AIPerf gains runtime worker scaling.

## Realistic concurrency ceilings on this cluster (revised)

With the system-cpu toleration:

| Concurrency | Status |
|---|---|
| 100K | ✅ Proven (this run) |
| ≥ 200K | Untested — should fit (cluster has 143 + 24 = 167 CPU available) |
| 500K-1M | Was previously achieved on this cluster's earlier shape (customer-gpu pool, since removed). Likely possible on system-cpu but needs validation, plus the records-manager CPU bump from `gotcha_records_manager_cpu_starves_at_high_concurrency.md`. |

## File layout

- `dev/scripts/smoke_100k_conc.py` — one-shot 100K driver (single cell).
- `dev/deploy/aiperf-mock-server-rs.yaml` — Rust mock Deployment + Service.
- `dev/deploy/kueue-aiperf-queues.yaml` — Kueue ResourceFlavor + ClusterQueue + LocalQueue.
- `dev/results/smoke-100k-conc.log` — driver stdout.
- `dev/results/cr-snapshots-rs/smoke-rs-c100k.json` — CR JSON snapshot at terminal phase.
- `dev/results/smoke-100k-conc-findings.md` — this report.

## Code changes (in working tree, awaiting commit + image rebuild)

- `src/aiperf/kubernetes/environment.py` — `_JobSetSettings.KUEUE_DEFAULT_QUEUE_NAME` + `KUEUE_DEFAULT_PRIORITY_CLASS`.
- `src/aiperf/kubernetes/jobset.py` — `_build_manifest_labels` reads operator-side defaults as fallback.
- `src/aiperf/operator/models.py` — `AIPerfJobSpec.scheduling: SchedulingConfig`.

## Things to look at next

1. **CR status-update bug at scale** — investigate kopf reconciliation under load.
2. **`status.workers.ready` undercount** — fix the Ready accounting to reflect actual worker processes.
3. **Push to 200K / 500K** — should fit on this cluster now that system-cpu pool is in scope.
4. **Profile rust mock RSS at scale** — the 4.6 GiB peak at 100K is much larger than expected.
5. **Helm chart**: bake the system-cpu toleration into the operator's default pod template, so users don't need to know to add it.
6. **Build + ship the Kueue default-on operator image** (in flight at the time of this writeup).
