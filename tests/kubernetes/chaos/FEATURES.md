<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Chaos Suite — Feature Inventory

A reference of what the suite under [`tests/kubernetes/chaos/`](.) actually contains, component by component. Companion to [`README.md`](README.md), which covers how to run scenarios and how they map to operator code. This document is for someone evaluating the suite's surface area — what helpers exist, what they do, and which pieces are generic vs. AIPerf-specific.

Each section flags the AIPerf-coupling level:

- **Generic** — works against any kubernetes operator/CRD with at most trivial renaming.
- **Mixed** — generic mechanism with AIPerf-shaped defaults (label selectors, namespace constants); parameterizable.
- **AIPerf-specific** — welded to AIPerfJob CRD lifecycle, JobSet labels, or operator-internal annotations.

## 1. Test scaffolding & conventions

**Pytest flags** ([`tests/kubernetes/conftest.py`](../conftest.py)). *Generic.*

- `--k8s-reuse-cluster` / `K8S_TEST_REUSE_CLUSTER` — skip Kind cluster create/destroy; reuse a long-lived `aiperf-pytest` cluster for fast iteration.
- `--k8s-skip-build` / `K8S_TEST_SKIP_BUILD` — skip image build; reuse existing images already loaded into the cluster.

**Markers.** *Generic.*

- `@pytest.mark.k8s_slow` — every chaos test carries this so the suite is opt-in (`-m k8s_slow`) and never runs on a default `pytest` invocation. Tests routinely wait 30 s for grace, 15 s for operator restart, or up to 2 minutes for benchmark completion.
- `@pytest.mark.asyncio` — all tests are async.
- `@pytest.mark.timeout(N)` — applied to long scenarios; current values range from 300 s to 1200 s (the sweep-controller kill).

**Convention: cleanup in `finally`.** *Generic.* Every chaos test wraps fault injection in `try/finally`. `finally` clauses force-delete the CR/namespace and call `await injector.reset()` (toxiproxy) or `await injector.restore()` (mock server) so a leaked fault never poisons the next test. Belt-and-suspenders intentional.

**Convention: `xfail` with flip-to-pass condition.** *Generic.* Scenarios that depend on unavailable infra ship as `@pytest.mark.xfail(strict=False, reason=...)` with a concrete condition in the reason string. `strict=True` is forbidden — chaos scenarios are exploratory by definition. No scenario currently carries an `xfail`; every test in the suite is expected to pass.

## 2. ChaosInjector (kubectl-level faults)

[`chaos_injector.py`](chaos_injector.py) — a single class wrapping `KubectlClient` with intent-revealing methods. Async throughout, no direct subprocess calls. Pytest fixture: function-scoped `chaos_injector` in [`conftest.py:38-41`](conftest.py).

### 2a. Generic helpers (drop-in for any cluster)

- `wait_for_pod_status_reason(namespace, label_selector, reason, timeout)` — poll `containerStatuses[*].state.waiting.reason` (and init-container statuses) for an exact match. Useful for `ImagePullBackOff`, `ErrImagePull`, `CrashLoopBackOff`, `CreateContainerError`. Returns the first pod name to surface the reason.
- `wait_for_container_restart(pod, container, namespace, *, since_count, timeout)` — snapshot `restartCount` before fault injection, then block until it advances. The standard idiom for "kill the container, prove it actually restarted." Returns the new restart count.
- `wait_for_cr_gone(namespace, name, timeout)` — block until a named CR disappears from the apiserver (not just "Terminating"). Returns wall-clock seconds elapsed. Currently hardcoded to `aiperfjob` kind but trivially generalizable.
- `wait_for_pods_gone(namespace, timeout)` — block until every pod in a namespace is reaped.
- `apply_resource_quota(namespace, name, hard_limits)` — apply an arbitrary `ResourceQuota` for K3-style quota-exhaustion tests.
- `delete_resource_quota(namespace, name)` — idempotent NotFound-swallowing delete, safe for unconditional `finally`.

### 2b. In-container fault injection

- `kill_container_in_pod(namespace, pod, container)` — `kubectl exec <pod> -c <container> -- sh -c 'kill -KILL 1'`. Kills only the target container; kubelet decides restart per `restartPolicy`. Generic kubectl pattern, no AIPerf assumptions in the helper itself.
- `kill_container_by_pid(pod, container_pid, namespace, exec_container)` — cross-container kill via shared PID namespace. Exec into a sidecar and issue `kill -9 <pid>` against a process in another container of the same pod. **Requires** the pod to have `spec.shareProcessNamespace: true`. *Generic mechanism, AIPerf-shaped only in that the caller must already know the target PID.*

### 2c. AIPerf-specific helpers (illustrative, not portable as-is)

The following are real and useful inside the AIPerf suite but assume AIPerfJob CRD shape, JobSet labels, or operator-internal annotations. Listed for completeness:

- `delete_cr_no_wait(namespace, name)`, `delete_cr_twice(namespace, name)` — hardcoded `aiperfjob` kind.
- `kill_operator_pod(force)` — hardcoded `aiperf-system` namespace and `app.kubernetes.io/name=aiperf-operator` selector.
- `stamp_completion_claim(namespace, name, timestamp_iso)`, `read_claim_annotation(namespace, name)` — read/write `aiperf.nvidia.com/completion-claimed` annotation.
- `wait_for_phase(namespace, name, phases, *, current_phase)` — `.status.phase` + `.status.currentPhase` composite waiter; the *idea* (composite waiter on phase + sub-phase) is generic, the field paths are not.
- `get_controller_pod_name(namespace, job_name)`, `get_worker_pod_names(namespace, job_name)` — hardcoded JobSet `replicatedjob-name` label and `aiperf-<job>` naming.
- `get_mock_server_pod_name(namespace, deployment)` — `app=aiperf-mock-server` selector.
- `wait_for_operator_ready(timeout)` — block until every container of the operator pod reports Ready; hardcoded `aiperf-system` namespace and operator selector.
- `create_invalid_cr(namespace, name, spec_patch)` — builds a minimal AIPerfJob manifest and applies it; the *idea* (apply a malformed CR via patch overlay) is generic, the embedded manifest is not.

### 2d. Timing dataclass

`ChaosTimings` — frozen dataclass with named timeouts (`cr_cleanup_seconds`, `pod_termination_grace`, `operator_recovery_seconds`, `completion_wait_seconds`) and one-line docstrings explaining each. Avoids magic numbers across tests. *Generic pattern, the values themselves are AIPerf-tuned.*

## 3. ToxiproxyInjector (network faults)

[`toxiproxy.py`](toxiproxy.py) — async REST client for a cluster-deployed [Toxiproxy](https://github.com/Shopify/toxiproxy) Deployment + Service. *Generic.*

**Fixture manifest.** [`fixtures/toxiproxy.yaml`](fixtures/toxiproxy.yaml) deploys Toxiproxy into namespace `aiperf-chaos-toxiproxy` with the admin API on `:8474` and a small pool of named listen ports (`proxy-0`..`proxy-5` on 20000-20005, plus `mock-server` on 20010) for per-test proxies. Cluster-internal DNS: `toxiproxy.aiperf-chaos-toxiproxy.svc`.

**Reserved listen ports** (stable across reruns so operator env-var overrides are deterministic):

| Port | Purpose |
|------|---------|
| 20000 | Operator -> apiserver (C15) |
| 20002 | Operator -> controller HTTP (C16) |
| 20010 | Benchmark -> mock-server (B3 latency injection) |
| 20001, 20003-20005 | Unreserved generic slots |

**Fixture lifecycle.** Package-scoped `toxiproxy_injector` in [`conftest.py:44-61`](conftest.py).

1. `ensure_deployed(kubectl)` — apply manifest, wait for rollout, open `kubectl port-forward` to admin API, probe `/version`.
2. Per-test: `add_proxy` + `add_toxic`, run assertions, `reset()` in `finally`.
3. `teardown(kubectl)` at package end — close port-forward, delete namespace.

**Methods on `ToxiproxyInjector`:**

- `add_proxy(name, listen, upstream)` — create a TCP proxy.
- `add_toxic(proxy_name, toxic_type, attributes, *, name, stream, toxicity)` — attach a toxic (`latency`, `timeout`, `bandwidth`, `slow_close`, `reset_peer`, etc.) with optional `stream` (`upstream`/`downstream`) and partial `toxicity`.
- `remove_toxic(proxy_name, toxic_name)`, `remove_proxy(name)` — targeted deletes.
- `reset()` — wipe every proxy and toxic. Idempotent; safe in `finally`.
- `teardown(kubectl)` — close port-forward and delete namespace.

**Non-obvious invariants** (documented in the module docstring, worth knowing if you port this):

- Every REST method opens its own short-lived `aiohttp.ClientSession`. Caching a session on the instance breaks under pytest-asyncio because the session binds to the loop it was created on, and chaos fixtures are package-scoped while individual tests run on function loops.
- `reset()` uses `DELETE /proxies/<name>` per proxy, not Toxiproxy's `POST /reset`. The latter re-enables proxies rather than removing them.
- The kubectl port-forward subprocess is kept alive across event loops, only the TCP port it binds is used by short-lived sessions.

## 4. Composite fixtures (toxiproxy-routed operator)

[`conftest.py:133-265`](conftest.py) — function-scoped fixtures that combine the toxiproxy injector with an operator redeploy so test code doesn't have to re-render the Deployment manifest. Both restore a plain (non-routed) operator on teardown even under `--k8s-skip-cleanup`, because the env overrides are shared mutable cluster state. *Mixed: mechanism is generic, env-var names are AIPerf-specific.*

- `operator_ready_toxiproxy_routed` — operator redeployed with `AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE` pointed at `http://toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local:20002`, so operator -> controller HTTP traverses Toxiproxy.
- `operator_ready_apiserver_toxiproxy_routed` — operator redeployed with `KUBERNETES_SERVICE_HOST` / `KUBERNETES_SERVICE_PORT` pinned at the Toxiproxy Service, `AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE=kubernetes.default.svc` to preserve SNI through the TCP proxy. The apiserver proxy is created *before* operator deployment because kopf logs in immediately on startup.

Both fixtures call a private `_assert_live_operator_env` precondition that diffs `kubectl set env deployment/aiperf-operator --list` against the expected env-var map and fails loudly when the override didn't land. Catches "redeployed but the env wasn't applied" without waiting for the assertion to time out 90 s later.

## 5. MockServerInjector (upstream-service faults)

[`mock_server_injector.py`](mock_server_injector.py) — chaos against the single-replica `aiperf-mock-server` Deployment that serves benchmark traffic in the k8s test harness. *Mixed: idiom is generic, default deployment name (`aiperf-mock-server`) and namespace (`default`) are AIPerf harness conventions.*

**Methods on `MockServerInjector`:**

- `restart(namespace, deployment)` — `kubectl rollout restart` for a rolling kill.
- `delete_pod(namespace, deployment)` — `--grace-period=0 --force` for ungraceful crash (skips SIGTERM window).
- `scale(namespace, replicas, deployment)` — change replica count; records prior count for restore.
- `patch_env(namespace, env_var, value, deployment)` — `kubectl set env` to inject behavior (e.g. `AIPERF_MOCK_FORCE_STATUS=500` to force 5xx responses).
- `restore()` — reverse every mutation applied during the test in LIFO order. Called automatically by the fixture teardown.

**Pattern worth porting.** Internally the class tracks an `_applied_ops: list[_AppliedOp]` and rolls back in `restore()` based on op kind (`env`, `scale`, `restart-annotation`). Each state-mutating fault method appends one entry; `delete_pod` appends none because the Deployment controller owns pod re-creation and there is nothing to unwind. Tests never track restore state themselves. Generic and worth lifting verbatim for any "perturb a Deployment in tests" use case.

## 6. Helm chart integration

[`deploy/helm/aiperf-operator/values.yaml:120-128`](../../../deploy/helm/aiperf-operator/values.yaml) ships `podTemplate.shareProcessNamespace: false` (production default), which [`templates/deployment.yaml`](../../../deploy/helm/aiperf-operator/templates/deployment.yaml) renders into the operator's `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` env var. When true, JobSet pods are rendered with `Pod.spec.shareProcessNamespace: true`, enabling cross-container `kill` via `kubectl exec` (Section 2b).

The opt-in-for-chaos, default-off-in-production tension is the load-bearing piece. *Generic pattern, AIPerf-specific env-var name.*

## 7. Documented scenarios (taxonomy)

The full scenario table lives in [`README.md`](README.md). The *taxonomy* — useful as a checklist when planning chaos coverage for any operator — is:

- **Cancellation & idempotence** — delete mid-state, rapid double-delete, recreate-same-name after fault.
- **Operator resilience** — operator pod kill mid-benchmark, recovery from durable mid-state markers left across operator restart.
- **Workload pods** — controller-container kill, worker-pod kill, sidecar restart, in-flight fetch during restart.
- **Helm lifecycle** — install -> run -> uninstall -> reinstall idempotence, mid-job upgrade, invalid values rollback, missing dependency CRDs.
- **Churn** — rapid create/delete cycles, parallel-jobs partial-delete stability, invalid-spec surfacing.
- **API disruption** — apiserver pause (TLS-preserving toxiproxy), intra-operator HTTP blackhole.
- **Workload runtime** — upstream 5xx burst, upstream Deployment restart, latency injection.
- **Infrastructure** — image-pull failures, DNS failures, namespace quota exhaustion.
- **Sweep controller** — kill the sweep-controller pod mid-sweep and assert the restarted pod resumes rather than re-running completed variations.

## 8. Bug shapes chaos runs have surfaced

Past chaos sessions found (and led to fixes for) recurring bug *shapes*: stale imports on rarely-exercised recovery paths, undersized memory limits that work in steady state but OOM during cleanup, RBAC gaps that 403 silently on cross-namespace recovery, and durable mid-state markers that leak when the marker-holder dies. Useful as pattern recognition for "what kinds of bugs k8s chaos surfaces."

## 9. Known gaps

- **PID discovery lives in a test module, not the injector.** `kill_container_by_pid` requires the caller to pass `container_pid: int`, so every cross-container kill first needs a PID. The only implementation of that lookup is the module-level `_find_pid_by_cmdline` helper in [`test_chaos_jobset_pods.py`](test_chaos_jobset_pods.py), which walks `/proc/[0-9]*/cmdline` through the shared PID namespace. Nothing in [`chaos_injector.py`](chaos_injector.py) exposes it, so a new scenario in another module has to import the private helper or reimplement the walk. Promoting it to `ChaosInjector` would close the gap.
- **`kill_container_by_pid`'s docstring points at a tool the runtime image lacks.** It suggests obtaining the PID via `kubectl exec <pod> -c <any> -- pgrep -n <name>`; the distroless-python runtime image has no `pgrep`, which is exactly why `_find_pid_by_cmdline` exists.
