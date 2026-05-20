# Dynamo-Specific Chaos Tests — Net-New Coverage

**Status:** Draft
**Date:** 2026-05-19
**Author:** Anthony Casagrande
**Scope:** The subset of the [D-series chaos suite](2026-05-19-dynamo-chaos-suite-design.md) that adds **genuinely new** coverage to dynamo — tests that dynamo's existing `tests/fault_tolerance/` suite does **not** cover today.

This document explicitly excludes D-series scenarios that overlap with dynamo's existing coverage (worker-migration with KV recovery, socket-level request cancellation, single-node etcd restart, GPU XID injection, SIGSTOP-rank-pause). For the full catalog including overlapping scenarios, see the D-series design doc.

## What dynamo already covers (excluded from this doc)

`/tmp/dynamo/tests/fault_tolerance/` ships these capabilities; tests overlapping them are out of scope here.

| Existing coverage | Path | Class of fault |
|---|---|---|
| Worker migration with KV-state recovery | `migration/utils.py` | `ManagedProcess` SIGKILL + checkpoint replay |
| Request cancellation mid-stream | `cancellation/utils.py` | Socket monkey-patching, mid-flight tear-down |
| Multi-node etcd HA (clean restart) | `etcd_ha/utils.py` (`EtcdCluster.terminate_replica`, `restart_replica`) | Single-node restart, leader change |
| Rank SIGSTOP / canary detection | `test_canary_rank_pause.py:196` (raw `os.kill`) | Engine-rank pause, hang detection |
| GPU XID hardware faults | `hardware/fault_injection_service/agents/gpu_fault_injector/agent.py:218` (per-node DaemonSet via `nsenter`+`kmsg`) | 27 pre-defined XIDs |
| Deployment-level pod restart scenarios | `deploy/scenarios.py` (`Failure(ABC)` hierarchy) | `kubectl delete pod`, restart-count assertions |

These are mature, upstream-maintained, and stay where they are. Our new tests **complement** them — they assert on a different layer (operator + CR-status surface, network plane, KV-transport plane, KV-router internals) that no `ManagedProcess`-based in-process test can reach.

## The eight gap-categories

Per the earlier coverage survey of dynamo's existing suite, the gaps are:

1. **No network-layer fault injection** — no Toxiproxy, no iptables, no NetworkPolicy chaos. Only application-level cancellation.
2. **No etcd corruption or split-brain** — `etcd_ha/` only does clean restarts.
3. **No asymmetric mixed-failure topologies** — single-worker crash, never "prefill alive, decode dying mid-batch."
4. **No slow-death / cascading-timeout scenarios** — workers crash fast or drain gracefully; nothing tests the in-between.
5. **No host-level resource starvation** — GPU XID exists, but no CPU throttling, disk-I/O saturation, host-memory pressure, or VRAM contention from a sidecar.
6. **No CRD / operator-status assertions** — existing tests assert against runtime internals and pod restart counts; nothing asserts `DynamoGraphDeployment.status.state` / `.status.conditions` transitions.
7. **No invalid-spec / validation-webhook testing.**
8. **No HF-Hub egress blackhole during weight load.**

The net-new tests below close each gap. Each section names the dynamo file:line that the test asserts against, the inject technique (file:line in the proposed chaos suite), and what the test would catch that no existing dynamo test catches.

---

## Gap 1 — Network-layer fault injection

**Why this is the highest-value gap:** dynamo's discovery (etcd), control plane (NATS), KV transport (NIXL side-channel), inter-rank coordination (NCCL), and OpenAI-compatible HTTP frontend all depend on network connectivity that *behaves perfectly* in every existing pytest. Toxiproxy lets us assert on degraded-but-not-broken: latency spikes, partial blackholes, mid-stream `reset_peer`, bandwidth caps, slow-close.

> **Nuance:** `tests/fault_tolerance/hardware/fault_injection_service/api_service/main.py:70, :513-814` ships a `NetworkPolicy`/ChaosMesh-backed network-fault injector that *can* block NATS, drop packets, add latency, etc. — but no pytest under `tests/fault_tolerance/` currently invokes it (`grep -rn fault_injection_service tests/ --include=test_*.py` returns empty). The infrastructure exists, the test coverage doesn't.

### D802 — etcd 30 s pause via Toxiproxy

- **Fault:** Inject a `timeout: 0` toxic on the etcd Service (`<release>-etcd:2379`) for 30 s, then heal.
- **Tests against:** Lease keep-alive at `lib/runtime/src/transports/etcd/lease.rs:136` (TTL/2 heartbeat), discovery lease default 60 s (`kvbm-config/src/discovery.rs:122`).
- **Assertion:** Frontend serves stale roster, then recovers; lease-expiry latency bounded by 60 s + 30 s timeout = ~90 s. No permanent outage. `dynamo_frontend_disconnected_clients` does not run away.
- **What dynamo doesn't cover:** `etcd_ha/` does only clean process restart. There is no test of the 30 s-degraded-but-reachable state, which is the realistic failure mode in production (apiserver flakes, network partition).

### D803 — NATS kill mid-traffic

- **Fault:** `kubectl delete pod -l app=nats --force --grace-period=0` under 8 concurrent SSE streams.
- **Tests against:** `lib/runtime/src/transports/nats.rs:49` (Client struct). No explicit `reconnect_buffer_size` / `retry_on_initial_connect` overrides exist; reconnect is whatever `async_nats` defaults to.
- **Assertion (refines D-series catalog):** Catalog says "KV-router load metrics go stale but routing falls back to round-robin; no crash." This net-new test refines that with concrete bounds — error rate <20% during outage, <5% after recovery; `dynamo_component_router_*` metrics flow again within 30 s of NATS restart.
- **What dynamo doesn't cover:** No NATS-side pytest exists in `tests/fault_tolerance/`. (`hardware/fault_injection_service/api_service/main.py:814` *does* ship a ChaosMesh-based NATS-block injector via `target_nats` parameter — but no pytest currently calls it.) NATS is the load-metric and KV-router-event bus; silent failure here would degrade routing quality without crashing anything.

### D804 — NATS slow-close toxic on stats subjects

- **Fault:** Toxiproxy `slow_close` toxic on the NATS service for 60 s.
- **Tests against:** Service-stats scrape subject (`$SRV.STATS.<service>`).
- **Assertion:** Service-stats requests time out cleanly; no permanent socket leak (assert against `dynamo_component_router_overhead_total_ms` histogram).
- **What dynamo doesn't cover:** No latency-class network faults exist anywhere in dynamo's suite.

### D203 — Backend stream inactivity timeout

- **Fault:** Toxiproxy with 5 s upstream `latency` exceeding `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`.
- **Tests against:** `lib/llm/src/http/service/disconnect.rs:195` (`monitor_for_disconnects` declaration; body runs through line 260).
- **Assertion:** Stream RST observed; `ErrorType::ResponseTimeout` metric increments; client receives clean termination not infinite hang.
- **What dynamo doesn't cover:** SSE timeout behavior is only tested via mock servers in dynamo's existing suite, not via real network-latency injection.

### D805 — DynamoWorkerMetadata watch RBAC revoked

- **Fault:** `kubectl patch role` denying `watch` on `dynamoworkermetadatas` CRD; assert reflector backoff behavior.
- **Tests against:** `lib/runtime/src/discovery/kube/daemon.rs:204` (EndpointSlice setup; the `DynamoWorkerMetadata` watch sits a few lines below).
- **Assertion:** Reflector backoff kicks in; snapshot ages but existing entries are not dropped (cache-validation at `daemon.rs:203, 249` masks the brief outage).
- **What dynamo doesn't cover:** No Kubernetes-discovery-backend fault testing exists (only etcd-backend HA tests).

### D806 — Worker lease keep-alive blackholed

- **Fault:** Toxiproxy `bandwidth: 0` on the etcd keepalive port for one worker only.
- **Tests against:** `lease.rs:179-188` (heartbeat loop), `lease.rs:104-109` (deadline cascade).
- **Assertion:** After 60 s lease TTL, that worker's endpoints auto-delete from discovery; frontend stops routing to it within ~90 s; other workers unaffected.
- **What dynamo doesn't cover:** Selective-worker partition is not in `etcd_ha/`; that suite only does full-cluster operations.

---

## Gap 2 — etcd corruption / split-brain

### D801 — etcd kill during decode-worker registration race

- **Fault:** `kubectl delete pod -l app.kubernetes.io/name=etcd -n dynamo-system --force` while a fresh decode worker is mid-registration.
- **Tests against:** Worker bootstrap path `components/src/dynamo/vllm/main.py` → `register_model` → `serve_endpoint` (in `worker_factory.py:398, 433`). Lease grant at `lease.rs:21`.
- **Assertion:** Worker either retries to registration success within the ~90 s lease-TTL window or fails cleanly (CrashLoopBackOff with clear error); no half-registered state where the worker is in the discovery roster but not actually serving.
- **What dynamo doesn't cover:** The registration race is a startup-time concern. `etcd_ha/` exercises steady-state cluster behavior, not the registration window.

### D807 — NATS partition split-brain between frontend replicas

- **Fault:** Toxiproxy partition between two frontend replicas' NATS connections.
- **Tests against:** Frontend stats scraping; KV-router state synchronization across replicas.
- **Assertion:** Both replicas eventually converge to the same worker view after the partition heals; no permanent disagreement.
- **What dynamo doesn't cover:** Split-brain semantics across stateless frontends are unverified. Dynamo's existing tests run a single frontend.

---

## Gap 3 — Asymmetric mixed-failure topologies

### D301 — NIXL `reset_peer` mid-KV-handoff (disagg only)

- **Fault:** Toxiproxy `reset_peer` toxic on `VLLM_NIXL_SIDE_CHANNEL_PORT` (stamped per-engine as `5600 + engineID` in `failover_vllm.go:35`) during an active prefill→decode handoff.
- **Tests against:** `lib/kvbm-physical/src/transfer/notifications/nixl_status.rs:30` ("NIXL transfer status check failed"); KV-transfer protocol in `lib/kvbm-{logical,physical,engine,consolidator}/`.
- **Assertion:** Frontend surfaces 500 + structured error per the two-frame SSE contract at `lib/llm/src/http/service/disconnect.rs:225-239` (error JSON frame + `data: [DONE]`; `ErrorMessage` JSON shape defined at `openai.rs:90-93`); decode pod does not restart-loop; `dynamo_component_errors_total{error_type="response_stream"}` increments by exactly the affected request count.
- **What dynamo doesn't cover:** dynamo's migration test (`migration/utils.py`) kills the *whole* worker process. It does not test a *transport-level* fault between prefill and decode while both processes remain alive — the exact asymmetric failure mode disagg deployments hit in practice.

### D304 — KVBM consolidator ZMQ socket close

- **Fault:** Patch the consolidator env to break its ZMQ socket binding; restart the container.
- **Tests against:** `lib/kvbm-consolidator/src/wire/` (ZMQ egress, KV-cache event dedup).
- **Assertion:** Prefill side logs lost-event warnings; decode reconnects within 30 s of consolidator restart; total KV-block dedup state is consistent (no orphan blocks).
- **What dynamo doesn't cover:** The KVBM consolidator is not separately fault-tested. Worker-migration kills the whole pod, never just the consolidator socket.

### D403 — Lost-block-deallocation memory leak (KV-router internal)

- **Fault:** Force a free event for a worker that was just removed from the topology (delete pod + send a request that completes elsewhere).
- **Tests against:** `lib/kv-router/src/sequences/multi_worker.rs:346-366` (replica-sync Free handler; lookups the request_index, silently drops on stale state).
- **Assertion:** `dynamo_component_inflight_requests` drains; no router OOM after N cycles of pod-recreate.
- **What dynamo doesn't cover:** KV-router internal state is opaque to dynamo's existing tests. The replica-sync bug (silent drop of Free events) would only surface as gradual memory growth, which no existing test asserts on.

---

## Gap 4 — Slow-death / cascading-timeout scenarios

### D303 — NIXL 60 s transfer stall (observational)

- **Fault:** Toxiproxy `bandwidth: 1KBps` on NIXL side-channel port.
- **Tests against:** `lib/kvbm-physical/src/transfer/notifications/mod.rs:54-75` (1 ms poll, 60 s warning, 30 s re-warning, **no hard timeout**).
- **Assertion (observational, not pass/fail):** Warning logs appear at 60 s, 90 s, 120 s. The test exists to surface the known no-hard-timeout gap to the maintainers; it ratchets the observed behavior, not the desired behavior.
- **What dynamo doesn't cover:** No test exists for "slow but not stopped" KV transfer. Dynamo's migration test only handles the binary "stopped" case.

### D405 — Prefill-complete event-loss queue deadlock

- **Fault:** Block the worker→router watch channel briefly (Toxiproxy short pause on the metrics path).
- **Tests against:** `lib/kv-router/src/scheduling/local.rs:157-180` (remote state listener; watch channel uses bounded buffer).
- **Assertion:** Queue drains after the channel recovers; no permanent deadlock. (If queue stays stuck, that's a real bug.)
- **What dynamo doesn't cover:** The KV-router queue's recovery semantics under transient backpressure are uncovered. Existing tests run with a healthy event bus.

### D404 — Pending-queue unbounded growth (router OOM)

- **Fault:** Set `threshold_frac` so all workers throttle; submit 10× normal load.
- **Tests against:** `lib/kv-router/src/scheduling/queue.rs:62` (BinaryHeap, no max size).
- **Assertion:** Queue grows but router does not OOM within 2 min. (Known gap — heap is unbounded; assertion exists to bound the regression surface.)
- **What dynamo doesn't cover:** Queue memory growth under sustained overload is not tested.

---

## Gap 5 — Host-level resource starvation

### D505 — Decode worker OOM via VRAM-pressure sidecar

- **Fault:** Inject a sidecar container running a CUDA stub that allocates above the worker's `--gpu-memory-utilization` headroom.
- **Tests against:** `HostCacheConfig` (G2 pinned-memory pool sizing) at `lib/kvbm-config/src/cache.rs:46-88`; the actual eviction policy is in `kvbm-physical`/`kvbm-engine`, but the test exercises the config surface (set tiny `DYN_KVBM_CPU_CACHE_GB`, observe behavior at the eviction layer indirectly via metrics and worker OOM).
- **Assertion:** Worker OOMs cleanly (kubelet OOMKilled), restarts, traffic recovers within 60 s. Does not segfault or hang.
- **What dynamo doesn't cover:** GPU memory exhaustion *from outside the worker* is not tested. Dynamo's tests run workers at their advertised memory budget, never under contention from a noisy neighbor on the same GPU.

### D305 — KVBM CPU cache pre-fill to capacity

- **Fault:** Set `DYN_KVBM_CPU_CACHE_GB` to a tiny value, push enough traffic to fill it.
- **Tests against:** `HostCacheConfig` (G2 pinned-memory pool sizing) at `lib/kvbm-config/src/cache.rs:46-88`; eviction itself lives downstream in `kvbm-physical`/`kvbm-engine`. Pinned-memory pressure observable via `/proc/<pid>/status` VmPin.
- **Assertion:** G3 (disk) spill triggers; no `cudaMallocHost()` OOM; latency spikes but no crash.
- **What dynamo doesn't cover:** Cache-pressure scenarios are not exercised. Existing tests run with default cache sizes that never fill.

### D703 — Namespace ResourceQuota exhausted

- **Fault:** Apply a `ResourceQuota` capping memory below the worker request.
- **Tests against:** Admission-time scheduling on DGD apply.
- **Assertion:** DGD reaches `state=failed` within 120 s; reason names the quota rejection; quota deleted in `finally`.
- **What dynamo doesn't cover:** Resource-quota admission failures are not surfaced by any existing test.

---

## Gap 6 — CRD / operator-status assertions

This is the entire D1xx family. **Every** test here is net-new — dynamo's existing suite has zero coverage of the operator-status surface (`DynamoGraphDeployment.status.state`, `.status.conditions`, `.status.components[]`). Existing tests assert on `kubectl get pods` and runtime metrics; nothing reads `.status` of the parent CR.

### D101 — Kill operator pod mid-DGD-apply

- **Fault:** `kill_operator_pod(force=True)` while applying a fresh DGD.
- **Tests against:** Reconcile loop at `deploy/operator/internal/controller/dynamographdeployment_controller.go:119`.
- **Assertion:** DGD reconciles to `status.state=successful` after operator restart; `status.observedGeneration` matches `metadata.generation`; no orphan child resources (Deployments, Services, ConfigMaps).
- **What dynamo doesn't cover:** Operator-pod resilience is not tested at all. Dynamo's tests assume the operator is healthy.

### D102 — Rapid double-delete DGD is idempotent

- **Fault:** `kubectl delete dgd <name>` twice within 1 s.
- **Tests against:** Finalizer path; child-resource cleanup.
- **Assertion:** `status.state=…` → CR-gone within 60 s; no stuck finalizer.
- **What dynamo doesn't cover:** Finalizer idempotency is not tested.

### D103 — Operator kill during failover-cascade

- **Fault:** Kill operator while a pod is `Failed`/`Terminating`; assert `FailoverCascadeReconciler.Reconcile` (`failover_cascade_controller.go:75`) resumes.
- **Tests against:** Cohort recreation logic in failover-cascade controller.
- **Assertion:** `spec.components[?(@.name=="<x>")].replicas` returns to spec after restart; per-component replica counts in `status.components` recover.
- **What dynamo doesn't cover:** Cascade-mid-flight resilience is not tested.

### D104 — Invalid DGD spec surfaces in status

- **Fault:** Apply DGD with `spec.components[?(@.name=="Frontend")].replicas: -1` or missing required component name.
- **Tests against:** Validation webhook + reconciler error handling.
- **Assertion:** `status.state=failed`; `status.conditions[?type=Ready].status=False`; reason names the validation failure.
- **What dynamo doesn't cover:** No invalid-spec tests exist.

### D105 — Rapid create-delete-create-same-name

- **Fault:** Apply, delete, re-apply within 5 s, same name.
- **Tests against:** Finalizer tombstone handling.
- **Assertion:** New DGD reaches `state=successful` despite the prior tombstone.
- **What dynamo doesn't cover:** No same-name-reuse test exists.

### D106 — Webhook validation pod down

- **Fault:** Scale validator webhook deployment to 0 mid-traffic.
- **Assertion:** Apply fails fast or is admitted-then-rejected; no half-created children.
- **What dynamo doesn't cover:** Webhook-availability scenarios are not tested.

### D107 — Operator RBAC revoked mid-reconcile

- **Fault:** `kubectl patch clusterrole` to drop `deployments: patch`.
- **Assertion:** Reconcile errors; CR `status.state` accurately reflects the failure (does not falsely flip to successful).
- **What dynamo doesn't cover:** RBAC degradation paths are not tested.

### D108 — Apiserver pause 30 s during reconcile

- **Fault:** Toxiproxy on apiserver with `timeout` toxic via `KUBERNETES_SERVICE_HOST` override (port 20000 reserved).
- **Tests against:** Operator's apiserver-client resilience.
- **Assertion:** Operator survives the pause, reconcile resumes, DGD reaches `successful` after the pause heals.
- **What dynamo doesn't cover:** Apiserver-side faults are not tested.

---

## Gap 7 — Invalid-spec / validation-webhook testing

Already enumerated under Gap 6 (D104, D106). These are net-new in their entirety.

---

## Gap 8 — HF Hub egress blackhole

### D704 — Block HF Hub during weight load

- **Fault:** NetworkPolicy denying egress to `0.0.0.0/0` except cluster CIDR; or an init-container writes `127.0.0.1 huggingface.co` to `/etc/hosts` via emptyDir share.
- **Tests against:** Worker bootstrap weight-load path; `components/src/dynamo/vllm/main.py:122-123` (`await fetch_model(config.model)`).
- **Assertion:** Worker surfaces a clean "weight-download failed" status, **not** opaque `CrashLoopBackOff` with no actionable error. DGD reaches `state=failed` with a reason naming the network failure.
- **What dynamo doesn't cover:** No HF-Hub-unreachable test exists. This is a real production failure mode (corporate firewalls, HF Hub outages) and the current UX on failure is opaque.
- **Dependency:** Needs a NetworkPolicy-aware CNI; kindnet does not honor NetworkPolicy. Run against Cilium-on-kind or a real cluster.

### D701 — ImagePullBackOff surfaces in CR

- **Fault:** Apply DGD with a non-existent container image (e.g. `image: "does-not-exist:nope"`).
- **Tests against:** Operator reconcile + child-pod status propagation; `dynamographdeployment_controller.go:119` (Reconcile) reads child-pod state and writes `.status.state`.
- **Assertion:** DGD reaches `state=failed` within 120 s of the kubelet pull failure; reason names `ImagePullBackOff` or `ErrImagePull`; pod surfaces in `containerStatuses[*].state.waiting.reason`.
- **What dynamo doesn't cover:** Infra-failure → CR-status propagation is not asserted by any existing test. Dynamo's tests assume images pull cleanly. This is the lowest-cost smoke test that catches "reconciles forever on a doomed pod" regressions.

---

## Gap 2/3 hybrid — KV-router-specific net-new

The D4xx family deserves a separate callout because the KV-router (`lib/kv-router/`) is the most dynamo-distinctive subsystem and has **zero** existing fault coverage in `tests/fault_tolerance/`. Each scenario below is genuinely new ground.

### D401 — Kill decode worker mid-request (no retry path)

- **Fault:** Kill the worker selected by the router during active generation.
- **Tests against:** `lib/kv-router/src/scheduling/queue.rs:173-210` (no retry logic after admission — documented gap).
- **Assertion:** Client gets clean error (not infinite hang); P99 of error-latency stays within configured timeout.
- **What dynamo doesn't cover:** Worker death mid-request from the *router's* perspective. Migration test handles worker death from the *worker's* perspective.

### D402 — Split-brain on rapid topology churn

- **Fault:** Rapid `kubectl delete pod` + recreate of a decode worker.
- **Tests against:** `lib/kv-router/src/sequences/multi_worker.rs:307-343` (replica-sync AddRequest race).
- **Assertion:** No duplicated slot entries; router state matches actual roster within 30 s.
- **What dynamo doesn't cover:** Topology-churn race conditions are not tested.

### D406 — Pinned-worker head-of-line block

- **Fault:** Submit a pinned request to an overloaded worker, then non-pinned requests behind it.
- **Tests against:** `lib/kv-router/src/scheduling/queue.rs:246-249` (documented TODO).
- **Assertion (observational):** Ratchet the observed HoL-block latency; test exists to detect regression, not to validate a fix.
- **What dynamo doesn't cover:** Documented but unmeasured router behavior.

---

## Frontend / SSE net-new

### D201 — Force-kill Frontend pod under 64 concurrent SSE streams

- **Fault:** `kubectl delete pod --force --grace-period=0` on the frontend pod.
- **Tests against:** `lib/llm/src/http/service/disconnect.rs:212` (`monitor_for_disconnects`) cleanup.
- **Assertion:** Clients see TCP RST or HTTP 503; the next replica serves immediately; no zombie traffic; `dynamo_frontend_disconnected_clients` increments by the expected count.
- **What dynamo doesn't cover:** Frontend force-kill under load is untested. Existing tests run with `--grace-period` and don't measure client-side observables.

### D202 — Scale Frontend 1→0→1 mid-traffic

- **Assertion:** New requests succeed within 30 s of recovery; no permanent 503.

### D207 — Streaming error frame format

- **Fault:** Kill upstream worker mid-stream.
- **Tests against:** Mid-stream error path at `disconnect.rs:225-239` (two-frame emission: error JSON event then `data: [DONE]`).
- **Assertion:** Client receives `{"error":{"message":...,"type":"internal_server_error","code":500}}` SSE frame **then** `data: [DONE]` (two-frame contract; `ErrorMessage` JSON shape at `openai.rs:90-93`).
- **What dynamo doesn't cover:** The mid-stream error-frame format is unverified by any test.

---

## What's *not* in this doc

Scenarios that overlap dynamo's existing coverage and are intentionally omitted here:

- D501 (kill worker; restart-count advances) — covered by `deploy/scenarios.py` `DeletePodFailure`.
- D502 (kill worker between register and serve) — partial overlap with migration tests.
- D504 (engine SIGSTOP) — direct overlap with `test_canary_rank_pause.py`.
- D706 (DCGM XID injection) — direct overlap with `hardware/fault_injection_service/`.
- D601 through D606 (MPI multinode) — partial overlap with existing multi-node deployment tests.

If we end up writing D501/D502/D504 anyway, the value-add is asserting on **CR-status surface** (which the existing tests don't), not on the underlying mechanism (which they already do).

## Wave-0 net-new subset

Of the ten Wave-0 scenarios in the D-series doc, **all ten** are net-new to dynamo:

| ID | Net-new because |
|---|---|
| D803 — NATS kill | dynamo has no NATS fault testing |
| D802 — etcd 30 s pause | `etcd_ha/` only does clean restart |
| D801 — etcd kill during registration | startup-race not exercised |
| D301 — NIXL `reset_peer` | KV transport plane uncovered |
| D401 — kill decode worker mid-request | router perspective uncovered |
| D201 — force-kill Frontend under SSE | frontend force-kill uncovered |
| D101 — kill operator mid-DGD-apply | operator-status surface uncovered |
| D704 — HF Hub egress blackhole | network egress block uncovered |
| D104 — invalid DGD spec | no invalid-spec tests |
| D701 — ImagePullBackOff | infra failure mode uncovered |

Every Wave-0 scenario closes a gap that no existing dynamo test can close. This is the strongest single argument for shipping Wave-0 in our repo even if we later upstream to dynamo: the value is immediate.

## Acceptance criteria for net-new coverage

A test counts as "genuinely new" if:

1. The fault is not already injected by any test under `/tmp/dynamo/tests/fault_tolerance/` or `/tmp/dynamo/tests/utils/`.
2. **OR** the assertion targets a layer (CRD status, KV-router internal state, NIXL transport, NATS metrics path, operator-status conditions) that no existing dynamo test asserts on.
3. **OR** the timing/scale (mid-stream, mid-handoff, mid-registration) is not exercised by any existing test.

Every test in this document satisfies at least one of these. Tests that satisfy zero (D504, D706, partial D5xx) are correctly excluded.

## Summary count

| Family | Net-new tests | Coverage gap closed |
|---|---:|---|
| D1xx (operator) | 8 (D101–D108) | Gap 6 (CRD status) entire |
| D2xx (frontend) | 4 (D201, D202, D203, D207) | Gap 1, gap 4 partial |
| D3xx (KV transport) | 4 (D301, D303, D304, D305) | Gaps 3, 4, 5 |
| D4xx (KV-router) | 6 (D401, D402, D403, D404, D405, D406) | Gaps 3, 4 (router-specific) |
| D5xx (workers) | 1 (D505 VRAM pressure) | Gap 5 |
| D7xx (infra) | 3 (D701, D703, D704) | Gaps 5, 6, 8 |
| D8xx (state store) | 7 (D801–D807) | Gaps 1, 2 |
| **Total** | **33** | All 8 gaps |

This is the net-new figure quoted earlier, in concrete terms: **33 of the 54 D-series scenarios (~61%)** add coverage dynamo doesn't have today.
