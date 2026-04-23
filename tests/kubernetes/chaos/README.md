<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Operator Chaos Suite

Fault-injection tests for the kopf operator that back the AIPerfJob CRD. This suite sits alongside `tests/kubernetes/` and reuses its cluster/operator fixtures; it lives under `tests/kubernetes/chaos/` so the standard `K8S_TEST_*` flags (Kind cluster, image loading, mock server) all apply unchanged.

Design documents:
- [`docs/superpowers/specs/2026-04-23-operator-chaos-testing-design.md`](../../../docs/superpowers/specs/2026-04-23-operator-chaos-testing-design.md) — v1 scenarios (C1–C5b).
- [`docs/superpowers/specs/2026-04-23-chaos-expansion-design.md`](../../../docs/superpowers/specs/2026-04-23-chaos-expansion-design.md) — v2 expansion (JobSet pods, Helm, churn, API disruption, benchmark-runtime).

## Running

```bash
# Full local suite (Kind; fresh cluster + image build)
uv run pytest tests/kubernetes/chaos/ -v -m k8s_slow

# Reuse an existing aiperf-pytest cluster (fastest iteration)
uv run pytest tests/kubernetes/chaos/ -v -m k8s_slow --k8s-reuse-cluster --k8s-skip-build

# Run a single scenario
uv run pytest tests/kubernetes/chaos/test_chaos_cancellation.py::test_c1_delete_aiperfjob_mid_ramp -v
```

Every test is marked `k8s_slow` because scenarios intentionally wait on pod termination grace (~30 s), benchmark duration (~2 min for the longrun CR), or operator recovery (~15 s). Expect ~5 minutes per scenario on a modest laptop Kind cluster.

## What's here

```
tests/kubernetes/chaos/
  __init__.py
  conftest.py                       # chaos_injector / toxiproxy_injector / mock_server_injector / operator_ready_shared_pid fixtures
  chaos_injector.py                 # intent-revealing kubectl wrapper for faults
  toxiproxy.py                      # ToxiproxyInjector: REST admin API wrapper for proxies + toxics
  mock_server_injector.py           # MockServerInjector: pause/resume/crash the mock-server Deployment
  fixtures/
    toxiproxy.yaml                  # Toxiproxy Deployment + Service manifest (aiperf-chaos-toxiproxy ns)
  test_chaos_cancellation.py        # C1, C3
  test_chaos_operator_resilience.py # C4, C5 (xfail)
  test_chaos_jobset_pods.py         # C6, C7, C8, C9
  test_chaos_helm.py                # H1, H2, H3, H4 (xfail)
  test_chaos_churn.py               # C10, C11, C12 (xfail)
  test_chaos_api_disruption.py      # C15 (xfail), C16 (xfail)
  test_chaos_benchmark.py           # B1, B2, B3 (xfail)
  findings-2026-04-23.md            # v1 run log
  findings-2026-04-23-v2.md         # v2 run log (populated during chaos session)
  README.md                         # this file
```

The `ChaosInjector` helper is the single entry point for kubectl-level faults — methods like `delete_cr_no_wait`, `kill_operator_pod`, `stamp_completion_claim`, `wait_for_phase` hide the kubectl details so new scenarios read as intent not plumbing. `ToxiproxyInjector` and `MockServerInjector` play the same role for their respective fault domains.

## Toxiproxy fixture

`fixtures/toxiproxy.yaml` deploys a [toxiproxy](https://github.com/Shopify/toxiproxy) Deployment + Service in namespace `aiperf-chaos-toxiproxy`, exposing the `:8474` admin REST API and a small pool of listen ports for per-test proxy endpoints. The session-scoped `toxiproxy_injector` fixture in `conftest.py` port-forwards the admin API, wraps it in a typed `ToxiproxyInjector` client, and yields it to tests. API-disruption tests (`test_chaos_api_disruption.py`) and latency tests (`test_chaos_benchmark.py` B3) add proxies + toxics per-test and always call `await toxiproxy_injector.reset()` in `finally` so the next test starts from an empty proxy table. The Service is reachable cluster-internally as `toxiproxy.aiperf-chaos-toxiproxy.svc`, which lets tests route in-cluster traffic (operator → apiserver, benchmark → mock-server) through the proxy when the relevant endpoint-URL overrides are available.

## Shared PID namespace

Wave 0 added an opt-in `podTemplate.shareProcessNamespace` Helm chart value (also exposed via the `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` operator env) that flips `Pod.spec.shareProcessNamespace` to true on JobSet pods. With it on, `kubectl exec <pod> -c <sidecar> -- kill ...` can reach the aiperf controller/worker PID in another container of the same pod; without it, each container has its own PID namespace and `kill` can only see PID 1 (the shim). The chaos suite sets it true for pods it drives via the `operator_ready_shared_pid` fixture; the production default remains false so normal deployments don't accidentally expose cross-container process introspection. The runtime image is distroless-python with only `bash` and a busybox multicall — there's no `pkill` or `pidof`, so `ChaosInjector._kill_process_by_cmdline_fragment` walks `/proc/*/cmdline` by hand to discover the target PID before invoking `kill`.

## Scenarios and how they map to operator code

### Cancellation & operator resilience

| Test | Design ID | Exercises |
|------|-----------|-----------|
| `test_c1_delete_aiperfjob_mid_ramp` | C1 | `lifecycle.on_delete` → `request_cancellation` + `close_progress_client` → owner-ref GC |
| `test_c3_rapid_double_delete_is_idempotent` | C3 | `request_cancellation` / `close_progress_client` idempotence |
| `test_c4_kill_operator_mid_benchmark_recovers` | C4 | kopf reconcile resume; durable `completion-claimed` annotation across operator restart |
| `test_c5_orphaned_claim_recovers` (xfail) | C5b | recovery from "operator died between claim and handle_completion" — `_recover_orphaned_completion_claim` in `monitor.py` |

### JobSet / workload faults

| Test | Design ID | Exercises |
|------|-----------|-----------|
| `test_c6_kill_controller_container_salvages` | C6 | `_maybe_recover_terminated_controller` salvage path after `kubectl exec` kill of `aiperf.cli` in control-plane container |
| `test_c7_kill_worker_pod_mid_benchmark` | C7 | JobSet pod restartPolicy + controller tolerance of transient worker loss |
| `test_c8_kill_event_bus_sidecar` | C8 | sidecar restart + event-bus reconnect inside controller pod (cross-container kill via shared PID ns) |
| `test_c9_kill_results_sidecar_mid_fetch` | C9 | `fetch_results_with_retry` backoff while the results sidecar restarts; interaction with orphaned-claim recovery |

### Helm / churn / API / benchmark

| Test | Design ID | Exercises |
|------|-----------|-----------|
| `test_h1_install_job_uninstall_reinstall_is_clean` | H1 | Helm install → run CR → uninstall → reinstall; CRD + finalizer cleanup idempotence |
| `test_h2_upgrade_with_inflight_job_preserves_cr` | H2 | `helm upgrade` mid-benchmark; CR + JobSet survive operator rollout |
| `test_h3_invalid_values_fail_fast_and_recover` | H3 | Chart template error, rollback, clean reinstall |
| `test_h4_missing_jobset_crd_surfaces_error` (xfail) | H4 | operator behavior when `jobsets.jobset.x-k8s.io` CRD is absent (exploratory — no contract yet) |
| `test_c10_rapid_create_delete_recreate_same_name` | C10 | `on_create` `clear_cancellation` unsticks a re-created CR with the old name |
| `test_c11_parallel_jobs_delete_subset` | C11 | parallel reconcile; deletion of 5/10 CRs doesn't destabilize the others |
| `test_c12_invalid_spec_surfaces_conditions` (xfail) | C12 | malformed `spec.benchmark` surfaces as `phase=Failed` (exploratory — CRD schema is permissive) |
| `test_c15_pause_apiserver_30s_recovers` (xfail) | C15 | 30 s apiserver pause via toxiproxy; operator reconcile retries + CR Completes (needs `KUBERNETES_SERVICE_HOST` override in operator Deployment) |
| `test_c16_block_operator_controller_http_falls_back` (xfail) | C16 | salvage path still completes CR when operator↔controller HTTP is dropped (needs `AIPERF_K8S_CONTROLLER_HTTP_URL` override) |
| `test_b1_mock_server_500s_mid_run` | B1 | benchmark tolerates 10 s burst of HTTP 500 from mock-server; error-rate metric populated |
| `test_b2_mock_server_restart_mid_run` | B2 | benchmark recovers across a mock-server Deployment restart (connection refused → reconnect) |
| `test_b3_mock_server_latency_injection` (xfail) | B3 | per-request latency spike via toxiproxy toxic — needs fixture to expose an in-cluster listen port for mock-server traffic |

### Xfail scenarios and why

Each xfail ships with `strict=False` so the test converts a controlled fault into a documented "not yet contracted" signal rather than a hard failure. Rationale from each Wave's report:

- **H4 — missing JobSet CRD.** Chart does not pre-validate the `jobsets.jobset.x-k8s.io` CRD, so operator behavior when it's absent is observed rather than contract-asserted. Flips to pass when the chart adds a `requires:` check or the operator itself fails fast with a typed error.
- **C12 — invalid spec.** The CRD's OpenAPI schema is deliberately permissive (flat `spec.benchmark`), so bogus config is accepted at admission and only surfaces at operator-side validation. Test xfails when the operator logs-and-retries instead of stamping `phase=Failed` within 60 s.
- **C15 — 30 s apiserver pause.** Requires redirecting the operator's apiserver traffic through toxiproxy, which needs the operator Deployment re-rendered with `KUBERNETES_SERVICE_HOST` / `KUBERNETES_SERVICE_PORT` pointed at the toxiproxy Service. Cross-cutting Helm change; deferred. Flips to pass when the chart adds an apiserver-URL override env.
- **C16 — block operator↔controller HTTP.** Operator resolves controller URLs from per-JobSet pod DNS directly, so there is no Service to patch for a clean in-cluster intercept. Needs an `AIPERF_K8S_CONTROLLER_HTTP_URL` operator-env override to route controller HTTP through toxiproxy.
- **B3 — mock-server latency injection.** The toxiproxy fixture Service does not currently expose listen port `:20010` for mock-server traffic, and the test refuses to mutate the fixture inline. Flips to pass when `fixtures/toxiproxy.yaml` advertises an extra port + a helper points `AIPerfJobConfig.endpoint_url` at `toxiproxy.<ns>.svc:20010`.

## Bugs this suite has already surfaced

All four bugs below were discovered in the 2026-04-23 v1 session and fixed in this branch:

- **Fixed:** stale `get_api` import in `src/aiperf/kubernetes/completion_signal.py` — every benchmark completion's stop hook crashed silently.
- **Fixed:** operator self-heals an orphaned `completion-claimed` annotation via `_recover_orphaned_completion_claim` (`src/aiperf/operator/handlers/monitor.py`); `test_c5_orphaned_claim_recovers` is no longer xfail.
- **Fixed:** `tests/kubernetes/helpers/operator.py` now sets `AIPERF_K8S_SERVER_METRICS_MANAGER_MEMORY=256Mi` (was 128Mi, OOMed and blocked `SystemController`).
- **Fixed:** `server_metrics_manager` auto-discovery uses the pod's own namespace (via `AIPERF_NAMESPACE` env / downward-API file) instead of endpoint-derived namespace — no more cross-namespace 403 spam.

V2 findings (populated during the next manual chaos run) live in `findings-2026-04-23-v2.md`.

## Adding a new scenario

1. Add a method to `ChaosInjector` (or `ToxiproxyInjector` / `MockServerInjector`) that expresses the fault at intent level — not raw kubectl args or REST calls.
2. Put the test in the most-related file (`test_chaos_cancellation.py`, `test_chaos_jobset_pods.py`, `test_chaos_benchmark.py`, etc.) or add a new `test_chaos_<family>.py`.
3. Tests MUST mark themselves `@pytest.mark.asyncio` and `@pytest.mark.k8s_slow` (via `pytestmark` at module top). Use `@pytest.mark.timeout(300)` on long scenarios.
4. If the scenario needs cross-container `kill`, depend on `operator_ready_shared_pid` (not `operator_ready`). If it needs toxiproxy, depend on `toxiproxy_injector`. If it needs mock-server disruption, depend on `mock_server_injector`.
5. Always wrap the injection + assertions in `try / finally` with a force-delete (and `toxiproxy_injector.reset()` / `mock_server_injector.restore()` as applicable) in `finally`. Chaos tests leak resources by definition; belt-and-suspenders cleanup keeps the cluster usable for the next test.
6. If the scenario depends on infra not yet in place, ship it as `@pytest.mark.xfail(strict=False, reason=...)` with a concrete flip-to-pass condition in the reason string. Do NOT use `strict=True` — chaos scenarios are allowed to be exploratory.
7. Document the new scenario in `findings-2026-04-23-v2.md` (or the next session log) and in this README's scenario table.
