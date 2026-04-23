<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Chaos Testing — Expansion Design (v2)

Follow-up to `2026-04-23-operator-chaos-testing-design.md`. The v1 session automated C1/C3/C4/C5 and left ~12 scenarios deferred behind tool limitations. This expansion lifts those limits — JobSet-pod kills, Helm chaos, parallel/validation churn, apiserver/controller-HTTP fault injection, and live benchmark-server faults — so the chaos suite covers the full real-world blast radius of the AIPerf stack on Kind.

## Goal

Fault-inject across the **three boundaries** a production deployment actually spans:

1. **Helm chart** — what happens to in-flight jobs across install / upgrade / invalid-values / CRD missing.
2. **Operator + JobSet pods** — kill every container we ship (controller, workers, event-bus sidecar, results sidecar), validate salvage + restart semantics.
3. **Benchmarking runtime** — introduce real inference-server faults mid-run (500s, restart, latency injection) and verify the benchmark completes with honest metrics.

Scope is still **correctness under fault**, not throughput.

## Non-goals

- GPU / vLLM chaos (separate DGX-cluster session)
- Kueue gang-scheduling chaos (out of scope)
- Discovering NEW chaos categories (we work from the existing v1 catalog)
- Load testing or perf regression

## Infrastructure prerequisites

Three one-time additions unlock the deferred scenarios:

### I1. `shareProcessNamespace` opt-in on JobSet pods

Add `share_process_namespace: bool = False` to `aiperf.config.deployment.PodTemplateConfig`. Thread into `_build_pod_spec` in `src/aiperf/kubernetes/jobset_specs.py` as `podSpec.shareProcessNamespace` when true. Expose via:

- Helm values key `podTemplate.shareProcessNamespace` (default false)
- Operator env var `AIPERF_K8S_SHARE_PROCESS_NAMESPACE` (default false, parsed in `tests/kubernetes/helpers/operator.py`)

Chaos suite flips it true; production default stays false. Unlocks C6/C8 via `kubectl exec <pod> -c <chaos-sidecar> -- kill -9 <pid>`.

### I2. Toxiproxy fixture

Add `tests/kubernetes/chaos/fixtures/toxiproxy.yaml` (Deployment + Service in namespace `aiperf-chaos-toxiproxy`). Add `ToxiproxyInjector` helper in `tests/kubernetes/chaos/toxiproxy.py` — wraps the toxiproxy REST API (`:8474`) with intent-revealing methods: `add_proxy`, `add_toxic(type, attrs)`, `remove_proxy`, `reset`. Kopf operator is redeployed with apiserver URL pointed at toxiproxy when the fixture is active. Unlocks C15/C16.

### I3. ChaosInjector extensions

Extend `tests/kubernetes/chaos/chaos_injector.py` with:

- `get_controller_pod_name(namespace, job_name) -> str`
- `get_worker_pod_names(namespace, job_name) -> list[str]`
- `kill_container_by_name(pod, container, namespace)` — uses shared PID namespace when available
- `wait_for_container_restart(pod, container, namespace, timeout)`
- `create_invalid_cr(namespace, name, spec_patch)` — builds a CR with a known-bad spec and applies it
- `apply_toxic(proxy_name, toxic_spec)` / `clear_toxics(proxy_name)` — delegates to ToxiproxyInjector

## Scenario catalog (expansion)

Scenario IDs match v1 where applicable. New IDs (H*, B*) are introduced for Helm and benchmarking.

### Wave 1 — JobSet / workload chaos (depends on I1)

| ID | Name | Exercises |
|----|------|-----------|
| C6 | Kill system-controller container mid-ramp | `_maybe_recover_terminated_controller` salvage path; operator converges CR to Completed |
| C7 | Kill one worker pod mid-benchmark | JobSet recreates worker; benchmark tolerates and Completes |
| C8 | Kill event-bus sidecar | Sibling-container death brings controller down → salvage path |
| C9 | Kill results sidecar mid-fetch | `fetch_results_with_retry` survives restart; readiness marker re-checked |

### Wave 2 — Helm chaos (independent of other waves)

| ID | Name | Exercises |
|----|------|-----------|
| H1 | install → job → uninstall → re-install | no orphaned CRDs/PVCs/finalizers; fresh install is clean |
| H2 | helm upgrade with in-flight job | operator upgrade does not drop Running CR; monitor timer resumes |
| H3 | invalid values (bad image + resources) | install fails fast; no partial deploy; re-install clean after fix |
| H4 | chart install with `--skip-crds` and no pre-existing JobSet CRD | operator surfaces missing-dependency error; CR creation blocked cleanly |

### Wave 3 — Churn + API chaos

| ID | Name | Exercises |
|----|------|-----------|
| C10 | Rapid create → delete → create same CR name | `clear_cancellation` on `on_create`; no stale flags starve cycle 2 |
| C11 | 3 parallel AIPerfJobs, delete 2 mid-run | per-CR isolation; unaffected jobs Complete |
| C12 | Apply invalid spec (bad endpoint URL) | status conditions surface validation error; CR reaches Failed |
| C15 | Pause apiserver via toxiproxy (30s) | operator reconcile degrades gracefully and recovers on unpause |
| C16 | Block operator ↔ controller HTTP (`:19090`) via toxiproxy | operator falls back to salvage path; CR still terminates |

### Wave 4 — Benchmark runtime chaos

Targets the mock-server pod used by the k8s test harness. New helper `tests/kubernetes/chaos/mock_server_injector.py` does restart / SIGTERM / scale-to-zero.

| ID | Name | Exercises |
|----|------|-----------|
| B1 | Mock server returns 500s mid-run | error rate metric reflects it; benchmark does not hang; CR Completes |
| B2 | Mock server restart mid-run | connection reset; worker reconnects; CR Completes |
| B3 | Mock server latency injection via toxiproxy | p99 latency reflects injected delay; no timeout cascade |

## Automation layout

```
tests/kubernetes/chaos/
  __init__.py
  conftest.py
  chaos_injector.py              # extended
  toxiproxy.py                   # NEW
  mock_server_injector.py        # NEW
  fixtures/
    toxiproxy.yaml               # NEW Deployment + Service manifest
  test_chaos_cancellation.py     # existing, kept
  test_chaos_operator_resilience.py   # existing, kept
  test_chaos_jobset_pods.py      # NEW — C6, C7, C8, C9
  test_chaos_helm.py             # NEW — H1, H2, H3, H4
  test_chaos_churn.py            # NEW — C10, C11, C12
  test_chaos_api_disruption.py   # NEW — C15, C16
  test_chaos_benchmark.py        # NEW — B1, B2, B3
  README.md                      # updated
  findings-2026-04-23.md         # v1 session log (kept)
  findings-2026-04-23-v2.md      # NEW expansion session log
```

All tests remain `@pytest.mark.asyncio` + `@pytest.mark.k8s_slow`. Per-scenario timeout ≤ 5 min. Teardown unconditionally deletes every CR the test touches and calls `ToxiproxyInjector.reset()` for the API-disruption suite.

## Helm-chart changes

- Add `podTemplate.shareProcessNamespace` value (default `false`) → threaded via operator env var to `PodTemplateConfig.share_process_namespace`.
- No other chart shape changes. Chaos-only fixtures live under `tests/kubernetes/chaos/fixtures/`, not in the main chart.

## Failure model

Each new test must:

1. Set up (deploy CR, wait for expected phase).
2. Inject one fault via `ChaosInjector` / `ToxiproxyInjector` / `MockServerInjector`.
3. Assert a specific observable: CR phase, claim annotation, container restart count, operator log line, results-ready marker, or measured metric.
4. Force-delete the CR in `finally`.
5. For toxiproxy tests, also call `toxiproxy.reset()` in `finally`.

When a test uncovers a real bug (as C5b did in v1), it is committed as `xfail(strict=False)` with a reproduction steps comment, and a bug finding is appended to `findings-2026-04-23-v2.md` with repro + verdict + hypothesis for the fix. Fixes land in the same branch unless the surface area is out of scope.

## Exit criteria

- All 16 scenarios (C1–C16 + H1–H4 + B1–B3) either automated (green) or documented in README with a "not yet automated" rationale.
- Helm chart surfaces `podTemplate.shareProcessNamespace` and passes existing `test_helm.py` (no regressions).
- Toxiproxy fixture + ToxiproxyInjector land with their own unit tests.
- `make check-ergonomics` + `make check-ruff-baselined` clean for all new files.
- `tests/unit/` green under `-n auto`.
- README + findings log committed; design doc (this file) referenced from README.

## Out of scope (explicit)

- Chaos scenarios that demand a real GPU/vLLM mock
- Kueue gang-scheduling chaos
- CNI replacement for NetworkPolicy enforcement (toxiproxy covers the interesting fault)
- Refactoring non-chaos operator code unless a bug surfaces and a targeted fix is trivial
