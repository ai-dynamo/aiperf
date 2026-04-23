<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Operator Chaos Testing — Design

## Goal

Chaos-test the AIPerf kopf operator on a real Kind cluster, hands-on. Exercise the paths hardened on `ajc/k8s`: durable completion claim (`try_claim_completion`), cooperative cancellation (`request_cancellation` / `is_cancellation_requested`), results-ready marker, progress-poll reconnect, JobSet pod-kill recovery.

Scope is controller-plane chaos — NOT load-testing. We care about correctness under fault, not throughput.

## Two phases

1. **Manual verification.** Bring up a real cluster, inject each fault by hand via `kubectl` / `docker exec`, watch the operator reconcile, record what happened.
2. **Automation.** For reproducible-and-informative scenarios only, codify as pytest under `tests/kubernetes/chaos/` using the existing `BenchmarkDeployer` / `OperatorDeployer` / `KubectlClient` fixtures from `tests/kubernetes/conftest.py`.

Non-automatable findings get a reproduction recipe in `tests/kubernetes/chaos/README.md`.

## Environment

- Fresh Kind cluster `aiperf-chaos` from `ajc/k8s` HEAD
- aiperf image + aiperf-mock-server image built locally and loaded into Kind
- JobSet controller CRD installed
- Operator deployed via the existing helm chart path used by `tests/kubernetes/test_helm.py`
- Findings captured under `/tmp/aiperf-chaos/<scenario>/` (CR YAML, operator logs, controller logs, events)

## Scenario catalog

Codes map to the scenarios enumerated in the planning transcript.

### Cancellation & completion (highest value — touches ajc/k8s work)

- **C1** — Delete `AIPerfJob` mid-ramp → cooperative cancellation
- **C2** — Delete CR after completion handler started → race with `try_claim_completion`
- **C3** — Two rapid deletes on same CR → idempotent cleanup
- **C4** — Kill operator pod mid-benchmark → kopf reconcile resume
- **C5** — Kill operator pod during completion handler → results marker semantics

### JobSet / workload disruption

- **C6** — Kill system-controller pod mid-ramp
- **C7** — Kill one worker pod mid-benchmark
- **C8** — Kill event-bus sidecar
- **C9** — Kill results sidecar → readiness-marker on restart

### API / churn

- **C10** — Rapid create → delete → create same-named CR
- **C11** — 10 parallel AIPerfJobs, delete 5 mid-run
- **C12** — Apply invalid spec → validation + status conditions

### Resource / scheduling

- **C13** — Controller OOM via tiny memory limit
- **C14** — Pod stuck Pending (unsatisfiable resources)

### Node/cluster-level

- **C15** — `docker pause` control-plane node ~30s (apiserver hiccup)
- **C16** — Block operator ↔ controller HTTP API (`:19090`)

## Automation layout

Peer to existing helper tree:

```
tests/kubernetes/chaos/
  __init__.py
  chaos_injector.py        # kill_operator / kill_controller / kill_worker / pause_node / etc.
  chaos_assertions.py      # CR reached terminal state, no dangling PVCs, events emitted
  test_chaos_cancellation.py    # C1-C3, C11
  test_chaos_pod_kills.py       # C4-C9
  test_chaos_churn.py           # C10, C12
  test_chaos_api_disruption.py  # C13-C16 (those that prove reproducible)
  README.md                # Scenario catalog + repro recipes (incl. non-automated)
```

All tests marked `@pytest.mark.k8s_slow` + `@pytest.mark.asyncio`. Hard per-test timeout 5 min. Teardown deletes CR unconditionally. If cluster wedges, the session-level fixture rebuilds the cluster.

## Out of scope

- Kueue gang-scheduling chaos
- GPU / vLLM chaos
- Operator code refactors (bugs found → noted, optional targeted fix commit)
- CNI swap for NetworkPolicy (C16 will use Service-level blocks if CNI is the bottleneck)

## Exit criteria

- All 16 scenarios classified (Pass / Fail / Interesting / N/A)
- At least one codified pytest chaos file green locally
- `tests/kubernetes/chaos/README.md` committed with scenario catalog + manual-repro recipes
