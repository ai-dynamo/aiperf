<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf Operator Chaos Suite

Fault-injection tests for the kopf operator that back the AIPerfJob CRD. This suite sits alongside `tests/kubernetes/` and reuses its cluster/operator fixtures; it lives under `tests/kubernetes/chaos/` so the standard `K8S_TEST_*` flags (Kind cluster, image loading, mock server) all apply unchanged.

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
  conftest.py                       # chaos_injector fixture
  chaos_injector.py                 # intent-revealing kubectl wrapper for faults
  test_chaos_cancellation.py        # C1, C3
  test_chaos_operator_resilience.py # C4, C5 (xfail)
  README.md                         # this file
```

The `ChaosInjector` helper is the single entry point — methods like `delete_cr_no_wait`, `kill_operator_pod`, `stamp_completion_claim`, `wait_for_phase` hide the kubectl details so new scenarios read as intent not plumbing.

## Scenarios and how they map to operator code

| Test | Design ID | Exercises |
|------|-----------|-----------|
| `test_c1_delete_aiperfjob_mid_ramp` | C1 | `lifecycle.on_delete` → `request_cancellation` + `close_progress_client` → owner-ref GC |
| `test_c3_rapid_double_delete_is_idempotent` | C3 | `request_cancellation` / `close_progress_client` idempotence |
| `test_c4_kill_operator_mid_benchmark_recovers` | C4 | kopf reconcile resume; durable `completion-claimed` annotation across operator restart |
| `test_c5_orphaned_claim_recovers` (xfail) | C5b | recovery from "operator died between claim and handle_completion" — presently **reveals an open bug** (see findings log) |

## Design context

The full design, scenario catalog, and every verdict are in `docs/superpowers/specs/2026-04-23-operator-chaos-testing-design.md`. The session findings log (reproduction steps, timestamps, bugs discovered) lives at `/tmp/aiperf-chaos/findings.md` while a session is active — copy interesting results into this README or a docs/ note before the tmp dir is reaped.

## Known open questions (deferred)

- **C6 — kill system-controller container deterministically.** The aiperf runtime container lacks a writable filesystem for ad-hoc `pkill`, and `kill -9 1` hits the tini/shim. Options: add `shareProcessNamespace: true` to the JobSet pod spec so `kubectl debug` can reach the aiperf PID, or ship a tiny `/usr/local/bin/aiperf-chaos-kill` helper in the image. Captured in findings.md #C6.
- **C7–C9 (worker/sidecar kills).** Low operator-logic signal; mostly tests JobSet + kubelet behaviour that Kubernetes already covers. Revisit if we add per-sidecar reconcile logic.
- **C11 (many parallel jobs).** Needs >= 8 vCPU / 16 GiB node; appropriate for the GPU/DGX chaos session, not for Kind.
- **C15 (apiserver pause).** `docker pause` on the kind node freezes the kubelet too — need toxiproxy or similar between kube-apiserver and the operator for a clean fault.
- **C16 (NetworkPolicy block).** `kindnet` doesn't enforce NetworkPolicy; switching to Calico just for this is yak-shaving.

## Bugs this suite has already surfaced

All four bugs below were discovered in the 2026-04-23 session:

- **Fixed:** stale `get_api` import in `src/aiperf/kubernetes/completion_signal.py` — every benchmark completion's stop hook crashed silently.
- **Fixed:** operator self-heals an orphaned `completion-claimed` annotation via `_recover_orphaned_completion_claim` (`src/aiperf/operator/handlers/monitor.py`); `test_c5_orphaned_claim_recovers` is no longer xfail.
- **Fixed:** `tests/kubernetes/helpers/operator.py` now sets `AIPERF_K8S_SERVER_METRICS_MANAGER_MEMORY=256Mi` (was 128Mi, OOMed and blocked `SystemController`).
- **Fixed:** `server_metrics_manager` auto-discovery uses the pod's own namespace (via `AIPERF_NAMESPACE` env / downward-API file) instead of endpoint-derived namespace — no more cross-namespace 403 spam.

## Adding a new scenario

1. Add a method to `ChaosInjector` (or reuse an existing one) that expresses the fault at intent level — not raw kubectl args.
2. Put the test in the most-related file (`test_chaos_cancellation.py`, `test_chaos_operator_resilience.py`, etc.) or add a new `test_chaos_<family>.py`.
3. Tests MUST mark themselves `@pytest.mark.asyncio` and `@pytest.mark.k8s_slow` (via `pytestmark` at module top).
4. Always wrap the injection + assertions in `try / finally` with a force-delete in `finally`. Chaos tests leak resources by definition; belt-and-suspenders cleanup keeps the cluster usable for the next test.
5. Document the new scenario in the findings log and in this README's scenario table.
