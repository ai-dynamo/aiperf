# AIPerf Chaos — Unified-API Sibling Suite

This directory holds **unified-API ports** of the 23 legacy AIPerf chaos
scenarios that live in [`tests/kubernetes/chaos/`](../chaos/). Each ported
scenario uses the `chaos_common` `faults.inject(...)` API instead of the
legacy `ChaosInjector` / `MockServerInjector` / `ToxiproxyInjector` calls.

**Legacy is NOT replaced.** The original `chaos/test_chaos_*.py` files stay
unchanged and continue to ship. The ports here are a parallel surface that
exercises the same scenarios through the unified ABC, validating that the
adapter injectors faithfully wrap the legacy mechanisms.

See:
- `tests/kubernetes/chaos_common/` — unified `FaultInjector` ABC + injectors.
- `tests/kubernetes/chaos_common/FEATURES.md` — capability matrix.

## Relationship to `chaos/`

| Surface | Where | Status |
|---|---|---|
| Legacy AIPerf scenarios | `tests/kubernetes/chaos/` | Continues to ship as-is |
| Unified-API ports | `tests/kubernetes/chaos_aiperf/` (this dir) | Parallel coverage via `faults.inject(...)` |
| Dynamo D-series | `tests/kubernetes/chaos_dynamo/` | Same unified API, Dynamo-targeting |
| Adapter unit tests | `tests/kubernetes/chaos_common/` | Echo-only registry |

The conftest in this package:

1. Re-exports the legacy `chaos/conftest.py` via `pytest_plugins`, so every
   ported test can still request `operator_ready`, `chaos_injector`,
   `toxiproxy_injector`, `mock_server_injector`,
   `operator_ready_toxiproxy_routed`, and
   `operator_ready_apiserver_toxiproxy_routed` exactly as legacy tests do.
2. Overrides the echo-only `faults` fixture from
   `tests/kubernetes/chaos_common/conftest.py` with an `InjectorRegistry`
   pre-loaded with every concrete injector (`pod`, `workload`, `crd`,
   `network`, `store`, `process`, `client`, `cluster`). The `CRDInjector`
   is parameterized for `aiperfjob` / `aiperf.nvidia.com` /
   `aiperf-system`; the `NetworkInjector` and `StoreInjector` are wired to
   the legacy `toxiproxy_injector` (namespace `aiperf-chaos-toxiproxy`) so
   the C15/C16/B3 reserved-port pool (20000, 20002, 20010) stays intact.
3. Exports two helpers as plain `async def`:
   - `wait_for_aiperfjob_phase(kubectl, namespace, name, phases, *, timeout, current_phase, poll_interval)`
     — polls `.status.phase` (and optionally `.status.currentPhase`) with
     the same JSONPath shape and TimeoutError style as
     `ChaosInjector.wait_for_phase`.
   - `scrape_aiperf_metrics(kubectl, namespace, *, deployment_name, metrics_port, timeout)`
     — opens a short-lived `kubectl port-forward` to the operator pod,
     hits `/metrics`, returns parsed Prometheus text as `{name: float}`.
     Stub-grade; ported metrics-shape scenarios extend it as they land.

## Running

    make kubernetes-chaos-aiperf-tests-ci

Or invoke pytest directly while reusing an existing cluster:

    uv run pytest tests/kubernetes/chaos_aiperf/ -v -m k8s_slow \
      --k8s-reuse-cluster --k8s-skip-build -n auto

Same fixtures as `chaos/`; same cluster (`aiperf-pytest` or `aiperf`). Run a
single scenario:

    uv run pytest tests/kubernetes/chaos_aiperf/test_chaos_cancellation_unified.py -v

## Per-file marker pattern

Every test file in this package must declare both markers at module level:

```python
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
```

`k8s_slow` opts the test into the slow-only run gate (these scenarios spin
up a real AIPerf operator + JobSet); `asyncio` is required because every
helper here is `async def` and the conftest fixtures are
`pytest_asyncio.fixture`-based.
