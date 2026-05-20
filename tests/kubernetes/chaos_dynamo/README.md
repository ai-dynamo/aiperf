# AIPerf Chaos Dynamo — D-series Scenario Suite

D-series chaos scenarios targeting a live Dynamo deployment, complementing
the unit-grade `chaos_common/` adapter tests. See:

- `~/.aiperf/docs/superpowers/specs/2026-05-19-dynamo-chaos-suite-design.md` — D-series catalog (54 scenarios).
- `~/.aiperf/docs/superpowers/specs/2026-05-19-dynamo-net-new-chaos-tests.md` — the 33-of-54 that close real gaps in dynamo's `tests/fault_tolerance/`.

## Status

Wave-0 — the 10 highest-leverage scenarios — ships here:
D101, D104, D701, D201, D401, D803, D802, D801, D301, D704.

## Fixture composition

Three layers, all rooted in `conftest.py`:

1. **Cluster + dynamo deployment** re-exported from
   `tests.kubernetes.gpu.dynamo.conftest`: `dynamo_operator`, `dynamo_config`,
   `dynamo_server`, `dynamo_endpoint_url`. Re-exported explicitly because
   `tests/kubernetes/gpu/__init__.py` forbids subdirectory `__init__.py`
   files, so pytest's conftest walk does not reach `gpu/dynamo/` from a
   sibling `chaos_dynamo/` directory.
2. **Toxiproxy** — `dynamo_toxiproxy` (package-scoped) deploys
   `tests/kubernetes/chaos_common/fixtures/toxiproxy.yaml` (the expanded
   17-port manifest in namespace `chaos-toxiproxy`) and yields a
   ready-to-use `ToxiproxyInjector`. Distinct from the legacy
   `toxiproxy_injector` in `tests/kubernetes/chaos/conftest.py` (different
   namespace, different port layout).
3. **Unified faults registry** — `faults` (function-scoped) overrides the
   echo-only registry from `tests/kubernetes/chaos_common/conftest.py` and
   pre-registers every concrete injector (`pod`, `workload`, `crd`,
   `network`, `store`, `process`, `client`, `cluster`). The `CRDInjector` is
   parameterized for `DynamoGraphDeployment` / `nvidia.com` / `dynamo-system`.

Pytest fixture resolution prefers the conftest closest to the test file, so
`tests/kubernetes/chaos_dynamo/test_*.py` resolves `faults` from this
package's conftest rather than the chaos_common one. Adapter unit tests
under `chaos_common/` continue to see the echo-only registry.

## Helpers

Two plain `async def` helpers (not fixtures) are exported from the conftest:

- `wait_for_dgd_state(kubectl, name, namespace, target_state, *, timeout, poll_interval)` —
  polls `kubectl get dynamographdeployment <name> -n <ns> -o jsonpath='{.status.state}'`
  until it matches. Used by D101, D104, D701.
- `scrape_frontend_metrics(kubectl, namespace, *, deployment_name, metrics_port, timeout)` —
  port-forwards the dynamo frontend pod, hits `/metrics`, and returns the
  parsed Prometheus text as `{name: float}`. Used by D803 and similar.

## Running

Full Wave-0 against a kind cluster (Cilium overlay required for D704):

    uv run pytest tests/kubernetes/chaos_dynamo/ -v -m k8s_slow -n auto

Reuse existing cluster + skip image build:

    uv run pytest tests/kubernetes/chaos_dynamo/ -v -m k8s_slow \
      --k8s-reuse-cluster --k8s-skip-build -n auto

A single scenario:

    uv run pytest tests/kubernetes/chaos_dynamo/test_chaos_d1xx_operator_admission.py \
      -k test_chaos_d101_operator_kill -v

The D704 test is `xfail(strict=True)` unless `KIND_HAS_CILIUM=1` is set in the
environment; see `tests/kubernetes/chaos_common/README.md` for the
Cilium-on-kind bring-up.

## Per-file marker pattern

Every D-series test file must apply the slow marker, and async scenarios must also apply `pytest.mark.asyncio`:

```python
import pytest

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
```

Use a module-level `asyncio` mark when every test in the file is async. If a consolidated file mixes async and synchronous scenarios, keep `pytest.mark.k8s_slow` at module level and apply `@pytest.mark.asyncio` only to async tests.
