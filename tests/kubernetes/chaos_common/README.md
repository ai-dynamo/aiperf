# AIPerf Chaos Common — Unified Fault-Injection Module

This module hosts the shared fault-injection primitives (registries, ABCs,
marks, and helpers) that the unified chaos suite uses to express
reproducible fault scenarios across the AIPerf data plane, control plane,
and Dynamo integration paths.

## Status

The shared injector registry and its hermetic adapter-contract tests are
present and run in the trusted Kubernetes gate. AIPerf cluster fault scenarios
live in `tests/kubernetes/chaos/` and `tests/kubernetes/chaos_aiperf/`.
The external Dynamo D-series suite from the historical Kubernetes branch is
not part of this AIPerf merge; the D-series commands below document that
separate environment and are not runnable from this worktree.

## Cilium-on-kind for D704

### Why this matters

D704 (`tests/kubernetes/chaos_dynamo/test_chaos_d7xx_infra.py::test_d704_hf_hub_egress_blackhole`,
a Phase 4 forward-reference; this file does not exist yet)
needs a NetworkPolicy-aware CNI to inject the fault it claims to: a
blackhole on egress to the Hugging Face Hub. The default Kind CNI
(`kindnet`) is NOT NetworkPolicy-aware — it silently ignores egress
rules. Without Cilium (or Calico, or a real cluster whose CNI honors
NetworkPolicy), the test cannot actually block the egress it asserts
against, and any "passing" result would be a false positive.

### The decision (option c)

D704 ships with `@cilium_on_kind_required` from
`tests/kubernetes/chaos_common/marks.py`. When `KIND_HAS_CILIUM` is unset
(the default), the mark expands to
`pytest.mark.xfail(condition=True, strict=True, reason=...)`, which
xfail-skips the test in CI while keeping it visibly present in the suite
with a documented flip-to-pass condition. When the env var is set, the
xfail constraint stays strict — a real failure on a Cilium-enforcing
cluster becomes a loud pytest failure rather than a silent skip.

### Recipe to make D704 actually pass

Three approaches, in order of preference:

#### Recipe A — Existing real cluster (recommended for CI)

If the test target is a real cluster whose CNI already honors
NetworkPolicy (DGX, GKE, EKS, AKS, any Cilium/Calico-equipped cluster),
just set the env var when invoking pytest:

```bash
export KIND_HAS_CILIUM=1
uv run pytest tests/kubernetes/chaos_dynamo/test_chaos_d7xx_infra.py -v
```

(The `chaos_dynamo/` path above is a Phase 4 forward-reference and does
not exist yet; the command becomes runnable once Phase 4 lands.)

One env var, no other changes. The name is `KIND_HAS_CILIUM` for
historical clarity (kindnet is the broken case the gate exists for), but
any CNI that enforces NetworkPolicy egress satisfies the gate.

#### Recipe B — Cilium on Kind

Bring up a dedicated kind cluster with the default CNI disabled, then
install Cilium via Helm:

```bash
kind create cluster --name aiperf-pytest-cilium --config <(cat <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  disableDefaultCNI: true
  kubeProxyMode: none
EOF
)

helm repo add cilium https://helm.cilium.io/
helm repo update

helm install cilium cilium/cilium --version 1.16.4 \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=aiperf-pytest-cilium-control-plane \
  --set k8sServicePort=6443

kubectl --context kind-aiperf-pytest-cilium \
  wait --for=condition=Ready pod -l k8s-app=cilium \
  -n kube-system --timeout=180s

export KIND_HAS_CILIUM=1
export KUBECONFIG=$HOME/.kube/config
```

Verify Cilium is actually enforcing policy before running D704 (a quick
sanity NetworkPolicy that blocks egress and a `kubectl exec ... curl`
against an external IP is sufficient — Cilium should drop the
connection). Bump the Cilium chart version to the current release at the
time of running if 1.16.4 has aged out; the helm repo URL
(`https://helm.cilium.io/`) is the long-term stable location.

#### Recipe C — Document non-coverage

If neither A nor B is feasible (e.g. a developer laptop without
permission to install Cilium and no access to a real cluster), D704
stays xfail-skipped and the suite's effective coverage is
9-of-10. Document this in the run report; do not silently delete the
test.

### Verifying the flip

Quick sanity check that the gate behaves correctly:

```bash
KIND_HAS_CILIUM=1 uv run pytest \
  tests/kubernetes/chaos_dynamo/test_chaos_d7xx_infra.py::test_d704_hf_hub_egress_blackhole -v
```

If Cilium is enforcing NetworkPolicy, the test passes. If the env var is
set but Cilium is NOT actually enforcing (misconfigured install, wrong
context, kindnet still on the path), the test fails LOUDLY because
`xfail(strict=True)` flips the constraint: an xfail-marked test that
unexpectedly passes is itself a failure. That loud failure is the
design — better than a silent false positive.

## Running the unified chaos suite

The suite has three runnable layers; each layer requires progressively more
infrastructure. Pick the lowest layer that exercises your change.

### Adapter unit tests (no cluster needed)

The injector adapters under `chaos_common/test_*_injector.py` use the
echo-only `faults` fixture from this package's `conftest.py` — they assert
the API contract (dispatch by prefix, LIFO restore, idempotent
`AppliedFault.restore`) without touching a real cluster.

```bash
uv run pytest tests/kubernetes/chaos_common/ -n auto
```

These run on every PR via `run-kubernetes-tests.yml` (no opt-in marker).

### External D-series scenarios (not included here)

In the historical Kubernetes branch, the Dynamo-side D-series under
`chaos_dynamo/test_chaos_d*.py` overrides
the `faults` fixture with a real registry against a live cluster + dynamo
deployment. Every test carries `@pytest.mark.k8s_slow` so a default
`pytest` invocation skips the whole directory.

```bash
uv run pytest tests/kubernetes/chaos_dynamo/ -v -m k8s_slow \
    --k8s-reuse-cluster --k8s-skip-build -n auto
```

`--k8s-reuse-cluster` and `--k8s-skip-build` are inherited from
`tests/kubernetes/conftest.py` and let local iteration skip the
~3-minute Kind bring-up plus image-build cycle.

A single scenario, no marker filter:

```bash
uv run pytest tests/kubernetes/chaos_dynamo/test_chaos_d1xx_operator_admission.py \
    -k test_chaos_d101_operator_kill -v
```

### D704 specifically (Cilium-enforcing CNI required)

`test_chaos_d704_hf_hub_egress_blackhole` injects a NetworkPolicy egress
deny against the Hugging Face Hub. Default Kind CNI (kindnet) silently
ignores NetworkPolicy, so the test ships with `@cilium_on_kind_required`
which expands to `xfail(strict=True)` unless `KIND_HAS_CILIUM=1` is set.

```bash
KIND_HAS_CILIUM=1 uv run pytest \
    tests/kubernetes/chaos_dynamo/test_chaos_d7xx_infra_control_plane.py \
    -k test_chaos_d704_hf_hub_egress_blackhole -v
```

See the "Cilium-on-kind for D704" section above for the bring-up recipe.

### Historical unified-suite shape

The historical `k8s-chaos.yml` workflow exercised three directories;
`chaos_common/` ran without `k8s_slow`, while the cluster suites required it:

```bash
uv run pytest tests/kubernetes/chaos_common/ -n auto
uv run pytest tests/kubernetes/chaos/ tests/kubernetes/chaos_dynamo/ \
    -m k8s_slow -n auto
```

See `tests/kubernetes/chaos_common/FEATURES.md` for the per-`fault_id`
shipped/planned/deferred matrix.
