---
name: aiperf-kube-run
description: Use when running, deploying, submitting, attaching to, or retrieving results from a single AIPerf benchmark on a Kubernetes cluster with the aiperf kube CLI (AIPerfJob CR, operator mode or direct mode).
---

# Running an AIPerf Benchmark on Kubernetes

Hub skill for the `aiperf kube` lifecycle. One `AIPerfJob` = one benchmark = one
controller pod + N worker pods, orchestrated by the in-cluster operator.

**Related skills:** `aiperf-kube-setup` (cluster/operator install),
`aiperf-kube-triage` (a run is stuck or failed), `aiperf-kube-sweep`
(multi-variation runs).

## Lifecycle

| Stage | Command | Touches cluster |
|---|---|---|
| 1. Scaffold config | `aiperf kube init -t <template> -o bench.yaml` | no |
| 2. Offline check | `aiperf kube validate bench.yaml` | no |
| 3. Cluster check | `aiperf kube preflight -i <image> -e <url> -w <n>` | yes |
| 4. Deploy | `aiperf kube profile --config bench.yaml --image <image>` | yes |
| 5. Watch | `aiperf kube attach` / `aiperf kube list --watch` | yes |
| 6. Collect | `aiperf kube results` | yes |
| 7. Clean up | `aiperf kube delete <name> --force` | yes |

Full lifecycle narrative, phase table, exit codes, and per-command flag
reference: `references/lifecycle.md` (bundled with this skill).

## Minimum viable run

```bash
aiperf kube preflight -i aiperf:latest -e http://server:8000 -w 8
aiperf kube profile \
    --model Qwen/Qwen3-0.6B \
    --url http://server:8000 \
    --image aiperf:latest \
    --total-workers 8 \
    --detach
aiperf kube attach          # live progress
aiperf kube list            # phase == Completed?
aiperf kube results         # -> ./artifacts/<name>/
```

Defaults: benchmark namespace `aiperf-benchmarks`, operator namespace
`aiperf-system` (auto-detected by cluster-wide pod-label search).

## Rules that bite

- **Foreground `profile` in operator mode hard-times-out at 600 s.** Anything
  longer than ~10 minutes MUST use `--detach` plus `aiperf kube attach`.
  Ctrl+C or the timeout does not stop the cluster-side run. The limit is
  `AIPERF_K8S_WATCH_DEFAULT_TIMEOUT_SECONDS` (600, max 86400) if you would
  rather raise it than detach.
- **Non-TTY stdout auto-detaches.** CI and piped invocations never block; do not
  write scripts that assume `profile` waits.
- **`--dry-run` never contacts the cluster**, so the mode it prints is *assumed*,
  not detected, and the CR is printed *before* validation. Pass `--no-operator`
  to preview direct mode. Use `aiperf kube validate` for a real verdict.
- **`~/.aiperf/last_kube_benchmark.json` is per-user, not per-context.** Every
  command that defaults its target reads it. When working across clusters or
  namespaces in one shell, always pass the job id positionally plus `-n` and
  `--kube-context`.
- **`kubectl delete` leaves that file stale**; `aiperf kube delete` clears it.
- **Results come from the operator PVC by default.** `--from-pods` is required
  for direct-mode runs, and on the default `--all` path the controller API is
  the only tier — there is no `kubectl cp` fallback unless `--summary-only`.
- **Cancel is not delete, but it is not a safe pause either.** `aiperf kube
  cancel` patches `spec.cancel: true`; the CR survives, but the operator
  deletes the JobSet and sets `phase=Cancelled` **without harvesting**, and
  every completion path short-circuits once cancellation is requested. Results
  survive only if a harvest already finished. Cancel an in-flight run and there
  is nothing on the PVC and no pods left for `--from-pods` — download first,
  cancel second.
- **Ambiguous names are refused, not guessed.** When an `AIPerfJob` and an
  `AIPerfSweep` share a name, destructive commands need `--kind job|sweep`.
  The refusal prints an error and **still exits 0**, so a script that only
  checks `$?` will believe it cancelled or deleted something.
- **`cancel` and `delete` only know about CRs, and say nothing useful when
  there isn't one.** Both resolve the target through `find_aiperf_cr`; a
  direct-mode (`--no-operator`) run has no `AIPerfJob`, so they print "No
  AIPerfJob or AIPerfSweep named ..." and exit 0 while the JobSet keeps
  running. Tear down direct-mode runs with `kubectl delete jobset <name>` (plus
  its ConfigMap/Role/RoleBinding), not with `aiperf kube delete`.

## Exit-code convention

| Group | Commands | Exits non-zero when |
|---|---|---|
| Gating | `validate`, `preflight`, `results`, `results list-runs` | any failure, including partial download |
| Addressing | `attach`, `logs`, `cancel`, `delete`, `shutdown`, `debug`, `list` | only `attach`/`logs`, only when the target does not exist (`list` also exits 1 on conflicting status filters). `cancel`/`delete` exit 0 even on not-found and on ambiguous-name refusal |

Use `--ignore-not-found` on `attach`/`logs` in teardown scripts. A target that
exists but has nothing to show exits 0 by design.

## Operator mode vs direct mode

Operator mode is chosen automatically when the `aiperfjobs.aiperf.nvidia.com`
CRD is present. It owns JobSet/ConfigMap/RBAC lifecycle via ownerReferences and
harvests results to a PVC on terminal phase (nothing is harvested mid-run).

Direct mode (`--no-operator`, or CRD absent) creates the resources itself, has
no CR and no PVC, and uses an 8-hour JobSet TTL (vs 5 minutes) so pods survive
long enough for `aiperf kube results --from-pods`. Direct-mode details are in
`references/lifecycle.md`.

`--operator` skips the cluster-scoped CRD probe on its own — that is how
namespace-scoped tenants submit without cluster-wide RBAC. Pair it with an
explicit `--namespace`, since skipping the probe also skips operator
discovery.

## Common mistakes

| Mistake | Consequence |
|---|---|
| Long run without `--detach` | `TimeoutError` at 600 s; run keeps going, you lose the stream |
| `aiperf kube results` right after submit | Operator harvests only on terminal phase; wait for `Completed` |
| Trusting `--dry-run` as validation | Unknown envelope keys print fine and reject on submit |
| Omitting `-n` after switching clusters | Reads the other cluster's last-benchmark record |
| Reusing a direct-mode name without deleting ConfigMap/Role/RoleBinding | Direct mode refuses to adopt existing resources |
| `aiperf kube delete` on a direct-mode run | Prints not-found, exits 0, JobSet keeps burning cluster capacity |
