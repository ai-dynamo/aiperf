# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `aiperf kube` Lifecycle Reference

Terminal reference for the single-benchmark lifecycle: operator mode, direct
mode, and the flag surface of every subcommand in the flow.

Every cluster-touching command accepts `-n`/`--namespace` (unset ->
`aiperf-benchmarks`), `--kubeconfig` (`~/.kube/config` or `$KUBECONFIG`), and
`--kube-context` (current context). The operator namespace defaults to
`aiperf-system`, auto-detected by cluster-wide pod-label search. Commands that
default their target read `~/.aiperf/last_kube_benchmark.json`.

## Operator-mode workflow

```bash
# 1-2. Scaffold + validate offline (no cluster contacted)
aiperf kube init -t goodput_slo \
    --model Qwen/Qwen3-0.6B --url http://server:8000 -o bench.yaml
aiperf kube validate --strict bench.yaml

# 3. Cluster readiness: connectivity, API versions, RBAC, node capacity,
#    image pull-ability, endpoint reachability
aiperf kube preflight -i aiperf:latest -e http://server:8000 -w 1

# 4. Submit one AIPerfJob CR; the operator owns the JobSet, ConfigMap,
#    Role and RoleBinding via ownerReferences
aiperf kube profile --config bench.yaml --image aiperf:latest \
    --total-workers 8 --detach

# 5. Watch
aiperf kube attach                 # WebSocket progress via port-forward
aiperf kube list --watch           # CR phase table, refreshed
aiperf kube logs -f                # controller/worker pod logs

# 6. Collect once phase == Completed (harvest happens only on terminal phase)
aiperf kube results                # -> ./artifacts/<name>/

# 7. Retire / remove
aiperf kube shutdown               # let the finished controller pod exit
aiperf kube delete <name> --force  # CR + GC'd children; PVC results survive
```

`-w` on `preflight` counts worker **pods**; `--total-workers` on `profile`
counts worker **processes**, packed `runtime.workersPerPod` (default 10) per
pod. `--total-workers 8` is one pod, so preflight it with `-w 1`; for 8 worker
pods use `--total-workers 80 -w 8`. A total that is neither <= workersPerPod nor
a multiple of it is rejected rather than silently collapsed onto one pod.


Operator mode is chosen automatically when the `aiperfjobs.aiperf.nvidia.com`
CRD is present. `--operator` forces it while skipping the cluster-scoped CRD
probe — how a namespace-scoped tenant submits without cluster-wide RBAC; pair
it with an explicit `--namespace`, since skipping the probe also skips operator
discovery. On a terminal phase the operator copies results to a shared PVC (on
failure it salvages partial checkpoints); nothing is harvested mid-run.

### `--dry-run` fidelity

`--dry-run` never contacts the cluster, so the mode it prints is assumed, not
detected, and the CR is printed *before* the spec is schema-validated.

With `--operator` or `--no-operator` the flagged mode always wins and no probe
happens. With neither flag: a real run probes the CRD and falls back to direct
when it is absent, while `--dry-run` skips the probe and always renders the
`AIPerfJob` CR (a stderr note says so).

A hand-authored CR passed with `--config` prints snake_case keys as authored
(`ttl_seconds_after_finished`) but submits camelCase, and unknown envelope keys
print fine yet exit 1 on real submission. Use `validate` for a real verdict.

## Direct mode

Triggered by `--no-operator`, or automatically when the `AIPerfJob` CRD is
absent. The CLI creates `Role`, `RoleBinding`, `ConfigMap`, `JobSet` — plus a
`Namespace` when `--namespace` is omitted — itself, in that creation order.

It keeps the same image, JobSet topology (one controller pod + N worker pods),
run-config ConfigMap, per-namespace Role/RoleBinding (read/watch on pods, pod
logs, jobs, ConfigMaps, Services, Endpoints, Events, plus `patch` on JobSets
and on `aiperfjobs`/`aiperfjobs/status`; no create/update/delete verb), and live `attach` stream. An `AIPerfJob` YAML
still works as input — its JobSet-compatible deployment fields are projected
into the raw manifests with operator-mode override precedence. It is *not* the
operator Deployment, a CR, a PVC, a dashboard, or cross-job analytics.

### Trade-off matrix

| Feature | Operator mode | Direct mode |
|---|---|---|
| Submission object | `AIPerfJob` CR | `Role` + `RoleBinding` + `ConfigMap` + `JobSet` (+ `Namespace` if auto-created) |
| Status surface | `AIPerfJob.status.phase` | `JobSet` + pod status only |
| Web dashboard (operator results server, port 8081) | yes | no |
| Results persistence | operator PVC (durable) | controller pod's ephemeral volume only |
| TTL after finish | `ttlSecondsAfterFinished` 300 s | 28800 s (8 h), so pods outlive the run for retrieval |
| Admission / preflight | operator-side before admitting the CR | none; run `preflight` + `validate` first |
| Sweeps / multi-run | `AIPerfSweep` + sweep-controller pod | unsupported |
| RBAC footprint | cluster-scoped operator ServiceAccount | one Role + RoleBinding in the benchmark namespace |
| `results` default path | operator PVC, works post-pod-GC | requires `--from-pods` while the pod lives |
| Name collisions | operator reconciles | fails closed on any same-named non-Namespace resource |

### When direct mode is appropriate

No cluster-admin rights to install the CRD/operator; a single tenant namespace
whose RoleBinding cannot go cluster-wide; a one-off benchmark needing no
cross-run history; or a CI smoke test on an ephemeral cluster. Otherwise use
operator mode — durable results, dashboard/analytics, multi-user management.

### End-to-end direct-mode sequence

```bash
# 1. Deploy
aiperf kube profile --model Qwen/Qwen3-0.6B --url http://server:8000 \
    --image ghcr.io/nvidia/aiperf:v1.2.3 --total-workers 10 \
    --concurrency 100 --namespace aiperf-bench --no-operator --detach

# 2. Reattach any time before the JobSet TTL expires
aiperf kube attach --namespace aiperf-bench

# 3. Pull results off the controller pod BEFORE the TTL reaps it
aiperf kube results --from-pods --shutdown --namespace aiperf-bench
```

Resource names are `aiperf-<name>` for the JobSet; the ConfigMap adds
`-config`, the Role adds `-role`, the RoleBinding adds `-binding`. `<name>` is
`--name` or the auto-generated `<model>-<endpoint-type>-<phase-type>` slug. The
CLI prints one `Created <Kind>/<name>` line each; a
`Created Namespace/aiperf-benchmarks` line is prepended only when `--namespace`
is omitted, and an existing namespace is reused as `already exists`.

Preview without submitting — works with no cluster and no kubeconfig; the
memory estimate goes to stderr, so the redirect yields clean YAML:

```bash
aiperf kube profile --model Qwen/Qwen3-0.6B --url http://server:8000 \
    --image aiperf:latest --no-operator --dry-run > bench.yaml && \
    kubectl apply -f bench.yaml
```

### Direct-mode results retrieval

`--from-pods` port-forwards to the controller pod's API service (port 9090)
and downloads the exported artifacts. The default `--all` path has **no**
fallback tier — if that API is unreachable, retrieval fails outright. Only
`--summary-only` retries with `kubectl cp` against the `control-plane`
container's `/results`. The pod also runs a results sidecar on port 9091 over
the same volume, but `--from-pods` never targets it (only the operator's
completion fetch does). `--run` is rejected with `--from-pods`, since pods hold
only the latest run. `--shutdown` (effective only with `--from-pods`) tells the
controller API to exit after a successful download so the pod terminates.

An interactive foreground direct-mode `profile` (TTY stdout, no `--detach`)
already waits for the controller pod, tails its `control-plane` container, and
on completion downloads all artifacts plus per-pod logs into
`./artifacts/<job_id>/`, so the explicit `results` call is only needed for
detached and non-TTY runs.

### Direct-mode cleanup

`delete` and `cleanup` resolve `AIPerfJob`/`AIPerfSweep` CRs only; on a
direct-mode run they print not-found and exit 0 while the JobSet keeps running.
Tear down manually:

```bash
kubectl delete jobset      aiperf-<name>        -n <namespace>
kubectl delete configmap   aiperf-<name>-config -n <namespace>
kubectl delete role        aiperf-<name>-role    -n <namespace>
kubectl delete rolebinding aiperf-<name>-binding -n <namespace>
kubectl delete namespace   <namespace>          # only if dedicated to the run
```

TTL reaps completed pods eventually; ConfigMap, Role and RoleBinding stay.

### Direct-mode limitations

- No cross-job analytics, dashboard, leaderboard, compare, or run history.
- Results are lost if the pod is deleted before `results --from-pods` runs.
- No operator-side admission/validation; `--skip-endpoint-check` is a no-op.
- No CR lifecycle reconciler: `timeoutSeconds`, `resultsTtlDays`, `cancel` and
  `failurePolicy` need the operator. JobSet-native `ttlSecondsAfterFinished`
  and `keepFailedPods` still apply.
- No CR-level status: use `kubectl get jobset`, `kubectl get pods -l app=aiperf`,
  `aiperf kube logs`, and `aiperf kube list` (which falls back to JobSets).
- Refuses to adopt an existing Role, RoleBinding, ConfigMap or JobSet of the
  same name, whatever its phase — no owner CR or run UID can prove ownership.
  Use a unique `--name` or delete the prior resources.

## Per-command flag reference

`attach`, `logs`, `results`, `results list-runs`, `cancel`, `delete` and
`shutdown` all take an optional positional `job_id` (or sweep name), defaulting
to the last deployed benchmark; `list` takes an optional `job_id` to show one.

### `aiperf kube init` (local only)

| Flag | Default | Semantic |
|---|---|---|
| `-t`, `--template` | `minimal` | Bundled template name |
| `-l`, `--list` | false | List templates by category |
| `-s`, `--search` | none | Keyword search over name/description/tags/features |
| `-c`, `--category` | none | Filter listing by category (substring) |
| `-v`, `--verbose` | false | Show tags/features/difficulty in listings |
| `--model` | none | Pre-fill model name |
| `--url` | none | Pre-fill endpoint URL |
| `-o`, `--output` | stdout | Output file (prompts before overwrite) |
| `--job-name` | `my-benchmark` | `metadata.name` on the generated CR |

### `aiperf kube validate` (local only)

Positional: one or more YAML paths (required).

| Flag | Default | Semantic |
|---|---|---|
| `-s`, `--strict` | false | Fail on warnings, e.g. unknown spec fields |
| `-o`, `--output` | `text` | `text` or `json` |

Checks YAML structure, `apiVersion`/`kind`/`metadata.name`/`spec.endpoint`,
RFC 1123 names, the CR spec schema for the detected kind, deployment-field
extraction, worker count >= 1, and unknown spec fields. Exits 1 on failure.

### `aiperf kube preflight`

| Flag | Default | Semantic |
|---|---|---|
| `-i`, `--image` | none | Image to verify accessibility |
| `--image-pull-secret`, `--image-pull-secrets` | none | Pull-secret name to verify (repeatable) |
| `--secret`, `--secrets` | none | Referenced Secret name to verify (repeatable) |
| `-e`, `--endpoint-url` | none | Endpoint to test connectivity |
| `-w`, `--workers` | `1` | Planned worker pods, for resource projection |
| `-o`, `--output` | `text` | `text` or `json` |

Exits 1 when any check fails.

### `aiperf kube profile`

Accepts the full benchmark CLI surface (`-m`/`--model`/`--model-names`,
`-u`/`--url`, `-f`/`--config`, `--concurrency`, `--request-count`,
`--streaming`, `--endpoint-type`, ...) minus `--workers-max`, superseded by
`--total-workers`.

| Flag | Default | Semantic |
|---|---|---|
| `-d`, `--detach` | false | Exit after deploy; auto-enabled on non-TTY stdout (with a warning) |
| `--no-wait` | false | Don't wait for pod readiness; in operator mode returns as soon as the CR exists |
| `--attach-port` | `0` (ephemeral) | Local port for API port-forward; **direct mode only** |
| `--skip-endpoint-check` | false | Skip endpoint health validation (no-op in direct mode) |
| `--dry-run` | false | Print payload to stdout, contact nothing |
| `--operator` | false | Force operator mode, skip the CRD probe |
| `--no-operator` | false | Force direct mode |
| `--name` | auto slug | Benchmark name (DNS label, max 40 chars) |
| `--image` | none | AIPerf container image |
| `--image-pull-policy` | none | `Always` / `IfNotPresent` / `Never` |
| `--total-workers` | `10` | Total workers spread over pods by `runtime.workersPerPod`; a non-multiple total runs on a single pod |
| `--ttl-seconds` | `300` (operator), `28800` when unset in direct mode | Seconds to keep pods after completion |
| `--node-selector` | `{}` | JSON object or repeated `key=value` |
| `--tolerations` | `[]` | JSON object or array of objects |
| `--queue-name` | none | Kueue LocalQueue for gang-scheduled admission |
| `--priority-class` | none | Kueue WorkloadPriorityClass |
| `--annotations`, `--labels` | `{}` | `KEY=VALUE`, JSON object, or `--annotations.KEY VALUE` |
| `--image-pull-secrets` | `[]` | Pull secret names |
| `--env-vars` | `{}` | `NAME=value`, JSON object, or `--env-vars.NAME value` |
| `--env-from-secrets` | `{}` | `ENV_NAME=secret_name/key` form |

`--operator` and `--no-operator` together is a startup error. A config carrying
`sweep:` — or `multi_run:` needing multiple trials — is rejected with a pointer
to the sweep path; an `AIPerfSweep` CR is rejected in favour of `kubectl apply`.

Foreground operator mode polls the CR until `Completed`/`Failed`/`Cancelled` or
a hard 600 s timeout (`AIPERF_K8S_WATCH_DEFAULT_TIMEOUT_SECONDS`, max 86400);
Ctrl+C and the timeout both leave the cluster-side run intact.

### `aiperf kube attach`

| Flag | Default | Semantic |
|---|---|---|
| `-p`, `--port` | `0` (ephemeral) | Local port for the port-forward |
| `-v`, `--variation` | none | Sweep child index 0..199 -> `<sweep>-v<idx:02d>[-t<trial>]` |
| `-t`, `--trial` | none | Trial 0..9 within a variation; requires `-v` |
| `--ignore-not-found` | false | Exit 0 instead of 1 when the target does not exist |

### `aiperf kube list`

| Flag | Default | Semantic |
|---|---|---|
| `-A`, `--all-namespaces` | **true** | Search all namespaces; ignored when `--namespace` is set |
| `--running` / `--completed` / `--failed` | false | Phase filter; more than one exits 1 |
| `-w`, `--wide` | false | Add model, endpoint, error columns |
| `--watch` | false | Refresh until interrupted |
| `--interval` | `5` | Refresh seconds, with `--watch` |

Falls back to listing JobSets when no `AIPerfJob` CRs match.

### `aiperf kube logs`

| Flag | Default | Semantic |
|---|---|---|
| `--container` | all | Specific container |
| `-f`, `--follow` | false | Stream in real time |
| `--tail` | all | Lines from the end |
| `-o`, `--output` | stdout | Directory for per-pod log files |
| `-v`/`-t` | none | Sweep child addressing |
| `--ignore-not-found` | false | Exit 0 instead of 1 when the target is missing |

### `aiperf kube results`

| Flag | Default | Semantic |
|---|---|---|
| `--output` | `./artifacts/<name>` | Output dir (`./artifacts/<ns>__<job>__<epoch>` with `--run`; `./artifacts/sweep__<ns>__<name>` for a sweep) |
| `--from-pods` | false | Pull from the controller API instead of the operator PVC |
| `--all` / `-a` (negated by `--summary-only`) | `--all` | All artifacts vs. summary only; only `--summary-only` has the `kubectl cp` fallback |
| `--shutdown` | false | Stop the API service after download; only with `--from-pods` |
| `--port` | `0` (ephemeral) | Local port for the port-forward |
| `--operator-namespace` | auto-detected | Where the operator runs |
| `--run` | latest | Pin a historical run by epoch; incompatible with `--from-pods` |
| `-v`, `--variation` / `-t`, `--trial` | none | Download one sweep child instead of the whole sweep |

Exits 1 when the target cannot be resolved or any requested file fails to
download, partial sweep downloads included.

### `aiperf kube results list-runs`

Lists a job's historical runs from the operator. `-o`/`--output` is `text`
(default) or `json`; `--preview` annotates which runs current retention would
reap (read-only, latest always protected); `--operator-namespace` and
`-v`/`-t` behave as in `results`.

### `aiperf kube cancel`

| Flag | Default | Semantic |
|---|---|---|
| `-v`, `--variation` / `-t`, `--trial` | none | Sweep child addressing |
| `--kind` | inferred | `job` or `sweep` when both share a name |

Patches `spec.cancel: true`; already-terminal targets are a no-op. The operator
deletes the JobSet and sets `phase=Cancelled` **without harvesting** — download
results before cancelling.

### `aiperf kube delete`

| Flag | Default | Semantic |
|---|---|---|
| `-f`, `--force` | false | Skip the confirmation prompt |
| `--delete-namespace` | false | Also delete the namespace, only if AIPerf generated it as `aiperf-<job_id>` with matching auto-generated/job-id labels, UID and resourceVersion |
| `--kind` | inferred | `job` or `sweep` when both share a name |

Deletes the CR; JobSet, pods and ConfigMap follow via ownerReferences, and PVC
results are untouched. Without a TTY the confirmation auto-declines (as it does
for `cleanup`), so `--force` is mandatory in CI.

### `aiperf kube cleanup`

| Flag | Default | Semantic |
|---|---|---|
| `-a`, `--all` | false | Also remove running benchmarks (cancelled first) |
| `-f`, `--force` | false | Skip the confirmation prompt |
| `--dry-run` | false | List what would go, delete nothing |

Default scope is terminal only — jobs `Completed`/`Failed`/`Cancelled`, sweeps
`Succeeded`/`Failed`/`Cancelled`/`PartiallyFailed`.

### `aiperf kube shutdown`

`--local-port` (default `0` = ephemeral) sets the API port-forward's local
port. POSTs `/api/shutdown` to the controller API; HTTP 409 while the run is
still going — cancel it instead. Harvested results remain available.
