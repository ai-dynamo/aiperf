<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# Kubernetes Sweeps and Adaptive Search — Terminal Reference

## The model

One `AIPerfSweep` CR -> one sweep-controller pod (a JobSet the operator
creates) -> one child `AIPerfJob` per **(variation, trial)**, created
**sequentially**: the controller awaits each child's terminal phase before
creating the next, so peak cluster demand is one controller + one worker set
regardless of grid size. Size worker counts for a single child; what scales
with the grid is wall clock, not capacity. The kopf operator is
planner-agnostic — it validates the spec, provisions RBAC, creates the
controller JobSet, and rolls child status up; the adaptive/BO loop lives only
inside the controller pod.

| Sweep `type` | Variation count |
|---|---|
| `grid` | product of every parameter list length |
| `zip` | shared parameter-list length (lists must be equal length) |
| `scenarios` | number of `runs` (names unique and path-safe) |
| `sobol`, `latin_hypercube` | `samples` |
| `adaptive_search` | `maxIterations` (upper bound; early stop shrinks it) |

Child names are `<sweep>-v<NN>[-t<N>]`, labeled `aiperf.nvidia.com/sweep=<sweep>`;
the `-t<N>` suffix appears only when `multiRun.numRuns > 1` or
`multiRun.convergence` is set. Each child carries
`aiperf.nvidia.com/run-identity`, a SHA-256 of its generated `AIPerfJob.spec`;
a deterministic name is reused on restart only when ownership **and** identity
both match, otherwise the sweep fails rather than feeding a different config's
metrics into planner history.

## Hard caps (admission-time, `kopf.PermanentError` -> `status.phase: Failed`)

| Cap | Value | Notes |
|---|---|---|
| Variations | 200 | variation index 0..199 |
| Trials per variation | 10 | `multiRun.numRuns` 1..10, also a schema bound |
| Sweep CR name | 27 chars | child job id capped at 35; worst-case `-vNNN-tN` suffix reserved |
| BO iterations | `maxIterations` 2..200 | required field, no default |
| Inline aggregate in CR status | 600 000 bytes | `AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES`, range 10 000..900 000 |

Over-cap grids are rejected from a cheap shape count **before** expansion, so a
million-cell grid fails fast instead of stalling the operator. Dry-run and config
validation both pass on an over-cap sweep — count your grid yourself.

## Minimal adaptive/Bayesian `AIPerfSweep`

```yaml
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
metadata:
  name: bo-conc-llama8b        # <= 27 chars
  namespace: my-benchmarks
spec:
  image: aiperf:latest
  benchmark:
    models: [meta-llama/Llama-3.1-8B-Instruct]
    endpoint:
      urls: [http://vllm.bench.svc.cluster.local:8000/v1/chat/completions]
      type: chat
      streaming: true
    datasets: [{name: main, type: synthetic}]
    phases:
      - {name: profiling, type: poisson, rate: 1.0, duration: 120}
  sweep:
    type: adaptive_search
    planner: bayesian
    searchSpace:
      - {path: phases.profiling.concurrency, lo: 1, hi: 1000, kind: int, prior: log-uniform}
    objectives:
      - {metric: output_token_throughput, stat: avg, direction: maximize}
    maxIterations: 30
    nInitialPoints: 5
    randomSeed: 42
  multiRun: {numRuns: 3, cooldownSeconds: 30}
```

`numRuns: 3` runs three child jobs per proposed point and feeds the pooled objective
to the planner. `numRuns: 1` = one child per point, fastest.

## Multi-dimensional search

```yaml
spec:
  sweep:
    type: adaptive_search
    planner: optuna
    optunaSampler: botorch
    optunaAcquisition: qlognehvi      # multi-objective acquisition
    searchSpace:
      - {path: phases.profiling.concurrency, lo: 1,   hi: 500,  kind: int}
      - {path: phases.profiling.rate,        lo: 1.0, hi: 50.0, kind: real}
    objectives:
      - {metric: output_token_throughput, stat: avg, direction: maximize}
      - {metric: time_to_first_token,     stat: p95, direction: minimize}
    outcomeConstraints:
      - {metric: request_latency, op: "<=", bound: 2000.0}
    slaFilters:
      - {metricTag: time_to_first_token, stat: p95, op: lt, threshold: 200.0}
    maxIterations: 40
    nInitialPoints: 8                 # rule of thumb: >= 2 * len(searchSpace)
    objectivePooling: pooled
  multiRun: {numRuns: 2}
```

Search-space paths are rooted **inside** `benchmark:` — a `benchmark.` prefix is
rejected, as are duplicate paths. `prior: log-uniform` requires `lo > 0`; `hi` must be `> lo`.

## `sweep` block fields (adaptive_search)

Wire form is camelCase; snake_case is accepted on input.

| Field | Type / values | Default |
|---|---|---|
| `type` | `adaptive_search` | required discriminator |
| `planner` | `bayesian` \| `optuna` \| `monotonic_sla` \| `smooth_isotonic` | `bayesian` |
| `searchSpace[]` | `path`, `lo`, `hi`, `kind` (`int`\|`real`), `prior` (`uniform`\|`log-uniform`) | `kind=real`, `prior=uniform`; min 1 dim |
| `objectives[]` | `metric`, `stat` (`avg`\|`p50`\|`p90`\|`p95`\|`p99`), `direction` (`maximize`\|`minimize`), `threshold` | `stat=avg`, `threshold=null`; min 1 |
| `outcomeConstraints[]` | `metric`, `op` (`<=`\|`>=`\|`==`), `bound` | `[]` |
| `slaFilters[]` | `metricTag`, `stat` (avg,p1..p99,min,max), `op` (`lt`\|`le`\|`gt`\|`ge`), `threshold` | `stat=p95`; `[]` |
| `slaTiers[]` | `label` + `filters[]`; 2-10 tiers activates the multi-tier planner | `[]` |
| `maxIterations` | int, 2..200 | **required** |
| `nInitialPoints` | int >= 1; must be `< maxIterations` for `bayesian`/`optuna` | `5` |
| `plateauWindow` | int >= 2 | `8` |
| `plateauThreshold` | float > 0 (CoV) | `0.01` |
| `improvementPatience` | int >= 2 | `10` |
| `randomSeed` | int >= 0 | `null` -> derived from the CR's `metadata.uid` |
| `optunaSampler` | `gp` \| `tpe` \| `botorch` | `botorch` (falls back to `tpe` if the optional stack is absent) |
| `optunaAcquisition` | `logei`,`qlogei`,`qnei`,`qlognei` (single-obj) / `qehvi`,`qnehvi`,`qlognehvi` (multi-obj) | `null` |
| `optunaTerminator` | `regret` \| `emmr` \| `none` | `none` |
| `objectivePooling` | `mean` \| `pooled` | `mean` (no-op when `stat=avg`) |
| `constraintMode` | `penalty` \| `eic` | `penalty` — deprecated and ignored; both planners use Optuna's native `constraints_func` |
| `monotonicStabilityTrials` | int >= 1 | `2` |
| `slaReplicates` | int >= 0 (0 = auto) | `0` |
| `slaPrecision` | `tight` (10000) \| `normal` (1000) \| `coarse` (300) requests | `normal` |
| `slaWarmupSeconds` | float >= 0, `null` = auto `max(30s, 3x inter-batch)` | `null` |
| `cooldownSeconds` | float >= 0, between variations/iterations | `0.0` |

- Acquisition arity must match objective count (`qlognehvi` with 1 objective,
  or `qlognei` with 2, is a hard error).
- `improvementPatience` / `plateauWindow` / `plateauThreshold` apply only to
  `bayesian` and `optuna`; with `monotonic_sla` / `smooth_isotonic` they emit a
  UserWarning and are ignored.
- `iterationOrder` and `sameSeed` exist only on grid-family sweeps.

## `multiRun` block (trials within a variation)

| Field | Range | Default |
|---|---|---|
| `numRuns` | 1..10 | `1` |
| `cooldownSeconds` | 0..86400 | `0.0` |
| `confidenceLevel` | 0 < x < 1 | `0.95` |
| `setConsistentSeed` | bool | `true` |
| `varySeedPerTrial` | bool | `false` |
| `disableWarmupAfterFirst` | bool | `true` |
| `convergence.metric` | metric tag | required when `convergence` present |
| `convergence.stat` | avg,p50,p90,p95,p99,min,max | `avg` |
| `convergence.mode` | `ci_width` \| `cv` \| `distribution` | `ci_width` |
| `convergence.threshold` | 0 < x < 1 | `null` -> ci_width 0.10, cv 0.05, distribution 0.05 |
| `convergence.minRuns` | >= 2, must be `<= numRuns` | `2` |

## CLI flags that actually exist on the sweep subcommand

Submission takes a YAML file: a bare AIPerf config with top-level `sweep:` /
`multiRun:`, or an `AIPerfJob` CR. An `AIPerfSweep` CR is rejected — apply it
directly instead. Only these sweep-specific flags exist:

| Flag | Maps to | Default |
|---|---|---|
| `--trials N` | `multiRun.numRuns` | unset |
| `--cooldown S` | `multiRun.cooldownSeconds` | `0.0` |
| `--convergence-metric M` | `multiRun.convergence.metric` | unset |
| `--min-runs N` | `multiRun.convergence.minRuns` | `3` |
| `--max-runs N` | `multiRun.numRuns` (raised to at least this) | `10` |
| `--convergence-threshold F` | `multiRun.convergence.threshold` | unset (per-mode default: `ci_width` 0.10, `cv` 0.05, `distribution` 0.05) |
| `-d`, `--detach` | reserved; submission always behaves as detached | `false` |
| `--dry-run` | print the rendered CR as JSON, submit nothing | `false` |

`--min-runs`, `--max-runs`, and `--convergence-threshold` are silently ignored
unless `--convergence-metric` is also passed. `--max-runs` raises
`multiRun.numRuns` to *at least* N; it never lowers an authored value.

`aiperf kube sweep` inherits the full benchmark CLI surface, so
`--convergence-stat`, `--convergence-mode`, `--confidence-level`, the
`--search-*` family (`--search-space`, `--search-metric`, `--search-stat`,
`--search-direction`, `--search-max-iterations`, `--search-initial-points`,
`--search-random-seed`, `--search-planner`, `--search-percentile-pooling`,
`--search-sla`, `--search-sla-tier`, `--search-recipe`, `--search-style`) and
the SLO/goodput flags (`--slo-attainment-fraction`, `--goodput`) all parse. The
table above lists only the sweep-specific flags; anything not listed is either
inherited from the benchmark surface or authored in YAML. `--bo-constraint-mode`
parses but is deprecated and ignored. Passing `--convergence-metric` also forces
`sweep.iterationOrder: independent` when unset, and synthesizes a one-cell
sweep when there is no `sweep:` block. `--dry-run` renders the CR and returns;
it never expands variations, so it shows neither the variation list nor whether
you are over the caps.

## Status fields worth watching

| Field | Meaning |
|---|---|
| `status.phase` | `Pending`, `Running`, `Aggregating`, `Succeeded`, `PartiallyFailed`, `Failed`, `Cancelled` |
| `status.totalVariations` | upper bound (`maxIterations` for adaptive); rewritten down on early stop |
| `status.maxTotalRuns` | `totalVariations * multiRun.numRuns`, an upper bound |
| `status.completedRuns` / `failedRuns` | authoritative child tallies; cancelled children are their own bucket, not failures |
| `status.runEpoch` | int64 epoch key, also the on-disk results directory name |
| `status.runStates` | open object: pending / running / completed / failed / cancelled counts |
| `status.currentChildRef`, `status.currentCell` | active child (`name`, `index`, `label`); `currentCell.label` is the CURRENT column |
| `status.aggregation`, `status.aggregate`, `status.aggregateRef` | aggregation progress and inlined/pointed aggregate |
| `status.runtimeRef` | `jobSetName` and `sweepControllerHost` |
| `status.apiUrl` | `/api/v1/sweeps/<ns>/<name>` on the operator service |
| `status.conditions`, `status.error` | `ConfigValid` / `Failed` conditions and the rejection message |

`status` is preserve-unknown, so unlisted keys may appear. The BO trajectory
(every proposed point, per-iteration objective, running best, convergence
reason) lives in `search_history.json`, never on the CR; a compact projection
is served as `search_summary` on the sweep detail API route, `null` for
grid-family sweeps and before harvest. A `slaFilterCount` of zero makes every
`feasible` flag vacuously true. `bestTrials[].objectiveValues` is positional
against `objectives` and keeps explicit `null` slots.

Aggregates above the inline cap drop `confidence` first, then omit `children`
and set a truncation marker; read large aggregates through the results download
path, not `kubectl get -o json`.

`failurePolicy.onChildFailure` is `continue` (default) or `abort`;
`failurePolicy.maxFailures` defaults to `0` = unbounded. Some failures within
budget ends `PartiallyFailed`. `spec.benchmark`, `spec.sweep`, `spec.multiRun`,
`spec.variables`, `spec.randomSeed`, `spec.failurePolicy`, and
`spec.childMetadata` are immutable after creation.

## Mutual exclusion and the operator-managed gate

| Combination | Outcome |
|---|---|
| `kind: AIPerfJob` with `spec.sweep` set | rejected at admission (CEL `!has(self.sweep)`) |
| `kind: AIPerfSweep` without `spec.sweep` | rejected at admission (CEL `has(self.sweep)`) |
| A `sweep:` block nested inside `spec.benchmark` | silently pruned by structural-schema pruning; sweep axes belong on the parent CR |
| Two sweep types at once | not expressible — `sweep.type` is a discriminated union |
| Magic-list flags (`--concurrency 10,20,30`) inside a child | rejected inside the child pod |

Every controller and worker pod gets `AIPERF_OPERATOR_MANAGED=1`. Under that
flag an in-process multi-variation plan hard-fails with `SystemExit`, so the
cluster can never sweep on top of a sweep. **Adaptive search is exempt by
construction**: the controller proposes one point at a time and each child sees
a single-config plan, so the multi-variation shape never arises. The controller
also strips the parent `multiRun` block out of each child.

## Cancellation

```bash
kubectl -n <ns> patch aiperfsweep <name> --type=merge -p '{"spec":{"cancel":true}}'
```

1. A background poller in the controller re-reads the parent CR and flips an
   in-process cancel flag; the adaptive loop and the child-wait loop both check
   it at their await boundaries.
2. The running child is patched with `spec.cancel: true`, drains to a terminal
   phase, and still contributes its results.
3. Remaining iterations are skipped; aggregation runs over completed children;
   `search_history.json` is flushed with `convergenceReason: "cancelled"`.

An explicit parent cancel yields `Cancelled`; out-of-band cancellation of
individual children counts in a separate bucket and yields `Cancelled` only
when nothing completed. `search_history.json` is rewritten after every
iteration, so the completed prefix survives immediately; after a controller
restart, terminal children with matching run-identity are replayed into a fresh
planner before execution advances.

Per-variation cancel targets one child by index (`0..199`), optionally with a
trial index (`0..9`); the trial index requires the variation index. Deleting
the parent garbage-collects children through ownerReferences; results already
harvested to the operator PVC survive.

## Output layout

Under the operator results PVC (`AIPERF_RESULTS_DIR`, default `/data`):

```
<results-dir>/
  <namespace>/
    sweeps/
      <sweep-name>/
        latest.txt                       # pointer to newest epoch
        <sweep-run-epoch>/
          search_history.json            # adaptive only: trajectory + convergenceReason
          aggregate.json                 # durable parent aggregate
          children.json                  # child manifest
          sweep_aggregate/
            profile_export_aiperf_sweep.json
            profile_export_aiperf_sweep.csv
            manifest.json                # epoch lineage of every child run
            sampling_design.json         # sobol / latin_hypercube only
    <sweep-name>-v00-t0/                 # child, same layer as a standalone job
      <child-run-epoch>/
        profile_export_aiperf.json
        ...
    <sweep-name>-v01-t0/
```

The controller's results dir is an `emptyDir`; the operator harvests the whole
tree onto the PVC before deleting the JobSet. `sweep_aggregate/` uses one
mode-agnostic per-combination schema grouped by stamped variation values, so
readers need not know whether the sweep was grid or adaptive. Downloading a
sweep whose children only partially succeeded exits non-zero even though the
successful children land on disk.

Plotting runs once, in the controller, after cross-variation aggregation, and
only when auto-plot resolves true. With `plotRequired: true` a plot failure
fails aggregation and withholds the ready marker; with `false` it is logged and
the aggregate still becomes ready.

## Credential transport for sweeps

**Credential-bearing axes are rejected before any child is created**, for every
sweep type. The check walks grid/zip `parameters` keys, adaptive/QMC
`searchSpace` (and `dimensions`) paths, and `scenarios` `runs` bodies. A path
is credential-bearing when any dotted segment (case- and punctuation-normalized)
is, or ends/starts with, one of: `apiKey`, `xApiKey`, `authorization`,
`proxyAuthorization`, `token`, `apiToken`, `authToken`, `accessToken`,
`bearerToken`, `idToken`, `refreshToken`, `secret`, `clientSecret`, `password`,
`passwd`, `credential`, `awsAccessKeyId`, `signature`,
`ocpApimSubscriptionKey`, `xGoogApiKey`, `xFunctionsKey`, `aegSasKey`,
`xAmzSecurityToken` — **plus any `endpoint.headers.*` path at all**, and any
value that looks like a credential-bearing URL. `variables.<name>` axes are
covered too, since the credential name still appears as a segment. Rejection
message: *"Kubernetes sweeps cannot vary credential-bearing values."*
Per-variation secret transport does not exist: sweep values are copied into
child annotations, parent status, aggregates, and API responses.

Keep credentials **fixed** and inject them through Secret-backed pod env vars:

| Variable | Carries |
|---|---|
| `AIPERF_INJECTED_API_KEY` | endpoint API key (alias `OPENAI_API_KEY` accepted, but it can only rehydrate a key the config authored-then-redacted) |
| `AIPERF_INJECTED_HEADERS` | JSON object, string values only — credential-bearing headers |
| `AIPERF_INJECTED_ENDPOINT_URLS` | JSON list of strings — full endpoint URLs, replacing the authored list |

Submission fails with *"Kubernetes endpoint credentials must come from
Secret-backed pod environment variables"* if the config carries an API key,
sensitive header, or credential-bearing URL and the pod template has no
matching `valueFrom.secretKeyRef` env entry. Wire them with
`--env-from-secrets.AIPERF_INJECTED_API_KEY <secret>/<key>`, or directly:

```yaml
spec:
  podTemplate:
    env:
      - name: AIPERF_INJECTED_API_KEY
        valueFrom:
          secretKeyRef: {name: llm-endpoint, key: api-key}
      - name: AIPERF_INJECTED_HEADERS
        valueFrom:
          secretKeyRef: {name: llm-endpoint, key: headers-json}
```

Authored credentials are redacted before persistence, so the serialized run and
any ConfigMap stay secret-free; injected values are popped from the process
environment (never left in `/proc/<pid>/environ` for child services) and
overlaid onto the redacted placeholders at run start. Injected headers override
same-named authored headers; injected URLs replace the authored list. Any
placeholder still present after the overlay is a hard error naming the missing
variable. Labels, annotations, status, manifests, and API responses all run
through the same redaction projection.
