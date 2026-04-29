# AIPerf Kubernetes / Operator — Slop & Hallucination Audit

**Date:** 2026-04-27
**Branch:** `ajc/k8s` vs `origin/main`
**Method:** 7 parallel Opus 4.7 agents, each given an independent slice with explicit instructions to *verify* every suspected bug (grep, read library code, cross-check CRD) before reporting — no speculation. Findings are P0/P1/P2 prioritized. Investigation only; no edits made by the auditor agents.

**Total:** ~91 findings — **14 P0**, ~30 P1, rest P2/P3.

---

## Closeout (2026-04-27 cycle)

Same-day fix wave executed by 12 follow-up subagents (5 P0 + 7 P1/P2). All P0 closed; ~25 P1 closed; ~30 P2 closed (a small set deferred or verified false-positive). Final integration check: `uv run pytest -n auto tests/unit/` → **13,402 passed**, 9 pre-existing failures unchanged, **0 new regressions**.

### P0 wave (5 commits, 14 P0s closed)

| SHA | Closes | Subject |
|---|---|---|
| `6c8ec3bec` | P0 #1, #2, #3 | UI ServerMetrics dispatcher dict→list **and** `progress_client` `/health`→`/healthz` + `worker_groups` (cross-bundled by index race; code is correct, attribution scrambled) |
| `616b25511` | P0 #4, #5a, #5b, #14 | kube CLI: `validate -o json` logger silencing; `logs.py` → `kube_console`; `debug.py` accepts `KubeManageOptions`; `dashboard.py` `verify_api=True` |
| `e287edd15` | P0 #6, #7 (+ bonus P1) | sweep TTL converges on `status.completionTime`; `BenchmarkPlan.trials`/`ConvergenceConfig.max_runs`/CRD aligned at `le=20`; `aggregation_failed` now promotes top-level `phase=Failed` |
| `fb9fd186e` | P0 #8, #9, #10, #11 | preflight: `_run_check` catches `Exception`; type-based transient classification (replaces string-match); taint-aware node resources; PSA `restricted` → WARN; RBAC `review.status=None` raises ApiException |
| `bb822d644` | P0 #12 | orphan FastAPI deletion: `routers/api.py` + `metrics_utils.py` + `prometheus_formatter.py` + their tests (-1613 LOC) |

### P1/P2 wave (7 commits + 1 orphaned-but-content-at-HEAD)

| SHA | Slice | Findings closed |
|---|---|---|
| `659afb2b9` | kube CLI | `_debug_extract.py` ApiClient threaded; `watch.py` `Literal["rich","text","json"]`; `init.py` errors → `kube_console.print_error`; `generate.py` stderr console; `sweep.py` JSON `highlight=False`; `list_.py` `console.clear()`; `-A`+`--namespace` help clarified. *Skipped: `watch.py` Ctrl-C UX polish.* |
| `933d615b7` | operator core | Cancel-claim race fixed by mirroring `monitor.py` pre-claim cancellation check; double `sb.finalize()` collapsed; `k8s_helpers.retry_with_backoff` and `lifecycle.send_shutdown` `except Exception` narrowed; dead `_CONDITIONS_FILE` and `_validate_namespace_and_job_id` deleted; `kopf.event` direct call routed through `events.index_update_failed`. |
| `bec4460fc` | preflight + memory | `parse_image_ref` returns 4-tuple with digest distinct from tag; `orjson` import hoisted; `_DEFAULT_PHASE_REQUEST_COUNT` and `_PHASE_AVG_SEC_PER_REQUEST` lifted to `constants.py`; `_get_worker_pod_limit_mib` unused params dropped; `_check_namespace` 403→SKIP; CLI `_check_jobset_crd` aligned to FAIL; `_check_dns` uses `k8s-app=kube-dns` label. *Skipped: `validate.py` source-path prefix — audit premise wrong (path already shown).* |
| `b29d447b3` + `e8bc273c8` | kubernetes lib | `watch_orchestrator` `get_running_loop`; `find_aiperf_job`/`find_aiperf_sweep` 404 returns None instead of cluster-wide leak; `EVENT_BUS_PROXY_PUB_FRONTEND`/`SUB_BACKEND` Field-promoted; `save_pod_logs --all-containers --prefix`; `goodput_rps` clamped; watchdog bg-tasks retained; `port_forward` routed through `subproc`; `_PortForwardSettings`/`_ProgressStreamSettings`/`_WatchSettings` Field groups; `delete_namespace` re-raises non-404; `_consume_ws_messages` handles `WSMsgType.CLOSE`; pollers aligned. *Skipped: `K8sWatchdogSource.create` — verified false-positive (used by E2E test helpers).* |
| `2ce7a1ede` (orphan; content at HEAD) | routers | `DatasetMixin` dropped from `FastAPIService`; orphan `ws_manager` removed; `routers/kubernetes.py` deleted; `get_service` canonical in `depends.py`; new `api/pod_state_rpc.py` shared by `progress.py`+`debug.py`; `_resolve_dest_path` rejects `('','. ','..')`; `_GET_POD_STATES_TIMEOUT` promoted to `Environment.API_SERVER`. *Skipped: `progress_models` retry constants (out of slice); `/api/v1/config/*` namespace co-ownership (contract break).* |
| `5f927606e` | sweep | RFC3339 `fromisoformat` for `creationTimestamp`/`completionTime`; grid variation order `sorted(variables.keys())`; `_idle_until_terminated` removed (pod returns 0/1; `restartPolicy: OnFailure` confirmed); `_AGGREGATE_INLINE_MAX_BYTES = 600_000` cap with `confidence` drop fallback; new `parent_running()` writer with atomic JSON-patch test+replace. *Skipped per audit guidance: `child_rollup` k8s_client consolidation; `(variation_index, trial_index)` tracking; `is_my_child` strictness; `_collect_run_result` empty-summary (mooted — `CIWidthCriterion` already handles missing).* |
| `f871e938e` | UI JS | dashboard active-card phantoms (`j.backend`/`j.gpuConfig`) dropped; breadcrumb double-`decodeURIComponent` removed; non-hash `<a href="/sweeps/…">` prefixed with `#`; dead `?? '---'` removed in `realtime-metrics.js`; Sweeps added to command palette; archived-children manifest table replaces all-`---` `JobTable` reuse. *No-op at HEAD: scatter `\n` tooltip, `cluster.gpuCount` fallbacks, `sortDir` hardening — all in user's WIP M-state, not at HEAD.* |

### Verifier wins (verify-before-fix protocol earned its keep)

- **Kubernetes lib agent:** `K8sWatchdogSource.create` audit-flagged as dead code → actually used by `tests/kubernetes/helpers/watchdog.py:19` for E2E helpers. Skipped deletion.
- **Preflight agent:** `validate.py` source-file path prefix audit-flagged → already exists in both display paths (CLI and JSON modes). Adding it would duplicate output. Skipped.
- **UI JS agent:** Several audit "bugs" (scatter `\n` tooltip, `cluster.gpuCount` fallbacks, sortDir hardening) turned out to be in the user's WIP M-state, not at HEAD. Skipped fixing what wasn't broken.

### Cleanup TODO before any PR

1. **Commit-message attribution scrambled** by parallel-agent index races. `e8bc273c8` titled "k8s-lib slice 4 P2s" actually contains sweep-slice-3 content too. `5f927606e` titled "sweep" contains slice-6 routers content. Code is correct on disk. Recommend `git rebase -i origin/main` to relabel.
2. **Out-of-scope follow-ups discovered:** ZMQ ports `5663/5664` also hardcoded in `src/aiperf/config/{_zmq_tcp,_models_comm,_zmq_dual_bind}.py` (different config plane, not in any audit slice). Pre-existing test failures (`test_kube.py::TestResultsCommand*` MagicMock-vs-AsyncMock; `test_resolvers.py::test_legacy_epoch_match`; `test_cli_kube_results_list.py` `_print_runs_table` ImportError) want their own triage pass — none caused by the audit fixes.

---

## Cross-slice patterns

Worth tackling holistically in a follow-up:

1. **Wire-protocol mismatches** between client and server — *the dominant failure mode.* 4 of the 14 P0s are clients reading field names that no longer exist (or never existed) on the response. UI dashboard ↔ FastAPI routers, operator `progress_client` ↔ controller API, sweep CR status fields. End-to-end contract test would catch all of these.
2. **Hallucinated frontend fields** — `j.backend`, `j.gpuConfig`, `j.tokenThroughput`, `cluster.gpuCount`, `cluster.nodeCount`. None on the Pydantic response models. `??` fallback chains hide the drift permanently.
3. **Status field naming chaos in sweep CRD** — `phase=Running` declared in CRD enum but never written by anyone; `aggregation_failed` writes nested `aggregation.phase=Failed` but never sets terminal top-level `status.phase=Failed`; TTL reaper reads `status.completedAt` while writers use nested `aggregation.completedAt` and CRD declares `completionTime`. Three writers, three name conventions, none matching the reader.
4. **Branch-only bare `except Exception` regressions** — added on `ajc/k8s` without `# noqa: BLE001`. Should fail `make check-ruff-baselined`. Worth checking why baseline enforcement let them through.
5. **Tunable constants in module scope** (per `feedback_constants_in_environment_py.md` rule "must live as a Field on `_XxxSettings`") — ZMQ proxy ports `5663/5664`, `_PORT_FORWARD_TIMEOUT` group, `_GET_POD_STATES_TIMEOUT`, `MAX_RETRIES`/`INITIAL_BACKOFF_SEC`, memory-estimator magic floats. Sweep into `_XxxSettings` together.
6. **Cross-namespace data leak** in `client_jobs.find_aiperf_job`/`find_aiperf_sweep` — when `namespace` is given and 404 returned, falls through to a cluster-wide list and matches by name, returning a same-named CR from a *different* namespace.

---

## P0 punch list (consolidated)

| # | File | Bug | Slice |
|---|---|---|---|
| 1 | `src/aiperf/operator/progress_client.py:361` | `check_health()` hits `/health`; server only serves `/healthz`. Every controller liveness probe reads "down forever." | routers |
| 2 | `src/aiperf/operator/progress_client.py:317` | `get_worker_startup_states()` reads `data["workers"]`; server response renamed to `worker_groups`. Always returns empty mapping. | routers |
| 3 | `src/aiperf/api/static-v2/lib/ws-dispatch.js:168` | Guards with `Array.isArray(msg.endpoint_summaries)` against a `dict[str, …]` payload. Entire ServerMetrics card silently dead. HTTP fallback in `app.js:52` discards its response too. | UI JS |
| 4 | `src/aiperf/cli_commands/kube/validate.py:60-85` | `-o json` mode does NOT silence the `aiperf.kube` logger. Every other `-o text|json` command does. JSON output corruptible by stray INFO logs. | CLI |
| 5a | `src/aiperf/cli_commands/kube/logs.py:52,75` | Raw `print(...)` for streaming pod log lines. Bypasses `kube_console`, bypasses logger gating. | CLI |
| 5b | `src/aiperf/cli_commands/kube/debug.py:227-256` | Re-rolls `--namespace`/`--kubeconfig`/`--kube-context` instead of accepting `KubeManageOptions`. Help-text drift inevitable. | CLI |
| 6 | `src/aiperf/operator/handlers/sweep/lifecycle.py:142` | TTL reaper reads `status.completedAt`. Nobody writes that field. CRD declares `completionTime`; writers populate nested `aggregation.completedAt`. TTL effectively measured from creation timestamp → CRs reaped before they finish. | sweep |
| 7 | `src/aiperf/sweep_controller/plan_builder.py:56` | `convergence.maxRuns > 10` crashes the sweep-controller pod with a Pydantic ValidationError: `BenchmarkPlan.trials` has `le=10`, but `ConvergenceConfig.max_runs` and the CRD `maxRuns` field have no upper bound. | sweep |
| 8 | `src/aiperf/operator/preflight/_checker.py:166-189` | `_run_check` only catches `(ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError)`. Tier 1/2 checks raising anything else (`RuntimeError`, `urllib3 MaxRetryError`, `ssl.SSLError`) propagate uncaught and abort the rest of preflight. | preflight |
| 9 | `src/aiperf/operator/preflight/_resources.py:29-92, 139-189` | `_check_node_resources` aggregates allocatable across all Ready nodes ignoring taints/tolerations. Reports "sufficient resources" on a cluster of 50 NoSchedule-tainted nodes. | preflight |
| 10 | `src/aiperf/operator/preflight/_infra.py:325-332` | `compatible_levels = {"privileged","baseline","restricted"}` — every standard PSA level rubber-stamped as compatible. Workload-vs-PSA mismatch only surfaces at admission time. | preflight |
| 11 | `src/aiperf/kubernetes/preflight_checks.py:194-230` | RBAC permission check returns `bool(review.status and review.status.allowed)` — `review.status=None` (transient/malformed apiserver response) is silently treated as "denied" and surfaces as a hard FAIL with no SKIP/WARN path. | preflight |
| 12 | `src/aiperf/api/routers/api.py` (orphan) | References `svc._metrics`, `svc._progress_tracker`, `svc._worker_tracker`, `svc.get_info_labels()` — none exist on `FastAPIService`. Loaded only by test conftest, but if anyone wires it into prod, every endpoint instant-AttributeErrors. Pair with orphan `metrics_utils.py`/`prometheus_formatter.py` (api/ duplicate of metrics/). | routers |
| 13 | (incident from #5b) | `KubeManageOptions` pattern parity broken — see #5b above. | CLI |
| 14 | `src/aiperf/cli_commands/kube/dashboard.py:164-191` | `start_port_forward(..., verify_api=False)` only confirms `kubectl port-forward` started its subprocess; doesn't wait for the operator HTTP server. `webbrowser.open(url)` runs immediately → "connection refused" tab when the pod is `ContainerCreating`. *Listed P1 by the CLI agent but raised here as P0-adjacent because it's the one a user notices first.* | CLI |

---

## Findings by slice

### Slice 1 — Kube CLI commands (`src/aiperf/cli_commands/kube/*`)

12 findings, mostly polish; 5 actionable P0/P1.

**P0**

- **validate -o json doesn't silence aiperf.kube logger** — `validate.py:60-85`. Mirror the `preflight.py:96-113` `try/finally` pattern.
- **logs.py uses raw `print(...)`** — `logs.py:52, 75`. Replace with `kube_console.console.print(line, highlight=False, markup=False)`.
- **debug.py ignores `KubeManageOptions`** — `debug.py:227-256`. Every sibling command accepts `manage_options: KubeManageOptions | None`. Drop local `--namespace`/`--kubeconfig`/`--kube-context` params.

**P1**

- **`_debug_extract.py:31`** instantiates `ApiClient()` outside `k8s_client()`. Pass the open client through.
- **`watch.py:34-40` `output: str`** instead of `Literal["rich","text","json"]`. Typos silently launch TUI.
- **`init.py:133`** `print(str(e))` to stdout for an error path. Use `kube_console.print_error` + `SystemExit(1)`.
- **`dashboard.py:164-191`** opens browser before the port-forward is actually serving. Pass `verify_api=True`.

**P2**

- `generate.py:106` raw `print(..., file=sys.stderr)` for memory-estimate banner.
- `sweep.py:123-125` JSON dry-run misses `highlight=False`.
- `list_.py:23-26, 87` `-A` default `True` + explicit `--namespace` silently demotes.
- `watch.py:111-124` no clean Ctrl-C exit message.
- `generate.py:148-153` raises `SystemExit("...")` instead of `cli_utils.raise_startup_error_and_exit`.
- `list_.py:157` raw `print("\033[2J\033[H")` — use `kube_console.console.clear()`.

### Slice 2 — Preflight, memory estimator, results sidecar

16 findings, **5 P0**.

**P0**

- **CLI RBAC `_run_permission_check`** silently maps `review.status=None` → denied. `preflight_utils.py:42-55` returns `bool(... and ...allowed)`. Distinguish `allowed=None` from `allowed=False`.
- **Operator `_run_check` exception tuple** too narrow. Tier 1/2 uncaught exceptions propagate out of `run_all`. Broaden to `Exception` like the CLI variant.
- **`_check_node_resources` ignores taints**. False PASS on unschedulable clusters.
- **PSA check rubber-stamps every level**. Only privileged/baseline are safe; restricted needs a dry-run.
- **Controller in-process `/api/results/list` and `/api/results/files/{filename}` don't gate on the ready marker** that the sidecar enforces. CLI `aiperf kube results --all` mid-run silently downloads partial exports.

**P1**

- `_run_check:174` "transient" downgrade by string-match `"connect" in error_str` — turns hard FAILs into WARNs on any error message containing "connect". Gate by exception type.
- `parse_image_ref` mis-classifies digest references as tags — `image@sha256:...` returns `tag="sha256:..."`. `_workload.py:106` `not tag` warning never fires for digests. Display in `preflight_capacity_checks.py:387-392` shows `Tag: sha256:abcdef...`.
- `_workload.py:175` `import orjson` inside dry-run failure path; should be module-top.
- Memory estimator magic literals `1000` (`params.py:215, 234`) and `2.0 sec/req` (`params.py:230-239`) — hoist to `constants.py` or `_XxxSettings` Field.
- `estimator.py:48-54` `_get_worker_pod_limit_mib(workers_per_pod, rp_per_pod)` ignores both args — misleading API.

**P2**

- `_check_namespace` 403 returns "HTTP 403" error instead of SKIP/WARN.
- CLI `_check_jobset_crd` returns WARN on non-404 errors but operator variant returns FAIL — align.
- `validate.py:172-176` validation messages don't include source-file path.
- `_check_dns` substring match on `coredns` matches `coredns-monitoring`. Use label selector.

### Slice 3 — Sweep controller + sweep handlers + worker_pod_manager

12 findings, **2 P0** + 6 P1s. Sweep status-machine is the densest source of bugs.

**P0**

- **TTL reaper field mismatch** — `sweep/lifecycle.py:142` reads `status.completedAt`; writers populate `aggregation.completedAt`; CRD declares `completionTime`. TTL never measures actual completion.
- **`convergence.maxRuns > 10` crashes controller pod** — `BenchmarkPlan.trials` `le=10` not mirrored on `ConvergenceConfig.max_runs` or the CRD.

**P1**

- `_epoch_from_creation_ts` uses `strptime("%Y-%m-%dT%H:%M:%SZ")`; sub-second RFC3339 timestamps return `"0"` and break epoch isolation. Use `datetime.fromisoformat(...)`.
- `_expand_grid_sweep` uses `list(variables.keys())` against an apiserver-rewritten map (alphabetized at storage per `gotcha_k8s_crd_object_map_keys_alphabetized.md`). Variation order shifts between submit and read.
- Sweep-controller pod idles forever (`while True: sleep(3600)`) when no `ttlSecondsAfterFinished` is set. JobSet `completions=1` — pod never exits, no SIGTERM, leaked pod.
- `_load_aggregate_for_cr` patch can exceed K8s 1MB CR size limit on large sweeps. Bound or rely solely on disk path.
- CRD declares `phase=Running` but no code writes it. Parent jumps `Pending → Aggregating`.
- `aggregation_failed` writes nested `aggregation.phase=Failed` but doesn't promote to top-level `phase=Failed` — CR stuck at `Aggregating`, TTL never fires.

**P2**

- `child_rollup` opens 4 separate `k8s_client()` contexts per tick. Hold one client per sweep.
- `BenchmarkPlan.trials=int` hard-coded uniform; adaptive convergence may emit fewer/more per cell. Track `(variation_index, trial_index)` in `RunResult`.
- `is_my_child` uid+label match too strict — false positive `ChildNameConflictError` on quick resubmit.
- `_collect_run_result` empty-summary returns `success=True` with empty `summary_metrics`, mis-feeding adaptive convergence.

### Slice 4 — Kubernetes lib (`src/aiperf/kubernetes/*` excl. preflight/results)

14 findings, no P0s.

**P1**

- **`watch_orchestrator.py:141`** `asyncio.get_event_loop()` inside async coro. DeprecationWarning today; `RuntimeError` on newer Python. Switch to `get_running_loop()`.
- **`client_jobs.find_aiperf_job:147` & `find_aiperf_sweep:328`** — when `namespace` is given and direct GET 404s, falls through to `list_cluster_custom_object(field_selector=None)` with name match in Python → returns same-named CR from a different namespace. Move cluster-wide fallback to the `namespace is None` branch only.
- **`jobset_builder.py:212-213`** ZMQ proxy ports `5663`/`5664` hardcoded — promote to `K8sEnvironment.PORTS` Fields.
- **`logs.py:57-65`** `save_pod_logs` only captures default container; controller pod has 5+. Pass `--all-containers=true --prefix`.

**P2**

- `watch_pollers.py:132-136` `goodput_rps` math can go negative when `error_count > request_count` from different windows. Clamp.
- `watchdog_pod_checks.py:177-178` `asyncio.create_task(...)` fire-and-forget; loop only weak-references. Retain in a set + `discard` callback.
- 5 sites of `except Exception: pass` swallowing 410-Gone, 403, 5xx alike (`watchdog.py:311, 364, 397, 421`, `watchdog_pod_checks.py:191`). At minimum log at DEBUG; differentiate 403 from 5xx.
- `port_forward.py:50-54, 159-163` direct `asyncio.create_subprocess_exec` instead of `subproc.run_command`.
- Module-level tunables in `port_forward.py`, `progress_stream.py`, `watch_diagnosis.py` should be `_XxxSettings` Fields.
- `client_jobsets.delete_namespace:225-235` swallows non-404 failures, returns `None` for both success and failure. Re-raise.
- `cli_helpers._open_api_client` and `K8sWatchdogSource.create` instantiate `ApiClient()` outside `k8s_client()`. The latter is dead-code (`watch.py:209-210` doesn't use it).
- `progress_stream._consume_ws_messages:29-39` checks `WSMsgType.CLOSED` but not `WSMsgType.CLOSE` — abnormal close codes (non-1000) don't surface as errors.
- `watch_pollers.PodPoller.poll:303` and `EventPoller.poll:338` only catch `ApiException`; `CRPoller._get_raw_cr:266` correctly catches `(aiohttp.ClientError, asyncio.TimeoutError, OSError)` too. Align.

### Slice 5 — Operator core handlers, lifecycle, completion claim

7 findings. The slice is well-organized — durable claim, kopf signatures, k8s_client all correct.

**P1**

- **`completion.handle_completion:92-97`** — `try_claim_completion` is called by `lifecycle.on_benchmark_complete:158`, then `handle_completion` short-circuits on `is_cancellation_requested` and returns silently. Claim annotation stays on the CR; if `on_delete` arrives before `_recover_orphaned_completion_claim` can run, the CR is GC'd and the claim is lost. Either release the claim before returning, or check cancellation before claiming.
- **`completion.handle_completion:129 & :386`** — `sb.finalize()` called twice on the index-failure path. `status.py:235` says exactly-once. Currently safe but breaks under future change.
- **`k8s_helpers.retry_with_backoff:52`** — branch-new file with bare `except Exception` and no `# noqa: BLE001`. Hides programmer errors (`TypeError` on `coro_factory`, etc.) behind quiet retry.
- **`lifecycle.on_benchmark_complete:172`** — branch-new `except Exception as e:` with no noqa. Narrow to `(aiohttp.ClientError, asyncio.TimeoutError, OSError)` matching monitor.py.

**P2**

- `sweep_union.py:31` dead `_CONDITIONS_FILE` constant.
- `_completion_fetch._validate_namespace_and_job_id:247-270` dead defensive code — guards traversal patterns that DNS-1123 inputs cannot produce.
- `completion.py:387` raw `kopf.event` instead of `events.post_event` wrapper.

### Slice 6 — Operator + API HTTP routers

16 findings, **3 P0** + 6 P1s.

**P0**

- **`progress_client.check_health:361`** — wrong path `/health`.
- **`progress_client.get_worker_startup_states:317`** — wrong response key `workers`.
- **Orphan `api/routers/api.py`** — references nonexistent `FastAPIService` attrs. Pair with orphan `api/metrics_utils.py` and `api/prometheus_formatter.py` (different `_extract_metric_labels` set than `metrics/prometheus_formatter.py`).

**P1**

- **Two `prometheus_formatter` modules** — `api/` keeps `{"model","concurrency"}`; `metrics/` excludes `{"config","version"}`. Different cardinality semantics.
- **`api_service.FastAPIService:50`** inherits `DatasetMixin` and re-subscribes `DATASET_CONFIGURED_NOTIFICATION` already handled by `DatasetRouter:73`. Double-handling; mixin state is dead.
- **`api_service.FastAPIService:79`** builds a `WebSocketManager` orphan; `/ws` uses `WebSocketRouter.ws_manager` (different instance). `service.ws_manager.broadcast` silently no-ops.
- **`api/routers/kubernetes.py`** orphan — duplicate of `core.py` `/healthz`+`/readyz`, never registered.
- **Two `get_service` definitions** in `api_service.py:39` and `depends.py:16`. Pick one.
- **`progress.py:170-208` and `debug.py:104-123`** duplicate the GET_POD_STATES RPC + bus-cache fallback inline. Already drifting (one rebuilds AggregateWorkerStatus, the other returns raw payload). Extract to a helper.

**P2**

- Module-level magic floats `_GET_POD_STATES_TIMEOUT` and `progress_models.MAX_RETRIES`/`INITIAL_BACKOFF_SEC`/`BACKOFF_MULTIPLIER` should be `_XxxSettings` Fields.
- `progress.py:200, debug.py:119` redundant `(asyncio.TimeoutError, Exception)` tuple — `Exception` already covers `TimeoutError`.
- `progress_client._resolve_dest_path:644-653` — `Path("..").name == ".."`, can write one dir up. Reject `('', '.', '..')`.
- `/api/v1/config/*` namespace co-owned by `config.py` (`/retention`) and `results_analytics.py` (`/{ns}/{job_id}`). Load-order dependent; CR named `retention` would misroute.
- Orphan `metrics_utils.format_metrics_json` diverges from prod `metrics.format_metrics_json` (different return type, different version lookup, different label set).

### Slice 7 — Operator UI JS (`ui-v1/`, `ui/`, `static-v2/`, `static/`)

14 findings, **1 P0** + 4 P1s.

**P0**

- **`static-v2/lib/ws-dispatch.js:168-172`** — `Array.isArray(msg.endpoint_summaries)` against a `dict[str, …]` payload. Entire ServerMetrics card silently dead. HTTP fallback in `static-v2/app.js:52` discards its response too. Convert dict → list at dispatch time and HTTP bootstrap.

**P1**

- **`ui-v1/pages/dashboard.js:142, 297, 376, 378, 487, 505`** + **`ui/views/run.js:648, 655, 1385`** read `j.backend`, `j.gpuConfig`, `j.tokenThroughput`. None on `AIPerfJobInfo` (`kubernetes/models.py:354-454`). `??` fallback chains hide the drift.
- **`ui-v1/pages/dashboard.js:161`** — scatter tooltip returns a single string with `\n`. Chart.js v4 needs `string[]` for multi-line; `\n` rendered literally.
- **`static-v2/app.js:52`** — `getServerMetrics` HTTP bootstrap discards its response body (apply is a comment).

**P2**

- `ui-v1/pages/dashboard.js:58-59` cluster banner `cluster?.gpuCount ?? cluster?.gpu_count ?? cluster?.nodeCount ?? cluster?.node_count` — none of those exist on `ClusterResponse`. Drop fallbacks.
- `ui-v1/components/breadcrumb.js:48-84` double `decodeURIComponent` (router already decoded). `URIError` on stray `%`.
- `ui-v1/components/job-table.js:177`, `pages/sweeps.js:247`, `pages/job-detail.js:1749` — `<a href="/sweeps/...">` non-hash paths. Middle-click 404s.
- `static-v2/components/realtime-metrics.js:256-257` `fmtInt(x) ?? '---'` — `fmtInt` already returns `'---'`. Dead code.
- `ui-v1/pages/sweep-detail.js:96-108` cleanup ordering can flash "Loading..." on terminal-state transition.
- `ui-v1/components/command-palette.js:6-12` PAGES list missing Sweeps.
- `ui-v1/lib/api.js:11` Content-Type on GET requests forces CORS preflight. Same-origin today, harmless.
- `ui-v1/components/job-table.js:46` `-sortDir` flip relies on numeric type; stale string → silent NaN.
- `ui-v1/components/chart-wrapper.js:78-85` `JSON.stringify(options)` change-detection drops closure callbacks → tooltip closures captured at first mount only.
- `ui-v1/pages/sweep-detail.js:137, 459` — archived-children flow feeds `ChildrenManifestEntry` into `JobTable`; missing `phase`/`workersReady`/`throughputRps`/`created` columns render `---`.

---

## Notes

- **No agent reported hallucinated kubernetes_asyncio APIs, kopf signatures, or cyclopts surfaces.** Library calls are well-grounded. Hallucinations cluster in *data shapes flowing across the wire* — exactly the boundaries where end-to-end tests are scarce.
- **Pre-existing baseline violations are not flagged.** Only NEW `BLE001`/`S110`/`S112` in branch-only files.
- **Memory-cited gotchas confirmed in code:** `gotcha_k8s_crd_object_map_keys_alphabetized.md` (sweep grid expansion), `feedback_constants_in_environment_py.md` (multiple module-level tunables), `feedback_never_aggregate_across_runs.md` (no violation found in sweep aggregator — clean).
