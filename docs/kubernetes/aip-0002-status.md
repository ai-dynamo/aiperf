# AIP-0002 Kubernetes Deployment — Implementation Status

Status tracker for [AIP-0002 Kubernetes Deployment Enhancement](https://raw.githubusercontent.com/ai-dynamo/enhancements/1a643db98375863ba9892130edeb5e076ee78e91/deps/AIP-0002-kubernetes-deployment.md) against the current `ajc/k8s` branch.

## Legend

| Emoji | Meaning |
|-------|---------|
| ✅ | **Done** — implemented, matches spec intent |
| 🌟 | **Exceeds spec** — implementation goes meaningfully beyond AIP-0002 (e.g., already in the spec's explicit "future work", more operational flexibility, richer lifecycle, production-tuned defaults validated at 1M+ concurrency) |
| 🟡 | **Partial** — some paths done, others missing |
| 🔵 | **Divergent** — implemented differently with no clear better/worse (naming, command namespacing, equivalent algorithms) |
| 🔴 | **Not started** — no code located |
| ⚪ | **Deferred** — explicitly out of MVP scope / future work |
| ❓ | **Unclear** — needs a human to confirm |

> **Status of AIP-0002 itself — needs amending.** The original AIP-0002 scopes the MVP to CLI + JobSet with the full operator as explicit post-MVP future work. The `ajc/k8s` branch has moved well past that: the operator + `AIPerfJob` CRD + dashboard + richer lifecycle is the intended design now, not a stretch goal. Many 🌟 rows below reflect this shift (e.g., `podTemplate.volumes` for dataset mounting beats `--dataset-pvc` by supporting any volume type). **Treat AIP-0002 as an outdated v0; this doc is the current source of truth until the AIP is revised or an AIP-0003 is authored for the operator design.**

> **High-level posture:** AIP-0002 scopes the MVP to **CLI + JobSet** and lists a full operator as explicit post-MVP future work. The `ajc/k8s` branch has already built that future work: a **kopf-based operator with an `AIPerfJob` CRD**, a per-job FastAPI router, a dashboard UI, per-service container isolation (finer than the spec's 4-container hybrid), cross-job results persistence via an operator-wide PVC, and production defaults validated at 1M+ concurrency. Where the branch has jumped past the MVP into future-spec territory, features are flagged 🌟 rather than 🔵. True 🔵 divergences below are mostly naming or command-namespace deltas (e.g., `aiperf kube profile` vs `aiperf profile --kubernetes`).

> **Other notable deltas before reading the tables:**
> - All kube CLI lives under `aiperf kube <cmd>`, not top-level `aiperf <cmd>`. Where the spec says `aiperf list`, today's equivalent is `aiperf kube list`.
> - `aiperf profile --kubernetes` does not exist; deployment is `aiperf kube profile`.
> - Default namespace is a shared `aiperf-benchmarks`, not per-job `aiperf-{job_id}`.
> - Worker-pod layout packs `worker-group-manager` + N workers + M record-processors into one pod, not the spec's 2-container Worker+Sidecar ratio (though the pod-level co-location intent is preserved).
> - HTTP endpoints are hosted by a dedicated `FastAPIService` (ServiceType.API) and by the operator pod's FastAPI, not directly by RecordsManager / DatasetManager processes.
> - Config delivery uses a single `run_config.json` (from `BenchmarkRun`) rather than separate `service_config.json` / `user_config.json`.

---

## REQ — Top-level requirements

| REQ | Requirement | Status | Notes |
|---|---|---|---|
| REQ 1 | Multi-pod distributed deployment via JobSet; Worker + RecordProcessor 1:1 sidecar | 🌟 | JobSet emitted (`kubernetes/jobset.py:235-256`); worker pod runs WPM + N workers + M RPs (scale factor default 1) — more flexible per-pod ratio than fixed 1:1. |
| REQ 2 | Direct Kubernetes API integration for all components | 🌟 | Full kopf operator + `AIPerfJob` CRD reconcile loop (explicit AIP-0002 post-MVP future work); CLI-direct fallback exists at `profile_deploy_direct.py`. |
| REQ 3 | Sustain ≥100K concurrency; scale to 1M+ | 🟡 | 1M+ ramps validated in memory (see `project_k8s_durability_ramp_*.md`), but integration test suite only exercises ≤10 workers (`tests/kubernetes/test_scaling.py`). |
| REQ 4 | Preserve ZMQ (TCP inter-pod, IPC intra-pod) | ✅ | Full dual-bind / locality-aware resolver implemented in `zmq/_zmq_dual_bind.py`, `zmq/zmq_base_client.py:164-241`. |
| REQ 5 | Lifecycle: deploy, execute, cleanup; `aiperf attach`; auto-cleanup on failure | 🌟 | All covered via richer surfaces: deploy/execute/attach ✅, cancel via `spec.cancel=true` CR field, cleanup via `kubectl delete aiperfjob` + ownerReferences GC + time-based results TTL. |
| REQ 6 | Simple DX for 1M+ concurrency; basic K8s security hygiene | 🟡 | Operator path works but requires Helm install + CRD; spec's single-command `aiperf profile --kubernetes` flow is absent. |

---

## Architecture — pod & container layout

| Feature | Status | Notes |
|---|---|---|
| JobSet CRD used to manage all components | ✅ | `jobset.py:235-256` emits `kind: JobSet` v1alpha2. |
| Controller ReplicatedJob (replicas: 1) with **4** containers | 🌟 | `replicas=1` ok; pod has 5 required + 2 optional + results-sidecar + event-bus-proxy containers (9+) — one per service, **more isolation** than the 4-container hybrid. |
| `control-plane` container: SystemController + WorkerManager | 🔵 | Container named `control-plane` (`jobset_builder.py:263-275`) but runs only SystemController; WorkerManager moved to per-worker-pod `worker-group-manager`. |
| `timing-manager` isolated container | ✅ | `jobset_builder.py:284-291`. |
| `dataset-manager` container | ✅ | `jobset_builder.py:276-283`. |
| `records-manager` container (bundling GPUTelemetry + ServerMetrics) | 🌟 | Records-manager exists; GPU-telemetry + server-metrics are separate containers (`jobset_builder.py:314-340`) — more isolation than optional bundle. |
| Controller containers share emptyDir for IPC sockets | ✅ | `jobset_helpers.py:113-126` mounts shared `ipc` emptyDir at `/aiperf/ipc`. |
| Workers ReplicatedJob (scalable) | ✅ | `jobset_builder.py:452-469` uses `spec.worker_replicas`. |
| Worker + RecordProcessor sidecar 1:1 | 🌟 | Pod runs `worker-group-manager` + N `worker-{i}` + M `record-processor-{i}` in one pod (`jobset_builder.py:382-424`); **configurable** ratio via `RECORD_PROCESSOR_SCALE_FACTOR` (default 1) — pod-level co-location intent preserved, tunability added. |
| Worker identity via `JOB_COMPLETION_INDEX` | 🔵 | Uses `AIPERF_POD_INDEX` from downward-API label `jobset.sigs.k8s.io/job-index` (`jobset_helpers.py:213-222`); `JOB_COMPLETION_INDEX` deliberately not used. |
| Headless Service providing JobSet DNS | ✅ | `jobset.py:244-246` sets `network.enableDNSHostnames: true`. |
| `startupPolicy: InOrder` (controller → workers) | 🌟 | Ordering enforced by ZMQ client-side reconnect (`zmq/zmq_base_client.py:149-150,203-204` RECONNECT_IVL=100ms–5s) + `WORKER_CONNECTION_PROBE_TIMEOUT=60s` (`environment.py:254-262`) + `WORKER_BACKOFF_LIMIT=20` worker-pod restart — more robust than strict sequencing (workers recover from transient controller-late situations). |
| `successPolicy: controller` | ✅ | `jobset.py:247-250` `targetReplicatedJobs=["controller"]`. |
| Controller `backoffLimit: 0` | ✅ | `environment.py:242-247` default 0. |
| Worker `backoffLimit: 2` | 🌟 | Default **20** with `le=20` hard ceiling (`environment.py:248-253`); docstring explicitly states "WORKER_BACKOFF_LIMIT absorbs transient first-deploy flakes" — production-tuned for high-scale ZMQ first-connect retries at 1M+ concurrency, where 2 is insufficient. |

---

## Communication — ZMQ

| Channel | Status | Notes |
|---|---|---|
| CREDIT_ROUTER IPC+TCP | ✅ | `_zmq_dual_bind.py:216-220, 251-253`; `credit/sticky_router.py:127-163` binds TCP additional address. |
| RAW_INFERENCE IPC-only (Worker → sidecar direct, bypassing proxy) | ✅ | Raw-inference proxy runs **inside each worker pod** via `WorkerGroupManager` (`workers/worker_pod_manager.py:126` `ProxyManager(enable_raw_inference=True)`); worker push → proxy → RP pull all happen over IPC in the shared emptyDir — no TCP. Proxy enables multi-worker fan-in to shared RPs at configurable RP:worker ratio. Spec's IPC intent met. |
| RECORDS IPC+TCP | ✅ | `records/records_manager.py:97-111` + `_zmq_dual_bind.py:209-213, 261-263`. |
| EVENT_BUS_PROXY IPC+TCP | ✅ | Both IPC + TCP (`ZMQDualBindProxyConfig`); defaults 5663/5664 via `zmq/zmq_proxy_base.py:111,118`. |
| Dual-bind proxies (IPC + TCP on same proxy) | ✅ | `zmq_base_client.py:164-241` binds primary, then `_bind_additional_address`. |
| Locality-based transport selection in services | ✅ | `_zmq_dual_bind.py:_resolve()` selects TCP when `controller_host` is set, IPC otherwise; workers get `controller_host` via `K8sEnvironment.ZMQ.CONTROLLER_HOST`. |
| Configurable ZMQ addresses (M1) | ✅ | Full Pydantic config tree in `config/_models_comm.py` (Ipc / Tcp / DualBind), resolved in `config/_comm.py` + `config/resolvers.py:205-250`. |

---

## CLI — `aiperf profile --kubernetes` options

| Option | Status | Notes |
|---|---|---|
| `--kubernetes` flag | 🔵 | No such flag; K8s lives under `aiperf kube profile` subcommand group (`cli_commands/kube/_app.py:9-70`, `cli.py:67-70`). |
| `--namespace` | ✅ | `KubeManageOptions.namespace` (`config/kube.py:68-72`); default `aiperf-benchmarks`. |
| `--image` (auto = current CLI version tag) | ⚪ | `--image` required (`config/kube.py:108-115`); auto-detection deferred (explicit per-run image is the intended workflow). |
| `--workers-max` | ✅ | `workers` field default 10 (`config/kube.py:127-135`). |
| `--kubeconfig` | ✅ | `config/kube.py:52-58`. |
| `--kubecontext` | 🔵 | Named `--kube-context` (hyphenated) at `config/kube.py:60-66`. |
| `--dataset-pvc` (RWX, skip HTTP download) | 🌟 | Generic `AIPerfJob.spec.podTemplate.volumes` + `volumeMounts` (`crd.yaml:140-150`, flowed via `operator/spec_converter.py:113-124` → `jobset_helpers.py:109,125`) supports any volume type (PVC, ConfigMap, hostPath, …) at any mount path — user writes `inputFile: /data/prompts.jsonl` and mounts the PVC at `/data`. Richer than a dedicated flag. |
| `--results-pvc` | 🔵 | Controlled at helm-install time via `storage.enabled`/`storage.storageClassName`/`storage.size` (`deploy/helm/aiperf-operator/values.yaml:157-176`, `templates/pvc.yaml`, `deployment.yaml:180-184`); operator PVC is cluster-wide, so per-job override isn't needed. |
| `--no-results-pvc` | 🔵 | `storage.enabled=false` yields an emptyDir-backed results volume with optional `storage.emptyDirSizeLimit` (`deploy/helm/aiperf-operator/values.yaml:158-180`, `templates/deployment.yaml:175`). |
| `--ttl-seconds` (int \| `none`) | ⚪ | `ttl_seconds: int\|None` default 300 (`config/kube.py:137-143`); accepts int or null — literal `none` string parser deferred (pass `null`/omit for same effect). |
| `--api-access` (port-forward / proxy / loadbalancer / nodeport) | ⚪ | Port-forward + helm `ingress.enabled` cover the intended use cases (dev + cloud); LB/NodePort deferred, mode selector unnecessary. |
| `--node-ip` | ⚪ | Tied to NodePort; deferred per access-methods decision. |
| `--skip-preflight` | 🌟 | Preflight is **not auto-invoked** by `aiperf kube profile` client-side; instead the operator runs preflight server-side during CR reconcile (`operator/handlers/create.py:150-189,375`), rejecting invalid CRs before JobSet creation. Client-side `--skip-endpoint-check` exists for the one client check (`cli_commands/kube/profile.py:36-40`). |
| Auto-generated namespace `aiperf-{job_id}` when `--namespace` unset | 🔵 | Default is static `aiperf-benchmarks` (`kubernetes/constants.py:151`, `resources.py:356-358`); no per-job namespace. |
| Auto worker-count: `ceil(concurrency / 500)` default | 🟡 | **Planned work**: flip to forward derivation (concurrency → workers via `AIPERF_HTTP_CONNECTION_LIMIT` default 500/worker). Today reverse-derived: CLI takes `--workers-max`, derives `connectionsPerWorker = ceil(concurrency / workers)` (`config/kube.py:247-257`, `kube/profile.py:92-98`). |

---

## CLI — lifecycle commands

| Command | Status | Notes |
|---|---|---|
| `aiperf list` (incl. `--all-namespaces`) | 🌟 | `aiperf kube list` with `-A` (`cli_commands/kube/list_.py:19-103`) lists **AIPerfJob CRs** plus JobSet fallback — richer than JobSet-only listing (CRs carry status, progress, cancel signalling). |
| `aiperf attach --job-id` (reconnect + progress) | 🔵 | `aiperf kube attach [job_id]` with positional + port-forward + WebSocket (`cli_commands/kube/attach.py:17-76`). |
| `aiperf results --job-id` | 🌟 | `aiperf kube results [job_id]` with `--from-pods`, `--summary-only`, `--shutdown` (`cli_commands/kube/results.py:17-60`) — more retrieval modes than the spec's single download. |
| `aiperf cancel --job-id` | 🔵 | No dedicated subcommand, but cancellation flows via `spec.cancel=true` on the AIPerfJob CR (`operator/models.py:288`, `operator/handlers/lifecycle.py:62-120`) and `POST /jobs/{ns}/{name}/cancel` (`operator/routers/jobs.py:407`). Triggerable by `aiperf kube results --shutdown` or `kubectl edit aiperfjob`. |
| `aiperf cleanup --job-id` | 🌟 | `kubectl delete aiperfjob` cascades via `ownerReferences` GC (JobSet + ConfigMap + Role reaped automatically, `handlers/lifecycle.py:35-59`); separate time-based results TTL (`handlers/cleanup.py:31-66`, `models.py:282-286` `resultsTtlDays`). Richer than a per-job flag. |
| `aiperf preflight` standalone | 🔵 | Exists as `aiperf kube preflight` (`cli_commands/kube/preflight.py:16-67`). |
| Ctrl+C cancellation with graceful termination (M6) | ✅ | `cancel_aiperf_job` sets `spec.cancel=true` (`kubernetes/client_jobs.py:231`); operator tears down (`handlers/monitor.py`, `completion.py:85,147`); CLI entry points catch KeyboardInterrupt; `watch_orchestrator.py:142` installs SIGINT/SIGTERM; integration test `tests/integration/test_ctrl_c_cancellation.py`. |

---

## `aiperf service` command (M0)

| Feature | Status | Notes |
|---|---|---|
| `aiperf service` subcommand exists | ✅ | Registered at `cli.py:55-59`, implemented at `cli_commands/service.py:12-102`. |
| `--type` accepts comma-separated list for multi-service containers | 🌟 | Per-container-per-service is the new design (`jobset_builder.py:259-360`) — each control-plane/worker service runs in its own container for independent OOM/liveness/resource isolation, making multi-service-per-container obsolete. |
| `--api-port` | ✅ | `cli_commands/service.py:60-68`. |
| `--health-port` | ✅ | `cli_commands/service.py:50-58` (plus `--health-host`). |
| `--id` | ⚪ | Exposed as `--service-id` (`cli_commands/service.py:32-40`); short `--id` alias deferred (cosmetic). |
| `system-controller,worker-manager` combo | 🌟 | Intentional redesign — WorkerManager was relocated to `worker-group-manager` inside each worker pod (`kubernetes/constants.py:134`, `jobset_builder.py:368-393`, `common/subprocess_manager.py:7-112`); controller-side combo obsolete. |
| `timing-manager`, `dataset-manager`, `records-manager` | ✅ | Registered service plugins via dynamic `ServiceType` enum (`plugin/enums.py:116-118`). |
| `worker`, `record-processor` | ✅ | Same registration; single-type launch only. |

---

## Configuration (ConfigMap + file loading)

| Feature | Status | Notes |
|---|---|---|
| CLI creates ConfigMap via K8s API from Pydantic models | 🔵 | `ConfigMapSpec.from_benchmark_run` (`resources.py:132-170`) serializes `BenchmarkRun` as single `run_config.json`, not separate service/user files. Direct-deploy path at `profile_deploy_direct.py:46-47`. |
| Pods mount ConfigMap at `/etc/aiperf/` | ✅ | `jobset_helpers.py:98,102` mounts `config` volume at `CONFIG_MOUNT_PATH` (`/etc/aiperf`). |
| `AIPERF_CONFIG_SERVICE_FILE` env / `--service-config` | 🔵 | Consolidated: containers bootstrap via `--benchmark-run /etc/aiperf/run_config.json` (`jobset_helpers.py:136-143`), with the full `BenchmarkRun` envelope written by `ConfigMapSpec.from_benchmark_run` (`resources.py:132-170`). Pydantic field `common/_env_services.py:34-38` retained but unused. |
| `AIPERF_CONFIG_USER_FILE` env / `--user-config` | 🔵 | Same consolidation as above — single `run_config.json` covers service- and user-scope config (`common/_env_services.py:39-43` unused). |
| Pydantic `to_json()` / `from_json()` round-trip (M0) | 🔵 | Uses `model_dump(mode="json")` + `orjson.dumps` for write (`resources.py:154-159`), `--benchmark-run` Path for read. |
| `ProfileConfigureCommand` for per-sweep dynamic config | ✅ | Command enum at `common/enums/communication_enums.py:49`; handlers in dataset/gpu/records/timing/server/worker (7 handler sites). |

---

## Dataset distribution

| Feature | Status | Notes |
|---|---|---|
| Local file → HTTP upload (`--input-file /local/...`) | ⚪ | Intended workflow: user creates a PVC out-of-band (e.g., `kubectl cp` to a staging pod, or dataset already in cluster storage), references it via `podTemplate.volumes` in the AIPerfJob CR, and `--input-file` becomes a local mounted filepath. HTTP upload of laptop-local files is out of scope. |
| `pvc://{name}/path` URI scheme | 🌟 | Not needed: `AIPerfJob.spec.podTemplate.volumes/volumeMounts` mount any PVC at any path, and `inputFile` takes a normal filepath — strictly more flexible than a URI scheme. |
| Remote URL (`--input-file https://...`) | ⚪ | Curated URLs served via `--public-dataset`; arbitrary URLs deferred (same philosophy as local files — bring data into the cluster via PVC). |
| `--public-dataset sharegpt` (in-cluster generation) | ✅ | `dataset/dataset_manager.py:266` via `load_public_dataset`. |
| Synthetic (`--isl`, `--num-dataset-entries`, etc.) | ✅ | `dataset/synthesis/` composer path runs inside `DatasetManager`. |
| Controller `POST /api/dataset` chunked + LZ4/ZSTD | ⚪ | Obsolete — replaced by `podTemplate` volume mounts. No upload endpoint needed. |
| Broadcast `DATASET_CONFIGURED_NOTIFICATION` | ✅ | Enum at `common/enums/communication_enums.py:99`; consumers in `timing/manager.py:79`, `workers/worker.py:396`, `workers/worker_pod_manager.py:230`, `api/routers/dataset.py:73`. |
| Worker `GET /api/dataset` → emptyDir → mmap | ✅ | `workers/worker_pod_dataset_download.py:51-83` + `dataset/memory_map_client.py:135`. |
| `--dataset-pvc` mount path | 🌟 | See `--dataset-pvc` row in CLI options — `podTemplate.volumes`/`volumeMounts` handle this generically. |

---

## HTTP API endpoints

| Endpoint | Host per spec | Status | Notes |
|---|---|---|---|
| `GET /api/progress` | RecordsManager | 🌟 | Served by dedicated `FastAPIService` (ServiceType.API, `controller/kubernetes_service_manager.py:50`) — cleaner separation of concerns (dedicated HTTP pod, not HTTP server embedded in RecordsManager). Operator pod additionally exposes `/api/v1/jobs` + polls via `operator/progress_client.py`. |
| `/ws` WebSocket | RecordsManager | 🌟 | `api/routers/websocket.py:122` on FastAPIService — same dedicated-HTTP-pod advantage. |
| `GET /api/metrics` | RecordsManager | 🌟 | `api/routers/metrics.py:80` on FastAPIService; also exposes `/metrics` Prometheus endpoint (not in spec). |
| `GET /api/artifacts/archive` (ZSTD) | RecordsManager | 🌟 | ZSTD is used throughout the storage layer: `operator/environment.py:76` stores result files as `.zst` on disk, `results_db.py:286` writes `profile_export_aiperf.json.zst`, and per-file endpoint `/api/results/files/{filename}` (`api/routers/results.py:107`) serves with HTTP content negotiation (raw zstd / decompressed / re-gzipped per client `Accept-Encoding`). Batch `/api/v1/results/{ns}/{job}.zip` (`operator/routers/results_files.py:248`) is intentionally zip format for tool compatibility — zstd entries are decompressed on the fly. |
| `GET /api/dataset` | DatasetManager | 🌟 | `api/routers/dataset.py:171-213` on FastAPIService exposes `/api/dataset/{data,index,state}` — split by artifact type, cleaner than single endpoint. |
| `POST /api/dataset` | DatasetManager | ⚪ | Obsolete — dataset upload replaced by `podTemplate.volumes` PVC mount workflow (see dataset-distribution section). |
| `/healthz` liveness | per container | ✅ | `common/health_server.py:68-80` aiohttp; auto-started via `HealthServerMixin` on `BaseService`. |
| `/readyz` readiness | per container | ✅ | `common/health_server.py:101-106` w/ optional `readiness_check`. |

---

## External API access methods

| Method | Status | Notes |
|---|---|---|
| port-forward (default subprocess) | ✅ | `kubernetes/port_forward.py` — `port_forward_to_controller`, `start_port_forward`, auto-restart, pod-disappearance watch. |
| kubectl proxy subprocess | 🔵 | Same localhost-TCP-tunnel outcome via `kubernetes/port_forward.py` (Python k8s client) + `ingress.yaml`. |
| LoadBalancer | ⚪ | Port-forward + ingress cover dev + cloud; LB Service type deferred. |
| NodePort (with `--node-ip` auto-discovery) | ⚪ | Bare-metal-without-ingress-controller deferred; users on such clusters deploy an ingress controller first. |

---

## Time synchronization

| Feature | Status | Notes |
|---|---|---|
| Credit `issued_at_ns` stamp at T1, T2-T1 on receive | ✅ | `credit/issuer.py:210-227` stamps; `workers/clock_offset_tracker.py:110-123` computes sample. |
| Worker EMA-smoothed offset tracking | 🌟 | Uses NTP-style min-filter over sliding window (`clock_offset_tracker.py:98-123`), plus optional baseline-RTT / one-way estimate — more robust than simple EMA, standard for clock-sync. |

---

## Results persistence

| Mode | Status | Notes |
|---|---|---|
| Auto-create `aiperf-results-{job_id}` PVC (default) | 🌟 | JobSet uses emptyDir; persistence is a **single cluster-wide operator PVC** `aiperf-operator-results` (`deploy/helm/aiperf-operator/templates/pvc.yaml`, `operator/environment.py:54`, `AIPERF_RESULTS_DIR=/data`). Flow: worker→controller→operator→CLI. Enables cross-job artifact retention + recovery; outlives any single JobSet. |
| `--results-pvc NAME` user-provided | 🔵 | Configured at helm-install time via `storage.storageClassName`/`storage.size` (`deploy/helm/aiperf-operator/values.yaml:157-176`); operator PVC survives cleanup by design. |
| `--no-results-pvc` (emptyDir only) | 🔵 | EmptyDir is the only mode in JobSet today — implicit always-on, not a flag. |
| ConfigMap `profile_export_aiperf.json` (always written) | 🌟 | Operator PVC persists summaries at `jobs/<ns>/<id>/profile_export_aiperf.json` (`operator/routers/jobs.py:209`); cross-job retention, no 1 MiB ConfigMap cap, recoverable after CR deletion — strictly richer than ConfigMap summary for standard install/upgrade/cleanup lifecycles. |
| Incremental progress persistence (to ConfigMap) | 🌟 | `RecordsManager._write_partial_checkpoint_task` (`records/records_manager.py:413`) writes periodic checkpoints to PVC; operator **recovers** from partial checkpoints (`operator/handlers/monitor.py:1042`). More robust than ConfigMap (size-limit free, restart-resistant). |
| TTL auto-cleanup (JobSet + owned PVC) | 🌟 | JobSet `ttlSecondsAfterFinished` default 300 (`environment.py:231`, `jobset.py:262`) cascades to pods. Operator PVC is cluster-lifecycle (not owned by JobSet) — richer than per-job PVC GC because results survive across jobs. Separate days-based results-TTL (`operator/main.py:148`, `operator/models.py:286`) reaps stale result dirs. Direct-mode default 28800s. |

---

## Failure handling

| Feature | Status | Notes |
|---|---|---|
| Controller fails → benchmark fails, ConfigMap checkpoint preserved | 🟡 | **Planned work**: mirror the partial-checkpoint summary into a ConfigMap on controller failure so users can `kubectl get cm` without operator access. Today: `CONTROLLER_BACKOFF_LIMIT=0`; partial checkpoints on PVC only. |
| Worker retries (backoffLimit); exhaustion tolerated | 🌟 | `WORKER_BACKOFF_LIMIT=20` production-tuned (spec's 2 is insufficient for high-scale ZMQ first-connect); `successPolicy` targets controller only, so exhaustion doesn't fail JobSet. |
| CLI disconnect → benchmark continues; reattach via `aiperf attach` | ✅ | `aiperf kube attach` (`cli_commands/kube/attach.py`, `kubernetes/attach.py`); benchmark runs independent of CLI. |
| CLI non-zero exit on fatal errors | ✅ | `exit_on_error` wraps kube subcommands (`cli_utils.py:61`; used across profile/attach/preflight/watch). |
| Graceful shutdown: CreditsComplete → ProcessRecordsResult → ShutdownCommand | ✅ | `timing/phase/publisher.py:91` → `records/records_manager.py:301,480` → `controller/system_controller.py:997,1026,1139`. |
| CLI retry w/ exponential backoff on API drops | ✅ | `operator/progress_client.py:60,149-181` `retry_with_backoff` with configurable `max_retries`/`initial_backoff`. |

---

## Pre-flight checks

| Check | Status | Notes |
|---|---|---|
| cluster-connectivity | ✅ | `check_cluster_connectivity` (`kubernetes/preflight_checks.py:85`), wired in `cli_commands/kube/preflight.py:279,301`. |
| rbac-permissions | ✅ | `check_rbac_permissions` (`preflight_checks.py:195`). |
| namespace-availability | ✅ | `check_namespace` (`preflight_checks.py:142`). |
| resource-quota (warning) | ✅ | `check_resource_quotas` in `preflight_capacity_checks.py`. |
| endpoint-reachability | ✅ | `check_endpoint_connectivity` (`preflight_checks.py:426`). |
| image-pull | ✅ | `check_image` (`preflight_capacity_checks.py:370`). |
| Standalone `aiperf preflight` command | 🌟 | Client-side `aiperf kube preflight` available; **server-side preflight runs in the operator reconcile loop** via `OperatorPreflightChecker` (`operator/handlers/create.py:150-189,375`), rejecting invalid CRs before JobSet creation — stronger than a client-side check (cannot be bypassed). |

---

## Error reporting channels

| Channel | Status | Notes |
|---|---|---|
| CLI progress stream (real-time error counts) | ✅ | `watch_pollers.py:130,165` emits `error_count`; rendered in `watch_render_rich.py:181`, `watch_render_text.py:83`; `progress_stream.py` streams via websocket. |
| `/api/progress` error counts + samples | 🟡 | **Planned work**: `/api/progress` exists and exposes `request_errors` (`progress_tracker_mixin.py:53,72,123`); "recent error samples" ring-buffer not yet implemented. |
| ConfigMap aggregated error summary | 🟡 | **Planned: add sample error-message payloads to AIPerfJob `.status`** (counts already surface via `records/error_tracker.py:25` `get_error_summary()` → operator PVC + CRD `.status` `requests_errors`/`error_rate` at `operator/models.py:103,126,189,229` + Prometheus `/metrics`; sample messages missing). |
| Pod logs (via `kubectl logs`) | ✅ | `cli_commands/kube/logs.py` + `kubernetes/logs.py`. |

---

## Resource allocation (defaults)

| Container | Spec (CPU req/limit, mem req/limit) | Status | Notes |
|---|---|---|---|
| control-plane | 200m/1000m, 256Mi/512Mi | 🌟 | `SYSTEM_CONTROLLER` default 500m/1Gi (`environment.py:305`) — tuned from 1M+ concurrency ramps; spec values were explicitly "preliminary estimates." |
| timing-manager | 100m/500m, 128Mi/256Mi | 🌟 | `TIMING_MANAGER` default 1000m/2Gi (`environment.py:307`) — production-validated, timing is latency-critical. |
| dataset-manager | 100m/500m, 256Mi/512Mi | 🌟 | `DATASET_MANAGER` default 1000m/2Gi (`environment.py:308`) — tuned for multi-GB synthetic datasets. |
| records-manager | 200m/1000m, 256Mi/1Gi | 🌟 | `RECORDS_MANAGER` default 1000m/2Gi (`environment.py:309`) — **spec defaults cause event-loop starvation at >500k requests** (see `gotcha_records_manager_cpu_starves_at_high_concurrency.md`); real floor is 4000m+. |
| worker | 250m/250m, 256Mi/512Mi | 🌟 | Pod-level `WORKER_POD` 4000m/12Gi split across WPM + workers + RPs via weighted shares (`jobset_resources.py:77-125`) — more efficient pod packing than per-container allocations; `resource_mode=guaranteed` achieves req==limit (default is `burstable` — no limits — so the controller can grow during aggregation). |
| record-processor | 100m/250m, 256Mi/512Mi | 🌟 | Shares worker-pod budget with weighted shares; per-RP CPU pinnable via `RECORD_PROCESSOR_CPU_REQUEST` (`environment.py:317-320`). |

---

## RBAC

| Permission | Status | Notes |
|---|---|---|
| ConfigMaps (CRUD) | ✅ | `resources.py:208-212`. |
| Services (CRUD) | 🌟 | Intentional least-privilege — get/list/watch/create/delete + endpoints, no update/patch (`resources.py:220-224`); nothing in the code path updates or patches Services post-creation, so narrower RBAC is safer. |
| PVCs (CRUD) | 🌟 | No per-job PVCs created at runtime — the single `aiperf-operator-results` PVC is provisioned at Helm install time with installer credentials (`deploy/helm/aiperf-operator/templates/pvc.yaml`, mounted via `deployment.yaml:180-184`); operator ServiceAccount intentionally least-privileged. |
| Pods (read) | ✅ | `resources.py:214-218`. |
| Pods/logs (read) | ✅ | Same rule covers `pods/log`. |
| Pods/exec (create) | 🔵 | Runtime operator/controller never calls `pods/exec`; `kube debug`/`logs`/`watch_diagnosis` use `pods/log` + K8s watch API. `pods/exec` is only granted in optional chaos-test mode via `SYS_PTRACE`; least-privilege is intentional. |
| Jobs (read) | ✅ | `resources.py:232-236`. |
| JobSets (CRUD) | ✅ | `resources.py:238-247`. |

---

## MVP Deliverables (M0–M6)

| Milestone | Item | Status | Notes |
|---|---|---|---|
| M0 | `aiperf service` CLI command | ✅ | See [`aiperf service`](#aiperf-service-command-m0) section. |
| M0 | Config Pydantic `to_json()` / `from_json()` | 🔵 | `model_dump(mode="json")` + `orjson.dumps`; no `to_json/from_json` helpers. |
| M0 | Config loading from `/etc/aiperf/` | 🔵 | Via `--benchmark-run /etc/aiperf/run_config.json`, not the spec's separate service/user files. |
| M0 | HTTP health server `/healthz` `/readyz` | ✅ | `common/health_server.py`, `HealthServerMixin`; routers `api/routers/core.py:41,49` + `api/routers/kubernetes.py:19,39`. |
| M1 | Configurable ZMQ addresses | ✅ | See [Communication](#communication--zmq). |
| M1 | Dual-bind ZMQ proxies (IPC + TCP) | ✅ | See [Communication](#communication--zmq). |
| M2-M3 | JobSet YAML generation | ✅ | `kubernetes/jobset.py`, `jobset_builder.py`. |
| M2-M3 | startupPolicy controller-first | 🌟 | Equivalent outcome via ZMQ reconnect + worker-pod backoff (see architecture row). |
| M2-M3 | Headless Service DNS | ✅ | `network.enableDNSHostnames: true`. |
| M2-M3 | ConfigMap for user/service config | 🔵 | Single `run_config.json` instead of separate files. |
| M2-M3 | Namespace + RBAC + Service creation | 🌟 | All covered: namespace ✅, RBAC at least-privilege (PVCs handled at helm-install, pods/exec intentionally narrowed), Service creation ✅. See RBAC table for details. |
| M2-M3 | Basic CLI integration (`--kubernetes`/`--kubeconfig`/`--image`/`--workers-max`) | 🔵 | Under `aiperf kube profile`, not top-level `aiperf profile --kubernetes`. |
| M4 | `GET /api/dataset` | 🌟 | Hosted by dedicated FastAPIService; split into `/api/dataset/{data,index,state}`. |
| M4 | Worker download + local mmap | ✅ | `workers/worker_pod_dataset_download.py` + `dataset/memory_map_client.py`. |
| M4 | CLI dataset upload `POST /api/dataset` | ⚪ | Replaced by `podTemplate.volumes` workflow. |
| M5 | `GET /api/progress` polling | 🌟 | Exists on dedicated FastAPIService (cleaner separation). |
| M5 | WebSocket progress streaming | 🌟 | `/ws` on FastAPIService. |
| M5 | port-forward setup for CLI↔controller | ✅ | `kubernetes/port_forward.py`. |
| M5 | `GET /api/artifacts/archive` ZSTD | 🌟 | ZSTD everywhere except batch zip envelope (intentional for tool compat) — see HTTP endpoints table. |
| M5 | ConfigMap summary storage | 🌟 | Summary persisted to operator PVC (`operator/routers/jobs.py:209`) + cross-job recoverable, not ConfigMap. |
| M6 | Automatic cleanup (delete JobSet after benchmark) | ✅ | `_maybe_delete_jobset_after_success` (`operator/handlers/completion.py:138`); TTL fallback on JobSet. |
| M6 | Ctrl+C cancellation with graceful termination | ✅ | See [lifecycle commands](#cli--lifecycle-commands) row. |

---

## Testing strategy

| Layer | Status | Notes |
|---|---|---|
| Unit tests (mocked K8s client) | ✅ | `tests/unit/kubernetes/` (46 files), `tests/unit/operator/` (27 files); includes JobSet/ConfigMap/preflight/progress_client. |
| Integration tests (kind/minikube) | ✅ | `tests/kubernetes/test_{benchmark,deployment,scaling,operator,kueue_*,helm}.py` against `local_cluster` kind/minikube fixture. |
| E2E tests (real cluster, 10→200+ workers) | 🟡 | **Planned: scheduled 100-200 worker CI** (nightly/weekly). Today: `tests/kubernetes/test_scaling.py:27-29` caps at 10 workers; 1M+ validated via manual ramps (see `project_k8s_durability_ramp_*.md`). No full spec-conformance (1M+/±5%) E2E is planned. |

---

## Explicit non-goals / deferred

| Item | Status | Notes |
|---|---|---|
| Automatic pod-failure recovery beyond JobSet built-ins | ⚪ | Non-goal (MVP) |
| Non-Kubernetes orchestrators | ⚪ | Non-goal |
| Web UI for job management | 🌟 | Non-goal (MVP) per spec — **already built**: operator dashboard + watch UI (`src/aiperf/operator/ui/`, `src/aiperf/cli_commands/kube/dashboard.py`, `kubernetes/watch_render_rich.py`). |
| Cross-cluster / multi-cloud | ⚪ | Non-goal |
| Replacing single-node mode | ⚪ | Both modes coexist |
| Advanced security (mTLS, NetworkPolicy, secrets) | ⚪ | Non-goal (MVP) |
| Graceful degradation / checkpoint-resume / worker replacement | 🌟 | Non-goal (MVP) per spec — **partial-checkpoint recovery already implemented** (`operator/handlers/monitor.py:1042` `_recover_from_partial_checkpoints`). |
| Unified `AIPerfJobSpec` YAML | 🌟 | Spec's post-MVP future — **already done** as the `AIPerfJob` CRD schema (`operator/models.py`). |
| Custom operator (full) | 🌟 | Spec's post-MVP future — **already done**, kopf operator + reconcile loop + CRD on this branch. |

---

## Summary scoreboard

Across **158** discrete spec items in the tables above:

| Status | Count | Read as |
|---|---|---|
| ✅ Done | 64 | Feature present and matches spec intent |
| 🌟 Exceeds spec | 48 | Branch is past MVP into explicit future-work territory, or ships a richer / production-validated variant |
| 🔵 Divergent | 22 | Equivalent user outcome through a different surface |
| ⚪ Deferred | 17 | Intentionally out of scope (spec non-goals + user-confirmed deferrals: laptop-file upload, arbitrary URLs, LB/NodePort, cosmetic polish, etc.) |
| 🟡 Partial | 7 | **Tracked planned work** — see list below |
| 🔴 Not started | 0 | — |

**Where the branch exceeds the spec (🌟 highlights):**
- Full kopf operator + `AIPerfJob` CRD (spec's explicit post-MVP future work — done now)
- Operator dashboard + rich watch UI (explicit MVP non-goal — done anyway)
- Cross-job operator-wide results PVC + partial-checkpoint recovery (more robust than spec's per-job JobSet PVC + ConfigMap summary)
- Per-service container isolation in controller pod (9+ containers vs spec's 4-container hybrid)
- Configurable RP:worker ratio via `RECORD_PROCESSOR_SCALE_FACTOR` (spec assumed fixed 1:1)
- Dedicated `FastAPIService` hosting all HTTP endpoints (cleaner separation than embedding HTTP in each service)
- Production-tuned resource defaults validated at 1M+ concurrency (spec values were preliminary estimates that break at >500k)
- NTP-style min-filter clock offset (more robust than simple EMA)
- **Server-side preflight** in the operator reconcile loop (cannot be bypassed)
- **ZMQ reconnect + backoff** replacing `startupPolicy: InOrder` (recovers from transient controller-late situations rather than requiring strict ordering)
- **`kubectl delete aiperfjob` ownerReferences GC** replacing `aiperf cleanup` (one-shot cleanup via native K8s cascade)
- **Helm-provisioned PVC with installer credentials** replacing runtime PVC CRUD (least-privilege operator ServiceAccount)
- **`podTemplate.volumes`/`volumeMounts`** replacing dedicated `--dataset-pvc` / `pvc://` URI (supports any volume type at any mount path)
- **ZSTD storage throughout** — datasets stored `.dat.zst`, results stored `.json.zst`, per-file API endpoint supports zstd HTTP content negotiation; only batch zip is uncompressed (intentional, for tool compat)
- Richer lifecycle commands (`aiperf kube results --from-pods --summary-only --shutdown`, `aiperf kube list` listing CRs+JobSets)

**Where the branch takes an equivalent-but-different surface (🔵 highlights):**
- `spec.cancel=true` CR field + HTTP route instead of `aiperf cancel` subcommand
- Helm `storage.*` values replacing `--results-pvc` / `--no-results-pvc` / `--ttl-seconds` overrides
- Single `run_config.json` replacing separate `service_config.json` / `user_config.json`
- `port-forward` (Python client) replacing `kubectl proxy` subprocess
- `aiperf kube <cmd>` subapp replacing top-level `aiperf <cmd>` (groups 13+ k8s commands that wouldn't fit as flags)
- `pods/log` + K8s watch API replacing `pods/exec` for debug

**Intentionally deferred (⚪):**
- Laptop-local `--input-file /path` upload (users stage via PVC first)
- Arbitrary URL `--input-file https://...` (use `--public-dataset` for curated sources)
- `POST /api/dataset` upload endpoint
- `type: LoadBalancer` Service + NodePort + `--node-ip` + `--api-access` mode selector (port-forward + ingress cover all supported cluster shapes)
- `startupPolicy: InOrder` (ZMQ reconnect is stronger)
- Advanced security (mTLS, NetworkPolicy, secrets), cross-cluster/multi-cloud, replacing single-node mode — per spec's own non-goal list

**Tracked planned work (🟡, 7 items):**
1. **Scheduled 100-200 worker CI** for scale-regression coverage (no full 1M+ automation planned; manual ramps suffice for those).
2. **Auto worker-count from concurrency** — flip derivation from reverse (`workers → conns/worker`) to forward (`concurrency / AIPERF_HTTP_CONNECTION_LIMIT`).
3. **Sample error messages in `/api/progress`** — add a ring-buffer of recent error payloads alongside existing `request_errors` count.
4. **Sample error messages in AIPerfJob `.status`** — CRD status carries counts + rates, needs sample payloads.
5. **ConfigMap failure-checkpoint mirror** — on controller failure, mirror partial-checkpoint summary into a ConfigMap so users can `kubectl get cm` without operator access.
6. **REQ 3 concurrency in CI** — 1M+ validated manually, item (1) above covers the CI path.
7. **REQ 6 DX** — operator-mode requires Helm + CRD install; evaluate whether a lower-friction first-run flow is needed.

**Decision needed:**
- **Amend AIP-0002** to reflect the operator-first posture and the 🌟 design decisions that diverge from the original MVP framing. Until then, this status doc is the current source of truth.
