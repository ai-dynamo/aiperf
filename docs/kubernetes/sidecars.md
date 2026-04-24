---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Controller Pod Sidecars
---

# Controller Pod Sidecars

Every AIPerf benchmark controller pod runs a short stack of sidecar containers alongside the control-plane process. Two of them — the **event-bus proxy** and the **results sidecar** — exist to offload load or provide a fallback surface that the SystemController itself cannot reliably host. They are invisible in normal usage, but anyone debugging fan-in hangs, startup races, or partial result retrieval needs to know what they do and how to tune them.

The source of truth is `src/aiperf/kubernetes/jobset_builder.py` (composition), `src/aiperf/kubernetes/results_sidecar.py` (FastAPI app), and `src/aiperf/kubernetes/environment.py` (resource defaults and ports).

---

## Overview

**Event-bus proxy** runs the XPUB/XSUB ZMQ proxy for the benchmark's pub/sub event bus in a dedicated container. It exposes `tcp://*:5663` (XSUB frontend — publishers connect here) and `tcp://*:5664` (XPUB backend — subscribers connect here) on the controller pod so that workers and record processors can connect and publish/subscribe without talking to the SystemController directly. It is an independent sidecar because, at high concurrency, hundreds of simultaneous RP and worker pub/sub connections arriving at pod startup previously starved the SystemController's event loop while it tried to forward socket I/O itself. Health endpoint is `:8088/healthz` and `:8088/readyz`.

**Results sidecar** runs a minimal FastAPI app on `:9091` that serves the controller pod's `/results` volume read-only. It exists as a fallback so the operator (and `aiperf kube results --from-pods`) can still retrieve exported artifacts after the main control-plane container exits — for example, when the controller completes exports and terminates but the JobSet TTL hasn't cleaned the pod up yet. Files are hidden until the controller writes the `.aiperf_results_ready.json` marker on clean exit, so clients never download half-written artifacts.

Both sidecars are always injected into controller pods by default. Disabling them is almost never correct.

---

## Event-bus proxy

### Architecture

```mermaid
flowchart LR
    subgraph workerpods[Worker pods N x]
        w1[workers]
        rp1[record processors]
    end
    subgraph controller[Controller pod]
        ebp[event-bus-proxy<br/>XSUB:5663 / XPUB:5664]
        sc[control-plane<br/>SystemController]
        mgrs[dataset-manager<br/>timing-manager<br/>records-manager<br/>...]
    end
    w1 -- PUB tcp:5663 --> ebp
    rp1 -- PUB tcp:5663 --> ebp
    ebp -- XPUB tcp:5664 --> sc
    ebp -- XPUB tcp:5664 --> mgrs
    sc -- PUB tcp:5663 --> ebp
```

All pub/sub traffic for the benchmark flows through the proxy, not the SystemController. The SystemController is just another subscriber.

### Ports

| Port | Protocol | Name          | Purpose                                    |
|------|----------|---------------|--------------------------------------------|
| 5663 | TCP      | pub-frontend  | XSUB socket that publishers connect to     |
| 5664 | TCP      | sub-backend   | XPUB socket that subscribers connect to    |
| 8088 | HTTP     | health        | `/healthz` liveness, `/readyz` readiness   |

### Resource budget

Defaults are set in `_K8sEnvironment.EVENT_BUS_PROXY`:

| Setting        | Default | Env var                                |
|----------------|---------|----------------------------------------|
| CPU request    | 2000m   | `AIPERF_K8S_EVENT_BUS_PROXY_CPU`       |
| Memory request | 1Gi     | `AIPERF_K8S_EVENT_BUS_PROXY_MEMORY`    |

The 2-core default is deliberate: the proxy is pure socket I/O forwarding, but at large fan-ins (hundreds of simultaneous PUB/SUB connections from workers and record processors during startup) it can briefly saturate a single core during the initial subscription storm. Giving it its own CPU envelope was the fix for several "SystemController heartbeat timeout during initialization" incidents seen at 500k+ concurrency.

### Startup ordering

The proxy container is prepended to the controller pod's container list, so the kubelet begins pulling and starting it before the control-plane container. The proxy's bind sockets come up in tens of milliseconds — well inside the 90-second client connection-probe timeout used by the rest of the services.

### Disabling (legacy fallback)

Set on the operator deployment:

```
AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED=false
```

This reverts to pre-sidecar behavior: the SystemController itself hosts the XPUB/XSUB proxy inside its own event loop. This is a legacy code path kept only for bisecting regressions — do not run production benchmarks with it disabled. At anything above ~50k concurrency it will reintroduce the startup fan-in stall that motivated the sidecar in the first place.

---

## Results sidecar

The results sidecar is a small FastAPI app launched via `python -m aiperf.kubernetes.results_sidecar`. It mounts the controller pod's `/results` PVC volume read-only and exposes three endpoints on port **9091**.

### Endpoint catalog

| Method | Path                              | Purpose                                        |
|--------|-----------------------------------|------------------------------------------------|
| GET    | `/healthz`                        | Liveness probe. Always `200 OK`.               |
| GET    | `/api/results/list`               | List available result files (JSON)             |
| GET    | `/api/results/files/{filename}`   | Download a result file (with content negotiation) |

The response models (`ResultFileInfo`, `ResultsListResponse`) are shared with the in-process controller API router at `src/aiperf/api/models/results.py`, so the two HTTP surfaces are contractually identical.

### `GET /healthz`

```bash
curl http://localhost:9091/healthz
# {"status":"ok"}
```

### `GET /api/results/list`

Returns the JSON body defined by `ResultsListResponse`:

```json
{
  "files": [
    {"name": "metrics.json",                "size": 41820},
    {"name": "profile_export_aiperf.json",  "size": 18331144},
    {"name": "profile_export_console.txt",  "size": 3402},
    {"name": "checkpoints/phase_0.parquet", "size": 1048576}
  ]
}
```

Readiness semantics:

- If `.aiperf_results_ready.json` exists, all top-level files under `/results` are listed (except the marker itself).
- If the marker is absent, only files under `/results/checkpoints/` are listed. Top-level exports are hidden until the controller finishes cleanly.

### `GET /api/results/files/{filename}`

Streams a single file. Honors `Accept-Encoding` — if the client advertises `zstd` or `gzip`, the sidecar streams the compressed body and sets `Content-Encoding` accordingly (via `aiperf.common.compression.stream_file_compressed`). Otherwise the file is returned as-is.

- Sets `Content-Disposition: attachment; filename="..."` and `X-Filename` headers.
- `Content-Type` is inferred from the extension: `.json`, `.jsonl`, `.csv`, `.parquet`, `.txt`; everything else is `application/octet-stream`.

Error semantics:

| Status | When                                                                          |
|--------|-------------------------------------------------------------------------------|
| 400    | Path traversal (`..`, symlink escaping `/results`) or reserved marker name    |
| 404    | Controller not ready (marker absent) and file is not under `checkpoints/`     |
| 404    | File path resolves correctly but does not exist                               |

### Path safety

`_safe_resolve` resolves the requested filename under the base directory and rejects anything that, after resolution, does not sit inside `base_dir.resolve()`. This catches `..`, absolute paths, and symlinks that would escape `/results`. The readiness marker `.aiperf_results_ready.json` is explicitly unfetchable.

### Readiness marker

The controller writes `/results/.aiperf_results_ready.json` via `write_ready_marker()` at the end of `SystemController` shutdown, once all exporters have flushed. The payload records whether the run was cancelled:

```json
{"ready": true, "was_cancelled": false}
```

Until the marker exists, the sidecar returns 404 on any top-level file. Checkpoint artifacts under `/results/checkpoints/` are served unconditionally so that in-flight progress can be inspected mid-run.

---

## When the results sidecar is used

The results sidecar is reached by the operator, not directly by `aiperf kube results --from-pods`:

1. **`aiperf kube results --from-pods`** tries two tiers: the **Controller API** (`http://controller-pod:9090/api/results/files/...`) via port-forward, then **`kubectl cp`** against the `control-plane` container's `/results` directory. Both require the control-plane container to still be running.
2. **Operator completion fetch** reaches the results sidecar directly via `_download_final_and_sidecar` in `src/aiperf/operator/handlers/_completion_fetch.py`. When the primary client returns without the key result files, the operator re-issues the download against the sidecar port (`:9091`). Because the sidecar outlives the control-plane container, it covers the window between export completion and pod deletion.

If neither path returns the files, results can only be recovered from operator-side storage (see `aiperf kube results` without `--from-pods`).

---

## Tuning

### Event-bus proxy

| Env var                                | Default | Notes                                                                     |
|----------------------------------------|---------|---------------------------------------------------------------------------|
| `AIPERF_K8S_EVENT_BUS_PROXY_CPU`       | `2000m` | Raise if the proxy pegs one core at >1M concurrency                       |
| `AIPERF_K8S_EVENT_BUS_PROXY_MEMORY`    | `1Gi`   | Rarely the bottleneck                                                     |
| `AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED` | `true`  | `false` only for bisecting regressions against the pre-sidecar code path  |

### Results sidecar

| Env var                              | Default | Notes                                                               |
|--------------------------------------|---------|---------------------------------------------------------------------|
| `AIPERF_K8S_RESULTS_SIDECAR_CPU`     | `250m`  | Adequate; streaming is compression-bound not CPU-bound              |
| `AIPERF_K8S_RESULTS_SIDECAR_MEMORY`  | `512Mi` | Adequate                                                            |
| `AIPERF_RESULTS_DIR`                 | `/results` | Volume mount path; do not change unless remounting the PVC        |
| `AIPERF_RESULTS_SIDECAR_PORT`        | `9091`  | Container port; set by the JobSet builder to match `PORTS.RESULTS_SIDECAR` |
| `AIPERF_RESULTS_SIDECAR_LOG_LEVEL`   | `info`  | uvicorn log level                                                   |

---

## When to disable

- **Event-bus proxy**: basically never. The pre-sidecar code path exists only as a bisection escape hatch and will reintroduce the SystemController startup starvation it was built to fix.
- **Results sidecar**: always-on. There is no supported toggle to turn it off, because the operator's completion-fetch path depends on it when the control-plane container has already exited. If you want to skip artifact retrieval entirely, set `ttlSecondsAfterFinished: 0` on the AIPerfJob and let the JobSet delete the pod immediately.

---

## Troubleshooting

### The `results-sidecar` container shows `Running` but `/api/results/list` returns only checkpoints (or 404 on key files)

The controller has not finished exporting. The sidecar filters out top-level files until `.aiperf_results_ready.json` is written. This is expected while the run is still in progress — checkpoint artifacts will appear as they are written, but `metrics.json` and the profile exports only show up after a clean controller exit.

If the controller has exited but the marker is still missing, the run was killed (OOM, SIGKILL, node eviction) before exporters flushed. Recover partial results by:

1. Downloading `checkpoints/` via the sidecar — these are written incrementally.
2. Falling back to `aiperf kube results` without `--from-pods` (operator-side storage, if the operator successfully fetched anything before termination).

### `event-bus-proxy` is `CrashLoopBackOff`

Check `aiperf kube logs -c event-bus-proxy <job>`. The two real failure modes:

- `AddressAlreadyInUse` on 5663/5664 — something else is bound to those ports inside the pod; almost always a sign that the main control-plane container was started with the legacy in-process proxy (`AIPERF_K8S_EVENT_BUS_SIDECAR_ENABLED=false` on one side and `true` on the other). Align both.
- Missing `run_config.json` — the ConfigMap didn't mount before the container started. `aiperf kube debug <job>` will show the volume error.

### `/healthz` returns `200` but connections to `5663`/`5664` hang

The proxy is up but has no subscribers yet. This is normal during pod startup before the control-plane container has registered as a subscriber on `tcp://127.0.0.1:5664`. If it persists more than ~30 seconds after the control-plane container reports `RUNNING`, inspect its logs for `EventBusProxy` connection errors — the control-plane may be trying to run its own in-process proxy instead of connecting to the sidecar.
