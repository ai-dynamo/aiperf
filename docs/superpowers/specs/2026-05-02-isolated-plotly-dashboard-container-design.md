# Isolated Plotly Dashboard Container — Design

**Status:** approved (2026-05-02)
**Owner:** Anthony Casagrande

## Problem

The Plotly Dash dashboard (`src/aiperf/operator/dashboard_mount.py` +
`src/aiperf/plot/dashboard/`, ~11k LOC, dash 3.1 / plotly 6.4 /
dbc 2.0) is currently mounted in-process inside the operator Pod's
`results-server` sidecar at `/dashboard/`. The sidecar runs with
`requests 100m/512Mi` and `limits 500m/1Gi` (Helm `values.yaml`).

`build_dashboard()` loads parquet/JSON for every run on the PVC into
pandas. At 1M-concurrency runs the build can blow past 1Gi and OOM-kill
the entire `results-server`, which also serves the SPA, the runs index,
the jobs router, and the WebSocket. A single rendering blow-up takes
down everything.

## Goal

Isolate the Plotly Dash app into its own container in the operator Pod
with a relaxed memory budget, behind a Helm toggle. Keep a single
externally-visible URL (the existing `results-server`), refresh the
dashboard whenever a run completes, and re-link `Plots ↗` from the SPA
top-nav.

Out of scope:
- Per-namespace scoping of the Plotly view (stays global across PVC).
- Authn on the dashboard route (inherits whatever the Ingress enforces).
- Dropping `dash`/`plotly`/`dbc` deps from the wheel — the new
  container still needs them, and we ship one image.

## Topology

```
                 ┌──────────────────────────────────────────────────┐
                 │  operator Pod (3 containers when enabled)        │
                 │                                                  │
   ─── HTTP ───▶ │  results-server :8081  ─── proxy /dashboard/* ─▶ │
   (Ingress      │  (FastAPI)                  localhost:8082       │
    or PF)       │                                                  │
                 │                       ┌────────────────────────┐ │
                 │                       │ aiperf-dashboard :8082 │ │
                 │                       │ (uvicorn + Dash WSGI)  │ │
                 │                       │ POST /admin/refresh    │ │
                 │                       └─────────┬──────────────┘ │
                 │                                 │ rebuild        │
                 │  aiperf-operator (kopf) ────────┘                │
                 │   completion handler hits localhost:8082         │
                 │                                                  │
                 │  PVC `/data` — RW in operator, RO in             │
                 │  results-server, RO in dashboard                 │
                 └──────────────────────────────────────────────────┘
```

- Same operator image, different `command:` —
  `python -m aiperf.operator.dashboard_server`.
- External callers hit one URL (port 8081). Inside the Pod,
  `results-server` reverse-proxies `/dashboard/*` to `localhost:8082`
  via an `httpx`-streamed proxy.
- New container is opt-in via `dashboard.enabled` (default `false`).
  When off, results-server's proxy route returns 503 with a friendly
  body and the SPA hides the `Plots ↗` nav entry.

## New module — `src/aiperf/operator/dashboard_server.py`

Tiny FastAPI app, `python -m`-runnable, owned by the new container.

1. **Startup.** Mount `WSGIMiddleware(DashboardProxy(_pending_app(...)))`
   at `/dashboard/`. Kick off `asyncio.create_task(build_dashboard(RESULTS_DIR))`
   to swap in the real Dash app once ready. Lift the existing
   `_mount_dashboard` logic from `results_server.py` verbatim.

2. **`POST /admin/refresh`.** Rebuild in a background task; hot-swap
   `dashboard_proxy.app` on success. Idempotent — a refresh in flight
   short-circuits a second concurrent call (one inflight flag).

3. **`GET /healthz`.** Always 200 once uvicorn is up. Liveness +
   readiness probe target. Readiness does *not* gate on
   "dashboard built" — "no runs yet" is a valid steady state and we
   want the Pod ready so the operator can fire `/admin/refresh`.

## Changes to `results_server.py`

- Remove `_mount_dashboard()` and the `dashboard_mount` import.
- Add a `_dashboard_proxy_router` that streams
  `httpx.AsyncClient.stream(method, url, ...)` requests through to
  `http://localhost:{AIPERF_DASHBOARD_PORT}/dashboard/{path}`.
  - Returns 503 with friendly body when toggle is off
    (`AIPERF_DASHBOARD_PROXY_ENABLED` unset/`"0"`) or upstream
    is unreachable.
  - Forwards method, body, headers (drop `host`, `content-length` is
    re-set by httpx); returns upstream status, body stream, and
    headers verbatim.
  - About 50 LOC; no path-rewriting needed since both sides agree on
    `/dashboard/` prefix.

## Changes to `dashboard_mount.py`

- Stays in tree; still used by the new server module.
- Already lazy-imports dash/plotly inside its functions
  (`_make_dash_app`, `build_dashboard`); no change needed.

## Refresh trigger

The operator's durable-completion site
(`src/aiperf/operator/client_cache.py::try_claim_completion()`) gains
a single fire-and-forget call on successful claim:

```python
if (port := DASHBOARD_PORT) > 0:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"http://localhost:{port}/admin/refresh")
    except (httpx.HTTPError, OSError) as exc:
        self.debug(lambda: f"dashboard refresh skipped: {exc}")
```

Failures (toggle off, sidecar not ready, sidecar down) are swallowed
at debug — refresh is best-effort, not load-bearing.

## Helm — `values.yaml`

```yaml
resultsServer:
  port: 8081
  resources: { ... }    # unchanged

dashboard:
  # dashboard.enabled toggles the isolated Plotly Dash sidecar container.
  # Disabled by default; flip to true to add the third container in the
  # operator Pod and surface "Plots ↗" in the SPA top-nav.
  enabled: false

  # dashboard.port is the Pod-local HTTP port the dashboard listens on.
  # results-server reverse-proxies /dashboard/* to localhost:<port>.
  port: 8082

  # dashboard.resources caps the dashboard container's CPU / memory.
  # Default permissive: 1Gi requested, no limit. Set
  # resources.limits.memory to enforce a ceiling for shared clusters.
  resources:
    requests:
      cpu: 100m
      memory: 1Gi
    limits: {}
```

## Helm — `templates/deployment.yaml`

Wrap a new `{{- if .Values.dashboard.enabled }}` block adding a third
container:

- `image`: same as operator/results-server.
- `command: [python, -m, aiperf.operator.dashboard_server]`
- `env`: `AIPERF_RESULTS_DIR`, `AIPERF_DASHBOARD_PORT`,
  `PYTHONUNBUFFERED=1`, `TMPDIR=/tmp`, `MPLCONFIGDIR=/tmp/matplotlib`.
- `volumeMounts`: PVC at `.Values.storage.mountPath`, `readOnly: true`;
  `tmp` emptyDir at `/tmp`.
- `securityContext`: matches `results-server` —
  `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`,
  drop ALL caps.
- `resources`: `toYaml .Values.dashboard.resources` (with
  `omitempty`-style behaviour on empty `limits`).
- Probes: liveness + readiness `GET /healthz` on
  `.Values.dashboard.port`.
- **No** `Service` port added — the proxy is in-Pod, so port 8082
  stays Pod-local.

The `aiperf-operator` and `results-server` containers each gain
`AIPERF_DASHBOARD_PORT` (defaults to `0` when disabled) and
`AIPERF_DASHBOARD_PROXY_ENABLED` (results-server only). When
`dashboard.enabled` is false, `AIPERF_DASHBOARD_PORT=0`,
`AIPERF_DASHBOARD_PROXY_ENABLED=0` — `try_claim_completion`'s refresh
call is short-circuited, and the proxy route returns 503.

## SPA gating

- `routers/config.py::create_config_router()` — the `/api/config`
  route the SPA already calls on boot — gains a `dashboard_enabled: bool`
  field derived from `os.environ["AIPERF_DASHBOARD_PROXY_ENABLED"]`.
- `src/aiperf/operator/ui-v1/components/top-nav.js` — the
  `NAV_GROUPS` array gains a conditional `Plots ↗` entry rendered as
  `<a target="_blank" href="/dashboard/">` (not the SPA router link)
  when `dashboard_enabled` is true.
- `src/aiperf/operator/ui-v1/app.js` — fetches `/api/config` on boot
  (already done); pipes `dashboard_enabled` to top-nav.

## Documentation

- `docs/kubernetes/dashboard-ui.md` — new section: "Isolated Plotly
  dashboard sidecar — opt-in toggle, default-off rationale, how to
  set memory caps, how the refresh flow works".
- `docs/kubernetes/configuration.md` — document
  `dashboard.enabled` / `dashboard.port` / `dashboard.resources` in
  the values reference.
- `docs/kubernetes/sidecars.md` — add the dashboard container to the
  sidecar inventory.
- `llms.txt` — no entry needed (existing dashboard-ui.md entry covers it).

## Testing

### Unit
- `tests/unit/operator/test_dashboard_server.py` (new) — startup
  mounts placeholder; `/admin/refresh` against an empty PVC keeps
  placeholder, against a populated PVC swaps to a real `dash.Dash`;
  concurrent `/admin/refresh` calls don't double-build.
- `tests/unit/operator/test_dashboard_mount.py` — keep as-is.
- `tests/unit/operator/test_results_server.py` — replace the
  `monkeypatch.setattr(results_server, "build_dashboard", ...)` test
  with an httpx-mocked proxy test:
  - `/dashboard/foo` is forwarded to the configured upstream URL.
  - Response body streams through.
  - 503 returned when upstream unreachable.
  - 503 returned when toggle is off.
- `tests/unit/operator/test_completion_refresh.py` (new) —
  `try_claim_completion` posts to dashboard refresh URL; httpx errors
  and timeouts are swallowed without affecting claim outcome.
- `tests/unit/api/test_config_router.py` — parametrized check that
  `dashboard_enabled` reflects the env var.

### E2E
- `tests/e2e/operator_ui/test_navigation.py` — parametrized variant:
  with `dashboard_enabled=true`, `Plots ↗` appears and opens in a new
  tab; with false, the entry is absent.

### Helm chart
- `tests/kubernetes/` — small `helm template` test verifying:
  - The new container appears iff `dashboard.enabled=true`.
  - `limits` block is omitted when `limits: {}`.
  - The `aiperf-operator` and `results-server` containers get
    `AIPERF_DASHBOARD_PORT` only when enabled.

### Manual smoke
1. `helm upgrade --set dashboard.enabled=true` → 3 containers in
   operator Pod, "Plots ↗" link visible.
2. Run a benchmark to completion → operator log shows
   `dashboard refresh posted`; refresh in dashboard log; clicking
   `Plots ↗` shows the new run.
3. `--set dashboard.enabled=false` → 2 containers, link gone,
   `/dashboard/` returns 503.
4. `--set dashboard.resources.limits.memory=512Mi` → cap enforced;
   OOMKill of dashboard does *not* restart results-server or operator.

## Migration & rollback

- Zero schema or CRD changes; CR-level config untouched.
- Rollback: `helm upgrade --set dashboard.enabled=false`.
- **Behavior change for existing deploys with no override.** They
  get the new opt-in default — they look identical to before this
  change *except* the in-process Dash mount inside `results-server`
  is gone, so `/dashboard/` 503s with a "feature disabled" message
  instead of returning the embedded plot. Anyone relying on the
  old in-process mount must flip the toggle.

## Risks

- **Two-hop latency.** Every `/dashboard/*` request crosses the proxy.
  Localhost loopback is sub-millisecond and Dash callback round-trips
  are already 100ms+ — not material.
- **Sidecar cold start.** First click after Pod start hits the
  placeholder app for a few seconds while `build_dashboard()` walks
  the PVC. The dashboard container does its own initial build at
  startup (mirrors today's `_mount_dashboard` behaviour) so no
  external trigger is needed for the first build.
- **PVC access mode.** The PVC is already mounted RW by the operator
  and RO by results-server; the third RO mount in the dashboard
  container imposes no new access-mode requirement (any
  `ReadWriteOnce` PVC is fine since all three mounts are in the same
  Pod).

