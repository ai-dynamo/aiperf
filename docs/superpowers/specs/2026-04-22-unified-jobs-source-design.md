# Design: Unified Jobs Source (CR + PVC Union)

**Status:** Draft
**Date:** 2026-04-22
**Scope:** Fix the split-source confusion in the operator web UI by making `/api/v1/jobs` the single source of truth — a union of cluster CRs and PVC result directories, keyed by `(namespace, name)`.

---

## 1. Problem

The operator UI shows jobs from two unrelated sources, and each page picks only one:

| Source | Lifecycle | Used by |
|---|---|---|
| **Cluster CRs** (k8s `AIPerfJob`) | seconds–hours; deleted on cleanup | Dashboard counts, Jobs list, Job Detail status pane |
| **Results PVC** (`<ns>/<name>/profile_export_aiperf.json`) | weeks–months; persists after CR deletion | Dashboard KPIs, Leaderboard, History, Compare, Job Detail metrics |

Consequences:
- A finished benchmark whose CR has been reaped **disappears from the Jobs page** but still appears in Leaderboard/History. Users think their run vanished.
- A currently-running benchmark appears on Jobs but not in Leaderboard (nothing written yet). Users think the Leaderboard is broken.
- The Dashboard "completed" count reflects *only* CRs, under-counting real history.

The UI doesn't badge which source a row came from, so the user has to reason about both data planes simultaneously.

## 2. Goals

1. **One logical concept of "a job"** at the API layer: every benchmark that ever ran or is running, keyed by `(namespace, name)`.
2. **Source-tagged entries** so the UI can render provenance badges.
3. **Jobs page becomes the complete index** — Running, Completed, Failed, *and* Archived (historical runs whose CR is gone).
4. **Job Detail gracefully handles archived** — no 404 when the CR is gone but the PVC dir exists.
5. **Dashboard counts** reflect the union.

## 3. Non-goals

- Changing the Leaderboard/History/Compare pages (they're already intentionally PVC-only — analytics is the point).
- Persisting more CR state to disk or replicating PVC state into etcd. The split at the *storage* layer is correct; only the *API and UI* layers are wrong.
- Garbage collecting historical results.
- Changing the CRD.

## 4. Data model

### 4.1 `source` discriminator

Add to `AIPerfJobInfo`:

```python
source: Literal["live", "archived", "both"] = Field(
    default="live",
    description=(
        "Provenance: 'live' = CR on cluster only (typically running or "
        "just-completed with no results persisted yet); 'archived' = PVC "
        "results only (CR no longer exists); 'both' = CR + PVC results "
        "(the common steady state for a completed run)."
    ),
)
```

### 4.2 Archived phase

When an entry is `source="archived"` (PVC-only), the CR is gone so there's no `status.phase`. Derive from the summary JSON's top-level `status` field if present (generator writes `"status": "Succeeded"`), else fall back to `"Archived"`.

### 4.3 Live-only fields when archived

For `source="archived"` entries:
- `workers_ready = workers_total = 0`
- `progress_percent = 100.0` (run is done by definition)
- `throughput_rps` / `latency_p99_ms` / `model` / `endpoint` read from summary JSON's `request_throughput.avg`, `request_latency.p99`, `input_config.models.items[0].name`, `input_config.endpoint.urls[0]` — the same fields DuckDB reads.
- `created` / `start_time` / `completion_time` read from summary JSON's `start_time`/`end_time` if present; else file mtime of `profile_export_aiperf.json`.

## 5. Architecture

### 5.1 New helper: `list_all_jobs`

**Location:** `src/aiperf/operator/job_union.py` (new file, one responsibility).

```python
async def list_all_jobs(
    api: ApiClient | None,
    results_dir: Path,
    *,
    all_namespaces: bool = True,
    namespace: str | None = None,
) -> list[AIPerfJobInfo]:
    """Return the UNION of cluster CRs and PVC result directories.

    Keyed by (namespace, name). Each entry's `source` field marks whether
    it came from the cluster, the PVC, or both. Live fields (workers, pods,
    progress) are populated for CR-backed entries; historical fields
    (throughput, latency, model) are populated for PVC-backed entries.
    `both` entries favor the CR for live fields and fall back to PVC data
    where the CR is silent.
    """
```

### 5.2 Find-one helper: `find_any_job`

```python
async def find_any_job(
    api: ApiClient | None,
    results_dir: Path,
    namespace: str,
    name: str,
) -> AIPerfJobInfo | None:
    """Return a single job by (namespace, name). Prefers CR; falls back to PVC."""
```

### 5.3 Router wiring

`src/aiperf/operator/routers/jobs.py`:

- `GET /api/v1/jobs` now calls `list_all_jobs(api, results_dir, all_namespaces=True)`.
- `GET /api/v1/jobs/{ns}/{name}` uses `find_any_job`. If `source="archived"`, returns `JobDetailResponse` with `status={}` (or a synthesized status from the summary JSON) and `pods=[]`; keeps the existing shape.
- `POST /api/v1/jobs/{ns}/{name}/cancel` returns 400 when `source="archived"` — there's no CR to patch.

The results directory is already passed to `create_jobs_router` via a new parameter (currently only `api_holder` is passed); we add `results_dir: Path` to the router factory signature.

### 5.4 UI changes

| Component | Change |
|---|---|
| `pages/jobs.js` | New tab "Archived" next to Running/Completed/Failed; counts from `.filter(j => j.source === 'archived').length`. Add a small source badge (`live`/`archived`/`both`) next to the status pill. |
| `components/job-table.js` | Add source badge column (or inline next to phase). |
| `pages/job-detail.js` | When `info.source === 'archived'`: hide Cancel, hide Pods card, show a single-line banner "This run's Kubernetes resource has been deleted. Showing archived results from the PVC." |
| `pages/dashboard.js` | Count tiles (Running/Completed/Failed) read from the new union. Running is still CR-only (archived can't be running), but Completed/Failed include archived-equivalents (status==Succeeded / status==Failed from the summary JSON). |

## 6. Edge cases

1. **Archived job with no summary JSON** — shouldn't happen (the PVC scan keys on `profile_export_aiperf.json` existing), but guard: skip such dirs, log a warning.
2. **CR present but name mismatched from PVC dir** — extremely rare; the operator writes to `<ns>/<name>/` verbatim. The union key is `(metadata.namespace, metadata.name)` for CRs and `(ns_dir.name, job_dir.name)` for PVC — they align by construction.
3. **Running job that already has partial results on disk** — source becomes `both`; CR wins for live fields (it has real-time data); PVC fills in what CR doesn't yet have.
4. **Archived job's cancel** — the cancel button is hidden; the API returns 400 defensively.
5. **Namespace filter narrowing** — `list_all_jobs(namespace="foo")` filters both CRs and PVC dirs. PVC scan only descends into `<base>/foo/`.
6. **Results scan performance** — `_scan_job_dirs` is O(N_jobs) directory reads. At 1000 archived jobs this is still fast. If it becomes a bottleneck, add a 5s in-memory cache keyed by `results_dir` mtime.

## 7. Testing

Unit tests (`tests/unit/operator/test_job_union.py`):
- `list_all_jobs` with CR-only → all entries `source="live"`.
- `list_all_jobs` with PVC-only → all entries `source="archived"`; fields derived from summary JSON.
- `list_all_jobs` with overlap → the overlap entries are `source="both"`, non-overlap keep their own source tag.
- `list_all_jobs` filters by namespace on both sides.
- `find_any_job` prefers CR when both exist; falls back to PVC; returns None when neither.
- Archived entry with `"status": "Failed"` in summary JSON → `phase="Failed"`.
- Archived entry with missing `status` key → `phase="Archived"`.

E2E tests (extend `tests/e2e/operator_ui/`):
- New fixture `archived_only_job` that writes a PVC dir without creating a matching CR.
- `test_jobs_shows_archived_jobs` — archived entries appear on the Jobs page.
- `test_jobs_archived_filter_tab` — filtering by Archived shows only archived rows.
- `test_job_detail_archived_banner_hides_cancel` — archived job detail shows banner, no Cancel button, no Pods card.
- `test_dashboard_counts_include_archived` — Completed count includes archived-Succeeded entries.

## 8. Backwards compatibility

- `AIPerfJobInfo.source` has a default of `"live"`, so existing consumers that construct it directly (CLI, tests) don't break.
- The existing `/api/v1/jobs` response shape is unchanged (`{"jobs": [...]}`), just with the additional `source` field on each entry.
- CLI `aiperf kube list` (if it consumes `/api/v1/jobs`) will silently ignore the new field.
