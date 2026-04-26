# AIPerfSweep — Native Support in `operator/ui-v1` (Design)

**Status:** Draft (brainstorming complete, plan pending)
**Date:** 2026-04-26
**Scope target:** Single PR (or a tight 2-PR split: backend dual-backed sweep API → UI-v1 pages)
**Related:**
- CRD design: `docs/superpowers/specs/2026-04-25-k8s-sweeps-design.md`
- CRD plan:   `docs/superpowers/plans/2026-04-25-k8s-sweeps.md`
- Strict-schema design: `docs/superpowers/plans/2026-04-26-aiperfconfig-strict-crd-schema.md`
- Sweep typesafety design: `docs/superpowers/plans/2026-04-26-aiperfsweep-spec-typesafety.md`

---

## 1. Problem

The `operator/ui-v1` Preact SPA (`src/aiperf/operator/ui-v1/`) has zero awareness of `AIPerfSweep`. Every page treats `AIPerfJob` instances as flat siblings:

- `/jobs` lists every CR in one table — children of a sweep show up undifferentiated next to one-shot benchmarks. Users cannot tell which jobs are part of which campaign, nor see rollup phase / progress.
- `/jobs/:ns/:name` renders per-job detail with no link back to the parent sweep.
- There is no place to view the parent sweep's `currentCell`, `completedRuns`, `failedRuns`, `cells.*` rollup, or the final aggregate JSON.
- `/leaderboard` and `/compare` are flat — they cannot answer "what did varying *concurrency* do to *throughput* in **this** sweep?"

Meanwhile the kopf operator already manages sweeps end-to-end: parent `AIPerfSweep` CRs with rollup status, child `AIPerfJob` CRs labelled `aiperf.nvidia.com/sweep=<sweep-name>` + `variation-index` + `variation-label` + `trial-index`, and aggregate JSON files persisted to the results PVC at sweep terminal.

Sweeps are also the primary multi-run workflow on the cluster. Without UI surface, users either drop to `kubectl get aiperfsweep` (loses the metric story) or stare at a dashboard that pretends every child is a one-off benchmark.

## 2. Goals (v1)

1. **First-class visibility** of sweeps in the UI: list page, detail page with rollup, child list, and per-cell aggregates.
2. **Comparison-axis story** baked into the sweep detail page: chart-or-table view of metrics across the swept dimension(s) — the unique value sweeps add over flat jobs.
3. **Durable, dual-backed** like the existing jobs surface: a sweep that finished months ago and was reaped by TTL still renders fully from the PVC.
4. **Child-side back-link**: `/jobs` and `/jobs/:ns/:name` show "↳ from sweep `<name>`" so users always know the campaign context.

## 3. Explicit Non-Goals (v1)

- **No cancel button.** Sweep cancellation stays kubectl-only for v1 (`kubectl patch aiperfsweep ... --type=merge -p '{"spec":{"cancel":true}}'`). The dual-backed endpoint shape leaves room to add this in v2 without API churn.
- **No create form.** Sweep submission stays kubectl/Helm/YAML-only. The CRD schema is rich enough that a useful form is a separate design.
- **No reuse of `/compare`.** Compare today is "pick N jobs and overlay time-series" — the sweep comparison story is "scan one metric across a swept axis." Different question, different UI; we do not pre-populate Compare with sweep children.
- **No new chart library.** Reuse `chart-wrapper.js` (Chart.js 4) verbatim.
- **No dimension auto-detection from `variation-label` text.** We get structured dimensions from the parent CR / aggregate JSON.

## 4. User-Facing Surface

### 4.1 Navigation

`TopNav` gains one item:

```
Dashboard | Jobs | Sweeps | Leaderboard | Compare | History
                  ^new
```

### 4.2 Routes

| Route                       | Page                | Notes                                           |
|-----------------------------|---------------------|-------------------------------------------------|
| `/sweeps`                   | `SweepsList` (new)  | Filter tabs + table, polled every 5 s.          |
| `/sweeps/:ns/:name`         | `SweepDetail` (new) | Header, rollup KPIs, conditions, cells, children. |
| `/jobs`                     | `Jobs` (existing)   | `JobTable` rows gain a small "↳ sweep:`<name>`" badge when the row is a child. |
| `/jobs/:ns/:name`           | `JobDetail` (existing) | Header gains a "Part of sweep `<name>`" link if owned by a sweep. |

`app.js` adds two route matches; everything else routes unchanged.

### 4.3 `SweepsList` page (`pages/sweeps.js`)

Mirrors `pages/jobs.js` shape:

- Filter tabs: All / Running / Completed / Failed (counts shown).
- Free-text search over name + namespace.
- Optional model filter (only renders when >1 distinct model in the result set, same as jobs).
- Polled `api.listSweeps()` every 5 s.

**Columns:**
- Name (links to detail)
- Namespace
- Phase (pill: Running / Aggregating / Succeeded / Failed / PartiallyFailed / Cancelled)
- Progress: `completedRuns / maxTotalRuns` with mini phase-bar (reuse `phase-bar.js`)
- Failed runs (red badge if >0)
- Variations (`status.totalVariations`)
- Model (from template snapshot)
- Source pill: `live` / `archived` / `both` (same chip as jobs page)
- Age

### 4.4 `SweepDetail` page (`pages/sweep-detail.js`)

Five panels stacked top-to-bottom:

1. **Header strip**
   - Name, namespace, phase pill, age, model from template.
   - Source chip (`live`/`archived`/`both`).
   - When running: `currentCell` line, e.g. `running variation 8/12 trial 4`.

2. **Rollup KPIs** — four `KpiCard`s reused from `components/kpi-card.js`:
   - Total variations
   - Completed runs
   - Failed runs
   - Total runs (incl. trials, i.e. `completed + failed + in_flight`)

3. **Conditions strip** — reuse `components/conditions.js`, same as job-detail.

4. **Cells panel** — toggle between **Chart** and **Table** views over the same per-cell aggregate data:
   - **Chart view (default):**
     - 1D sweep: x = swept dimension values, y = selected metric, error bars from trial spread (when `trials > 1`).
     - 2D sweep: small-multiples — one chart per value of the second dimension, x = first dimension.
     - 3+ D: chart view degrades gracefully to "select two dimensions" — beyond that, the table view is the answer; we do not invent a 3-D chart.
     - Metric selector reuses `components/metric-selector.js`. Default metric: `request_throughput`.
     - Stat selector (avg / p50 / p90 / p95 / p99 / min / max) — defaults: `avg` for throughput-like, `p99` for latency-like (same heuristic the leaderboard uses).
   - **Table view:**
     - Rows = cells (variations).
     - Columns: variation label, dimension values (one column per swept dim), `trials_completed/trials_failed`, then metric columns (configurable, default: `request_throughput`, `ttft`, `itl`) showing chosen stat with sparkline of trial spread.
     - Last column links to first child of the cell; row-click expands an inline list of children.

5. **Children panel** — reuse `JobTable` filtered to this sweep's children. Clicking a row navigates to existing `/jobs/:ns/:name`. No filter tabs duplicated; the list is already scoped.

A "Open in Compare" link sits at the bottom for the rare "I want time-series overlay" case — pre-fills `/compare` with this sweep's children.

### 4.5 Child back-link in existing pages

- `JobTable` row: when `job.sweep_name` is set, render a small italic link `↳ sweep: <name>` under the job name. Clicking the link navigates to `/sweeps/:ns/:sweep_name` and stops propagation.
- `JobDetail` header: when `job.sweep_name` is set, render `Part of sweep [<name>]` directly under the job name pill, link to the sweep detail.

Both surfaces work for archived children too (see §6 for how `sweep_name` is durable).

## 5. API Additions

All read-only. New file `src/aiperf/operator/routers/sweeps.py` parallel to `routers/jobs.py`, factored via the same `create_sweeps_router(api_holder, results_dir)` pattern. Pydantic response models in new `routers/sweeps_models.py`.

### 5.1 Endpoints

```
GET /api/v1/sweeps
  → SweepListResponse { sweeps: SweepSummary[] }
  Lists AIPerfSweep records cluster-wide. Dual-backed via sweep_union
  (live CRs + archived PVC dirs), source-tagged.

GET /api/v1/sweeps/{namespace}/{name}
  → SweepDetailResponse {
      sweep:        SweepSummary,
      status:       <raw .status or synthesized-from-aggregate.json>,
      spec_summary: {
        sweep_type:  "grid" | "scenarios",
        dimensions:  [{ name: str, values: list[Any] }],
        multi_run:   { trials: int, ... } | None,
        convergence: { metric: str, threshold: float, ... } | None,
      },
      children:     ActiveJobSummary[],
    }

GET /api/v1/sweeps/{namespace}/{name}/cells
  → CellAggregatesResponse {
      dimensions: [{ name: str, values: list[Any] }],
      cells: [{
        variation_index: int,
        variation_label: str,
        values:          { dim_name: dim_value, ... },
        trials_completed: int,
        trials_failed:    int,
        metrics: { metric_name: { avg, p50, p90, p95, p99, min, max, stddev } },
        children: [{ namespace, name, trial_index, phase }],
      }],
      source: "live" | "archived" | "both",
    }
```

### 5.2 Job-side touch-ups (additive, no breaking change)

`ActiveJobSummary` (in `routers/jobs_models.py`) gains three optional fields, all `None` for non-sweep children:

```python
sweep_name:      str | None
variation_index: int | None
variation_label: str | None
```

Populated by `job_union.list_all_jobs`:
- For live children: read from labels (`aiperf.nvidia.com/sweep`, `variation-index`, `variation-label`).
- For archived children: read from the child's `sweep.json` marker (see §6.4).

### 5.3 `lib/api.js` additions

```js
api.listSweeps()                    // GET /sweeps
api.getSweep(ns, name)              // GET /sweeps/:ns/:name
api.getSweepCells(ns, name)         // GET /sweeps/:ns/:name/cells
```

`/sweeps` page polls `listSweeps` at 5 s. Detail page polls `getSweep` at 5 s while phase is non-terminal, then once-on-load when terminal. Cells endpoint is fetched once on mount + on every metric/stat selector change (the response is the same data; client-side stat selection happens against `metrics`).

## 6. Durability — `sweep_union` & Archived Sweeps

Mirror `job_union` precisely. Without dual-backing, sweeps disappear from the UI on TTL reap.

### 6.1 Persistence layout (additive)

```
<results_dir>/<ns>/sweeps/<sweep-name>/
  aggregate.json     # parent manifest, written by sweep-controller at terminal:
                     # spec snapshot, per_cell_aggregates[], child_runs[],
                     # final phase, completedRuns, failedRuns, totalVariations,
                     # maxTotalRuns, completedAt
  conditions.json    # parent conditions snapshot (parallel to job conditions.json)
```

The aggregate JSON is the durable parent manifest. Everything `/sweeps/{ns}/{name}` computes from `.status` + `.spec` is recoverable from it. The aggregate path resolution is centralized in `aiperf.operator.results_layout.resolve_sweep_dir(results_dir, ns, name)` (sibling to the existing `resolve_run_dir`).

### 6.2 `src/aiperf/operator/sweep_union.py`

Parallel to `job_union.py`:

```python
@dataclass
class SweepRecord(AIPerfBaseModel):
    namespace: str
    name: str
    source: Literal["live", "archived", "both"]
    phase: str
    total_variations: int
    completed_runs: int
    failed_runs: int
    age_seconds: int
    model: str | None
    spec_summary: SpecSummary
    aggregate_path: str | None     # resolved if archived/both, else None

async def list_all_sweeps(
    api: ApiClient,
    results_dir: Path,
    *,
    all_namespaces: bool = True,
) -> list[SweepRecord]: ...

async def find_any_sweep(
    api: ApiClient,
    results_dir: Path,
    namespace: str,
    name: str,
) -> SweepRecord | None: ...

def synthesize_sweep_status_from_aggregate(
    namespace: str,
    name: str,
    aggregate: dict[str, Any],
    conditions: list[dict[str, Any]] | None,
) -> dict[str, Any]: ...
```

Source-tagging follows the jobs convention exactly: `live` if only the CR exists, `archived` if only the directory exists, `both` if both exist (CR values win on live fields; aggregate values backfill historical-only fields).

### 6.3 Children resolution under archive

| Sweep `source` | Children query                                                      |
|----------------|---------------------------------------------------------------------|
| `live`/`both`  | List `AIPerfJob` by label selector `aiperf.nvidia.com/sweep=<name>` and join with archived children of the same name (existing `job_union` path). |
| `archived`     | Read `child_runs[]` from `aggregate.json`; for each entry, resolve via `find_any_job(api, results_dir, ns, name)` — which dual-backs itself. No apiserver call required if every child is archived. |

### 6.4 Child-side back-link durability

The sweep-controller drops a tiny marker in each child's results dir at child-create time, *before* the child CR is created (so it survives early failures and parent reap):

```
<results_dir>/<ns>/<child-job-name>/sweep.json
  {
    "sweep_name":      "...",
    "variation_index": 7,
    "variation_label": "concurrency-128-rate-50",
    "trial_index":     4
  }
```

`job_union` reads this alongside the existing per-job summary read — one cheap stat-and-read per archived child. Live children prefer labels; archived children read the marker.

### 6.5 Cancel under archive

Out of scope for v1 UI but the endpoint shape needs to be consistent for v2: `POST /sweeps/{ns}/{name}/cancel` against an archived sweep returns `400 "Cannot cancel archived sweep ... the Kubernetes resource no longer exists"` — the same shape as the existing job-side cancel rejection.

## 7. Architecture & File Layout

### 7.1 New files (backend)

```
src/aiperf/operator/
  routers/
    sweeps.py                       # FastAPI router
    sweeps_models.py                # Pydantic response models
  sweep_union.py                    # live + archived join (mirrors job_union.py)

src/aiperf/operator/results_layout.py
  + resolve_sweep_dir(...)          # sibling helper to resolve_run_dir
```

### 7.2 New files (frontend)

```
src/aiperf/operator/ui-v1/
  pages/
    sweeps.js                       # SweepsList
    sweep-detail.js                 # SweepDetail (header, KPIs, conditions, cells, children)
  components/
    cells-chart.js                  # 1D / 2D-faceted Chart.js wrapper for cells
    cells-table.js                  # cell-rows table with trial-spread sparklines
```

### 7.3 Edited files

```
src/aiperf/operator/results_server.py    # register sweeps router
src/aiperf/operator/job_union.py         # add sweep_name/variation_index/variation_label,
                                         # read sweep.json marker for archived children
src/aiperf/operator/routers/jobs_models.py  # add three optional fields to ActiveJobSummary
src/aiperf/operator/handlers/sweep/create.py  # write sweep.json marker before child CR
src/aiperf/operator/ui-v1/app.js         # +2 routes (/sweeps, /sweeps/:ns/:name)
src/aiperf/operator/ui-v1/components/top-nav.js  # +1 nav item
src/aiperf/operator/ui-v1/components/job-table.js  # render sweep-back-link badge
src/aiperf/operator/ui-v1/lib/api.js     # +3 methods (listSweeps, getSweep, getSweepCells)
src/aiperf/operator/ui-v1/lib/state.js   # +sweeps signal
```

### 7.4 Sweep-controller responsibility

The sweep-controller pod is responsible for two new persistence side-effects (both already implied by the dual-backed durability story; calling them out so the implementation plan owns them):

1. **Per-child marker write.** Before creating each child `AIPerfJob` CR, write `<results_dir>/<ns>/<child-job-name>/sweep.json`. Idempotent — overwrite is fine; the deterministic child name is the apiserver-anchored identity.
2. **Parent aggregate write.** At sweep terminal, write `<results_dir>/<ns>/sweeps/<sweep-name>/aggregate.json` and `conditions.json`. Atomic write (`*.tmp` + `os.replace`) to avoid the half-written-JSON read race.

## 8. Data Flow

### 8.1 Live sweep, mid-run

```
SweepDetail page
  ├── poll getSweep(ns, name) every 5 s
  │     └── /sweeps/{ns}/{name} reads CR via k8s API
  │         + sweep_union enriches with aggregate_path if PVC dir exists
  │         + spec_summary derived from CR.spec
  │         + children = job_union.list_all_jobs filtered by sweep label
  │
  └── fetch getSweepCells(ns, name) on mount + metric/stat changes
        └── /sweeps/{ns}/{name}/cells:
              if aggregate.json exists (interim writes possible)
                 → read per_cell_aggregates[]
              else
                 → for each child by label, fetch its profile_export_aiperf.json
                   from PVC, group by variation_index, compute per-cell stats
                   on the fly
```

### 8.2 Archived sweep (CR gone)

```
SweepDetail page
  ├── poll getSweep(ns, name)
  │     └── /sweeps/{ns}/{name}:
  │           CR fetch returns 404 → sweep_union returns archived record
  │           status synthesized from aggregate.json + conditions.json
  │           children = child_runs[] from aggregate.json, each resolved via
  │                      find_any_job (dual-backed itself)
  │
  └── getSweepCells reads per_cell_aggregates[] from aggregate.json directly.
```

### 8.3 Child back-link

```
JobTable row render
  └── job.sweep_name set?
        ├── live child:    set in job_union from aiperf.nvidia.com/sweep label
        └── archived child: set in job_union from sweep.json marker
```

## 9. Error Handling

- `/sweeps` listing: PVC scan failures fall back to empty (best-effort) and only the live half returns; CR fetch failures (non-404) surface `HTTPException` verbatim. Mirrors current jobs behavior.
- `/sweeps/{ns}/{name}`: 404 only if neither the CR nor the PVC dir exists.
- `/sweeps/{ns}/{name}/cells`: live mid-run with no aggregate.json yet and no child summaries → return `cells: []` + `dimensions` from spec, not 404. The chart panel renders an empty state ("No cells completed yet") rather than an error banner.
- aggregate.json malformed (truncated mid-write, decode error): treat as archived-but-corrupted — the record still appears in the list with phase="Unknown", but detail returns `503` with a clear message ("aggregate.json failed to decode at line N").
- Archived child whose `sweep.json` references a non-existent sweep: log at WARN, surface the back-link as a dead link (italic, no anchor) — the child still renders, the user sees that the parent is gone.

## 10. Testing Strategy

### 10.1 Backend unit tests (new)

```
tests/unit/operator/
  test_sweep_union.py
    - live-only sweep: source=live, status from CR
    - archived-only sweep: source=archived, status from aggregate.json
    - both: CR values win on live fields; aggregate backfills completedAt
    - corrupt aggregate.json: surfaces as Unknown phase, not crash
    - children resolution: live (label selector) vs archived (child_runs[])
  test_sweeps_router.py
    - GET /sweeps (live, archived, mixed)
    - GET /sweeps/{ns}/{name} (404 when both absent)
    - GET /sweeps/{ns}/{name}/cells (live mid-run, archived terminal)
    - synthesize_sweep_status_from_aggregate
  test_jobs_router.py (extend existing)
    - ActiveJobSummary.sweep_name set from labels (live)
    - ActiveJobSummary.sweep_name set from sweep.json marker (archived)
    - sweep.json absent → all three new fields are None
  test_results_layout.py (extend existing)
    - resolve_sweep_dir returns expected path
    - resolve_sweep_dir returns None when missing
```

### 10.2 Frontend smoke tests

The existing UI test harness uses `data-testid` selectors. Add:

- `data-testid="page-sweeps"` on `SweepsList` root
- `data-testid="page-sweep-detail"` on `SweepDetail` root
- `data-testid="sweep-cells-chart"` and `data-testid="sweep-cells-table"` on the cells panel views
- `data-testid="job-row-sweep-link"` on the back-link in `JobTable`

JS unit smoke (`tests/_js_cache/`): render `SweepsList` against a mocked `/api/v1/sweeps` response, assert filter tabs + row count.

### 10.3 Integration

`tests/integration/test_sweep_ui_flow.py` (new): submits an `AIPerfSweep` against a kind cluster, polls `/api/v1/sweeps` until terminal, asserts:
- Sweep appears in list with `source="live"`, then `source="both"`, then (after CR delete) `source="archived"`.
- `/cells` returns same `dimensions` and `len(cells)==totalVariations` across all three states.
- Each child's `/jobs/{ns}/{name}` response carries `sweep_name`, `variation_index`, `variation_label`.

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `aggregate.json` write is non-atomic and a torn read crashes the API. | `*.tmp` + `os.replace` write in sweep-controller; reader catches `orjson.JSONDecodeError` and surfaces a clear 503 (per §9). |
| `sweep.json` marker is dropped *before* child CR creation, so a sweep-controller crash leaves orphan markers on PVC. | Markers are content-addressed by child-job-name (deterministic). On replay, the same marker is overwritten. Stale markers for never-created children are harmless: nothing reads `sweep.json` except `job_union`, which only reads it when an actual child summary is also present. |
| Cells endpoint is hot in the UI (mounted per detail render); aggregate.json could be hundreds of KB. | Cells response is computed once per request from on-disk JSON; UI fetches once on mount, re-fetches only on metric/stat selector change (which re-uses the cached client-side response — server hit only when explicitly refreshed). HTTP caching headers `Cache-Control: max-age=2` on terminal sweeps. |
| 2D sweep with many points → chart unreadable. | Faceted small-multiples on the second dimension; cells table view is the fallback for high-cardinality sweeps. 3+ D explicitly degrades to "pick two dims" + table. |
| Archived child whose `sweep.json` points at a sweep whose `aggregate.json` is missing. | UI renders the back-link as a dead-link italic; `/sweeps/{ns}/{name}` returns 404 as expected. No broken render. |
| Sweep CR with `metadata.name == <existing one-shot job name>` would collide on `/jobs` back-link routing. | Sweep CRs and AIPerfJob CRs are different kinds; the routes (`/sweeps/...` vs `/jobs/...`) cannot collide. The display-name collision is purely cosmetic. |
| `variation_label` not DNS-safe historical sweeps may already be deployed. | The marker stores the original (pre-sanitization) label too if available; otherwise the sanitized label is fine for display. |

## 12. Migration / Rollout

- All endpoints are additive — no existing route changes shape.
- `ActiveJobSummary` gains three nullable fields — existing clients that don't know about them ignore unknown JSON keys.
- `sweep.json` marker is written by future sweep-controller releases; sweeps run before this lands have no marker, so the back-link on archived children of *those* sweeps is missing. Mitigation: a one-shot backfill script (`tools/backfill_sweep_markers.py`) walks `<results_dir>/<ns>/sweeps/*/aggregate.json`, reads `child_runs[]`, and writes the markers. Optional — running without backfill just means older children show no back-link, which is current behavior anyway.
- No CRD schema changes. No Helm value changes (the operator already has `results_dir` mounted). No new RBAC.

## 13. Out of Scope (Future Work)

- Cancel button on `SweepDetail` (v2; endpoint shape already accommodated).
- Create-sweep wizard (separate design — needs CRD-form generator or YAML editor).
- Convergence-trace plot per cell (when `convergence` is set, plot trial-by-trial stat vs threshold).
- Per-cell time-series overlay in the chart panel (currently only the cell aggregate is plotted; trial-level series would require pulling each child's parquet).
- Sweep diffing (compare two sweeps that vary the same dimension) — natural extension of `/compare`.

---

## 14. Open Questions

None. All design decisions converged during brainstorming. Ready for plan.
