# Multi-Epoch Support — AIPerfJob + AIPerfSweep + ui-v1 (Design)

**Status:** Draft (brainstorming complete, plan pending)
**Date:** 2026-04-26
**Spec target:** Single feature spec; expect a 2-PR plan (backend + UI).
**Related:**
- Sweep CRD design: `docs/superpowers/specs/2026-04-25-k8s-sweeps-design.md`
- Sweep UI-v1 design: `docs/superpowers/specs/2026-04-26-aiperfsweep-ui-v1-native-support-design.md`

---

## 1. Problem

The on-disk layout already encodes a per-CR run-history dimension (`<base>/<ns>/<name>/<epoch>/`) and `latest.txt` resolves "current," but only one backend surface (`/api/v1/results/.../runs/...`) exposes it. Every other endpoint and the entire ui-v1 collapse to the latest epoch:

- `/api/v1/jobs/{ns}/{name}` → latest only.
- `/api/v1/sweeps/{ns}/{name}` and `/cells` → latest only.
- DuckDB analytics tables are keyed by `(ns, name)` — no epoch column.
- `pages/job-detail.js`, `pages/sweep-detail.js`, breadcrumb, list pages, compare, leaderboard — no epoch awareness.

Re-applying the same `AIPerfJob` CR re-runs and writes a new epoch to disk; older summaries persist on the PVC but are shadowed everywhere except the file-download endpoint. Re-applying an `AIPerfSweep` CR currently overwrites `aggregate.json` in place — there is no sweep epoch story at all.

## 2. Goals (v1)

1. **First-class multi-epoch UI.** Job detail + sweep detail pages get an epoch dropdown and a `/jobs/:ns/:name/runs/:epoch` URL form (sweep equivalent). Latest stays the default at `/jobs/:ns/:name` and `/sweeps/:ns/:name`.
2. **Sweep parity with jobs.** Re-applying a sweep CR creates a new sweep epoch with its own children, results, and aggregate. No data shared between sweep epochs.
3. **Explicit linkage.** A `children.json` per sweep epoch records exactly which (child name, child epoch) tuples belong to that sweep epoch. Authoritative even after the CR is reaped.
4. **Analytics epoch-aware.** DuckDB tables gain an `epoch` column so leaderboard/compare/summary can scope to a specific epoch. Default behavior (no `?epoch`) remains "latest only," preserving today's UX.
5. **No backward compatibility.** The pre-epoch on-disk shape is wiped from the cluster and the codebase. `LEGACY_EPOCH`, `migrate_legacy_layout`, and the `^legacy$` branch in `EPOCH_RE` are deleted. A one-shot wipe script handles existing data.

## 3. Explicit Non-Goals (v1)

- **No imperative "rerun" trigger.** Re-applying the CR is the rerun gesture (consistent with how jobs already work today). No `kubectl annotate aiperfsweep ... aiperf.nvidia.com/rerun=true`.
- **No mid-run partial-epoch surface.** A sweep epoch is created, runs, terminates. While running, only that epoch is visible (older epochs are still browsable). We do not show "epoch 3 (partial) overlaid with epoch 2 (final)" in the same view.
- **No automatic compare-on-rerun.** Compare can opt in to multi-epoch comparison (`?epoch=3`), but the default landing on `/compare` keeps showing one row per `(ns, name)` at latest.
- **No new chart library.**
- **No epoch GC controls in the UI.** Retention is operator-side configuration (`enforce_retention`), not a per-CR knob.

## 4. Sweep Epoch Model & On-Disk Layout

### 4.1 Trigger

A new sweep epoch is created exactly when the kopf operator's `@kopf.on.create` (or `@kopf.on.resume`) handler fires for an `AIPerfSweep` CR — same trigger model as jobs today.

The epoch identifier is `epoch_key_from_body(body)`, i.e. the decimal epoch-seconds parsed from the CR's `metadata.creationTimestamp` (the existing `results_layout.epoch_key_from_body` helper). Apiserver-monotonic per CR-creation, no clock skew, no need to enumerate existing epochs to pick a number — first-ever run and rerun use the same derivation. Re-applying the same CR after a terminal phase forces a new `creationTimestamp` (the user `kubectl delete`s + `kubectl apply`s, or the controller rejects in-place mutation), giving a fresh epoch.

`spec.runEpoch` is **owned by the operator** — never user-supplied. The CRD validation rejects user-set values (`x-kubernetes-validations` rule). `status.runEpoch` mirrors `spec.runEpoch` for kubectl observability.

### 4.2 On-disk layout

```
<base>/<ns>/sweeps/<name>/
  latest.txt                           ← decimal epoch string
  <epoch>/
    aggregate.json                     ← parent rollup
    conditions.json                    ← parent conditions snapshot
    children.json                      ← NEW: explicit (child_name, child_epoch) manifest
```

Mirrors jobs' `<base>/<ns>/<name>/<epoch>/` exactly. `resolve_sweep_dir(base, ns, name)` becomes `resolve_sweep_dir(base, ns, name, epoch=None)` — None falls through to `latest.txt`.

### 4.3 `children.json` shape

```json
{
  "sweep_run_epoch": 1714069323,
  "children": [
    {
      "namespace": "bench",
      "name":      "satsweep-e1714069323-v0007-t04",
      "variation_index": 7,
      "variation_label": "concurrency-128-rate-50",
      "trial_index": 4,
      "child_run_epoch": 1714069324
    }
  ]
}
```

This is the authoritative back-link after parent CR reap. `sweep_union` resolves `latest.txt` → `<epoch>/aggregate.json` + `<epoch>/children.json`; the per-child `sweep.json` marker (already landed) continues to provide the forward-link from a child's results dir to its parent + sweep epoch.

### 4.4 Child naming

Child AIPerfJob CR names embed the sweep epoch:

```
<sweep_name>-e<sweep_epoch>-v<variation_index:04d>-t<trial_index:02d>
```

Each sweep epoch creates its own child CRs in fresh paths. The two epoch axes — sweep epoch and child epoch — are orthogonal:

- Sweep epoch advances when the parent CR is re-applied.
- Child epoch advances when an individual child CR is re-applied (rare, but possible — same mechanism as today's jobs).

Most rerun workflows only bump sweep epoch; children stay at child-epoch 1.

### 4.5 CRD changes

`AIPerfSweep` `status` schema:
- `runEpoch: integer` — already present, semantics tightened to "monotonic, operator-owned."
- `childRunEpochsRef: object | null` — already present (unused). Becomes a structured pointer: `{ epoch: int, count: int, childrenRef: "PVC:<path>/children.json" }`.

`AIPerfJob` `status`:
- `runEpoch: integer` — add if missing (same shape).

## 5. URL & Route Grammar

| URL | Resolves to |
|---|---|
| `/jobs/:ns/:name` | latest epoch (existing route) |
| `/jobs/:ns/:name/runs/:epoch` | specific epoch |
| `/sweeps/:ns/:name` | latest epoch (existing) |
| `/sweeps/:ns/:name/runs/:epoch` | specific epoch |

Vocabulary matches the existing API surface (`/api/v1/results/{ns}/{name}/runs/{epoch}`). No new separators (no `@`, no `epochs/`). Latest is implicit, not `/runs/latest` — keeps shareable links stable across reruns when the user wants "always show me the freshest."

The existing legacy `ui/` (not v1) used the same `/runs/:epoch` shape. We adopt it in ui-v1 too.

## 6. API Surface

All additive — no breaking changes. `?epoch=` always defaults to "latest" when omitted.

```
# Jobs
GET /api/v1/jobs                                    → unchanged (latest of each)
GET /api/v1/jobs/{ns}/{name}                        → latest (existing)
GET /api/v1/jobs/{ns}/{name}?epoch={N}              → NEW: specific epoch
GET /api/v1/jobs/{ns}/{name}/epochs                 → NEW: history list

# Sweeps
GET /api/v1/sweeps                                  → unchanged
GET /api/v1/sweeps/{ns}/{name}                      → latest
GET /api/v1/sweeps/{ns}/{name}?epoch={N}            → NEW: specific epoch
GET /api/v1/sweeps/{ns}/{name}/epochs               → NEW: history list
GET /api/v1/sweeps/{ns}/{name}/cells?epoch={N}      → NEW: epoch-scoped cells
GET /api/v1/sweeps/{ns}/{name}/children?epoch={N}   → NEW: explicit children manifest

# Existing results-files endpoints stay as-is and remain epoch-aware
```

`EpochSummary` response shape (used by both `/jobs/.../epochs` and `/sweeps/.../epochs`):

```python
class EpochSummary(BaseModel):
    epoch: int
    is_latest: bool
    started_at: str        # ISO-8601
    completed_at: str | None
    phase: str             # terminal phase or "Running" for the in-flight epoch
    summary_url: str       # link to ?epoch={N} for convenience
```

## 7. DuckDB Schema Migration

Existing tables in `aiperf.operator.results_db` are keyed by `(namespace, job_id)`. Add an `epoch` integer column to:

- `runs` (one row per persisted job-epoch summary)
- `cells` (one row per sweep-cell aggregate; PK becomes `(ns, sweep_name, sweep_epoch, variation_index)`)

Default-current behaviour (`?epoch=` omitted) → SQL filter `WHERE epoch = (SELECT MAX(epoch) FROM runs WHERE ns=? AND name=?)`. With `?epoch=N` set → exact match.

Migration: since the design wipes existing data (§3), the schema change is destructive — the wipe script also drops and recreates the DuckDB file. No ALTER TABLE in production.

## 8. UI-v1 Changes

### 8.1 Routes (`app.js`)

Add two routes:

```js
matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
matchRoute('/sweeps/:ns/:name/runs/:epoch', currentRoute);
```

Both render their existing detail pages with an `epoch` prop set.

### 8.2 `pages/job-detail.js`

- New header element: epoch dropdown + "viewing epoch N of M" line + click-to-latest button when not on latest.
- New API call on mount: `api.getJobEpochs(ns, name)`. Populates the dropdown.
- Existing data-fetch becomes epoch-aware: `api.getJob(ns, name, epoch)`.
- Conditions panel, KPIs, pods/events panels — all scoped to the chosen epoch (events from k8s API only available for the live epoch; older epochs say "Events not retained for archived epochs").

### 8.3 `pages/sweep-detail.js`

Same shape as job-detail epoch UI:
- Epoch dropdown at top.
- Header rollup, KPIs, conditions, cells panel, children panel — all driven by the chosen epoch.
- Children panel rows now show `(child_name, child_epoch)` and link to `/jobs/:ns/:name/runs/:epoch` with the recorded child epoch.

### 8.4 List pages

`pages/jobs.js` and `pages/sweeps.js` get a new "Epochs" column (count, e.g. `3 ↻`) that links to the latest with the dropdown pre-loaded. No row-expansion; the count is the affordance.

### 8.5 Breadcrumb

`components/breadcrumb.js` renders `bench / satsweep / runs / 3` when the route includes `runs/:epoch`. Clicking `satsweep` goes to latest; clicking `runs` goes to the epochs index page (a thin overlay/sidebar would be over-engineering — clicking `runs` opens the dropdown on the detail page).

### 8.6 Compare and leaderboard (read-only opt-in)

- `lib/api.js`: `compareJobs([{ns, name, epoch}, ...])` — `epoch` optional per item, defaults to latest.
- `pages/compare.js`: when picking jobs/sweeps, the picker shows recent epochs as expandable rows. Selecting an older epoch adds it as a comparison member. Default flow (search-and-add) still picks latest.
- Leaderboard stays single-epoch (latest). No cross-epoch leaderboard in v1.

## 9. Wipe Step (no-backcompat)

`tools/wipe_pre_epoch_results.py`:

- Walks `<base>/<ns>/<name>/` and `<base>/<ns>/sweeps/<name>/`.
- For each candidate dir: if its immediate children are NOT all integer epoch subdirs (i.e. it has files like `profile_export_aiperf.json` directly, or the literal `legacy/` subdir, or `aggregate.json` outside an epoch dir), `rm -rf` the whole `<name>/`.
- For DuckDB: drops and recreates the database file.
- Default: dry-run with verbose log of what would be deleted. `--apply` actually runs.
- Run-once via `kubectl exec` against the operator pod's `/data` mount.

The script does NOT touch CRs in the apiserver (no kubectl). Cluster operators run `kubectl delete aiperfjobs --all -A; kubectl delete aiperfsweeps --all -A` separately if they want a fresh tabula rasa, but that's optional.

Codebase changes that ride along:

- Delete `LEGACY_EPOCH`, `migrate_legacy_layout`, and the `^legacy$` branch in `EPOCH_RE` from `src/aiperf/operator/results_layout.py`.
- Delete the legacy-handling test paths in `tests/unit/operator/test_results_layout.py`.
- `EPOCH_RE` becomes `re.compile(r"^\d{9,11}$")` exactly.

## 10. File Map

### 10.1 Backend — new files

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/routers/epochs.py` | `EpochSummary` model + helper `list_epochs(ns, name, kind=Literal["jobs","sweeps"])` for the two `/epochs` endpoints. |
| `tools/wipe_pre_epoch_results.py` | One-shot pre-epoch wipe. |

### 10.2 Backend — edited files

| Path | Change |
|---|---|
| `src/aiperf/operator/results_layout.py` | Add `resolve_sweep_dir(..., epoch=None)`. Drop `LEGACY_EPOCH`, `migrate_legacy_layout`, `^legacy$`. |
| `src/aiperf/operator/job_union.py` | Optional `epoch` param on `find_any_job`, `_archived_from_summary`, `_scan_pvc_jobs`. |
| `src/aiperf/operator/sweep_union.py` | Optional `epoch` param on `find_any_sweep`, `list_all_sweeps` stays latest-only. New `_record_from_archive(..., epoch)`. |
| `src/aiperf/operator/routers/jobs.py` | `GET /jobs/{ns}/{name}` accepts `?epoch=`. New `GET /jobs/{ns}/{name}/epochs`. |
| `src/aiperf/operator/routers/sweeps.py` | `GET /sweeps/{ns}/{name}` accepts `?epoch=`. New `GET /sweeps/{ns}/{name}/epochs`, `GET /sweeps/{ns}/{name}/children`. `GET /cells` accepts `?epoch=`. |
| `src/aiperf/operator/routers/jobs_models.py` | Add `JobEpochsResponse`. |
| `src/aiperf/operator/routers/sweeps_models.py` | Add `SweepEpochsResponse`, `ChildrenManifestResponse`. |
| `src/aiperf/operator/handlers/sweep/create.py` | Compute new epoch on (re)create from PVC; write `spec.runEpoch` ; embed sweep epoch in child names; drop `children.json` after children are created. |
| `src/aiperf/operator/handlers/sweep/lifecycle.py` | `latest.txt` write at sweep terminal (alongside aggregate.json). |
| `src/aiperf/sweep_controller/aggregator.py` | `write_sweep_aggregate` writes into `<base>/<ns>/sweeps/<name>/<epoch>/...`. New `write_children_manifest`. |
| `src/aiperf/sweep_controller/k8s_executor.py` | Sweep epoch in child name template; pass `child_run_epoch` into `sweep.json` marker. |
| `src/aiperf/operator/results_db.py` | Add `epoch` column to schema; default queries select `MAX(epoch)`. |
| `src/aiperf/operator/results_server.py` | Wire any new routers if extracted. |
| `deploy/helm/aiperf-operator/templates/crd.yaml` | `runEpoch` validation rules on AIPerfJob status. |
| `deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml` | `runEpoch` rules + structured `childRunEpochsRef`. |

### 10.3 Frontend — new files

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/ui-v1/components/epoch-selector.js` | Reusable dropdown showing the epoch list, "viewing N of M," click-to-latest. |

### 10.4 Frontend — edited files

| Path | Change |
|---|---|
| `app.js` | Two new route matches (`/jobs/:ns/:name/runs/:epoch`, `/sweeps/:ns/:name/runs/:epoch`). |
| `lib/api.js` | New methods: `getJobEpochs`, `getJob(ns, name, epoch?)`, `getSweepEpochs`, `getSweep(ns, name, epoch?)`, `getSweepCells(ns, name, epoch?)`, `getSweepChildren(ns, name, epoch)`, `compareJobs([{ns,name,epoch?}, ...])`. |
| `pages/job-detail.js` | Mount `EpochSelector`; epoch-aware data fetch. |
| `pages/sweep-detail.js` | Mount `EpochSelector`; epoch propagated to all panels. |
| `pages/jobs.js` | New "Epochs" column. |
| `pages/sweeps.js` | New "Epochs" column. |
| `pages/compare.js` | Per-pick epoch override; opt-in epoch comparison. |
| `components/breadcrumb.js` | Render `runs/<epoch>` segments. |

## 11. Data Flow

### 11.1 First-ever sweep run

```
User: kubectl apply -f sweep.yaml (name=satsweep)
operator.handlers.sweep.create:
  walk <base>/<ns>/sweeps/satsweep/ → no existing epoch
  derive epoch = epoch_key_from_body(body)            # creationTimestamp seconds
  patch: spec.runEpoch=<epoch>
  create JobSet for sweep-controller pod
sweep-controller (in pod):
  determines child names: satsweep-e<epoch>-v<v>-t<t>
  creates child AIPerfJob CRs (ownerRef=AIPerfSweep)
  drops sweep.json marker per child (with sweep_run_epoch + child_run_epoch=<epoch>)
  runs each to terminal, collects RunResults
  on terminal:
    write <base>/<ns>/sweeps/satsweep/<epoch>/aggregate.json
    write <base>/<ns>/sweeps/satsweep/<epoch>/conditions.json
    write <base>/<ns>/sweeps/satsweep/<epoch>/children.json
    write <base>/<ns>/sweeps/satsweep/latest.txt = "<epoch>"
```

### 11.2 Sweep rerun

```
User: kubectl apply -f sweep.yaml (same name) after prior run finished
operator.handlers.sweep.create:
  walk <base>/<ns>/sweeps/satsweep/ → finds <prev_epoch>/, latest.txt
  derive epoch = epoch_key_from_body(body)            # new creationTimestamp
  patch: spec.runEpoch=<epoch>
  proceeds identically — fresh child CRs, fresh dirs.
```

### 11.3 UI epoch fetch

```
SweepDetail mount (ns=bench, name=satsweep, epoch=undefined)
  → api.getSweepEpochs("bench", "satsweep")  → [<e1>, <e2>, <e3>] with is_latest on e3
  → api.getSweep("bench", "satsweep")         → latest = e3 detail
  → api.getSweepCells("bench", "satsweep")    → e3 cells
  user picks e2 in dropdown → navigate("/sweeps/bench/satsweep/runs/<e2>")
SweepDetail (epoch=e2):
  → api.getSweep("bench", "satsweep", e2)
  → api.getSweepCells("bench", "satsweep", e2)
```

## 12. Error Handling

- Missing `latest.txt`: every persisted dir has it post-wipe; if absent, return 500 with a clear message ("PVC corrupted: latest.txt missing"). Not a fall-through case anymore.
- `?epoch=N` for an N that doesn't exist: 404 with `Epoch <N> not found for <ns>/<name>; available: [...]`.
- `?epoch=N` syntactically invalid (non-digit): 400 immediately at the FastAPI Path validator (use `re.fullmatch(r"\d{9,11}", epoch)` or `int(epoch)` with explicit catch).
- `children.json` missing on an archived sweep: degrade gracefully — render rollup + cells, render children panel as "Children list unavailable for this archived epoch (children.json missing)." Should be impossible after this design lands; possible only on a sweep written by an older controller mid-deploy.
- DuckDB query without `epoch` filter: silently uses `MAX(epoch)` per `(ns, name)`. No cross-epoch joins by accident.

## 13. Testing Strategy

### 13.1 Unit tests (new/extended)

- `tests/unit/operator/test_results_layout.py` — extend `resolve_sweep_dir` to take epoch; assert legacy paths (`LEGACY_EPOCH`, `^legacy$`) are gone.
- `tests/unit/operator/test_sweep_union_epochs.py` — sweep epoch enumeration, archived rendering.
- `tests/unit/operator/test_jobs_router_epochs.py` — `?epoch=`, `/epochs` listing.
- `tests/unit/operator/test_sweeps_router_epochs.py` — `?epoch=`, `/epochs`, `/children`.
- `tests/unit/operator/test_results_db_epoch.py` — DuckDB MAX(epoch) default + explicit filter.
- `tests/unit/sweep_controller/test_aggregator_epoch.py` — `write_sweep_aggregate` writes under `<epoch>/`.
- `tests/unit/sweep_controller/test_k8s_executor_child_naming.py` — child name embeds sweep epoch.
- `tests/unit/operator/handlers/test_sweep_create_epoch.py` — first-run vs rerun epoch derivation.
- `tests/unit/tools/test_wipe_pre_epoch_results.py` — dry-run + apply behaviors.

### 13.2 Integration

- `tests/integration/test_multi_epoch_flow.py` (new) — kind cluster, submit sweep, verify epoch=N1 lands; resubmit, verify epoch=N2 lands; verify both visible via API; verify `children.json` per epoch; verify DuckDB has rows tagged with epoch.

### 13.3 Frontend

- `data-testid="epoch-selector"` on the dropdown; `data-testid="epoch-banner-not-latest"` when off-latest.
- Smoke render against mocked `/epochs` response.

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Sweep-controller crashes mid-run after creating some children but before writing `children.json`. | The controller always rebuilds the children list from labels (`aiperf.nvidia.com/sweep=<name>` + `sweep-run-epoch=<epoch>`) on resume. `children.json` is a snapshot of that label scan, written at terminal — restartable. |
| Two sweep epochs racing because of clock skew across nodes. | Epoch derives from `creationTimestamp` of the CR, set by the apiserver — single source. Concurrent reapply of the same name is an apiserver conflict, not an epoch race. |
| User reapplies a sweep CR while the prior epoch is still mid-run. | Apiserver enforces uniqueness on `(kind, namespace, name)` — `kubectl apply` on a name that already has a live CR mutates the existing CR, it does not create a new one. To start a new epoch the user must `kubectl delete aiperfsweep <name>` first; the prior CR's `metadata.deletionTimestamp` propagates to the controller (cooperative cancel), and only then can the user `kubectl apply` a fresh one — which produces a new `creationTimestamp` and therefore a new epoch. Same gesture as job rerun today. |
| DuckDB schema rebuild breaks existing local dev data. | Wipe step is part of the rollout. Devs running locally get a fresh DB — that's the explicit trade-off for "no backcompat." |
| URL change breaks bookmarks. | `/jobs/:ns/:name` and `/sweeps/:ns/:name` (latest) remain unchanged. Only `runs/:epoch` is new. No existing bookmark is affected. |
| Old children with no sweep epoch in their name collide with new ones. | After the wipe step there are no old children. Going forward, names always include `e<sweep_epoch>` so no collision possible. |

## 15. Out of Scope (Future Work)

- Cross-epoch leaderboard / "show me how request-throughput drifted across the last 5 epochs of this sweep." Natural compare extension; tracked separately.
- Convergence-per-epoch trace plotting.
- Annotation-based rerun trigger (`aiperf.nvidia.com/rerun=true`) — not needed if "kubectl apply after terminal" is the rerun gesture.
- Per-epoch retention controls in the UI (today: operator config only).
- Diffing two epochs side-by-side at the cell level ("epoch 3 vs 5: which cells improved?").

## 16. Open Questions

None. Brainstorming converged. Ready for plan.
