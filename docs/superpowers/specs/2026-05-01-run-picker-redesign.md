# Run Picker Redesign (job-detail page)

**Status:** Spec — pending implementation plan
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-05-01
**Scope:** `src/aiperf/operator/ui-v1/` (frontend) + the `/api/v1/jobs/{ns}/{name}/epochs` endpoint in `src/aiperf/operator/routers/jobs.py`. No CRD or operator-handler changes.

## Problem

The job-detail page in `ui-v1/` ships **two** controls for picking which run (epoch) to view, both bound to the same router state:

- `EpochSelector` — a compact pill-with-overlay-`<select>` inline in the title row at `pages/job-detail.js:2116`
- `RunSelectorCard` — a full-width horizontal pills bar on its own row at `pages/job-detail.js:2193`

The dual control is itself confusing — same selection, two surfaces, slightly different vocabulary. On top of that, both controls use the raw decimal-seconds `epoch` string as the row label (e.g. `"1714150923"`), and the pills bar carries a separate `Live`/`Latest` pseudo-row that *aliases* the most-recent epoch. The result: users can't tell which epoch is "current", which is "the live run", or whether `Live` and `Run 4` (where `Run 4` is the latest epoch) refer to the same thing — they do, but the UI presents them as separate options.

When a job has accumulated 10+ runs, the pills wrap to multiple rows and the lookup task ("which run did I cancel halfway?", "which one finished cleanly?") becomes a hover-and-squint exercise — there is no per-row signal of success/failure or duration; only the artifact-dir mtime and file count.

## Goals

- **One** picker on the page, not two.
- The currently-viewed run is unambiguous in both the collapsed button label and the menu list.
- Per-row status is a glanceable signal (success / failure / cancelled / running), derived once on the server.
- The picker fits in the existing title row — no extra page row, no horizontal wrap.
- Picking a run is one click for "go to latest" and two clicks (open + select) for any other run.
- "Latest" is not a separate entry; it's a property of the most recent row.

## Non-Goals

- **No cross-config comparison.** Same-config epochs of one `AIPerfJob` only. Cross-job/cluster comparison stays on the compare page.
- **No sweep-variation picking.** Sweep variations are a different layer with different data; the sweep-detail page handles those independently.
- **No headline metric column.** Per-row signal is intentionally minimal (status dot + relative time). Adding a metric brings layout/units/sort questions that aren't motivated by a real user task here.
- **No URL change.** Routes stay `/jobs/<ns>/<name>` (latest live) and `/jobs/<ns>/<name>/runs/<epoch>` (pinned). The picker reads/writes the existing router state.
- **No legacy `ui/` (v0) changes.** This redesign is `ui-v1/` only.
- **No behavior change to `runHref`.** It stays in `lib/run-selector.js` and continues to be used by other links (history, dashboard, breadcrumbs).

## Architecture

### Component shape

One new component replaces both existing controls:

```
src/aiperf/operator/ui-v1/components/run-picker.js

export function RunPicker({
  namespace,    // string
  name,         // string
  epochs,       // list[EpochSummary]   — server response, see API section
  current,      // string | undefined   — the epoch the user is viewing (undefined === latest)
  onPick,       // (epoch: string | undefined) => void
})
```

Deletions:

- `components/epoch-selector.js` — entire file removed.
- The `RunSelectorCard` definition and its render site in `pages/job-detail.js` (lines 1521–1572 and 2193–2200) — removed.
- `lib/run-selector.js::buildRunSelectorRows` — removed (only consumer was `RunSelectorCard`).
- `runHref` stays. Callsites in dashboard / history / breadcrumb keep using it unchanged.

The single `RunPicker` callsite slots into the title row at the location of today's `EpochSelector` (`pages/job-detail.js:2116`).

### Backend: API change

Endpoint: `GET /api/v1/jobs/{namespace}/{name}/epochs` (defined at `routers/jobs.py:707`, implemented in `_list_job_epochs_impl` at `routers/jobs.py:386`).

The response shape is widened. Today:

```python
class JobEpochSummary(AIPerfBaseModel):
    epoch: str
    is_latest: bool
    mtime_epoch: int
    file_count: int
```

After:

```python
class JobEpochSummary(AIPerfBaseModel):
    epoch: str
    is_latest: bool
    mtime_epoch: int
    file_count: int
    status: Literal["running", "succeeded", "failed", "cancelled", "unknown"]   # NEW
    started_at: int | None    # NEW — unix seconds, None if unknown
    ended_at: int | None      # NEW — unix seconds, None if still running or unknown
```

`status` is a single normalized enum the UI maps directly to a dot color. The frontend never reconciles `phase` + `error` + CR-level `runEpoch` — that reconciliation is the server's job.

### Backend: status derivation

A single helper computes `status` per row, server-side:

```python
def derive_run_status(
    row: RunIndexRow,
    *,
    live_running_epoch: str | None,
) -> Literal["running", "succeeded", "failed", "cancelled", "unknown"]:
    if live_running_epoch is not None and row.epoch == live_running_epoch:
        return "running"
    if row.error:
        return "failed"
    phase = (row.phase or "").lower()
    if phase == "succeeded":
        return "succeeded"
    if phase == "failed":
        return "failed"
    if phase == "cancelled":
        return "cancelled"
    return "unknown"
```

`live_running_epoch` is computed once per request from the AIPerfJob CR fetched alongside the runs:

- Fetch the CR (already done elsewhere in `routers/jobs.py`; the epochs endpoint will fetch it once).
- If `cr.status.phase == "Running"` and `cr.status.runEpoch` is set, that's the live epoch; otherwise `None`.

The `runs` SQLite index already carries `phase` (`runs_index.py:58`), `error` (`:64`), `start_time` (`:60`), `end_time` (`:61`). No new ingestion or schema changes — pure projection.

### Backend: implementation switch

`_list_job_epochs_impl` switches from the lean `list_runs_async` (which projects through `RunEntry` and drops `phase`/`error`/`start_time`/`end_time`) to reading rich `RunIndexRow` rows via `runs_index.list_runs_for_job(ns, name)`. When the index has no rows for this job (cold cache / never indexed), the implementation falls back to `list_runs_async` and emits `status="unknown"`, `started_at=None`, `ended_at=None` for every entry — a graceful degradation that matches the "we don't know" intent of the unknown enum value.

Ordering remains ascending by `mtime_epoch` (latest at tail), matching today's contract.

### Frontend: API client

`api.getJobEpochs(namespace, name)` already wraps this endpoint in `lib/api.js`. The wrapper keeps its signature; the `epochs` array it returns gains the three new fields. Type docs in the wrapper are updated.

## Components

### `<RunPicker>` — the only public component

Two visible parts: a **collapsed button** that's always rendered (when there's anything to render at all), and a **popover menu** that opens on click.

#### Collapsed button label by state

| State | Button reads | Visual |
|---|---|---|
| Viewing latest, status=running | `Run N · running` | Pulsing blue dot · chevron |
| Viewing latest, status=succeeded | `Run N · 12m ago` | Green dot · chevron |
| Viewing latest, status=failed | `Run N · 12m ago` | Red dot · chevron |
| Viewing latest, status=cancelled | `Run N · 12m ago` | Amber dot · chevron |
| Viewing **older** run (any status) | `Run N · 2h ago  ·  not latest` | Status dot · "not latest" pill (amber background) · chevron |
| Exactly 1 epoch total | `Run 1 · 12m ago` | Status dot · **no chevron** · button has `aria-disabled="true"`, no hover affordance, click is a no-op (popover does not open) |
| 0 persisted epochs | `null` (component renders nothing) | The existing `Live` indicator on `pages/job-detail.js:2101–2107` carries the running signal on its own |

`Run N` numbering is **ordinal by `mtime_epoch` ascending across the response**, so the oldest persisted run is `Run 1` and the newest is `Run M`. Numbering is computed client-side from the response; the server stays opaque about display labels. The raw `epoch` string is shown in the button's `title` tooltip and in the menu row's `title` for the rare cases where a user copy-pastes one into a URL or talks about a specific run by id.

#### Popover menu

Rendered in a custom popover (not native `<select>` — native can't render multi-line entries with dots, badges, and a keyboard-accessible "jump to latest" action). The popover is absolutely positioned relative to the button, dismisses on outside click, `Esc`, or selection, and traps focus while open. This is a new pattern in `ui-v1/`; one prior popover precedent (`components/job-table.js` row menu) covers outside-click dismissal but not full keyboard handling, so the keyboard logic is original to this component.

Order: **newest at the top** (reverse of the API's ascending order — the picker is for navigation, where "most recent first" matches user expectation).

Sticky row at the top of the menu, rendered iff the user is viewing an older run (`current !== undefined && current !== latestEpoch`): `↩ Jump to latest`. Selecting it calls `onPick(undefined)` and dismisses. This replaces the inline "jump to latest" affordance the current `EpochSelector` provides.

Each non-sticky row:

```
[●]  Run N            [latest]            12m ago
```

- `●` — colored dot per `status`.
- `Run N` — same ordinal labelling as the button.
- `[latest]` — a small badge, only on the row matching `is_latest`.
- Right-aligned relative time, computed from `started_at` if present (falls back to `mtime_epoch`).
- The currently-selected row gets a blue tint background and `aria-selected="true"`.

Keyboard:

- `↑` / `↓` move focus through rows (skipping the disabled "jump to latest" sticky if it's not present).
- `Enter` selects the focused row.
- `Esc` dismisses without changing selection.
- `Tab` exits the popover (return focus to the button).

#### Edge cases

- **Stale epoch in URL** that's not in the response — button shows `Run ?(<epoch>) · unknown` with an amber dot and the "not latest" pill. The menu omits the orphan entry but still offers "Jump to latest".
- **>50 entries** — popover has `max-height: 60vh; overflow-y: auto`. No virtualization needed; menu is rendered in a single pass.
- **Server returns `status="unknown"` for some rows** (e.g., index miss) — those rows show a gray dot. Selection still works — picking an unknown-status row just navigates to its epoch URL.

### Data flow inside `<JobDetail>`

The existing wiring is reused unchanged:

- `epochs` is fetched via `api.getJobEpochs(namespace, name)` (`pages/job-detail.js:1665–1666`).
- `current` is the route param `epoch` (`pages/job-detail.js:1619`).
- `pickEpoch` (`pages/job-detail.js:1671`) is passed as `onPick`. It already navigates to the correct URL (latest on `undefined`, run-pinned otherwise).

The page no longer renders `RunSelectorCard` and no longer imports `EpochSelector`. The title-row block keeps every other element (name, phase badge, ns pill, model pill, similar-runs link, relative time, live/completed indicator) and adds `<RunPicker>` where `<EpochSelector>` used to sit.

## Routing

Unchanged.

- `/jobs/<ns>/<name>` — latest live run (no pinned epoch).
- `/jobs/<ns>/<name>/runs/<epoch>` — pinned run.

`runHref` (in `lib/run-selector.js`) is the canonical link builder for both shapes and stays in place.

## Testing

### Backend

- `tests/unit/operator/routers/test_jobs_epochs.py` (or extend the existing module if present): assert the new endpoint shape contains `status`, `startedAt`, `endedAt` per entry; parametrize over `phase`/`error`/`live_running_epoch` combinations exhaustively against `derive_run_status`. One row per status enum value.
- Index-miss / disk-fallback case: assert every entry in the response has `status="unknown"` and `startedAt is None`/`endedAt is None`.
- Live-running case: assert the row whose `epoch == cr.status.runEpoch` carries `status="running"` even if its phase column is stale (the index lags the CR briefly).

### Frontend

- `tests/unit/ui/components/test_run_picker.py` (the existing UI tests are pytest-driven, asserting against parsed JS DOM via the test harness — match the conventions of `tests/unit/ui/components/`):
  - Button label by state (running, completed-latest, viewing-older, single-run inert, zero-run null).
  - Menu rendering: ordinal labelling, latest badge only on `is_latest`, relative time fallback when `started_at` is absent.
  - Status-to-color mapping table (one assertion per enum value).
  - "Jump to latest" sticky shown iff `current !== undefined && current !== latest`.
  - Keyboard nav: `↑`/`↓`/`Enter`/`Esc`/`Tab`.
  - Stale-epoch in URL: button shows `Run ?(<epoch>)`, menu omits the orphan, "Jump to latest" still offered.

### Integration smoke

The existing job-detail integration test that asserts the picker renders and clicking changes route is updated to:

- Assert exactly one picker is in the page (no `data-testid="job-detail-run-selector"` from the deleted card).
- Assert the button carries `data-testid="job-detail-run-picker"` and the popover items carry `data-testid="job-detail-run-picker-row"`.
- Assert clicking a menu row navigates to `/jobs/<ns>/<name>/runs/<epoch>` and clicking "Jump to latest" navigates to `/jobs/<ns>/<name>`.

## Migration / rollout

No feature flag. The `ui-v1/` operator UI is not yet GA, so there are no in-the-wild bookmarks to support. The endpoint shape change is additive — existing API clients ignore the new fields; only the frontend in this repo consumes them.

## Out of Scope

- Per-run headline metric column. Same-config epochs make the metric the comparison signal — but inline in the picker the metric needs units, sort order, and a "best so far" badge that grow the design beyond what the user-task ("which run am I looking at?") needs. The metric belongs on the run cards in the body of the page, not in the picker.
- Cross-namespace / cross-model run comparison. Compare page handles that.
- Sweep variation selection. Different domain object, different page.
- Renaming `epoch` to `run` in the API. The on-disk and CR vocabulary uses `epoch` consistently and changing that is a much wider blast radius than this redesign warrants. The frontend display label is `Run N`; the wire format is `epoch`.
