# Namespace-Workflow Operator UI Redesign

**Status:** Spec — pending implementation plan
**Owner:** Anthony Casagrande (acasagrande@nvidia.com)
**Date:** 2026-04-26
**Scope:** Frontend redesign of `src/aiperf/operator/ui/` only. No backend / operator API changes.

## Problem

The current operator UI treats Kubernetes namespace as a *grouping field* on jobs — runs flow into a single global Home, Archive, and Launch surface; namespace is a column. Operators in practice live in one namespace at a time (their team's), occasionally needing a cross-namespace view. The UI shape doesn't match that workflow: every action takes a namespace decision out of muscle memory and into a per-screen prompt.

This redesign makes namespace a first-class navigation context: pick a namespace, then work inside it. Cross-namespace views remain for situational awareness and analytics, but they are no longer the default landing surface.

## Goals

- Namespace is the primary axis of operational navigation.
- The URL alone tells you which namespace an action will affect.
- Switching namespaces is one click and is visible at all times in the breadcrumb.
- Cross-namespace situational awareness ("is anything broken anywhere?") is preserved.
- Cross-namespace comparison/analytics remains possible (cluster-key driven).

## Non-Goals

- No backend / operator API changes. The picker derives namespace summaries client-side from the existing `api.listJobs()` response.
- No per-namespace operator config (e.g., per-namespace retention overrides). Retention stays global. A future spec can add this; not bundled here.
- No feature-flag or `?v2=true` toggle. The operator UI has not shipped publicly, so there are no in-the-wild URLs to support and no migration shims for legacy paths.
- No left sidebar / persistent namespace rail. Top-rail breadcrumb pill is the switcher.
- No changes to the analysis/compare logic itself — only its mount path stays in the cross-namespace tier.
- The legacy `ui-v1/` tree is unrelated to this work and is unchanged.

## Architecture

### Two-tier route model

- **Cross-namespace tier** (unprefixed): `/`, `/analysis`, `/log`, `/settings`. Used for situational awareness across namespaces and for analytics that benefit from a wider lens.
- **Per-namespace tier** (`/ns/<name>/...`): every operational view — overview, launch, archive, single run. The namespace in the URL is the authoritative scope for any state-changing action.

### Route table

| Route | Purpose |
|---|---|
| `/` | Cross-namespace picker (tiles with mini-status). On app mount, redirects to `/ns/<last>` if `localStorage.aiperf.ui.lastNamespace` is set and that namespace appears in the cluster's namespace list (derived from current jobs). |
| `/ns/:ns` | Per-namespace overview — replaces the current `/` Home, scoped to one namespace. |
| `/ns/:ns/launch` | Launch into `:ns`. YAML's `namespace:` is auto-filled from URL and locked (see Launch contract below). |
| `/ns/:ns/archive` | Namespace history — runs in `:ns` only. |
| `/ns/:ns/run/:name` | Single-run workbench. |
| `/ns/:ns/run/:name/runs/:epoch` | Single-run epoch view. |
| `/analysis` | Cross-namespace analysis (cluster-key-aware). Unchanged in shape. |
| `/log` | Durable run log (cross-namespace). Unchanged. |

### Resolution rule (in `app.js`)

`resolveView()` matches `/ns/:ns/...` patterns first, then the unprefixed analytics routes. There are no legacy patterns (no `/jobs`, no `/fleet`, no `/launch` without ns, no `/run/:ns/:name`). Unmatched routes fall through to `/` (the picker), so stale bookmarks land somewhere usable.

A landing-redirect effect runs once at app mount. If `route.value === '/'` and `localStorage.aiperf.ui.lastNamespace` is set and that namespace currently has at least one job in `jobs.value` (i.e., it appears in the picker), `navigate('/ns/<last>')`. Otherwise stay at `/` and render the picker.

## Components

### `<NamespacePicker>` — mounted at `/`

Renders one tile per namespace observed in `jobs.value`:

- Tile content: namespace name (large), one-line activity summary (`3 active · 41 total`), per-phase chip row (`Running 2 · Failed 1`), last-activity timestamp, left-edge state tint (`live` if any active, `fault` if any failed within the last 24 h and zero live, `quiet` otherwise).
- A search box filters tiles by namespace name (typeahead, exact substring — namespace names are short enough that fuzzy is overkill).
- Click a tile: writes `localStorage.aiperf.ui.lastNamespace = name`, then `navigate('/ns/<name>')`.
- Empty cluster (no namespaces visible to the operator's RBAC, i.e., no jobs at all) renders an empty state pointing at the kube preflight docs. We do not show a launch CTA at the cross-namespace tier because launch requires a namespace.

A namespace tile only appears once that namespace has at least one job (current or historical). Empty-but-deployable namespaces are not surfaced. Adding a `/api/v1/namespaces` endpoint to enumerate every RBAC-visible namespace is out of scope; it slots in cleanly later because the picker is namespace-driven.

### `<NamespaceOverview>` — mounted at `/ns/:ns`

The per-namespace dashboard. Same layout as the current `views/home.js` but the data is filtered to `j.namespace === ns`:

- Stats hero (running / passed / failed / total / GPUs).
- Active runs strip (`ActiveCard` per live run).
- Recent runs table.

Empty namespace (zero active *and* zero historical):

- Renders the namespace name in the empty-state heading: "No runs yet in `<ns>`".
- A single "Launch in `<ns>`" CTA navigating to `/ns/<ns>/launch`.
- No inline templates and no auto-redirect to launch — the URL the user typed is the URL they get.

Per-namespace UI preferences (see `localStorage` schema below) drive:

- `pinnedRunNames`: pinned runs surface at the top of the recent-runs table.
- `lastLaunchTemplateId`: pre-loaded into the editor next time `/ns/<ns>/launch` opens.
- `overviewMetricKey`: which throughput/latency series the overview chart highlights.

### `<TopRail>` and breadcrumb (modified)

- At `/`: breadcrumb is the existing `AIPERF` callsign only. LAUNCH CTA is hidden.
- At `/ns/:ns/...`: breadcrumb format `<ns ▾> › <segment> [› <segment>]`. Segments derive from the route — `launch`, `archive`, `run/<name>`, `runs/<epoch>` collapsed to `epoch <short>`.
- Clicking the namespace pill opens a `<NamespaceSwitcher>` dropdown — a slim variant of the picker (same tile content, compact, with typeahead). Selecting a namespace navigates to `/ns/<chosen>` (overview), not the analogous sub-route. Switching is a context change, not a sideways deep-link.
- A "View all namespaces" item at the bottom of the dropdown navigates to `/`.
- ⌘K command palette is unchanged in shape; results are namespace-aware: when in `/ns/foo/...`, `foo`'s runs surface above a divider, others below. In cross-namespace routes, results group by namespace (current behavior).

### Reused components, namespace-filtered data

`Run` view (`/ns/:ns/run/:name`), `Archive` (now per-namespace at `/ns/:ns/archive`), `Launch` — same components, route param `:ns` becomes the filter input. No structural changes beyond the prop wiring and route-key updates.

## Data flow

### Backend: zero changes

This redesign is entirely frontend. The picker derives namespace tiles client-side by grouping `api.listJobs()` by `j.namespace`. Per-tile aggregation:

- `running = count(j.phase ∈ {pending, initializing, running})`
- `failed = count(j.phase ∈ {failed, error}) within last 24h`
- `completed = count(j.phase ∈ {completed, succeeded})`
- `lastActivity = max(j.lastUpdate ?? j.startTime)`

### Polling

Single global poll of `api.listJobs()` continues at the app level (current behavior). The `jobs` signal stays global. Per-namespace views compute their slice via `jobs.value.filter(j => j.namespace === ns)`. No per-page polling; switching namespaces is instant (no refetch).

### Launch validation contract

At `/ns/:ns/launch`:

1. **On view mount.** If the editor is empty, insert a starter YAML with `namespace: <ns>` populated. If a template is loaded via `lastLaunchTemplateId`, the template's `namespace` field is overwritten with `<ns>` before display.
2. **On every keystroke** (debounced ~150 ms). A lightweight YAML parse extracts the `namespace:` field. If the field is absent, the divergence flag is *not* raised — absence is normal during editing and the operator submit path will use the URL's namespace as the default (the same value the auto-fill inserted). If the field is present and diverges from `:ns`, the breadcrumb namespace pill gets a `chip--bad` tint and the LAUNCH button disables. Hover-tooltip on the disabled button: "YAML namespace `<typed>` doesn't match `<ns>`. Switch namespaces or fix the YAML."
3. **On submit.** The submit handler refuses to call `api.createJob()` while the divergence flag is set. This is a UI lock; the backend API itself is unchanged.

### `localStorage` schema

| Key | Type | Purpose |
|---|---|---|
| `aiperf.ui.lastNamespace` | `string` | Sticky last-used namespace |
| `aiperf.ui.ns.<ns>.pinnedRunNames` | `string[]` | Per-namespace pinned runs surfaced on overview |
| `aiperf.ui.ns.<ns>.lastLaunchTemplateId` | `string \| null` | Template auto-loaded next time `/ns/<ns>/launch` opens |
| `aiperf.ui.ns.<ns>.overviewMetricKey` | `string` | Which series the overview chart highlights |

All keys are best-effort: missing key ⇒ default behavior. Quota errors are swallowed silently. A small `lib/ns-prefs.js` module wraps these (`getNsPref(ns, key, default)` / `setNsPref(ns, key, value)`) so views don't sprinkle raw `localStorage` calls.

### State on namespace switch

Switching namespaces is just `navigate('/ns/<other>')`. Component state is route-keyed via Preact's standard re-mount semantics (`<NamespaceOverview ns={ns} key={ns} />` where state isolation matters). The `jobs` signal is shared, so the new overview's filter takes effect immediately on the next render. The `lastNamespace` localStorage write happens on successful navigation (in the click handler), not on mount, so a deep-link visit to `/ns/foo` from a Slack URL also updates stickiness.

### Cross-namespace analysis

`/analysis` continues to compute cluster keys as `(namespace, model, settingsHash)`. No code change here — the existing logic already namespace-prefixes cluster keys. We just stop advertising "compare across namespaces" as a *primary* operational path; users assembling cluster keys from two different namespaces remains technically possible because `/analysis` is in the cross-namespace tier. This respects the project rule that runs are only comparable within `ns + model + settings`.

## Testing

### `tests/e2e/operator_ui/` changes

| Existing file | Action |
|---|---|
| `_pages.py` | Update `BASE_PATH` for every page object — `LaunchPage` → `/ns/<ns>/launch`, `RunPage` → `/ns/<ns>/run/<name>`, `ArchivePage` → `/ns/<ns>/archive`. Add a new `NamespacePicker` page object for `/`. |
| `test_dashboard.py` | Split into `test_namespace_picker.py` (cross-namespace `/`: tile rendering, search filter, sticky-redirect, mini-status chips) and `test_namespace_overview.py` (per-namespace `/ns/:ns`: stats hero, active strip, recent rows scoped to one namespace, empty-namespace state). |
| `test_history.py` | Rename to `test_namespace_archive.py`; update routes; drop cross-namespace grouping assertions (now in the picker). |
| `test_navigation.py` | Add coverage: sticky `/` → `/ns/<last>` redirect, the namespace switcher dropdown opening from the breadcrumb pill, "View all namespaces" returning to `/`, ⌘K results being namespace-aware in `/ns/:ns/...`. |
| `test_launch.py` | Add: auto-fill of `namespace: <ns>` on view mount, divergence-detection lock disables LAUNCH and chips the pill `bad`, corrected YAML re-enables LAUNCH. |
| `test_jobs.py`, `test_unified_jobs.py` | Delete. Coverage moves to `test_namespace_overview.py` and `test_namespace_archive.py`. |
| `test_job_detail.py` | Rename to `test_run_detail.py`; update routes. |
| `test_compare.py`, `test_leaderboard.py` | Minor route updates only — analysis tier shape is unchanged. |
| `test_robustness.py`, `test_xss.py` | Minor route updates. Add an explicit XSS test for namespace names containing HTML-special chars rendered into the picker tile and breadcrumb pill. |

### `tests/unit/ui/` changes

- `test_ns_prefs.py` (new): round-trip, missing-key default, quota-error swallow for `lib/ns-prefs.js`.
- Router tests grow cases for the new `/ns/:ns/...` patterns and the `/` sticky-redirect logic.

### Required green before merge

```bash
ruff format . && ruff check --fix .
uv run pytest tests/unit/ -n auto
uv run pytest -m component_integration -n auto
uv run pytest tests/e2e/operator_ui/ -n auto
make check-ergonomics && make check-ruff-baselined
```

## Rollout

Single PR. No feature flag. The operator UI has not shipped publicly, so there are no in-the-wild bookmarks/users to support — clean break is the right call.

## Documentation

- `docs/kubernetes/dashboard-ui.md` — rewrite of the navigation model.
- `docs/media/images/api-dashboard-v2.png` — re-shoot from the new picker, overwrite in place (per the project's dashboard-screenshot rule: latest in place, no dated variants).
- No CLAUDE.md / copilot-instructions / cursor changes — this is UI architecture, not a coding pattern.

## Open questions

None — all clarifying decisions captured above.
