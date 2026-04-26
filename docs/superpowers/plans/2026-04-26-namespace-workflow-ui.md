# Namespace-Workflow Operator UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the operator UI so namespace is the primary navigation context — picker at `/`, workspace at `/ns/<name>/...` — with no legacy URL shims.

**Architecture:** Two-tier hash-based router. Cross-namespace tier (`/`, `/analysis`, `/log`) for situational awareness; per-namespace tier (`/ns/:ns/...`) for every operational view. Frontend-only — no backend / operator API changes. Picker derives namespace tiles client-side by grouping the existing `api.listJobs()` response.

**Tech Stack:** Preact + htm + signals (existing), Chart.js (existing), Playwright (e2e), pytest (unit), `localStorage` for per-namespace UI prefs. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-26-namespace-workflow-ui-design.md`

**Working branch:** Current branch `ajc/k8s` (per user preference — no worktrees, commit on current branch).

**Conventions for this plan:**
- All file paths absolute from repo root.
- Each task ends with one commit. Commit messages follow Conventional Commits (`feat(operator-ui): ...`, `test(operator-ui): ...`, `refactor(operator-ui): ...`).
- Test commands: `uv run pytest -n auto <path>`. Single-folder per invocation. Use the e2e suite path explicitly when relevant.
- Pre-commit runs normally; do NOT pass `--no-verify` for this work (the `--no-verify` rule in memory is aiperf-rs only).
- Scratch helper for verifying tests fail before implementing: `uv run pytest -n auto <path> -x` is OK; `--lf` is OK to focus on the new test only.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `src/aiperf/operator/ui/lib/ns-prefs.js` | `localStorage` wrapper: `getNsPref`, `setNsPref`, `getLastNamespace`, `setLastNamespace`. |
| `src/aiperf/operator/ui/lib/yaml-namespace.js` | Lightweight `extractNamespaceField(yamlText): string \| null` for the launch divergence check. |
| `src/aiperf/operator/ui/views/namespace-picker.js` | The new `/` view: tiles per namespace observed in `jobs.value`. |
| `src/aiperf/operator/ui/views/namespace-overview.js` | The new `/ns/:ns` view: scoped Home (refactored from `views/home.js`). |
| `src/aiperf/operator/ui/components/namespace-switcher.js` | Compact dropdown rendered from the breadcrumb namespace pill. |
| `tests/unit/ui/test_ns_prefs.py` | Unit-test harness driving `lib/ns-prefs.js` via the existing JS test runner used by `test_ui_utils.py`. |
| `tests/e2e/operator_ui/test_namespace_picker.py` | E2E for `/` (tile rendering, search, sticky redirect, mini-status). |
| `tests/e2e/operator_ui/test_namespace_overview.py` | E2E for `/ns/:ns` (stats hero, active strip, recent rows, empty state). |

**Modified files:**

| Path | What changes |
|---|---|
| `src/aiperf/operator/ui/lib/router.js` | Add navigation helpers: `lastNamespaceRedirect()` is host-resolved, but `matchRoute` is unchanged (already pattern-driven). |
| `src/aiperf/operator/ui/app.js` | Replace `resolveView()` with the new route table. Add the `/` → `/ns/<last>` mount-effect. Delete every legacy branch. |
| `src/aiperf/operator/ui/components/top-rail.js` | Breadcrumb root becomes the namespace switcher pill. Hide LAUNCH at `/`. |
| `src/aiperf/operator/ui/components/command-palette.js` | Namespace-aware result ordering when current route is `/ns/:ns/...`. |
| `src/aiperf/operator/ui/views/launch.js` | Auto-fill `namespace: <ns>` from URL; debounced YAML parse; LAUNCH lock on divergence. |
| `src/aiperf/operator/ui/views/archive.js` | Drop cross-namespace grouping; scope to URL `:ns`. |
| `src/aiperf/operator/ui/views/run.js` | Adjust route prop wiring; drop legacy `/run/:ns/:name` in tests/links. |
| `src/aiperf/operator/ui/views/home.js` | DELETE — content split between `namespace-picker.js` and `namespace-overview.js`. |
| `src/aiperf/operator/ui/style.css` | Add styles for picker tiles, switcher dropdown, namespace-overview empty state, launch divergence chip. |
| `tests/e2e/operator_ui/_pages.py` | Update `BASE_PATH` and test-ids for every page object. Add `NamespacePicker` and rename others. |
| `tests/e2e/operator_ui/test_dashboard.py` | DELETE — replaced by `test_namespace_picker.py` + `test_namespace_overview.py`. |
| `tests/e2e/operator_ui/test_history.py` | RENAME to `test_namespace_archive.py`; update routes; drop cross-namespace grouping assertions. |
| `tests/e2e/operator_ui/test_jobs.py` | DELETE — coverage moves to overview/archive. |
| `tests/e2e/operator_ui/test_unified_jobs.py` | DELETE — same reason. |
| `tests/e2e/operator_ui/test_job_detail.py` | RENAME to `test_run_detail.py`; update routes. |
| `tests/e2e/operator_ui/test_launch.py` | Add divergence-lock cases. Update route. |
| `tests/e2e/operator_ui/test_navigation.py` | Add sticky-redirect, switcher dropdown, ⌘K namespace-aware cases. |
| `tests/e2e/operator_ui/test_compare.py` | Route updates only. |
| `tests/e2e/operator_ui/test_leaderboard.py` | Route updates only. |
| `tests/e2e/operator_ui/test_robustness.py` | Route updates. |
| `tests/e2e/operator_ui/test_xss.py` | Route updates + namespace-name XSS case. |
| `tests/unit/ui/test_aiperf_dashboard_ui.py` | Update assertions to match the picker shape. |
| `tests/unit/ui/test_ui_utils.py` | Add cases for the new router patterns and `extractNamespaceField`. |
| `docs/kubernetes/dashboard-ui.md` | Rewrite navigation section. |
| `docs/media/images/api-dashboard-v2.png` | Re-shoot from the new picker. |

---

## Task 1: Add `lib/ns-prefs.js` localStorage wrapper

**Files:**
- Create: `src/aiperf/operator/ui/lib/ns-prefs.js`
- Test: `tests/unit/ui/test_ns_prefs.py`

The two preceding sibling files (`lib/state.js`, `lib/format.js`) are pure helper modules. Match that style — small, no imports, ESM exports.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ui/test_ns_prefs.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``src/aiperf/operator/ui/lib/ns-prefs.js``.

Driven via the same Playwright-page-eval shim used in
``test_ui_utils.py`` — load the module from disk in a blank page,
call its exports, assert on the return values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.async_api import async_playwright

NS_PREFS_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "ns-prefs.js"
)


async def _eval_with_module(script: str) -> object:
    """Run ``script`` in a blank page after importing ns-prefs.js as ``M``."""
    src = NS_PREFS_PATH.read_text()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context()
        page = await ctx.new_page()
        await page.goto("about:blank")
        # Inline the module on the page and re-export under window.M.
        await page.add_script_tag(content=f"const __m = (() => {{ const exports = {{}}; {src.replace('export ', 'exports.')} ; return exports; }})(); window.M = __m;")
        result = await page.evaluate(script)
        await browser.close()
        return result


@pytest.mark.asyncio
async def test_get_ns_pref_missing_returns_default():
    out = await _eval_with_module("M.getNsPref('foo', 'pinnedRunNames', ['fallback'])")
    assert out == ["fallback"]


@pytest.mark.asyncio
async def test_set_then_get_round_trip():
    out = await _eval_with_module(
        "(() => { M.setNsPref('foo', 'pinnedRunNames', ['a','b']); "
        "return M.getNsPref('foo', 'pinnedRunNames', []); })()"
    )
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_last_namespace_round_trip():
    out = await _eval_with_module(
        "(() => { M.setLastNamespace('team-llama'); return M.getLastNamespace(); })()"
    )
    assert out == "team-llama"


@pytest.mark.asyncio
async def test_get_last_namespace_missing_returns_null():
    out = await _eval_with_module("M.getLastNamespace()")
    assert out is None


@pytest.mark.asyncio
async def test_set_pref_quota_error_swallowed():
    """Force a throw on setItem; the helper must not propagate."""
    out = await _eval_with_module(
        "(() => { const orig = Storage.prototype.setItem; "
        "Storage.prototype.setItem = () => { throw new Error('quota'); }; "
        "try { M.setNsPref('foo', 'k', 'v'); return 'ok'; } "
        "finally { Storage.prototype.setItem = orig; } })()"
    )
    assert out == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest -n auto tests/unit/ui/test_ns_prefs.py
```
Expected: FAIL — module file does not yet exist (FileNotFoundError on `read_text`).

- [ ] **Step 3: Implement `lib/ns-prefs.js`**

Create `src/aiperf/operator/ui/lib/ns-prefs.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-namespace UI preferences persisted in ``localStorage``.
 *
 * Keys:
 *   ``aiperf.ui.lastNamespace``                    sticky last-used namespace
 *   ``aiperf.ui.ns.<ns>.pinnedRunNames``           pinned runs surfaced on overview
 *   ``aiperf.ui.ns.<ns>.lastLaunchTemplateId``     auto-loaded launch template
 *   ``aiperf.ui.ns.<ns>.overviewMetricKey``        chart series key on overview
 *
 * Best-effort: missing key returns the supplied default; quota / disabled
 * storage errors are swallowed so the UI never crashes on persistence.
 */

const LAST_NS_KEY = 'aiperf.ui.lastNamespace';

function nsKey(ns, key) {
  return `aiperf.ui.ns.${ns}.${key}`;
}

export function getNsPref(ns, key, fallback) {
  try {
    const raw = window.localStorage.getItem(nsKey(ns, key));
    if (raw == null) return fallback;
    return JSON.parse(raw);
  } catch (_e) {
    return fallback;
  }
}

export function setNsPref(ns, key, value) {
  try {
    window.localStorage.setItem(nsKey(ns, key), JSON.stringify(value));
  } catch (_e) {
    // quota / disabled storage / SecurityError — drop on the floor
  }
}

export function getLastNamespace() {
  try {
    const raw = window.localStorage.getItem(LAST_NS_KEY);
    return raw == null ? null : raw;
  } catch (_e) {
    return null;
  }
}

export function setLastNamespace(ns) {
  try {
    if (ns) window.localStorage.setItem(LAST_NS_KEY, ns);
  } catch (_e) { /* swallow */ }
}
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest -n auto tests/unit/ui/test_ns_prefs.py
```
Expected: PASS — five tests.

- [ ] **Step 5: Commit**

```
git add src/aiperf/operator/ui/lib/ns-prefs.js tests/unit/ui/test_ns_prefs.py
git commit -s -m "feat(operator-ui): add ns-prefs localStorage wrapper for per-namespace UI prefs"
```

---

## Task 2: Add `lib/yaml-namespace.js` extractor

**Files:**
- Create: `src/aiperf/operator/ui/lib/yaml-namespace.js`
- Modify: `tests/unit/ui/test_ui_utils.py` — add a new test class for it

The launch divergence check needs `extractNamespaceField(yamlText): string | null`. We do NOT want a full YAML parser dependency for this single field — a regex over top-level lines is enough. Multi-document, indented, and quoted values must be handled.

- [ ] **Step 1: Add failing tests to `test_ui_utils.py`**

Append to `tests/unit/ui/test_ui_utils.py`:

```python
import pytest
from playwright.async_api import async_playwright

YAML_NS_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src" / "aiperf" / "operator" / "ui" / "lib" / "yaml-namespace.js"
)


async def _extract(yaml_text: str):
    src = YAML_NS_PATH.read_text()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await (await browser.new_context()).new_page()
        await page.goto("about:blank")
        await page.add_script_tag(content=f"const __m = (() => {{ const exports = {{}}; {src.replace('export ', 'exports.')} ; return exports; }})(); window.M = __m;")
        out = await page.evaluate("(y) => M.extractNamespaceField(y)", yaml_text)
        await browser.close()
        return out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "yaml_text,expected",
    [
        pytest.param("namespace: foo\n", "foo", id="bare-top-level"),
        pytest.param("namespace: 'team-llama'\n", "team-llama", id="single-quoted"),
        pytest.param('namespace: "team-llama"\n', "team-llama", id="double-quoted"),
        pytest.param("apiVersion: v1\nnamespace: bar\nkind: AIPerfJob\n", "bar", id="not-first-key"),
        pytest.param("metadata:\n  namespace: indented\n", None, id="indented-not-top-level"),
        pytest.param("# namespace: commented\n", None, id="comment-ignored"),
        pytest.param("", None, id="empty"),
        pytest.param("namespace:\n", None, id="empty-value"),
        pytest.param("namespace: foo  # trailing comment\n", "foo", id="trailing-comment"),
    ],
)
async def test_extract_namespace_field(yaml_text, expected):
    assert await _extract(yaml_text) == expected
```

- [ ] **Step 2: Run to verify fail**

```
uv run pytest -n auto tests/unit/ui/test_ui_utils.py::test_extract_namespace_field
```
Expected: FAIL — file not found.

- [ ] **Step 3: Implement `lib/yaml-namespace.js`**

Create `src/aiperf/operator/ui/lib/yaml-namespace.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Extract the top-level ``namespace:`` field from an AIPerfJob YAML body.
 *
 * Used by the launch view's divergence check — full YAML parsing is
 * overkill (and would pull a parser into the bundle) for one field that
 * is, by spec, top-level on the AIPerfJob CR. Indented ``namespace:``
 * (e.g. ``metadata.namespace``) is intentionally ignored: the AIPerfJob
 * shape places ``namespace`` at the top of the spec body the launch
 * editor produces.
 *
 * @param {string} yamlText raw editor contents
 * @returns {string|null} the unquoted value, or null if absent / empty
 */
export function extractNamespaceField(yamlText) {
  if (!yamlText) return null;
  const lines = yamlText.split('\n');
  for (const raw of lines) {
    // Strip line-start whitespace check: if non-zero indent, it's not top-level.
    if (raw.length === 0) continue;
    if (raw[0] === ' ' || raw[0] === '\t') continue;
    if (raw[0] === '#') continue;
    const m = /^namespace:\s*(.*)$/.exec(raw);
    if (!m) continue;
    let v = m[1].trim();
    if (!v) return null;
    // Strip trailing inline comment (only when not inside quotes).
    if (v[0] !== '"' && v[0] !== "'") {
      const hashIdx = v.indexOf('#');
      if (hashIdx >= 0) v = v.slice(0, hashIdx).trim();
    }
    if (!v) return null;
    if ((v[0] === '"' && v[v.length - 1] === '"') ||
        (v[0] === "'" && v[v.length - 1] === "'")) {
      v = v.slice(1, -1);
    }
    return v || null;
  }
  return null;
}
```

- [ ] **Step 4: Run to verify pass**

```
uv run pytest -n auto tests/unit/ui/test_ui_utils.py::test_extract_namespace_field
```
Expected: PASS — nine parametrized cases.

- [ ] **Step 5: Commit**

```
git add src/aiperf/operator/ui/lib/yaml-namespace.js tests/unit/ui/test_ui_utils.py
git commit -s -m "feat(operator-ui): add yaml-namespace extractor for launch divergence check"
```

---

## Task 3: Build `<NamespacePicker>` view (without wiring into the router yet)

**Files:**
- Create: `src/aiperf/operator/ui/views/namespace-picker.js`
- Modify: `src/aiperf/operator/ui/style.css` — add `.np-*` rule block
- Test: `tests/e2e/operator_ui/test_namespace_picker.py`

We build the component first, then wire it into the router in Task 6 (after the overview is also ready, so the route swap is one atomic step). Until then, `app.js` does not yet route to it; the e2e test in this task asserts only on the rendered DOM by directly visiting `#/__npicker_preview` (a temporary inline-mounted preview route gated to debug builds is *not* introduced — instead the e2e test imports and renders the component standalone via Playwright `page.evaluate` against a stub HTML harness).

**Simpler path:** wire into the router in this task too. The legacy `/` Home stays mounted *via fall-through* until Task 6, so picker is visible at `/` *now*. The intermediate state is acceptable because there are no shipped users.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/e2e/operator_ui/test_namespace_picker.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E coverage for the cross-namespace picker mounted at ``/``."""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import NamespacePickerPage


@pytest.mark.asyncio
async def test_picker_renders_one_tile_per_namespace(page, base_url, seeded_results_dir):
    p = NamespacePickerPage(page=page, base_url=base_url)
    await p.goto()
    # Golden fixture has runs in two namespaces: 'default' and 'bench-perf'.
    await expect(p.tile("default")).to_be_visible()
    await expect(p.tile("bench-perf")).to_be_visible()


@pytest.mark.asyncio
async def test_picker_tile_shows_phase_chips(page, base_url, seeded_results_dir):
    p = NamespacePickerPage(page=page, base_url=base_url)
    await p.goto()
    tile = p.tile("default")
    await expect(tile.locator(".np-chip-running")).to_be_visible()


@pytest.mark.asyncio
async def test_picker_search_filters_tiles(page, base_url, seeded_results_dir):
    p = NamespacePickerPage(page=page, base_url=base_url)
    await p.goto()
    await p.search().fill("bench")
    await expect(p.tile("bench-perf")).to_be_visible()
    await expect(p.tile("default")).not_to_be_visible()


@pytest.mark.asyncio
async def test_clicking_tile_navigates_to_namespace_overview(page, base_url, seeded_results_dir):
    p = NamespacePickerPage(page=page, base_url=base_url)
    await p.goto()
    await p.tile("default").click()
    await expect(page).to_have_url(lambda url: url.endswith("#/ns/default"))
```

- [ ] **Step 2: Add the `NamespacePickerPage` page object**

Edit `tests/e2e/operator_ui/_pages.py` — append the new class (full update of all page objects happens in Task 8; this is just the additive piece):

```python
class NamespacePickerPage(BasePage):
    """The ``/`` view — cross-namespace picker with one tile per namespace."""

    async def goto(self) -> None:
        await self._goto("/")
        await expect(self.page.get_by_test_id("page-namespace-picker")).to_be_visible()

    def tile(self, namespace: str) -> Locator:
        return self.page.get_by_test_id(f"np-tile-{namespace}")

    def search(self) -> Locator:
        return self.page.get_by_test_id("np-search")
```

- [ ] **Step 3: Run e2e to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_picker.py
```
Expected: FAIL — `page-namespace-picker` test-id not present (current `/` is the legacy Home).

- [ ] **Step 4: Implement `views/namespace-picker.js`**

Create `src/aiperf/operator/ui/views/namespace-picker.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE PICKER — the cross-namespace landing surface mounted at ``/``.
 *
 * One tile per namespace observed in ``jobs.value`` (no separate API
 * call — we group the existing job list by ``j.namespace``). Each tile
 * surfaces "is anything broken / live here?" at a glance: phase chips
 * for live counts, last-activity timestamp, left-edge state tint.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { setLastNamespace } from '../lib/ns-prefs.js';
import { fmtRelative } from '../lib/format.js';

const FAILED_RECENT_WINDOW_MS = 24 * 60 * 60 * 1000;

function bucketForJob(j) {
  const p = (j.phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function aggregate(nsJobs) {
  let running = 0, failedRecent = 0, completed = 0, total = nsJobs.length;
  let lastActivity = 0;
  const now = Date.now();
  for (const j of nsJobs) {
    const b = bucketForJob(j);
    if (b === 'live') running += 1;
    if (b === 'passed') completed += 1;
    const ts = (j.lastUpdate ?? j.startTime) ? Date.parse(j.lastUpdate ?? j.startTime) : 0;
    if (ts > lastActivity) lastActivity = ts;
    if (b === 'fault' && (now - ts) <= FAILED_RECENT_WINDOW_MS) failedRecent += 1;
  }
  let tint = 'quiet';
  if (running > 0) tint = 'live';
  else if (failedRecent > 0) tint = 'fault';
  return { running, failedRecent, completed, total, lastActivity, tint };
}

function NamespaceTile({ name, agg, onPick }) {
  return html`
    <div
      class=${'np-tile np-tile--' + agg.tint}
      data-testid=${'np-tile-' + name}
      onclick=${() => onPick(name)}
    >
      <div class="np-tile-name">${name}</div>
      <div class="np-tile-summary">${agg.running} active · ${agg.total} total</div>
      <div class="np-tile-chips">
        ${agg.running > 0 && html`<span class="np-chip np-chip-running">Running ${agg.running}</span>`}
        ${agg.failedRecent > 0 && html`<span class="np-chip np-chip-failed">Failed ${agg.failedRecent}</span>`}
        ${agg.completed > 0 && html`<span class="np-chip np-chip-completed">Completed ${agg.completed}</span>`}
      </div>
      <div class="np-tile-time">${agg.lastActivity ? fmtRelative(agg.lastActivity) : '—'}</div>
    </div>
  `;
}

export function NamespacePicker() {
  const [query, setQuery] = useState('');
  const list = jobs.value ?? [];

  const tiles = useMemo(() => {
    const groups = new Map();
    for (const j of list) {
      const ns = j.namespace || 'default';
      if (!groups.has(ns)) groups.set(ns, []);
      groups.get(ns).push(j);
    }
    const out = [];
    for (const [name, nsJobs] of groups) {
      out.push({ name, agg: aggregate(nsJobs) });
    }
    out.sort((a, b) => b.agg.lastActivity - a.agg.lastActivity);
    return out;
  }, [list]);

  const filtered = query
    ? tiles.filter(t => t.name.toLowerCase().includes(query.toLowerCase()))
    : tiles;

  function pick(name) {
    setLastNamespace(name);
    navigate('/ns/' + encodeURIComponent(name));
  }

  return html`
    <div class="page-namespace-picker" data-testid="page-namespace-picker">
      <div class="np-header">
        <h1 class="np-title">Pick a namespace</h1>
        <input
          class="np-search"
          data-testid="np-search"
          placeholder="filter namespaces…"
          value=${query}
          oninput=${(e) => setQuery(e.target.value)}
        />
      </div>
      ${tiles.length === 0 && html`
        <div class="np-empty" data-testid="np-empty">
          <p>No AIPerfJob runs visible in any namespace yet.</p>
          <p>If you expected to see runs here, check the operator's RBAC against your kubeconfig context.</p>
        </div>
      `}
      <div class="np-grid">
        ${filtered.map(t => html`
          <${NamespaceTile} key=${t.name} name=${t.name} agg=${t.agg} onPick=${pick} />
        `)}
      </div>
    </div>
  `;
}
```

- [ ] **Step 5: Add CSS rules to `style.css`**

Append to `src/aiperf/operator/ui/style.css`:

```css
/* ---- Namespace Picker (page-namespace-picker) ----------------------- */
.page-namespace-picker { padding: 24px; }
.np-header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.np-title { font-size: 22px; font-weight: 600; margin: 0; }
.np-search {
  flex: 1; max-width: 360px; padding: 8px 12px;
  background: var(--surface-2); border: 1px solid var(--border-1);
  border-radius: 6px; color: var(--text-1);
}
.np-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.np-tile {
  background: var(--surface-1);
  border: 1px solid var(--border-1);
  border-left-width: 4px;
  border-radius: 8px;
  padding: 14px 16px;
  cursor: pointer;
  transition: background 120ms;
}
.np-tile:hover { background: var(--surface-2); }
.np-tile--live   { border-left-color: var(--accent-info); }
.np-tile--fault  { border-left-color: var(--accent-bad);  }
.np-tile--quiet  { border-left-color: var(--border-1);    }
.np-tile-name { font-weight: 600; font-size: 16px; margin-bottom: 4px; }
.np-tile-summary { color: var(--text-2); font-size: 13px; margin-bottom: 8px; }
.np-tile-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.np-chip { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.np-chip-running   { background: var(--chip-info-bg);   color: var(--chip-info-fg); }
.np-chip-failed    { background: var(--chip-bad-bg);    color: var(--chip-bad-fg); }
.np-chip-completed { background: var(--chip-good-bg);   color: var(--chip-good-fg); }
.np-tile-time { color: var(--text-3); font-size: 12px; }
.np-empty { padding: 40px; text-align: center; color: var(--text-2); }
```

- [ ] **Step 6: Wire into `app.js` `resolveView` (additive — Home still mounted as fallback)**

Edit `src/aiperf/operator/ui/app.js`. At the top of the file, add the import:

```javascript
import { NamespacePicker } from './views/namespace-picker.js';
```

Replace the `return { kind: 'home' };` line in `resolveView` with:

```javascript
  if (currentRoute === '/') return { kind: 'namespace-picker' };
  return { kind: 'home' };
```

In the `App` rendering switch, add a new case for `'namespace-picker'`:

```javascript
    case 'namespace-picker':
      return html`<${NamespacePicker} />`;
```

(Find the existing `switch (resolved.kind)` block in `App()` and insert the new branch.)

- [ ] **Step 7: Run e2e test to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_picker.py
```
Expected: PASS — four tests.

- [ ] **Step 8: Commit**

```
git add src/aiperf/operator/ui/views/namespace-picker.js src/aiperf/operator/ui/style.css src/aiperf/operator/ui/app.js tests/e2e/operator_ui/test_namespace_picker.py tests/e2e/operator_ui/_pages.py
git commit -s -m "feat(operator-ui): namespace picker at root, replaces global home"
```

---

## Task 4: Build `<NamespaceOverview>` view at `/ns/:ns`

**Files:**
- Create: `src/aiperf/operator/ui/views/namespace-overview.js` (refactor from `views/home.js`)
- Modify: `src/aiperf/operator/ui/app.js` — add the route
- Modify: `src/aiperf/operator/ui/style.css` — add `.no-empty` empty-state rules
- Test: `tests/e2e/operator_ui/test_namespace_overview.py`

The overview reuses everything from `home.js` (StatTile, ActiveCard, recent-runs table) — but filters jobs to the URL `:ns` and shows the empty-state when zero jobs match.

- [ ] **Step 1: Write the failing e2e test**

Create `tests/e2e/operator_ui/test_namespace_overview.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E coverage for the per-namespace overview at ``/ns/:ns``."""

from __future__ import annotations

import pytest
from playwright.async_api import expect

from ._pages import NamespaceOverviewPage


@pytest.mark.asyncio
async def test_overview_renders_only_jobs_in_namespace(page, base_url, seeded_results_dir):
    p = NamespaceOverviewPage(page=page, base_url=base_url, namespace="default")
    await p.goto()
    # Golden fixture has at least one run in 'default' and one in 'bench-perf'.
    rows = page.locator("[data-testid^='no-row-']")
    count = await rows.count()
    assert count > 0
    for i in range(count):
        testid = await rows.nth(i).get_attribute("data-testid")
        assert testid.startswith("no-row-default-")


@pytest.mark.asyncio
async def test_overview_empty_namespace_renders_launch_cta(page, base_url, seeded_results_dir):
    p = NamespaceOverviewPage(page=page, base_url=base_url, namespace="empty-ns")
    await p.goto()
    await expect(page.get_by_test_id("no-empty")).to_be_visible()
    await expect(page.get_by_test_id("no-empty-launch-cta")).to_be_visible()


@pytest.mark.asyncio
async def test_overview_empty_launch_cta_navigates(page, base_url, seeded_results_dir):
    p = NamespaceOverviewPage(page=page, base_url=base_url, namespace="empty-ns")
    await p.goto()
    await page.get_by_test_id("no-empty-launch-cta").click()
    await expect(page).to_have_url(lambda u: u.endswith("#/ns/empty-ns/launch"))
```

- [ ] **Step 2: Add the page object**

Append to `tests/e2e/operator_ui/_pages.py`:

```python
@dataclass
class NamespaceOverviewPage(BasePage):
    namespace: str = "default"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}")
        await expect(self.page.get_by_test_id("page-namespace-overview")).to_be_visible()

    def stats(self) -> Locator:
        return self.page.get_by_test_id("no-stats")

    def row(self, name: str) -> Locator:
        return self.page.get_by_test_id(f"no-row-{self.namespace}-{name}")
```

- [ ] **Step 3: Run e2e to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_overview.py
```
Expected: FAIL — route not yet registered.

- [ ] **Step 4: Implement `views/namespace-overview.js`**

This view is structurally a copy of `views/home.js` adjusted for namespace scoping. Read `src/aiperf/operator/ui/views/home.js` first to copy the StatTile, ActiveCard, and recent-runs row markup verbatim. Then create the new file:

Create `src/aiperf/operator/ui/views/namespace-overview.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE OVERVIEW — per-namespace dashboard mounted at ``/ns/:ns``.
 *
 * Same shape as the prior global Home but filtered to ``j.namespace === ns``.
 * Renders an empty-state with a single "Launch in <ns>" CTA when the
 * namespace has zero current and zero historical jobs.
 */

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration, fmtInt, fmtNumber } from '../lib/format.js';

// --- helpers (copied from home.js verbatim — keep in sync if home.js helpers change) ---
function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function titleCase(s) {
  if (!s) return '—';
  const lower = String(s).toLowerCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function modelShort(model) {
  if (!model) return '';
  return String(model).split('/').pop();
}

function progressPct(j) {
  if (j.progressPct != null) return Math.max(0, Math.min(100, Number(j.progressPct)));
  if (j.requestsCompleted != null && j.requestsTotal) {
    return Math.max(0, Math.min(100, (j.requestsCompleted / j.requestsTotal) * 100));
  }
  return null;
}

function StatTile({ label, value, sub, mod }) {
  return html`
    <div class=${'no-stat no-stat--' + mod}>
      <div class="no-stat-label">${label}</div>
      <div class="no-stat-val">${value}</div>
      ${sub && html`<div class="no-stat-sub">${sub}</div>`}
    </div>
  `;
}

function ActiveCard({ job, ns }) {
  const pct = progressPct(job);
  const elapsed = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/ns/${encodeURIComponent(ns)}/run/${encodeURIComponent(job.name)}`;
  return html`
    <div
      class="no-active-card"
      data-testid=${'no-active-' + ns + '-' + job.name}
      onclick=${() => navigate(href)}
    >
      <div class="no-active-card-head">
        <div>
          <div class="no-active-card-name">${job.name}</div>
          <div class="no-active-card-ns">${modelShort(job.model) || 'no model'}</div>
        </div>
        <span class="chip chip--info">${titleCase(job.phase) || 'Running'}</span>
      </div>
      ${pct != null && html`
        <div class="no-active-card-track">
          <div class="no-active-card-fill" style=${'width:' + pct + '%'}></div>
        </div>
      `}
      <div class="no-active-card-stats">
        <div><div class="no-active-card-stat-lab">Throughput</div><div class="no-active-card-stat-val">${job.throughputRps != null ? fmtNumber(job.throughputRps, 1) : '—'}</div></div>
        <div><div class="no-active-card-stat-lab">Latency p99</div><div class="no-active-card-stat-val">${job.latencyP99Ms != null ? fmtInt(job.latencyP99Ms) + ' ms' : '—'}</div></div>
        <div><div class="no-active-card-stat-lab">Elapsed</div><div class="no-active-card-stat-val">${elapsed != null ? fmtDuration(elapsed) : '—'}</div></div>
      </div>
    </div>
  `;
}

function RecentRow({ job, ns }) {
  const href = `/ns/${encodeURIComponent(ns)}/run/${encodeURIComponent(job.name)}`;
  return html`
    <tr
      class=${'no-row no-row--' + phaseBucket(job.phase)}
      data-testid=${'no-row-' + ns + '-' + job.name}
      onclick=${() => navigate(href)}
    >
      <td class="no-row-name">${job.name}</td>
      <td>${modelShort(job.model)}</td>
      <td>${titleCase(job.phase)}</td>
      <td>${job.throughputRps != null ? fmtNumber(job.throughputRps, 1) : '—'}</td>
      <td>${job.latencyP99Ms != null ? fmtInt(job.latencyP99Ms) + ' ms' : '—'}</td>
    </tr>
  `;
}

export function NamespaceOverview({ ns }) {
  const all = jobs.value ?? [];
  const list = useMemo(() => all.filter(j => (j.namespace || 'default') === ns), [all, ns]);

  const counts = useMemo(() => {
    const c = { live: 0, passed: 0, fault: 0, total: list.length };
    for (const j of list) c[phaseBucket(j.phase)] = (c[phaseBucket(j.phase)] || 0) + 1;
    return c;
  }, [list]);

  if (list.length === 0) {
    return html`
      <div class="page-namespace-overview" data-testid="page-namespace-overview">
        <div class="no-empty" data-testid="no-empty">
          <h1 class="no-empty-title">No runs yet in <code>${ns}</code></h1>
          <p class="no-empty-sub">Launch your first benchmark in this namespace.</p>
          <button
            class="btn btn--primary no-empty-cta"
            data-testid="no-empty-launch-cta"
            onclick=${() => navigate('/ns/' + encodeURIComponent(ns) + '/launch')}
          >Launch in ${ns}</button>
        </div>
      </div>
    `;
  }

  const active = list.filter(j => phaseBucket(j.phase) === 'live');
  const recent = [...list]
    .sort((a, b) => Date.parse(b.lastUpdate ?? b.startTime ?? 0) - Date.parse(a.lastUpdate ?? a.startTime ?? 0))
    .slice(0, 25);

  const gpus = clusterInfo.value?.gpus ?? null;

  return html`
    <div class="page-namespace-overview" data-testid="page-namespace-overview">
      <div class="no-stats" data-testid="no-stats">
        <${StatTile} label="Running" value=${fmtInt(counts.live)} mod="live" />
        <${StatTile} label="Passed"  value=${fmtInt(counts.passed)} mod="passed" />
        <${StatTile} label="Failed"  value=${fmtInt(counts.fault)} mod="fault" />
        <${StatTile} label="Total"   value=${fmtInt(counts.total)} mod="total" />
        <${StatTile} label="GPUs"    value=${gpus != null ? fmtInt(gpus) : '—'} mod="gpus" />
      </div>
      ${active.length > 0 && html`
        <div class="no-active">
          ${active.map(j => html`<${ActiveCard} key=${j.name} job=${j} ns=${ns} />`)}
        </div>
      `}
      <table class="no-recent">
        <thead><tr><th>Name</th><th>Model</th><th>Phase</th><th>RPS</th><th>p99</th></tr></thead>
        <tbody>${recent.map(j => html`<${RecentRow} key=${j.name} job=${j} ns=${ns} />`)}</tbody>
      </table>
    </div>
  `;
}
```

- [ ] **Step 5: Add CSS rules**

Append to `src/aiperf/operator/ui/style.css`:

```css
/* ---- Namespace Overview (page-namespace-overview) ------------------- */
.page-namespace-overview { padding: 24px; }
.no-stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }
.no-stat  { background: var(--surface-1); border: 1px solid var(--border-1); border-radius: 8px; padding: 14px 16px; }
.no-stat-label { color: var(--text-2); font-size: 12px; }
.no-stat-val   { font-size: 28px; font-weight: 600; }
.no-stat-sub   { color: var(--text-3); font-size: 12px; }
.no-active     { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; margin-bottom: 24px; }
.no-active-card { background: var(--surface-1); border: 1px solid var(--border-1); border-radius: 8px; padding: 14px 16px; cursor: pointer; }
.no-active-card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.no-active-card-name { font-weight: 600; }
.no-active-card-ns   { color: var(--text-2); font-size: 12px; }
.no-active-card-track { height: 4px; background: var(--surface-2); border-radius: 2px; overflow: hidden; margin: 8px 0; }
.no-active-card-fill  { height: 100%; background: var(--accent-info); }
.no-active-card-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.no-active-card-stat-lab { color: var(--text-2); font-size: 11px; }
.no-active-card-stat-val { font-weight: 600; }
.no-recent { width: 100%; border-collapse: collapse; }
.no-recent th, .no-recent td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border-1); }
.no-row { cursor: pointer; }
.no-row:hover { background: var(--surface-2); }
.no-row--live   { border-left: 3px solid var(--accent-info); }
.no-row--passed { border-left: 3px solid var(--accent-good); }
.no-row--fault  { border-left: 3px solid var(--accent-bad); }
.no-empty { text-align: center; padding: 80px 24px; }
.no-empty-title { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
.no-empty-sub   { color: var(--text-2); margin-bottom: 24px; }
```

- [ ] **Step 6: Wire route into `app.js`**

Edit `src/aiperf/operator/ui/app.js`. Add the import:

```javascript
import { NamespaceOverview } from './views/namespace-overview.js';
```

In `resolveView`, add the match before any existing `/ns/:ns/run/:name` matches:

```javascript
  const nsOverviewMatch = matchRoute('/ns/:ns', currentRoute);
  if (nsOverviewMatch) return { kind: 'namespace-overview', params: nsOverviewMatch };
```

In the `App` switch, add:

```javascript
    case 'namespace-overview':
      return html`<${NamespaceOverview} ns=${resolved.params.ns} />`;
```

- [ ] **Step 7: Run e2e to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_overview.py
```
Expected: PASS — three tests.

- [ ] **Step 8: Commit**

```
git add src/aiperf/operator/ui/views/namespace-overview.js src/aiperf/operator/ui/style.css src/aiperf/operator/ui/app.js tests/e2e/operator_ui/test_namespace_overview.py tests/e2e/operator_ui/_pages.py
git commit -s -m "feat(operator-ui): namespace overview at /ns/:ns with empty-state launch CTA"
```

---

## Task 5: Build `<NamespaceSwitcher>` and rewire breadcrumb

**Files:**
- Create: `src/aiperf/operator/ui/components/namespace-switcher.js`
- Modify: `src/aiperf/operator/ui/components/top-rail.js` — breadcrumb root becomes the switcher pill
- Modify: `src/aiperf/operator/ui/style.css` — add `.ns-switcher-*` rules
- Test: `tests/e2e/operator_ui/test_navigation.py` — append switcher cases

- [ ] **Step 1: Add failing tests to `test_navigation.py`**

Append to `tests/e2e/operator_ui/test_navigation.py`:

```python
@pytest.mark.asyncio
async def test_breadcrumb_namespace_pill_opens_switcher(page, base_url, seeded_results_dir):
    await page.goto(base_url + "/#/ns/default")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.get_by_test_id("ns-switcher-pill").click()
    await expect(page.get_by_test_id("ns-switcher-dropdown")).to_be_visible()


@pytest.mark.asyncio
async def test_switcher_navigates_to_other_namespace(page, base_url, seeded_results_dir):
    await page.goto(base_url + "/#/ns/default")
    await expect(page.get_by_test_id("page-namespace-overview")).to_be_visible()
    await page.get_by_test_id("ns-switcher-pill").click()
    await page.get_by_test_id("ns-switcher-item-bench-perf").click()
    await expect(page).to_have_url(lambda u: u.endswith("#/ns/bench-perf"))


@pytest.mark.asyncio
async def test_switcher_view_all_returns_to_picker(page, base_url, seeded_results_dir):
    await page.goto(base_url + "/#/ns/default")
    await page.get_by_test_id("ns-switcher-pill").click()
    await page.get_by_test_id("ns-switcher-view-all").click()
    await expect(page).to_have_url(lambda u: u.endswith("#/") or u.endswith("#"))
```

- [ ] **Step 2: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py::test_breadcrumb_namespace_pill_opens_switcher tests/e2e/operator_ui/test_navigation.py::test_switcher_navigates_to_other_namespace tests/e2e/operator_ui/test_navigation.py::test_switcher_view_all_returns_to_picker
```
Expected: FAIL — `ns-switcher-pill` test-id absent.

- [ ] **Step 3: Implement `<NamespaceSwitcher>`**

Create `src/aiperf/operator/ui/components/namespace-switcher.js`:

```javascript
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE SWITCHER — compact dropdown rendered from the breadcrumb pill.
 *
 * Same data source as the picker (group ``jobs.value`` by namespace),
 * but compact: name + a single phase-chip row + last-activity. Selecting
 * an item navigates to the namespace overview (``/ns/<chosen>``); the
 * "View all namespaces" footer item navigates to ``/``.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { setLastNamespace } from '../lib/ns-prefs.js';

export function NamespaceSwitcher({ currentNs, onClose }) {
  const [query, setQuery] = useState('');
  const list = jobs.value ?? [];

  const items = useMemo(() => {
    const groups = new Map();
    for (const j of list) {
      const ns = j.namespace || 'default';
      if (!groups.has(ns)) groups.set(ns, { name: ns, running: 0, total: 0 });
      const g = groups.get(ns);
      g.total += 1;
      const p = (j.phase ?? '').toLowerCase();
      if (p === 'running' || p === 'initializing' || p === 'pending') g.running += 1;
    }
    return Array.from(groups.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [list]);

  const filtered = query
    ? items.filter(i => i.name.toLowerCase().includes(query.toLowerCase()))
    : items;

  function pick(name) {
    setLastNamespace(name);
    onClose?.();
    navigate('/ns/' + encodeURIComponent(name));
  }

  return html`
    <div class="ns-switcher-dropdown" data-testid="ns-switcher-dropdown">
      <input
        class="ns-switcher-search"
        data-testid="ns-switcher-search"
        autofocus
        placeholder="filter namespaces…"
        value=${query}
        oninput=${(e) => setQuery(e.target.value)}
      />
      <div class="ns-switcher-list">
        ${filtered.map(i => html`
          <button
            class=${'ns-switcher-item' + (i.name === currentNs ? ' ns-switcher-item--current' : '')}
            data-testid=${'ns-switcher-item-' + i.name}
            onclick=${() => pick(i.name)}
          >
            <span class="ns-switcher-name">${i.name}</span>
            <span class="ns-switcher-meta">${i.running} / ${i.total}</span>
          </button>
        `)}
      </div>
      <button
        class="ns-switcher-view-all"
        data-testid="ns-switcher-view-all"
        onclick=${() => { onClose?.(); navigate('/'); }}
      >View all namespaces →</button>
    </div>
  `;
}
```

- [ ] **Step 4: Update `top-rail.js`**

Read `src/aiperf/operator/ui/components/top-rail.js` first to see the existing breadcrumb shape, then modify it. The pill replaces the existing first breadcrumb segment when on a `/ns/:ns/...` route.

Key changes inside `top-rail.js`:
1. Import `NamespaceSwitcher` and `useState`.
2. Add a `[switcherOpen, setSwitcherOpen]` hook in the rail component.
3. When on `/` — render the existing callsign only (no breadcrumb, no LAUNCH CTA).
4. When on `/ns/:ns/...` — render the breadcrumb starting with a clickable pill `data-testid="ns-switcher-pill"` that toggles `switcherOpen`. When open, render `<NamespaceSwitcher currentNs={ns} onClose={() => setSwitcherOpen(false)} />` underneath.

Concrete diff (apply at the start of the breadcrumb-rendering block in `TopRail`; preserve the rest of the rail):

```javascript
import { useState } from 'preact/hooks';
import { NamespaceSwitcher } from './namespace-switcher.js';
import { matchRoute } from '../lib/router.js';
// ... existing imports unchanged ...

export function TopRail({ currentRoute }) {
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const nsMatch = matchRoute('/ns/:ns', currentRoute)
    ?? matchRoute('/ns/:ns/launch', currentRoute)
    ?? matchRoute('/ns/:ns/archive', currentRoute)
    ?? matchRoute('/ns/:ns/run/:name', currentRoute)
    ?? matchRoute('/ns/:ns/run/:name/runs/:epoch', currentRoute);
  const ns = nsMatch?.ns ?? null;
  const showLaunchCta = ns != null;

  return html`
    <header class="top-rail">
      <div class="top-rail-left">
        <span class="top-rail-callsign">AIPERF</span>
        ${ns && html`
          <span class="top-rail-sep">›</span>
          <button
            class="ns-switcher-pill"
            data-testid="ns-switcher-pill"
            onclick=${() => setSwitcherOpen(o => !o)}
          >${ns} ▾</button>
          ${switcherOpen && html`
            <${NamespaceSwitcher} currentNs=${ns} onClose=${() => setSwitcherOpen(false)} />
          `}
        `}
        <!-- existing remaining breadcrumb segments here, unchanged in style -->
      </div>
      <div class="top-rail-right">
        ${showLaunchCta && html`
          <button class="btn btn--primary" onclick=${() => navigate('/ns/' + encodeURIComponent(ns) + '/launch')}>LAUNCH</button>
        `}
        <!-- existing ⌘K palette button etc. -->
      </div>
    </header>
  `;
}
```

(The existing segments after the pill — e.g. `archive`, `run/<name>` — are unchanged in derivation and styling. Preserve any breadcrumb-derivation logic already present; just slot the pill into segment-zero.)

- [ ] **Step 5: Add CSS**

Append to `src/aiperf/operator/ui/style.css`:

```css
.ns-switcher-pill {
  background: var(--surface-2); border: 1px solid var(--border-1);
  color: var(--text-1); padding: 4px 10px; border-radius: 12px;
  font-size: 13px; cursor: pointer;
}
.ns-switcher-pill:hover { background: var(--surface-3); }
.ns-switcher-dropdown {
  position: absolute; top: 48px; left: 80px;
  width: 280px; max-height: 360px; overflow: auto;
  background: var(--surface-1); border: 1px solid var(--border-1);
  border-radius: 8px; padding: 6px; z-index: 200;
  box-shadow: 0 6px 24px rgba(0,0,0,.35);
}
.ns-switcher-search {
  width: 100%; padding: 6px 8px; margin-bottom: 6px;
  background: var(--surface-2); border: 1px solid var(--border-1); color: var(--text-1);
  border-radius: 4px;
}
.ns-switcher-item {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%; padding: 6px 8px; text-align: left;
  background: transparent; border: none; color: var(--text-1); cursor: pointer; border-radius: 4px;
}
.ns-switcher-item:hover { background: var(--surface-2); }
.ns-switcher-item--current { background: var(--surface-2); font-weight: 600; }
.ns-switcher-meta { color: var(--text-3); font-size: 11px; }
.ns-switcher-view-all {
  width: 100%; padding: 6px 8px; margin-top: 6px;
  background: transparent; border-top: 1px solid var(--border-1); color: var(--text-2);
  text-align: left; cursor: pointer;
}
.ns-switcher-view-all:hover { color: var(--text-1); }
```

- [ ] **Step 6: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py
```
Expected: the three new tests PASS. Existing tests in this file may fail until Task 9 — note any failures and skip-mark them with `pytest.mark.xfail(reason="updated in subsequent tasks")` if they block the build.

- [ ] **Step 7: Commit**

```
git add src/aiperf/operator/ui/components/namespace-switcher.js src/aiperf/operator/ui/components/top-rail.js src/aiperf/operator/ui/style.css tests/e2e/operator_ui/test_navigation.py
git commit -s -m "feat(operator-ui): namespace switcher dropdown via breadcrumb pill"
```

---

## Task 6: Migrate launch view — auto-fill + lock at `/ns/:ns/launch`

**Files:**
- Modify: `src/aiperf/operator/ui/views/launch.js`
- Modify: `src/aiperf/operator/ui/app.js` — add the route
- Modify: `tests/e2e/operator_ui/test_launch.py` — divergence-lock cases + route update
- Modify: `tests/e2e/operator_ui/_pages.py` — `LaunchPage` BASE_PATH

- [ ] **Step 1: Update `LaunchPage` and add failing tests**

Edit `tests/e2e/operator_ui/_pages.py` — change the existing `LaunchPage`:

```python
@dataclass
class LaunchPage(BasePage):
    namespace: str = "default"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/launch")
        await expect(self.page.get_by_test_id("page-launch")).to_be_visible()

    def editor(self) -> Locator:
        return self.page.get_by_test_id("launch-editor")

    def submit(self) -> Locator:
        return self.page.get_by_test_id("launch-submit")
```

Replace `tests/e2e/operator_ui/test_launch.py` body — keep file-level header — with:

```python
@pytest.mark.asyncio
async def test_launch_autofills_namespace_from_url(page, base_url, seeded_results_dir):
    p = LaunchPage(page=page, base_url=base_url, namespace="team-llama")
    await p.goto()
    contents = await p.editor().input_value()
    assert "namespace: team-llama" in contents


@pytest.mark.asyncio
async def test_launch_locks_when_yaml_namespace_diverges(page, base_url, seeded_results_dir):
    p = LaunchPage(page=page, base_url=base_url, namespace="team-llama")
    await p.goto()
    await p.editor().fill("namespace: other-team\nmodel: meta/llama-3-8b\n")
    await expect(p.submit()).to_be_disabled()
    pill = page.get_by_test_id("ns-switcher-pill")
    await expect(pill).to_have_class(lambda cls: "ns-switcher-pill--bad" in cls)


@pytest.mark.asyncio
async def test_launch_lock_lifts_when_yaml_namespace_corrected(page, base_url, seeded_results_dir):
    p = LaunchPage(page=page, base_url=base_url, namespace="team-llama")
    await p.goto()
    await p.editor().fill("namespace: other-team\nmodel: meta/llama-3-8b\n")
    await expect(p.submit()).to_be_disabled()
    await p.editor().fill("namespace: team-llama\nmodel: meta/llama-3-8b\n")
    await expect(p.submit()).to_be_enabled()
```

(Preserve any existing `from ._pages import LaunchPage` and other imports in the test file.)

- [ ] **Step 2: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_launch.py
```
Expected: FAIL — route `/ns/:ns/launch` not yet wired and divergence lock not implemented.

- [ ] **Step 3: Modify `views/launch.js`**

Read `src/aiperf/operator/ui/views/launch.js` to find the existing editor wiring and submit handler. Apply these changes:

1. Accept `ns` as a prop.
2. Pre-fill the editor on first mount with a starter that includes `namespace: <ns>` (or, if a `lastLaunchTemplateId` is present, load that template and overwrite its `namespace` field with `<ns>`).
3. On editor change, debounce 150 ms and call `extractNamespaceField(text)`. Maintain a `divergence` state: `null` if the field is absent or matches `ns`; the typed value if it diverges.
4. Disable submit when `divergence != null`. Add a chip class to the pill via a global signal `launchDivergence` exported from `lib/state.js` so `top-rail.js` can read it.

Add to `src/aiperf/operator/ui/lib/state.js`:

```javascript
export const launchDivergence = signal(null);  // string | null
```

Updated `views/launch.js` (essential additions; preserve the rest of the file's existing layout/components):

```javascript
import { useEffect, useState, useRef } from 'preact/hooks';
import { extractNamespaceField } from '../lib/yaml-namespace.js';
import { getNsPref } from '../lib/ns-prefs.js';
import { launchDivergence } from '../lib/state.js';
import { api } from '../lib/api.js';

const STARTER = (ns) => `# AIPerfJob — edit and click LAUNCH
namespace: ${ns}
model: meta/llama-3-8b-instruct
endpoint: http://my-endpoint:8000
concurrency: 64
`;

export function Launch({ ns }) {
  const [text, setText] = useState('');
  const [divergence, setDivergence] = useState(null);
  const debounceRef = useRef(0);

  // First-mount: load template if any, else starter — overwriting any
  // namespace field with the URL value before display.
  useEffect(() => {
    const tplId = getNsPref(ns, 'lastLaunchTemplateId', null);
    let initial = STARTER(ns);
    if (tplId) {
      // Templates are static module imports keyed by id elsewhere; if
      // the resolver returns a body, replace its namespace field.
      const resolved = resolveTemplateBody(tplId);
      if (resolved) initial = resolved.replace(/^namespace:.*$/m, `namespace: ${ns}`);
    }
    setText(initial);
  }, [ns]);

  // Debounced divergence check.
  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const v = extractNamespaceField(text);
      const d = (v != null && v !== ns) ? v : null;
      setDivergence(d);
      launchDivergence.value = d;
    }, 150);
    return () => clearTimeout(debounceRef.current);
  }, [text, ns]);

  // Reset divergence on unmount so the pill clears.
  useEffect(() => () => { launchDivergence.value = null; }, []);

  async function onSubmit() {
    if (divergence) return;
    await api.createJob(text);
    // existing post-submit navigation behavior preserved
  }

  return html`
    <div class="page-launch" data-testid="page-launch">
      <textarea
        class="launch-editor"
        data-testid="launch-editor"
        value=${text}
        oninput=${(e) => setText(e.target.value)}
      ></textarea>
      <button
        class="btn btn--primary launch-submit"
        data-testid="launch-submit"
        disabled=${divergence != null}
        title=${divergence ? `YAML namespace '${divergence}' doesn't match '${ns}'. Switch namespaces or fix the YAML.` : ''}
        onclick=${onSubmit}
      >LAUNCH</button>
    </div>
  `;
}

// Stub for the template resolver — preserve the existing template
// loader if `views/launch.js` already had one. If not, this no-op
// keeps behavior identical to "no template".
function resolveTemplateBody(_id) { return null; }
```

(If `views/launch.js` already has a template-resolver function, keep it and reuse it instead of the stub. Do not duplicate.)

- [ ] **Step 4: Wire route + pill class into `top-rail.js`**

Edit `src/aiperf/operator/ui/components/top-rail.js`:

```javascript
import { launchDivergence } from '../lib/state.js';
// ... and where the pill is rendered, expand the className:
const pillClass = 'ns-switcher-pill' + (launchDivergence.value ? ' ns-switcher-pill--bad' : '');
// ... use pillClass on the button.
```

Add CSS:

```css
.ns-switcher-pill--bad {
  border-color: var(--accent-bad);
  color: var(--accent-bad);
}
```

Edit `src/aiperf/operator/ui/app.js`. Add to `resolveView`:

```javascript
  const launchMatch = matchRoute('/ns/:ns/launch', currentRoute);
  if (launchMatch) return { kind: 'launch', params: launchMatch };
```

In the App switch:

```javascript
    case 'launch':
      return html`<${Launch} ns=${resolved.params.ns} />`;
```

Remove the legacy `/launch` (no ns) match-branch and any `case 'launch':` that used the old shape — replace with the above.

- [ ] **Step 5: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_launch.py
```
Expected: PASS — three tests.

- [ ] **Step 6: Commit**

```
git add src/aiperf/operator/ui/views/launch.js src/aiperf/operator/ui/lib/state.js src/aiperf/operator/ui/components/top-rail.js src/aiperf/operator/ui/style.css src/aiperf/operator/ui/app.js tests/e2e/operator_ui/test_launch.py tests/e2e/operator_ui/_pages.py
git commit -s -m "feat(operator-ui): namespace-aware launch with auto-fill and divergence lock"
```

---

## Task 7: Migrate archive view to `/ns/:ns/archive`

**Files:**
- Modify: `src/aiperf/operator/ui/views/archive.js`
- Modify: `src/aiperf/operator/ui/app.js` — add the route, drop the unprefixed `/archive`
- Rename: `tests/e2e/operator_ui/test_history.py` → `tests/e2e/operator_ui/test_namespace_archive.py`
- Modify: `tests/e2e/operator_ui/_pages.py` — `ArchivePage` BASE_PATH

- [ ] **Step 1: Update `_pages.py`**

```python
@dataclass
class ArchivePage(BasePage):
    namespace: str = "default"

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/archive")
        await expect(self.page.get_by_test_id("page-archive")).to_be_visible()

    def row(self, name: str) -> Locator:
        return self.page.get_by_test_id(f"arch-row-{self.namespace}-{name}")

    def search(self) -> Locator:
        return self.page.get_by_test_id("arch-search")
```

- [ ] **Step 2: Rename test file and update assertions**

```
git mv tests/e2e/operator_ui/test_history.py tests/e2e/operator_ui/test_namespace_archive.py
```

Inside `test_namespace_archive.py` — wherever a test instantiates `ArchivePage()`, pass an explicit namespace; drop any cross-namespace grouping assertions; update test-id matches to use the namespace-scoped form. Concrete patch shape (apply per-test):

```python
async def test_archive_renders_only_namespace_jobs(page, base_url, seeded_results_dir):
    p = ArchivePage(page=page, base_url=base_url, namespace="default")
    await p.goto()
    rows = page.locator("[data-testid^='arch-row-default-']")
    assert await rows.count() > 0
    other = page.locator("[data-testid^='arch-row-bench-perf-']")
    assert await other.count() == 0
```

Drop any assertion about a namespace-group header, namespace selector, or "All namespaces" toggle.

- [ ] **Step 3: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_archive.py
```
Expected: FAIL — route not yet wired.

- [ ] **Step 4: Modify `views/archive.js`**

Read the existing file. Two edits:

1. Accept `ns` as a prop.
2. Replace the existing `groupBy(j.namespace)` rendering with a single flat list filtered to `j.namespace === ns`.
3. The search box stays; remove any namespace-filter affordance from the toolbar (the route is the namespace filter).

Specifically, change the export signature and filter:

```javascript
export function Archive({ ns }) {
  const list = (jobs.value ?? []).filter(j => (j.namespace || 'default') === ns);
  // ... rest of the existing rendering, but never group by namespace.
  // Each row's testid is `arch-row-${ns}-${j.name}`.
}
```

Drop any `for (const [ns, group] of namespaceGroups)` rendering loop — flatten to one rows table.

- [ ] **Step 5: Wire route into `app.js`**

```javascript
import { Archive } from './views/archive.js';
// In resolveView:
  const archiveMatch = matchRoute('/ns/:ns/archive', currentRoute);
  if (archiveMatch) return { kind: 'archive', params: archiveMatch };
// Drop any branch that resolves bare `/archive` or `/jobs` to archive.
// In switch:
    case 'archive':
      return html`<${Archive} ns=${resolved.params.ns} />`;
```

- [ ] **Step 6: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_namespace_archive.py
```
Expected: PASS.

- [ ] **Step 7: Commit**

```
git add src/aiperf/operator/ui/views/archive.js src/aiperf/operator/ui/app.js tests/e2e/operator_ui/_pages.py tests/e2e/operator_ui/test_namespace_archive.py
git rm tests/e2e/operator_ui/test_history.py 2>/dev/null || true
git commit -s -m "refactor(operator-ui): scope archive to /ns/:ns/archive, drop cross-ns grouping"
```

---

## Task 8: Migrate run view to `/ns/:ns/run/:name`

**Files:**
- Modify: `src/aiperf/operator/ui/views/run.js` — only outbound href patterns
- Modify: `src/aiperf/operator/ui/app.js` — replace `/run/:ns/:name` with the prefixed form; same for `/runs/:epoch`
- Rename: `tests/e2e/operator_ui/test_job_detail.py` → `tests/e2e/operator_ui/test_run_detail.py`
- Modify: `tests/e2e/operator_ui/_pages.py` — `RunPage` BASE_PATH

- [ ] **Step 1: Update `_pages.py` `RunPage`**

```python
@dataclass
class RunPage(BasePage):
    namespace: str
    name: str

    async def goto(self) -> None:
        await self._goto(f"/ns/{self.namespace}/run/{self.name}")
        await expect(self.page.get_by_test_id("page-job-detail")).to_be_visible()
```

- [ ] **Step 2: Rename test file and update**

```
git mv tests/e2e/operator_ui/test_job_detail.py tests/e2e/operator_ui/test_run_detail.py
```

Update each test in the file to construct `RunPage(page=page, base_url=base_url, namespace="default", name="<existing-test-name>")` instead of the old positional or single-string form. The internal page-content assertions are unchanged.

- [ ] **Step 3: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_run_detail.py
```
Expected: FAIL.

- [ ] **Step 4: Update `app.js` resolveView**

Find and DELETE these branches:

```javascript
const runEpochMatch = matchRoute('/run/:ns/:name/runs/:epoch', currentRoute)
  ?? matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
if (runEpochMatch) return { kind: 'run', params: runEpochMatch };
const runMatch = matchRoute('/run/:ns/:name', currentRoute)
  ?? matchRoute('/jobs/:ns/:name', currentRoute);
if (runMatch) return { kind: 'run', params: runMatch };
```

Replace with:

```javascript
const runEpochMatch = matchRoute('/ns/:ns/run/:name/runs/:epoch', currentRoute);
if (runEpochMatch) return { kind: 'run', params: runEpochMatch };
const runMatch = matchRoute('/ns/:ns/run/:name', currentRoute);
if (runMatch) return { kind: 'run', params: runMatch };
```

- [ ] **Step 5: Update outbound links inside `run.js`**

Read `src/aiperf/operator/ui/views/run.js`. Anywhere the file builds an href to itself or peer routes (epoch links, "back to archive"), update from `/run/${ns}/${name}` to `/ns/${ns}/run/${name}`. Update `archive` back-links to `/ns/${ns}/archive`.

- [ ] **Step 6: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_run_detail.py
```
Expected: PASS.

- [ ] **Step 7: Commit**

```
git add src/aiperf/operator/ui/views/run.js src/aiperf/operator/ui/app.js tests/e2e/operator_ui/_pages.py tests/e2e/operator_ui/test_run_detail.py
git rm tests/e2e/operator_ui/test_job_detail.py 2>/dev/null || true
git commit -s -m "refactor(operator-ui): move single-run view to /ns/:ns/run/:name"
```

---

## Task 9: Make command palette namespace-aware

**Files:**
- Modify: `src/aiperf/operator/ui/components/command-palette.js`
- Modify: `tests/e2e/operator_ui/test_navigation.py` — append palette case

- [ ] **Step 1: Add failing test**

```python
@pytest.mark.asyncio
async def test_palette_groups_current_namespace_first(page, base_url, seeded_results_dir):
    await page.goto(base_url + "/#/ns/default")
    await page.keyboard.press("Meta+K")  # or 'Control+K' on linux CI
    await page.keyboard.type("bench")
    items = page.locator("[data-testid^='cmdp-job-']")
    first_id = await items.first.get_attribute("data-testid")
    # Items in the current namespace surface above the divider.
    assert first_id.startswith("cmdp-job-default-")
```

- [ ] **Step 2: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py::test_palette_groups_current_namespace_first
```

- [ ] **Step 3: Modify `command-palette.js`**

Read `src/aiperf/operator/ui/components/command-palette.js`. Two changes:

1. Each rendered candidate row's `data-testid` must include the namespace: `cmdp-job-${j.namespace}-${j.name}`. If the existing component emits a different format (e.g. `cmdp-job-${j.name}`), update it.
2. Find the section that builds the candidate list from `jobs.value`. After the existing flat list is computed, partition it by current namespace:

```javascript
import { matchRoute } from '../lib/router.js';
import { route } from '../lib/router.js';

// ... inside the palette component:
const nsMatch = matchRoute('/ns/:ns', route.value)
  ?? matchRoute('/ns/:ns/launch', route.value)
  ?? matchRoute('/ns/:ns/archive', route.value)
  ?? matchRoute('/ns/:ns/run/:name', route.value);
const currentNs = nsMatch?.ns ?? null;

const ranked = currentNs
  ? [
      ...candidates.filter(c => (c.namespace || 'default') === currentNs),
      ...candidates.filter(c => (c.namespace || 'default') !== currentNs),
    ]
  : candidates;
```

When rendering, if `currentNs` is set, insert a `<div class="cmdp-divider">other namespaces</div>` between the two partitions (only when the second is non-empty).

- [ ] **Step 4: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py::test_palette_groups_current_namespace_first
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/aiperf/operator/ui/components/command-palette.js tests/e2e/operator_ui/test_navigation.py
git commit -s -m "feat(operator-ui): namespace-aware command palette ordering"
```

---

## Task 10: Sticky `/` → `/ns/<last>` redirect on app mount

**Files:**
- Modify: `src/aiperf/operator/ui/app.js`
- Modify: `tests/e2e/operator_ui/test_navigation.py` — append redirect case

- [ ] **Step 1: Add failing test**

```python
@pytest.mark.asyncio
async def test_root_redirects_to_last_namespace_when_known(page, base_url, seeded_results_dir):
    # First visit: pick 'default' to set lastNamespace.
    await page.goto(base_url + "/#/")
    await page.get_by_test_id("np-tile-default").click()
    await expect(page).to_have_url(lambda u: u.endswith("#/ns/default"))
    # Reload root: should redirect.
    await page.goto(base_url + "/#/")
    await expect(page).to_have_url(lambda u: u.endswith("#/ns/default"))


@pytest.mark.asyncio
async def test_root_renders_picker_when_last_namespace_absent_from_jobs(page, base_url, seeded_results_dir):
    # Manually seed lastNamespace to a value that is not present in jobs.
    await page.goto(base_url + "/#/")
    await page.evaluate("window.localStorage.setItem('aiperf.ui.lastNamespace', 'ghost-ns')")
    await page.goto(base_url + "/#/")
    await expect(page.get_by_test_id("page-namespace-picker")).to_be_visible()
```

- [ ] **Step 2: Run to verify fail**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py::test_root_redirects_to_last_namespace_when_known tests/e2e/operator_ui/test_navigation.py::test_root_renders_picker_when_last_namespace_absent_from_jobs
```

- [ ] **Step 3: Implement the mount-effect in `app.js`**

Edit `src/aiperf/operator/ui/app.js`. Add inside the `App` component, after the existing polling effects:

```javascript
import { getLastNamespace } from './lib/ns-prefs.js';

// ... inside App():
useEffect(() => {
  if (route.value !== '/') return;
  const last = getLastNamespace();
  if (!last) return;
  // Wait one tick for the first poll to populate jobs.value, then redirect
  // only if the namespace appears in the observed list.
  const t = setTimeout(() => {
    const present = (jobs.value ?? []).some(j => (j.namespace || 'default') === last);
    if (present && route.value === '/') navigate('/ns/' + encodeURIComponent(last));
  }, 200);
  return () => clearTimeout(t);
}, []);
```

- [ ] **Step 4: Run to verify pass**

```
uv run pytest -n auto tests/e2e/operator_ui/test_navigation.py::test_root_redirects_to_last_namespace_when_known tests/e2e/operator_ui/test_navigation.py::test_root_renders_picker_when_last_namespace_absent_from_jobs
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/aiperf/operator/ui/app.js tests/e2e/operator_ui/test_navigation.py
git commit -s -m "feat(operator-ui): sticky / -> /ns/<last> redirect on app mount"
```

---

## Task 11: Delete legacy code paths and tests

**Files:**
- Delete: `src/aiperf/operator/ui/views/home.js`
- Delete: `tests/e2e/operator_ui/test_dashboard.py`, `test_jobs.py`, `test_unified_jobs.py`
- Modify: `src/aiperf/operator/ui/app.js` — drop the legacy `case 'home'` and any remaining unprefixed branches (`/jobs`, `/fleet`, `/leaderboard`, `/history`)
- Modify: `tests/e2e/operator_ui/_pages.py` — delete `HomePage` if still present
- Modify: `tests/e2e/operator_ui/test_compare.py`, `test_leaderboard.py`, `test_robustness.py`, `test_xss.py` — route updates only
- Modify: `tests/unit/ui/test_aiperf_dashboard_ui.py` — assertion updates

- [ ] **Step 1: Delete legacy view files and tests**

```
git rm src/aiperf/operator/ui/views/home.js
git rm tests/e2e/operator_ui/test_dashboard.py tests/e2e/operator_ui/test_jobs.py tests/e2e/operator_ui/test_unified_jobs.py
```

- [ ] **Step 2: Strip legacy branches from `app.js`**

Edit `resolveView`: remove every `if (currentRoute === '/launch')`, `'/fleet'`, `'/jobs'`, `'/leaderboard'`, `'/history'`, `'/archive'` (without ns), and `'/compare'` legacy form (the `/analysis` and `/log` branches stay as cross-namespace tier). Remove the `case 'home':` rendering branch and the `Home` import.

The final shape of `resolveView` is:

```javascript
function resolveView(currentRoute) {
  if (currentRoute === '/') return { kind: 'namespace-picker' };
  const launchMatch = matchRoute('/ns/:ns/launch', currentRoute);
  if (launchMatch) return { kind: 'launch', params: launchMatch };
  const archiveMatch = matchRoute('/ns/:ns/archive', currentRoute);
  if (archiveMatch) return { kind: 'archive', params: archiveMatch };
  const runEpochMatch = matchRoute('/ns/:ns/run/:name/runs/:epoch', currentRoute);
  if (runEpochMatch) return { kind: 'run', params: runEpochMatch };
  const runMatch = matchRoute('/ns/:ns/run/:name', currentRoute);
  if (runMatch) return { kind: 'run', params: runMatch };
  const nsMatch = matchRoute('/ns/:ns', currentRoute);
  if (nsMatch) return { kind: 'namespace-overview', params: nsMatch };
  if (currentRoute === '/analysis')                 return { kind: 'analysis' };
  if (currentRoute === '/log')                      return { kind: 'log' };
  return { kind: 'namespace-picker' };  // unmatched falls through to picker
}
```

The App switch keeps only the kinds in this resolver: `namespace-picker`, `namespace-overview`, `launch`, `archive`, `run`, `analysis`, `log`. Delete any other case bodies and the `Home` import.

- [ ] **Step 3: Update remaining e2e route literals**

Quick `grep`:

```
grep -rE "'/jobs|'/leaderboard|'/history|'/fleet|'/run/(?!:)|/#/run/|/#/jobs|/#/leaderboard|/#/history|/#/fleet" tests/e2e/operator_ui/
```

For each hit in `test_compare.py`, `test_leaderboard.py`, `test_robustness.py`, `test_xss.py`:
- `/run/<ns>/<name>` → `/ns/<ns>/run/<name>`
- `/archive` (bare) → `/ns/<ns>/archive` (whichever namespace the test was using)
- `/launch` (bare) → `/ns/<ns>/launch`
- `/jobs`, `/leaderboard`, `/history`, `/fleet` references → delete (the surrounding test will fail, fix it by aiming at the canonical replacement: `/`, `/analysis`, or `/log`).

In `test_xss.py`, add one new case:

```python
@pytest.mark.asyncio
async def test_namespace_name_with_html_chars_renders_escaped(page, base_url, with_namespace):
    # Inject a job into a namespace whose name contains HTML-special chars.
    nasty = "ns<script>alert(1)</script>"
    with_namespace(nasty)
    await page.goto(base_url + "/#/")
    tile = page.locator(f"[data-testid='np-tile-{nasty}']")
    inner = await tile.inner_html()
    assert "<script>" not in inner
    assert "&lt;script&gt;" in inner or "alert" not in inner
```

(The `with_namespace` fixture is a thin wrapper over the existing fixture builders that creates one job in the supplied namespace; if not already present, add it to `_builders.py` based on the existing `single_job` builder.)

- [ ] **Step 4: Update `test_aiperf_dashboard_ui.py`**

This unit test asserts on a screenshot or rendered DOM of the dashboard. Update the route under test from `/` (which used to be Home) to `/#/ns/default` (the new overview), and update the test-id selectors from `hm-*` to `no-*`.

- [ ] **Step 5: Run the full e2e suite to verify**

```
uv run pytest -n auto tests/e2e/operator_ui/
```
Expected: all tests PASS.

```
uv run pytest -n auto tests/unit/
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```
git add src/aiperf/operator/ui/app.js tests/e2e/operator_ui/_pages.py tests/e2e/operator_ui/test_compare.py tests/e2e/operator_ui/test_leaderboard.py tests/e2e/operator_ui/test_robustness.py tests/e2e/operator_ui/test_xss.py tests/unit/ui/test_aiperf_dashboard_ui.py
git commit -s -m "refactor(operator-ui): drop legacy routes and dashboards (clean break)"
```

---

## Task 12: Update operator-UI documentation

**Files:**
- Modify: `docs/kubernetes/dashboard-ui.md`

- [ ] **Step 1: Read current doc**

```
cat docs/kubernetes/dashboard-ui.md
```

- [ ] **Step 2: Rewrite the Navigation section**

Replace the existing navigation section with a new one matching the redesign. The new section must cover:

1. The two-tier model (cross-namespace tier vs per-namespace tier).
2. The route table from the spec (`/`, `/ns/:ns`, `/ns/:ns/launch`, `/ns/:ns/archive`, `/ns/:ns/run/:name`, `/analysis`, `/log`).
3. The sticky last-namespace behavior — first visit shows the picker, subsequent visits land on the last namespace.
4. The breadcrumb-pill switcher.
5. The launch divergence lock (auto-fill, locks LAUNCH when YAML and URL disagree).
6. Note that the picker only shows namespaces with at least one observed job — empty-but-deployable namespaces are not surfaced.

Keep the rest of the document (sidecar / API surface / data flow sections) unchanged unless that section currently references a deleted route — in which case update the route reference and leave the prose alone.

- [ ] **Step 3: Commit**

```
git add docs/kubernetes/dashboard-ui.md
git commit -s -m "docs(kubernetes): document namespace-workflow operator UI"
```

---

## Task 13: Re-shoot dashboard screenshot

**Files:**
- Modify: `docs/media/images/api-dashboard-v2.png` (overwrite in place per project convention — no dated variants)

- [ ] **Step 1: Bring up a local operator with the golden fixtures**

Use whatever path the project already supports for serving the operator UI against the e2e golden tree. Two known-working paths:

1. **`make` target** — search the repo's `Makefile` for any target serving `aiperf.operator.results_server` (e.g. `make serve-operator-ui`). If present, use it.
2. **Direct uvicorn** — adapt the fixture in `tests/e2e/operator_ui/conftest.py` (look for `_serve_app()` or the `uvicorn.Server` instantiation). Run a one-off script that calls `create_app(results_dir=tests/fixtures/operator_ui/results)` on a known port. Persist the server with `run_in_background=true` and `PYTHONUNBUFFERED=1`.

Whichever path you pick, the goal is a running server hosting the e2e golden fixtures so `/#/` renders the picker with realistic tiles.

- [ ] **Step 2: Capture the picker tile grid**

Open `http://127.0.0.1:19090/#/` in a browser. Take a 1440×900 PNG. Save to:

```
docs/media/images/api-dashboard-v2.png
```

(Overwriting in place — no dated suffix, per the durable rule.)

- [ ] **Step 3: Stop server and commit**

```
git add docs/media/images/api-dashboard-v2.png
git commit -s -m "docs: refresh dashboard screenshot for namespace picker"
```

Stop any background server you started. If you used `run_in_background=true`, send SIGTERM via the harness's process control rather than a raw `kill`.

---

## Final Verification

- [ ] Run the full test gates:

```
ruff format . && ruff check --fix .
uv run pytest -n auto tests/unit/
uv run pytest -n auto -m component_integration
uv run pytest -n auto tests/e2e/operator_ui/
make check-ergonomics
make check-ruff-baselined
```

All green. The redesign is single-PR-shippable on `ajc/k8s`.

---

## Deferred (scaffolded, not consumed in this plan)

The `lib/ns-prefs.js` wrapper from Task 1 supports four keys; this plan wires only two of them:

| Key | Wired in this plan? |
|---|---|
| `aiperf.ui.lastNamespace` | yes (Tasks 3, 5, 10) |
| `aiperf.ui.ns.<ns>.lastLaunchTemplateId` | yes (Task 6) |
| `aiperf.ui.ns.<ns>.pinnedRunNames` | no — storage layer only |
| `aiperf.ui.ns.<ns>.overviewMetricKey` | no — storage layer only |

The pinned-runs section on the overview and the per-namespace chart-key memory ship in a follow-up. Defining the keys in `ns-prefs.js` now lets the follow-up be additive: a new section in `views/namespace-overview.js` reading the existing prefs, no migration required. The spec calls these out under "Initial preferences captured" but does not require them to be consumed in the same PR as the navigation rework — splitting them keeps this PR focused on the route/structure change.
