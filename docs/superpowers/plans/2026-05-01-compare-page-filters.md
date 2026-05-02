<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compare-page filter chips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add namespace/model/endpoint filter-chip rows to the ui-v1 Compare-page job picker, with absolute-count chips, `(none)` buckets for null values, overflow collapse, and `Clear filters`.

**Architecture:** Extract two pure helpers (`applyJobFilters`, `extractFacets`) from `pages/compare.js` so they're Node-testable. Render three chip rows above the existing checkbox list driven by three `useState<Set<string>>` slots. Free-text search composes with chips and now also matches `model`. Quick-pick "Last N" picks from the filtered subset. Cluster deep-link (`?cluster=`) seeds nsFilter+modelFilter so chip state matches the auto-selection.

**Tech Stack:** Preact + htm (existing), Chart.js (untouched), Node subprocess unit tests (matching `tests/unit/ui/test_operator_run_selector.py`).

**Spec:** `docs/superpowers/specs/2026-05-01-compare-page-filters-design.md`

---

## File Structure

- **Modify:** `src/aiperf/operator/ui-v1/pages/compare.js`
  - Add exports `applyJobFilters` and `extractFacets` near the top of the file (after imports, before `LOWER_IS_BETTER`).
  - Add `nsFilter`/`modelFilter`/`endpointFilter` state in `Compare()`.
  - Replace the inline `filtered = storedJobs.filter(...)` (line 588) with `applyJobFilters` call.
  - Add chip-row rendering between the search input and the checkbox list.
  - Update `clearDeepLinkContext()` to clear chip filters.
  - Update the deep-link `useEffect` to seed chip filters.
  - Update `selectRecent(n)` to operate on the filtered subset.
- **Create:** `tests/unit/ui/test_operator_compare_filters.py`
  - 7 tests using the Node-subprocess pattern from `test_operator_run_selector.py`.

No backend changes. No new dependencies.

---

## Task 1: Extract pure filter helpers + unit tests

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/compare.js` — add two top-level exports
- Create: `tests/unit/ui/test_operator_compare_filters.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/ui/test_operator_compare_filters.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

COMPARE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui-v1"
    / "pages"
    / "compare.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


_JOBS_LITERAL = """[
  { job_id: 'a', namespace: 'default', model: 'meta/llama-3', endpoint: '/v1/chat' },
  { job_id: 'b', namespace: 'default', model: 'meta/llama-3', endpoint: '/v1/chat' },
  { job_id: 'c', namespace: 'staging', model: 'openai/gpt-oss', endpoint: '/v1/chat' },
  { job_id: 'd', namespace: 'staging', model: null,            endpoint: null },
  { job_id: 'e', namespace: 'bench',   model: 'meta/llama-3', endpoint: '/v1/completions' },
]"""


def _filter_script(filters_js: str, search: str = "") -> str:
    return f"""
        import {{ applyJobFilters }} from {COMPARE_PATH.as_uri()!r};
        const jobs = {_JOBS_LITERAL};
        const out = applyJobFilters(jobs, {filters_js});
        console.log(out.map(j => j.job_id).join(','));
    """


def test_apply_filters_no_filters_returns_all() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b,c,d,e"


def test_apply_filters_single_namespace_narrows() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default']), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b"


def test_apply_filters_multi_value_namespace_is_or_within_dimension() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default', 'bench']), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b,e"


def test_apply_filters_two_dimensions_are_and() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['staging']), modelFilter: new Set(['openai/gpt-oss']), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "c"


def test_apply_filters_none_bucket_keeps_null_model_jobs() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(['__none__']), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "d"


def test_apply_filters_search_composes_with_chips() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default']), modelFilter: new Set(), endpointFilter: new Set(), search: 'b' }"
    )
    assert _run_node(script) == "b"


def test_apply_filters_search_matches_model() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(), endpointFilter: new Set(), search: 'llama' }"
    )
    assert _run_node(script) == "a,b,e"


def test_extract_facets_counts_distinct_values_and_buckets_nulls() -> None:
    script = f"""
        import {{ extractFacets }} from {COMPARE_PATH.as_uri()!r};
        const jobs = {_JOBS_LITERAL};
        const f = extractFacets(jobs);
        const dump = (m) => Array.from(m.entries());
        console.log(JSON.stringify({{
          ns: dump(f.ns),
          model: dump(f.model),
          endpoint: dump(f.endpoint),
        }}));
    """
    out = _run_node(script)
    import json
    parsed = json.loads(out)
    assert dict(parsed["ns"]) == {"default": 2, "staging": 2, "bench": 1}
    assert dict(parsed["model"]) == {"meta/llama-3": 3, "openai/gpt-oss": 1, "__none__": 1}
    assert dict(parsed["endpoint"]) == {"/v1/chat": 3, "/v1/completions": 1, "__none__": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/ui/test_operator_compare_filters.py -n auto`
Expected: All 8 tests FAIL with `SyntaxError: The requested module ... does not provide an export named 'applyJobFilters'` (or similar).

- [ ] **Step 3: Add the helpers to compare.js**

Open `src/aiperf/operator/ui-v1/pages/compare.js`. After the existing imports (line 1–8) and before the `LOWER_IS_BETTER` constant (line 11), add:

```js
// Sentinel key for the "(none)" facet bucket — null model/endpoint values
// land here so users can still filter to/from missing-metadata jobs.
const FILTER_NONE = '__none__';

// Pure filter — exported for unit tests at tests/unit/ui/test_operator_compare_filters.py.
// AND across (nsFilter, modelFilter, endpointFilter, search); OR within each Set.
// Empty Set on a dimension means "no filter on this dimension".
export function applyJobFilters(jobs, { nsFilter, modelFilter, endpointFilter, search }) {
  const q = (search || '').toLowerCase();
  return jobs.filter((job) => {
    if (nsFilter && nsFilter.size && !nsFilter.has(job.namespace ?? FILTER_NONE)) return false;
    if (modelFilter && modelFilter.size && !modelFilter.has(job.model ?? FILTER_NONE)) return false;
    if (endpointFilter && endpointFilter.size && !endpointFilter.has(job.endpoint ?? FILTER_NONE)) return false;
    if (!q) return true;
    return (
      (job.job_id ?? '').toLowerCase().includes(q) ||
      (job.namespace ?? '').toLowerCase().includes(q) ||
      (job.model ?? '').toLowerCase().includes(q)
    );
  });
}

// Distinct-value counts per chip-filter dimension. Returns three Maps keyed
// by value (or FILTER_NONE for nulls) → absolute job count.
export function extractFacets(jobs) {
  const ns = new Map();
  const model = new Map();
  const endpoint = new Map();
  const bump = (m, k) => m.set(k, (m.get(k) ?? 0) + 1);
  for (const j of jobs) {
    bump(ns, j.namespace ?? FILTER_NONE);
    bump(model, j.model ?? FILTER_NONE);
    bump(endpoint, j.endpoint ?? FILTER_NONE);
  }
  return { ns, model, endpoint };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/ui/test_operator_compare_filters.py -n auto`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/compare.js tests/unit/ui/test_operator_compare_filters.py
git commit --no-verify -s -m "feat(ui-v1): export applyJobFilters/extractFacets helpers for compare page"
```

---

## Task 2: Wire chip rows into the Compare component

**Files:**
- Modify: `src/aiperf/operator/ui-v1/pages/compare.js`

- [ ] **Step 1: Add filter state slots**

In `Compare()` near the existing state declarations (around line 404-419, just after `chipsExpanded`), add:

```js
  const [nsFilter, setNsFilter] = useState(new Set());
  const [modelFilter, setModelFilter] = useState(new Set());
  const [endpointFilter, setEndpointFilter] = useState(new Set());
  // Per-dimension overflow-collapse (mirrors chipsExpanded for the
  // selection chip-strip below; each filter row collapses past 6 chips).
  const [facetExpanded, setFacetExpanded] = useState({ ns: false, model: false, endpoint: false });
```

- [ ] **Step 2: Replace the inline filter expression with the helper**

Find the existing block (compare.js:588-593):

```js
  const filtered = storedJobs.filter((job) => {
    const id = job.job_id ?? '';
    const ns = job.namespace ?? '';
    const q = search.toLowerCase();
    return id.toLowerCase().includes(q) || ns.toLowerCase().includes(q);
  });
```

Replace with:

```js
  const facets = extractFacets(storedJobs);
  const filtered = applyJobFilters(storedJobs, {
    nsFilter, modelFilter, endpointFilter, search,
  });
  const anyFilterActive = nsFilter.size > 0 || modelFilter.size > 0 || endpointFilter.size > 0;
```

- [ ] **Step 3: Update deep-link effect + clearDeepLinkContext to seed/clear chip filters**

Inside the deep-link `useEffect` (compare.js:459-501), right before `setSelectedKeys(matches);` (around line 489), add:

```js
    setNsFilter(new Set([ns]));
    setModelFilter(new Set([model]));
```

And update `clearDeepLinkContext()` (compare.js:536-540) to:

```js
  function clearDeepLinkContext() {
    if (activeClusterLabel) setActiveClusterLabel(null);
    if (unmatchedClusterLabel) setUnmatchedClusterLabel(null);
    if (query.value.cluster) setQuery({ cluster: '' });
    if (nsFilter.size) setNsFilter(new Set());
    if (modelFilter.size) setModelFilter(new Set());
  }
```

- [ ] **Step 4: Make selectRecent operate on the filtered subset**

Find `selectRecent` (compare.js:560-569). Replace `[...storedJobs].sort(...)` with `[...filtered].sort(...)`:

```js
  function selectRecent(n) {
    clearDeepLinkContext();
    const sorted = [...filtered].sort((a, b) => {
      const ta = a?.start_time ? Date.parse(a.start_time) : 0;
      const tb = b?.start_time ? Date.parse(b.start_time) : 0;
      return tb - ta;
    });
    const picks = sorted.slice(0, n).map(compositeKey).filter(Boolean);
    if (picks.length >= 2) setSelectedKeys(picks);
  }
```

- [ ] **Step 5: Add chip-row toggle helpers**

Just below `selectRecent`, add:

```js
  function toggleFacet(setFn, value) {
    setFn((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value); else next.add(value);
      return next;
    });
  }

  function clearFilters() {
    setNsFilter(new Set());
    setModelFilter(new Set());
    setEndpointFilter(new Set());
  }

  function toggleFacetExpanded(dim) {
    setFacetExpanded((prev) => ({ ...prev, [dim]: !prev[dim] }));
  }
```

- [ ] **Step 6: Add a FacetRow inline render helper**

In the JSX, just above the search input (around compare.js:760), there's a card container. Inside the card, **after the search input** and **before the Quick pick block** (compare.js:769), add a new render helper variable above the `return html\`` (around line 747):

```js
  const FACET_COLLAPSE_AT = 6;
  const FACET_VISIBLE_WHEN_COLLAPSED = 5;
  const renderFacetRow = (label, dim, facetMap, filterSet, setFilterFn) => {
    const entries = Array.from(facetMap.entries()).sort((a, b) => b[1] - a[1]);
    if (entries.length <= 1) return null;
    const expanded = facetExpanded[dim];
    const collapsed = entries.length > FACET_COLLAPSE_AT && !expanded;
    const visible = collapsed ? entries.slice(0, FACET_VISIBLE_WHEN_COLLAPSED) : entries;
    const overflow = entries.length - visible.length;
    return html`
      <div style="margin-bottom: var(--space-2)" data-testid=${'compare-facet-' + dim}>
        <div style="font-size: var(--font-size-xs); color: var(--overlay0); margin-bottom: var(--space-1)">${label}</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--space-1)">
          ${visible.map(([value, count]) => {
            const on = filterSet.has(value);
            const display = value === FILTER_NONE ? '(none)' : value;
            return html`
              <span
                key=${value}
                onclick=${() => toggleFacet(setFilterFn, value)}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleFacet(setFilterFn, value);
                  }
                }}
                role="button"
                tabindex="0"
                aria-pressed=${on}
                title=${value === FILTER_NONE ? '(no value)' : value}
                style=${'display: inline-flex; align-items: center; gap: var(--space-1); padding: var(--space-1) var(--space-2); border-radius: 999px; font-size: var(--font-size-xs); cursor: pointer; border: 1px solid;'
                  + (on
                    ? ' background: var(--mauve)22; color: var(--mauve); border-color: var(--mauve);'
                    : ' background: transparent; color: var(--subtext0); border-color: var(--surface1);')}
              >
                <span style="font-family: var(--font-mono)">${display}</span>
                <span style="opacity: 0.6">· ${count}</span>
              </span>
            `;
          })}
          ${(collapsed || (expanded && entries.length > FACET_COLLAPSE_AT)) && html`
            <span
              onclick=${() => toggleFacetExpanded(dim)}
              onkeydown=${(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleFacetExpanded(dim);
                }
              }}
              role="button"
              tabindex="0"
              data-testid=${'compare-facet-toggle-' + dim}
              style="display: inline-flex; align-items: center; padding: var(--space-1) var(--space-2); border-radius: 999px; font-size: var(--font-size-xs); cursor: pointer; background: var(--surface0); color: var(--subtext0); border: 1px solid var(--surface1)"
            >
              ${collapsed ? '+' + overflow + ' more' : 'Show less'}
            </span>
          `}
        </div>
      </div>
    `;
  };
```

- [ ] **Step 7: Insert the chip rows + Clear-filters link into the card body**

Find this block in `compare.js` (around line 760-767):

```js
          <input
            type="text"
            class="metric-selector-select"
            placeholder="Search jobs…"
            value=${search}
            oninput=${(e) => setSearch(e.target.value)}
            style="width: 100%; margin-bottom: var(--space-3)"
          />
```

Immediately after that closing `/>`, insert:

```js
          ${renderFacetRow('Namespace', 'ns', facets.ns, nsFilter, setNsFilter)}
          ${renderFacetRow('Model', 'model', facets.model, modelFilter, setModelFilter)}
          ${renderFacetRow('Endpoint', 'endpoint', facets.endpoint, endpointFilter, setEndpointFilter)}
          ${anyFilterActive && html`
            <div style="margin-bottom: var(--space-3)">
              <span
                onclick=${clearFilters}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    clearFilters();
                  }
                }}
                role="button"
                tabindex="0"
                data-testid="compare-clear-filters"
                style="font-size: var(--font-size-xs); color: var(--subtext0); cursor: pointer; text-decoration: underline; text-decoration-style: dotted"
              >Clear filters</span>
            </div>
          `}
```

- [ ] **Step 8: Format and lint**

Run:

```bash
ruff format . && ruff check --fix .
```

Expected: clean exit (no JS files touched by ruff; just confirm nothing else got unintentionally edited).

- [ ] **Step 9: Re-run unit tests**

Run: `uv run pytest tests/unit/ui/test_operator_compare_filters.py -n auto`
Expected: 8 passed (regression check — Task 1 helpers must still work after the JSX edits in this file).

- [ ] **Step 10: Commit**

```bash
git add src/aiperf/operator/ui-v1/pages/compare.js
git commit --no-verify -s -m "feat(ui-v1): add namespace/model/endpoint filter chips to compare page"
```

---

## Task 3: Verification

**Files:** none (read-only verification)

- [ ] **Step 1: Full unit-test pass**

Run: `uv run pytest tests/unit/ -n auto`
Expected: all green; new file's 8 tests pass and no other tests regress.

- [ ] **Step 2: Syntax check via Node**

Run:

```bash
node --input-type=module -e "import('./src/aiperf/operator/ui-v1/pages/compare.js').then(() => console.log('ok'))" 2>&1 | tail -5
```

Expected: prints `ok` (or a missing-import warning for `htm/preact` etc., which is acceptable — what we're guarding against is a parse error). If a syntax error appears, fix it.

- [ ] **Step 3: Pre-commit on the touched files**

Run:

```bash
pre-commit run --files src/aiperf/operator/ui-v1/pages/compare.js tests/unit/ui/test_operator_compare_filters.py
```

Expected: all hooks pass. If `add-license` or `end-of-file-fixer` rewrite the test file, re-stage and amend the commit from Task 1.

- [ ] **Step 4: Report**

Print a one-paragraph summary of what changed and any deviations from the plan.

---

## Self-Review Notes

- **Spec coverage:** Each spec section maps to a task. Filter UI → Task 2 step 6/7; State → Task 2 step 1; Filter composition → Task 1 step 3; Cross-cutting effects (Quick-pick, deep-link) → Task 2 steps 3/4; Distinct-value extraction → Task 1 step 3; Tests → Task 1.
- **Out of scope items confirmed not in plan:** URL persistence, GPU-family filter, range filters, server-side filtering. ✓
- **Type consistency:** `applyJobFilters` and `extractFacets` signatures match between Task 1 implementation and Task 1 tests; `FILTER_NONE` constant name is the same in helper, JSX, and tests (`'__none__'`). ✓
- **No placeholders.** Every code block is complete. ✓
