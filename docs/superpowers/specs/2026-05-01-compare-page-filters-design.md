<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compare-page filter chips (ui-v1)

## Problem

`src/aiperf/operator/ui-v1/pages/compare.js` left panel currently offers a
single substring search (lines 588–593) over `job_id` and `namespace`. As soon
as a cluster accumulates a few dozen completed runs across multiple namespaces,
models, and endpoints, the user has to remember exact identifiers to narrow the
list. The data already returned by `/results` (`JobEntry` in
`src/aiperf/operator/routers/results_schemas.py`) carries the categorical
dimensions needed to filter — `namespace`, `model`, `endpoint` — but the UI
does not expose them.

## Goal

Let the user filter the stored-job picker by the three categorical dimensions
already on the response, with chip-toggle UI, while keeping the existing
text-search and the existing checkbox-driven selection flow.

## Non-goals

- URL persistence of filter state (`?ns=…&model=…`). The cluster deep-link
  (`?cluster=`) stays as-is.
- GPU-family / hardware filtering. Telemetry is only on the compare-response
  `meta`, not on `/results`. Adding it would require a backend change and is
  not in scope.
- Numeric / range filters (file count, size).
- Server-side filtering. The result set is small enough that client-side is
  fine.

## UI

Three chip groups insert between the search input (compare.js:760–767) and the
checkbox list (compare.js:802–826), in this order: **Namespace → Model →
Endpoint**.

Each group:

- Renders one chip per distinct value seen in `storedJobs` for that dimension.
- Suffixes the chip with an absolute count, e.g. `default · 4`. Counts are
  computed against `storedJobs` (not the post-filter view) so numbers stay
  stable as the user toggles other dimensions.
- Renders a `(none)` chip for the bucket of jobs whose `model` or `endpoint`
  is null, so those jobs remain selectable rather than vanishing into a
  null-only blind spot.
- Hides itself entirely when there is exactly one distinct value (filtering on
  the only option does nothing).
- Uses the existing chip-overflow pattern (compare.js:906–967): show first
  6 chips, collapse the rest behind a `+N more` toggle.

When any filter is active, a small `Clear filters` link appears in the chip
row.

The free-text search remains above the chip groups. Its haystack expands to
include `model` (so the user can type "llama" and still narrow when chips are
unset).

## State

```js
const [nsFilter, setNsFilter]             = useState(new Set());
const [modelFilter, setModelFilter]       = useState(new Set());
const [endpointFilter, setEndpointFilter] = useState(new Set());
```

Empty Set = "no filter on this dimension." Null model/endpoint values are
keyed under the string `'__none__'` in the Set so `Set.has()` works.

## Filter composition

The existing `filtered = storedJobs.filter(...)` becomes a single pure
helper, extracted to keep `compare.js` testable from Node (the pattern used
elsewhere in `tests/unit/ui/`):

```js
// src/aiperf/operator/ui-v1/pages/compare.js (or a sibling helpers file)
export function applyJobFilters(jobs, { nsFilter, modelFilter, endpointFilter, search }) {
  const q = (search || '').toLowerCase();
  const NONE = '__none__';
  return jobs.filter((job) => {
    if (nsFilter.size && !nsFilter.has(job.namespace ?? NONE)) return false;
    if (modelFilter.size && !modelFilter.has(job.model ?? NONE)) return false;
    if (endpointFilter.size && !endpointFilter.has(job.endpoint ?? NONE)) return false;
    if (!q) return true;
    return (
      (job.job_id ?? '').toLowerCase().includes(q) ||
      (job.namespace ?? '').toLowerCase().includes(q) ||
      (job.model ?? '').toLowerCase().includes(q)
    );
  });
}
```

Truth table: AND across dimensions, OR within each dimension.

## Cross-cutting effects

- **Quick-pick "Last N"** (compare.js:560–569) operates on the *filtered*
  subset, not raw `storedJobs`. Filter to `staging`, click "Last 3" → three
  most recent staging runs.
- **Cluster deep-link** (compare.js:459–501): when the `?cluster=ns · model`
  effect fires, also seed `nsFilter = new Set([ns])` and
  `modelFilter = new Set([model])`. `clearDeepLinkContext()` clears both
  filters too, so the cluster pill and the chip selection always agree.

## Distinct-value extraction

A small memoized helper:

```js
function extractFacets(jobs) {
  const ns = new Map(), model = new Map(), endpoint = new Map();
  const NONE = '__none__';
  for (const j of jobs) {
    bump(ns, j.namespace ?? NONE);
    bump(model, j.model ?? NONE);
    bump(endpoint, j.endpoint ?? NONE);
  }
  return { ns, model, endpoint };
}
function bump(map, key) { map.set(key, (map.get(key) ?? 0) + 1); }
```

`useMemo` keyed on `storedJobs` so the chip rows don't recompute on every
search keystroke.

## Tests

Follow the existing `tests/unit/ui/test_operator_run_selector.py` pattern: the
filter helper lives at module scope and is exercised by a Node subprocess that
imports it from the compare module. Cases:

1. Empty filters return all jobs.
2. Single-namespace filter keeps only matching jobs.
3. Multi-value namespace filter is OR within the dimension.
4. Filters across two dimensions are AND.
5. `(none)` bucket: jobs with `model = null` are kept iff `'__none__'` is in
   the Set.
6. Search composes with chip filters: chips narrow first, search narrows
   further.
7. Search matches `model` (regression: previously only matched `job_id`/`ns`).

E2E: `tests/e2e/operator_ui/test_compare.py` currently targets the legacy
`ui/views/analysis.js`, not ui-v1, so no changes there. Hand-verify the new
chip rows in the dev server before shipping.

## File touch list

- `src/aiperf/operator/ui-v1/pages/compare.js` — add chip rows + state +
  filter wiring; export `applyJobFilters` helper.
- `tests/unit/ui/test_operator_compare_filters.py` — new file, mirrors
  `test_operator_run_selector.py` shape.

No backend changes. No new dependencies.

## Risks

- **Cardinality blowup.** A cluster with 50+ namespaces would render 50
  chips. The `+N more` overflow already handles this per dimension; if it
  becomes a real problem we add a per-dimension type-ahead later.
- **Stale absolute counts.** Counts reflect `storedJobs`, not the post-filter
  view. A user filtering by namespace=staging and seeing `model: llama · 4`
  may expect "4 staging-llama jobs" but the 4 is the global count. Decision
  documented here; switch to live counts if the confusion surfaces.
