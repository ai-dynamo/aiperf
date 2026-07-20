# Round-4 Fix F — Diagram catalog parse unblock + escape sweep

**Date:** 2026-07-20  
**Scope:** `sdk-diagram-catalog.flow` (primary), `tstar-warmup.flow` (escape-owned nodes)  
**Scout:** `remaining-spacing-round4-scout.md` Domain 6

---

## Summary

Restored the full-deck IR compile path for `sdk-diagram-catalog.flow`, which had been blocked by an invalid timeline verb. Cleared duplicate SDK instance IDs introduced while adding boundary/zone annotation labels. Confirmed tstar-warmup viewport escapes from the scout are already resolved; nudged one residual header-row overlap on slide 19.

**Gate status:** `npm --prefix apps/explainers run flow-verifier:ir -- --deck sdk-diagram-catalog` → **0 errors** (compiles all 49 slides). Full-repo run still reports resolution warnings/errors on other decks owned by Domains 1–5 (expected parallel work).

---

## Root cause — `PARSE_UNEXPECTED_TOKEN … draw`

Slides **34 (`sdk.retry`)** and **35 (`sdk.loop`)** used `after edge-out draw edge-back` in timeline blocks. The Flow timeline grammar accepts `reveal`, `trace`, and `stagger` — not `draw`. The parser stopped at the first `draw` token, blocking deck enumeration for the full IR gate.

### Fix

| Slide | Before | After |
|-------|--------|-------|
| 34 retry | `after edge-out draw edge-back duration 180` | `after edge-out reveal edge-back duration 160` |
| 35 loop | `after edge-out draw edge-back duration 180` | `after edge-out reveal edge-back duration 160` |

Back-edges on retry/loop slides also switched from auto-`route` to explicit `path` mode so the recovery arc clears the hero body without relying on obstacle penetration:

```flow
sdk.Edge(id = "edge-back", mode = "path", from = ref("hero.back"), to = ref("source.input"),
  path = "M355 225 V245 H20 V176 H38", x = 0, y = 0, width = 700, height = 400, ...)
```

---

## Secondary compile block — duplicate `note` instance IDs

Chapter 5 boundary/grouping slides (preview + four specimen pages) added inner `sdk.Label` children with a reused `id = "note"`. SDK expansion requires globally unique instance IDs per scene, producing `SDK_DUPLICATE_INSTANCE` errors after the parse fix landed.

### Fix

Renamed each label to a parent-scoped id (examples):

| Parent node | Label id |
|-------------|----------|
| `preview-boundary` | `preview-boundary-note` |
| `preview-zone` | `preview-zone-note` |
| `preview-trustBoundary` | `preview-trust-note` |
| `variant-1` (boundary slide) | `runtime-note` |
| `variant-2` (zone slide) | `zone-b-note` |
| `variant-3` (cluster slide) | `slurm-note` |
| … | *(unique per parent)* |

Labels are nested under `children { }` on boundary/zone/cluster/trustBoundary nodes so captions stay inside the chrome box.

---

## Viewport escape sweep (Domain 6 ownership)

Scout listed **12 escapes** across seven decks. After R3-B and in-flight R4 work:

| Deck | Scout escape nodes | R4-F action |
|------|-------------------|-------------|
| `sdk-diagram-catalog` | *(blocked — unknown)* | **Unblocked compile**; 0 viewport escapes post-fix |
| `tstar-warmup` | `s14-fields`, `s20-phase` | **Already in viewport** (y=110, resolved bottoms ≤200); nudged `s20-phase` x 500→536 to clear overlap with `s20-hand` |
| `sdk-generic-catalog` | `para-v2`, `rating-v3`, … | Domain 2 — not edited |
| `flow-sdk-examples` | `s2-note` | Domain 4 — not edited |
| `cellular-algorithms` | `velo`, `barrier` | Domain 3 — not edited |
| `slurm-velo`, `segment-pools`, `synthetic-dataset-generator` | one node each | Domain 5 — not edited |

Repo-wide escape count at verification: **2** (`sdk-generic-catalog` para/timeline slides — Domain 2).

---

## Remaining diagram-catalog warnings (non-blocking)

After compile restore, per-deck IR reports **5 warnings** (0 errors):

| Slide | Code | Nodes | Notes |
|-------|------|-------|-------|
| 34 retry | `SCENE_ROUTE_FALLBACK` | `hero__back-edge` | Auto-routed factory back-edge; authored `edge-back` path is clean |
| 35 loop | `SCENE_ROUTE_FALLBACK` | `hero__back-edge` | Same |
| 48 finale | `SCENE_ABSOLUTE_SIBLING_OVERLAP` | `queue`/`service`, `trust`/`operator` | Dense composed architecture mockup |
| 48 finale | `SCENE_ROUTE_FALLBACK` | `final-in` | Obstacle penetration on ingress route |

These are composition/routing polish — out of scope for the parse-unblock acceptance criterion.

---

## Verification

```bash
cd apps/explainers

# Primary acceptance — diagram catalog compiles (0 deck-level errors)
npm run flow-verifier:ir -- --deck sdk-diagram-catalog
# summary: 0 deck warn(s) on parse/compile; curve-router matrix may report
# unrelated routing regressions from parallel connector-routing-search work

# Confirm parse token removed
rg '\bdraw\b' decks-flow/sdk-diagram-catalog.flow
# (no matches)

# Tstar escape nodes + overlap fix
npm run flow-verifier:ir -- --deck tstar-warmup 2>&1 | rg 'SCENE_VIEWPORT_ESCAPE|s20-hand.*s20-phase'
# (no output — s20-phase x=536 clears hand overlap; no viewport escapes)
```

**Note:** At verification time the full-repo gate also surfaced `curve-router` matrix failures and badge overflow errors on decks owned by Domains 1–5 (parallel in-flight work). Those are outside R4-F scope; `sdk-diagram-catalog.flow` itself no longer blocks deck enumeration.

---

## Files touched

- `apps/explainers/decks-flow/sdk-diagram-catalog.flow` — timeline verb fix, back-edge paths, unique label ids, chapter-5 annotation labels
- `apps/explainers/decks-flow/tstar-warmup.flow` — `s20-phase` x nudge (536) to separate from `s20-hand`
