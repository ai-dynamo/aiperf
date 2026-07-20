# Round-3 Fixer A Report — sdk-generic-catalog variant-column relayout

**Date:** 2026-07-20  
**Scope:** `apps/explainers/decks-flow/sdk-generic-catalog.flow` only  
**Method:** Y/X nudges on right-column variant stacks; caption labels placed after resolved growth bands; selective authored height bumps for sub-72 boxes.

## Problem

After R2 presentation/diagram intrinsic growth, ~59 verifier `SCENE_ABSOLUTE_SIBLING_OVERLAP` warnings in this deck. Primary pattern: variant specimens at x≈430–435 kept pre-growth Y while quotes, code blocks, lists, and caption labels resolved taller.

## Changes

Re-spaced **30+ slides** with right-column variant+caption stacks using **variant → +10px → caption → +12px → next variant** rhythm. Representative fixes:

| Slide | Adjustment |
|-------|------------|
| `sdk.richText`, `sdk.codeBlock`, `sdk.quote`, `sdk.list`, `sdk.propertyList` | Pushed v2 rows and caption labels down ~20–55px |
| `sdk.paragraph` | Narrowed hero (220×88); moved variant column to y=240+ |
| `sdk.iconLabel`, `sdk.alert`, `sdk.statusCard`, `sdk.emptyState` | Three-band stacks respaced |
| `sdk.stat`, `sdk.metric`, `sdk.table`, `sdk.tableRow`, `sdk.tagList` | Two-band stacks respaced |
| Navigation variants (`breadcrumb`, `tabs`, `pagination`, `timeline`, `timelineItem`) | Caption labels lowered |
| `sdk.rating` | Vertical stack infeasible (resolved height ~110px/item); switched to horizontal trio at y=100 like `sdk.gauge` |
| `sdk.semaphore` | Raised authored height 28→36; widened label gaps |
| Composition variants (`section`, `splitPane`, `mediaObject`, `toolbar`) | Preventive relayout |

IDs, timelines, and slide order unchanged.

## Verification

```bash
cd apps/explainers
npm run flow-verifier -- --ir-only 2>&1 | rg 'sdk-generic-catalog.*SCENE_ABSOLUTE_SIBLING_OVERLAP' | wc -l
```

| Metric | Before | After |
|--------|-------:|------:|
| `sdk-generic-catalog` sibling overlaps | **59** | **37** |

**−22 warnings (−37%).** All targeted variant-column stacks (quote, list, props, code, icon-label, alert, status-card, empty-state, timeline, etc.) now clear. `sdk.rating` variants use horizontal layout with captions at y=340 (resolved rating height ~230px). Remaining 37 are outside this fix domain: gauge/progress/meter internal `__track`/`__value` chrome, opener/finale composition slides, media-object hero slot crowding, and chapter intro collisions.

No commit created.
