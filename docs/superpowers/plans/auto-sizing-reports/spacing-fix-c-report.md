# Spacing Fix C — Pipeline Stage Defaults

**Date:** 2026-07-20  
**Scope:** `apps/explainers/src/flow/sdk/generic/topology.ts`, `topology.test.ts`

## Problem

Scout §1 / §2D: `sdk.pipeline` stage boxes defaulted to **96×56**, smaller than panel/card presets (160×64, 80px card height). Short stage titles clipped; stages sat closer than panel-scale chrome expects.

## Change

| Constant | Before | After | Rationale |
|----------|--------|-------|-----------|
| `PIPELINE_DEFAULT_NODE_WIDTH` | 96 | **120** | ~25% wider; fits one-line panel title at 0.9 scale |
| `PIPELINE_DEFAULT_NODE_HEIGHT` | 56 | **64** | Aligns with `PANEL_DEFAULT_GEOMETRY.height`; matches title band + inset |
| `PIPELINE_DEFAULT_GAP` | 24 | **28** | Slight bump as nodes grow so stages do not visually collide |

`placePipelineNode` unchanged: `geometry.width/height > 0` still wins over defaults.

## Tests

Added to `topology.test.ts`:

1. **floors unset stage geometry** — zero-sized slot roots → 120×64, gap 28, group width `2×120+28`.
2. **preserves explicit stage sizes** — authored 200×100 stage keeps size; sibling still gets defaults; group height follows tallest stage.

Existing multi-root placement test unchanged (uses explicit 80×40 geometry).

## Verification

```bash
npm --prefix apps/explainers test -- src/flow/sdk/generic/topology.test.ts
```

## Out of scope

No changes to `layout.ts`, `index.css`, or `.flow` decks per Fixer C charter.
