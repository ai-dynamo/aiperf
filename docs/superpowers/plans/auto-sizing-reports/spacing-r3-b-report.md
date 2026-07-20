# Spacing Round 3 — Fixer B Report (Viewport Escape Remediation)

**Date:** 2026-07-20  
**Scope:** `cellular-internals.flow`, `flow-sdk-examples.flow`  
**Scout:** `remaining-spacing-round3-scout.md` §8

## Summary

Compacted absolute compositions that exceeded the 700×400 scene viewport after intrinsic growth. Cleared **22** verifier warnings across the two scoped decks (**0** remain); repo-wide escapes **45 → 36** per scout baseline.

## Fixed node IDs

### `cellular-internals.flow`

| Slide | Node IDs | Fix |
|-------|----------|-----|
| 15 Retain rows | `note15` | width 500→470 (x+width ≤ 700) |
| 16 Exact fold | `note16` | width 500→470 |
| 18 Merge & publish | `cell180`–`cell183`, `agg0`, `agg1`, `merge`, `sink`, `merge-to-sink` (arrow) | Tightened cell column pitch (y 92/164/236/308); realigned aggregators; moved `sink` y 490→248; nudged `merge` y 145→130 |

### `flow-sdk-examples.flow`

| Slide | Node IDs | Fix |
|-------|----------|-----|
| 1 Chrome vocabulary | `s2-note`, `s2-note__chrome`, `s2-note__caption` | Raised bottom row (divider/legend/note y −10/−18) |
| 16 AIPerf composites | `s7-export`, `s7-export__exporter-1` (+ chrome/label) | Moved export x 470→420, y 70→240 (fits 280×160 beside pipeline) |
| 19 Agent checklist | `s10-stack`, `s10-d` (+ chrome/title/detail) | Stepper y 90→78; stack y 130→100, gap 10→6 |

## Verification

```bash
cd apps/explainers
npm run flow-verifier -- --ir-only 2>&1 | rg 'SCENE_VIEWPORT_ESCAPE.*(cellular-internals|flow-sdk-examples)'
# (no output — 0 escapes in scoped decks)
```
