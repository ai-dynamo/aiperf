# Spacing Round 2 — Fixer B Report (Factory Preset Floors)

**Date:** 2026-07-20  
**Scope:** `apps/explainers/src/flow/sdk/generic/chrome.ts`, `catalog.ts`, tests  
**Scout:** `remaining-spacing-round2-scout.md` §3 (Default presets still below minimum)

## Summary

Raised SDK factory default geometry floors so pre-resolve IR, SDK previews, and the Node deck verifier see slack above resolver minima. Width defaults unchanged.

## Changes

| Constant / preset | Before | After | Rationale |
|-------------------|--------|-------|-----------|
| `HEADER_DEFAULT_GEOMETRY.height` | 44 | **66** | Above ~62 px header resolver minimum |
| `PANEL_DEFAULT_GEOMETRY.height` | 64 | **70** | Two-band ~62 px floor + wrap slack |
| `CARD_SIZE_PRESETS.*.height` | 80 | **88** | Three-band ~78 px + subtitle margin |
| `NOTE_DEFAULT_GEOMETRY.height` | 40 | **48** | Title-only ~38 px + padding |
| `LABEL_DEFAULT_GEOMETRY.height` | 16 | **22** | Scale-aware 12 px text band |
| `CALLOUT_DEFAULT_GEOMETRY.height` | 40 | **48** | 12 px label + vertical padding |
| `sdk.iconLabel` catalog default height | 32 | **40** | Icon-label presentation chrome |

## Tests

Added `sdk chrome factory default geometry floors` in `chrome.test.ts` (header, panel, card presets, note, label, callout). Added `sdk.iconLabel` height floor assertion in `catalog.test.ts`.

```bash
npm --prefix apps/explainers test -- \
  src/flow/sdk/generic/chrome.test.ts \
  src/flow/sdk/generic/catalog.test.ts
# 38 passed
```

## Out of scope (unchanged)

- `layout.ts`, `index.css`, product `.flow` decks
- Presentation/diagram intrinsic resolvers (Domain 1)
