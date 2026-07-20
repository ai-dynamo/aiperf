# Spacing Fix E — SDK Catalog Deck Crowding

**Date:** 2026-07-20  
**Worker:** Fixer E (SDK catalog decks)  
**Scope:** `sdk-diagram-catalog.flow`, `sdk-generic-catalog.flow`

---

## Summary

Raised authored variant-box heights across the diagram catalog deck and expanded indicator/card clusters in the generic catalog deck. No changes to `diagram/catalog.ts` — deck overrides were the sole crowding source.

---

## Changes

### `sdk-diagram-catalog.flow`

| Pattern | Before | After | Count |
|---------|--------|-------|-------|
| Storage variants | 160×62 | 160×78 | 18 |
| Control variants | 150×62 | 150×72 | 24 |
| All other variants | *×62 | *×72 | 85 |

127 variant nodes total. Component IDs, slide order, and x positions unchanged. Viewport is 700×400; tallest variant row (y=306 + h=78) ends at 384 — within bounds.

### `sdk-generic-catalog.flow`

| Slide | Change |
|-------|--------|
| `sdk.gauge` | Variant gauges 80×58 → 80×68; y 110→108; caption y 195→198 |
| `sdk.statusCard` | Variant cards 220×82 → 220×88; sc-v2 y 215→218 (20 px inter-row gap preserved) |
| `sdk.semaphore` | Moved `sem-vl` from x=540/y=175 (overlapped sem-v2 column) to x=430/y=288 below variant stack |

---

## Verification

```
npm test -- src/flow/dev-tools/verify-geometry.test.ts   # 5/5 pass
verifyPackageIr(sdk-diagram-catalog)                      # errors=0 warns=0
verifyPackageIr(sdk-generic-catalog)                      # errors=0 warns=0
```

`diagram/catalog.ts` defaults (144×82) left untouched — catalog deck systematically overrides geometry per variant row.

---

## Not changed (intentional)

- Chapter opener `sdk.Label` notes at y=300 (separate slides, no variant overlap)
- Inset compact demo frames (90×80) — intentional tight-specimen layout
- `final-status` card 250×62 on closing slide — single-line title, adequate
