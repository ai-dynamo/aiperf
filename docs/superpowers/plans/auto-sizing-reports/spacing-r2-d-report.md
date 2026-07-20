# Spacing Fix R2-D — Cellular + Diagram Catalog Tail

**Date:** 2026-07-20  
**Worker:** Round-2 Fixer D  
**Scope:** `cellular-internals.flow`, `sdk-diagram-catalog.flow` (composition slide)

## Problem

Scout §6: Round 1 missed merge-slide three-band cards (64–70 px), sketch-slide `exact`/`approx` cards (64 px), and catalog composition `results` at 120×60 with sibling diagram nodes at 120×70.

## Changes

### `cellular-internals.flow`

| Slide | IDs | Before → After | Notes |
|-------|-----|----------------|-------|
| 17 · Sketch | `exact`, `approx` | 210×**64** → 210×**88** | Three-band subtitle cards |
| 18 · Merge | `cell180`–`cell183` | 120×**64** → 120×**88**; y **100/172/244/316** → **100/196/292/388** | 8 px inter-card gap preserved |
| 18 · Merge | `agg0`, `agg1` | 150×**70** → 150×**88**; y **120/260** → **148/340** | Re-centered on taller cell pairs |
| 18 · Merge | `sink` | 210×**48** @ y **340** → 210×**52** @ y **490** | Clears `agg1` / `cell183`; title-only slack |

### `sdk-diagram-catalog.flow` — Catalog composition

| ID | Before → After | Notes |
|----|----------------|-------|
| `client`, `queue`, `service` | 120×**70** → **140×72**; x respaced **24 / 184 / 344** | Two-band diagram row inside trust boundary |
| `results` | 120×**60** → **150×72**; x **190** → **175** | Scout hotspot; centered under queue |
| `operator` | 105×**70** → 105×**72** | Two-band parity with Round-1 E |

## Scan (height 56|60|62|64|70)

- `cellular-internals.flow`: **0** remaining Card/Panel/diagram nodes in target set.
- `sdk-diagram-catalog.flow`: boundary nested `member-*` at **68** px and hero glyphs at **105** px left unchanged (outside 56–70 scan set; adequate two-band slack).

## Unchanged

- All `sdk.Header` 664×44 entries, slide 19 atlas title-only panels (52 px), IDs, timelines, narration.

## Verification

```bash
cd apps/explainers
node scripts/flow-verifier.mjs --ir-only --deck cellular-internals
node scripts/flow-verifier.mjs --ir-only --deck sdk-diagram-catalog
npm test -- src/flow/dev-tools/verify-geometry.test.ts
```

`verify-geometry.test.ts`: **5/5 pass**. Full `flow-verifier` compile currently fails on pre-existing `SDK_TIMELINE_UNSUPPORTED_ACTION` errors in `sdk-diagram-catalog.flow` (unrelated to geometry edits); geometry-only scan of edited nodes shows zero remaining 56–70 px hotspots in scope.

## Out of scope

No TS/CSS/resolver edits; no commit.
