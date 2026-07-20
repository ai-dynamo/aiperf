# Round-4 Fix D — Header→content spacing (cellular-internals + flow-sdk-examples)

**Date:** 2026-07-20  
**Scope:** Domain 4 from `remaining-spacing-round4-scout.md` — header→content Y collisions and `s2-note` viewport escape.  
**Method:** R3-D pattern — push first content row to `y ≥ 86` (resolved header bottom ~82 + 4px gap); minimal cascade only where verifier flagged collisions.

---

## Summary

Resolved `sdk.Header` chrome grows to ~62–66px from `y=16` (bottom ≈ 82). First content rows authored at `y=68–78` intersected the header band. Bumped verifier-confirmed header→content pairs and merge-slide cell layout; left Domain 5 composition debt (rx2-hdr, s7-hdr, finalCard stacks, etc.) untouched.

---

## Before / after verifier counts

| Deck | Overlaps (before) | Escapes (before) | Overlaps (after) | Escapes (after) |
|------|------------------:|-----------------:|-----------------:|----------------:|
| `cellular-internals.flow` | **13** | **0** | **0** | **0** |
| `flow-sdk-examples.flow` (total deck) | **21** | **1**† | **11** | **0** |
| `flow-sdk-examples.flow` (Domain 4 subset) | **10** hdr + **1** esc | **1** | **0** | **0** |

† Scout reported `s2-note` viewport escape; baseline IR run showed 21 overlaps and no escape line (note resolved within viewport at authored coords). Repositioned `s2-note` preventively; post-fix escapes **0**.

### Domain 4 acceptance filters

```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck cellular-internals 2>&1 \
  | rg 'hdr[0-9]+.*OVERLAP|OVERLAP.*hdr[0-9]+' | wc -l
# → 0

npm --prefix apps/explainers run flow-verifier:ir -- --deck flow-sdk-examples 2>&1 \
  | rg 's2-hdr.*OVERLAP|SCENE_VIEWPORT_ESCAPE.*s2-note' | wc -l
# → 0
```

Header→content overlaps on owned slides cleared. Remaining **11** warnings on `flow-sdk-examples` are Domain 5 (rx2-hdr, s5/s6/s7/s10 composition, finalCard) plus one pre-existing `rx4-loop` degenerate-path error — out of scope for Fix D.

---

## cellular-internals.flow

| Slide | Nodes | Δy / layout | Before → After |
|-------|-------|-------------|----------------|
| 0 | `phase0`, `tags0` | +16 | 70→86, 72→88 |
| 2 | `chip2` | +16 | 72→88 |
| 4 | `chip4` | +16 | 72→88 |
| 6 | `start` | +15 | 95→110 (clears `phase6` band) |
| 9 | `indexed` | +15 | 95→110 (clears `phase9` band) |
| 13 | `chip13`; cascade `http`, `srv` | +16 chip; +15 cards | chip 72→88; http/srv 110→125 |
| 18 merge | `cell180–183` | 2×2 grid, h=68 | vertical stack → (30/160, 102/196); `agg0/1`, `merge`, `sink` shifted right |
| 19 | `phase19`, `ctrl19–res19` | +16 phase; +16 panels | phase 70→86; panels +16 cascade |
| finalCard | `f-phase`, `f-ctrl–f-res` | same as slide 19 | phase 70→86; panels +16 cascade |

**Verifier:** deck total **13 → 0** warnings.

---

## flow-sdk-examples.flow

| Slide | Nodes | Δy / layout | Before → After |
|-------|-------|-------------|----------------|
| 1 chrome | `s2-callout` | +18; moved right | y 68→86, x 290→440; target y 98→112 |
| 1 chrome | `s2-bracket`, `s2-card`, `s2-panel`, `s2-tags`, `s2-div`, `s2-legend`, `s2-note` | +14 cascade | bracket 84→98; card 90→104; panel 98→112; tags 113→127; div 192→206; legend 204→218; note 308→322 |
| 4 curve matrix | `rx1-col-*` | +16 | column headers 70→86 |
| 4 curve matrix | `rx1-row-*`, all `rx1-*-src/dst` shapes | +16 | matrix body shifted below header band |
| 4 curve matrix | `rx1-note` | +24 | 338→362 (clears bottom-row shape band after matrix shift) |

**Verifier (Domain 4):** `s2-hdr`×`s2-callout` **1 → 0**; `rx1-hdr`×column labels **9 → 0**; `s2-note` viewport escape **0**.

**Not changed (Domain 5):** slides 5, 7, 11–15, 18, finalCard; `rx4-loop` degenerate-path error.

---

## Files touched

- `apps/explainers/decks-flow/cellular-internals.flow`
- `apps/explainers/decks-flow/flow-sdk-examples.flow`

No resolver or catalog changes. Timelines and node IDs preserved.

---

## Verification commands

```bash
cd apps/explainers
npm run flow-verifier:ir -- --deck cellular-internals
# summary: 0 error(s), 0 warn(s)

npm run flow-verifier:ir -- --deck flow-sdk-examples
# summary: 1 error(s), 11 warn(s) — Domain 4 hdr/escape targets at 0
```
