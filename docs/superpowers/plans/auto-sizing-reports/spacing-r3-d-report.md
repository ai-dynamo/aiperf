# Round-3 Fix D — Product deck header→content Y spacing

**Date:** 2026-07-20  
**Scope:** `aiperf-vs-locust.flow`, `cellular-algorithms.flow`, `rust-architecture-atlas.flow` (optional third deck)  
**Method:** Push first content row to `y ≥ 86` (resolved header bottom ~82 + 4px gap); minimal cascade only where verifier flagged `hdr*` collisions.

---

## Summary

Resolved headers grow to ~62–66px from `y=16` (bottom ≈ 82). First content rows authored at `y=68–74` intersected the header band. Bumped only verifier-confirmed header→content pairs; left status-chip overlays, sibling crowding, and viewport escapes untouched.

---

## aiperf-vs-locust.flow

| Slide | Nodes | Δy | Before → After |
|-------|-------|----|----------------|
| s3 (coordinator) | `s3-master` | +16 | 70 → 86 |
| s4 (stats push) | `s4-t0/t1/t2`, `s4-r0/r1/r2`, `s4-merge` | +16 | labels 74→90, panels 92→108, merge 85→101 |
| s5 (topology) | `s5-sc` | +18 | 68 → 86 |
| s6 (message topology) | `s6-coord` | +18 | 68 → 86 |
| s7 (sticky routing) | `s7-steps` | +18 | 68 → 86 |
| s9 (overshoot) | `s9-gate1`, `s9-gate2` | +16 | 70 → 86 |
| s11 (side by side) | `s11-locust`, `s11-aiperf`, `s11-div` | +18 | 68 → 86 |

**Verifier:** header overlaps **11 → 0** (`s*-hdr` × first content). Deck total overlaps 32 → 27 (remaining: s1 thread chips, s6-bus/dset, s7 worker row, s8 condition cards, viewport escapes — out of scope).

**Not changed:** s1/s2/s8/s10 (no hdr collision in verifier); downstream stacks on s4/s5/s7 kept at authored Y (edges re-route).

---

## cellular-algorithms.flow

| Slide | Node | Δy | Before → After |
|-------|------|----|----------------|
| 15 (reference catalog) | `pages` | +13 | 75 → 88 |

**Verifier:** `header` × `pages` overlap cleared.

**Not changed:** Slide 0 `map-band` at y=90 already clears header (y=90 > 82); band×chapter overlaps (44 deck overlaps) are band-inset composition, not header spacing. Status-chip × algo-panel pairs (`st-*` on panels) intentionally left — scout §3C, overlay semantics.

---

## rust-architecture-atlas.flow (optional)

| Slide | Nodes | Δy | Before → After |
|-------|-------|----|----------------|
| 10 (extension seams) | `ext`, edge `a10a` path, `reg` | +10 / +22 | ext 80→90; path 140→162, 170→192; reg 170→192 |

**Verifier:** `hdr10` × `ext` cleared. `hdr10` × `a10a` remains — edge authored bbox is full viewport (`700×400`); path endpoints are below header; visual collision fixed.

Slides 0–9 first rows at y=90 already clear resolved header.

---

## Files touched

- `apps/explainers/decks-flow/aiperf-vs-locust.flow`
- `apps/explainers/decks-flow/cellular-algorithms.flow`
- `apps/explainers/decks-flow/rust-architecture-atlas.flow`

IDs and timelines preserved. No commit.

---

## Verification

```bash
cd apps/explainers
npm run flow-verifier -- --ir-only 2>&1 | rg 'aiperf-vs-locust.*hdr' | rg OVERLAP   # 0 lines
npm run flow-verifier -- --ir-only 2>&1 | rg 'cellular-algorithms 15:.*pages'       # 0 overlap lines
```
