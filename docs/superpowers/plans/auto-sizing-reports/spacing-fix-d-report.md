# Spacing Fix D — Cellular Internals Deck Geometry

**Date:** 2026-07-20  
**Scope:** `apps/explainers/decks-flow/cellular-internals.flow` only

## Problem

Scout §4: authored boxes fight auto-sizing for three-band cards (title + detail + subtitle). Worst cases: 56px START banner, 80×70 register chips, 70px policy/ownership stacks with tight vertical pitch, 130×70 cell slices, 96×70 event rail, 52px merge-tree cells.

## Changes

| Slide | IDs | Before → After | Why |
|-------|-----|----------------|-----|
| 2 · Author Config v2 | `cell0`–`cell3` | 130×70 → **145×88** | Long subtitle/detail strings; three bands need ~78px + slack |
| 3 · Self-exec | `stdio` | 220×64 → **220×88** | Three text bands in handoff card |
| 4 · Controller promotion | `single`, `roles` | 150×70 / 200×70 → **150×82 / 220×88** | Wider title on `roles`; subtitle band slack |
| 6 · Modulo partition | `own0`–`own2` | 220×64 @ y 95/173/251 → **220×80 @ y 95/185/275** | 64px clipped third band; 10px inter-card gap |
| 7 · START barrier | `c0`–`c3` | 80×70 → **95×88**; x nudged +5 | `"register"` detail width; subtitle overflow |
| 7 · START barrier | `start` | 230×56 @ y 105 → **230×88 @ y 95** | Critical scout hotspot: 56px vs ~78px three-band minimum |
| 8 · START policies | `pol0`–`pol2` | 190×70 @ y 105/190/275 → **190×82 @ y 105/200/295** | 85px pitch − 70px height left ~5px gap after grow |
| 8 · START policies | `ev7` | y 280 → **y 300** | Clears taller `pol2` stack |
| 10 · Ownership index | `indexed`, `inflight`, `completed` | 160×70 @ y 95/180/265 → **160×82 @ y 95/192/289** | Same triple-stack pitch issue as policies |
| 11 · Autonomous cell | `env`, `inputs`, `torigin` | 160×64 @ y 100/180/260 → **160×80 @ y 100/192/284** | Left-column triple stack; 12px gaps |
| 13 · Global ordinal | `global-ord` | 460×80 → **460×92** | Long detail formula string |
| 15 · Captured record | `ev0`–`ev4` | 96×70 → **108×82**; x respaced | `"admission"` / `"terminal"` width-bound at 96px |
| 19 · Merge & publish | `cell180`–`cell183` | 120×52 → **120×64**; y 100/162/224/286 → **100/172/244/316** | Three-band store cards; 8px vertical gap |
| 19 · Merge & publish | `agg1`, `sink` | agg1 y 250 → **260**; sink y 310 → **340** | Re-center agg1 on taller cell pair; sink clears cell183 |

## Unchanged (intentional)

- All `sdk.Header` 664×44 entries — scout noted viewport cost but headers are two-band chrome with resolver support; left for Domain 5 / CSS work.
- Slide 1 hub cards (90px height), slide 5 validation cards (80–88px), slide 12 worker shards (74px) — already at or above two-band minimum with modest slack.
- Narration, timeline targets, slide structure, and node IDs preserved.

## Verification

```bash
npm --prefix apps/explainers run flow-verifier -- decks-flow/cellular-internals.flow
```

Visual spot-check recommended on slides **7–8** (START barrier / policies) and **19** (merge tree) at 700×360 scene space.

## Out of scope

No TypeScript, CSS, resolver, or commit changes per Fixer D charter.
