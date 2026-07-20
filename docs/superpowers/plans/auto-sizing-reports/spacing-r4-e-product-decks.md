# Round-4 Fix E — Product narrative deck composition

**Date:** 2026-07-20  
**Scope:** Domain 5 product decks under `apps/explainers/decks-flow/`  
**Method:** Deck-only relayout — increase Y gaps, shrink authored heights where nodes grow intrinsically, nudge X for dual columns, pull nodes up/in from viewport edges for escapes. No resolver or SDK catalog changes.

---

## Summary

Cleared **99 → 0** resolution warnings across nine owned product narrative decks by spacing composition rows below resolved headers (~82px bottom + 4px gap), widening horizontal chip/column pitch, moving inset task/credit panels outside parent card bands, converting the atlas slide-10 viewport-sized path edge to a connector, and tightening finalCard file-list vertical rhythm (36px authored height, ~72px Y stride).

---

## Before / after (flow-verifier:ir --deck)

| Deck | Overlaps before | Escapes before | Total before | Overlaps after | Escapes after | Total after |
|------|----------------:|---------------:|-------------:|---------------:|--------------:|------------:|
| aiperf-vs-locust | 26 | 0 | **26** | 0 | 0 | **0** |
| slurm-velo | 14 | 1 | **15** | 0 | 0 | **0** |
| synthetic-dataset-generator | 13 | 1 | **14** | 0 | 0 | **0** |
| segment-pools | 12 | 1 | **13** | 0 | 0 | **0** |
| rust-architecture-atlas | 11 | 0 | **11** | 0 | 0 | **0** |
| tstar-warmup | 3 | 2 | **5** | 0 | 0 | **0** |
| velo-deep-dive | 2 | 0 | **2** | 0 | 0 | **0** |
| dynosim | 2 | 0 | **2** | 0 | 0 | **0** |
| rust-architecture | 1 | 0 | **1** | 0 | 0 | **0** |
| **Totals** | **84** | **5** | **89** | **0** | **0** | **0** |

Baseline counts from Round-4 scout (`remaining-spacing-round4-scout.md` Domain 5). Post-fix verification run 2026-07-20 on the same command line.

---

## Per-deck changes

### aiperf-vs-locust.flow (−26)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0 | Thread chips + credit column horizontal/vertical stack | Wider X pitch (64px panels), tdots/cdots Y +10–18px, credit row aligned at y=188 |
| 4 | `s5-wm` × `s5-rm` | Panels y=175, width 135 |
| 6 | Stepper/worker/sticky row | Workers column y=118/168/218 h=36; map/t2 aligned |
| 7 | Condition card row | Cards w=118, x pitch 30→165→300→435 |
| 10 | Dual-column compare | Locust/AIPerf panels h=72; timeline row y=172 |

### slurm-velo.flow (−15)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 1 | Tasks inside `s1-alloc` card | Tasks moved to y=238 below card; card h=130 |
| 4 | `s4-cells` viewport escape | Card x=448 w=210 (was 470×224) |
| 5 | SEND/REPLY labels × cards | Label y nudged; card h=130 |
| 13–14 | Cell stack + note | Cell pitch y=96/168/240 h=38; note y=310 |

### synthetic-dataset-generator.flow (−14)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0–7 | Header × first content | First rows y≥90; prefix/raw callouts pulled down |
| 1 | `media-join` escape | x=500 w=160 |
| 6 | Lineage bracket × panels | Bracket y=132 h=78 |
| 8 | `fork` × `generate` | fork w=130, generate x=180 |
| finalCard | f1–f4 stack | y=90/162/234/306 h=36 |

### segment-pools.flow (−13)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0 | Stepper/pool/callout/bracket band | phases y=88; composer/pool y=142/132; callout y=118 |
| 2 | `domain-trace` escape | x=560 w=88 |
| 5 | `raw` × `token` | token y=168 h=60 |
| finalCard | f1–f5 stack | y=90/162/234/306/378 h=36 |

### rust-architecture-atlas.flow (−11)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0 | execute/mock/cell/dyno stack | Target row y=320 h=64; execute/cell y=220 h=72 |
| 10 | `a10a` full-viewport path × ext/reg/ds… | Path edge → `connector` ext→reg; registry row y=262 h=64 |

### tstar-warmup.flow (−5)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 1 | `s2-hdr` × `s2-steps` | Stepper y=86 |
| 2, 8 | cut labels × header | `s3-cut-l` / `s9-cut-l` y=88 |
| 14, 20 | `s14-fields` / `s20-phase` escapes | Panels pulled left (x=460 / x=500) |

### velo-deep-dive.flow (−2)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 4 | `s4-hdr` × `s4-c0` | c0 y=86 |
| finalCard | `final-hdr` × `f1` | f1 y=88 |

### dynosim.flow (−2)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0 | `s0-hdr`/`s0-modes` × `s0-callout` | Rail y=86; callout y=88 |

### rust-architecture.flow (−1)

| Slide | Pattern | Fix |
|-------|---------|-----|
| 0 | `header` × `binary` | binary y=88 h=88 |

---

## Files touched

- `apps/explainers/decks-flow/aiperf-vs-locust.flow`
- `apps/explainers/decks-flow/slurm-velo.flow`
- `apps/explainers/decks-flow/synthetic-dataset-generator.flow`
- `apps/explainers/decks-flow/segment-pools.flow`
- `apps/explainers/decks-flow/rust-architecture-atlas.flow`
- `apps/explainers/decks-flow/tstar-warmup.flow`
- `apps/explainers/decks-flow/velo-deep-dive.flow`
- `apps/explainers/decks-flow/dynosim.flow`
- `apps/explainers/decks-flow/rust-architecture.flow`

No TypeScript resolver, SDK catalog, or out-of-scope deck edits.

---

## Verification

```bash
cd apps/explainers
for deck in aiperf-vs-locust slurm-velo synthetic-dataset-generator segment-pools \
  rust-architecture-atlas tstar-warmup velo-deep-dive dynosim rust-architecture; do
  ov=$(npm run flow-verifier:ir -- --deck "$deck" 2>&1 | rg -c SCENE_ABSOLUTE_SIBLING_OVERLAP || true)
  es=$(npm run flow-verifier:ir -- --deck "$deck" 2>&1 | rg -c SCENE_VIEWPORT_ESCAPE || true)
  echo "$deck overlaps=$ov escapes=$es total=$((ov+es))"
done
# All nine decks: overlaps=0 escapes=0 total=0
```

Acceptance target was ≤3 residual warnings per deck; all nine decks reached **0**.
