# Remaining Spacing Scout — Round 2 (Post Fix A–E)

**Date:** 2026-07-20  
**Scope:** Read-only reconnaissance in `apps/explainers` after Round 1 spacing fixes (subtitle resolver, callout/legend, stage CSS trim, pipeline 120×64/28, cellular deck pass, sdk catalog deck growth).  
**Thoroughness:** Medium.

---

## Executive summary

Round 1 closed the highest-impact resolver gaps (subtitle panels, callouts, legend factory width, pipeline defaults, cellular START slides, catalog variant rows). **Round 2 crowding is mostly:**

1. **`diagram.*` nodes still identity-sized** — no intrinsic resolver; factory uses fixed 22/20 px bands and a 46 px glyph gutter while decks author 60–72 px boxes.
2. **Presentation chrome (`code-block`, `quote`, `icon-label`) paints but does not size** — `core.group` + `presentation` has semantic chrome only; long monospace or icon+label strings clip at factory/deck geometry.
3. **SDK factory preset floors remain below resolver minimums** — especially `HEADER_DEFAULT` 44 vs ~62, three-band cards at 80 vs ~78 with zero slack, `LABEL` height 16 vs scale-aware text metrics.
4. **Seven product explainer decks still author 56–70 px two-band boxes** — resolver can grow height at runtime, but authored heights fight layout spacing and the Node deck verifier still uses raw geometry.
5. **Cellular merge slide + diagram composition slide missed Round 1 tail** — three-band cards still authored at 64–70 px; diagram trust-boundary children at 120×60.
6. **Lane/frame title bands (28/48 px) and footer shell padding** — long lane titles and `max(64px, 6vh)` footer floor still compress diagram viewport on short laptops.

---

## Round 1 baseline (what is already fixed)

| Fix | Report | Status |
|-----|--------|--------|
| Subtitle in `resolvePanelLayout` | spacing-fix-a | `layout.ts:373-389` includes subtitle height/width |
| Callout + legend sizing | spacing-fix-a | `resolveCalloutLayout` at `layout.ts:401-431`; legend width in `chrome.ts:1090-1103` |
| Stage hero / subtitle CSS trim | spacing-fix-b | `index.css:1252`, `748-769` |
| Pipeline 120×64, gap 28 | spacing-fix-c | `topology.ts:504-507` |
| Cellular deck geometry | spacing-fix-d | Slides 2–19 except tail noted below |
| sdk-diagram / sdk-generic catalog boxes | spacing-fix-e | Variant rows → 72/78; gauge/semaphore fixes |

---

## 1. Diagram-node intrinsic layout still missing

**Verdict:** Yes — still a Round 2 domain.

`LAYOUT_CAPABILITIES` registers panel/header/text/stepper/lane/etc. but **no** `diagram.*` entry:

```1083:1102:apps/explainers/src/core/diagram/capabilities/layout.ts
export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  // ...
  { capabilityId: "core.stepper", resolveLayout: resolveStepperLayout },
  { capabilityId: "core.circle", resolveLayout: resolveEllipseLayout },
  { capabilityId: "core.ellipse", resolveLayout: resolveEllipseLayout },
];
```

Diagram nodes emit `capabilityId: diagram.${category}` (`diagram/catalog.ts:187`) with **fixed** title/detail placement:

```332:338:apps/explainers/src/flow/sdk/diagram/catalog.ts
      semanticTarget(
        titleId,
        { x: 46, y: detail === undefined ? 20 : 12, width: Math.max(box.width - 56, 0), height: 22 },
        label,
```

Semantic chrome mirrors the same fixed bands (`capabilities/chrome.ts:248-290`: title `TITLE_HEIGHT`, detail at `y + 38`, glyph gutter 46 px).

| Source | Geometry | Issue |
|--------|----------|-------|
| Catalog default | `144×82` (`diagram/catalog.ts:128-129`) | OK for short labels; no width growth for long titles |
| Round 1 deck overrides | `150×72`, `160×78` (`spacing-fix-e`) | Better slack; still no intrinsic growth |
| **Composition slide (missed)** | `sdk.Database` **120×60** (`sdk-diagram-catalog.flow:1602`) | Two-band minimum ~62 px; detail band ends ~y+58 inside 60 px box |
| Pipeline diagram nodes | `120×70` (`sdk-diagram-catalog.flow:1599-1601`) | Tight for long titles (`"Load client"`, `"Ready turns"`) at 104 px text width (150−56 gutter) |

Deck overrides at 150×72/160×78 help static layout but **do not replace** a `resolveDiagramLayout` for authored-long titles or the 60 px composition outlier.

---

## 2. Presentation chrome without sizing

**Verdict:** Yes — independent resolver gap.

`hasNativeSemanticChrome` includes presentation modes (`capabilities/chrome.ts:97-101`), and tests verify **rendering** only (`layout.test.ts:340-384`), not layout growth:

| Presentation | Chrome placement | Default / deck geometry | Gap |
|--------------|------------------|---------------------------|-----|
| `code-block` | `y+10`, `height−20`, `fontSize: 12`, `whiteSpace: pre` (`chrome.ts:129-150`) | Catalog `320×140` (`catalog.ts:120`); deck variants **215×70** (`sdk-generic-catalog.flow:320-321`) | No resolver; multi-line `pre` clips at 70 px |
| `quote` | italic, padded box (`chrome.ts:129-150`) | Catalog `280×88` (`catalog.ts:121`); deck **215×80** (`sdk-generic-catalog.flow:338-339`) | No width/height growth for wrapped quotes |
| `avatar` | circle `rootBox` only (`chrome.ts:157-166`) | Catalog `48×48` (`catalog.ts:127`) | Size is intentional; OK when square |
| `icon-label` | label at `x+40`, `width−48`, `fontSize: 12` (`chrome.ts:169-190`) | Catalog default **160×32** (`catalog.ts:128`); hero **340×70** (`sdk-generic-catalog.flow:484`) | 32 px default clips any non-trivial label; no intrinsic width |

Factories emit `core.group` with `presentation` props (`catalog.ts:550-667`) — **outside** `resolvePanelLayout` / `resolveTextLayout` paths.

---

## 3. Default presets still below minimum — should factories raise floors?

**Verdict:** Yes for defaults; resolvers already grow at resolve time for panel/header/text when resolution runs.

Triplicated constants remain in `text-metrics.ts:12-15` and `generic/chrome.ts:57-88`:

| Preset | Location | Value | Resolver minimum (0.9 scale) | Verdict |
|--------|----------|-------|------------------------------|---------|
| `HEADER_DEFAULT_GEOMETRY.height` | `chrome.ts:63` | **44** | `INSET×2 + TITLE + DETAIL + 4` ≈ **62** (`resolveHeaderLayout` `layout.ts:453-456`) | Default still 18 px short before resolve |
| `CARD_SIZE_PRESETS.*.height` | `chrome.ts:85-87` | **80** | Three-band card ≈ **78** (`layout.ts:385-389`) | 1–2 px slack; zero margin for subtitle wrap |
| `PANEL_DEFAULT_GEOMETRY.height` | `chrome.ts:64` | **64** | Two-band ≈ **62** | Barely OK; no detail wrap slack |
| `NOTE_DEFAULT_GEOMETRY.height` | `chrome.ts:66` | **40** | Title-only ≈ **38** | Tight; width-only growth |
| `LABEL_DEFAULT_GEOMETRY.height` | `chrome.ts:67` | **16** | `resolveTextLayout` uses default `fontSize: 14` → scaled **12.6** (`layout.ts:478-488`, `text-metrics.ts:9-27`) | Height OK for 12 px labels; **width** still tight for specimen captions |
| `CALLOUT_DEFAULT_GEOMETRY` | `chrome.ts:68` | 140×40 | Resolver grows (`layout.ts:424-427`) | Factory default still small; resolver compensates |

**Recommendation:** Raise factory floors in `generic/chrome.ts` so pre-resolve IR, SDK previews, and the Node verifier (`geometry.mjs` — still authored-geometry-only) see less cramped boxes. Add +4 px slack on three-band card presets.

Label factory passes authored height through unchanged (`chrome.ts:968-973`) — deck labels at `height = 16` with long `text` (e.g. `sdk-generic-catalog.flow:898` `width = 200, height = 18`) rely on `resolveTextLayout` width growth only.

---

## 4. Other decks still cramped (sample)

Resolver-backed **two-band** panels at **70 px** are above the ~62 px floor but leave only ~8 px slack and tight horizontal fit for long titles/details. **56 px** and **three-band 64 px** boxes remain problematic.

### `segment-pools.flow`

| Lines | Node | Geometry | Issue |
|-------|------|----------|-------|
| 53-57 | `composer` | 150×**70** | `"SegmentPool"`-scale titles OK; many siblings 70×90 mixed |
| 353-357 | `hash-recipe` | 400×**70** | Title `"HASH_VERSION + domain + parent id"` width-bound at 400 px |
| 57, 94, 140, 238, 446, 517, 625 | various panels | **70** height | Systematic two-band 70 px grid |

### `rust-architecture.flow` / `rust-architecture-atlas.flow`

| Lines | Pattern |
|-------|---------|
| `rust-architecture.flow:262-272` | Registry hub **200×70** + satellite panels **110×70** |
| `rust-architecture.flow:467-475` | Coordinator **200×70**, workers **110×70** |
| `rust-architecture-atlas.flow:37-167` | Dense atlas: **14+ panels at 70×80**, spawn cell **70×70** |

### `tstar-warmup.flow`

| Lines | Pattern |
|-------|---------|
| 163-211, 415-822, 1416-2220 | **40+** panels/cards at **70×80** or **110×70** |
| 1010-1057 | Domain cards **90×80** abreast — horizontal crowding at y=100 |

No `subtitle` props in this deck — two-band resolver path applies.

### `dynosim.flow`

| Lines | Pattern |
|-------|---------|
| 531, 602 | Central panels **200×70** |
| 899-907 | Comparison panels **180×80** |

### `aiperf-vs-locust.flow`

| Lines | Pattern |
|-------|---------|
| 270-315 | Four abreast **70** px panels (`width` 90–160) — `"Execute task A"` / `"runs the simulated request"` |
| 343-351 | Row 2: **280×70** — long detail strings |

### `flow-sdk-examples.flow`

| Lines | Pattern |
|-------|---------|
| 376-378, 1367-1370, 1640-1656 | Panels **100×56** — **below** two-band ~62 px minimum |
| 386-388 | Pipeline stage panels **72×56** inside `sdk.pipeline` — below Round 1 topology default **64** height (explicit geometry wins) |

### Catalog decks (post Round 1)

| Lines | Pattern |
|-------|---------|
| `sdk-generic-catalog.flow:114-115, 283-284` | `RichText` / `Paragraph` variants **height = 62** |
| `sdk-generic-catalog.flow:1012` | `final-status` **250×62** — single title, acceptable |
| `sdk-diagram-catalog.flow:1602` | **`results` 120×60** inside trust boundary — **new** diagram crowding |

---

## 5. Lane / swimlane / frame band constants still tight

```32-38:apps/explainers/src/core/diagram/capabilities/layout.ts
const LANE_TITLE_BAND = 28;
const LANE_INSET = 10;
const DEFAULT_CHILD_HEIGHT = 64;
const FRAME_TITLE_BAND = 28;
const FRAME_DETAIL_BAND = 48;
```

| Constant | Used at | Issue |
|----------|---------|-------|
| `LANE_TITLE_BAND = 28` | `resolveLaneLayout` `layout.ts:891-909` | Lane title text not measured; long titles overflow band |
| `FRAME_TITLE_BAND / FRAME_DETAIL_BAND` | `resolveFrameLayout` `layout.ts:824` | Fixed 28/48 px; not derived from `estimateTextWidth` / scale |
| Deck sample | `velo-deep-dive.flow:111` | `sdk.Lane` **280×70**, title `"tcp:// · uds:// · SLURM_* · k8s"` — title band 28 px vs multi-token string |

Swimlane label gutter `SWIMLANE_LABEL_WIDTH = 72` (`layout.ts:36`) is similarly fixed.

---

## 6. Absolute overlaps after Round 1

Round 1 fixed **`sem-vl` overlap** (`spacing-fix-e`: moved to y=288). Remaining hotspots:

| Location | Evidence | Severity |
|----------|----------|----------|
| `cellular-internals.flow:807-812` | Three-band cards at **64–70 px** (`agg0`, `agg1`, `cell180–183`) — subtitle band extends past box bottom (~78 px needed) | **Clip**, not xy overlap |
| `cellular-internals.flow:769-770` | `exact` / `approx` cards **210×64** with subtitle | Same clip |
| `cellular-internals.flow:814` | `sink` panel **210×48**, title only | Title band ~38 px min in 48 px — tight |
| `sdk-generic-catalog.flow:340-341` | `quote-v1-l` at y=**192**; `quote-v1` y=105 h=**80** → bottom 185 | 7 px gap — OK |
| `sdk-generic-catalog.flow:116-117` | `rich-v1-l` y=**175**; `rich-v1` bottom 170 | 5 px — tight but non-overlapping |
| `sdk-generic-catalog.flow:562` | `TagList` **52×48** | Intentional micro-specimen; tags overflow box by design |
| `sdk-generic-catalog.flow:948-949` | `toolbar-v2` **220×46** with badge + caption | Horizontal crowding inside rail |

No new **absolute-position** sibling overlaps found in cellular/catalog decks after Round 1; remaining issues are **intrinsic clip** and **tight vertical pitch** between stacked nodes.

---

## Residual infrastructure gaps

- **Node deck verifier:** `scripts/flow-verifier/geometry.mjs` still validates authored geometry (Round 1 note unchanged) — product deck 56 px boxes may pass verifier while clipping at render time.
- **Factory vs resolve integration:** Presentation and diagram nodes lack end-to-end `resolveScene` tests beyond manual node shapes.
- **Header 664×44 on every slide:** Still consumes ~12% of 360 px scene viewport (`cellular-internals.flow:855`, passim); resolver grows height when caption present, but authored 44 px remains the floor.

---

## Top 5 parallel fix domains (disjoint file ownership)

Ordered for parallel workers. **Do not split Domain 1** across workers — it owns all new `LAYOUT_CAPABILITIES` entries.

### Domain 1 — Diagram + presentation intrinsic resolvers

| | |
|--|--|
| **Files** | `apps/explainers/src/core/diagram/capabilities/layout.ts`, `layout.test.ts`, `capabilities/chrome.ts` (diagram band y offsets), `apps/explainers/src/flow/sdk/diagram/catalog.ts` |
| **Problem** | `diagram.*` and `core.group` presentation modes have semantic chrome but identity layout; long titles, monospace blocks, and icon+label pairs clip at deck/catalog geometry. |
| **Acceptance** | Register `resolveDiagramLayout` (prefix match or per-category) and `resolvePresentationLayout` for `code-block` / `quote` / `icon-label`; tests prove width/height grow for long strings; `sdk-diagram-catalog` composition `results` node resolves to ≥62 px height; `npm test -- layout.test.ts` green. |

### Domain 2 — SDK factory preset floors

| | |
|--|--|
| **Files** | `apps/explainers/src/flow/sdk/generic/chrome.ts`, `apps/explainers/src/flow/sdk/generic/catalog.ts` (catalog default sizes for `iconLabel`, optional `codeBlock`/`quote` floors) |
| **Problem** | `HEADER_DEFAULT` 44, `CARD` 80 (zero three-band slack), `LABEL` 16, `NOTE` 40, `PANEL` 64, `iconLabel` 32 — factories emit cramped IR before resolve. |
| **Acceptance** | `HEADER_DEFAULT.height ≥ 66`; `CARD_SIZE_PRESETS.*.height ≥ 82`; `iconLabel` default height ≥ 40; factory/unit tests or catalog tests assert new floors; no regression in `catalog.test.ts`. |

### Domain 3 — Product architecture deck geometry pass

| | |
|--|--|
| **Files** | `decks-flow/segment-pools.flow`, `rust-architecture.flow`, `rust-architecture-atlas.flow`, `tstar-warmup.flow`, `dynosim.flow`, `aiperf-vs-locust.flow`, `flow-sdk-examples.flow` |
| **Problem** | Systematic **56–70 px** authored panels; `flow-sdk-examples` still uses **56 px** pipeline/panel heights below resolver minimum; long title/detail strings width-bound. |
| **Acceptance** | Bump two-band panels to **≥82 px** height (or **≥64** where resolver-only); raise `flow-sdk-examples` pipeline stages to **≥120×64**; `flow-verifier` / `verify-geometry` zero errors on sampled slides; spot-check `segment-pools` hash-recipe and `aiperf-vs-locust` slide 2. |

### Domain 4 — Cellular + diagram catalog tail

| | |
|--|--|
| **Files** | `decks-flow/cellular-internals.flow`, `decks-flow/sdk-diagram-catalog.flow` (composition slide only) |
| **Problem** | Round 1 missed merge-slide three-band cards (`cell180–183` **64 px**, `agg0/agg1` **70 px**, `exact/approx` **64 px**); diagram catalog trust-boundary `results` **120×60**. |
| **Acceptance** | Three-band cards ≥**88 px** height with respaced y; `results` ≥**72 px**; slides 18–19 and catalog composition slide verify without text extending past rects. |

### Domain 5 — Lane/frame bands + footer shell

| | |
|--|--|
| **Files** | `apps/explainers/src/core/diagram/capabilities/layout.ts` (`LANE_TITLE_BAND`, `FRAME_*`), `apps/explainers/src/index.css` (`.ex-stage-footer` padding), optionally `decks-flow/velo-deep-dive.flow` lane height |
| **Problem** | Fixed 28/48 px frame/lane bands ignore scale-aware title width; footer `padding-bottom: max(64px, 6vh)` (`index.css:739`) still competes with diagram on 720–760 px viewports after hero trim. |
| **Acceptance** | `resolveLaneLayout` title band derives from `estimateTextWidth(title, 14, bold)` with floor 28; footer bottom padding reduced to `max(48px, 5vh)` without lede/subtitle overlap; Playwright snapshot at 1280×720 shows ≥10% diagram height gain vs Round 1 baseline. |

**Conflict note:** Domains 1 and 5 both touch `layout.ts` — run **sequentially** or assign one worker to both band + resolver work. Domains 2–4 are disjoint from each other.

---

## Suggested verification commands

```bash
npm --prefix apps/explainers test -- \
  src/core/diagram/capabilities/layout.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/flow/dev-tools/verify-geometry.test.ts \
  src/flow/sdk/generic/topology.test.ts

npm --prefix apps/explainers run flow-verifier -- \
  decks-flow/cellular-internals.flow \
  decks-flow/segment-pools.flow \
  decks-flow/flow-sdk-examples.flow \
  decks-flow/sdk-diagram-catalog.flow
```

Adjust script names to match local Makefile wrappers if needed.
