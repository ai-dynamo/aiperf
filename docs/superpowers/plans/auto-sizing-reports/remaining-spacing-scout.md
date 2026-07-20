# Remaining Spacing Scout — Post Auto-Sizing

**Date:** 2026-07-20  
**Scope:** Read-only reconnaissance of remaining diagram spacing/crowding in `apps/explainers` after the diagram node auto-sizing work (Tasks 1–3 + final-review gap fixes).  
**Thoroughness:** Medium.

---

## Executive summary

Auto-sizing landed the shared `SCENE_TEXT_SCALE` (0.9), bottom-up resolution in `resolve-scene.ts`, intrinsic resolvers for chip/panel/note/header/text, and verifier parity in TypeScript. **Most remaining crowding falls into four buckets:**

1. **Chrome band/preset defaults** still sized for pre-scale nominal font boxes; several presets sit at or below the arithmetic minimum for their text bands.
2. **`resolvePanelLayout` omits the subtitle band**, so three-line `sdk.card` nodes can grow for title+detail yet still clip subtitle text.
3. **Callout, legend, and diagram-node capabilities have no intrinsic layout hooks** — they remain identity-sized from authored/factory defaults.
4. **Stage shell CSS + deck-authored geometry** consume vertical space independently of SVG auto-sizing; catalog and cellular decks still author tight boxes and occasional overlaps.

---

## 1. SDK chrome defaults still crowding content

Constants are **triplicated** across `text-metrics.ts`, `capabilities/chrome.ts`, and `flow/sdk/generic/chrome.ts` (generic copy at lines 56–87). Layout resolvers import from `text-metrics.ts`; SDK factories use local copies that can drift.

### Band heights vs 0.9 text scale

| Constant | Location | Value | Issue relative to 0.9 scale |
|----------|----------|-------|-----------------------------|
| `TITLE_HEIGHT` | `text-metrics.ts:13`, `generic/chrome.ts:57` | 22 | Bold 14px → scaled ~12.6px; 22px band OK. Bold 13px header → ~11.7px; OK. |
| `DETAIL_HEIGHT` | `text-metrics.ts:14`, `generic/chrome.ts:58` | 20 | Normal 11.5px → ~10.4px; OK with small margin. |
| `SUBTITLE_HEIGHT` | `generic/chrome.ts:59` only | 16 | Renderer uses `fontSize: 10` (`capabilities/chrome.ts:304`); scaled ~9px. Band OK but **not included in `resolvePanelLayout` height** (see §2). |
| `HEADER_TEXT_HEIGHT` | `generic/chrome.ts:60` | 24 | Unused in layout path; header resolver uses `TITLE_HEIGHT` + `DETAIL_HEIGHT` instead. |
| `STEPPER_CHIP_HEIGHT` | `text-metrics.ts:17` | 26 | Matches chip default height; OK for 11px bold labels. |
| `LEGEND_ROW_HEIGHT` | `generic/chrome.ts:73` | 20 | Row text at `fontSize: 11` (`generic/chrome.ts:1143`) without scale-aware width sizing; long entry labels clip horizontally. |
| `LANE_TITLE_BAND` | `capabilities/layout.ts:31` | 28 | Lane title band; tight if lane labels use 14px bold. |
| `FRAME_TITLE_BAND` / `FRAME_DETAIL_BAND` | `capabilities/layout.ts:36-37` | 28 / 48 | Frame chrome; not re-derived from scaled metrics. |

### Default geometry presets (still tight)

```56:87:apps/explainers/src/flow/sdk/generic/chrome.ts
const INSET = 8;
const TITLE_HEIGHT = 22;
const DETAIL_HEIGHT = 20;
const SUBTITLE_HEIGHT = 16;
// ...
const HEADER_DEFAULT_GEOMETRY = { x: 18, y: 16, width: 664, height: 44 } as const;
const PANEL_DEFAULT_GEOMETRY = { width: 160, height: 64 } as const;
const CHIP_DEFAULT_GEOMETRY = { width: 84, height: 26 } as const;
const NOTE_DEFAULT_GEOMETRY = { width: 160, height: 40 } as const;
const LABEL_DEFAULT_GEOMETRY = { width: 120, height: 16 } as const;
const CALLOUT_DEFAULT_GEOMETRY = { width: 140, height: 40 } as const;
// ...
const CARD_SIZE_PRESETS = {
  compact: { width: 150, height: 80 },
  standard: { width: 190, height: 80 },
  wide: { width: 250, height: 80 },
};
```

| Preset | Authored size | Minimum content (title + detail, no subtitle) | Verdict |
|--------|---------------|-----------------------------------------------|---------|
| `HEADER_DEFAULT_GEOMETRY.height = 44` | 44 | `INSET*2 + TITLE + DETAIL + 4` = 16+22+24 = **62** | Default below resolver minimum; auto-grow helps **only when resolution runs** and caption present. |
| `PANEL_DEFAULT_GEOMETRY.height = 64` | 64 | **62** for two bands | Barely sufficient; no room for wrapped detail. |
| `CARD_SIZE_PRESETS.*.height = 80` | 80 | Three bands: **78** (`16+22+2+20+2+16`) | **1–2px margin**; subtitle stack has zero slack. |
| `NOTE_DEFAULT_GEOMETRY.height = 40` | 40 | Note text treated as title → min **38** (`16+22`) | Tight; long notes rely on width growth only. |
| `LABEL_DEFAULT_GEOMETRY.height = 16` | 16 | Default label `fontSize: 11` (`generic/chrome.ts:977`); scaled 14px default text → **12.6px** | **Height smaller than scaled 14px default** used by `resolveTextLayout`. |
| `CHIP_DEFAULT_GEOMETRY` | 84×26 | OK for short labels; `FEATURE-GATED` etc. need width growth (chip resolver handles width). | |
| `CALLOUT_DEFAULT_GEOMETRY` | 140×40 | Text at `fontSize: 12` centered (`generic/chrome.ts:1254-1255`); no intrinsic growth | Long callout text clips. |

### Layout gaps (managed containers)

| Default gap | File:line | Notes |
|-------------|-----------|-------|
| Stack/grid/rail `gap: 12` | `generic/layout.ts:191,253,302` | Reasonable; rails with full-bleed children still overflow authored width (see `catalog.test.ts:805`). |
| Pipeline `PIPELINE_DEFAULT_GAP = 24`, node **96×56** | `generic/topology.ts:496-498` | Stage boxes smaller than panel/card presets; long stage labels overflow unless panels inside pipeline auto-grow. |
| Inset `gap: 8` default | `generic/layout.ts:479` | Compact padding demo (`sdk-generic-catalog.flow:223`) uses 90×80 inset frames. |
| `DEFAULT_GAP = 8` | `capabilities/layout.ts:34` | Managed stack/grid fallback. |
| `LANE_INSET = 10` | `capabilities/layout.ts:32` | Swimlane/lane chrome; separate from `INSET = 8`. |

### Diagram catalog defaults (separate from generic chrome)

```124:129:apps/explainers/src/flow/sdk/diagram/catalog.ts
function geometry(props: Props, spec: DiagramSpec): GeometryIr {
  // ...
  width: numberProp(props, "width", spec.width ?? 144),
  height: numberProp(props, "height", spec.height ?? 82),
```

Title/detail child targets use **fixed 22px title band** and glyph offset (`catalog.ts:334`) without scale-aware sizing. Deck variant rows systematically override to **150×62** (`sdk-diagram-catalog.flow` passim), below the 82px catalog fallback and tight for title+detail+gly ph layout.

---

## 2. Layout resolver completeness and remaining overflow paths

### Registered and complete (post gap-fix)

```1041:1058:apps/explainers/src/core/diagram/capabilities/layout.ts
export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  // ...
  { capabilityId: "core.chip", resolveLayout: resolveChipLayout },
  { capabilityId: "core.panel", resolveLayout: resolvePanelLayout },
  { capabilityId: "core.note", resolveLayout: resolvePanelLayout },
  { capabilityId: "core.header", resolveLayout: resolveHeaderLayout },
  { capabilityId: "core.text", resolveLayout: resolveTextLayout },
  // ...
];
```

Bottom-up resolution is wired in `resolve-scene.ts:203-228` and consumed by `SceneRenderer.tsx:4076`.

| Resolver | Registered | Status |
|----------|------------|--------|
| `resolveChipLayout` | `core.chip` | Complete for label width; respects `clipsOverflow`. |
| `resolvePanelLayout` | `core.panel`, `core.note` | **Incomplete for subtitle**; note `props.text` maps to title (OK per gap fix). |
| `resolveHeaderLayout` | `core.header` | Complete for title+caption. |
| `resolveTextLayout` | `core.text` | Complete for `node.text` + `fontSize`. |
| `resolveStepperLayout` | `core.stepper` | Complete (scale-aware chip widths). |
| `resolveRailLayout` | `layout.rail` | Expands to intrinsic child main-axis sum; equal slot assignment can still compress visual weight when parent `fixedWidth`. |

### Remaining overflow paths

#### A. Card subtitle stacking (high impact)

`resolvePanelLayout` height formula (`layout.ts:377-380`):

```377:380:apps/explainers/src/core/diagram/capabilities/layout.ts
  const contentHeight =
    INSET * 2 +
    (title.length > 0 ? TITLE_HEIGHT : 0) +
    (detail.length > 0 ? DETAIL_HEIGHT + 4 : 0);
```

**No `subtitle` term**, yet renderer paints subtitle (`capabilities/chrome.ts:296-308`) and SDK card factory stacks three bands (`generic/chrome.ts:686-733`).

**Concrete deck failure:** `cellular-internals.flow:302` — `sdk.Card(id = "start", … height = 56)` with title+detail+subtitle. Resolver grows to ~62px for two bands; subtitle band starts at y≈56 inside the box and extends to ~72 → **vertical clip**.

#### B. Callout (`core.callout`)

- Factory: fixed `CALLOUT_DEFAULT_GEOMETRY` 140×40, text fills box (`generic/chrome.ts:1206-1258`).
- **No** `resolveCalloutLayout` in `LAYOUT_CAPABILITIES`; identity layout only.
- Stem geometry is separate; label overflow is the main issue.

#### C. Legend (`sdk.legend` → `core.group`)

- Factory computes height from rows (`generic/chrome.ts:1167-1172`) but **entry label width is not measured**; `buildText` rows use fixed `width` (`generic/chrome.ts:1137-1141`).
- **No** legend-specific resolver; long entry labels clip inside the authored/default 180px width.

#### D. Pipeline stage boxes

- `sdk.pipeline` places child nodes at `PIPELINE_DEFAULT_NODE_WIDTH/HEIGHT` (96×56) when slot geometry absent (`topology.ts:496-577`).
- Child panels/cards inside pipeline slots can auto-grow **only if** resolution runs on expanded IR; authored pipeline height may still clip tall children unless container uses managed layout with vertical expansion.

#### E. Diagram nodes (`diagram.*`)

- Emit `capabilityId: diagram.${category}` (`diagram/catalog.ts:187,466`).
- **Not** in `LAYOUT_CAPABILITIES`; identity layout preserves authored 150×62 variant boxes regardless of title string length.
- Semantic chrome uses diagram-specific title/detail placement (`capabilities/chrome.ts:248-290`) with fixed band heights (22 / 20).

#### F. Presentation chrome (`code-block`, `quote`, `avatar`, `icon-label`)

- `hasNativeSemanticChrome` true (`capabilities/chrome.ts:96-102`) but **no** intrinsic layout; identity bounds only.

#### G. Clipped leaves (partially fixed)

Chip/panel/note/header/text respect `clipsOverflow` (`layout.ts:328-329,359-360,398-399,432-433`). **Callout/legend/diagram nodes** do not have clip-aware resolvers because they have no resolvers at all.

#### H. Verifier parity gap

`scripts/flow-verifier/geometry.mjs` still uses authored geometry only (noted in `task-3-report.md:18-20`). TypeScript verifier has parity; Node deck verifier may miss resolved overflow.

---

## 3. Stage/shell CSS crowding

Cinematic presentation (`ExplainerShell`) uses a **hero + footer grid**; diagram SVG lives in `.ex-stage-hero`, copy/subtitles in `.ex-stage-footer`.

### Stage hero padding (eats vertical diagram room)

```1245:1280:apps/explainers/src/index.css
.ex-stage-hero.ex-content-card__diagram {
  padding: clamp(54px, 7vh, 76px) clamp(24px, 4vw, 72px) 8px;
  /* ... */
}
.ex-shell--present .ex-stage-hero.ex-content-card__diagram {
  padding: clamp(34px, 4vh, 48px) clamp(18px, 3vw, 48px) 6px;
}
```

- **Top padding alone:** 34–76px before SVG content, plus progress chrome at `top: 54px` (`.ex-chrome--progress`, `index.css:1328-1332`).
- **Bottom padding on hero:** only 6–8px; most bottom competition comes from footer.

### Footer + subtitles (compete with diagram height)

```730:778:apps/explainers/src/index.css
.ex-stage-footer {
  gap: clamp(8px, 1.2vh, 12px);
  padding: 0 clamp(18px, 4vw, 72px) max(64px, 6vh);
}
.ex-subtitles {
  padding: 9px 10px;
}
.ex-subtitles__text {
  font-size: clamp(16px, 1.7vw, 24px);
  line-height: 1.35;
}
```

- Footer bottom padding **`max(64px, 6vh)`** ≈ 64–77px on typical viewports.
- Subtitle text up to **24px** with label row (`ex-subtitles__label` 10px + 8px margin) consumes **~60–80px** when karaoke row visible (`ExplainerShell.subtitles.test.tsx` confirms footer placement).
- Slide title/lede in `.ex-stage-copy` add **`clamp(28px, 3.5vw, 50px)`** title (`index.css:1296-1308`) below the diagram grid row.
- **Net effect:** on 760px-height viewports, `@media (max-height: 760px)` still leaves `padding-bottom: 64px` on footer (`index.css:1490-1492`) while shrinking hero padding only modestly.

### Non-cinematic diagram cap

```642:646:apps/explainers/src/index.css
.ex-content-card__diagram svg {
  max-height: min(68vh, 650px);
}
```

Hub/split layout caps SVG independently of scene auto-sizing.

### Chrome-hidden mode

```1400:1402:apps/explainers/src/index.css
.ex-shell--chrome-hidden .ex-stage-hero.ex-content-card__diagram {
  padding-top: 26px;
}
```

Presentation mode with hidden chrome gains diagram room; default present mode remains tight vertically.

---

## 4. Deck-authored crowding hotspots

### `cellular-internals.flow`

| Location | Issue |
|----------|-------|
| `297-300` | Four **`sdk.Card` 80×70** with title+detail+subtitle (`c0`–`c3`); width tight for `"register"` detail at 11.5px. |
| `302` | **`start` card 230×56** with three text bands — subtitle overflow after partial auto-grow (§2A). |
| `345-347` | Policy cards **190×70** ×3 stacked with 85px vertical pitch — edges between `pol1`/`pol2` have **5px net gap** between 70px-tall boxes at y=105/190/275. |
| `91-95` | Cell slice cards **130×70** with long subtitle `"single source of truth"` / `"same request"` — width-bound. |
| Passim | **`sdk.Header` 664×44** on every slide consumes ~12% of nominal 360px diagram viewport before scene content. |

### `sdk-diagram-catalog.flow`

| Pattern | Issue |
|---------|-------|
| Variant rows **`150×62`** at y≈292 on most slides (e.g. lines 100-102, 423-425) | Two-line title+detail + glyph in **62px** height; long titles (`"Backpressured"`, `"conversation-ownership"`) overflow width without diagram intrinsic layout. |
| Hero nodes **140×76** (e.g. lines 69-71) | Better, but still fixed. |
| **`sdk.Label` chapter notes** at y=300, height 24 | Sits below variant row at y=292+62=354 on some slides — check overlap on shorter viewports. |

### `sdk-generic-catalog.flow`

| Location | Issue |
|----------|-------|
| `800-803` | Three **`sdk.Gauge` 80×58** abreast (420–590px); labels `"idle"`, `"nominal"`, `"saturated"` dense; hero gauge 180×110 above. |
| `854-857` | **`sdk.Semaphore` 90×30** variants; label **`sem-vl` at x=540, width=120** overlaps sem-v3 at x=430+90=520 (`857-858`). |
| `532` | **`sdk.TagList` 52×48** — extremely small facet demo. |
| `223-224` | **`sdk.Inset` 90×80** frames with badge children — demonstrates tight inset, not overflow bug. |
| `905-908` | **`sdk.Toolbar` 220×46** with badge + caption in v2 — horizontal crowding. |
| Passim | **`sdk.Header` 664×44** on every specimen slide (same as cellular). |

---

## 5. Top parallel fix domains (minimal file conflicts)

Domains ordered for **parallel workers** with disjoint primary files.

### Domain 1 — Stage/footer CSS rebalance

| | |
|--|--|
| **Files** | `apps/explainers/src/index.css` (optional read of `ExplainerShell.tsx` for grid structure) |
| **Problem** | Cinematic hero top padding + footer bottom padding + large karaoke subtitles consume 120–180px vertical, compressing diagram SVG on laptop viewports despite scene auto-sizing. |
| **Acceptance** | Screenshot or Playwright snapshot of `cellular-internals` slide 6 at 1280×720 and 1280×760: diagram SVG viewBox uses ≥15% more vertical pixels; subtitles remain readable; no overlap between footer title and subtitles. Compare `.ex-stage-hero` computed height before/after. |

### Domain 2 — Pipeline & topology stage defaults

| | |
|--|--|
| **Files** | `apps/explainers/src/flow/sdk/generic/topology.ts`, `apps/explainers/src/flow/sdk/generic/topology.test.ts`, optionally `apps/explainers/decks-flow/flow-sdk-examples.flow` (pipeline slide panels 72×56) |
| **Problem** | `PIPELINE_DEFAULT_NODE_HEIGHT = 56` and gap 24 are below panel/card band minimums; pipeline stage boxes clip before child auto-sizing applies. |
| **Acceptance** | `topology.test.ts` updated for new defaults; `flow-sdk-examples` pipeline slide resolves without child overflow at `resolveScene` for stage labels `"1"`, `"2"`, `"3"`. |

### Domain 3 — Diagram catalog presets + variant deck rows

| | |
|--|--|
| **Files** | `apps/explainers/src/flow/sdk/diagram/catalog.ts`, `apps/explainers/decks-flow/sdk-diagram-catalog.flow` |
| **Problem** | Systematic **150×62** variant geometry and 144×82 defaults do not fit title+gly ph+detail at 0.9 scale; no `diagram.*` intrinsic resolver. |
| **Acceptance** | Bump spec defaults and variant row sizes (e.g. 150×72 or 160×76); run `npm --prefix apps/explainers run verify-decks -- sdk-diagram-catalog` (or deck verifier equivalent) with zero geometry overlap diagnostics on chapter opener slides. |

### Domain 4 — Cellular deck geometry pass

| | |
|--|--|
| **Files** | `apps/explainers/decks-flow/cellular-internals.flow` only |
| **Problem** | Authored 80px-wide cell cards, 56px START banner, and 70px-tall triple-stack cards fight auto-sizing and connector routing. |
| **Acceptance** | Re-verify slides 6–7 (`start` barrier, START policies) at 700×360 scene coordinate space: no text extends past card rects; edge anchors clear of text. Visual screenshot optional. |

### Domain 5 — Core chrome bands, subtitle resolver, callout/legend intrinsic layout

| | |
|--|--|
| **Files** | `apps/explainers/src/core/diagram/text-metrics.ts`, `capabilities/layout.ts`, `capabilities/layout.test.ts`, `capabilities/chrome.ts`, `flow/sdk/generic/chrome.ts` |
| **Problem** | (a) `resolvePanelLayout` ignores `subtitle`; (b) band/preset constants at arithmetic floor; (c) callout/legend lack resolvers; (d) generic chrome duplicates constants. |
| **Acceptance** | New tests: three-line card grows to ≥78px height; long callout/legend entry expands width; `CARD_SIZE_PRESETS.standard.height` ≥ computed minimum + 4px slack; `npm --prefix apps/explainers test -- src/core/diagram/capabilities/layout.test.ts` green. |

**Conflict note:** Domain 5 touches `layout.ts`; Domains 1–4 do not. Domain 3 touches `diagram/catalog.ts` only, not generic chrome. **Do not split Domain 5 across workers** without coordinating on `layout.ts`.

---

## Residual risks

- **Factory vs resolver tests:** Layout tests cover manual node shapes (`layout.test.ts:223-238` note path) but not full SDK factory → `resolveScene` integration for cards with subtitle, callouts, or legends.
- **Node flow verifier:** `geometry.mjs` may report false negatives/positives until it imports the TS layout registry or a generated bundle.
- **Rail full-bleed children:** `catalog.test.ts:805` documents intentional overflow for some rail demos; distinguish product decks from catalog specimens when fixing.

---

## Suggested verification commands

```bash
npm --prefix apps/explainers test -- \
  src/core/diagram/capabilities/layout.test.ts \
  src/core/diagram/resolution/resolve-scene.test.ts \
  src/flow/dev-tools/verify-geometry.test.ts

npm --prefix apps/explainers run verify-decks -- cellular-internals sdk-diagram-catalog sdk-generic-catalog
```

(Adjust `verify-decks` invocation to match project Makefile/script names if wrapped.)
