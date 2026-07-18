<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flow-Backed Explainers Design

**Date:** 2026-07-18  
**Status:** Approved  
**Host:** `apps/explainers` (legacy SPA shell)  
**Authoring:** `.flow` only  

## Goal

Authors write explainer decks as `.flow` files. The real `@aiperf/flow-compiler` pipeline compiles them into a stable **DeckPackage** artifact. The legacy `apps/explainers` shell loads those packages and plays them with **full animation** and **voiced narration**, matching today’s SPA behavior and visual parity.

**Done means all eight current `DECK_REGISTRY` decks are `.flow`-backed, animated, voiced, and free of React `MentalModel` / hand-authored `content.ts` on the registry path.** There is **no** MentalModel escape hatch.

## Locked decisions

| Decision | Choice |
|---|---|
| Parity target | Legacy `apps/explainers` SPA (pixel + behavior) |
| Runtime host | Keep `ExplainerShell`, hub, speech, storage, routes |
| Diagrams | Scene IR only (`render: @scene { ... }` + timeline) |
| Escape hatch | **None** — no `@mental_model`, no React MentalModel at runtime |
| Voice | Legacy Web Speech path (voice picker, word highlight, auto-advance) |
| Animation | Required: every diagram slide has a Flow timeline |
| Scope | All 8 registry decks |

## Current registry (must remain route/id stable)

1. `rust-architecture` → `/rust-architecture`
2. `rust-architecture-atlas` → `/rust-architecture-atlas`
3. `segment-pools` → `/segment-pools`
4. `slurm-velo` → `/slurm-velo`
5. `velo-deep-dive` → `/velo-deep-dive`
6. `cellular-internals` → `/cellular-internals`
7. `cellular-algorithms` → `/cellular-algorithms`
8. `dynosim` → `/dynosim`

Bookmarks, hub cards, and tests that key off these ids/routes must keep working.

## Architecture

```text
decks/*.flow
    │
    ▼
@aiperf/flow-language  (explainer + slide + @scene + timeline)
    │
    ▼
@aiperf/flow-compiler  (parse → symbols → link → validate → lower → pack)
    │
    ▼
DeckPackage artifact (JSON or generated TS module)
    │
    ▼
apps/explainers adapter  packageToDeckDefinition(pkg)
    │
    ├─► DeckDefinition fields (hub, slides text, css tokens, routes)
    ├─► MentalModel slot → SceneRenderer(sceneIr, timeline, clock)
    └─► ExplainerShell (unchanged chrome + speech + timed advance)
```

**Ownership**

- **Authoring:** `.flow` source under a dedicated explainers decks tree (e.g. `apps/aiperf-flow/explainers/decks/` or `apps/explainers/decks-flow/`). One file (or package entry) per deck.
- **Compile:** real Flow compiler only. Delete `apps/aiperf-flow/scripts/compile-explainer-flows.mjs` and the hand-maintained `compiled-decks.ts` regex path once the real pipeline is green.
- **Runtime:** `apps/explainers` never parses `.flow` at runtime; it only loads DeckPackage artifacts.
- **Diagrams:** port every React `MentalModel.tsx` into animated `@scene` + timeline IR.

## Data model — DeckPackage

```ts
type DeckPackage = {
  schemaVersion: 1;
  id: string;
  route: string;
  topic: string;
  storagePrefix: string;
  classPrefix: string;
  eyebrowLabel: string;
  startGateTitle: string;
  hub: {
    title: string;
    highlight: string;
    description: string;
  };
  /** Optional deck-scoped CSS that styles ExplainerShell chrome only — not diagrams. */
  css?: string;
  /** Optional end card as Scene IR (preferred) or structured final-card IR. */
  finalCard?: { kind: "scene"; scene: SceneIr };
  slides: SlidePackage[];
  glossary: { word: string; meaning: string }[];
};

type SlidePackage = {
  id: string;
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: string[];
  caption: string;
  /** Required when the legacy deck showed a MentalModel for that slide. */
  render?: { kind: "scene"; scene: SceneIr };
};
```

`SceneIr` is the existing Flow scene IR (roots, capabilities, theme roles, **timeline cues**). No alternate render kinds.

### Authoring surface

```flow
explainer "Rust Architecture" {
  id: "rust-architecture"
  route: "/rust-architecture"
  topic: "architecture"
  storagePrefix: "rust-arch-explainer"
  classPrefix: "rust-arch"
  eyebrowLabel: "RUST ARCHITECTURE"
  startGateTitle: "Rust architecture walkthrough"

  hub: {
    title: "from scratch"
    highlight: "Rust architecture"
    description: "Narrated walkthrough of the native workspace..."
  }

  slide "Product shell" {
    eyebrow: "Product shell"
    title: "One binary is both CLI and engine"
    lede: "..."
    narration: "AIPerf ships as one native aiperf binary..."
    term: { word: "aiperf-cli", meaning: "..." }
    points: [ "...", "...", "..." ]
    caption: "..."

    render: @scene {
      roots: [ /* boxes, labels, arrows */ ]
      timeline: [
        /* enter / draw / emphasis cues — required for diagram slides */
      ]
    }
  }
}
```

Forbidden in source and IR:

- `@mental_model(...)`
- `render.kind !== "scene"`
- Runtime imports of deck-local `MentalModel.tsx` from the registry path

## Compiler & language

1. Integrate `explainer` into `parseDocument` (grammar already exists under `packages/language/src/grammar/explainer.ts`; wire it into the main document parser if not already).
2. Symbol/link/validate: treat explainer decks as top-level documents; validate uniqueness of `id`/`route` across a multi-file build set.
3. Lower:
   - Deck metadata → DeckPackage fields
   - Each slide → `SlidePackage`
   - `@scene` → Scene IR via existing scene lowering (including timeline)
4. Pack: emit DeckPackage JSON (and optionally a typed TS module wrapper for Vite).
5. Schema: add Zod (or equivalent) `DeckPackage` / `SlidePackage` schemas in `@aiperf/flow-schema` with `schemaVersion: 1` and unknown-field rejection.
6. Validation fail-closed:
   - empty `title` or `narration`
   - duplicate ids/routes in a build
   - invalid scene / unknown capability
   - diagram slides missing `timeline` when `render` is present (or when a compile flag `--require-animated-diagrams` is on — default **on** for explainers builds)

## Animation (required)

- Every slide that has `render: @scene` **must** include a non-empty `timeline` that drives enter/draw/emphasis using Flow motion roles (`motion.enter`, `motion.draw`, `motion.emphasis`, `motion.stagger`, `motion.easing`).
- Legacy SVG/CSS motion (e.g. `rust-arch-motion` paths) is rewritten as timeline cues + scene nodes — not left as React/CSS keyframes on diagram content.
- `SceneRenderer` in `apps/explainers`:
  - Plays timeline from slide start
  - Restarts on Back / pill / keyboard revisit (same as today’s shell restart semantics)
  - Honors `prefers-reduced-motion` (skip or collapse motion; still show final frame; narration still runs)

## Voice (required)

- Keep `ExplainerShell` + `useTimedSlideshow` + `narration.ts` Web Speech path.
- `.flow` `narration:` is the single source; compile copies into `SlidePackage.narration`.
- Behavior preserved:
  - play with or without audio
  - voice picker / persistent voice URI
  - synchronized word highlighting / subtitles
  - auto-advance on narration complete
  - stop on last slide
  - Back restarts narration + diagram timeline
- Kokoro / aiperf-flow narrator backends are **optional later**; not required for done. Parity is legacy voiced slideshow behavior.

## Runtime adapter (`apps/explainers`)

```ts
function packageToDeckDefinition(pkg: DeckPackage): DeckDefinition
```

- Maps package fields onto existing `DeckDefinition`.
- `MentalModel` becomes a thin wrapper that selects `pkg.slides[i].render?.scene` and mounts `SceneRenderer`.
- `FinalCard` mounts `SceneRenderer` when `finalCard` is present.
- `css` may still style shell chrome (layout wrappers); diagram pixels come from Scene IR + theme.
- `DECK_REGISTRY` imports compiled packages only (no `content.ts`, no `MentalModel.tsx` imports).

## Error handling

| Stage | Behavior |
|---|---|
| Compile | Fail with diagnostics; no package emitted |
| Build registry | Fail if any of the eight required decks missing |
| Runtime corrupt package | Hub shows error; route fails closed (no blank silent deck) |
| Missing scene on a formerly-diagram slide | Treat as content bug; caught by visual regression / slide-count + render-presence tests |

## Migration

### Phase 1 — Pipeline
1. DeckPackage schema + tests
2. Compiler explainer lowering + golden fixture (`rust-architecture` minimal)
3. `SceneRenderer` + `packageToDeckDefinition` in `apps/explainers`
4. Build script: compile all deck `.flow` → packages consumed by Vite

### Phase 2 — Port all eight decks
For each deck:

1. Move slide text from `content.ts` into `.flow`
2. Rebuild each MentalModel frame as `@scene` + timeline (viewport ~700×400, mobile via shell)
3. Port FinalCard if present as scene or structured end card
4. Wire package into registry; remove old TS content/MentalModel from registry path
5. Visual + voice smoke vs legacy

Order (suggested): `rust-architecture` → `slurm-velo` → `dynosim` → `segment-pools` → `velo-deep-dive` → `cellular-internals` → `cellular-algorithms` → `rust-architecture-atlas`.

### Phase 3 — Cleanup
- Delete unused `MentalModel.tsx` / `content.ts` / deck `styles.ts` diagram rules once unused
- Delete regex compile script and `compiled-decks.ts` generation path
- CI: registry is package-only; compile check on all eight `.flow` files

## Testing / done bar

**Unit**

- Schema round-trip for DeckPackage
- Compiler: each deck `.flow` compiles; unique ids/routes; narration non-empty; timelines present on diagram slides

**Integration**

- `validateDeckRegistry` still passes on package-backed registry
- Slide counts match legacy (or documented intentional deltas)
- Adapter produces working `DeckDefinition` for ExplainerShell

**E2E / visual**

- Playwright (or existing screenshot harness) golden frames for representative slides of all eight decks vs pre-migration baselines
- Narration path: speech API mocked or exercised; auto-advance still fires
- Reduced-motion: final frame visible; advance still works

**Done checklist**

- [ ] All 8 decks authored only as `.flow`
- [ ] Real compiler produces DeckPackages (no regex compiler)
- [ ] Registry imports packages only
- [ ] No React MentalModel on registry path
- [ ] Diagram slides animated via timeline
- [ ] Voiced narration works through ExplainerShell
- [ ] Routes/ids unchanged
- [ ] Visual parity gates green (or accepted diffs documented)

## Out of scope

- Replacing `ExplainerShell` with aiperf-flow preview chrome
- Moving the hub into aiperf-flow
- Requiring Kokoro / immersive cinematic host for explainers
- Adding new decks beyond the eight (e.g. mock-server from canvas-ports design) — follow-on
- Arbitrary CSS/scripts inside `.flow` diagram bodies outside Scene IR + theme roles

## Relationship to other specs

- Supersedes the “hybrid MentalModel escape hatch” direction discussed during brainstorming; that path is rejected.
- Complements `2026-07-17-explainer-canvas-ports-design.md` (content already in registry) by changing **authoring/runtime source** to Flow.
- Does not replace cinematic immersive preview designs; explainers remain the narrated slideshow product surface.
- Voice completion behavior stays with legacy `useTimedSlideshow`; aiperf-flow voice-completion design applies to FlowApp scenes, not this shell, unless later unified.
