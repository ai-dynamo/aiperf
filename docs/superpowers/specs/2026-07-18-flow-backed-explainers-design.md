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

**Done means all eight current `DECK_REGISTRY` decks are `.flow`-backed (embedded `@scene` parse → DeckPackage), packages-only on the registry path, animated, voiced, and free of React `MentalModel` / hand-authored `content.ts` / dual-load fallback.** There is **no** MentalModel escape hatch.

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

- **Authoring:** Exactly **one `.flow` file per deck**. Path: `apps/explainers/decks-flow/<deck-id>.flow`. That single file must contain the full deck: hub/metadata, every slide’s text + narration, every diagram as inline `render: @scene { … }` with timelines, and optional `finalCard` scene. **No** companion fragment trees (`decks-flow/scenes/…`, `.flowfrag`, per-slide sidecar files, or MentalModel React modules on the registry path).
- **Expressiveness:** The `explainer` + nested `@scene` surface must be rich enough to replace today’s React MentalModels (nested boxes/labels/arrows/paths, multi-cue timelines, theme roles). If a diagram feature exists in a legacy MentalModel, it must be expressible inside that one `.flow` file — extend grammar/lowering/SceneRenderer, do not split the deck across files.
- **Compile:** real Flow compiler only — explainer documents go through embedded scene parse (`parsePackageSceneBody` / `parseNativeEmbeddedScene`) and `lowerExplainerScene` / `compileExplainerSource`. Delete `apps/aiperf-flow/scripts/compile-explainer-flows.mjs` and the hand-maintained `compiled-decks.ts` regex path once the real pipeline is green.
- **Runtime:** `apps/explainers` never parses `.flow` at runtime; it only loads DeckPackage artifacts. Registry is **packages-only** (no legacy dual-load).
- **Diagrams:** port every React `MentalModel.tsx` into animated `@scene` + timeline IR **inside the same deck `.flow`**.

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

## Language surface

Authoring is **one `.flow` file per deck** at `apps/explainers/decks-flow/<deck-id>.flow`. That file is the entire deck: hub/metadata, every slide’s copy + narration, every diagram, and optional `finalCard`. The compiler does not assemble decks from companion trees.

**Packages-only runtime:** `DECK_REGISTRY` loads **only** DeckPackage artifacts from `apps/explainers/src/decks-generated/` via `packageToDeckDefinition`. There is no dual-load / legacy React deck fallback on the registry path. Compile emits packages; the shell never parses `.flow` at runtime.

### Slide diagrams — embedded `@scene` parse

Diagram slides use only `render: @scene { … }`. Bodies are captured and parsed through the shared **embedded scene** path in `@aiperf/flow-language` (`embedded-scene.ts` + `parseNativeEmbeddedScene`), then lowered by `@aiperf/flow-compiler` (`lowerExplainerScene`). There is no alternate `render` kind and no regex / `Function()` scene parse.

```flow
render: @scene {
  roots: [ /* scene nodes */ ]
  timeline: [ /* non-empty cues */ ]
}
```

Lowering emits `{ kind: "scene"; scene: SceneIr }` into `SlidePackage.render` (and into `finalCard` when authored as a scene).

Two dialects share one capture/lower path:

| Dialect | Body shape | Parse | Lower |
|---|---|---|---|
| **package** (decks-flow — required for the eight registry decks) | `roots: […]`, optional `timeline` / `camera` | `captureEmbeddedScene` → `parsePackageSceneBody` → `package-scene` | `lowerExplainerScene` normalizes to strict `SceneIr` |
| **native** (cinematic / shared scene rules) | `rect` / `text` / `connector` / `timeline` / `camera` statements | `embedded-scene-source` + `parseNativeEmbeddedScene` (same Chevrotain scene rules as cinematic examples) | existing document `lower()` via `lowerExplainerScene` |

Dialect is detected by a leading `roots:` / `timeline:` / `camera:` field; otherwise the body is native. `@theme.*` package style refs stay as strings for runtime theme resolution. Public exports: `captureEmbeddedScene`, `detectEmbeddedSceneForm`, `parsePackageSceneBody`, `parseNativeEmbeddedScene`, `lowerExplainerScene`, `compileExplainerSource`.

### Roots capabilities

Scene `roots` (and nested `children`) are Scene IR nodes. The explainer diagram vocabulary used by `SceneRenderer` and deck ports is:

| Capability | Role |
|---|---|
| `core.rect` | Boxes / panels |
| `core.text` | Labels |
| `core.arrow` | Directed edges (also accepts line/path-shaped arrow nodes) |
| `core.connector` | Links between nodes |

Nodes carry `id`, `capability`, `layout` (or geometry), optional `style` / theme roles (`@theme.…`), optional `text`, and optional nested `children`. Extend grammar/lowering/`SceneRenderer` if a legacy MentalModel feature cannot be expressed with these primitives — do not invent a second render path.

### Timeline cues

Every slide with `render: @scene` **must** carry a non-empty `timeline` array. Cues are finite `{ id, at, duration, target, action }` entries (enter / draw / emphasis and related motion roles). The explainer compile path fail-closes on empty timelines. Runtime plays cues from slide start, restarts on revisit, and honors `prefers-reduced-motion`.

### No `@mental_model` / no registry dual-load

- Source must not use `@mental_model(...)`.
- DeckPackage / schema reject any `render.kind` other than `"scene"` (including `"mental_model"`).
- `DECK_REGISTRY` must not import deck-local `MentalModel.tsx` / `content.ts` modules and must not fall back to legacy deck modules when a package is missing (fail closed / build gate instead).

### Forbidden fragments

Do **not** author or consume:

- `decks-flow/scenes/` (or any parallel scene tree)
- `.flowfrag` / per-slide sidecar `.flow` files
- `#include` / fragment assembly that splits one deck across files
- Hand-maintained React MentalModel modules on the registry path
- Dual-load helpers that prefer package-then-legacy for registry entries

Discard any such trees if they appear in worktrees or ports. One deck → one `.flow` → one DeckPackage → packages-only registry entry.

### Authoring example

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
      roots: [
        {
          id: "shell"
          capability: "core.rect"
          layout: { x: 80, y: 120, width: 540, height: 160 }
          children: [
            {
              id: "label"
              capability: "core.text"
              text: "aiperf binary"
              layout: { x: 100, y: 180, width: 500, height: 40 }
            }
          ]
        }
      ]
      timeline: [
        { id: "enter-shell", at: 0, duration: 400, target: "shell", action: "enter" }
      ]
    }
  }
}
```

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
- Diagram slot selects `pkg.slides[i].render?.scene` and mounts `SceneRenderer`.
- `FinalCard` mounts `SceneRenderer` when `finalCard` is present.
- `css` may still style shell chrome (layout wrappers); diagram pixels come from Scene IR + theme.
- **Packages-only:** `DECK_REGISTRY` imports / loads compiled DeckPackages only (no `content.ts`, no `MentalModel.tsx`, no legacy dual-load fallback).

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
- Remove dual-load (`deckFromPackageOrLegacy` / legacy deck imports) so registry is packages-only
- CI: registry is package-only; compile check on all eight `.flow` files; MentalModel registry import assert hard-fails

## Testing / done bar

**Unit**

- Schema round-trip for DeckPackage
- Embedded scene parse: package-form `roots`/`timeline` via `parsePackageSceneBody`; native form via `parseNativeEmbeddedScene`; dialect detection; lowering to `SceneIr`
- Compiler: each deck `.flow` compiles via `compileExplainerSource` / explainer lowerer; unique ids/routes; narration non-empty; timelines present on diagram slides

**Integration**

- `validateDeckRegistry` passes on a **packages-only** registry (no dual-load)
- Assert script / CI: `deck-registry` does not statically import any deck `MentalModel.tsx`
- Slide counts match legacy (or documented intentional deltas)
- Adapter produces working `DeckDefinition` for ExplainerShell from generated packages only

**E2E / visual**

- Playwright (or existing screenshot harness) golden frames for representative slides of all eight decks vs pre-migration baselines
- Narration path: speech API mocked or exercised; auto-advance still fires
- Reduced-motion: final frame visible; advance still works

**Done checklist**

- [ ] All 8 decks authored only as `.flow` (package-form embedded `@scene`)
- [ ] Real compiler + embedded scene parse produce DeckPackages (no regex / `Function()` scene parse)
- [x] Registry is packages-only (`decks-generated` → `packageToDeckDefinition`; no legacy dual-load)
- [ ] No React MentalModel / `content.ts` on registry path
- [ ] Diagram slides animated via timeline
- [ ] Voiced narration works through ExplainerShell
- [ ] Routes/ids unchanged
- [ ] Visual parity gates green (or accepted diffs documented)

## Tooling (build + gates)

Authoring and runtime stay packages-only: **one** `apps/explainers/decks-flow/<deck-id>.flow` per deck → real compiler (embedded `@scene` parse → `lowerExplainerScene` / `compileExplainerSource`) → `apps/explainers/src/decks-generated/*.package.json` → `packageToDeckDefinition` → `SceneRenderer`. There is **no** MentalModel registry path and **no** dual-load fallback.

| Command | What it does |
|---|---|
| `make build-explainer-packages` | Compiles every `decks-flow/*.flow` via `apps/aiperf-flow` → `npm run build:explainer-packages` (`scripts/build-explainer-packages.mjs`) |
| `make assert-deck-packages` | Requires all eight generated packages; non-empty slide narration; non-empty `scene.timeline` when `render` is present (`apps/explainers/scripts/assert-deck-packages.mjs`) |
| `make assert-no-mentalmodel-registry` | Hard-fails if `deck-registry.ts` transitively imports any `MentalModel.tsx` (`apps/explainers/scripts/assert-no-mentalmodel-registry.mjs`) |
| `make assert-explainer-packages` | Runs build + both asserts |

npm equivalents (from `apps/aiperf-flow`): `build:explainer-packages`, `assert:deck-packages`, `assert:no-mentalmodel-registry`, `assert:explainer-packages`. Explainers package mirrors the two assert scripts under `apps/explainers`.

Agent instruction files (`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/python.mdc`) keep one shared body from `# AIPerf` (preambles may differ). Design record: this spec + `docs/superpowers/plans/2026-07-18-flow-backed-explainers.md`.

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
