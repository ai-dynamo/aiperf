# Native Flow Explainers System

**Date:** 2026-07-18  
**Status:** Design (awaiting approval)  
**Scope:** Port explainers system to native Flow language; full integration with narrator, themes, immersive preview, causal replay  
**Deliverables:** Language primitives, runtime engine, 4 ported decks + 3-5 new flow-specific topics, comprehensive test suite

---

## Executive Summary

Transform the AIPerf explainers system (currently a standalone React SPA) into a **first-class Flow language feature** where explainer decks are authored in `.flow` files and rendered as native Flow scenes. This is meta-didactic: teaching Flow visualization through Flow itself.

**Key outcomes:**
- Explainers become navigable, themed, narrated Flow documents
- Tight integration with narrator, themes, immersive preview, causal replay
- 4 existing explainer topics ported (rust-architecture, slurm-velo, dynosim, aiperf-flow)
- 3-5 new flow-specific explainer topics
- Full language support: `explainer` as first-class primitive

---

## Problem & Context

### Current State
- Explainers app is a **separate React SPA** (apps/explainers)
- Content is hardcoded React components + slide definitions
- No connection to Flow visualization capabilities
- Tight coupling to React/slideshow UI patterns

### Vision
- **Explainers as Flow documents** — authored in `.flow` language, compiled to IR, rendered as scenes
- **Native integration** — explainer scenes are first-class Flow scenes, not special-cased React components
- **Educational meta-loop** — teaching Flow visualization through Flow itself
- **Full system integration** — narrator narrates slides, themes style diagrams, immersive mode shows explainer scenes cinematically

---

## Language Design

### `explainer` Block (Top-Level Language Construct)

```flow
explainer "Flow IR Fundamentals" {
  id: "flow-ir-basics"
  route: "/explainers/flow-ir-basics"
  topic: "introduction"
  
  slide "What is Flow IR?" {
    eyebrow: "Foundations"
    title: "The Flow Intermediate Representation"
    lede: "A declarative format for describing AI visualization scenes"
    
    narration: "Flow IR is a JSON schema that describes interactive cinematic visualizations of AI system behavior. Unlike imperative rendering, Flow IR expresses what to show, not how to show it. This separation enables backend neutrality: the same IR renders on canvas, SVG, or semantic HTML."
    
    term: {
      word: "Intermediate Representation"
      meaning: "A structured format that captures intent without prescribing implementation details"
    }
    
    points: [
      "Scene-graph structured visualization language",
      "Backend-neutral: canvas, SVG, semantic rendering",
      "Deterministic, reproducible output",
      "Composable: scenes nest, inherit, override"
    ]
    
    caption: "Flow IR bridges authoring intent and runtime rendering across all backends"
    
    render: @scene {
      // Native Flow scene showing IR structure diagram
      // This is embedded directly; renders through Flow evaluation
      roots: [...]
    }
  }
  
  slide "Scene Evaluation" {
    eyebrow: "Runtime"
    title: "How Scenes Execute"
    narration: "A Flow scene is evaluated at runtime through a deterministic process..."
    points: [...]
    render: @scene { ... }
  }
  
  // More slides...
}
```

### Language Grammar

**Top-level:**
- `explainer STRING { deck-body }` — declares an explainer deck
- Decks are scoped and registered in a global explainer registry

**Deck body fields:**
- `id: STRING` — unique identifier (used for routing, storage)
- `route: STRING` — URL path for this deck (e.g., `/explainers/flow-ir-basics`)
- `topic: STRING` — category (e.g., "introduction", "architecture", "performance")
- `hub: { title, highlight, description }` — metadata for explainers hub/index page
- `eyebrowLabel: STRING` — breadcrumb label
- `startGateTitle: STRING` — title shown before first slide plays
- `slide BLOCK...` — sequence of slide blocks

**Slide fields:**
- `eyebrow: STRING` — breadcrumb context
- `title: STRING` — slide headline
- `lede: STRING` — subtitle/subheading
- `narration: STRING` — speech text (required; routed to narrator backend)
- `term: { word, meaning }` — optional glossary entry
- `points: [STRING, ...]` — bullet points
- `caption: STRING` — image/diagram caption
- `render: @scene { ... }` — embedded mental model scene (optional)

### Compiler Lowering

**Input:** `.flow` file with `explainer` blocks

**Output:** 
1. **Explainer IR** (new schema type) — structured representation of deck, slides, narration
2. **Scene IRs** — one scene per slide (mental model visualization)
3. **Deck metadata** — registration entry (id, route, topic, hub metadata)

**Validation:**
- Explainer ID uniqueness (no duplicate ids across all decks)
- Route uniqueness
- Narration text non-empty
- @scene blocks are valid Flow scene IR

---

## Runtime Architecture

### Core Components

#### **1. SlideshowController**
Orchestrates slide progression, narrator sync, timing.

```typescript
export interface SlideshowController {
  // State
  currentSlideIndex: number;
  totalSlides: number;
  isPlayingNarration: boolean;
  
  // Navigation
  nextSlide(): Promise<void>;
  prevSlide(): Promise<void>;
  jumpToSlide(index: number): Promise<void>;
  
  // Narrator integration
  startNarrationForSlide(index: number): Promise<void>;
  stopNarration(): void;
  
  // Scene transitions
  transitionTo(slideIndex: number): Promise<void>;
}
```

**Responsibilities:**
- Maintains slide state (current index, narration state, timing)
- Coordinates with narrator backend (start/stop speech, wait for completion)
- Triggers scene transitions with animation
- Handles keyboard/button navigation and skip controls

#### **2. SlideRenderer (React Component)**
Renders a single slide: title, points, scene, narration UI.

```typescript
export interface SlideRendererProps {
  slide: SlideDefinition;
  sceneIr: SceneIr;
  theme: ResolvedTheme;
  isActive: boolean;
  showNarrationUI: boolean;
}

export function SlideRenderer(props: SlideRendererProps) {
  // Render slide layout: eyebrow, title, points, mental model scene, caption
  // Scene is evaluated through standard Flow evaluation (theme-aware)
  // Narration UI shows speaker icon, transcription, timing
}
```

#### **3. ExplainerLayout (React Component)**
Unified shell for all explainer decks.

```typescript
export interface ExplainerLayoutProps {
  deck: ExplainerDefinition;
  slideIndex: number;
  onNavigate: (index: number) => void;
  narrator: NarratorBackend;
  theme: ResolvedTheme;
}

export function ExplainerLayout(props: ExplainerLayoutProps) {
  // Topbar: deck title, progress indicator, theme selector
  // Main: slide renderer
  // Sidebar: slide thumbnails + outline
  // Bottom: narrator controls, next/prev buttons
  // All styled by active theme
}
```

#### **4. ExplainerRegistry**
Registry for all explainer decks (similar to endpoint/transport registries).

```typescript
export interface ExplainerRegistry {
  register(deckDef: ExplainerDefinition): void;
  getDeck(id: string): ExplainerDefinition | undefined;
  getAllDecks(): readonly ExplainerDefinition[];
  getRouteMap(): Map<string, string>; // route -> deck id
}
```

---

### Integration Points

#### **Narrator Integration**

Each slide's `narration` text is:
1. Routed to active narrator backend (Kokoro or browser speech)
2. Slideshow pauses while narration plays
3. On narration completion, advance prompt shown
4. User can skip narration, auto-advance on completion, or manual next/prev

**API:**
```typescript
export interface NarratorBinding {
  speakSlide(slide: SlideDefinition): Promise<void>;
  stopSpeech(): void;
  onNarrationComplete: () => void;
}
```

#### **Theme Integration**

Active theme applies to:
- Slide background, text color, accent colors
- Mental model diagram colors (shape fills, strokes, text)
- UI chrome (topbar, buttons, progress indicator)

**Implementation:**
- Explainer slides are regular Flow scenes; theme colors are CSS variables/Flow theme values
- `ExplainerLayout` passes `theme: ResolvedTheme` to `SlideRenderer`
- Mental model `@scene` blocks reference theme roles (e.g., `fill: theme.ink.primary`)

#### **Immersive Preview Integration**

Explainer scenes support immersive mode:
- Expand mental model scene to full viewport
- Apply cinematic controls (play, speed, causal trace)
- Show explainer title/narration in overlay (not embedded in scene)

**Implementation:**
- Explainer routes registered in preview app
- On immersive mode, explainer scene evaluated with cinematic context
- Narration continues in background; user controls pace

#### **Causal Replay Integration**

Architecture/system-flow explainer slides can embed causal traces:
- Show causality relationships in diagrams
- Highlight signal flow as narration mentions components
- Time-synchronized with narrator

**Implementation:**
- Slide's `@scene` block can reference causal events
- Timeline visualization (from causal-replay) integrated into mental model

---

## Content Structure

### Explainer Topics & Decks

#### **Ported from apps/explainers (4 decks)**

1. **Rust Architecture** (`rust-architecture`)
   - Modules: AIPerf system topology, worker execution, transport layer
   - Slides: ~12-15
   - Mental models: box diagrams showing module relationships, data flow arrows
   - Topics: cellular execution, phase orchestration, metric aggregation

2. **SLURM + Velo** (`slurm-velo`)
   - Modules: Distributed execution, cluster orchestration, cross-host communication
   - Slides: ~10-12
   - Mental models: cluster topology, SLURM rank mapping, Velo message flow
   - Topics: srun allocation, controller/cell model, network protocol

3. **Dynamo Simulation** (`dynosim`)
   - Modules: Discrete-event simulation, replay mechanics, timing models
   - Slides: ~10-12
   - Mental models: event queue, timeline visualization, state transitions
   - Topics: SimClock, deterministic ordering, replay vs. live execution

4. **AIPerf Flow System** (`aiperf-flow-system`)
   - Modules: (replaces generic AIPerf intro) Request lifecycle in Flow context
   - Slides: ~8-10
   - Mental models: request→response journey, Flow scene progression
   - Topics: endpoint binding, transport, metric capture

#### **New flow-specific explainers (3-5 decks)**

1. **Flow IR Fundamentals** (`flow-ir-basics`)
   - Modules: IR schema, scene composition, backend-neutral rendering
   - Slides: ~10-12
   - Mental models: IR structure diagram, scene graph nesting, schema overview
   - Topics: declarative vs. imperative, schema versioning, IR validation

2. **Visualization Capabilities** (`viz-capabilities`)
   - Modules: Core shapes, layout algorithms, composition patterns
   - Slides: ~12-15
   - Mental models: capability taxonomy, shape examples, layout demonstrations
   - Topics: span-map, segment-strip, waterfall, connector routing, text rendering

3. **Scene Evaluation Engine** (`scene-evaluation`)
   - Modules: Evaluation phases, data binding, semantic projection, display lists
   - Slides: ~10-12
   - Mental models: evaluation pipeline, data flow, output structure
   - Topics: determinism, incremental evaluation, display instruction generation

4. **Theme System** (`theme-system`) *(optional, but natural given recent work)*
   - Modules: Theme IR, inheritance, role mapping, application
   - Slides: ~8-10
   - Mental models: theme hierarchy, color palette demonstration, contrast validation
   - Topics: role taxonomy, light/dark variants, backend rendering consistency

5. **Immersive & Causal Replay** (`immersive-causal`) *(optional, advanced topic)*
   - Modules: Cinematic control, causal tracing, forensic visualization
   - Slides: ~8-10
   - Mental models: causal event tree, timeline correlation, highlight propagation
   - Topics: deterministic replay, cause-effect discovery, interactive exploration

---

## Component Architecture

### New Flow Capability: `explainer.slide-deck`

**Purpose:** Registers explainer scenes as navigable deck routes; provides slide state API.

**API (pseudo-schema):**

```flow
capability "explainer.slide-deck" version "1.0.0" {
  // Input: current slide index
  input: {
    slideIndex: number
    deckId: string
  }
  
  // Output: rendered slide scene + UI state
  output: {
    scene: SceneIr
    title: string
    narration: string
    points: string[]
    slideCount: number
    canAdvance: boolean
    canRetreat: boolean
  }
  
  // Exposed state for navigation
  state: {
    onNext: () => void
    onPrev: () => void
    onJumpTo: (index: number) => void
  }
}
```

### Reusable Mental Model Components

These are authored as `.flow` scenes and can be composed into explainer mental models.

**Component library (native Flow scenes):**
- `ArchitectureBox` — labeled box with ports (shows modules)
- `DataFlowArrow` — animated arrow showing data/signal movement
- `GridLayout` — grid positioning for components
- `SequenceTimeline` — time-ordered event visualization
- `ThemePalette` — color role demonstration
- `ContrastMatrix` — role-to-role mapping visualization
- `LegendBlock` — key/legend with color swatches and labels

**Implementation:**
- Each component is a composable Flow symbol/macro
- Takes theme colors as parameters
- Supports optional animation/causal tracing

---

## Testing Strategy

### Unit Tests (Language & Compiler)

- **Parser:** Valid/invalid explainer syntax
- **Compiler:** Lowering explainer → IR (narration text preserved, scene IDs correct)
- **Validation:** Duplicate ID detection, route conflicts, schema compliance

### Runtime Tests

- **SlideshowController:** Navigation (next, prev, jump), narrator sync, state transitions
- **Theme integration:** Explainer slides render with theme colors applied
- **Narrator integration:** Slide narration routed correctly, timing respected
- **Scene rendering:** Each slide's mental model scene evaluates without error

### E2E Tests

- **Full deck playthrough:** All slides navigate correctly, narration plays, transitions smooth
- **Theme switching:** Deck responds to theme changes mid-playthrough
- **Immersive mode:** Explainer scene expands to full viewport, cinematic controls work
- **Causal replay integration:** Architecture deck shows causal trace overlay

### Content Tests

- **Deck schema validation:** All ported + new decks are schema-valid
- **Scene rendering:** Each deck's slides render on all backends (canvas, SVG, semantic)
- **Narration completeness:** All slides have narration text, no stubs or TODOs

---

## Phasing & Task Breakdown

Suitable for **parallel implementer wave** (15-20 Haiku models, 1-2 Opus for architecture/compilation).

### Phase 1: Language & Runtime (6-7 tasks)

1. **Explainer language syntax + parser** (Opus)
   - Design grammar precisely; integrate into Flow language parser
   - Tests: parser accepts/rejects valid/invalid explainer blocks

2. **Compiler lowering** (Opus)
   - Transform explainer AST → ExplainerDefinition + scene IRs
   - Tests: lowering roundtrip, ID/route uniqueness

3. **SlideshowController + state machine** (Haiku)
   - Slide progression, navigation logic, timing
   - Tests: all navigation paths, state consistency

4. **Narrator integration** (Haiku)
   - Route narration to narrator backend, sync timing, handle completion
   - Tests: narration routed correctly, pause/resume, skip logic

5. **Theme integration** (Haiku)
   - Apply theme to explainer layout + scene rendering
   - Tests: theme colors applied, contrast validated, dark/light variants

6. **Immersive preview support** (Haiku)
   - Register explainer routes, cinematic controls, overlay UI
   - Tests: scene expands full viewport, controls functional

7. **Causal replay binding** (Haiku)
   - Embed causal traces in architecture explainer scenes
   - Tests: timeline visualization, highlight sync with narration

### Phase 2: Components & Layouts (4-5 tasks)

8. **ExplainerLayout shell** (Haiku)
   - Topbar (title, progress, theme selector), sidebar (outline), main (slide renderer), bottom (narrator controls)
   - Tests: layout responsive, all controls functional

9. **Mental model diagram primitives** (Haiku)
   - ArchitectureBox, DataFlowArrow, GridLayout, SequenceTimeline as reusable Flow symbols
   - Tests: components compose, theme colors apply, animation works

10. **Glossary & term UI** (Haiku)
    - Render term definition on slide, glossary index
    - Tests: term display, index navigation

11. **Scene transition animations** (Haiku)
    - Fade, slide, or morph between slides
    - Tests: transitions smooth, theme-aware (color palette matching)

12. **Mobile/immersive responsive layout** (Haiku)
    - Stack/reflow for narrow viewports; full-screen immersive mode
    - Tests: responsive breakpoints, immersive fullscreen, touch controls

### Phase 3: Content Porting & Creation (4-5 tasks)

13. **Rust architecture deck** (Haiku + reference implementation)
    - Port 12-15 slides from apps/explainers; use ArchitectureBox + DataFlowArrow components
    - Tests: all slides render, narration intact, theme colors applied

14. **SLURM + Velo deck** (Haiku)
    - Port 10-12 slides; cluster topology + rank mapping visualizations
    - Tests: slides render, causal trace works for distributed execution examples

15. **Dynamo simulation deck** (Haiku)
    - Port 10-12 slides; discrete-event simulation timeline + state transitions
    - Tests: timeline visualization, event ordering correct

16. **AIPerf flow system deck** (Haiku)
    - New deck covering AIPerf in Flow context (replaces generic intro)
    - 8-10 slides; request lifecycle, endpoint binding, transport
    - Tests: slides render, integration with flow-specific topics

17. **Flow IR fundamentals deck** (Haiku)
    - New deck: 10-12 slides on IR schema, scene composition, backend-neutral rendering
    - Use IR structure diagram component
    - Tests: schema diagrams render correctly, concept clarity

18. **Visualization capabilities deck** (Haiku)
    - New deck: 12-15 slides on core shapes, layout, composition patterns
    - Showcase each P0 capability (span-map, segment-strip, waterfall)
    - Tests: capability examples render, diversity of visualizations

### Phase 4: Polish & Integration (3-4 tasks)

19. **E2E test suite** (Haiku)
    - Full playthrough tests for all decks; theme switching, immersive mode, narrator
    - Tests: all decks pass e2e without error

20. **Theme consistency** (Haiku)
    - Audit all decks for consistent color application, contrast, readability
    - Tests: all decks readable in light/dark/reduced-motion modes

21. **Narrator quality & timing** (Haiku)
    - Polish narration audio (if using Kokoro), refine timing (e.g., pause lengths between slides)
    - Tests: narration clarity, pacing feels natural

22. **Performance optimization** (Haiku)
    - Lazy load scene IRs, memoize evaluations, cache theme-rendered scenes
    - Tests: smooth navigation, no jank on slide transitions

---

## Design Decisions

### 1. Language Embedding
**Decision:** `explainer` is a top-level language construct (like `flow` or `capability`).  
**Rationale:** Explainers are not scenes or symbols; they're structural composition of scenes. Top-level makes this clear.

### 2. Mental Model Rendering
**Decision:** Mental models are embedded `@scene` blocks in slide definitions.  
**Rationale:** Keeps content self-contained, enables theme + narrator + immersive integration. Alternative (separate files) would fragment content and complicate routing.

### 3. Narrator Backend Selection
**Decision:** Use Kokoro when available; fallback to browser speech API if not configured.  
**Rationale:** Kokoro provides higher quality narration (matching immersive preview), but browser speech is always available as fallback for accessibility.

### 4. Route Registration
**Decision:** Explainer routes auto-registered on compiler output (explainer IR includes route).  
**Rationale:** Decentralized: each deck declares its own route. Preview app subscribes to all registered routes and builds navigation menu.

### 5. Deck Scoping
**Decision:** Explainer IDs must be globally unique (validated at compile time).  
**Rationale:** Prevents routing collisions; simplifies registry lookups. Enforced in `ExplainerRegistry.register()`.

---

## Success Criteria

### Functional
- ✅ All 4 explainer decks port to native Flow with identical content
- ✅ 3-5 new flow-specific decks authored and deployed
- ✅ Narration plays correctly; slides advance on completion or manual control
- ✅ Theme switching updates explainer colors live
- ✅ Immersive mode works; cinematic controls functional
- ✅ Causal replay traces show in architecture explanations

### Quality
- ✅ 100% of slides render without error
- ✅ >95% E2E test pass rate
- ✅ Narration timing feels natural (no rushed/slow pacing)
- ✅ All decks readable in light/dark/reduced-motion modes
- ✅ Mental model diagrams clear and accurate

### Performance
- ✅ Slide transitions <500ms
- ✅ Narration start latency <200ms
- ✅ No memory leaks on long deck playthrough
- ✅ Mobile immersive mode smooth (60fps)

---

## Open Questions (Resolved by Design)

1. **Language embedding** ✅ — Top-level construct, alongside `flow`, `capability`
2. **Mental models** ✅ — Embedded `@scene` blocks
3. **Narrator backend** ✅ — Kokoro first, browser speech fallback
4. **Route registration** ✅ — Auto-registered, globally unique IDs
5. **Deck composition** ✅ — One `.flow` file per deck (or group related in monolithic file)

---

## Next Steps

1. **User review** — Approve design, suggest refinements
2. **Write implementation plan** (superpowers:writing-plans)
3. **Fan out Phase 1 tasks** to parallel implementers (1-2 Opus + 5-6 Haiku)
4. **Iterate per phase** — review → adjust → next phase

---

## Appendix: Example Explainer Deck (Sketch)

```flow
explainer "Flow IR Fundamentals" {
  id: "flow-ir-basics"
  route: "/explainers/flow-ir-basics"
  topic: "introduction"
  
  hub: {
    title: "Flow IR Fundamentals"
    highlight: "Learn how Flow expresses visualization declaratively"
    description: "Understand the structure, schema, and rendering pipeline of Flow Intermediate Representation."
  }
  
  eyebrowLabel: "Flow Concepts"
  startGateTitle: "Ready to learn Flow IR?"
  
  slide "What is Flow IR?" {
    eyebrow: "Foundations"
    title: "The Flow Intermediate Representation"
    lede: "Declarative visualization language for AI systems"
    
    narration: "Flow IR is a JSON schema that describes interactive cinematic visualizations of AI system behavior. Unlike imperative rendering instructions, Flow IR expresses *what* to show, not *how* to show it. This separation enables backend neutrality: the same IR renders identically on canvas, SVG, or semantic HTML."
    
    term: {
      word: "Intermediate Representation"
      meaning: "A structured format that captures intent without prescribing implementation"
    }
    
    points: [
      "Declarative: expresses intent, not implementation",
      "Backend-neutral: renders on canvas, SVG, semantic HTML",
      "Deterministic: same IR always produces identical output",
      "Composable: scenes nest and inherit from parent definitions"
    ]
    
    caption: "Flow IR bridges authoring intent and runtime rendering"
    
    render: @scene {
      roots: [
        {
          id: "ir-concept-box"
          capability: "core.rect"
          layout: { x: 100, y: 100, width: 300, height: 200 }
          style: { fill: @theme.surface.primary }
          children: [
            {
              id: "ir-label"
              capability: "core.text"
              text: "Flow IR"
              layout: { x: 10, y: 10, width: 280, height: 30 }
              style: { fontSize: 24, fill: @theme.ink.primary }
            }
          ]
        }
      ]
    }
  }
  
  slide "Scene Evaluation" {
    eyebrow: "Runtime"
    title: "How Scenes Execute"
    lede: "The journey from IR to pixels"
    
    narration: "When a Flow scene evaluates, the runtime traverses the scene graph, resolving data bindings, applying theme colors, and generating display instructions. This is a deterministic process: given the same IR and input data, evaluation always produces the same display output."
    
    points: [
      "Resolve data bindings (input values, theme roles)",
      "Generate display instructions (shapes, text, paths)",
      "Apply theme colors to diagram elements",
      "Produce backend-neutral display list"
    ]
    
    caption: "The evaluation pipeline: IR → theme → display → render"
    
    render: @scene {
      // Timeline visualization of evaluation phases
      roots: [...]
    }
  }
  
  // More slides...
}
```

---

## Approval Checkpoints

- [ ] Language design approved (explainer syntax, grammar, compiler)
- [ ] Runtime architecture approved (components, integration points)
- [ ] Content scope approved (4 ported + 3-5 new decks)
- [ ] Testing strategy approved (unit, runtime, E2E, content)
- [ ] Task phasing approved (parallel implementer wave sizing)
