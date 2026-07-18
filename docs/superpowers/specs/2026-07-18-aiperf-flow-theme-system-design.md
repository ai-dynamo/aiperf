# AIPerf Flow Theme System Design

## Status

Approved design for a typed AIPerf Flow theme system and its first bundled
theme, **Systems Chalk**.

This record refines the typed visual-system direction in
[`2026-07-17-aiperf-flow-design.md`](2026-07-17-aiperf-flow-design.md). It
defines the first implementable theme slice without introducing the complete
materials, lighting, filters, or responsive type system described by the
broader product design.

## Goal

AIPerf Flow authors can select a named theme in a `.flow` document, define a
typed custom theme, and allow a host application to override the selected
theme at runtime. The same resolved theme controls scene rendering and a
restrained set of player-chrome properties.

The first bundled theme presents inference systems as clean technical
explainers:

- charcoal surfaces;
- chalk-white structure and labels;
- restrained cyan, teal, blue, and yellow semantic accents;
- precise rounded geometry;
- causal draw-on animation;
- subtle analog character without distorted shapes or pervasive handwriting.

Systems Chalk is an original AIPerf Flow theme. It takes visual direction from
the supplied dark technical-explainer references without copying third-party
names, logos, assets, layouts, or illustrations.

## Non-goals

This increment does not add:

- arbitrary CSS in `.flow` source;
- backend-specific theme branches;
- user-authored shaders, filters, patterns, or materials;
- light, print, or high-contrast variants as separately named themes;
- theme-defined layout, camera, narration, semantics, or interaction;
- runtime injection of unvalidated token dictionaries;
- nondeterministic line jitter;
- a general animation scripting system.

Reduced motion, high contrast, backend quality, and other accessibility or
capability policies remain orthogonal runtime axes. A theme supplies defaults
that those policies may constrain.

## Design principles

### Semantic before literal

Components request visual roles such as `surface.canvas`, `ink.primary`, or
`accent.execution`. They do not embed a Systems Chalk color or ask which theme
is active.

### Resolve before rendering

Theme roles resolve during scene evaluation. The display list contains literal,
backend-neutral paint and motion values. Canvas and SVG renderers do not contain
theme-specific conditionals.

### Strict authoring, immutable runtime

Theme declarations use a closed, typed role vocabulary. Compilation rejects
unknown roles and invalid values. Runtime registration revalidates untrusted IR,
resolves inheritance, and freezes the result.

### Explicit local overrides survive

Existing authored style values and component props remain valid. A direct
author override takes precedence over a semantic theme default.

### Motion explains causality

Systems Chalk reveals an entity, draws its outgoing connector toward the
destination, reveals that destination, and then adds annotation. Motion is not
continuous decoration.

## Authoring model

### Selecting a theme

A document selects its default theme with a top-level declaration:

```flow
use theme systems_chalk
```

If omitted, the runtime's existing default visual behavior remains active.
This preserves current documents and avoids silently restyling authored work.

### Defining a custom theme

A custom theme extends exactly one complete theme and overrides typed roles:

```flow
theme lab_chalk extends systems_chalk {
  color accent.control = "#78dce8"
  color accent.execution = "#ffd866"
  number stroke.standard = 2
  duration motion.draw = 420ms
}

use theme lab_chalk
```

The first version requires a base theme. It does not support standalone partial
themes, multiple inheritance, nested theme scopes, or scene-local themes.
Single inheritance keeps resolution deterministic and gives every custom theme
a complete fallback.

Theme IDs use the language's identifier rules. Bundled theme IDs are reserved;
a document cannot redefine `systems_chalk`.

### Referring to a role explicitly

Foundation-node styles may opt into a semantic role:

```flow
rect "router" {
  fill theme(surface.raised)
  stroke theme(accent.control)
}
```

`theme(role)` remains unresolved in IR so a runtime theme override can change
the result. Existing `token(name)` references remain compile-time document
tokens and continue to lower to literals.

### Precedence

Visual values resolve in this order:

1. explicit component prop or node style;
2. custom-theme override;
3. inherited base-theme value;
4. schema-defined component default.

A literal author value and an authored `token(name)` reference are both
explicit overrides. A `theme(role)` reference resolves against the active
theme.

## Typed role vocabulary

The schema package owns role names and value types. The initial vocabulary is
intentionally small.

### Colors

- `surface.canvas`
- `surface.panel`
- `surface.raised`
- `surface.control`
- `ink.primary`
- `ink.muted`
- `ink.inverse`
- `line.structural`
- `line.guide`
- `accent.control`
- `accent.execution`
- `accent.compute`
- `accent.attention`
- `accent.success`
- `accent.danger`
- `accent.focus`

Color values use strict CSS hexadecimal syntax in this increment. Supporting
other color spaces remains future work.

### Typography

- `font.display`
- `font.body`
- `font.data`
- `weight.regular`
- `weight.label`
- `weight.emphasis`
- `size.caption`
- `size.body`
- `size.label`
- `size.title`

Font roles are ordered family stacks. Weight values are integers from 100 to
900. Size values are finite positive logical pixels.

### Stroke and shape

- `stroke.hairline`
- `stroke.standard`
- `stroke.emphasis`
- `stroke.cap`
- `stroke.join`
- `shape.cornerRadius`

Stroke widths are finite nonnegative numbers. The initial cap vocabulary is
`butt`, `round`, and `square`; the join vocabulary is `bevel`, `round`, and
`miter`. Corner radius is a finite nonnegative logical-pixel number.

### Motion

- `motion.draw`
- `motion.enter`
- `motion.emphasis`
- `motion.stagger`
- `motion.easing`

Durations are nonnegative integer milliseconds. The initial easing vocabulary
is `linear`, `ease_in`, `ease_out`, and `ease_in_out`. Reduced-motion policy may
replace draw and enter motion with immediate state changes.

## Schema and IR

The schema package adds:

```ts
type ThemeRole =
  | "surface.canvas"
  | "surface.panel"
  // remaining closed roles
  | "motion.easing";

type ThemeValueIr =
  | Readonly<{ kind: "color"; value: string }>
  | Readonly<{ kind: "font"; value: readonly string[] }>
  | Readonly<{ kind: "number"; value: number }>
  | Readonly<{ kind: "duration"; valueMs: number }>
  | Readonly<{ kind: "enum"; value: string }>;

type FlowThemeIr = Readonly<{
  id: string;
  extends: string;
  values: Readonly<Partial<Record<ThemeRole, ThemeValueIr>>>;
  sourceMap: SourceRange;
}>;

type ThemeRoleReferenceIr = Readonly<{
  kind: "theme-role";
  role: ThemeRole;
}>;
```

`FlowIr` gains:

```ts
themes: readonly FlowThemeIr[];
defaultTheme?: string;
```

Style values gain `ThemeRoleReferenceIr` alongside existing scalar values.
The IR remains strict and rejects unknown fields. A compatible IR version bump
is required because consumers of `style` currently assume scalar values.

Compiler output contains authored custom themes and the selected default ID.
Bundled theme definitions are supplied by the versioned runtime registry rather
than duplicated into each document.

## Compilation

The language package parses `theme`, `extends`, `use theme`, typed assignments,
and `theme(role)` references. The compiler then:

1. collects declarations across linked modules;
2. rejects duplicate or reserved IDs;
3. validates role names and value kinds;
4. resolves base IDs against authored and bundled themes;
5. detects inheritance cycles;
6. verifies that the selected default exists;
7. emits custom theme definitions and unresolved role references.

Diagnostics include the theme ID, role or base ID, expected type, received
value, and source range. Imported modules may export themes, but a linked
document has one active default.

## Runtime registry and resolution

The runtime owns a transactional `ThemeRegistry` analogous to the capability
evaluator registry:

- bundled themes register during runtime bootstrap;
- document themes register as one atomic batch;
- duplicates fail before any document theme becomes visible;
- freezing returns an immutable lookup;
- resolved themes are cached by ID;
- role lookup is constant time after resolution.

`ThemeRegistry.resolve(id)` walks the single-parent chain, detects cycles again
for untrusted IR, validates the complete result, and returns a deeply frozen
`ResolvedTheme`.

The active ID is:

```text
explicit runtime override
→ document default
→ existing runtime default
```

An unknown explicit override is an error. Player UI controls only offer IDs
present in the frozen registry, so a normal user cannot create that error.

## Evaluation integration

`CapabilityEvaluationContext` gains the resolved theme. Foundation-node
evaluation and every capability evaluator use shared typed helpers such as:

```ts
theme.color("accent.execution")
theme.number("stroke.standard")
theme.font("font.body")
```

The helpers return concrete values and fail closed on a kind mismatch. Shared
component visual-role mappings keep Queue, Waterfall, SegmentStrip, SpanMap,
GlyphRun, and semantic morph contributions consistent.

The scene evaluator resolves explicit `ThemeRoleReferenceIr` values before it
creates draw commands. Literal authored styles bypass role lookup.

No evaluator reads CSS variables or imports a bundled theme directly.

## Display-list and renderer integration

The display-list path contract adds:

- `lineCap`;
- `lineJoin`;
- `strokeReveal`, a finite value clamped to `[0, 1]`.

`strokeReveal` expresses draw progress without naming a theme. SVG implements
it with normalized path length and stroke dash offset. Canvas applies the same
dash behavior using a cached path length. A maintained path-metrics package is
preferred over a custom SVG path parser.

When reduced motion is active, evaluation emits `strokeReveal: 1`. Fill-only
shapes do not use stroke reveal. Hit regions and semantic output are complete
for every frame; visual reveal never temporarily removes accessibility
semantics.

The display list continues to contain literal fill, stroke, font, width, cap,
join, and reveal values. Backend conformance tests assert equivalent output.

## Player chrome

`FlowApp` accepts an optional runtime theme override and exposes the active theme
ID to the host. The preview shell adds a theme selector for available registry
entries; embedders are not required to render one.

The runtime maps a restrained subset of `ResolvedTheme` to CSS custom
properties on the `.aiperf-flow` root:

- canvas and panel surfaces;
- primary and muted ink;
- control, focus, and danger accents;
- body and data fonts;
- standard stroke width.

Scene rendering does not read these CSS properties. They exist only to keep the
player, controls, subtitles, focus treatment, and scene surround coherent.
Control geometry and navigation layout do not change by theme.

## Systems Chalk

### Palette

- `surface.canvas`: `#232526`
- `surface.panel`: `#292C2D`
- `surface.raised`: `#303334`
- `surface.control`: `#383C3E`
- `ink.primary`: `#F1F3F2`
- `ink.muted`: `#AEB4B5`
- `ink.inverse`: `#232526`
- `line.structural`: `#D7DADA`
- `line.guide`: `#777D80`
- `accent.control`: `#71D8D0`
- `accent.execution`: `#69C8BA`
- `accent.compute`: `#77B8DE`
- `accent.attention`: `#F0CF58`
- `accent.success`: `#7DCE82`
- `accent.danger`: `#F07972`
- `accent.focus`: `#9BDBF5`

The palette uses color to distinguish semantic roles. Large structural regions
remain neutral.

### Typography

Systems Chalk uses a softly rounded, legible sans for titles and labels and a
tabular monospaced face for metrics. Font assets must be bundled with the
runtime so output does not depend on host-installed fonts. The implementation
selects mature, redistributable font packages and records their licenses.

Handwriting fonts are intentionally excluded. The references derive their
clarity from clean labels, not simulated handwriting.

### Shape and line

Systems Chalk uses:

- two-logical-pixel standard outlines;
- rounded caps and joins;
- restrained twelve-logical-pixel corner radii;
- solid structural edges;
- dotted or dashed guide and in-flight edges;
- no ambient shadows or glow;
- no random wobble.

### Signature motion

Causal draw-on is the signature:

1. reveal the source entity;
2. draw the connector toward its target;
3. reveal the target entity;
4. add its label or metric annotation.

Existing scene and component timelines determine causal order. Theme motion
roles provide durations and easing, not semantic ordering. Components without
causal anchors use a simple enter transition and do not invent dependencies.

## Accessibility and quality policy

Systems Chalk must meet WCAG AA contrast for normal text and non-text
boundaries on its default surfaces. The compiler validates syntax and type;
runtime theme registration computes contrast for required role pairs and
rejects a custom theme that does not meet the minimum.

Required contrast pairs include:

- `ink.primary` on every surface;
- `ink.muted` on canvas and panel;
- each interactive accent on `surface.control`;
- `accent.focus` against canvas, panel, and control surfaces.

High-contrast policy may substitute stronger runtime values after normal theme
resolution. Reduced-motion policy disables progressive stroke reveal and
stagger. Degraded quality cannot remove structural lines, labels, focus
indication, or semantic color distinctions.

## Errors

Public errors use specific classes and actionable messages for:

- duplicate theme IDs;
- reserved bundled IDs;
- unknown base themes;
- inheritance cycles;
- unknown roles;
- role/value kind mismatches;
- invalid colors, numbers, durations, or enums;
- incomplete resolved themes;
- insufficient required contrast;
- unknown document defaults;
- unknown explicit runtime overrides.

Errors identify the theme and role and preserve source ranges when the error
originates in authored source.

## Testing strategy

### Language and compiler

- parse valid selection, inheritance, typed overrides, and role references;
- reject every invalid declaration category;
- compile imported themes deterministically;
- preserve `theme(role)` references in IR;
- keep existing `token(name)` lowering unchanged.

### Schema

- round-trip theme IR and role references;
- reject unknown fields, non-finite numbers, and incorrect discriminants;
- verify the IR version boundary.

### Runtime

- test atomic registration, freeze behavior, inheritance, caching, and
  precedence;
- test every role kind and required contrast pair;
- test runtime override, document default, and legacy default selection;
- verify exact error messages for unknown IDs and cycles.

### Evaluation and backends

- replace hard-coded defaults in Queue, Waterfall, SegmentStrip, SpanMap,
  GlyphRun, semantic morph, and foundation text/path evaluation;
- assert Systems Chalk colors, fonts, strokes, caps, and joins in evaluated
  display lists;
- assert Canvas and SVG parity for full and partial stroke reveal;
- assert reduced motion emits complete strokes;
- retain complete semantic projections and hit regions during reveal.

### Application and end-to-end

- verify the active theme is reflected on the Flow root and player chrome;
- verify runtime switching does not recompile the document;
- render the deterministic request-lifecycle cinematic fixture with Systems
  Chalk in Canvas and SVG fallback;
- assert key raw display-list values and accessible semantics;
- add focused visual snapshots only for stable reference scenes.

## Compatibility and migration

Existing documents with literal styles or document tokens retain their output.
Existing capability props remain valid. The runtime default remains the current
visual behavior until a document or host selects a theme.

Component hard-coded fallback colors migrate to semantic roles in the same
implementation. Tests that currently assert those literals change to assert
resolved values under an explicit test theme.

The IR version bump makes the new style-value union explicit. Old IR can be
upgraded by supplying empty themes, no default theme, and scalar-only styles.

## Completion criteria

The feature is complete when:

- `.flow` can declare, import, select, and extend typed themes;
- compiler and schema reject invalid theme data with source-aware errors;
- runtime selection supports document defaults and explicit overrides;
- all current core evaluators consume semantic theme roles;
- Canvas and SVG render the same resolved Systems Chalk output;
- causal draw-on works deterministically and respects reduced motion;
- player chrome adopts the restrained active-theme subset;
- existing unthemed documents remain visually compatible;
- language, compiler, runtime, backend, accessibility, and cinematic E2E tests
  pass.
